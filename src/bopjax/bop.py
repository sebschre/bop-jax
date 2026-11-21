"""Analytic BOP total energy as one differentiable function.

    E(shifts, positions, cell, params)

``shifts`` are the changes of the onsite levels: one per atom and shell, and
additionally one per spin channel in magnetic calculations.  The energy is the
tight-binding bond model functional

    E = sum U_bond + sum E0 (N - N0) + 1/2 sum_i J_ii q_i^2
        + 1/2 sum_{i != j} J_ij q_i q_j - 1/4 sum I m^2 + E_rep

and self-consistency is its stationary point with respect to ``shifts``.
Forces, stress, magnetic torques and parameter gradients at that point are
plain partial derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import expansion as ex
from . import pairwise as pw
from .geometry import Topology, bond_vectors, pair_vectors, reverse_slots
from .hamiltonian import (L_OF, angular_momentum, bond_blocks, cosine_cutoff,
                          powerlaw, screening_factors, shell_slices)

SPIN_MODES = ("none", "collinear", "noncollinear")
MU_B = 5.7883818060e-5  # Bohr magneton in eV / T
G_SPIN = 2.00231930436  # electron spin g-factor


@dataclass(frozen=True)
class SpeciesSettings:
    """Shells and expansion settings of one species (``BOPConfig.species``).

    ``basis`` is the subset of ``BOPConfig.basis`` this species carries; its
    other shells are absent: no hopping into or out of them, no electrons and
    no energy.  Entries left at None take the value of the ``BOPConfig``.
    ``nmom`` may not exceed ``BOPConfig.nmom``, which sets how far the
    moments are walked for every atom.
    """

    basis: tuple[str, ...]
    nmom: int | None = None
    nexp: int | None = None
    bandwidth: str | None = None
    terminator: str | None = None
    kernel: str | None = None
    bandwidth_guard: bool | None = None
    # in magnetic calculations: False gives the species one onsite shift for
    # both spin channels (no exchange splitting of its own; a moment can
    # still be induced through its neighbours)
    spin_polarized: bool = True
    # one onsite shift for all shells of the species (their mean), as
    # BOPConfig.atomic_shifts does for every species
    atomic_shifts: bool = False


@dataclass(frozen=True)
class BOPConfig:
    nmom: int = 6  # moments calculated exactly
    nexp: int | None = None  # length of the Chebyshev expansion (default nmom)
    bandwidth: str = "lowest"  # see expansion.BANDWIDTHS
    global_ab: tuple[float, float] = (0.0, 1.0)  # for bandwidth = "global"
    terminator: str = "none"  # see expansion.TERMINATORS
    alpha_weight: float = 1.0  # for terminator = "weighted"
    kernel: str = "none"  # "none" or "jackson"
    # widen b_inf where the scaled recursion chain has eigenvalues outside
    # [-1, 1], where the Chebyshev expansion would diverge
    bandwidth_guard: bool = False
    rcut: float = 3.0  # bond integrals
    dcut: float = 0.0
    rcut2: float | None = None  # repulsion, embedding, Coulomb
    dcut2: float = 0.0
    basis: tuple[str, ...] = ("d",)
    spin: str = "none"  # see SPIN_MODES
    spin_orbit: bool = False  # onsite xi L.S (spin = "noncollinear" only)
    repulsion: pw.Repulsion | None = None
    yukawa: bool = False  # environment-dependent Yukawa repulsion (params["yukawa"])
    coulomb: str = "diagonal"  # "diagonal" (onsite only) or "frauenheim"
    screening: bool = False  # environment-dependent d-d bond integrals
    batch: int = 64  # atoms whose moments are evaluated at once (memory)
    remat: bool = False  # recompute the moment walks in the backward pass (memory)
    kT: float = 0.0  # electronic temperature (eV); the energy is then the free energy
    smearing_nodes: int = 24  # quadrature nodes of the Fermi kernel (kT > 0)
    # per-species shells and expansion settings, one SpeciesSettings per
    # species; None: every species has all of ``basis`` and the settings above
    species: tuple[SpeciesSettings, ...] | None = None
    # one onsite shift per atom (and spin) for all its shells, instead of one
    # per shell: self-consistency in the atomic charge (and moment) only
    atomic_shifts: bool = False

    @property
    def n_expansion(self) -> int:
        return self.nmom if self.nexp is None else max(self.nexp, self.nmom)

    @property
    def magnetic(self) -> bool:
        return self.spin != "none"


def _per_atom(fn, topo: Topology, batch: int, remat: bool = False, *extra):
    """Apply ``fn(cluster_origin_i, cluster_nbr_i, *extra_i)`` to every atom, in
    batches so that memory stays bounded for large cells.  With ``remat`` the
    walk of each batch is recomputed in the backward pass instead of being
    stored."""
    if remat:
        fn = jax.checkpoint(fn)
    xs = (jnp.asarray(topo.cluster_origin), jnp.asarray(topo.cluster_nbr)) + extra
    if topo.n_atoms <= batch:
        return jax.vmap(fn)(*xs)
    return jax.lax.map(lambda x: fn(*x), xs, batch_size=batch)


def _walk(topo: Topology, nmom: int, start, hop):
    """Vectors H^a |start> for a = 0..ceil(nmom / 2) on a growing cluster.

    ``hop(origins, w_neighbours, w_self)`` applies H to the first len(origins)
    cluster members.  After a applications only atoms within a hops are
    non-zero, so each step works on the smallest cluster that holds them.
    """
    sizes = topo.cluster_sizes
    def run(origin, nbr, start=start):
        ws = [start[None]]
        for a in range((nmom + 1) // 2):
            n_in, n_out = sizes[a], sizes[a + 1]
            w = ws[-1]
            zero = jnp.zeros((n_out + 1 - n_in,) + w.shape[1:], dtype=w.dtype)
            w_pad = jnp.concatenate([w, zero])  # slot n_out is the outside world
            idx = jnp.where(nbr[:n_out] < n_in, nbr[:n_out], n_out)
            ws.append(hop(origin[:n_out], w_pad[idx], w_pad[:n_out]))
        return ws
    return run


def moments(hblk, onsite, topo: Topology, nmom: int, batch: int = 64, remat: bool = False,
            starts=None):
    """Diagonal moments <i a|H^n|i a> for all unit-cell orbitals, (N, norb, nmom+1).

    hblk   : (N, K, norb, norb) hopping blocks of the unit-cell atoms
    onsite : (N, norb) onsite levels
    starts : optional (N, norb, ncol) start vectors of each atom; the result
             is then <start_c|H^n|start_c>, shape (N, ncol, nmom+1)
    """
    norb = onsite.shape[1]
    sizes = topo.cluster_sizes

    def hop(origins, w_nbr, w_self):  # w_nbr: (C, K, norb, norb_start)
        return (jnp.einsum("ckab,ckbd->cad", hblk[origins], w_nbr)
                + onsite[origins][:, :, None] * w_self)

    walk = _walk(topo, nmom, jnp.eye(norb, dtype=hblk.dtype), hop)

    def one(origin, nbr, *start):
        ws = walk(origin, nbr, *start)
        mu = []
        for k in range(nmom + 1):
            a, b = k // 2, k - k // 2
            mu.append(jnp.einsum("cbd,cbd->d", ws[a], ws[b][: sizes[a]]))
        return jnp.stack(mu, axis=-1)

    if starts is None:
        return _per_atom(one, topo, batch, remat)
    return _per_atom(one, topo, batch, remat, jnp.asarray(starts, dtype=hblk.dtype))


PAULI = np.array(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex
)


def spin_projectors(axes):
    """Projectors on spin up and down along each atom's axis, (N, 2, 2, 2)."""
    e = axes / jnp.linalg.norm(axes, axis=-1, keepdims=True)
    e_sigma = jnp.einsum("nx,xab->nab", e.astype(complex), PAULI)
    eye = jnp.eye(2, dtype=complex)
    return jnp.stack([0.5 * (eye + e_sigma), 0.5 * (eye - e_sigma)], axis=1)


def spin_orbit_blocks(basis) -> np.ndarray:
    """``L.S`` of each shell as a matrix in (orbital, spin), embedded in the
    orbitals of the atom: (nshell, norb, 2, norb, 2), with S = sigma / 2 in the
    global spin frame.  Its eigenvalues are l/2 and -(l+1)/2."""
    slices, norb = shell_slices(basis)
    out = np.zeros((len(basis), norb, 2, norb, 2), dtype=complex)
    for k, (sh, (a, b)) in enumerate(zip(basis, slices)):
        out[k, a:b, :, a:b, :] = 0.5 * np.einsum("xab,xtu->atbu", angular_momentum(sh), PAULI)
    return out


def orbital_blocks(basis) -> np.ndarray:
    """``L`` of all shells in the orbitals of the atom, times the unit matrix
    in spin: (3, norb, 2, norb, 2)."""
    slices, norb = shell_slices(basis)
    out = np.zeros((3, norb, 2, norb, 2), dtype=complex)
    for sh, (a, b) in zip(basis, slices):
        out[:, a:b, :, a:b, :] = np.einsum("xab,tu->xatbu", angular_momentum(sh), np.eye(2))
    return out


def spin_blocks(basis) -> np.ndarray:
    """Pauli matrices times the unit matrix in the orbitals of the atom:
    (3, norb, 2, norb, 2), in the global spin frame."""
    _, norb = shell_slices(basis)
    return np.einsum("xtu,ab->xatbu", PAULI, np.eye(norb))


def _field_sources(params, n, dtype):
    """Onsite sources ``h_i`` (coupling ``h_i.sigma``) and ``b_i`` (coupling
    ``b_i.L``) in eV, each (N, 3) or None, from ``params``.

    An external field B (Tesla) enters both:  h = -mu_B g/2 B,  b = -mu_B B,
    so that the energy falls by mu_B B.M for a magnetic moment
    M = g/2 <sigma> + <L> along B (to first order)."""
    h = b = None
    if "magnetic_field" in params:
        field = jnp.broadcast_to(jnp.asarray(params["magnetic_field"], dtype), (n, 3))
        h, b = -0.5 * G_SPIN * MU_B * field, -MU_B * field
    if "spin_field" in params:
        h = params["spin_field"] if h is None else h + params["spin_field"]
    if "orbital_field" in params:
        b = params["orbital_field"] if b is None else b + params["orbital_field"]
    return h, b


def moments_noncollinear(hblk, onsite, axes, topo: Topology, nmom: int, batch: int = 64,
                         spin_orbit=None, remat: bool = False):
    """Moments of the spinor Hamiltonian projected on each atom's local spin
    axis, shape (N, 2, norb, nmom+1).

    onsite     : (N, 2, norb) levels of the local spin-up and spin-down channels
    axes       : (N, 3) local spin quantisation axes (need not be normalised)
    spin_orbit : None, or an extra onsite matrix (N, norb, 2, norb, 2) in the
                 global spin frame, e.g. ``xi * spin_orbit_blocks(basis)``, plus
                 any Zeeman terms

    The start states stay |real orbital a, spin up/down along the axis>.  With
    spin-orbit coupling these are no longer eigenstates of the onsite block,
    which then couples the orbitals of a shell; the hopping stays diagonal in
    spin.
    """
    norb = onsite.shape[2]
    sizes = topo.cluster_sizes
    proj = spin_projectors(axes)  # (N, 2, 2, 2)
    # onsite block of orbital a on atom i: sum_s level_s P_s
    on_blk = jnp.einsum("nsa,nstu->natu", onsite.astype(complex), proj)
    hb = hblk.astype(complex)

    if spin_orbit is None:
        def hop(origins, w_nbr, w_self):  # w_nbr: (C, K, norb, 2, norb_start, 2)
            return (jnp.einsum("ckab,ckbtdu->catdu", hb[origins], w_nbr)
                    + jnp.einsum("catv,cavdu->catdu", on_blk[origins], w_self))
    else:
        # full onsite matrix in (orbital, spin): (N, norb, 2, norb, 2)
        on_full = (jnp.einsum("natu,ab->natbu", on_blk, jnp.eye(norb))
                   + spin_orbit.astype(complex))

        def hop(origins, w_nbr, w_self):
            return (jnp.einsum("ckab,ckbtdu->catdu", hb[origins], w_nbr)
                    + jnp.einsum("catbv,cbvdu->catdu", on_full[origins], w_self))

    start = jnp.einsum("ad,tu->atdu", jnp.eye(norb), jnp.eye(2)).astype(complex)
    walk = _walk(topo, nmom, start, hop)

    def one(origin, nbr, p):
        ws = walk(origin, nbr)
        mu = []
        for k in range(nmom + 1):
            a, b = k // 2, k - k // 2
            # 2x2 spin matrix of the moment for each start orbital
            m = jnp.einsum("catdu,catdv->duv", jnp.conj(ws[a]), ws[b][: sizes[a]])
            mu.append(jnp.real(jnp.einsum("svu,duv->sd", p, m)))
        return jnp.stack(mu, axis=-1)

    xs = (jnp.asarray(topo.cluster_origin), jnp.asarray(topo.cluster_nbr), proj)
    if remat:
        one = jax.checkpoint(one)
    if topo.n_atoms <= batch:
        return jax.vmap(one)(*xs)
    return jax.lax.map(lambda x: one(*x), xs, batch_size=batch)


@dataclass(frozen=True)
class _SpeciesLayout:
    """Static bookkeeping of ``BOPConfig.species``."""

    shells: tuple  # per species: indices of its shells in cfg.basis
    orb_mask: np.ndarray  # (nspecies, norb) 1 for the orbitals a species has
    shell_mask: np.ndarray  # (nspecies, nshell)
    starts: np.ndarray  # (N, norb, ncol) start vectors of each atom
    columns: tuple  # per species, per own shell: start columns of that shell
    atoms: tuple  # per species: its atoms in the unit cell
    settings: tuple  # per species: (nmom, nexp, bandwidth, terminator, kernel, guard)


def _species_layout(cfg: BOPConfig, species: np.ndarray) -> _SpeciesLayout:
    slices, norb = shell_slices(cfg.basis)
    if species.size and species.max() >= len(cfg.species):
        raise ValueError("BOPConfig.species has fewer entries than the topology has species")
    shells, orbs, settings = [], [], []
    for st in cfg.species:
        if not set(st.basis) <= set(cfg.basis):
            raise ValueError(f"species basis {st.basis} is not part of {cfg.basis}")
        own = tuple(k for k, sh in enumerate(cfg.basis) if sh in st.basis)
        shells.append(own)
        orbs.append([o for k in own for o in range(*slices[k])])
        nmom = cfg.nmom if st.nmom is None else st.nmom
        if nmom > cfg.nmom:
            raise ValueError("a species' nmom may not exceed BOPConfig.nmom")
        nexp = st.nexp if st.nexp is not None else (cfg.nexp if st.nmom is None else None)
        nexp = nmom if nexp is None else max(nexp, nmom)
        settings.append((nmom, nexp,
                         cfg.bandwidth if st.bandwidth is None else st.bandwidth,
                         cfg.terminator if st.terminator is None else st.terminator,
                         cfg.kernel if st.kernel is None else st.kernel,
                         cfg.bandwidth_guard if st.bandwidth_guard is None
                         else st.bandwidth_guard))
    nsp = len(cfg.species)
    orb_mask = np.zeros((nsp, norb))
    shell_mask = np.zeros((nsp, len(cfg.basis)))
    for s_, (own, o) in enumerate(zip(shells, orbs)):
        orb_mask[s_, o] = 1.0
        shell_mask[s_, list(own)] = 1.0
    # each atom walks only from its own orbitals: column c of species s starts
    # on orbital orbs[s][c]; spare columns start from zero and are ignored
    ncol = max(len(o) for o in orbs)
    starts = np.zeros((len(species), norb, ncol))
    for i, s_ in enumerate(species):
        starts[i, orbs[s_], np.arange(len(orbs[s_]))] = 1.0
    columns = tuple(tuple([c for c, o in enumerate(orbs[s_]) if slices[k][0] <= o < slices[k][1]]
                          for k in shells[s_]) for s_ in range(nsp))
    atoms = tuple(np.nonzero(species == s_)[0] for s_ in range(nsp))
    return _SpeciesLayout(tuple(shells), orb_mask, shell_mask, starts, columns, atoms,
                          tuple(settings))


def make_energy_fn(topo: Topology, cfg: BOPConfig, radial=powerlaw,
                   screening_radial=None):
    """Return ``energy(shifts, positions, cell, params) -> (E, aux)``.

    params is a dict of arrays:
      "onsite"    (nspecies, nshell)   free-atom onsite levels E0
      "n0"        (nspecies, nshell)   free-atom shell occupations
      "hubbard"   (nspecies,)          onsite Coulomb J_ii
      "bond"      {channel: (nspecies, nspecies, npar)}
      "stoner"    (nspecies, nshell)   Stoner integrals        [magnetic]
      "spin_axes" (N, 3)               local spin directions   [noncollinear]
      "spin_orbit" (nspecies, nshell)  spin-orbit constants xi [cfg.spin_orbit]
      "magnetic_field" (3,) or (N, 3)  optional external field B (Tesla)
                                       [magnetic]
      "spin_field"    (N, 3)           optional source h_i of a term h_i.sigma
                                       on each atom (eV)       [magnetic]
      "orbital_field" (N, 3)           optional source b_i of a term b_i.L on
                                       each atom (eV)          [noncollinear]
      "repulsion" {"pair", "density": (nspecies, nspecies, npar),
                   "embed": (nspecies, npar)}                  [cfg.repulsion]
      "taper"     (nspecies, nspecies, 2)   R1, R2             [cfg.coulomb]
      "yukawa"    {"B", "r_core", "C", "upsilon": (nspecies, nspecies),
                   "lambda0", "m": (nspecies,)}               [cfg.yukawa]
      "screening" {"sds", "sds_overlap", "dds_overlap", "ddp_overlap",
                   "ddd_overlap": (nspecies, nspecies, npar)}  [cfg.screening]

    With ``cfg.screening`` the d-d bond integrals are screened by a valence s
    orbital on the surrounding atoms (see ``hamiltonian.screening_factors``).
    "sds" is the sd-sigma bond integral and the other entries are overlaps;
    for "sds*" the entry [a, b] has the s orbital on species a.  They are
    evaluated with ``screening_radial`` (default: ``radial``) and share the
    cutoff of the bond integrals.

    With ``cfg.spin_orbit`` the onsite term ``xi L.S`` is added to the spinor
    Hamiltonian (``L`` in units of hbar, ``S = sigma / 2``, so the j = l + 1/2
    and j = l - 1/2 levels of a free atom are ``xi (2l + 1) / 2`` apart).  It
    ties the spin axes to the lattice: the energy is then invariant only under
    a common rotation of positions and axes, and it acquires a magnetic
    anisotropy.  The local-axis ansatz is unchanged, and with it what the
    functional can describe: the exchange field of each atom stays along its
    axis, the moment is the spin moment projected on the axis, and the
    orbital moment and the spin density perpendicular to the axis, both of
    first order in xi, are not fed back.  The orbital moment can be evaluated
    as the derivative of the energy with respect to ``params["orbital_field"]``
    (see ``make_orbital_moment``).

    An external field ``params["magnetic_field"]`` adds the Zeeman term
    ``-mu_B B.(L + g/2 sigma)`` to the onsite Hamiltonian (see
    ``bopjax.zeeman``).  Moments are counted as everywhere in bopjax, with the
    magnetisation along ``m e``, so the field favours ``m e`` along B.  The
    part of the spin term along each local axis acts like a free-atom level:
    it splits the spin-up and spin-down levels by ``-/+ h.e`` and adds
    ``(h.e) m`` to the energy (``aux["energy_zeeman"]``), so the shifts remain
    the internal (Stoner and Coulomb) potentials.  The perpendicular part and
    ``b.L`` enter the spinor Hamiltonian as matrices, like ``xi L.S``.  In
    collinear calculations only the component along z couples, to the spin
    only.

    With ``cfg.kT > 0`` the band filling has an electronic temperature: the
    filling is averaged over the Fermi kernel and the bond energy becomes the
    Mermin free energy (``expansion.smeared_filling_and_bond_energy``), so the
    energy is the free energy E - TS, still variational in the shifts.

    ``shifts`` has shape ``energy.shift_shape``: (N, nshell), or (N, 2, nshell)
    for magnetic calculations (spin up, spin down).  With
    ``cfg.atomic_shifts`` the last axis has length 1: every shell of an atom
    is shifted by the same amount (per spin channel), so that only the atomic
    charge and moment are made self-consistent.  With several shells the
    relative position of their levels is then fixed by ``onsite``; with one
    shift per shell it is a further unknown that, in a truncated expansion,
    need not have a stationary point (seen for carbon in iron).
    """
    if cfg.spin not in SPIN_MODES:
        raise ValueError(f"unknown spin mode {cfg.spin!r}")
    if cfg.spin_orbit and cfg.spin != "noncollinear":
        raise ValueError('spin-orbit coupling needs spin="noncollinear"')
    slices, norb = shell_slices(cfg.basis)
    nshell = len(cfg.basis)
    nspin = 2 if cfg.magnetic else 1
    nexp = cfg.n_expansion
    # electrons a full shell holds per spin channel
    degeneracy = np.array([(2.0 / nspin) * (2 * L_OF[s] + 1) for s in cfg.basis])
    species = topo.species
    orb_shell = np.concatenate([[k] * (b - a) for k, (a, b) in enumerate(slices)])
    ls_blocks = spin_orbit_blocks(cfg.basis) if cfg.spin_orbit else None
    noncollinear = cfg.spin == "noncollinear"
    l_blocks = orbital_blocks(cfg.basis) if noncollinear else None
    s_blocks = spin_blocks(cfg.basis) if noncollinear else None
    if cfg.screening and "d" not in cfg.basis:
        raise ValueError("screening acts on d-d bond integrals; the basis has no d shell")
    nbr_reverse = reverse_slots(topo) if cfg.screening else None
    screening_radial = radial if screening_radial is None else screening_radial
    needs_pairs = cfg.repulsion is not None or cfg.coulomb != "diagonal" or cfg.yukawa
    if needs_pairs and topo.nbr2_mask is None:
        raise ValueError("build the topology with rcut2 for repulsion or Coulomb terms")
    layout = None if cfg.species is None else _species_layout(cfg, np.asarray(species))
    unpolarized = (np.array([not cfg.species[s_].spin_polarized for s_ in species])
                   if layout is not None else np.zeros(len(species), bool))
    tied_shells = (np.array([cfg.species[s_].atomic_shifts for s_ in species])
                   if layout is not None and not cfg.atomic_shifts
                   else np.zeros(len(species), bool))
    if layout is not None:
        nexp = max(st[1] for st in layout.settings)

    def expansion_fn(nmom_, nexp_, bandwidth_, terminator_, kernel_, guard_=False):
        kern = ex.jackson_kernel(nexp_) if kernel_ == "jackson" else np.ones(nexp_ + 1)

        def local(mu_):
            a, b = ex.recursion_coefficients(mu_, nmom_)
            ainf, binf = ex.band_centre_and_width(a, b, nmom_, bandwidth_, cfg.global_ab)
            if guard_:
                binf = ex.guarded_width(a, b, ainf, binf, nmom_, nexp_, terminator_,
                                        cfg.alpha_weight)
            sigma = ex.sigma_coefficients(a, b, ainf, binf, nmom_, nexp_, kern,
                                          terminator_, cfg.alpha_weight)
            return ainf, binf, sigma, sigma[1] / kern[1]

        return jnp.vectorize(local, signature="(m)->(),(),(k),()")

    if cfg.kT < 0.0:
        raise ValueError("kT must not be negative")
    smear_offsets = smear_weights = None
    if cfg.kT > 0.0:
        smear_offsets, smear_weights = ex.fermi_nodes(cfg.kT, cfg.smearing_nodes)

    def local_expansion(mu):
        """(a_inf, b_inf, sigma, gamma0), each (N, nspin, nshell[, nexp + 1])."""
        if layout is None:
            return expansion_fn(cfg.nmom, nexp, cfg.bandwidth, cfg.terminator,
                                cfg.kernel, cfg.bandwidth_guard)(mu)
        # absent shells: sigma = 0, i.e. no states; a_inf, b_inf only keep
        # the Fermi-level bracket finite
        shape = mu.shape[:3]
        ainf, binf = jnp.zeros(shape, mu.dtype), jnp.ones(shape, mu.dtype)
        sigma = jnp.zeros(shape + (nexp + 1,), mu.dtype)
        gamma0 = jnp.zeros(shape, mu.dtype)
        for atoms, own, st in zip(layout.atoms, layout.shells, layout.settings):
            if len(atoms) == 0:
                continue
            idx = np.ix_(atoms, np.arange(shape[1]), np.asarray(own))
            a_s, b_s, sig_s, g_s = expansion_fn(*st)(mu[idx][..., : st[0] + 1])
            ainf, binf, gamma0 = ainf.at[idx].set(a_s), binf.at[idx].set(b_s), \
                gamma0.at[idx].set(g_s)
            sigma = sigma.at[idx].set(jnp.pad(sig_s, [(0, 0)] * 3 + [(0, nexp - st[1])]))
        return ainf, binf, sigma, gamma0

    def occupations(loc, ef):
        """Electrons and bond energy per atom, spin channel and shell."""
        ainf, binf, sigma, gamma0 = loc
        if cfg.kT > 0.0:
            def smeared(sig, a_, b_, e_, g_):
                return ex.smeared_filling_and_bond_energy(sig, a_, b_, e_, g_,
                                                          smear_offsets, smear_weights)
            fn = jnp.vectorize(smeared, signature="(k),(),(),(),()->(),()")
        else:
            fn = jnp.vectorize(ex.filling_and_bond_energy, signature="(k),(),(),(),()->(),()")
        fill, bond = fn(sigma, ainf, binf, ef, gamma0)
        return degeneracy * fill, degeneracy * bond

    def fermi_level(loc, n_total):
        """Fermi level for fixed total electron count.

        Bisection without gradients, then two differentiable Newton steps: these
        carry exact first, second and third derivatives of the Fermi level.
        """
        def excess(ef, loc_):
            return jnp.sum(occupations(loc_, ef)[0]) - n_total

        frozen = jax.lax.stop_gradient(loc)
        lo = jnp.min(frozen[0] - 2.0 * frozen[1]) - 40.0 * cfg.kT
        hi = jnp.max(frozen[0] + 2.0 * frozen[1]) + 40.0 * cfg.kT

        def bisect(_, bracket):
            lo_, hi_ = bracket
            mid = 0.5 * (lo_ + hi_)
            too_many = excess(mid, frozen) > 0.0
            return jnp.where(too_many, lo_, mid), jnp.where(too_many, mid, hi_)

        lo, hi = jax.lax.fori_loop(0, 100, bisect, (lo, hi))
        ef = jax.lax.stop_gradient(0.5 * (lo + hi))
        for _ in range(2):
            f, dos = jax.value_and_grad(excess)(ef, loc)
            ef = ef - f / dos
        return ef

    def pair_terms(positions, cell, params, charge):
        """Repulsion/embedding and intersite Coulomb energy per atom."""
        rvec = pair_vectors(positions, cell, topo)
        r = jnp.linalg.norm(rvec, axis=-1)
        mask = jnp.asarray(topo.nbr2_mask, dtype=r.dtype)
        sp_i = np.broadcast_to(species[:, None], topo.nbr2_origin.shape)
        sp_j = species[topo.nbr2_origin]
        e_rep = jnp.zeros(topo.n_atoms, dtype=r.dtype)
        e_coul = jnp.zeros(topo.n_atoms, dtype=r.dtype)
        if cfg.repulsion is not None:
            rp = params["repulsion"]
            fc = cosine_cutoff(r, cfg.rcut2, cfg.dcut2) * mask
            pair = jnp.vectorize(cfg.repulsion.pair, signature="(p),()->()")
            dens = jnp.vectorize(cfg.repulsion.density, signature="(p),()->()")
            emb = jnp.vectorize(cfg.repulsion.embed, signature="(p),()->()")
            rho = jnp.sum(fc * dens(rp["density"][sp_i, sp_j], r), axis=1)
            e_rep = emb(rp["embed"][species], rho) + 0.5 * jnp.sum(
                fc * pair(rp["pair"][sp_i, sp_j], r), axis=1)
        if cfg.yukawa:
            e_rep = e_rep + pw.yukawa_energy(params["yukawa"], r, mask, species, sp_i, sp_j,
                                             topo.nbr2_origin, cfg.rcut2, cfg.dcut2)
        if cfg.coulomb != "diagonal":
            e_coul = coulomb_energy(r, mask, sp_i, sp_j, params, charge)
        return e_rep, e_coul

    def coulomb_energy(r, mask, sp_i, sp_j, params, charge):
        """Intersite Coulomb energy per atom."""
        if cfg.coulomb != "frauenheim":
            raise ValueError(f"unknown Coulomb model {cfg.coulomb!r}")
        jii = params["hubbard"]
        tp = params["taper"][sp_i, sp_j]
        in_range = jnp.where(r < cfg.rcut2, 1.0, 0.0) * mask
        jij = in_range * pw.taper(tp[..., 0], tp[..., 1], r) * pw.frauenheim_coulomb(
            jii[sp_i], jii[sp_j], r)
        return 0.5 * charge * jnp.sum(jij * charge[topo.nbr2_origin], axis=1)

    def double_counting(occupations, positions, cell, params):
        """The part of the energy that depends on the occupations (N, nspin,
        nshell) explicitly: onsite and intersite Coulomb, Stoner."""
        n_shell = jnp.sum(occupations, axis=1)
        n0_ = params["n0"][species]
        if layout is not None:
            n0_ = n0_ * layout.shell_mask[species]
        charge = jnp.sum(n_shell - n0_, axis=1)
        e = 0.5 * jnp.sum(params["hubbard"][species] * charge**2)
        if cfg.magnetic:
            moment = occupations[:, 0] - occupations[:, 1]
            e = e - 0.25 * jnp.sum(params["stoner"][species] * moment**2)
        if cfg.coulomb != "diagonal":
            rvec = pair_vectors(positions, cell, topo)
            r = jnp.linalg.norm(rvec, axis=-1)
            mask = jnp.asarray(topo.nbr2_mask, dtype=r.dtype)
            sp_i = np.broadcast_to(species[:, None], topo.nbr2_origin.shape)
            e = e + jnp.sum(coulomb_energy(r, mask, sp_i, species[topo.nbr2_origin], params,
                                           charge))
        return e

    def potential_target(occupations, positions, cell, params):
        """Onsite shifts that the occupations imply, dE_dc/dN, in the shape of
        the shifts.  For exact tight binding the stationary point of the
        energy has shifts == potential_target(occupations)."""
        grad = jax.grad(double_counting)(occupations, positions, cell, params)
        grad = grad.reshape(topo.n_atoms, nspin, nshell)
        # the same tying of shifts as in the energy: one per atom, per own
        # shells of a species, or one for both spins
        if cfg.atomic_shifts:
            grad = jnp.mean(grad, axis=2, keepdims=True)
        elif layout is not None and tied_shells.any():
            own = jnp.asarray(layout.shell_mask[species], grad.dtype)
            mean = (jnp.sum(grad * own[:, None, :], axis=2, keepdims=True)
                    / jnp.sum(own, axis=1)[:, None, None])
            grad = jnp.where(tied_shells[:, None, None], mean, grad)
        if layout is not None and cfg.magnetic and unpolarized.any():
            grad = jnp.where(unpolarized[:, None, None], jnp.mean(grad, axis=1, keepdims=True),
                             grad)
        return grad.reshape((topo.n_atoms,) + energy.shift_shape[1:])

    def energy(shifts, positions, cell, params):
        rij = bond_vectors(positions, cell, topo)
        species_i = np.broadcast_to(species[:, None], topo.nbr_origin.shape)
        dd_screen = None
        if cfg.screening:
            dd_screen = screening_factors(
                rij, topo.nbr_mask, species, topo.nbr_origin, nbr_reverse,
                params["screening"], cfg.rcut, cfg.dcut, screening_radial)
        hblk = bond_blocks(rij, topo.nbr_mask, species_i,
                           species[topo.nbr_origin], params["bond"], cfg.basis,
                           cfg.rcut, cfg.dcut, radial, dd_screen)
        starts = None
        if layout is not None:
            om = layout.orb_mask
            hblk = hblk * (om[species][:, None, :, None] * om[species[topo.nbr_origin]][:, :, None, :])
            starts = layout.starts
        e0 = params["onsite"][species]  # (N, nshell)
        nshift = 1 if cfg.atomic_shifts else nshell
        shifts = shifts.reshape(topo.n_atoms, nspin, nshift)
        if layout is not None and tied_shells.any():
            # one shift for the species' own shells: their mean
            own = jnp.asarray(layout.shell_mask[species], shifts.dtype)  # (N, nshell)
            mean = (jnp.sum(shifts * own[:, None, :], axis=2, keepdims=True)
                    / jnp.sum(own, axis=1)[:, None, None])
            shifts = jnp.where(tied_shells[:, None, None], mean, shifts)
        if layout is not None and cfg.magnetic and unpolarized.any():
            # one shift for both spins: their mean, so the difference is idle
            mean = jnp.mean(shifts, axis=1, keepdims=True)
            shifts = jnp.where(unpolarized[:, None, None], mean, shifts)
        levels = e0[:, None, :] + shifts
        h_field, b_field = _field_sources(params, topo.n_atoms, rij.dtype)
        if (h_field is not None or b_field is not None) and not cfg.magnetic:
            raise ValueError("a magnetic field needs a magnetic calculation (cfg.spin)")
        if not noncollinear:
            if "orbital_field" in params:
                raise ValueError('an orbital field needs spin="noncollinear"')
            b_field = None  # real orbitals and no spin-orbit coupling: quenched
        h_par = None
        if h_field is not None:
            if noncollinear:
                e_axis = params["spin_axes"] / jnp.linalg.norm(
                    params["spin_axes"], axis=-1, keepdims=True)
                h_par = jnp.sum(h_field * e_axis, axis=-1)
            else:
                h_par = h_field[:, 2]
            levels = levels + jnp.stack([h_par, -h_par], axis=1)[:, :, None]
        orb_levels = levels[:, :, orb_shell]  # (N, nspin, norb)
        if noncollinear:
            extra = []
            if cfg.spin_orbit:
                extra.append(jnp.einsum("nk,katbu->natbu", params["spin_orbit"][species],
                                        ls_blocks))
            if b_field is not None:
                extra.append(jnp.einsum("nx,xatbu->natbu", b_field, l_blocks))
            if h_field is not None:
                h_perp = h_field - h_par[:, None] * e_axis
                extra.append(jnp.einsum("nx,xatbu->natbu", h_perp, s_blocks))
            onsite_matrix = sum(extra[1:], extra[0]) if extra else None
            mu_orb = moments_noncollinear(hblk, orb_levels, params["spin_axes"],
                                          topo, cfg.nmom, cfg.batch, onsite_matrix,
                                          cfg.remat)
        else:
            mu_orb = jax.vmap(lambda lev: moments(hblk, lev, topo, cfg.nmom, cfg.batch,
                                                  cfg.remat, starts),
                              in_axes=1, out_axes=1)(orb_levels)
        # average over the orbitals of each shell: (N, nspin, nshell, nmom+1)
        if layout is None:
            mu = jnp.stack([jnp.mean(mu_orb[:, :, a:b], axis=2) for a, b in slices], axis=2)
        else:
            if cfg.spin == "noncollinear":  # walked from every orbital: pick own
                mu_orb = jnp.einsum("nsom,noc->nscm", mu_orb,
                                    jnp.asarray(layout.starts, mu_orb.dtype))
            mu = jnp.zeros(mu_orb.shape[:2] + (nshell, mu_orb.shape[-1]), mu_orb.dtype)
            for atoms, own, cols in zip(layout.atoms, layout.shells, layout.columns):
                for k, c in zip(own, cols):
                    mu = mu.at[atoms, :, k].set(jnp.mean(mu_orb[atoms][:, :, np.asarray(c)], axis=2))

        loc = local_expansion(mu)
        n0 = params["n0"][species]
        if layout is not None:
            n0 = n0 * layout.shell_mask[species]
        ef = fermi_level(loc, jnp.sum(n0))
        n_spin, u_bond = occupations(loc, ef)  # (N, nspin, nshell)

        n_shell = jnp.sum(n_spin, axis=1)
        charge = jnp.sum(n_shell - n0, axis=1)
        e_bond = jnp.sum(u_bond, axis=(1, 2))
        e_prom = jnp.sum(e0 * (n_shell - n0), axis=1)
        e_ion = 0.5 * params["hubbard"][species] * charge**2
        e_atom = e_bond + e_prom + e_ion
        aux = {"fermi": ef, "n_shell": n_shell, "charge": charge, "energy_bond": e_bond,
               "occupations": n_spin}
        if cfg.magnetic:
            moment = n_spin[:, 0] - n_spin[:, 1]  # per shell, along the local axis
            e_mag = -0.25 * jnp.sum(params["stoner"][species] * moment**2, axis=1)
            e_atom = e_atom + e_mag
            aux.update(n_spin=n_spin, moment=jnp.sum(moment, axis=1), energy_mag=e_mag)
            if h_par is not None:
                e_zeeman = h_par * jnp.sum(moment, axis=1)
                e_atom = e_atom + e_zeeman
                aux.update(energy_zeeman=e_zeeman)
        if needs_pairs:
            e_rep, e_coul = pair_terms(positions, cell, params, charge)
            e_atom = e_atom + e_rep + e_coul
            aux.update(energy_rep=e_rep, energy_coulomb=e_coul)

        squeeze = (lambda x: x) if cfg.magnetic else (lambda x: x[:, 0])
        aux.update(energy_atom=e_atom, a_inf=squeeze(loc[0]), b_inf=squeeze(loc[1]),
                   sigma=squeeze(loc[2]), moments=squeeze(mu))
        return jnp.sum(e_atom), aux

    nshift = 1 if cfg.atomic_shifts else nshell
    energy.shift_shape = (topo.n_atoms, 2, nshift) if cfg.magnetic else (topo.n_atoms, nshift)
    energy.config = cfg
    energy.topology = topo
    energy.double_counting = double_counting
    energy.potential_target = potential_target
    return energy


def make_orbital_moment(energy_fn):
    """Return ``orbital_moment(shifts, positions, cell, params) -> (N, 3)``, the
    orbital angular momentum ``<L>`` of each atom in units of hbar.

    It is the derivative of the energy with respect to a field ``b_i`` that
    couples to the orbital angular momentum of atom i through an onsite term
    ``b_i.L`` (Hellmann-Feynman), taken at ``b = 0``.  At self-consistent
    ``shifts`` the partial derivative is the total one.  ``<L>`` is in the same
    convention as the spin moment ``m e = 2 <S>``: for a d shell that is more
    than half full it is parallel to ``m e`` when ``xi > 0`` (Hund's third
    rule), and the orbital magnetic moment in Bohr magnetons is ``<L>`` where
    the spin magnetic moment is ``m e``.

    Without spin-orbit coupling the orbital moment is quenched and the result
    is zero.  It is an observable only: nothing in the energy depends on it.
    """
    if not energy_fn.config.spin_orbit:
        raise ValueError("the orbital moment needs BOPConfig(spin_orbit=True)")
    n = energy_fn.topology.n_atoms

    def orbital_moment(shifts, positions, cell, params):
        def with_field(field):
            return energy_fn(shifts, positions, cell, dict(params, orbital_field=field))[0]
        return jax.grad(with_field)(jnp.zeros((n, 3), dtype=jnp.asarray(positions).dtype))

    return orbital_moment


def stoner_shifts(moments, stoner):
    """Starting onsite shifts for given shell moments (N, nshell): the exchange
    splitting -/+ I m / 2 for spin up/down.  Returns (N, 2, nshell)."""
    split = 0.5 * jnp.asarray(stoner) * jnp.asarray(moments)
    return jnp.stack([-split, split], axis=1)
