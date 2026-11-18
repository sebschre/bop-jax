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
class BOPConfig:
    nmom: int = 6  # moments calculated exactly
    nexp: int | None = None  # length of the Chebyshev expansion (default nmom)
    bandwidth: str = "lowest"  # see expansion.BANDWIDTHS
    global_ab: tuple[float, float] = (0.0, 1.0)  # for bandwidth = "global"
    terminator: str = "none"  # see expansion.TERMINATORS
    alpha_weight: float = 1.0  # for terminator = "weighted"
    kernel: str = "none"  # "none" or "jackson"
    rcut: float = 3.0  # bond integrals
    dcut: float = 0.0
    rcut2: float | None = None  # repulsion, embedding, Coulomb
    dcut2: float = 0.0
    basis: tuple[str, ...] = ("d",)
    spin: str = "none"  # see SPIN_MODES
    spin_orbit: bool = False  # onsite xi L.S (spin = "noncollinear" only)
    repulsion: pw.Repulsion | None = None
    coulomb: str = "diagonal"  # "diagonal" (onsite only) or "frauenheim"
    screening: bool = False  # environment-dependent d-d bond integrals
    batch: int = 64  # atoms whose moments are evaluated at once (memory)

    @property
    def n_expansion(self) -> int:
        return self.nmom if self.nexp is None else max(self.nexp, self.nmom)

    @property
    def magnetic(self) -> bool:
        return self.spin != "none"


def _per_atom(fn, topo: Topology, batch: int):
    """Apply ``fn(cluster_origin_i, cluster_nbr_i)`` to every atom, in batches
    so that memory stays bounded for large cells."""
    xs = (jnp.asarray(topo.cluster_origin), jnp.asarray(topo.cluster_nbr))
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
    def run(origin, nbr):
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


def moments(hblk, onsite, topo: Topology, nmom: int, batch: int = 64):
    """Diagonal moments <i a|H^n|i a> for all unit-cell orbitals, (N, norb, nmom+1).

    hblk   : (N, K, norb, norb) hopping blocks of the unit-cell atoms
    onsite : (N, norb) onsite levels
    """
    norb = onsite.shape[1]
    sizes = topo.cluster_sizes

    def hop(origins, w_nbr, w_self):  # w_nbr: (C, K, norb, norb_start)
        return (jnp.einsum("ckab,ckbd->cad", hblk[origins], w_nbr)
                + onsite[origins][:, :, None] * w_self)

    walk = _walk(topo, nmom, jnp.eye(norb, dtype=hblk.dtype), hop)

    def one(origin, nbr):
        ws = walk(origin, nbr)
        mu = []
        for k in range(nmom + 1):
            a, b = k // 2, k - k // 2
            mu.append(jnp.einsum("cbd,cbd->d", ws[a], ws[b][: sizes[a]]))
        return jnp.stack(mu, axis=-1)

    return _per_atom(one, topo, batch)


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
                         spin_orbit=None):
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
    if topo.n_atoms <= batch:
        return jax.vmap(one)(*xs)
    return jax.lax.map(lambda x: one(*x), xs, batch_size=batch)


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

    ``shifts`` has shape ``energy.shift_shape``: (N, nshell), or (N, 2, nshell)
    for magnetic calculations (spin up, spin down).
    """
    if cfg.spin not in SPIN_MODES:
        raise ValueError(f"unknown spin mode {cfg.spin!r}")
    if cfg.spin_orbit and cfg.spin != "noncollinear":
        raise ValueError('spin-orbit coupling needs spin="noncollinear"')
    slices, norb = shell_slices(cfg.basis)
    nshell = len(cfg.basis)
    nspin = 2 if cfg.magnetic else 1
    nexp = cfg.n_expansion
    kernel = ex.jackson_kernel(nexp) if cfg.kernel == "jackson" else np.ones(nexp + 1)
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
    needs_pairs = cfg.repulsion is not None or cfg.coulomb != "diagonal"
    if needs_pairs and topo.nbr2_mask is None:
        raise ValueError("build the topology with rcut2 for repulsion or Coulomb terms")

    def local_expansion(mu):
        a, b = ex.recursion_coefficients(mu, cfg.nmom)
        ainf, binf = ex.band_centre_and_width(a, b, cfg.nmom, cfg.bandwidth, cfg.global_ab)
        sigma = ex.sigma_coefficients(a, b, ainf, binf, cfg.nmom, nexp, kernel,
                                      cfg.terminator, cfg.alpha_weight)
        return ainf, binf, sigma

    def occupations(loc, ef):
        """Electrons and bond energy per atom, spin channel and shell."""
        ainf, binf, sigma = loc
        gamma0 = sigma[..., 1] / kernel[1]
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
        lo = jnp.min(frozen[0] - 2.0 * frozen[1])
        hi = jnp.max(frozen[0] + 2.0 * frozen[1])

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
        charge = jnp.sum(n_shell - params["n0"][species], axis=1)
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
        e0 = params["onsite"][species]  # (N, nshell)
        levels = e0[:, None, :] + shifts.reshape(topo.n_atoms, nspin, nshell)
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
                                          topo, cfg.nmom, cfg.batch, onsite_matrix)
        else:
            mu_orb = jax.vmap(lambda lev: moments(hblk, lev, topo, cfg.nmom, cfg.batch),
                              in_axes=1, out_axes=1)(orb_levels)
        # average over the orbitals of each shell: (N, nspin, nshell, nmom+1)
        mu = jnp.stack([jnp.mean(mu_orb[:, :, a:b], axis=2) for a, b in slices], axis=2)

        loc = jnp.vectorize(local_expansion, signature="(m)->(),(),(k)")(mu)
        n0 = params["n0"][species]
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

    energy.shift_shape = (topo.n_atoms, 2, nshell) if cfg.magnetic else (topo.n_atoms, nshell)
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
