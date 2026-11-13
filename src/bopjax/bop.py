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
from .geometry import Topology, bond_vectors, pair_vectors
from .hamiltonian import L_OF, bond_blocks, cosine_cutoff, powerlaw, shell_slices

SPIN_MODES = ("none", "collinear", "noncollinear")


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
    repulsion: pw.Repulsion | None = None
    coulomb: str = "diagonal"  # "diagonal" (onsite only) or "frauenheim"

    @property
    def n_expansion(self) -> int:
        return self.nmom if self.nexp is None else max(self.nexp, self.nmom)

    @property
    def magnetic(self) -> bool:
        return self.spin != "none"


def moments(hblk, onsite, topo: Topology, nmom: int):
    """Diagonal moments <i a|H^n|i a> for all unit-cell orbitals, (N, norb, nmom+1).

    hblk   : (N, K, norb, norb) hopping blocks of the unit-cell atoms
    onsite : (N, norb) onsite levels
    """
    n, norb = onsite.shape
    hb = hblk[topo.origin]  # (M, K, norb, norb)
    on = onsite[topo.origin]  # (M, norb)

    def apply_h(w):  # w: (S, M, norb, norb_start)
        hop = jnp.einsum("mkab,smkbc->smac", hb, w[:, topo.nbr_super])
        return hop + on[None, :, :, None] * w

    # one block of start vectors per unit-cell atom (supercell index s == s)
    w0 = jnp.zeros((n, topo.n_super, norb, norb), dtype=hblk.dtype)
    w0 = w0.at[np.arange(n), np.arange(n)].set(jnp.eye(norb, dtype=hblk.dtype))
    ws = [w0]
    for _ in range((nmom + 1) // 2):
        ws.append(apply_h(ws[-1]))

    mu = [jnp.einsum("smbc,smbc->sc", ws[k // 2], ws[k - k // 2])
          for k in range(nmom + 1)]
    return jnp.stack(mu, axis=-1)


PAULI = np.array(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex
)


def spin_projectors(axes):
    """Projectors on spin up and down along each atom's axis, (N, 2, 2, 2)."""
    e = axes / jnp.linalg.norm(axes, axis=-1, keepdims=True)
    e_sigma = jnp.einsum("nx,xab->nab", e.astype(complex), PAULI)
    eye = jnp.eye(2, dtype=complex)
    return jnp.stack([0.5 * (eye + e_sigma), 0.5 * (eye - e_sigma)], axis=1)


def moments_noncollinear(hblk, onsite, axes, topo: Topology, nmom: int):
    """Moments of the spinor Hamiltonian projected on each atom's local spin
    axis, shape (N, 2, norb, nmom+1).

    onsite : (N, 2, norb) levels of the local spin-up and spin-down channels
    axes   : (N, 3) local spin quantisation axes (need not be normalised)
    """
    n, _, norb = onsite.shape
    proj = spin_projectors(axes)  # (N, 2, 2, 2)
    # onsite block of orbital a on atom i: sum_s level_s P_s
    on_blk = jnp.einsum("nsa,nstu->natu", onsite.astype(complex), proj)[topo.origin]
    hb = hblk[topo.origin].astype(complex)

    def apply_h(w):  # w: (S, M, norb, 2, norb_start, 2)
        hop = jnp.einsum("mkab,smkbtcu->smatcu", hb, w[:, topo.nbr_super])
        return hop + jnp.einsum("matv,smavcu->smatcu", on_blk, w)

    w0 = jnp.zeros((n, topo.n_super, norb, 2, norb, 2), dtype=complex)
    start = jnp.einsum("ac,tu->atcu", jnp.eye(norb), jnp.eye(2)).astype(complex)
    w0 = w0.at[np.arange(n), np.arange(n)].set(start)
    ws = [w0]
    for _ in range((nmom + 1) // 2):
        ws.append(apply_h(ws[-1]))

    mu = []
    for k in range(nmom + 1):
        # 2x2 spin matrix of the moment for each start orbital
        m = jnp.einsum("smatcu,smatcv->scuv", jnp.conj(ws[k // 2]), ws[k - k // 2])
        mu.append(jnp.real(jnp.einsum("nsvu,ncuv->nsc", proj, m)))
    return jnp.stack(mu, axis=-1)


def make_energy_fn(topo: Topology, cfg: BOPConfig, radial=powerlaw):
    """Return ``energy(shifts, positions, cell, params) -> (E, aux)``.

    params is a dict of arrays:
      "onsite"    (nspecies, nshell)   free-atom onsite levels E0
      "n0"        (nspecies, nshell)   free-atom shell occupations
      "hubbard"   (nspecies,)          onsite Coulomb J_ii
      "bond"      {channel: (nspecies, nspecies, npar)}
      "stoner"    (nspecies, nshell)   Stoner integrals        [magnetic]
      "spin_axes" (N, 3)               local spin directions   [noncollinear]
      "repulsion" {"pair", "density": (nspecies, nspecies, npar),
                   "embed": (nspecies, npar)}                  [cfg.repulsion]
      "taper"     (nspecies, nspecies, 2)   R1, R2             [cfg.coulomb]

    ``shifts`` has shape ``energy.shift_shape``: (N, nshell), or (N, 2, nshell)
    for magnetic calculations (spin up, spin down).
    """
    if cfg.spin not in SPIN_MODES:
        raise ValueError(f"unknown spin mode {cfg.spin!r}")
    slices, norb = shell_slices(cfg.basis)
    nshell = len(cfg.basis)
    nspin = 2 if cfg.magnetic else 1
    nexp = cfg.n_expansion
    kernel = ex.jackson_kernel(nexp) if cfg.kernel == "jackson" else np.ones(nexp + 1)
    # electrons a full shell holds per spin channel
    degeneracy = np.array([(2.0 / nspin) * (2 * L_OF[s] + 1) for s in cfg.basis])
    species = topo.species
    orb_shell = np.concatenate([[k] * (b - a) for k, (a, b) in enumerate(slices)])
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
        if cfg.coulomb == "frauenheim":
            jii = params["hubbard"]
            tp = params["taper"][sp_i, sp_j]
            in_range = jnp.where(r < cfg.rcut2, 1.0, 0.0) * mask
            jij = in_range * pw.taper(tp[..., 0], tp[..., 1], r) * pw.frauenheim_coulomb(
                jii[sp_i], jii[sp_j], r)
            e_coul = 0.5 * charge * jnp.sum(jij * charge[topo.nbr2_origin], axis=1)
        elif cfg.coulomb != "diagonal":
            raise ValueError(f"unknown Coulomb model {cfg.coulomb!r}")
        return e_rep, e_coul

    def energy(shifts, positions, cell, params):
        rij = bond_vectors(positions, cell, topo)
        species_i = np.broadcast_to(species[:, None], topo.nbr_origin.shape)
        hblk = bond_blocks(rij, topo.nbr_mask, species_i,
                           species[topo.nbr_origin], params["bond"], cfg.basis,
                           cfg.rcut, cfg.dcut, radial)
        e0 = params["onsite"][species]  # (N, nshell)
        levels = e0[:, None, :] + shifts.reshape(topo.n_atoms, nspin, nshell)
        orb_levels = levels[:, :, orb_shell]  # (N, nspin, norb)
        if cfg.spin == "noncollinear":
            mu_orb = moments_noncollinear(hblk, orb_levels, params["spin_axes"],
                                          topo, cfg.nmom)
        else:
            mu_orb = jax.vmap(lambda lev: moments(hblk, lev, topo, cfg.nmom),
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
        aux = {"fermi": ef, "n_shell": n_shell, "charge": charge, "energy_bond": e_bond}
        if cfg.magnetic:
            moment = n_spin[:, 0] - n_spin[:, 1]  # per shell, along the local axis
            e_mag = -0.25 * jnp.sum(params["stoner"][species] * moment**2, axis=1)
            e_atom = e_atom + e_mag
            aux.update(n_spin=n_spin, moment=jnp.sum(moment, axis=1), energy_mag=e_mag)
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
    return energy


def stoner_shifts(moments, stoner):
    """Starting onsite shifts for given shell moments (N, nshell): the exchange
    splitting -/+ I m / 2 for spin up/down.  Returns (N, 2, nshell)."""
    split = 0.5 * jnp.asarray(stoner) * jnp.asarray(moments)
    return jnp.stack([-split, split], axis=1)
