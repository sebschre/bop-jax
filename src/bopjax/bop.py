"""Analytic BOP total energy as one differentiable function.

    E(shifts, positions, cell, params)

``shifts`` are the changes of the onsite levels, one per atom and shell.  The
energy is the tight-binding bond model functional

    E = sum_i U_bond,i + sum_ia E0_ia (N_ia - N0_ia) + 1/2 sum_i J_i q_i^2

and self-consistency is its stationary point with respect to ``shifts``.  Forces
and parameter gradients at that point are plain partial derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import expansion as ex
from .geometry import Topology, bond_vectors
from .hamiltonian import L_OF, bond_blocks, powerlaw, shell_slices


@dataclass(frozen=True)
class BOPConfig:
    nmom: int = 6  # moments calculated exactly
    nexp: int | None = None  # length of the Chebyshev expansion (default nmom)
    bandwidth: str = "lowest"
    kernel: str = "none"  # "none" or "jackson"
    rcut: float = 3.0
    dcut: float = 0.0
    basis: tuple[str, ...] = ("d",)

    @property
    def n_expansion(self) -> int:
        return self.nmom if self.nexp is None else max(self.nexp, self.nmom)


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


def make_energy_fn(topo: Topology, cfg: BOPConfig, radial=powerlaw):
    """Return ``energy(shifts, positions, cell, params) -> (E, aux)``.

    params is a dict of arrays:
      "onsite"  (nspecies, nshell)   free-atom onsite levels E0
      "n0"      (nspecies, nshell)   free-atom shell occupations
      "hubbard" (nspecies,)          onsite Coulomb J_i
      "bond"    {channel: (nspecies, nspecies, npar)}
    """
    slices, norb = shell_slices(cfg.basis)
    nshell = len(cfg.basis)
    nexp = cfg.n_expansion
    kernel = ex.jackson_kernel(nexp) if cfg.kernel == "jackson" else np.ones(nexp + 1)
    degeneracy = np.array([2.0 * (2 * L_OF[s] + 1) for s in cfg.basis])
    species = topo.species
    orb_shell = np.concatenate([[k] * (b - a) for k, (a, b) in enumerate(slices)])

    def local_expansion(mu):
        a, b = ex.recursion_coefficients(mu, cfg.nmom)
        ainf, binf = ex.band_centre_and_width(a, b, cfg.nmom, cfg.bandwidth)
        sigma = ex.sigma_coefficients(a, b, ainf, binf, cfg.nmom, nexp, kernel)
        return ainf, binf, sigma

    def occupations(loc, ef):
        ainf, binf, sigma = loc
        gamma0 = sigma[..., 1] / kernel[1]
        fn = jax.vmap(jax.vmap(ex.filling_and_bond_energy, (0, 0, 0, None, 0)),
                      (0, 0, 0, None, 0))
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

    def energy(shifts, positions, cell, params):
        rij = bond_vectors(positions, cell, topo)
        species_i = np.broadcast_to(species[:, None], topo.nbr_origin.shape)
        hblk = bond_blocks(rij, topo.nbr_mask, species_i,
                           species[topo.nbr_origin], params["bond"], cfg.basis,
                           cfg.rcut, cfg.dcut, radial)
        e0 = params["onsite"][species]  # (N, nshell)
        levels = e0 + shifts
        mu_orb = moments(hblk, levels[:, orb_shell], topo, cfg.nmom)
        # average over the orbitals of each shell
        mu = jnp.stack([jnp.mean(mu_orb[:, a:b], axis=1) for a, b in slices], axis=1)

        loc = jax.vmap(jax.vmap(local_expansion))(mu)
        n0 = params["n0"][species]
        ef = fermi_level(loc, jnp.sum(n0))
        n_shell, u_bond = occupations(loc, ef)

        charge = jnp.sum(n_shell - n0, axis=1)
        e_bond = jnp.sum(u_bond, axis=1)
        e_prom = jnp.sum(e0 * (n_shell - n0), axis=1)
        e_ion = 0.5 * params["hubbard"][species] * charge**2
        e_atom = e_bond + e_prom + e_ion
        aux = {
            "fermi": ef,
            "n_shell": n_shell,
            "charge": charge,
            "energy_atom": e_atom,
            "energy_bond": e_bond,
            "a_inf": loc[0],
            "b_inf": loc[1],
            "sigma": loc[2],
            "moments": mu,
        }
        return jnp.sum(e_atom), aux

    energy.shift_shape = (topo.n_atoms, nshell)
    return energy
