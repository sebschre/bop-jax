"""Pairwise terms on the second neighbour list: repulsion, embedding, Coulomb.

Functional forms follow BOPfox (``eamfct``: generic EAM energy with the "v0"
functions; ``buildhop``: Frauenheim-type Coulomb integrals with a taper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

COULOMB_CONSTANT = 14.4  # eV Angstrom


@dataclass(frozen=True)
class Repulsion:
    """E_rep,i = embed(sum_j density(r_ij)) + 1/2 sum_j pair(r_ij).

    pair(p, r) and density(p, r) take the parameter row of the species pair,
    embed(p, rho) the parameter row of the species.
    """

    pair: Callable
    density: Callable
    embed: Callable


def _exp_pair(p, r):
    return p[0] * jnp.exp(-p[1] * (r / p[2] - 1.0))


def _power_embed(p, rho):
    safe = jnp.where(rho > 0.0, rho, 1.0)
    return jnp.where(rho > 0.0, p[0] * safe ** p[1], 0.0)


# BOPfox RepVersion "v0": exponential pair term and density, power-law embedding
V0 = Repulsion(pair=_exp_pair, density=_exp_pair, embed=_power_embed)

# pair term only, p[0] * (p[2] / r) ** p[1]
POWERLAW = Repulsion(
    pair=lambda p, r: p[0] * (p[2] / r) ** p[1],
    density=lambda p, r: 0.0 * r,
    embed=lambda p, rho: 0.0 * rho,
)


def exp_sum(p, r):
    """Sum of stretched exponentials  sum_k c_k exp(-lambda_k r^n_k), with
    p = [c_1, lambda_1, n_1, c_2, lambda_2, n_2, ...]; unused terms have c = 0.
    Bond integrals and pair repulsion of Egorov et al., arXiv:2208.12973 (FeCo BOP)."""
    terms = p.reshape(-1, 3)
    return jnp.sum(terms[:, 0] * jnp.exp(-terms[:, 1] * r ** terms[:, 2]))


def _egorov_pair(p, r):
    """p = [c, lambda, n, p1, p2]: c exp(-lambda r^n) plus the short-range core
    repulsion p2 (p1 - r)^3 / r exp(p1 - r) for r < p1."""
    core = jnp.where(r < p[3], p[4] * jnp.clip(p[3] - r, 0.0) ** 3 / r * jnp.exp(p[3] - r), 0.0)
    return p[0] * jnp.exp(-p[1] * r ** p[2]) + core


def _gaussian_density(p, r):
    """p = [p1, p2, p3]: p1 exp(-p2 (r - p3)^2)."""
    return p[0] * jnp.exp(-p[1] * (r - p[2]) ** 2)


# Egorov et al.: pair and core repulsion, square-root embedding of Gaussian
# densities (embed row [-1, 0.5])
EGOROV = Repulsion(pair=_egorov_pair, density=_gaussian_density, embed=_power_embed)


def yukawa_energy(yp, r, mask, species, sp_i, sp_j, nbr_origin, rcut, dcut):
    """Environment-dependent Yukawa repulsion per atom (Znam et al., Phil. Mag. 83,
    415 (2003); form of Egorov et al.):

        E_i = 1/2 sum_j B / r_ij exp(-lambda_ij (r_ij - 2 r_core)),
        lambda_ij = (lambda_i + lambda_j) / 2,
        lambda_i = lambda0 + (sum_l C exp(-upsilon r_il))^(1/m),

    all sums with the cosine cutoff of the second list.
    """
    from .hamiltonian import cosine_cutoff
    fc = cosine_cutoff(r, rcut, dcut) * mask
    env = jnp.sum(fc * yp["C"][sp_i, sp_j] * jnp.exp(-yp["upsilon"][sp_i, sp_j] * r), axis=1)
    safe = jnp.where(env > 0.0, env, 1.0)
    lam = yp["lambda0"][species] + jnp.where(env > 0.0, safe ** (1.0 / yp["m"][species]), 0.0)
    lam_ij = 0.5 * (lam[:, None] + lam[nbr_origin])
    return 0.5 * jnp.sum(fc * yp["B"][sp_i, sp_j] / r
                         * jnp.exp(-lam_ij * (r - 2.0 * yp["r_core"][sp_i, sp_j])), axis=1)


def taper(r1, r2, r):
    """Seventh-order polynomial switching off the Coulomb integral at r2."""
    d = (r2 - r1) ** 7
    t0 = (-35 * r1**3 * r2**4 + 21 * r1**2 * r2**5 + 7 * r1 * r2**6 + r2**7) / d
    t1 = 140 * r1**3 * r2**3 / d
    t2 = -210 * (r1**3 * r2**2 + r1**2 * r2**3) / d
    t3 = 140 * (r1**3 * r2 + 3 * r1**2 * r2**2 + r1 * r2**3) / d
    t4 = -35 * (r1**3 + 9 * r1**2 * r2 + r2**3) / d
    t5 = 84 * (r1**2 + 3 * r1 * r2 + r2**2) / d
    t6 = -70 * (r1 + r2) / d
    t7 = 20 / d
    poly = t0 + r * (t1 + r * (t2 + r * (t3 + r * (t4 + r * (t5 + r * (t6 + r * t7))))))
    return jnp.where(r > r2, 0.0, poly)


def frauenheim_coulomb(j_i, j_j, r):
    """Screened Coulomb integral between two exponential charge clouds whose
    widths are set by the onsite integrals J_ii and J_jj."""
    k = COULOMB_CONSTANT
    a = 3.2 * j_i
    b = 3.2 * j_j
    same = k / r - jnp.exp(-b * r / k) * (
        k / r + 11.0 / 16.0 * b + 3.0 / 16.0 * b**2 * r / k + b**3 * (r / k) ** 2 / 48.0
    )
    equal = jnp.abs(a - b) < 1e-12
    a_s = jnp.where(equal, a + 1.0, a)  # keep the unused branch finite
    c = a_s**2 - b**2
    diff = (
        k / r
        - jnp.exp(-a_s * r / k) * (0.5 * b**4 * a_s / c**2 - k * b**4 * (b**2 - 3 * a_s**2) / c**3 / r)
        - jnp.exp(-b * r / k) * (0.5 * a_s**4 * b / c**2 + k * a_s**4 * (a_s**2 - 3 * b**2) / c**3 / r)
    )
    return jnp.where(equal, same, diff)
