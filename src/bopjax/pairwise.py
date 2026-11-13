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
