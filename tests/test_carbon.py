"""The fitted sp carbon model: structural order and diamond stability.

Fixed geometries at each phase's fitted equilibrium, so no scipy is needed.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax.hamiltonian import v0_bond
from bopjax.models import CARBON_CONFIG, carbon

FCC = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]])
A_DIAMOND = 3.5653


def diamond(a):
    return a * FCC, a * np.array([[0, 0, 0], [0.25, 0.25, 0.25]])


def lonsdaleite(a):
    ah = a / np.sqrt(2.0)
    cell = np.array([[ah, 0, 0], [-0.5 * ah, 0.5 * np.sqrt(3) * ah, 0],
                     [0, 0, np.sqrt(8 / 3) * ah]])
    frac = np.array([[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5],
                     [1 / 3, 2 / 3, 3 / 8], [2 / 3, 1 / 3, 7 / 8]])
    return cell, frac @ cell


def graphene(a):
    cell = np.array([[a, 0, 0], [-0.5 * a, 0.5 * np.sqrt(3) * a, 0], [0, 0, 8.0]])
    return cell, np.array([[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]]) @ cell


def sc(a):
    return a * np.eye(3), np.zeros((1, 3))


def solve(structure, strain=None):
    cell, pos = structure
    if strain is not None:
        cell, pos = cell @ (np.eye(3) + strain), pos @ (np.eye(3) + strain)
    cfg = bj.BOPConfig(**CARBON_CONFIG)
    topo = bj.build_topology(pos, [0] * len(pos), cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=v0_bond)
    pot = bj.make_potential(energy)
    e, _ = pot(jnp.asarray(pos), jnp.asarray(cell), carbon())
    return float(e) / len(pos), pot, (jnp.asarray(pos), jnp.asarray(cell))


@pytest.fixture(scope="module")
def e_diamond():
    return solve(diamond(A_DIAMOND))[0]


def test_structural_order(e_diamond):
    e_gr = solve(graphene(2.4477))[0]
    e_lon = solve(lonsdaleite(A_DIAMOND))[0]
    e_sc = solve(sc(1.9449))[0]
    # each phase at its own equilibrium: graphene +0.012, lonsdaleite
    # +0.003, sc +2.68 eV/atom above diamond
    assert 0.0 < e_gr - e_diamond < 0.06
    assert 0.0 < e_lon - e_diamond < 0.02
    assert 2.0 < e_sc - e_diamond < 3.5


def test_diamond_equilibrium_and_shear():
    """No pressure at the fitted lattice parameter; C' > 0."""
    _, pot, args = solve(diamond(A_DIAMOND))
    stress = bj.make_stress(pot)(*args, carbon()) * 160.21766
    assert np.allclose(np.asarray(stress), 0.0, atol=1.0)  # GPa
    d = 0.01
    e0 = solve(diamond(A_DIAMOND))[0]
    e1 = solve(diamond(A_DIAMOND), np.diag([d, d, -2 * d]))[0]
    assert e1 - e0 > 0
