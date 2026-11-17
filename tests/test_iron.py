"""The fitted iron model: structural order, moment and bcc stability.

Fixed geometries near each phase's equilibrium, so no scipy is needed.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax.hamiltonian import v0_bond
from bopjax.models import IRON_CONFIG, iron

A_BCC = 2.854
V0 = A_BCC**3 / 2


def bcc(a):
    return a * np.eye(3), a * np.array([[0, 0, 0], [0.5, 0.5, 0.5]])


def fcc(a):
    return a * np.eye(3), a * np.array([[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])


def hcp(a, ca=1.58):
    cell = np.array([[a, 0, 0], [-0.5 * a, 0.5 * np.sqrt(3) * a, 0], [0, 0, ca * a]])
    return cell, np.array([[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]]) @ cell


def solve(structure, moments=None, strain=None):
    cell, pos = structure
    if strain is not None:
        cell, pos = cell @ (np.eye(3) + strain), pos @ (np.eye(3) + strain)
    cfg = bj.BOPConfig(spin="none" if moments is None else "collinear", **IRON_CONFIG)
    topo = bj.build_topology(pos, [0] * len(pos), cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=v0_bond)
    params = iron()
    start = None
    if moments is not None:
        start = bj.stoner_shifts(np.asarray(moments, float)[:, None], params["stoner"][0, 0])
    res = bj.make_scf(energy)(jnp.asarray(pos), jnp.asarray(cell), params, shifts0=start,
                              tol=1e-10, max_iter=60)
    assert res.converged
    return res.energy / len(pos), res.aux


@pytest.fixture(scope="module")
def bcc_fm():
    return solve(bcc(A_BCC), (2.5, 2.5))


def test_bcc_ferromagnet_moment(bcc_fm):
    _, aux = bcc_fm
    assert np.allclose(np.asarray(aux["moment"]), 2.43, atol=0.03)


def test_structural_order(bcc_fm):
    e_fm = bcc_fm[0]
    e_hcp = solve(hcp((4 * 0.899 * V0 / (np.sqrt(3) * 1.58)) ** (1 / 3)))[0]
    e_fcc = solve(fcc((4 * 0.900 * V0) ** (1 / 3)))[0]
    e_bcc_nm = solve(bcc((2 * 0.93 * V0) ** (1 / 3)))[0]
    # fitted: hcp NM +69, fcc +107, bcc NM +309 meV/atom (at their own equilibria)
    assert 0.05 < e_hcp - e_fm < 0.09
    assert 0.08 < e_fcc - e_fm < 0.13
    assert e_hcp < e_fcc
    assert 0.25 < e_bcc_nm - e_fm < 0.37


def test_bcc_shear_stable():
    """C' > 0: tetragonal strain at constant volume costs energy."""
    d = 0.01
    strain = np.diag([d, d, -2 * d])
    e0 = solve(bcc(A_BCC), (2.5, 2.5))[0]
    e1 = solve(bcc(A_BCC), (2.5, 2.5), strain)[0]
    assert e1 - e0 > 0
