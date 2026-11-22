"""bopfox-rs against bopjax: energy, forces, virial, dE/dshifts and the
quantities along the way, for the models both codes support.

Run from the bop-jax root after ``maturin develop`` (see bopfox-rs/README.md):
    uv run pytest bopfox-rs/tests
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax import hamiltonian, pairwise
from bopjax.models import (CARBON_CONFIG, IRON_CONFIG, IRON_EGOROV_CONFIG, canonical_d,
                           carbon, iron, iron_egorov)

bopfox_rs = pytest.importorskip("bopfox_rs")

RNG = np.random.default_rng(7)


def lattice(base, a, n, noise):
    grid = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    pos = a * (grid[:, None, :] + np.asarray(base)[None]).reshape(-1, 3)
    return pos + RNG.normal(scale=noise, size=pos.shape), a * n * np.eye(3)


FCC = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
BCC = [[0, 0, 0], [0.5, 0.5, 0.5]]
DIAMOND = FCC + [[x + 0.25 for x in p] for p in FCC]


def compare(cfg, params, radial, pos, cell, shifts, tol=1e-9):
    n = len(pos)
    topo = bj.build_topology(pos, [0] * n, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=radial)
    shifts = np.asarray(shifts).reshape(energy.shift_shape)
    p, c, s = jnp.array(pos), jnp.array(cell), jnp.array(shifts)

    e_fn = lambda s_, p_, c_: energy(s_, p_, c_, params)[0]  # noqa: E731
    e_ref, aux = energy(s, p, c, params)
    g_shift, g_pos = jax.grad(e_fn, argnums=(0, 1))(s, p, c)

    def strained(eps):
        d = jnp.eye(3) + 0.5 * (eps + eps.T)
        return e_fn(s, p @ d, c @ d)

    virial = jax.grad(strained)(jnp.zeros((3, 3)))

    rs = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, params, radial))
    out = rs.compute(pos, [0] * n, cell, shifts)

    scale = max(1.0, abs(float(e_ref)))
    assert out["energy"] == pytest.approx(float(e_ref), abs=tol * scale)
    assert out["fermi"] == pytest.approx(float(aux["fermi"]), abs=1e-9)
    np.testing.assert_allclose(out["forces"], -np.asarray(g_pos), atol=tol * 10)
    np.testing.assert_allclose(out["virial"], np.asarray(virial), atol=tol * 10 * n)
    np.testing.assert_allclose(out["shift_gradient"].reshape(shifts.shape), np.asarray(g_shift),
                               atol=tol * 10)
    np.testing.assert_allclose(out["energy_atom"], np.asarray(aux["energy_atom"]), atol=tol * 10)
    return out, aux


def test_canonical_d_band():
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged")
    pos, cell = lattice(FCC, 3.5, 2, 0.05)
    compare(cfg, canonical_d(hubbard=8.0), hamiltonian.powerlaw, pos, cell,
            RNG.normal(scale=0.01, size=(32, 1)))


@pytest.mark.parametrize("basis", [("d",), ("s", "d")])
def test_magnetic_iron(basis):
    cfg = bj.BOPConfig(spin="collinear", basis=basis, **IRON_CONFIG)
    pos, cell = lattice(BCC, 2.83, 2, 0.04)
    n, nsh = len(pos), len(basis)
    split = np.zeros((n, 2, nsh))
    split[:, 0, -1], split[:, 1, -1] = -1.2, 1.2  # exchange splitting of the d shell
    shifts = split + RNG.normal(scale=0.01, size=split.shape)
    out, aux = compare(cfg, iron(basis=basis), hamiltonian.v0_bond, pos, cell, shifts)
    np.testing.assert_allclose(out["moment"], np.asarray(aux["moment"]), atol=1e-9)


def test_carbon_sp():
    cfg = bj.BOPConfig(**CARBON_CONFIG)
    pos, cell = lattice(DIAMOND, 3.57, 1, 0.04)
    compare(cfg, carbon(), hamiltonian.v0_bond, pos, cell,
            RNG.normal(scale=0.02, size=(8, 2)))


def test_egorov_iron_without_yukawa():
    conf = dict(IRON_EGOROV_CONFIG, yukawa=False)
    cfg = bj.BOPConfig(spin="collinear", **conf)
    pos, cell = lattice(BCC, 2.84, 2, 0.03)
    shifts = np.zeros((len(pos), 2, 1))
    shifts[:, 0], shifts[:, 1] = -1.0, 1.0
    compare(cfg, iron_egorov(), pairwise.exp_sum, pos, cell, shifts)
