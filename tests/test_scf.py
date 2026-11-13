"""Newton self-consistency beyond the single-species d-band case."""

import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax.models import canonical_d

A = 3.5
CELL = jnp.array(A * np.eye(3))
POS = A * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
POS[0] += [0.15, -0.08, 0.05]
POS[2] += [-0.05, 0.10, 0.02]


def fd_force(scf, pos, params, shifts, i, d, h=1e-3):
    def e(delta):
        return scf(pos.at[i, d].add(delta), CELL, params, shifts0=shifts, tol=1e-13).energy

    return -(-e(2 * h) + 8 * e(h) - 8 * e(-h) + e(-2 * h)) / (12 * h)


def test_two_species_charge_transfer():
    cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5)
    topo = bj.build_topology(POS, [0, 1, 0, 1], cfg.rcut, cfg.nmom, np.array(CELL))
    energy = bj.make_energy_fn(topo, cfg)
    scf = bj.make_scf(energy)
    params = canonical_d(n_electrons=[4.0, 5.6], onsite=[0.3, -0.2], hubbard=[6.0, 8.0])
    pos = jnp.array(POS)
    res = scf(pos, CELL, params, tol=1e-11)

    assert res.converged and len(res.history) <= 5
    assert float(jnp.sum(res.aux["charge"])) == pytest.approx(0.0, abs=1e-9)
    f = bj.make_forces(energy)(res.shifts, pos, CELL, params)
    assert float(f[1, 0]) == pytest.approx(fd_force(scf, pos, params, res.shifts, 1, 0), abs=1e-8)


def test_two_shells_per_atom():
    cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5, basis=("s", "d"))
    topo = bj.build_topology(POS, [0] * 4, cfg.rcut, cfg.nmom, np.array(CELL))
    energy = bj.make_energy_fn(topo, cfg)
    scf = bj.make_scf(energy)

    def channel(prefactor, power):
        return jnp.array([[[prefactor, 2.081, power]]])

    params = {
        "onsite": jnp.array([[0.4, 0.0]]),
        "n0": jnp.array([[1.0, 4.8]]),
        "hubbard": jnp.array([6.0]),
        "bond": {
            "sss": channel(-1.0, 2.0),
            "sds": channel(-0.9, 3.5),
            "dds": channel(-1.458, 5.0),
            "ddp": channel(0.972, 5.0),
            "ddd": channel(-0.243, 5.0),
        },
    }
    pos = jnp.array(POS)
    res = scf(pos, CELL, params, tol=1e-11)

    assert res.converged and len(res.history) <= 8
    assert res.shifts.shape == (4, 2)
    f = bj.make_forces(energy)(res.shifts, pos, CELL, params)
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-10)
    assert float(f[0, 1]) == pytest.approx(fd_force(scf, pos, params, res.shifts, 0, 1), abs=1e-8)
