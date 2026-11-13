"""Comparison with BOPfox for the canonical d-band model (W_LIM_I, 6 moments).

Reference numbers:
  * perfect fcc: bopfox/BOPtest/tests/BOP/energy/bulk.fcc.1x1x1/runlog.reference
  * distorted fcc: produced by running the BOPfox binary with rcut = 3.3,
    dcut = 0.5, Broyden mixing, scftol = 1e-10 (BOPfox then uses Jii = 8).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax.models import canonical_d

A = 3.5
FCC = A * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
CELL = A * np.eye(3)


def test_perfect_fcc_matches_reference_log():
    cfg = bj.BOPConfig(nmom=6, rcut=3.01, dcut=0.0)
    topo = bj.build_topology(FCC, [0] * 4, cfg.rcut, cfg.nmom, CELL)
    energy = bj.make_energy_fn(topo, cfg)
    e, aux = energy(jnp.zeros((4, 1)), jnp.array(FCC), jnp.array(CELL), canonical_d())

    assert e / 4 == pytest.approx(-5.82994578275954, abs=1e-11)
    assert aux["fermi"] == pytest.approx(0.02525345, abs=1e-8)
    assert aux["a_inf"][0, 0] == pytest.approx(-0.31753865, abs=1e-8)
    assert aux["b_inf"][0, 0] == pytest.approx(1.32390275, abs=1e-8)
    assert aux["moments"][0, 0, 2] == pytest.approx(1.75271848053838, abs=1e-12)
    np.testing.assert_allclose(aux["n_shell"][:, 0], 4.8, atol=1e-10)


@pytest.fixture(scope="module")
def distorted():
    pos = FCC.copy()
    pos[0] += [0.15, -0.08, 0.05]
    pos[2] += [-0.05, 0.10, 0.02]
    cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5)
    topo = bj.build_topology(pos, [0] * 4, cfg.rcut, cfg.nmom, CELL)
    energy = bj.make_energy_fn(topo, cfg)
    scf = bj.make_scf(energy)
    params = canonical_d(hubbard=8.0)
    pos, cell = jnp.array(pos), jnp.array(CELL)
    result = scf(pos, cell, params, tol=1e-12)
    return energy, scf, params, pos, cell, result


def test_scf_matches_bopfox(distorted):
    *_, result = distorted
    assert result.converged
    assert result.energy / 4 == pytest.approx(-6.0112485448437187, abs=1e-10)
    np.testing.assert_allclose(
        result.shifts[:, 0],
        [-3.2427167768e-3, 4.1529574320e-3, -1.3968614396e-3, 4.8662078441e-4],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.aux["charge"],
        [2.001241198650e-4, -5.18733079637e-5, 2.22429422658e-5, -1.704937541867e-4],
        atol=1e-9,
    )


def test_newton_converges_quadratically(distorted):
    *_, result = distorted
    # BOPfox needs 12 Broyden steps for this case
    assert len(result.history) <= 5
    assert result.history[1] < 1e-2 * result.history[0]


def test_forces_match_bopfox(distorted):
    energy, _, params, pos, cell, result = distorted
    f = bj.make_forces(energy)(result.shifts, pos, cell, params)
    f_bopfox = np.array(
        [
            [4.9859884934, -2.4231962888, 1.2439578230],
            [0.8291859687, -0.3280010053, -0.7088860647],
            [-3.5410851762, 2.7658615584, -0.2853256569],
            [-2.2740892253, -0.0146642081, -0.2497461152],
        ]
    )
    # BOPfox's analytic forces differ from the exact derivative of its own
    # self-consistent energy at the 1e-7 level; see the finite-difference test.
    np.testing.assert_allclose(f, f_bopfox, atol=1e-6)
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-12)


def test_forces_are_total_derivative_of_scf_energy(distorted):
    """Stationarity in the shifts means no response term is needed."""
    energy, scf, params, pos, cell, result = distorted
    f = bj.make_forces(energy)(result.shifts, pos, cell, params)

    def e_scf(delta):
        p = pos.at[0, 0].add(delta)
        return scf(p, cell, params, shifts0=result.shifts, tol=1e-13).energy

    h = 1e-3
    fd = -(-e_scf(2 * h) + 8 * e_scf(h) - 8 * e_scf(-h) + e_scf(-2 * h)) / (12 * h)
    assert f[0, 0] == pytest.approx(fd, abs=1e-8)
