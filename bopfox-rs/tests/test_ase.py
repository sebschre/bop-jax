"""The ASE calculator: forces and stress against ASE's numerical derivatives,
energy conservation in molecular dynamics, warm start, relaxation.

    uv run pytest bopfox-rs/tests/test_ase.py
"""

import numpy as np
import pytest

ase = pytest.importorskip("ase")
bopfox_rs = pytest.importorskip("bopfox_rs")

from ase import units  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402
from ase.optimize import BFGS  # noqa: E402

import bopjax as bj  # noqa: E402
from bopjax import hamiltonian  # noqa: E402
from bopjax.models import IRON_CONFIG, iron  # noqa: E402
from bopfox_rs.calculator import BOPCalculator  # noqa: E402


def iron_calc(**kwargs):
    cfg = bj.BOPConfig(spin="collinear", **IRON_CONFIG)
    return BOPCalculator.from_bopjax(cfg, iron(), hamiltonian.v0_bond, **kwargs)


def iron_cell(n=2, a=2.83, rattle=0.05):
    atoms = bulk("Fe", "bcc", a=a, cubic=True).repeat(n)
    atoms.rattle(rattle, seed=3)
    atoms.set_initial_magnetic_moments([2.5] * len(atoms))
    return atoms


@pytest.mark.parametrize("condition", ["stationary", "potential"])
def test_forces_and_stress_match_numerical(condition):
    atoms = iron_cell()
    atoms.calc = iron_calc(condition=condition, tol=1e-11)
    f = atoms.get_forces()
    s = atoms.get_stress()
    assert np.all(atoms.get_magnetic_moments() > 1.5)
    f_num = atoms.calc.calculate_numerical_forces(atoms, d=1e-4)
    s_num = atoms.calc.calculate_numerical_stress(atoms, d=1e-5)
    np.testing.assert_allclose(f, f_num, atol=1e-6)
    np.testing.assert_allclose(s, s_num, atol=1e-7)


def test_md_conserves_energy_and_warm_start_saves_work():
    atoms = iron_cell(rattle=0.0)
    MaxwellBoltzmannDistribution(atoms, temperature_K=600, rng=np.random.default_rng(1))
    calc = iron_calc(tol=1e-9)
    atoms.calc = calc
    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
    energies, evaluations = [], []

    def record():
        energies.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        evaluations.append(calc.scf_info["evaluations"])

    dyn.attach(record, interval=1)
    dyn.run(60)
    drift = max(energies) - min(energies)
    assert drift < 2e-4 * len(atoms), f"total energy varies by {drift} eV"
    # warm start, extrapolation and the carried Anderson history: a
    # fraction of the evaluations of a cold start at the same configuration
    cold = iron_calc(tol=1e-9, warm_start=False)
    final = atoms.copy()
    final.calc = cold
    final.get_potential_energy()
    cold = cold.scf_info["evaluations"]
    assert np.mean(evaluations[5:]) < 0.4 * cold, (cold, evaluations)


def test_relaxation_of_a_vacancy():
    atoms = iron_cell(n=3, rattle=0.0)
    del atoms[13]
    atoms.calc = iron_calc(tol=1e-9)
    e0 = atoms.get_potential_energy()
    opt = BFGS(atoms, logfile=None)
    assert opt.run(fmax=1e-3, steps=200)
    assert atoms.get_potential_energy() < e0
    assert np.abs(atoms.get_forces()).max() < 1e-3


def test_cold_start_without_moments_stays_nonmagnetic():
    atoms = iron_cell()
    atoms.set_initial_magnetic_moments(None)
    atoms.calc = iron_calc()
    atoms.get_potential_energy()
    assert np.abs(atoms.get_magnetic_moments()).max() < 1e-6


def test_noncollinear_with_spin_orbit():
    """Axes from vector initial moments; forces against numerical ones and
    the magnetic anisotropy: with spin-orbit coupling the energy depends on
    a common rotation of the moments."""
    cfg = bj.BOPConfig(spin="noncollinear", spin_orbit=True, **dict(IRON_CONFIG, nmom=7))
    params = dict(iron(), spin_orbit=np.array([[0.06]]))
    atoms = iron_cell(n=1, rattle=0.03)
    tilted = np.array([[0.0, 0.3, 2.4], [0.2, -0.1, 2.5]])
    atoms.set_initial_magnetic_moments(None)  # scalar -> vector moments
    atoms.set_initial_magnetic_moments(tilted)
    atoms.calc = BOPCalculator.from_bopjax(cfg, params, hamiltonian.v0_bond, tol=1e-11)
    f = atoms.get_forces()
    np.testing.assert_allclose(f, atoms.calc.calculate_numerical_forces(atoms, d=1e-4), atol=1e-6)
    m = atoms.get_magnetic_moments()
    assert m.shape == (2, 3)
    np.testing.assert_allclose(np.cross(m, tilted), 0.0, atol=1e-9)  # along the axes
    g = atoms.calc.results["axes_gradient"]
    assert np.abs(g).max() > 1e-6  # anisotropy: torques on the tilted moments
    e_tilted = atoms.get_potential_energy()
    atoms.set_initial_magnetic_moments([[0.0, 0.0, 2.5]] * 2)
    assert abs(atoms.get_potential_energy() - e_tilted) > 1e-7


def test_magnetic_field_moment_and_energy():
    """In a field the energy falls by about mu_B B.M, and the reported moment
    is -(1/mu_B) dE/dB of the self-consistent energy."""
    from bopfox_rs import MU_B
    atoms = iron_cell(rattle=0.02)
    e = {}
    for b in (-2.0, 0.0, 2.0):
        atoms.calc = iron_calc(tol=1e-11, magnetic_field=[0.0, 0.0, b])
        e[b] = atoms.get_potential_energy()
        if b == 0.0:
            m = atoms.calc.results["magnetic_moment"].sum(axis=0)
    fd = -(e[2.0] - e[-2.0]) / 4.0 / MU_B
    assert m[2] == pytest.approx(fd, rel=1e-4)
    assert m[2] > 30  # 16 atoms of about 2.3 Bohr magnetons
