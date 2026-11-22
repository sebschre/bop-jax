"""Spin dynamics and spin-lattice dynamics (bopfox_rs.spin): bopjax
trajectories, Larmor precession, Gilbert damping, thermal equilibrium,
energy conservation.

    uv run pytest bopfox-rs/tests/test_spin.py
"""

import numpy as np
import pytest

bopfox_rs = pytest.importorskip("bopfox_rs")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import bopjax as bj  # noqa: E402
from bopjax import hamiltonian  # noqa: E402
from bopjax.models import IRON_CONFIG, iron  # noqa: E402
from bopfox_rs import G_SPIN, MU_B  # noqa: E402
from bopfox_rs.spin import (HBAR, TIME_UNIT_FS, SpinDynamics,  # noqa: E402
                            SpinLatticeDynamics)

A = 2.83
CFG = bj.BOPConfig(spin="noncollinear", **IRON_CONFIG)  # 9 moments


def potential(xi=0.0):
    cfg = bj.BOPConfig(spin="noncollinear", spin_orbit=xi > 0, **IRON_CONFIG)
    params = dict(iron(), spin_orbit=jnp.array([[xi]])) if xi else iron()
    return bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, params, hamiltonian.v0_bond))


def four_atoms(rattle=0.03):
    pos = A * np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 0, 0], [1.5, 0.5, 0.5]])
    pos = pos + np.random.default_rng(11).normal(scale=rattle, size=pos.shape)
    axes = np.array([[0.0, 0.0, 1.0], [0.15, 0.05, 1.0], [-0.1, 0.15, 1.0], [0.05, -0.15, 1.0]])
    return pos, np.diag([2 * A, A, A]), axes / np.linalg.norm(axes, axis=1, keepdims=True)


def converged_shifts(pot, pos, cell, axes, **kw):
    s0 = np.zeros((len(pos), 2, 1))
    s0[:, 0], s0[:, 1] = -1.25, 1.25
    out = pot.scf(pos, [0] * len(pos), cell, s0, axes=axes, tol=1e-11, **kw)
    assert out["converged"]
    return out["shifts"]


def one_atom_cell():
    """Primitive bcc cell: every image shares the axis, a rigid ferromagnet."""
    cell = 0.5 * A * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    return np.zeros((1, 3)), cell


def test_trajectory_matches_bopjax():
    from bopjax import spin as bspin
    pot = potential()
    pos, cell, axes = four_atoms()
    shifts = converged_shifts(pot, pos, cell, axes)
    dt = 0.05
    sd = SpinDynamics(pot, pos, [0] * 4, cell, axes, shifts, dt, tol=1e-11, on_failure="raise")
    topo = bj.build_topology(pos, [0] * 4, CFG.rcut, CFG.nmom, cell, rcut2=CFG.rcut2)
    energy = bj.make_energy_fn(topo, CFG, radial=hamiltonian.v0_bond)
    init, apply = bspin.llg(energy, cell, iron(), dt / TIME_UNIT_FS, tol=1e-11, max_iter=30)
    state = init(jnp.array(pos), jnp.array(axes), jnp.array(shifts))
    apply = jax.jit(apply)
    for _ in range(10):
        sd.step()
        state = apply(state)
    np.testing.assert_allclose(sd.axes, np.asarray(state.axes), atol=1e-9)
    assert sd.energy == pytest.approx(float(state.energy), abs=1e-9)


def precession_rate(pot, field_z, theta=np.radians(30.0), dt=0.5, steps=60):
    """Rate (rad/fs) and final polar angle of a rigid ferromagnet in a field along z."""
    pos, cell = one_atom_cell()
    field = np.array([0.0, 0.0, field_z])
    axes = np.array([[np.sin(theta), 0.0, np.cos(theta)]])
    shifts = converged_shifts(pot, pos, cell, axes, magnetic_field=field)
    sd = SpinDynamics(pot, pos, [0], cell, axes, shifts, dt, magnetic_field=field, tol=1e-11,
                      on_failure="raise")
    phi = []
    for _ in range(steps):
        sd.step()
        phi.append(np.arctan2(sd.axes[0, 1], sd.axes[0, 0]))
    rate = -np.polyfit(dt * np.arange(1, steps + 1), np.unwrap(phi), 1)[0]
    return rate, np.arccos(sd.axes[0, 2])


def test_larmor_precession():
    """A rigid ferromagnet in a field precesses about the field at constant
    angle, at the Larmor frequency g mu_B B / hbar as B -> 0.  The
    transverse part of the field also induces a transverse spin density
    (and the moment grows in the field), so the rate deviates from it in
    proportion to B (-0.53% at 50 T, -1.15% at 100 T for this model,
    independent of the time step: the response of the model, not the
    integrator)."""
    pot = potential()
    theta = np.radians(30.0)
    deviation = {}
    for b in (50.0, 100.0):
        rate, polar = precession_rate(pot, b, theta)
        larmor = G_SPIN * MU_B * b / HBAR
        deviation[b] = rate / larmor - 1.0
        assert abs(deviation[b]) < 1.5e-4 * b
        assert polar == pytest.approx(theta, abs=1e-6)
    assert 1.9 < deviation[100.0] / deviation[50.0] < 2.4


def test_damping_lowers_energy_at_the_damping_rate():
    pot = potential(xi=0.06)
    pos, cell, axes = four_atoms()
    shifts = converged_shifts(pot, pos, cell, axes)
    alpha, dt = 0.2, 0.05
    sd = SpinDynamics(pot, pos, [0] * 4, cell, axes, shifts, dt, alpha=alpha, tol=1e-11,
                      on_failure="raise")
    energies, power = [sd.energy], [sd.damping_power]
    for _ in range(15):
        sd.step()
        energies.append(sd.energy)
        power.append(sd.damping_power)
    energies, power = np.array(energies), np.array(power)
    assert np.all(np.diff(energies) < 0)
    # dE/dt = -damping power (trapezoidal rule over each step)
    # (the trapezoidal rule is good to a few percent at the largest torques)
    np.testing.assert_allclose(np.diff(energies), -0.5 * dt * (power[1:] + power[:-1]),
                               rtol=5e-2)


def test_precession_conserves_energy_to_second_order():
    pot = potential()
    pos, cell, axes = four_atoms()
    shifts = converged_shifts(pot, pos, cell, axes)
    drifts = []
    for dt, steps in ((0.04, 10), (0.02, 20)):
        sd = SpinDynamics(pot, pos, [0] * 4, cell, axes, shifts, dt, tol=1e-11,
                          on_failure="raise")
        e0 = sd.energy
        sd.run(steps)
        drifts.append(abs(sd.energy - e0))
    assert drifts[1] < 0.35 * drifts[0], drifts  # second order: a quarter


@pytest.mark.slow
def test_thermal_spin_samples_boltzmann():
    """With damping and stochastic torques a rigid moment in a field samples
    exp(-E / kT): <cos theta> is the Langevin function of mu B / kT."""
    pot = potential()
    pos, cell = one_atom_cell()
    field = np.array([0.0, 0.0, 1000.0])
    axes = np.array([[0.0, 0.0, 1.0]])
    shifts = converged_shifts(pot, pos, cell, axes, magnetic_field=field)
    kT = 0.1
    sd = SpinDynamics(pot, pos, [0], cell, axes, shifts, 0.5, alpha=0.5, kT=kT,
                      magnetic_field=field, tol=1e-9, rng=np.random.default_rng(3))
    sd.run(400)  # equilibrate
    cos, moments = [], []
    for _ in range(6000):
        sd.step()
        cos.append(sd.axes[0, 2])
        moments.append(sd.moment[0])
    x = 0.5 * G_SPIN * MU_B * field[2] * np.mean(moments) / kT
    langevin = 1.0 / np.tanh(x) - 1.0 / x
    assert np.mean(cos) == pytest.approx(langevin, abs=0.04)


def test_spin_lattice_conserves_energy_and_matches_bopjax():
    from bopjax import spinlattice
    pot = potential()
    pos, cell, axes = four_atoms()
    shifts = converged_shifts(pot, pos, cell, axes)
    dt, mass = 0.1, 55.845
    rng = np.random.default_rng(5)
    sl = SpinLatticeDynamics(pot, pos, [0] * 4, cell, axes, shifts, dt, mass,
                             temperature=300.0, tol=1e-11, rng=rng, on_failure="raise")
    momenta0 = sl.momenta.copy()
    topo = bj.build_topology(pos, [0] * 4, CFG.rcut, CFG.nmom, cell, rcut2=CFG.rcut2)
    energy = bj.make_energy_fn(topo, CFG, radial=hamiltonian.v0_bond)
    init, apply = spinlattice.nve(energy, cell, iron(), dt / TIME_UNIT_FS, tol=1e-11, max_iter=30)
    state = init(jax.random.PRNGKey(0), jnp.array(pos), jnp.array(axes), mass, 0.0,
                 jnp.array(shifts))
    # the same starting momenta: bopjax's unit of momentum is amu A / time unit
    state = state._replace(momentum=jnp.array(momenta0 * TIME_UNIT_FS))
    apply = jax.jit(apply)
    e0 = sl.energy + sl.kinetic_energy
    for _ in range(10):
        sl.step()
        state = apply(state)
    np.testing.assert_allclose(sl.positions, np.asarray(state.position), atol=1e-9)
    np.testing.assert_allclose(sl.axes, np.asarray(state.axes), atol=1e-8)
    # the same scheme: the same total energy, drift included (it is that of
    # the midpoint step of the fast precession and falls with the time step)
    total = sl.energy + sl.kinetic_energy
    assert total == pytest.approx(float(state.energy) + float(spinlattice.kinetic_energy(state)),
                                  abs=1e-8)
    assert abs(total - e0) < 2e-3
