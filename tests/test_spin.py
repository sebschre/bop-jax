"""Spin dynamics (Landau-Lifshitz-Gilbert) integrator.

The integrator is checked on model energies with known solutions, then on the
non-collinear BOP energy of the distorted fcc cell of test_features.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax import md, spin
from bopjax.models import canonical_d

FIELD = 0.1  # eV


def model(energy, moments):
    """``evaluate`` for spin.integrator from ``energy(axes)`` and fixed moments."""
    value_and_grad = jax.value_and_grad(energy)
    moments = jnp.asarray(moments, dtype=float)

    def evaluate(axes, position, guess):
        e, g = value_and_grad(axes)
        return e, g, moments, jnp.zeros(1), jnp.zeros((), jnp.int32)

    return evaluate


def zeeman(axes):
    return -FIELD * jnp.sum(axes[:, 2])


def propagate(evaluate, axes, dt, steps, **kwargs):
    init, apply = spin.integrator(evaluate, dt, **kwargs)
    state = init(jnp.zeros((len(axes), 3)), jnp.asarray(axes), key=jax.random.PRNGKey(1))
    return jax.lax.scan(lambda s, _: (apply(s), s.axes), state, None, length=steps)


TILTED = np.array([[np.sin(1.0), 0.0, np.cos(1.0)]])


@pytest.mark.parametrize("moment", [2.0, -2.0])
def test_precession_in_a_field(moment):
    # d(S e)/dt = e x B with S = hbar m / 2: the angle about the field is -B t / S
    dt, steps = 0.05, 200
    state, _ = propagate(model(zeeman, [moment]), TILTED, dt, steps)
    phi = -FIELD * dt * steps / (0.5 * spin.HBAR * moment)
    expected = [np.sin(1.0) * np.cos(phi), np.sin(1.0) * np.sin(phi), np.cos(1.0)]
    np.testing.assert_allclose(state.axes[0], expected, atol=1e-10)
    assert float(state.energy) == pytest.approx(-FIELD * np.cos(1.0), abs=1e-13)


@pytest.mark.parametrize("moment", [2.0, -2.0])
def test_damping_in_a_field(moment):
    # tan(theta / 2) decays with the rate alpha / (1 + alpha^2) * B / |S| for
    # either sign of the moment: damping lowers the energy, it does not follow m
    alpha, time = 0.3, 2.0
    rate = alpha / (1.0 + alpha**2) * FIELD / (0.5 * spin.HBAR * 2.0)
    exact = np.cos(2.0 * np.arctan(np.tan(0.5) * np.exp(-rate * time)))
    errors = []
    for steps in (50, 100):
        state, _ = propagate(model(zeeman, [moment]), TILTED, time / steps, steps, alpha=alpha)
        errors.append(abs(float(state.axes[0, 2]) - exact))
    assert errors[0] < 1e-4
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.15)  # second order


def test_heisenberg_ring_conserves_energy_and_total_spin():
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(6, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    moments = np.array([2.0, 2.5, -2.0, 1.5, 2.0, -3.0])

    def heisenberg(e):  # exchange between the true spin directions sign(m) e
        n = e * jnp.sign(moments)[:, None]
        return -0.05 * jnp.sum(n * jnp.roll(n, 1, axis=0))

    drift, total = [], []
    for steps in (200, 400):
        state, _ = propagate(model(heisenberg, moments), axes, 4.0 / steps, steps)
        start = float(heisenberg(jnp.array(axes)))
        drift.append(abs(float(state.energy) - start))
        np.testing.assert_allclose(jnp.linalg.norm(state.axes, axis=1), 1.0, atol=1e-13)
        total.append(float(jnp.max(jnp.abs(spin.magnetisation(state) - moments @ axes))))
        np.testing.assert_allclose(jnp.sum(spin.torque(state), axis=0), 0.0, atol=1e-14)
    # both are constants of the motion; the scheme keeps them to second order
    assert drift[0] < 1e-5 and drift[1] < 0.3 * drift[0]
    assert total[0] < 1e-3 and total[1] < 0.3 * total[0]
    # the axes did move
    assert float(jnp.max(jnp.abs(state.axes - axes))) > 0.5


def test_fixed_axes_stay():
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.0, 0.8]])
    energy = lambda e: -0.05 * jnp.sum(e * jnp.roll(e, 1, axis=0))  # noqa: E731
    state, _ = propagate(model(energy, [2.0] * 3), axes, 0.05, 50, alpha=0.2,
                         free=[1, 0, 1])
    np.testing.assert_array_equal(state.axes[1], axes[1])
    assert float(jnp.abs(state.axes[0] - axes[0]).max()) > 0.1


def test_thermal_directions_follow_boltzmann():
    # independent moments in a field: <cos theta> = coth(x) - 1/x, x = B / kT
    n, x = 4000, 2.0
    axes = np.tile([0.0, 0.0, 1.0], (n, 1))
    _, path = propagate(model(zeeman, np.full(n, 2.0)), axes, 0.02, 1500, alpha=0.5,
                        kT=FIELD / x)
    np.testing.assert_allclose(jnp.linalg.norm(path[-1], axis=1), 1.0, atol=1e-12)
    cos = float(jnp.mean(path[500:, :, 2]))
    assert cos == pytest.approx(1.0 / np.tanh(x) - 1.0 / x, abs=5e-3)
    assert float(jnp.mean(path[500:, :, 0])) == pytest.approx(0.0, abs=5e-3)


# --- bond-order potential --------------------------------------------------------

A = 3.5
CELL = A * np.eye(3)
POS = A * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
POS[0] += [0.15, -0.08, 0.05]
POS[2] += [-0.05, 0.10, 0.02]
AXES = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [2.0, 1.0, 0.0], [-2.0, 0.5, 0.0]])
STONER = 0.95
PARAMS = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[STONER]]))


def bop_energy(spin_mode="noncollinear"):
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged", spin=spin_mode)
    topo = bj.build_topology(POS, [0] * 4, cfg.rcut, cfg.nmom, CELL)
    return bj.make_energy_fn(topo, cfg)


@pytest.fixture(scope="module")
def precession():
    init, apply = spin.llg(bop_energy(), CELL, PARAMS, 0.2 / md.TIME_UNIT_FS)
    init, apply = jax.jit(init), jax.jit(apply)
    start = init(jnp.array(POS), AXES, bj.stoner_shifts(np.full((4, 1), 2.0), STONER))
    states = [start]
    for _ in range(4):
        states.append(apply(states[-1]))
    return init, apply, states


def test_bop_start_is_the_self_consistent_state(precession):
    _, _, states = precession
    # energy and moments of test_noncollinear_matches_bopfox
    assert float(states[0].energy) / 4 == pytest.approx(-4.883417474987876, abs=1e-9)
    np.testing.assert_allclose(states[0].moment, [2.308853, 2.396763, 2.356156, 2.028613],
                               atol=1e-6)
    np.testing.assert_allclose(jnp.sum(spin.torque(states[0]), axis=0), 0.0, atol=1e-8)


def test_bop_precession_conserves_energy(precession):
    _, _, states = precession
    end = states[-1]
    assert float(jnp.max(jnp.abs(end.axes - states[0].axes))) > 0.02
    assert float(end.energy) == pytest.approx(float(states[0].energy), abs=1e-6)
    np.testing.assert_allclose(jnp.linalg.norm(end.axes, axis=1), 1.0, atol=1e-14)
    # warm start: fewer Newton steps than the five of the first solve
    assert int(jnp.max(end.scf_iterations)) <= 3


def test_bop_axis_and_moment_can_be_reversed_together(precession):
    init, apply, states = precession
    moments0 = np.array([2.0, 2.0, 2.0, -2.0])
    flip = np.array([1.0, 1.0, 1.0, -1.0])
    state = init(jnp.array(POS), AXES * flip[:, None],
                 bj.stoner_shifts(moments0[:, None], STONER))
    for _ in range(2):
        state = apply(state)
    np.testing.assert_allclose(state.moment, states[2].moment * flip, atol=1e-6)
    np.testing.assert_allclose(state.axes, states[2].axes * flip[:, None], atol=1e-6)
    assert float(state.energy) == pytest.approx(float(states[2].energy), abs=1e-7)


def test_bop_damping_removes_the_dissipated_energy():
    dt, alpha = 0.5 / md.TIME_UNIT_FS, 1.0
    init, apply = spin.llg(bop_energy(), CELL, PARAMS, dt, alpha=alpha)
    apply = jax.jit(apply)
    state = init(jnp.array(POS), AXES, bj.stoner_shifts(np.full((4, 1), 2.0), STONER))
    for _ in range(2):
        new = apply(state)
        # dE/dt = -damping_power, integrated over the step by the trapezoidal rule
        lost = 0.5 * dt * float(spin.damping_power(state, alpha) + spin.damping_power(new, alpha))
        assert lost > 1e-2
        assert float(state.energy - new.energy) == pytest.approx(lost, rel=5e-3)
        state = new


def test_llg_rejects_unsuitable_settings():
    with pytest.raises(ValueError, match="noncollinear"):
        spin.llg(bop_energy("collinear"), CELL, PARAMS, 0.01)
    with pytest.raises(ValueError, match="alpha"):
        spin.llg(bop_energy(), CELL, PARAMS, 0.01, kT=0.01)
