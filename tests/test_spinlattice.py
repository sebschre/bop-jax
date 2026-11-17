"""Coupled spin-lattice integrator.

Checked on a chain of spins whose exchange depends on the bond length, then on
the non-collinear BOP energy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax import md, spin, spinlattice
from bopjax.models import canonical_d

MOMENTS = jnp.array([2.0, 2.5, 1.5, 2.0])
MASS = 20.0


def chain_energy(position, axes):
    """Springs between neighbours of a ring of four atoms along x, with an
    exchange that grows as the bond shortens."""
    bond = jnp.roll(position[:, 0], -1) - position[:, 0]
    bond = bond.at[-1].add(4.0)  # the ring closes over the period
    exchange = 0.05 * jnp.exp(-2.0 * (bond - 1.0))
    return jnp.sum(0.5 * 3.0 * (bond - 1.0) ** 2
                   - exchange * jnp.sum(axes * jnp.roll(axes, -1, axis=0), axis=1))


def chain(position, axes, guess):
    energy, (d_position, d_axes) = jax.value_and_grad(chain_energy, argnums=(0, 1))(position, axes)
    return energy, -d_position, d_axes, MOMENTS, jnp.zeros(1), jnp.zeros((), jnp.int32)


def start():
    rng = np.random.default_rng(3)
    position = np.zeros((4, 3))
    position[:, 0] = np.arange(4.0) + 0.05 * rng.normal(size=4)
    axes = np.array([0.0, 0.0, 1.0]) + 0.5 * rng.normal(size=(4, 3))
    return jnp.array(position), jnp.array(axes)


def propagate(dt, steps, kT=0.0, **kwargs):
    init, apply = spinlattice.integrator(chain, dt, **kwargs)
    state = init(jax.random.PRNGKey(0), *start(), mass=MASS, kT=kT)
    final, _ = jax.lax.scan(lambda s, _: (apply(s), None), state, None, length=steps)
    return state, final


def total_energy(state):
    return float(state.energy + spinlattice.kinetic_energy(state))


def total_spin(state):
    return jnp.sum(state.moment[:, None] * state.axes, axis=0)


def test_energy_momentum_and_spin_are_conserved():
    first, last = propagate(0.02, 500, kT=0.01)
    # the two subsystems exchange energy ...
    assert abs(float(last.energy - first.energy)) > 1e-3
    assert float(jnp.max(jnp.abs(last.axes - first.axes))) > 0.1
    # ... while the sum, the momentum and the total spin stay (the errors are
    # of second order in the time step and do not grow)
    assert total_energy(last) == pytest.approx(total_energy(first), abs=2e-5)
    np.testing.assert_allclose(jnp.sum(last.momentum, axis=0), 0.0, atol=1e-13)
    np.testing.assert_allclose(total_spin(last), total_spin(first), atol=3e-4)
    np.testing.assert_allclose(jnp.linalg.norm(last.axes, axis=1), 1.0, atol=1e-13)


def test_second_order_and_corrector_passes():
    reference = propagate(0.0025, 800)[1]
    errors = [float(jnp.max(jnp.abs(propagate(dt, steps)[1].axes - reference.axes)))
              for dt, steps in ((0.04, 50), (0.02, 100))]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.2)
    # the passes of the midpoint iteration approach the time-reversible limit,
    # which conserves the energy better than the explicit scheme of one pass
    drift = []
    for correctors in (1, 3):
        first, last = propagate(0.08, 100, correctors=correctors)
        drift.append(abs(total_energy(last) - total_energy(first)))
    assert drift[1] < 0.2 * drift[0]


def test_fixed_atoms_reproduce_spin_dynamics():
    position, axes = start()
    _, coupled = propagate(0.02, 50, mobile=np.zeros(4))

    def evaluate(axes, position, guess):
        energy, _, gradient, moment, shifts, iterations = chain(position, axes, guess)
        return energy, gradient, moment, shifts, iterations

    init, apply = spin.integrator(evaluate, 0.0025)  # Heun's scheme, fine steps
    state = init(position, axes)
    for _ in range(400):
        state = apply(state)
    np.testing.assert_allclose(coupled.axes, state.axes, atol=2e-4)
    np.testing.assert_allclose(coupled.position, position, atol=0.0)


def test_fixed_axes_reproduce_velocity_verlet():
    dt, steps = 0.02, 50
    first, coupled = propagate(dt, steps, kT=0.01, free=np.zeros(4))
    force = jax.grad(lambda r: -chain_energy(r, first.axes))
    r, p, f = first.position, first.momentum, first.force
    for _ in range(steps):
        p = p + 0.5 * dt * f
        r = r + dt * p / MASS
        f = force(r)
        p = p + 0.5 * dt * f
    np.testing.assert_allclose(coupled.position, r, atol=1e-13)
    np.testing.assert_allclose(coupled.axes, first.axes, atol=0.0)


def test_collinear_spins_do_not_turn():
    # without spin-orbit coupling a collinear state has no torques, whatever
    # the atoms do
    init, apply = spinlattice.integrator(chain, 0.02)
    position, _ = start()
    axes = jnp.array([[0.6, 0.0, 0.8]] * 4) * jnp.array([1.0, -1.0, 1.0, 1.0])[:, None]
    state = init(jax.random.PRNGKey(1), position, axes, mass=MASS, kT=0.02)
    for _ in range(50):
        state = apply(state)
    assert float(jnp.max(jnp.abs(state.position - position))) > 0.01
    np.testing.assert_allclose(state.axes, axes, atol=1e-13)


# --- BOP energy: bcc ferromagnet with pair repulsion, four (001) planes ----------

STONER, A = 0.6, 2.8


@pytest.fixture(scope="module")
def bop_run():
    from bopjax import pairwise as pw

    positions = A * np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1], [0.5, 0.5, 1.5]])
    cell = A * np.diag([1.0, 1.0, 2.0])
    params = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[STONER]]),
                  repulsion={"pair": jnp.array([[[0.3, 16.0, 2.4]]]),
                             "density": jnp.zeros((1, 1, 3)), "embed": jnp.zeros((1, 2))})
    cfg = bj.BOPConfig(nmom=6, rcut=3.6, dcut=0.5, rcut2=3.6, dcut2=0.5, bandwidth="averaged",
                       repulsion=pw.POWERLAW, spin="noncollinear")
    topo = bj.build_topology(positions, [0] * 4, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy_fn = bj.make_energy_fn(topo, cfg)
    angle = np.radians([20.0, 20.0, -20.0, -20.0])
    axes = jnp.array(np.stack([np.sin(angle), 0 * angle, np.cos(angle)], axis=1))
    levels = bj.make_scf(energy_fn)(
        jnp.array(positions), jnp.array(cell), dict(params, spin_axes=axes),
        shifts0=bj.stoner_shifts(np.full((4, 1), 2.0), STONER)).shifts
    init, apply = spinlattice.nve(energy_fn, cell, params, 0.5 / md.TIME_UNIT_FS, tol=1e-6)
    states = [init(jax.random.PRNGKey(0), jnp.array(positions), axes, mass=55.85,
                   shifts0=levels)]
    apply = jax.jit(apply)
    for _ in range(10):
        states.append(apply(states[-1]))
    return energy_fn, cell, params, states


def test_bop_forces_come_from_the_spins(bop_run):
    energy_fn, cell, params, states = bop_run
    start = states[0]
    # exchange striction: the planes with parallel moments attract each other;
    # the same lattice with ferromagnetic order has no forces by symmetry
    np.testing.assert_allclose(start.force[:, 2], 0.0544 * np.array([1, -1, 1, -1]), atol=2e-4)
    np.testing.assert_allclose(start.force[:, :2], 0.0, atol=1e-8)
    np.testing.assert_allclose(jnp.sum(spin.torque(start), axis=0), 0.0, atol=1e-8)
    # the force is the derivative of the self-consistent energy
    potential = jax.jit(bj.make_potential(energy_fn, tol=1e-11))
    step = np.zeros((4, 3))
    step[1, 2] = 1e-4
    par = dict(params, spin_axes=start.axes)
    e_plus = potential(start.position + step, jnp.array(cell), par, start.shifts)[0]
    e_minus = potential(start.position - step, jnp.array(cell), par, start.shifts)[0]
    assert float(start.force[1, 2]) == pytest.approx(float(e_minus - e_plus) / 2e-4, abs=1e-6)


def test_bop_conserves_energy_while_spins_and_atoms_move(bop_run):
    _, _, _, states = bop_run
    start, end = states[0], states[-1]
    assert float(jnp.max(jnp.abs(end.axes - start.axes))) > 0.1
    assert float(spinlattice.kinetic_energy(end)) > 1e-5
    assert total_energy(end) == pytest.approx(total_energy(start), abs=5e-6)
    np.testing.assert_allclose(jnp.sum(end.momentum, axis=0), 0.0, atol=1e-12)


def test_nve_rejects_unsuitable_settings(bop_run):
    energy_fn, cell, params, _ = bop_run
    with pytest.raises(ValueError, match="alpha"):
        spinlattice.nve(energy_fn, cell, params, 0.01, kT=0.01)
