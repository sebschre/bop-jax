"""Coupled spin-lattice dynamics: ``bopjax.md.nve`` and ``bopjax.spin.llg`` in
one integrator.

The self-consistent energy ``E(R, e)`` depends on the atomic positions and on
the spin axes.  Its two gradients drive the two subsystems,

    M_i d2R_i/dt2 = -dE/dR_i                       (forces depend on the spins)
    S_i de_i/dt   = -e_i x dE/de_i  (+ damping)    (torques depend on the positions)

and without damping the sum ``E + E_kin`` is conserved while energy moves
between the lattice and the spin directions.

No spin-orbit coupling is needed for this.  The coupling is the dependence of
the exchange interaction on the geometry (exchange striction): the bond
integrals change with distance, and with them the energy a pair of atoms gains
from aligning its moments.  What the model lacks without spin-orbit coupling
is a preferred direction in space for the spins.  The energy is unchanged when
all axes are rotated together, so the torques sum to zero and the forces exert
no torque on the lattice: energy is exchanged, angular momentum is not.  A
strictly collinear configuration has no torques at all and stays collinear
however the atoms move.

Damping and thermal torques act on the spins as in ``bopjax.spin``; the atoms
have no thermostat.  With ``alpha > 0`` and ``kT > 0`` the spins are coupled to
a bath of that temperature and the lattice exchanges energy with it only
through the spins.

The scheme splits the step as half kick, half drift, spin step, half drift,
half kick.  The atoms follow velocity Verlet; the axes take one step of the
midpoint rule at the positions of the middle of the step, each update applied
as a rotation.  The midpoint rule is implicit and is solved by ``correctors``
passes of fixed-point iteration, one energy evaluation each, and one more
evaluation at the end of the step gives the forces.  With one pass this is an
explicit second-order scheme at the cost of ``spin.llg``; with three the
result is close to the time-reversible limit, which conserves the total energy
much better and allows a time step twice as long.  The time step is set by the
precession, a fraction of a femtosecond.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .md import KB_EV
from .scf import make_potential
from .spin import HBAR, _bounded, _unit, angular_velocity, rotate


class SpinLatticeState(NamedTuple):
    position: jnp.ndarray  # (N, 3)
    momentum: jnp.ndarray  # (N, 3)
    force: jnp.ndarray  # (N, 3) -dE/d(position)
    mass: jnp.ndarray  # (N, 1)
    axes: jnp.ndarray  # (N, 3) unit vectors
    gradient: jnp.ndarray  # (N, 3) dE/d(axes)
    moment: jnp.ndarray  # (N,) moments along the axes in Bohr magnetons
    energy: jnp.ndarray  # potential energy at ``position`` and ``axes``
    shifts: jnp.ndarray  # self-consistent onsite shifts
    shifts_history: jnp.ndarray  # (2, ...) those of the two steps before
    scf_iterations: jnp.ndarray  # Newton steps of each energy evaluation of the last step
    key: jnp.ndarray  # random key for the thermal torques
    mobile: jnp.ndarray  # (N, 1), 1.0 for atoms that move
    free: jnp.ndarray  # (N, 1), 1.0 for axes that move


def integrator(evaluate, dt, alpha=0.0, kT=0.0, min_moment=0.1, mobile=None,
               free=None, extrapolate=True, correctors=3):
    """Spin-lattice integrator for any energy model.

    evaluate : ``(position, axes, guess) -> (energy, force, gradient, moment,
               shifts, iterations)`` with ``force = -dE/d(position)`` and
               ``gradient = dE/d(axes)``; the rest as in ``spin.integrator``.

    Returns ``(init_fn, apply_fn)``; see ``nve``.
    """
    def velocity(axes, gradient, moment, noise, mask):
        width = jnp.sqrt(alpha * HBAR * jnp.abs(_bounded(moment, min_moment)) * kT / dt)
        return mask * angular_velocity(axes, gradient + width[:, None] * noise, moment,
                                       alpha, min_moment)

    def init_fn(key, positions, axes, mass=1.0, kT=0.0, shifts0=None):
        """``kT`` is the starting temperature of the atoms (eV)."""
        positions = jnp.asarray(positions)
        axes = _unit(jnp.asarray(axes, dtype=positions.dtype))
        n = positions.shape[0]

        def mask(values):
            return (jnp.ones((n, 1), positions.dtype) if values is None
                    else jnp.asarray(values, dtype=positions.dtype).reshape(n, 1))

        moving, turning = mask(mobile), mask(free)
        mass = jnp.broadcast_to(jnp.asarray(mass, dtype=positions.dtype).reshape(-1, 1), (n, 1))
        key, sub = jax.random.split(key)
        momentum = jnp.sqrt(mass * kT) * jax.random.normal(sub, positions.shape, positions.dtype)
        drift = jnp.sum(moving * momentum, axis=0) / jnp.maximum(jnp.sum(moving), 1.0)
        momentum = momentum - drift
        momentum = moving * momentum
        energy, force, gradient, moment, shifts, iterations = evaluate(positions, axes, shifts0)
        return SpinLatticeState(
            positions, momentum, moving * force, mass, axes, gradient, moment, energy, shifts,
            jnp.stack([shifts, shifts]), jnp.stack([iterations] * (correctors + 1)), key, moving,
            turning)

    def apply_fn(state):
        key, sub = jax.random.split(state.key)
        noise = jax.random.normal(sub, state.axes.shape, state.axes.dtype)
        s0, (s1, s2) = state.shifts, state.shifts_history
        guess = 3.0 * s0 - 3.0 * s1 + s2 if extrapolate else s0

        momentum = state.momentum + 0.5 * dt * state.force
        middle = state.position + 0.5 * dt * momentum / state.mass
        position = state.position + dt * momentum / state.mass
        # midpoint rule for the axes at the positions of the middle of the step
        w, iterations = velocity(state.axes, state.gradient, state.moment, noise, state.free), []
        for _ in range(correctors):
            half = rotate(state.axes, 0.5 * dt * w)
            _, _, gradient, moment, guess, used = evaluate(middle, half, guess)
            w = velocity(half, gradient, moment, noise, state.free)
            iterations.append(used)
        axes = rotate(state.axes, dt * w)
        energy, force, gradient, moment, shifts, used = evaluate(position, axes, guess)
        iterations.append(used)
        force = state.mobile * force
        momentum = momentum + 0.5 * dt * force
        return SpinLatticeState(
            position, momentum, force, state.mass, axes, gradient, moment, energy, shifts,
            jnp.stack([s0, s1]), jnp.stack(iterations), key, state.mobile,
            state.free)

    return init_fn, apply_fn


def nve(energy_fn, cell, params, dt, alpha=0.0, kT=0.0, tol=1e-7, max_iter=20,
        min_moment=0.1, mobile=None, free=None, extrapolate=True, correctors=3):
    """Spin-lattice dynamics of a non-collinear bopjax model.

    energy_fn : result of ``bopjax.make_energy_fn`` with ``spin="noncollinear"``
    dt        : time step in ``md.TIME_UNIT_FS`` femtoseconds
    alpha, kT : Gilbert damping and temperature (eV) of the bath of the spins;
                both zero for dynamics at constant total energy
    mobile    : (N,) mask, 1 for atoms that move and 0 for atoms held fixed
    free      : (N,) mask, 1 for axes that move and 0 for axes held fixed
    correctors: passes of the midpoint iteration of the axes (see above); a
                step costs ``correctors + 1`` self-consistent evaluations

    Returns ``(init_fn, apply_fn)``:

        state = init_fn(key, positions, axes, mass, kT, shifts0)
        state = apply_fn(state)        # jit-compatible

    ``kT`` of ``init_fn`` is the starting temperature of the atoms and
    ``shifts0`` selects the magnetic solution, as in ``spin.llg``.  The solver
    inside takes plain Newton steps, so ``shifts0`` should be close: converge
    the starting configuration with ``bopjax.make_scf`` and pass its shifts.

    ``spin.torque``, ``spin.damping_power`` and ``spin.magnetisation`` accept
    the state of this integrator as well.
    """
    if energy_fn.config.spin != "noncollinear":
        raise ValueError('spin-lattice dynamics needs BOPConfig(spin="noncollinear")')
    if kT > 0.0 and alpha <= 0.0:
        raise ValueError("a temperature needs damping: set alpha > 0")
    potential = make_potential(energy_fn, tol=tol, max_iter=max_iter)
    cell = None if cell is None else jnp.asarray(cell)
    value_and_grad = jax.value_and_grad(
        lambda variables, guess: potential(
            variables[0], cell, dict(params, spin_axes=variables[1]), guess), has_aux=True)

    def evaluate(position, axes, guess):
        (energy, aux), (d_position, d_axes) = value_and_grad((position, axes), guess)
        return energy, -d_position, d_axes, aux["moment"], aux["shifts"], aux["scf_iterations"]

    return integrator(evaluate, dt, alpha, kT, min_moment, mobile, free, extrapolate,
                      correctors)


def kinetic_energy(state: SpinLatticeState):
    return 0.5 * jnp.sum(state.momentum**2 / state.mass)


def temperature(state: SpinLatticeState):
    """Instantaneous temperature of the atoms in Kelvin."""
    return 2.0 * kinetic_energy(state) / (3.0 * jnp.sum(state.mobile) * KB_EV)

