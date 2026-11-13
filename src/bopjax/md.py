"""Molecular dynamics interface for JAX-MD (https://github.com/jax-md/jax-md).

JAX-MD integrators take ``energy_fn(R, **kwargs) -> scalar`` and a ``shift_fn``
that moves positions.  ``jaxmd_energy_fn`` provides the first, with the onsite
levels made self-consistent inside the compiled function, and ``jaxmd_space``
the second.

Periodic boundaries are handled by bopjax itself through the lattice
translations stored in the topology, so positions must NOT be wrapped back
into the cell: use the free-space shift returned by ``jaxmd_space``.

Units: with positions in Angstrom, energies in eV and masses in atomic mass
units, JAX-MD's time unit is ``TIME_UNIT_FS`` femtoseconds.

Neighbour lists are static.  Build the topology with a ``skin`` and call
``bopjax.needs_rebuild(positions, topo)`` every few steps; when it returns
True, rebuild the topology and the energy function.

Warm start.  Self-consistency started from zero costs about three Newton
steps per force evaluation.  The onsite levels change little between MD steps,
so starting from the previous ones saves most of that.  A JAX-MD integrator
cannot carry the levels in its state, so two routes are offered:

* ``nve`` in this module is a velocity-Verlet integrator whose state holds the
  onsite shifts and extrapolates them to the next step.  No extra cost.
* ``jaxmd_energy_fn`` accepts ``shifts0`` as a keyword, which JAX-MD forwards
  from ``apply_fn(state, shifts0=...)``.  Get the shifts for the next step
  from ``jaxmd_shifts_fn``; that is a second, already converged solve.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .scf import make_potential

TIME_UNIT_FS = 10.180505710  # Angstrom * sqrt(amu / eV) in femtoseconds
KB_EV = 8.617333262e-5  # Boltzmann constant in eV / K


def jaxmd_energy_fn(energy_fn, cell, params, self_consistent=True, tol=1e-9,
                    max_iter=20):
    """Wrap a bopjax energy function for JAX-MD.

    energy_fn : result of ``bopjax.make_energy_fn``
    cell      : (3, 3) lattice vectors or None; a ``box`` keyword passed by
                JAX-MD at call time overrides it
    params    : model parameters; a ``params`` keyword overrides them

    The returned function is differentiable: JAX-MD obtains forces from
    ``jax.grad``, which is exact because the self-consistent energy is
    stationary in the onsite levels.
    """
    potential = make_potential(energy_fn, tol=tol, max_iter=max_iter,
                               self_consistent=self_consistent)
    cell = None if cell is None else jnp.asarray(cell)

    def energy(positions, **kwargs):
        box = kwargs.get("box", cell)
        return potential(positions, box, kwargs.get("params", params),
                         kwargs.get("shifts0"))[0]

    return energy


def jaxmd_shifts_fn(energy_fn, cell, params, tol=1e-9, max_iter=20):
    """``shifts(positions, shifts0=None)``: the self-consistent onsite shifts,
    for passing to a JAX-MD ``apply_fn`` as ``shifts0`` on the next step."""
    potential = make_potential(energy_fn, tol=tol, max_iter=max_iter)
    cell = None if cell is None else jnp.asarray(cell)
    return lambda positions, shifts0=None: potential(positions, cell, params, shifts0)[1]["shifts"]


def jaxmd_space():
    """``(displacement_fn, shift_fn)`` to use with bopjax energies: free space,
    so that atoms are never wrapped across the cell boundary."""
    from jax_md import space

    return space.free()


class NVEState(NamedTuple):
    position: jnp.ndarray
    momentum: jnp.ndarray
    force: jnp.ndarray
    mass: jnp.ndarray  # (N, 1)
    shifts: jnp.ndarray  # self-consistent onsite shifts at ``position``
    shifts_history: jnp.ndarray  # (2, ...) those of the two steps before
    energy: jnp.ndarray  # potential energy at ``position``
    scf_iterations: jnp.ndarray  # Newton steps the last force evaluation took


def nve(energy_fn, cell, params, dt, tol=1e-7, max_iter=20, extrapolate=True):
    """Velocity-Verlet integrator with warm-started self-consistency.

    Returns ``(init_fn, apply_fn)`` in the style of JAX-MD:

        state = init_fn(key, positions, mass, kT)
        state = apply_fn(state)

    Each force evaluation starts the Newton solver from the shifts of the
    previous step, or with ``extrapolate`` from the quadratic extrapolation of
    the last three steps.  The solver still converges to ``tol`` (the largest
    remaining dE/dshift, in electrons), so the accuracy does not depend on the
    guess; ``state.scf_iterations`` reports the Newton steps taken.
    """
    potential = make_potential(energy_fn, tol=tol, max_iter=max_iter)
    cell = None if cell is None else jnp.asarray(cell)
    value_and_grad = jax.value_and_grad(
        lambda r, guess: potential(r, cell, params, guess), has_aux=True)

    def evaluate(position, guess):
        (energy, aux), grad = value_and_grad(position, guess)
        return energy, -grad, aux["shifts"], aux["scf_iterations"]

    def init_fn(key, positions, mass=1.0, kT=0.0):
        positions = jnp.asarray(positions)
        mass = jnp.broadcast_to(jnp.asarray(mass, dtype=positions.dtype).reshape(-1, 1),
                                (positions.shape[0], 1))
        momentum = jnp.sqrt(mass * kT) * jax.random.normal(key, positions.shape, positions.dtype)
        momentum = momentum - jnp.mean(momentum, axis=0, keepdims=True)  # no drift
        energy, force, shifts, iterations = evaluate(positions, None)
        return NVEState(positions, momentum, force, mass, shifts,
                        jnp.stack([shifts, shifts]), energy, iterations)

    def apply_fn(state):
        momentum = state.momentum + 0.5 * dt * state.force
        position = state.position + dt * momentum / state.mass
        s0, (s1, s2) = state.shifts, state.shifts_history
        guess = 3.0 * s0 - 3.0 * s1 + s2 if extrapolate else s0
        energy, force, shifts, iterations = evaluate(position, guess)
        momentum = momentum + 0.5 * dt * force
        return NVEState(position, momentum, force, state.mass, shifts,
                        jnp.stack([s0, s1]), energy, iterations)

    return init_fn, apply_fn


def kinetic_energy(state: NVEState):
    return 0.5 * jnp.sum(state.momentum**2 / state.mass)


def temperature(state: NVEState):
    """Instantaneous temperature in Kelvin (three degrees of freedom per atom)."""
    return 2.0 * kinetic_energy(state) / (3.0 * state.position.shape[0] * KB_EV)
