"""Adiabatic spin dynamics: the magnetic counterpart of ``bopjax.md``.

In a non-collinear calculation every atom carries a local spin axis ``e_i``
(``params["spin_axes"]``) and the self-consistent energy is a function
``E(e_1 .. e_N)``.  Its gradient gives the torque ``-e_i x dE/de_i`` in the
same way as the gradient with respect to positions gives the force, and the
axes follow the Landau-Lifshitz-Gilbert equation

    S_i de_i/dt = -e_i x dE/de_i - alpha |S_i| e_i x de_i/dt,
    S_i = hbar m_i / 2

where ``m_i = n_up - n_down`` is the moment along the axis in Bohr magnetons,
so that ``S_i e_i`` is the spin angular momentum of the atom.  This is the
adiabatic spin dynamics of Antropov et al., PRB 54, 1019 (1996): the electrons
(onsite levels, and with them the lengths of the moments) follow the
directions instantaneously, while the directions are slow variables.

* ``alpha = 0``  precession at constant energy: the counterpart of NVE.
* ``alpha > 0``  Gilbert damping; the energy falls until the torques vanish:
  the counterpart of a damped relaxation.
* ``kT > 0``     a fluctuating torque that obeys the fluctuation-dissipation
  theorem with the damping (stochastic LLG): the counterpart of a Langevin
  thermostat.  The directions sample exp(-E / kT).

The sign of ``m_i`` matters and is handled: an axis ``e`` with moment ``m``
describes the same state as ``-e`` with ``-m``, and both move alike.

The integrator is Heun's predictor-corrector scheme with the update written as
a rotation of each axis, so the axes stay unit vectors to machine precision.
It is of second order in ``dt`` and, with noise, converges to the Stratonovich
solution, which is the physical one.  A step costs two self-consistent
gradient evaluations; both are warm-started from the onsite levels carried in
the state, as in ``bopjax.md.nve``.

Units: time in the unit of ``bopjax.md`` (``md.TIME_UNIT_FS`` femtoseconds),
energies in eV.  Precession is fast: the frequency is ``2 |dE/de| / (hbar m)``,
about 0.15 rad/fs for a gradient of 0.1 eV and a moment of 2 Bohr magnetons,
so time steps are a fraction of a femtosecond.

Atoms whose moment vanishes have no direction to integrate; ``min_moment``
bounds |m| from below in the equation of motion.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .md import TIME_UNIT_FS
from .scf import make_potential

HBAR = 0.6582119569 / TIME_UNIT_FS  # eV * time unit
MU_B = 5.7883818060e-5  # Bohr magneton in eV / T


class SpinState(NamedTuple):
    axes: jnp.ndarray  # (N, 3) unit vectors
    gradient: jnp.ndarray  # (N, 3) dE/d(axes) at ``axes``
    moment: jnp.ndarray  # (N,) moments along the axes in Bohr magnetons
    energy: jnp.ndarray  # energy at ``axes``
    shifts: jnp.ndarray  # self-consistent onsite shifts at ``axes``
    shifts_history: jnp.ndarray  # (2, ...) those of the two steps before
    scf_iterations: jnp.ndarray  # Newton steps of the two evaluations of the last step
    position: jnp.ndarray  # (N, 3) atomic positions (fixed)
    key: jnp.ndarray  # random key for the thermal torques
    free: jnp.ndarray  # (N, 1), 1.0 for axes that move, 0.0 for fixed ones


def rotate(vectors, rotation):
    """Rotate each vector (N, 3) about ``rotation`` (N, 3) by the angle
    |rotation| (Rodrigues' formula).  The length is preserved exactly."""
    angle2 = jnp.sum(rotation**2, axis=-1, keepdims=True)
    small = angle2 < 1e-16
    angle = jnp.sqrt(jnp.where(small, 1.0, angle2))
    sinc = jnp.where(small, 1.0 - angle2 / 6.0, jnp.sin(angle) / angle)
    cosc = jnp.where(small, 0.5 - angle2 / 24.0, (1.0 - jnp.cos(angle)) / angle**2)
    wxv = jnp.cross(rotation, vectors)
    return vectors + sinc * wxv + cosc * jnp.cross(rotation, wxv)


def angular_velocity(axes, gradient, moment, alpha=0.0, min_moment=0.1):
    """Vector ``w`` (N, 3) with ``d(axes)/dt = w x axes`` for the
    Landau-Lifshitz-Gilbert equation, in rad per time unit.

    Only the part of ``w`` perpendicular to the axis moves it, but the part
    along it must be the same function of the axes in both stages of the
    integrator, so the gradient is used as it comes.  In a constant field the
    rotation about ``w`` is then the exact solution."""
    spin = 0.5 * HBAR * _bounded(moment, min_moment)[:, None]
    return (gradient / spin - alpha * jnp.cross(axes, gradient) / jnp.abs(spin)) / (
        1.0 + alpha**2)


def _bounded(moment, min_moment):
    sign = jnp.where(moment < 0.0, -1.0, 1.0)
    return sign * jnp.maximum(jnp.abs(moment), min_moment)


def _unit(axes):
    return axes / jnp.linalg.norm(axes, axis=-1, keepdims=True)


def integrator(evaluate, dt, alpha=0.0, kT=0.0, min_moment=0.1, free=None,
               extrapolate=True):
    """Landau-Lifshitz-Gilbert integrator for any energy model.

    evaluate : ``(axes, position, guess) -> (energy, gradient, moment, shifts,
               iterations)``.  ``gradient`` is dE/d(axes) (N, 3), ``moment``
               (N,) in Bohr magnetons.  ``shifts`` is internal state of the
               model whose previous values come back as ``guess`` (None on the
               first call); models without any return a dummy array.

    Returns ``(init_fn, apply_fn)``; see ``llg``.
    """
    def velocity(axes, gradient, moment, noise, mask):
        # the random torque enters like a gradient: <xi xi> = 2 alpha |S| kT / dt
        width = jnp.sqrt(alpha * HBAR * jnp.abs(_bounded(moment, min_moment)) * kT / dt)
        return mask * angular_velocity(axes, gradient + width[:, None] * noise, moment,
                                       alpha, min_moment)

    def init_fn(positions, axes, shifts0=None, key=None):
        positions = jnp.asarray(positions)
        axes = _unit(jnp.asarray(axes, dtype=positions.dtype))
        n = axes.shape[0]
        mask = (jnp.ones((n, 1), axes.dtype) if free is None
                else jnp.asarray(free, dtype=axes.dtype).reshape(n, 1))
        key = jax.random.PRNGKey(0) if key is None else key
        energy, gradient, moment, shifts, iterations = evaluate(axes, positions, shifts0)
        return SpinState(axes, gradient, moment, energy, shifts,
                         jnp.stack([shifts, shifts]), jnp.stack([iterations, iterations]),
                         positions, key, mask)

    def apply_fn(state):
        key, sub = jax.random.split(state.key)
        noise = jax.random.normal(sub, state.axes.shape, state.axes.dtype)
        s0, (s1, s2) = state.shifts, state.shifts_history
        guess = 3.0 * s0 - 3.0 * s1 + s2 if extrapolate else s0

        w0 = velocity(state.axes, state.gradient, state.moment, noise, state.free)
        trial = rotate(state.axes, dt * w0)
        _, gradient, moment, shifts, it_trial = evaluate(trial, state.position, guess)
        w1 = velocity(trial, gradient, moment, noise, state.free)
        axes = rotate(state.axes, 0.5 * dt * (w0 + w1))
        energy, gradient, moment, shifts, it_final = evaluate(axes, state.position, shifts)
        return SpinState(axes, gradient, moment, energy, shifts,
                         jnp.stack([s0, s1]), jnp.stack([it_trial, it_final]),
                         state.position, key, state.free)

    return init_fn, apply_fn


def llg(energy_fn, cell, params, dt, alpha=0.0, kT=0.0, tol=1e-7, max_iter=20,
        min_moment=0.1, free=None, extrapolate=True):
    """Spin dynamics of a non-collinear bopjax model at fixed atomic positions.

    energy_fn  : result of ``bopjax.make_energy_fn`` with ``spin="noncollinear"``
    params     : model parameters; ``params["spin_axes"]`` is not needed, the
                 axes are part of the state
    dt         : time step in ``md.TIME_UNIT_FS`` femtoseconds
    alpha      : Gilbert damping (dimensionless)
    kT         : temperature of the spin directions in eV; needs ``alpha > 0``
    min_moment : lower bound of |m| in the equation of motion (Bohr magnetons)
    free       : (N,) mask, 1 for axes that move and 0 for axes held fixed

    Returns ``(init_fn, apply_fn)``:

        state = init_fn(positions, axes, shifts0, key)
        state = apply_fn(state)        # jit-compatible

    ``shifts0`` is the starting guess of the onsite shifts and selects the
    magnetic solution: pass ``bopjax.stoner_shifts(moments, stoner)`` or
    converged shifts, since from zero the solver finds the non-magnetic state.
    ``tol`` and ``max_iter`` are those of ``bopjax.make_potential``.
    """
    if energy_fn.config.spin != "noncollinear":
        raise ValueError('spin dynamics needs BOPConfig(spin="noncollinear")')
    if kT > 0.0 and alpha <= 0.0:
        raise ValueError("a temperature needs damping: set alpha > 0")
    potential = make_potential(energy_fn, tol=tol, max_iter=max_iter)
    cell = None if cell is None else jnp.asarray(cell)
    value_and_grad = jax.value_and_grad(
        lambda axes, position, guess: potential(
            position, cell, dict(params, spin_axes=axes), guess), has_aux=True)

    def evaluate(axes, position, guess):
        (energy, aux), gradient = value_and_grad(axes, position, guess)
        return energy, gradient, aux["moment"], aux["shifts"], aux["scf_iterations"]

    return integrator(evaluate, dt, alpha, kT, min_moment, free, extrapolate)


def torque(state: SpinState):
    """Torques ``-e x dE/de`` on the spins in eV, (N, 3).  They vanish at a
    stationary magnetic configuration and, without spin-orbit coupling, sum to
    zero."""
    return -jnp.cross(state.axes, state.gradient)


def damping_power(state: SpinState, alpha, min_moment=0.1):
    """Rate ``-dE/dt`` at which Gilbert damping removes energy, in eV per time
    unit: ``alpha / (1 + alpha^2) * sum |e x dE/de|^2 / |S|``."""
    spin = 0.5 * HBAR * jnp.abs(_bounded(state.moment, min_moment))
    return alpha / (1.0 + alpha**2) * jnp.sum(jnp.sum(torque(state)**2, axis=1) / spin)


def effective_field(state: SpinState, min_moment=0.1):
    """Effective field ``-dE/d(m e)`` on each moment in Tesla, (N, 3)."""
    return -state.gradient / (MU_B * _bounded(state.moment, min_moment)[:, None])


def magnetisation(state: SpinState):
    """Total moment ``sum m_i e_i`` in Bohr magnetons, (3,)."""
    return jnp.sum(state.moment[:, None] * state.axes, axis=0)
