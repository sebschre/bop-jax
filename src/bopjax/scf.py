"""Self-consistent onsite levels by Newton's method, and derived quantities.

The residual is g = dE/d(shifts); the Jacobian is the Hessian d2E/d(shifts)^2.
Both come from automatic differentiation of the energy function.

Two solvers are provided:

* ``make_scf``       Python loop with a damped fallback and iteration history.
* ``make_potential`` fully jit-compiled ``E(positions, cell, params)`` with the
  self-consistency inside, so that ``jax.grad`` gives forces directly.  This is
  the form molecular dynamics codes need.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class SCFResult:
    shifts: jnp.ndarray
    energy: float
    aux: dict
    converged: bool
    history: list  # max |dE/dshift| per iteration


def _newton_step(hess, g):
    """Minimum-norm Newton step.  When all band edges follow the levels, a
    uniform shift only moves the Fermi level and the Hessian is singular along
    it; the pseudo-inverse leaves that direction untouched."""
    return -jnp.linalg.pinv(hess, rtol=1e-10, hermitian=True) @ g


def make_scf(energy_fn):
    """Build a Newton solver for ``energy_fn(shifts, positions, cell, params)``."""
    shape = energy_fn.shift_shape
    size = int(np.prod(shape))

    def flat_energy(x, positions, cell, params):
        return energy_fn(x.reshape(shape), positions, cell, params)

    val_grad = jax.jit(jax.value_and_grad(flat_energy, has_aux=True))
    hessian = jax.jit(jax.jacfwd(jax.grad(lambda *a: flat_energy(*a)[0])))

    def solve(positions, cell, params, shifts0=None, tol=1e-10, max_iter=30,
              verbose=False):
        x = jnp.zeros(size) if shifts0 is None else jnp.ravel(shifts0)
        (e, aux), g = val_grad(x, positions, cell, params)
        history = [float(jnp.max(jnp.abs(g)))]
        for it in range(max_iter):
            if verbose:
                print(f"scf {it:3d}  E = {float(e):.12f}  max|dE/dshift| = {history[-1]:.3e}")
            if history[-1] < tol:
                break
            hess = hessian(x, positions, cell, params)
            norm = float(jnp.linalg.norm(g))
            # Plain Newton first.  Far from the solution the BOP Hessian can be
            # indefinite or nearly singular, so if the residual does not drop we
            # fall back to Levenberg-Marquardt steps on |g|^2 with growing damping.
            normal = hess.T @ hess
            damping = 0.0
            for _ in range(12):
                if damping == 0.0:
                    step = _newton_step(hess, g)
                else:
                    step = -jnp.linalg.solve(normal + damping * jnp.eye(size), hess.T @ g)
                x_new = x + step
                (e_new, aux_new), g_new = val_grad(x_new, positions, cell, params)
                norm_new = float(jnp.linalg.norm(g_new))
                if np.isfinite(norm_new) and norm_new < norm:
                    break
                damping = 1e-4 * float(jnp.max(jnp.diag(normal))) if damping == 0.0 else 10.0 * damping
            x, e, aux, g = x_new, e_new, aux_new, g_new
            history.append(float(jnp.max(jnp.abs(g))))
        return SCFResult(x.reshape(shape), float(e), aux, history[-1] < tol, history)

    return solve


def make_forces(energy_fn):
    """Forces at fixed shifts.  At self-consistency these are the full forces,
    because the energy is stationary with respect to the shifts."""
    grad = jax.jit(jax.grad(lambda s, p, c, par: energy_fn(s, p, c, par)[0], argnums=1))
    return lambda shifts, positions, cell, params: -grad(shifts, positions, cell, params)


def make_potential(energy_fn, tol=1e-10, max_iter=20, self_consistent=True):
    """Return a jit-compatible ``potential(positions, cell, params, shifts0=None)``
    giving ``(E, aux)`` with the onsite levels made self-consistent inside.

    The converged shifts enter through ``stop_gradient``: because the energy is
    stationary in them, ``jax.grad`` of the result with respect to positions,
    cell or parameters is the exact total derivative.  ``aux["shifts"]`` holds
    the converged shifts and ``aux["scf_residual"]`` the final max |dE/dshift|.
    """
    shape = energy_fn.shift_shape
    size = int(np.prod(shape))

    def flat_energy(x, positions, cell, params):
        return energy_fn(x.reshape(shape), positions, cell, params)[0]

    grad = jax.grad(flat_energy)
    hessian = jax.jacfwd(grad)

    def converge(x0, positions, cell, params):
        def cond(state):
            _, g, it = state
            return (jnp.max(jnp.abs(g)) > tol) & (it < max_iter)

        def body(state):
            x, g, it = state
            x = x + _newton_step(hessian(x, positions, cell, params), g)
            return x, grad(x, positions, cell, params), it + 1

        x, g, _ = jax.lax.while_loop(cond, body, (x0, grad(x0, positions, cell, params), 0))
        return x, jnp.max(jnp.abs(g))

    def potential(positions, cell, params, shifts0=None):
        x = jnp.zeros(size, dtype=positions.dtype) if shifts0 is None else jnp.ravel(shifts0)
        residual = jnp.max(jnp.abs(grad(x, positions, cell, params)))
        if self_consistent:
            frozen = jax.lax.stop_gradient((x, positions, cell, params))
            x, residual = converge(*frozen)
        x = jax.lax.stop_gradient(x)
        e, aux = energy_fn(x.reshape(shape), positions, cell, params)
        aux = dict(aux, shifts=x.reshape(shape), scf_residual=jax.lax.stop_gradient(residual))
        return e, aux

    return potential


def make_stress(potential):
    """Stress tensor (3, 3) = (1/V) dE/d(strain) for a periodic cell.

    ``potential`` is the result of ``make_potential``.  Positive values are
    tensile: the energy rises when the cell is stretched.
    """

    def strained_energy(strain, positions, cell, params, shifts0):
        deform = jnp.eye(3, dtype=positions.dtype) + 0.5 * (strain + strain.T)
        return potential(positions @ deform, cell @ deform, params, shifts0)[0]

    grad = jax.grad(strained_energy)

    def stress(positions, cell, params, shifts0=None):
        zero = jnp.zeros((3, 3), dtype=positions.dtype)
        volume = jnp.abs(jnp.linalg.det(cell))
        return grad(zero, positions, cell, params, shifts0) / volume

    return stress
