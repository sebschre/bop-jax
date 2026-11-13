"""Self-consistent onsite levels by Newton's method.

The residual is g = dE/d(shifts); the Jacobian is the Hessian d2E/d(shifts)^2.
Both come from automatic differentiation of the energy function.
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


def make_scf(energy_fn):
    """Build a Newton solver for ``energy_fn(shifts, positions, cell, params)``."""
    shape = energy_fn.shift_shape
    size = int(np.prod(shape))

    def flat_energy(x, positions, cell, params):
        return energy_fn(x.reshape(shape), positions, cell, params)

    val_grad = jax.jit(jax.value_and_grad(flat_energy, has_aux=True))
    hessian = jax.jit(jax.jacfwd(jax.grad(lambda *a: flat_energy(*a)[0])))
    # A uniform shift of all levels only moves the Fermi level: the Hessian has
    # a null vector there, which this projector removes.
    gauge = jnp.ones((size, size)) / size

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
            hess = hessian(x, positions, cell, params) + gauge
            norm = float(jnp.linalg.norm(g))
            # Plain Newton first.  Far from the solution the BOP Hessian can be
            # indefinite or nearly singular, so if the residual does not drop we
            # fall back to Levenberg-Marquardt steps on |g|^2 with growing damping.
            normal = hess.T @ hess
            damping = 0.0
            for _ in range(12):
                if damping == 0.0:
                    step = -jnp.linalg.solve(hess, g)
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
        x = x - jnp.mean(x)  # fix the gauge: shifts average to zero
        return SCFResult(x.reshape(shape), float(e), aux, history[-1] < tol, history)

    return solve


def make_forces(energy_fn):
    """Forces at fixed shifts.  At self-consistency these are the full forces,
    because the energy is stationary with respect to the shifts."""
    grad = jax.jit(jax.grad(lambda s, p, c, par: energy_fn(s, p, c, par)[0], argnums=1))
    return lambda shifts, positions, cell, params: -grad(shifts, positions, cell, params)
