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


DENSE_LIMIT = 16  # above this many unknowns "auto" switches to Krylov


def _newton_step(hess, g):
    """Minimum-norm Newton step.  When all band edges follow the levels, a
    uniform shift only moves the Fermi level and the Hessian is singular along
    it; the pseudo-inverse leaves that direction untouched."""
    return -jnp.linalg.pinv(hess, rtol=1e-10, hermitian=True) @ g


def _krylov_step(grad, x, g, atol=0.0, rtol=1e-6, restart=4, maxiter=6):
    """Newton step from GMRES with Hessian-vector products.

    Each product costs about two gradient evaluations and no Hessian is ever
    stored, so the cost per step does not grow with the number of unknowns
    beyond that of the gradient itself.  The system is symmetric but can be
    indefinite and singular (see ``_newton_step``); GMRES started from zero
    stays in the range of the Hessian and so returns the same minimum-norm step.

    The Hessian is well conditioned in practice (eigenvalues within a factor of
    about three of each other for the d-band test cells), so about ten products
    reduce the linear residual by 1e-6.  The solve stops once that residual is
    below ``atol``: from a good starting guess (molecular dynamics) a handful
    of products is then enough.  JAX's GMRES only checks between restart
    cycles, hence the short cycle.
    """
    def hvp(v):
        return jax.jvp(grad, (x,), (v,))[1]

    # keep the Krylov space smaller than the rank of the Hessian, which is at
    # most size - 1 when a uniform shift is a null mode
    restart = max(1, min(restart, g.shape[0] - 1))
    step, _ = jax.scipy.sparse.linalg.gmres(hvp, -g, tol=rtol, atol=atol, restart=restart,
                                            maxiter=maxiter)
    return jnp.where(jnp.all(jnp.isfinite(step)), step, jnp.zeros_like(step))


def _use_krylov(method, size):
    if method not in ("auto", "dense", "krylov"):
        raise ValueError(f"unknown solver {method!r}")
    return method == "krylov" or (method == "auto" and size > DENSE_LIMIT)


def make_scf(energy_fn, method="auto"):
    """Build a Newton solver for ``energy_fn(shifts, positions, cell, params)``.

    method : "dense" builds the full Hessian, "krylov" uses Hessian-vector
             products, "auto" picks by the number of unknowns.
    """
    shape = energy_fn.shift_shape
    size = int(np.prod(shape))
    krylov = _use_krylov(method, size)

    def flat_energy(x, positions, cell, params):
        return energy_fn(x.reshape(shape), positions, cell, params)

    val_grad = jax.jit(jax.value_and_grad(flat_energy, has_aux=True))
    plain_grad = jax.grad(lambda *a: flat_energy(*a)[0])
    hessian = jax.jit(jax.jacfwd(plain_grad))
    krylov_step = jax.jit(lambda x, g, positions, cell, params, atol: _krylov_step(
        lambda y: plain_grad(y, positions, cell, params), x, g, atol))

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
            norm = float(jnp.linalg.norm(g))
            if krylov:
                # Newton direction, shortened if the residual does not drop
                step = krylov_step(x, g, positions, cell, params, 0.3 * tol)
                for _ in range(8):
                    x_new = x + step
                    (e_new, aux_new), g_new = val_grad(x_new, positions, cell, params)
                    norm_new = float(jnp.linalg.norm(g_new))
                    if np.isfinite(norm_new) and norm_new < norm:
                        break
                    step = 0.5 * step
            else:
                # Plain Newton first.  Far from the solution the BOP Hessian can
                # be indefinite or nearly singular, so if the residual does not
                # drop we fall back to Levenberg-Marquardt steps on |g|^2 with
                # growing damping.
                hess = hessian(x, positions, cell, params)
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
                    damping = (1e-4 * float(jnp.max(jnp.diag(normal))) if damping == 0.0
                               else 10.0 * damping)
            x, e, aux, g = x_new, e_new, aux_new, g_new
            history.append(float(jnp.max(jnp.abs(g))))
        return SCFResult(x.reshape(shape), float(e), aux, history[-1] < tol, history)

    return solve


def make_forces(energy_fn):
    """Forces at fixed shifts.  At self-consistency these are the full forces,
    because the energy is stationary with respect to the shifts."""
    grad = jax.jit(jax.grad(lambda s, p, c, par: energy_fn(s, p, c, par)[0], argnums=1))
    return lambda shifts, positions, cell, params: -grad(shifts, positions, cell, params)


CONDITIONS = ("stationary", "potential")


def _anderson(residual, y0, tol, max_iter, mixing, history):
    """Anderson (Pulay) mixing for ``residual(y) = 0`` with ``residual(y) =
    target(y) - y``, in a ``lax.while_loop``.  Returns ``(y, max|r|, iterations)``.

    The least-squares problem for the mixing coefficients is regularised
    relative to its diagonal.  The best iterate is kept: when the residual
    grows to ten times the best one, or is not finite, the iteration restarts
    from the best iterate with an empty history and half the mixing.
    """
    k = y0.shape[0]
    dtype = y0.dtype

    def mix(y, f, dy, df, count, beta):
        valid = (jnp.arange(history) < count).astype(dtype)
        a = (df @ df.T) * valid[:, None] * valid[None, :]
        a = a + jnp.diag(1e-10 * jnp.max(jnp.diag(a)) + (1.0 - valid) + 1e-300)
        gamma = jnp.linalg.solve(a, (df @ f) * valid) * valid
        return y + beta * f - (dy + beta * df).T @ gamma

    def norm(f):
        return jnp.where(jnp.all(jnp.isfinite(f)), jnp.max(jnp.abs(f)), jnp.inf)

    def cond(state):
        f = state[1]
        return (norm(f) > tol) & (state[-1] < max_iter)

    def body(state):
        y, f, dy, df, count, beta, y_best, f_best, it = state
        y_new = mix(y, f, dy, df, count, beta)
        y_new = jnp.where(jnp.all(jnp.isfinite(y_new)), y_new, y + beta * f)
        f_new = residual(y_new)
        better = norm(f_new) < norm(f_best)
        y_best = jnp.where(better, y_new, y_best)
        f_best = jnp.where(better, f_new, f_best)
        restart = norm(f_new) > 10.0 * norm(f_best)
        dy = jnp.roll(dy, 1, axis=0).at[0].set(y_new - y)
        df = jnp.roll(df, 1, axis=0).at[0].set(f_new - f)
        count = jnp.where(restart, 0, jnp.minimum(count + 1, history))
        beta = jnp.where(restart, 0.5 * beta, beta)
        y_new = jnp.where(restart, y_best, y_new)
        f_new = jnp.where(restart, f_best, f_new)
        return y_new, f_new, dy, df, count, beta, y_best, f_best, it + 1

    zeros = jnp.zeros((history, k), dtype=dtype)
    f0 = residual(y0)
    state = (y0, f0, zeros, zeros, 0, jnp.asarray(mixing, dtype), y0, f0, 0)
    state = jax.lax.while_loop(cond, body, state)
    y, f, y_best, f_best, it = state[0], state[1], state[6], state[7], state[-1]
    use_best = norm(f_best) < norm(f)
    return jnp.where(use_best, y_best, y), jnp.minimum(norm(f), norm(f_best)), it


def make_potential(energy_fn, tol=1e-10, max_iter=None, self_consistent=True,
                   method="auto", free=None, condition="stationary", mixing=0.1,
                   history=8, adjoint_tol=1e-8):
    """Return a jit-compatible ``potential(positions, cell, params, shifts0=None)``
    giving ``(E, aux)`` with the onsite levels made self-consistent inside.

    The converged shifts enter through ``stop_gradient``: because the energy is
    stationary in them, ``jax.grad`` of the result with respect to positions,
    cell or parameters is the exact total derivative.  ``aux["shifts"]`` holds
    the converged shifts, ``aux["scf_residual"]`` the final max |dE/dshift| and
    ``aux["scf_iterations"]`` the number of Newton steps taken.  ``shifts0`` is
    the starting guess; passing the shifts of a nearby configuration (the
    previous molecular dynamics step) saves most of the iterations.
    ``method`` selects the linear solver as in ``make_scf``.

    ``free`` (bool, (N,)) restricts the self-consistency to the onsite levels
    of those atoms; the others keep the levels passed in ``shifts0``, like
    atoms held at fixed positions.  The energy is then stationary in the free
    levels only, which is all the forces on atoms need as long as the frozen
    levels do not depend on the positions.  Use it for the outer shell of a
    cluster or cylinder, whose surface moments can come close to saturation
    and leave the self-consistency with nearly flat directions.
    ``aux["scf_residual"]`` is then the residual of the free levels.

    ``condition`` selects what "self-consistent" means:

    * ``"stationary"`` (default): dE/d(shifts) = 0, the condition BOPfox
      solves, by Newton's method (``max_iter`` default 20).
    * ``"potential"``: the shifts equal the potentials their occupations imply,
      ``shifts = dE_dc/dN`` (``energy_fn.potential_target``; onsite and
      intersite Coulomb, Stoner), solved by Anderson mixing with parameter
      ``mixing`` and ``history`` previous steps (``max_iter`` default 200).
      This is the usual charge and spin self-consistency of tight binding.
      For exact tight binding the two conditions coincide (in a perfect
      crystal they also coincide for the BOP); in inhomogeneous systems they
      differ by the error of the moment expansion, the amount by which the
      BOP bond energy is not variational.  Where moments approach saturation
      (surfaces), dN/d(shifts) nearly vanishes and the stationary point can
      lie far away or be out of reach of Newton's method, while the
      potential condition stays well conditioned.  The energy is then not
      stationary in the shifts, so gradients carry the response of the
      shifts: they are obtained by implicit differentiation of the
      condition, one adjoint linear solve (GMRES) per gradient, to the
      relative tolerance ``adjoint_tol``; each GMRES step costs about one
      gradient evaluation, and the solve usually dominates the cost.
      ``aux["scf_residual"]`` is max |shifts - dE_dc/dN|.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown self-consistency condition {condition!r}")
    if max_iter is None:
        max_iter = 20 if condition == "stationary" else 200
    shape = energy_fn.shift_shape
    size = int(np.prod(shape))
    if free is None:
        index = np.arange(size)
    else:
        mask = np.broadcast_to(np.asarray(free, bool).reshape((-1,) + (1,) * (len(shape) - 1)),
                               shape)
        index = np.flatnonzero(mask.ravel())
    krylov = _use_krylov(method, len(index))

    def flat_energy(x, positions, cell, params):
        return energy_fn(x.reshape(shape), positions, cell, params)[0]

    full_grad = jax.grad(flat_energy)

    def converge(x0, positions, cell, params):
        def grad(y):  # gradient with respect to the free levels
            return full_grad(x0.at[index].set(y), positions, cell, params)[index]

        hessian = jax.jacfwd(grad)

        def cond(state):
            _, g, it = state
            return (jnp.max(jnp.abs(g)) > tol) & (it < max_iter)

        def body(state):
            y, g, it = state
            if krylov:
                y = y + _krylov_step(grad, y, g, 0.3 * tol)
            else:
                y = y + _newton_step(hessian(y), g)
            return y, grad(y), it + 1

        y0 = x0[index]
        y, g, it = jax.lax.while_loop(cond, body, (y0, grad(y0), 0))
        return x0.at[index].set(y), jnp.max(jnp.abs(g)), it

    def potential_residual(y, x0, positions, cell, params):
        x = x0.at[index].set(y)
        _, aux = energy_fn(x.reshape(shape), positions, cell, params)
        target = energy_fn.potential_target(aux["occupations"], positions, cell, params)
        return jnp.ravel(target)[index] - y

    @jax.custom_vjp
    def solve_potential(y0, x0, positions, cell, params):
        return _anderson(lambda y: potential_residual(y, x0, positions, cell, params),
                         y0, tol, max_iter, mixing, history)

    def solve_fwd(y0, x0, positions, cell, params):
        out = solve_potential(y0, x0, positions, cell, params)
        return out, (out[0], x0, positions, cell, params)

    def solve_bwd(res, cotangent):
        # r(y(theta), theta) = 0  =>  theta_bar = -(dr/dtheta)^T u,
        # (dr/dy)^T u = y_bar
        y, x0, positions, cell, params = res
        y_bar = cotangent[0]
        _, vjp_y = jax.vjp(lambda v: potential_residual(v, x0, positions, cell, params), y)
        u, _ = jax.scipy.sparse.linalg.gmres(lambda v: vjp_y(v)[0], y_bar, tol=adjoint_tol,
                                             atol=0.0, restart=5, maxiter=60)
        _, vjp_t = jax.vjp(lambda p, c, par: potential_residual(y, x0, p, c, par),
                           positions, cell, params)
        p_bar, c_bar, par_bar = vjp_t(-u)
        return (jnp.zeros_like(y), jnp.zeros_like(x0), p_bar, c_bar, par_bar)

    solve_potential.defvjp(solve_fwd, solve_bwd)

    def potential(positions, cell, params, shifts0=None):
        x = jnp.zeros(size, dtype=positions.dtype) if shifts0 is None else jnp.ravel(shifts0)
        iterations = jnp.zeros((), dtype=jnp.int32)
        if self_consistent and condition == "potential":
            x0 = jax.lax.stop_gradient(x)
            y, residual, iterations = solve_potential(x0[index], x0, positions, cell, params)
            residual, iterations = jax.lax.stop_gradient((residual, iterations))
            x = x0.at[index].set(y)
            e, aux = energy_fn(x.reshape(shape), positions, cell, params)
            aux = dict(aux, shifts=jax.lax.stop_gradient(x.reshape(shape)),
                       scf_iterations=iterations, scf_residual=residual)
            return e, aux
        if self_consistent:
            frozen = jax.lax.stop_gradient((x, positions, cell, params))
            x, residual, iterations = converge(*frozen)
        else:
            residual = jnp.max(jnp.abs(full_grad(x, positions, cell, params)[index]))
        x = jax.lax.stop_gradient(x)
        e, aux = energy_fn(x.reshape(shape), positions, cell, params)
        aux = dict(aux, shifts=x.reshape(shape), scf_iterations=iterations,
                   scf_residual=jax.lax.stop_gradient(residual))
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
