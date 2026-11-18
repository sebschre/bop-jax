"""Climbing-image nudged elastic band (NEB) for minimum energy paths.

Henkelman, Uberuaga and Jonsson, J. Chem. Phys. 113, 9901 (2000) (climbing
image) and Henkelman and Jonsson, J. Chem. Phys. 113, 9978 (2000) (tangent).
The images are relaxed together with FIRE (Bitzek et al., PRL 97, 170201
(2006)).

The energy and forces of an image come from a user function
``force_fn(index, positions) -> (energy, forces)``, so that each image can
keep its own state, such as warm-started onsite levels or its own neighbour
list.  The two end points are not moved and should be relaxed minima.

Units are those of the force function (eV and A for bopjax); the FIRE time
step is in units with mass 1, so a displacement of ``dt^2 F`` per step at
rest.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np


class NEBResult(NamedTuple):
    images: np.ndarray  # (P, N, 3)
    energies: np.ndarray  # (P,)
    path: np.ndarray  # (P,) arc length along the band (A)
    slopes: np.ndarray  # (P,) dE/d(path) from the forces along the tangent (eV / A)
    climbing: int  # index of the climbing image, or -1
    converged: bool
    history: list  # max |NEB force| per step


class _Fire:
    """FIRE steps for unit masses, with the largest displacement of an atom
    bounded by ``max_step``."""

    def __init__(self, dt, dt_max, max_step):
        self.dt0, self.dt_max, self.max_step = dt, dt_max, max_step
        self.v = None
        self.reset()

    def reset(self):
        self.v, self.alpha, self.n_pos, self.dt = None, 0.1, 0, self.dt0

    def step(self, force):
        if self.v is None:  # first step from rest
            v = np.zeros_like(force)
        elif np.sum(force * self.v) > 0.0:
            v = self.v
            v = (1 - self.alpha) * v + self.alpha * force * np.linalg.norm(v) / max(
                np.linalg.norm(force), 1e-300)
            self.n_pos += 1
            if self.n_pos > 5:
                self.dt = min(self.dt * 1.1, self.dt_max)
                self.alpha *= 0.99
        else:
            v = np.zeros_like(force)
            self.alpha, self.n_pos, self.dt = 0.1, 0, self.dt * 0.5
        v = v + self.dt * force
        dx = self.dt * v
        largest = float(np.max(np.linalg.norm(dx.reshape(-1, 3), axis=-1)))
        if largest > self.max_step:
            dx *= self.max_step / largest
        self.v = v
        return dx


def relax(force_fn: Callable, positions, mobile=None, fmax=0.01, steps=500, dt=0.1,
          dt_max=0.5, max_step=0.1, verbose=False):
    """Relax one configuration with FIRE.

    ``force_fn(positions) -> (energy, forces)``.  Returns ``(positions,
    energy, converged)``.
    """
    x = np.array(positions, dtype=float)
    move = np.ones(len(x), bool) if mobile is None else np.asarray(mobile, bool)
    fire = _Fire(dt, dt_max, max_step)
    for step in range(steps):
        e, f = force_fn(x)
        f = np.where(move[:, None], np.asarray(f), 0.0)
        fnorm = float(np.max(np.linalg.norm(f, axis=-1)))
        if verbose and (step % 10 == 0 or fnorm < fmax):
            print(f"relax {step:4d}  E = {float(e):.6f}  max|F| = {fnorm:.4f}", flush=True)
        if fnorm < fmax:
            return x, float(e), True
        x = x + fire.step(f)
    e, _ = force_fn(x)
    return x, float(e), False


def _tangents(x, e):
    """Improved tangents of the interior images (Henkelman and Jonsson)."""
    tau = []
    for i in range(1, len(x) - 1):
        fwd, bwd = x[i + 1] - x[i], x[i] - x[i - 1]
        if e[i + 1] > e[i] > e[i - 1]:
            t = fwd
        elif e[i + 1] < e[i] < e[i - 1]:
            t = bwd
        else:
            hi = max(abs(e[i + 1] - e[i]), abs(e[i - 1] - e[i]))
            lo = min(abs(e[i + 1] - e[i]), abs(e[i - 1] - e[i]))
            t = fwd * hi + bwd * lo if e[i + 1] > e[i - 1] else fwd * lo + bwd * hi
        tau.append(t / np.linalg.norm(t))
    return tau


def neb(force_fn: Callable, images, mobile=None, spring=0.1, climb=True, fmax=0.01,
        climb_fmax=None, steps=500, dt=0.1, dt_max=0.5, max_step=0.1, verbose=False):
    """Relax a band of images into a minimum energy path.

    force_fn  : ``force_fn(index, positions) -> (energy, forces)`` for image
                ``index``, positions and forces (N, 3)
    images    : (P, N, 3) initial band including both end points
    mobile    : (N,) bool, atoms that move (default: all)
    spring    : spring constant between neighbouring images (eV / A^2)
    climb     : turn the highest image into a climbing image once the largest
                NEB force is below ``climb_fmax`` (default 3 fmax)
    fmax      : convergence threshold on the largest NEB force (eV / A)

    Returns an ``NEBResult``.  ``slopes`` are dE/ds along the band from the
    forces projected on the tangent (zero at the end points), for
    interpolating the energy profile.
    """
    x = np.array(images, dtype=float)
    p, n = x.shape[:2]
    if p < 3:
        raise ValueError("a band needs at least one image between the end points")
    move = np.ones(n, bool) if mobile is None else np.asarray(mobile, bool)
    climb_fmax = 3.0 * fmax if climb_fmax is None else climb_fmax

    e = np.zeros(p)
    f = np.zeros_like(x)
    for i in (0, p - 1):
        e[i], fi = force_fn(i, x[i])
        f[i] = np.asarray(fi)

    def evaluate(x):
        for i in range(1, p - 1):
            e[i], fi = force_fn(i, x[i])
            f[i] = np.where(move[:, None], np.asarray(fi), 0.0)

    def neb_forces(x, climbing):
        tau = _tangents(x, e)
        out = np.zeros_like(x)
        for k, i in enumerate(range(1, p - 1)):
            t = tau[k]
            f_par = np.sum(f[i] * t) * t
            if i == climbing:
                out[i] = f[i] - 2.0 * f_par
            else:
                d = np.linalg.norm(x[i + 1] - x[i]) - np.linalg.norm(x[i] - x[i - 1])
                out[i] = f[i] - f_par + spring * d * t
            out[i][~move] = 0.0
        return out, tau

    evaluate(x)
    climbing = -1
    fire = _Fire(dt, dt_max, max_step)
    history = []
    converged = False
    for step in range(steps):
        force, tau = neb_forces(x, climbing)
        fnorm = float(np.max(np.linalg.norm(force[1:-1], axis=-1)))
        history.append(fnorm)
        if verbose:
            print(f"neb {step:4d}  max|F| = {fnorm:.4f}  E - E0 = "
                  + " ".join(f"{v_:+.4f}" for v_ in e - e[0]) + (
                      f"  climbing {climbing}" if climbing >= 0 else ""), flush=True)
        if climb and climbing < 0 and fnorm < climb_fmax:
            climbing = 1 + int(np.argmax(e[1:-1]))
            fire.reset()
            continue
        if fnorm < fmax and (climbing >= 0 or not climb):
            converged = True
            break
        x = x + fire.step(force)
        evaluate(x)

    # arc length and slopes along the final band
    seg = np.linalg.norm((x[1:] - x[:-1]).reshape(p - 1, -1), axis=1)
    path = np.concatenate([[0.0], np.cumsum(seg)])
    tau = _tangents(x, e)
    slopes = np.zeros(p)
    for k, i in enumerate(range(1, p - 1)):
        slopes[i] = -np.sum(f[i] * tau[k])
    return NEBResult(x, e.copy(), path, slopes, climbing, converged, history)
