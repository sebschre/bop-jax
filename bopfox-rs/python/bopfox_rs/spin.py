"""Spin dynamics and coupled spin-lattice dynamics with bopfox-rs.

A port of ``bopjax.spin`` and ``bopjax.spinlattice``.  In a non-collinear
model every atom carries a local spin axis ``e_i`` and the self-consistent
energy ``E(R, e)`` gives torques ``-e_i x dE/de_i`` as it gives forces.  The
axes follow the Landau-Lifshitz-Gilbert equation

    S_i de_i/dt = -e_i x dE/de_i - alpha |S_i| e_i x de_i/dt,   S_i = hbar m_i / 2

with ``m_i`` the moment along the axis (Bohr magnetons): adiabatic spin
dynamics (Antropov et al., PRB 54, 1019 (1996)), the electrons following the
directions instantaneously.

* ``alpha = 0``: precession at constant energy;
* ``alpha > 0``: Gilbert damping, the energy falls until the torques vanish;
* ``kT > 0`` (with ``alpha > 0``): stochastic torques that obey the
  fluctuation-dissipation theorem, so the directions sample exp(-E / kT).

``SpinDynamics`` integrates the axes at fixed positions with Heun's
predictor-corrector scheme, each update applied as a rotation (second order,
Stratonovich with noise, unit axes to machine precision; two evaluations per
step).  ``SpinLatticeDynamics`` adds the atoms: half kick, half drift, a
midpoint step of the axes at the middle positions solved by ``correctors``
fixed-point passes, half drift, half kick (``correctors + 1`` evaluations).
Without damping ``E + E_kin`` is conserved.

Every evaluation is a self-consistent bopfox-rs solve, warm-started from the
shifts extrapolated from the three previous evaluations, with the Anderson
history of the previous solve.  External fields (``magnetic_field``, Tesla)
add Zeeman torques; with them a ferromagnet precesses at the Larmor
frequency g mu_B B / hbar.

Units: time in femtoseconds, masses in amu, energies in eV.  bopjax uses
``bopjax.md.TIME_UNIT_FS`` = 10.1805 fs as its time unit.
"""

from __future__ import annotations

import numpy as np

from . import MU_B, Potential

HBAR = 0.6582119569  # eV fs
TIME_UNIT_FS = 10.180505710  # bopjax's time unit (Angstrom sqrt(amu / eV)) in fs
AMU_A2_FS2 = 103.642697  # amu Angstrom^2 / fs^2 in eV
KB_EV = 8.617333262e-5  # eV / K


def rotate(vectors, rotation):
    """Rotate each vector (N, 3) about ``rotation`` (N, 3) by the angle
    |rotation| (Rodrigues); lengths are preserved exactly."""
    angle2 = np.sum(rotation**2, axis=-1, keepdims=True)
    small = angle2 < 1e-16
    angle = np.sqrt(np.where(small, 1.0, angle2))
    sinc = np.where(small, 1.0 - angle2 / 6.0, np.sin(angle) / angle)
    cosc = np.where(small, 0.5 - angle2 / 24.0, (1.0 - np.cos(angle)) / angle**2)
    wxv = np.cross(rotation, vectors)
    return vectors + sinc * wxv + cosc * np.cross(rotation, wxv)


def _bounded(moment, min_moment):
    return np.where(moment < 0.0, -1.0, 1.0) * np.maximum(np.abs(moment), min_moment)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angular_velocity(axes, gradient, moment, alpha=0.0, min_moment=0.1):
    """``w`` (N, 3, rad/fs) with ``de/dt = w x e`` for the LLG equation."""
    spin = 0.5 * HBAR * _bounded(moment, min_moment)[:, None]
    return (gradient / spin - alpha * np.cross(axes, gradient) / np.abs(spin)) / (1.0 + alpha**2)


class _Evaluator:
    """Self-consistent energy, forces and axis gradient, warm-started."""

    def __init__(self, potential, species, cell, tol, magnetic_field, condition, scf_options,
                 on_failure="warn"):
        if on_failure not in ("warn", "raise", "ignore"):
            raise ValueError("on_failure must be 'warn', 'raise' or 'ignore'")
        self.on_failure = on_failure
        self.failures = 0  # evaluations that did not reach the tolerance
        self.pot = potential if isinstance(potential, Potential) else Potential(potential)
        if self.pot.model.get("spin") != "noncollinear":
            raise ValueError('spin dynamics needs a model with spin = "noncollinear"')
        self.species = species
        self.cell = cell
        self.kwargs = dict(tol=tol, condition=condition, magnetic_field=magnetic_field,
                           **(scf_options or {}))
        self.evaluations = 0  # gradient evaluations of the self-consistency, in total

    def __call__(self, positions, axes, guess):
        out = self.pot.scf(positions, self.species, self.cell, guess, axes=axes,
                           reuse_history=guess is not None, **self.kwargs)
        if not out["converged"]:
            # The stationary condition can lack an exact root (as with nearly
            # saturated or strongly canted moments; bopjax's Newton solver
            # stalls there too).  The best iterate is used; condition =
            # "potential" stays well conditioned in such states.
            self.failures += 1
            message = f"self-consistency not converged (residual {out['residual']:.2e})"
            if self.on_failure == "raise":
                raise RuntimeError(message)
            if self.on_failure == "warn":
                import warnings
                warnings.warn(message + "; continuing with the best iterate", RuntimeWarning,
                              stacklevel=3)
        self.evaluations += out["evaluations"]
        return out


class SpinDynamics:
    """LLG dynamics of the spin axes at fixed atomic positions.

    potential      ``bopfox_rs.Potential`` (or model) with spin = "noncollinear"
    shifts0        starting onsite shifts (N, 2, nshell); they select the
                   magnetic solution (an exchange splitting, or converged
                   shifts), since from zero the non-magnetic state is found
    dt             time step (fs); precession is fast, use a fraction of a fs
    alpha, kT      Gilbert damping and temperature (eV) of the spin bath
    free           (N,) mask of the axes that move
    magnetic_field external field in Tesla, (3,) or (N, 3)
    rng            numpy Generator for the thermal torques
    condition      "stationary" (BOPfox) or "potential" self-consistency
    on_failure     what an evaluation that misses ``tol`` does: "warn"
                   (default) or "ignore" continue with the best iterate,
                   "raise" stops; ``self.evaluate.failures`` counts them
    """

    def __init__(self, potential, positions, species, cell, axes, shifts0, dt, alpha=0.0,
                 kT=0.0, tol=1e-9, min_moment=0.1, free=None, magnetic_field=None,
                 condition="stationary", extrapolate=True, rng=None, scf_options=None,
                 on_failure="warn"):
        if kT > 0.0 and alpha <= 0.0:
            raise ValueError("a temperature needs damping: set alpha > 0")
        self.evaluate = _Evaluator(potential, species, cell, tol, magnetic_field, condition,
                                   scf_options, on_failure)
        self.positions = np.asarray(positions, dtype=float)
        self.axes = _unit(axes)
        n = len(self.axes)
        self.free = np.ones((n, 1)) if free is None else np.asarray(free, float).reshape(n, 1)
        self.dt, self.alpha, self.kT, self.min_moment = dt, alpha, kT, min_moment
        self.extrapolate = extrapolate
        self.rng = np.random.default_rng() if rng is None else rng
        self.time = 0.0
        self._update(self.evaluate(self.positions, self.axes, np.asarray(shifts0, dtype=float)))
        self._shifts = [self.shifts] * 3

    def _update(self, out):
        self.result = out
        self.energy = out["energy"]
        self.gradient = out["axes_gradient"]
        self.moment = out["moment"]
        self.shifts = out["shifts"]

    def _velocity(self, axes, gradient, moment, noise):
        width = np.sqrt(self.alpha * HBAR * np.abs(_bounded(moment, self.min_moment)) * self.kT
                        / self.dt)
        return self.free * angular_velocity(axes, gradient + width[:, None] * noise, moment,
                                            self.alpha, self.min_moment)

    def _guess(self):
        s0, s1, s2 = self._shifts
        return 3.0 * s0 - 3.0 * s1 + s2 if self.extrapolate else s0

    def step(self):
        noise = self.rng.normal(size=self.axes.shape)
        w0 = self._velocity(self.axes, self.gradient, self.moment, noise)
        trial = rotate(self.axes, self.dt * w0)
        out = self.evaluate(self.positions, trial, self._guess())
        w1 = self._velocity(trial, out["axes_gradient"], out["moment"], noise)
        self.axes = rotate(self.axes, 0.5 * self.dt * (w0 + w1))
        self._update(self.evaluate(self.positions, self.axes, out["shifts"]))
        self._shifts = [self.shifts] + self._shifts[:2]
        self.time += self.dt

    def run(self, steps, callback=None, interval=1):
        for k in range(steps):
            self.step()
            if callback is not None and (k + 1) % interval == 0:
                callback(self)

    @property
    def torque(self):
        """``-e x dE/de`` (eV), (N, 3)."""
        return -np.cross(self.axes, self.gradient)

    @property
    def magnetisation(self):
        """``sum m_i e_i`` (Bohr magnetons), (3,)."""
        return np.sum(self.moment[:, None] * self.axes, axis=0)

    @property
    def damping_power(self):
        """``-dE/dt`` of the Gilbert damping (eV / fs)."""
        spin = 0.5 * HBAR * np.abs(_bounded(self.moment, self.min_moment))
        return self.alpha / (1.0 + self.alpha**2) * np.sum(np.sum(self.torque**2, axis=1) / spin)

    @property
    def effective_field(self):
        """``-dE/d(m e)`` on each moment (Tesla), (N, 3)."""
        return -self.gradient / (MU_B * _bounded(self.moment, self.min_moment)[:, None])


class SpinLatticeDynamics(SpinDynamics):
    """Coupled dynamics of atoms and spin axes (``bopjax.spinlattice.nve``).

    masses       amu, scalar or (N,)
    temperature  starting temperature of the atoms (K); momenta are drawn
                 from the Maxwell distribution without drift
    mobile       (N,) mask of the atoms that move
    correctors   fixed-point passes of the midpoint step of the axes; a step
                 costs ``correctors + 1`` evaluations
    """

    def __init__(self, potential, positions, species, cell, axes, shifts0, dt, masses,
                 temperature=0.0, alpha=0.0, kT=0.0, tol=1e-9, min_moment=0.1, mobile=None,
                 free=None, magnetic_field=None, condition="stationary", extrapolate=True,
                 correctors=3, rng=None, scf_options=None, on_failure="warn"):
        super().__init__(potential, positions, species, cell, axes, shifts0, dt, alpha, kT,
                         tol, min_moment, free, magnetic_field, condition, extrapolate, rng,
                         scf_options, on_failure)
        n = len(self.positions)
        self.masses = np.broadcast_to(np.asarray(masses, dtype=float), (n,)).reshape(n, 1)
        self.mobile = (np.ones((n, 1)) if mobile is None
                       else np.asarray(mobile, float).reshape(n, 1))
        self.correctors = correctors
        # momenta in amu A / fs
        sigma = np.sqrt(self.masses * KB_EV * temperature / AMU_A2_FS2)
        p = sigma * self.rng.normal(size=(n, 3))
        p -= np.sum(self.mobile * p, axis=0) / max(np.sum(self.mobile), 1.0)
        self.momenta = self.mobile * p
        self.forces = self.mobile * self.result["forces"]

    def _update(self, out):
        super()._update(out)
        if hasattr(self, "mobile"):
            self.forces = self.mobile * out["forces"]

    def step(self):
        noise = self.rng.normal(size=self.axes.shape)
        guess = self._guess()
        kick = 0.5 * self.dt / AMU_A2_FS2  # eV/A * fs -> amu A / fs
        momenta = self.momenta + kick * self.forces
        velocity = momenta / self.masses
        middle = self.positions + 0.5 * self.dt * velocity
        positions = self.positions + self.dt * velocity
        w = self._velocity(self.axes, self.gradient, self.moment, noise)
        for _ in range(self.correctors):
            half = rotate(self.axes, 0.5 * self.dt * w)
            out = self.evaluate(middle, half, guess)
            guess = out["shifts"]
            w = self._velocity(half, out["axes_gradient"], out["moment"], noise)
        self.axes = rotate(self.axes, self.dt * w)
        self.positions = positions
        self._update(self.evaluate(self.positions, self.axes, guess))
        self.momenta = momenta + kick * self.forces
        self._shifts = [self.shifts] + self._shifts[:2]
        self.time += self.dt

    @property
    def kinetic_energy(self):
        return 0.5 * AMU_A2_FS2 * np.sum(self.momenta**2 / self.masses)

    @property
    def temperature(self):
        """Instantaneous temperature of the atoms (K)."""
        return 2.0 * self.kinetic_energy / (3.0 * np.sum(self.mobile) * KB_EV)
