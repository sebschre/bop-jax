"""Phases of iron and the properties the iron model is fitted to.

Shared by ``examples/fit_iron.py`` and ``examples/iron_properties.py``.
Lengths in Angstrom, energies in eV, elastic constants in GPa, moments in
Bohr magnetons (the d moment of the Stoner model, m = N_up - N_down).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar

import bopjax as bj
from bopjax.hamiltonian import v0_bond
from bopjax.models import IRON_CONFIG, iron

EV_A3_TO_GPA = 160.21766


# --- structures: (cell, positions) for a cubic or hexagonal lattice parameter a ---


def bcc(a):
    return a * np.eye(3), a * np.array([[0, 0, 0], [0.5, 0.5, 0.5]])


def fcc(a):
    # conventional cell: atoms 0, 1 in z = 0 planes, 2, 3 in z = 1/2 planes,
    # so moments (+, +, -, -) are the single-layer antiferromagnet AFM-I
    return a * np.eye(3), a * np.array([[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])


def hcp(a, c_over_a=1.58):
    cell = np.array([[a, 0, 0], [-0.5 * a, 0.5 * np.sqrt(3) * a, 0], [0, 0, c_over_a * a]])
    return cell, np.array([[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]]) @ cell


@dataclass(frozen=True)
class Phase:
    name: str
    lattice: callable  # a -> (cell, positions)
    a_guess: float  # lattice parameter the neighbour lists are built for
    moments: tuple | None  # starting moments, None for a non-magnetic calculation

    @property
    def n_atoms(self):
        return len(self.lattice(self.a_guess)[1])


PHASES = {
    "bcc FM": Phase("bcc FM", bcc, 2.86, (2.5, 2.5)),
    "bcc AFM": Phase("bcc AFM", bcc, 2.82, (2.0, -2.0)),
    "bcc NM": Phase("bcc NM", bcc, 2.76, None),
    "fcc AFM": Phase("fcc AFM", fcc, 3.52, (2.0, 2.0, -2.0, -2.0)),
    "fcc FM": Phase("fcc FM", fcc, 3.60, (2.6, 2.6, 2.6, 2.6)),
    "fcc NM": Phase("fcc NM", fcc, 3.46, None),
    "hcp NM": Phase("hcp NM", hcp, 2.46, None),
    "hcp AFM": Phase("hcp AFM", hcp, 2.48, (1.5, -1.5)),
    "hcp FM": Phase("hcp FM", hcp, 2.50, (2.5, 2.5)),
}


class Calculator:
    """Compiled self-consistent energy and stress of one phase.

    Built once per phase; the parameters are an argument, so a fit does not
    recompile.  Neighbour lists carry a skin of ``skin`` so that lattice
    parameters within about +-6 % of ``phase.a_guess`` need no rebuild.
    """

    def __init__(self, phase: Phase, skin=0.5, basis=("d",), **config):
        self.phase = phase
        self.basis = tuple(basis)
        cfg_kw = dict(IRON_CONFIG, basis=self.basis, **config)
        radial = v0_bond
        spin = "none" if phase.moments is None else "collinear"
        self.cfg = bj.BOPConfig(spin=spin, **cfg_kw)
        cell, pos = phase.lattice(phase.a_guess)
        topo = bj.build_topology(pos, [0] * len(pos), self.cfg.rcut, self.cfg.nmom, cell,
                                 rcut2=self.cfg.rcut2, skin=skin)
        energy = bj.make_energy_fn(topo, self.cfg, radial=radial)
        potential = bj.make_potential(energy, tol=1e-10, max_iter=25)
        self._potential = jax.jit(potential)
        self._scf = bj.make_scf(energy)  # damped Newton, the fallback
        self._stress = jax.jit(bj.make_stress(potential))
        self._forces = jax.jit(lambda pos, cell, params, start: -jax.grad(
            lambda q: potential(q, cell, params, start)[0])(pos))
        self.extra = ()  # further lattice arguments, e.g. (c/a,) for hcp
        self._last = None  # shifts of the last call: warm start for the next

    def _start(self, params):
        if self.phase.moments is None:
            return None
        m = np.asarray(self.phase.moments, float)[:, None] * (
            np.array(self.basis) == "d")[None, :]  # the moment sits on the d shell
        return bj.stoner_shifts(m, params["stoner"][0])

    def _geometry(self, a, strain=None):
        cell, pos = self.phase.lattice(a, *self.extra)
        if strain is not None:
            deform = np.eye(3) + strain
            cell, pos = cell @ deform, pos @ deform
        return jnp.asarray(pos), jnp.asarray(cell)

    def energy(self, params, a, strain=None):
        """(energy per atom, aux) at lattice parameter a.

        Magnetic phases start from the Stoner-split levels of the phase's
        starting moments; the non-magnetic state is stationary as well, so
        a warm start from a collapsed solution would never find the moments
        again.  The previous solution is tried too and the lower one kept.
        """
        pos, cell = self._geometry(a, strain)
        e, aux = self._potential(pos, cell, params, self._start(params))
        if self.phase.moments is not None:
            if self._last is not None:
                e2, aux2 = self._potential(pos, cell, params, self._last)
                bad = float(aux["scf_residual"]) > 1e-7
                good2 = float(aux2["scf_residual"]) < 1e-7
                if good2 and (bad or float(e2) < float(e) - 1e-9):
                    e, aux = e2, aux2
        if float(aux["scf_residual"]) > 1e-7:
            # plain Newton can cycle (seen with an s shell): damped steps
            res = self._scf(pos, cell, params, self._start(params), tol=1e-10, max_iter=60)
            if res.converged:
                e, aux = self._potential(pos, cell, params, res.shifts)
        if self.phase.moments is not None:
            self._last = aux["shifts"]
        self.residual = float(aux["scf_residual"])
        if self.residual > 1e-6:  # no self-consistent state: not a candidate
            return 1e3, aux
        return float(e) / self.phase.n_atoms, aux

    def stress(self, params, a, strain=None):
        """Stress (3, 3) in GPa, positive tensile."""
        pos, cell = self._geometry(a, strain)
        start = self._last if self._last is not None else self._start(params)
        return np.asarray(self._stress(pos, cell, params, start)) * EV_A3_TO_GPA

    def equilibrium(self, params, rel=0.06):
        """Lattice parameter minimising the energy, with the energy and aux there."""
        self._last = None
        a0 = self.phase.a_guess
        res = minimize_scalar(lambda a: self.energy(params, a)[0],
                              bounds=((1 - rel) * a0, (1 + rel) * a0),
                              method="bounded", options={"xatol": 5e-5})
        e, aux = self.energy(params, res.x)
        return res.x, e, aux

    def relax(self, params, pos, cell, ftol=1e-5, steps=200):
        """Atomic positions at fixed cell, by damped steepest descent with
        a Barzilai-Borwein step (cells of a few atoms only)."""
        start = self._last if self._last is not None else self._start(params)
        pos = jnp.asarray(pos)
        f = self._forces(pos, cell, params, start)
        step = 0.01
        for _ in range(steps):
            if float(jnp.max(jnp.abs(f))) < ftol:
                break
            new = pos + step * f
            f_new = self._forces(new, cell, params, start)
            dx, df = jnp.ravel(new - pos), jnp.ravel(f_new - f)
            denom = float(-(dx @ df))
            step = float(dx @ dx) / denom if denom > 0 else 0.01
            step = min(max(step, 1e-3), 0.2)
            pos, f = new, f_new
        return pos

    def elastic_tensor(self, params, a, delta=0.004, relax=False):
        """6x6 elastic constants (GPa, Voigt order xx yy zz yz zx xy) from
        central differences of the exact stress, with the atoms relaxed in
        each strained cell if ``relax`` (needed for hcp)."""
        voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        c = np.zeros((6, 6))
        for j, (k, l) in enumerate(voigt):
            sig = []
            for sign in (1.0, -1.0):
                eps = np.zeros((3, 3))
                eps[k, l] = eps[l, k] = sign * (delta if k == l else 0.5 * delta)
                pos, cell = self._geometry(a, eps)
                if relax:
                    pos = self.relax(params, pos, cell)
                start = self._last if self._last is not None else self._start(params)
                s = np.asarray(self._stress(pos, cell, params, start)) * EV_A3_TO_GPA
                sig.append(np.array([s[m, n] for m, n in voigt]))
            c[:, j] = (sig[0] - sig[1]) / (2 * delta)
        return 0.5 * (c + c.T)

    def cubic_elastic(self, params, a, delta=0.004):
        """C11, C12, C44 (GPa) by central differences of the exact stress."""
        e11 = np.zeros((3, 3))
        e11[0, 0] = delta
        e44 = np.zeros((3, 3))
        e44[0, 1] = e44[1, 0] = 0.5 * delta  # engineering shear strain delta
        s_p, s_m = self.stress(params, a, e11), self.stress(params, a, -e11)
        c11 = (s_p[0, 0] - s_m[0, 0]) / (2 * delta)
        c12 = 0.5 * (s_p[1, 1] - s_m[1, 1] + s_p[2, 2] - s_m[2, 2]) / (2 * delta)
        t_p, t_m = self.stress(params, a, e44), self.stress(params, a, -e44)
        c44 = (t_p[0, 1] - t_m[0, 1]) / (2 * delta)
        return c11, c12, c44


def atomic_volume(phase: Phase, a):
    cell, pos = phase.lattice(a)
    return abs(np.linalg.det(cell)) / len(pos)


def mean_abs_moment(aux):
    if "moment" not in aux:
        return 0.0
    return float(jnp.mean(jnp.abs(aux["moment"])))


__all__ = ["PHASES", "Calculator", "Phase", "atomic_volume", "bcc", "fcc", "hcp", "iron",
           "mean_abs_moment"]
