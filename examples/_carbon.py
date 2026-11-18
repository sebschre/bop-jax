"""Phases of carbon and the calculator the carbon model is fitted with.

Used by ``examples/fit_carbon.py``.
Lengths in Angstrom, energies in eV, elastic constants in GPa.  Graphene and
the linear chain sit in cells with 8 A of vacuum, far beyond the cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar

import bopjax as bj
from bopjax.hamiltonian import v0_bond
from bopjax.models import CARBON_CONFIG, carbon

EV_A3_TO_GPA = 160.21766
VACUUM = 8.0
FCC_PRIM = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]])
BCC_PRIM = 0.5 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1.0]])


# --- structures: (cell, positions) for a lattice parameter a ---


def diamond(a):
    """Primitive cell of cubic diamond, a = cubic lattice parameter."""
    return a * FCC_PRIM, a * np.array([[0, 0, 0], [0.25, 0.25, 0.25]])


def lonsdaleite(a):
    """Hexagonal diamond with ideal c/a = sqrt(8/3) and u = 3/8; a is the
    cubic-diamond parameter of the same bond length (a_hex = a / sqrt 2)."""
    ah = a / np.sqrt(2.0)
    cell = np.array([[ah, 0, 0], [-0.5 * ah, 0.5 * np.sqrt(3) * ah, 0],
                     [0, 0, np.sqrt(8 / 3) * ah]])
    frac = np.array([[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5],
                     [1 / 3, 2 / 3, 3 / 8], [2 / 3, 1 / 3, 7 / 8]])
    return cell, frac @ cell


def graphene(a):
    """Flat graphene, a = in-plane lattice parameter (bond a / sqrt 3)."""
    cell = np.array([[a, 0, 0], [-0.5 * a, 0.5 * np.sqrt(3) * a, 0], [0, 0, VACUUM]])
    return cell, np.array([[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]]) @ cell


def chain(a):
    """Linear chain with equal bonds (cumulene), a = bond length."""
    return np.diag([a, VACUUM, VACUUM]), np.zeros((1, 3))


def sc(a):
    return a * np.eye(3), np.zeros((1, 3))


def bcc(a):
    return a * BCC_PRIM, np.zeros((1, 3))


def fcc(a):
    return a * FCC_PRIM, np.zeros((1, 3))


@dataclass(frozen=True)
class Phase:
    name: str
    lattice: callable  # a -> (cell, positions)
    a_guess: float  # lattice parameter the neighbour lists are built for
    bond: float  # nearest-neighbour distance per unit of a

    @property
    def n_atoms(self):
        return len(self.lattice(self.a_guess)[1])


PHASES = {
    "diamond": Phase("diamond", diamond, 3.567, np.sqrt(3) / 4),
    "lonsdaleite": Phase("lonsdaleite", lonsdaleite, 3.567, np.sqrt(3) / 4),
    "graphene": Phase("graphene", graphene, 2.46, 1 / np.sqrt(3)),
    "chain": Phase("chain", chain, 1.30, 1.0),
    "sc": Phase("sc", sc, 1.80, 1.0),
    "bcc": Phase("bcc", bcc, 2.30, np.sqrt(3) / 2),
    "fcc": Phase("fcc", fcc, 2.90, 1 / np.sqrt(2)),
}


class Calculator:
    """Compiled energy, forces and stress of one phase, non-magnetic.

    Built once per phase; the parameters are an argument, so a fit does not
    recompile.  The neighbour lists carry a skin so that lattice parameters
    within about +-10 % of ``phase.a_guess`` need no rebuild.
    """

    def __init__(self, phase: Phase, skin=0.6, **config):
        self.phase = phase
        self.cfg = bj.BOPConfig(**dict(CARBON_CONFIG, **config))
        cell, pos = phase.lattice(phase.a_guess)
        topo = bj.build_topology(pos, [0] * len(pos), self.cfg.rcut, self.cfg.nmom, cell,
                                 rcut2=self.cfg.rcut2, skin=skin)
        energy = bj.make_energy_fn(topo, self.cfg, radial=v0_bond)
        potential = bj.make_potential(energy, tol=1e-10, max_iter=25)
        self._potential = jax.jit(potential)
        self._stress = jax.jit(bj.make_stress(potential))
        self._forces = jax.jit(lambda pos, cell, params: -jax.grad(
            lambda q: potential(q, cell, params)[0])(pos))

    def _geometry(self, a, strain=None):
        cell, pos = self.phase.lattice(a)
        if strain is not None:
            deform = np.eye(3) + strain
            cell, pos = cell @ deform, pos @ deform
        return jnp.asarray(pos), jnp.asarray(cell)

    def energy(self, params, a, strain=None):
        """(energy per atom, aux) at lattice parameter a."""
        pos, cell = self._geometry(a, strain)
        e, aux = self._potential(pos, cell, params)
        e = float(e) / self.phase.n_atoms
        if not np.isfinite(e) or float(aux["scf_residual"]) > 1e-6:
            return 1e3, aux
        return e, aux

    def equilibrium(self, params, rel=0.1):
        """Lattice parameter minimising the energy, with the energy and aux there."""
        a0 = self.phase.a_guess
        res = minimize_scalar(lambda a: self.energy(params, a)[0],
                              bounds=((1 - rel) * a0, (1 + rel) * a0),
                              method="bounded", options={"xatol": 2e-5})
        e, aux = self.energy(params, res.x)
        return res.x, e, aux

    def relax(self, params, pos, cell, ftol=1e-6, steps=200):
        """Atomic positions at fixed cell, steepest descent with a
        Barzilai-Borwein step (cells of a few atoms only)."""
        pos = jnp.asarray(pos)
        f = self._forces(pos, cell, params)
        step = 0.005
        for _ in range(steps):
            if float(jnp.max(jnp.abs(f))) < ftol:
                break
            new = pos + step * f
            f_new = self._forces(new, cell, params)
            dx, df = jnp.ravel(new - pos), jnp.ravel(f_new - f)
            denom = float(-(dx @ df))
            step = float(dx @ dx) / denom if denom > 0 else 0.005
            step = min(max(step, 1e-4), 0.05)
            pos, f = new, f_new
        return pos

    def stress(self, params, a, strain=None, relax=False):
        """Stress (3, 3) in GPa, positive tensile, atoms relaxed if ``relax``."""
        pos, cell = self._geometry(a, strain)
        if relax:
            pos = self.relax(params, pos, cell)
        return np.asarray(self._stress(pos, cell, params)) * EV_A3_TO_GPA

    def cubic_elastic(self, params, a, delta=0.004):
        """C11, C12, C44 (GPa) by central differences of the exact stress,
        atoms relaxed in each strained cell (C44 of diamond has internal
        relaxation)."""
        e11 = np.zeros((3, 3))
        e11[0, 0] = delta
        e44 = np.zeros((3, 3))
        e44[0, 1] = e44[1, 0] = 0.5 * delta  # engineering shear strain delta
        s_p, s_m = self.stress(params, a, e11), self.stress(params, a, -e11)
        c11 = (s_p[0, 0] - s_m[0, 0]) / (2 * delta)
        c12 = 0.5 * (s_p[1, 1] - s_m[1, 1] + s_p[2, 2] - s_m[2, 2]) / (2 * delta)
        t_p = self.stress(params, a, e44, relax=True)
        t_m = self.stress(params, a, -e44, relax=True)
        c44 = (t_p[0, 1] - t_m[0, 1]) / (2 * delta)
        return c11, c12, c44


def bond_length(phase: Phase, a):
    return phase.bond * a


def atomic_volume(phase: Phase, a):
    """Volume per atom; graphene and the chain have none (their cells hold vacuum)."""
    cell, pos = phase.lattice(a)
    return abs(np.linalg.det(cell)) / len(pos)


__all__ = [
    "PHASES",
    "Calculator",
    "Phase",
    "atomic_volume",
    "bcc",
    "bond_length",
    "carbon",
    "chain",
    "diamond",
    "fcc",
    "graphene",
    "lonsdaleite",
    "sc",
]
