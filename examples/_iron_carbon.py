"""Iron-carbon cells and the calculator the Fe-C cross terms are fitted with.

Used by ``examples/fit_iron_carbon.py``.  Species 0 is Fe (the d model
``IRON_FIT``), species 1 is C (the sp model ``CARBON_FIT``).  Lengths in
Angstrom, energies in eV, stresses in GPa, moments in Bohr magnetons.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax.models import FE, IRON_CARBON_CONFIG, C, iron_carbon, tapered_exp

EV_A3_TO_GPA = 160.21766
A_FE = 2.8540  # bcc FM equilibrium of IRON_FIT
A_DIAMOND = 3.5651  # diamond equilibrium of CARBON_FIT


def bcc_supercell(n=2, a=A_FE):
    """n x n x n conventional bcc cells of iron."""
    base = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    grid = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    frac = (grid[:, None, :] + base[None]).reshape(-1, 3)
    return a * n * np.eye(3), a * frac, [FE] * len(frac)


def fe_with_carbon(site, n=2, a=A_FE):
    """bcc iron with one carbon atom in an octahedral site (between two Fe
    along z, at a/2) or a tetrahedral site.  Returns cell, positions, species."""
    cell, pos, sp = bcc_supercell(n, a)
    frac = {"octahedral": [0.5, 0.5, 0.0], "tetrahedral": [0.5, 0.25, 0.0]}[site]
    return cell, np.vstack([pos, a * np.array(frac)]), sp + [C]


def cementite(a=5.0896, b=6.7443, c=4.5248):
    """Fe3C, Pnma, 12 Fe + 4 C, at the experimental lattice (Wood et al.,
    J. Appl. Cryst. 37, 2004: Fe 4c (0.0367, 1/4, 0.8402),
    Fe 8d (0.1816, 0.0666, 0.3374), C 4c (0.8778, 1/4, 0.4337))."""
    def four_c(x, z):
        return [[x, 0.25, z], [0.5 - x, 0.75, z + 0.5], [-x, 0.75, -z], [x + 0.5, 0.25, 0.5 - z]]

    def eight_d(x, y, z):
        return [[x, y, z], [0.5 - x, -y, z + 0.5], [-x, y + 0.5, -z], [x + 0.5, 0.5 - y, 0.5 - z],
                [-x, -y, -z], [x + 0.5, y, 0.5 - z], [x, 0.5 - y, z], [0.5 - x, y + 0.5, z + 0.5]]

    frac = np.array(four_c(0.0367, 0.8402) + eight_d(0.1816, 0.0666, 0.3374)
                    + four_c(0.8778, 0.4337)) % 1.0
    cell = np.diag([a, b, c])
    return cell, frac @ cell, [FE] * 12 + [C] * 4


def diamond(a=A_DIAMOND):
    fcc = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]])
    return a * fcc, a * np.array([[0, 0, 0], [0.25, 0.25, 0.25]]), [C, C]


def bcc_fe(a=A_FE):
    return a * np.eye(3), a * np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), [FE, FE]


@dataclass
class Cell:
    name: str
    cell: np.ndarray
    positions: np.ndarray
    species: list
    magnetic: bool = True
    fe_moment: float = 2.4  # starting moment of each Fe atom


class Calculator:
    """Compiled self-consistent energy, forces and stress of one cell.

    The parameters are an argument, so a fit does not recompile.  The last
    self-consistent levels and relaxed positions are kept as warm starts.
    """

    def __init__(self, c: Cell, skin=0.4):
        self.c = c
        self.species = np.asarray(c.species)
        self.cfg = bj.BOPConfig(spin="collinear" if c.magnetic else "none",
                                **IRON_CARBON_CONFIG)
        topo = bj.build_topology(c.positions, c.species, self.cfg.rcut, self.cfg.nmom,
                                 c.cell, rcut2=self.cfg.rcut2, skin=skin)
        energy = bj.make_energy_fn(topo, self.cfg, radial=tapered_exp)
        potential = bj.make_potential(energy, tol=1e-9, max_iter=30)
        self._potential = jax.jit(potential)
        self._scf = bj.make_scf(energy)
        self._stress = jax.jit(bj.make_stress(potential))
        self._forces = jax.jit(lambda pos, cell, params, start: -jax.grad(
            lambda q: potential(q, cell, params, start)[0])(pos))
        self.cell = jnp.asarray(c.cell)
        self.positions = jnp.asarray(c.positions)
        self._last = None
        self.n_atoms = len(c.positions)

    def _start(self, params):
        if not self.c.magnetic:
            return None
        m = np.zeros((self.n_atoms, 3))
        m[self.species == FE, 2] = self.c.fe_moment
        return bj.stoner_shifts(m, float(params["stoner"][FE, 2]))

    def solve(self, params, positions=None):
        """(total energy, aux) at the given (default: current) positions."""
        pos = self.positions if positions is None else positions
        start = self._last if self._last is not None else self._start(params)
        e, aux = self._potential(pos, self.cell, params, start)
        if float(aux["scf_residual"]) > 1e-7:
            res = self._scf(pos, self.cell, params, self._start(params), tol=1e-10,
                            max_iter=80)
            e, aux = self._potential(pos, self.cell, params, res.shifts)
        if float(aux["scf_residual"]) > 1e-6:
            raise RuntimeError(f"{self.c.name}: no self-consistent state")
        self._last = aux["shifts"]
        return float(e), aux

    def forces(self, params, positions):
        start = self._last if self._last is not None else self._start(params)
        return self._forces(positions, self.cell, params, start)

    def relax(self, params, ftol=2e-3, steps=300, fixed=None):
        """Relax the atoms at fixed cell (FIRE), starting from the current
        positions; ``fixed`` atoms do not move.  Returns the energy and aux."""
        pos = self.positions
        keep = np.ones((self.n_atoms, 1))
        if fixed is not None:
            keep[list(fixed)] = 0.0
        v = jnp.zeros_like(pos)
        dt, dt_max, alpha, n_pos = 0.05, 0.2, 0.1, 0
        self.solve(params, pos)
        f = self.forces(params, pos) * keep
        for _ in range(steps):
            if float(jnp.max(jnp.abs(f))) < ftol:
                break
            power = float(jnp.vdot(f, v))
            if power > 0:
                vn, fn = jnp.linalg.norm(v), jnp.linalg.norm(f)
                v = (1 - alpha) * v + alpha * f * vn / fn
                n_pos += 1
                if n_pos > 5:
                    dt, alpha = min(1.1 * dt, dt_max), 0.99 * alpha
            else:
                v, dt, alpha, n_pos = jnp.zeros_like(v), 0.5 * dt, 0.1, 0
            v = v + dt * f  # unit masses (A, eV, time unit sqrt(amu) scaled away)
            pos = pos + dt * v
            self.solve(params, pos)
            f = self.forces(params, pos) * keep
        self.positions = pos
        self.max_force = float(jnp.max(jnp.abs(f)))
        return self.solve(params, pos)

    def stress(self, params):
        start = self._last if self._last is not None else self._start(params)
        return np.asarray(self._stress(self.positions, self.cell, params, start)) * EV_A3_TO_GPA

    def volume(self):
        return float(abs(np.linalg.det(np.asarray(self.cell))))


def fe_moments(aux, species):
    return np.asarray(aux["moment"])[np.asarray(species) == FE]


__all__ = ["A_DIAMOND", "A_FE", "Calculator", "Cell", "bcc_fe", "bcc_supercell", "cementite",
           "diamond", "fe_moments", "fe_with_carbon", "iron_carbon"]
