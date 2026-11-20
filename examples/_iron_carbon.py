"""Iron-carbon cells and the calculator the Fe-C cross terms are fitted with.

Used by ``examples/fit_iron_carbon.py``.  Species 0 is Fe (the d model of
Egorov et al., ``iron_egorov``), species 1 is C (the sp model ``CARBON_FIT``).  Lengths in
Angstrom, energies in eV, stresses in GPa, moments in Bohr magnetons.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax.models import FE, IRON_CARBON_CONFIG, C, iron_carbon, tapered_exp_sum

EV_A3_TO_GPA = 160.21766
A_FE = 2.8394  # bcc FM equilibrium of the Egorov iron model (V = 11.45 A^3)
A_DIAMOND = 3.5653  # diamond equilibrium of CARBON_FIT


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
    """Self-consistent energy, forces and stress of one cell.

    The parameters are an argument, so a fit does not recompile.  The onsite
    shifts satisfy the potential condition (``make_potential(condition=
    "potential")``: shifts = dE_dc/dN, Anderson mixing), which stays well
    conditioned where the stationary condition stalls; forces and stress come
    by implicit differentiation.  ``stationarity`` is max |dE/dshift| at the
    converged state: the energy is off from the stationary one to first order
    in it (docs/formulation.md), so it is checked.  States with an occupation
    outside the capacity of a shell are rejected (a broken expansion).
    """

    def __init__(self, c: Cell, skin=0.4, config=None, mixing=0.1):
        self.c = c
        self.species = np.asarray(c.species)
        self._config = IRON_CARBON_CONFIG if config is None else config
        # remat: recompute the moment walks in the backward pass; the forces
        # of Fe16C otherwise need more than 15 GB
        self.cfg = bj.BOPConfig(spin="collinear" if c.magnetic else "none",
                                **dict(dict(remat=True), **self._config))
        topo = bj.build_topology(c.positions, c.species, self.cfg.rcut, self.cfg.nmom,
                                 c.cell, rcut2=self.cfg.rcut2, skin=skin)
        energy = bj.make_energy_fn(topo, self.cfg, radial=tapered_exp_sum)
        self.shape = energy.shift_shape
        potential = bj.make_potential(energy, tol=1e-8, condition="potential",
                                      mixing=mixing, max_iter=400)
        self._potential = jax.jit(potential)
        self._gshift = jax.jit(jax.grad(lambda x, pos, cell, p: energy(x, pos, cell, p)[0]))
        self._forces = jax.jit(lambda pos, cell, p, x0: -jax.grad(
            lambda q: potential(q, cell, p, x0)[0])(pos))
        self._stress = jax.jit(bj.make_stress(potential))
        self.cell = jnp.asarray(c.cell)
        self.positions = jnp.asarray(c.positions)
        self.n_atoms = len(c.positions)
        self._x = None  # last self-consistent shifts: the warm start

    def _start(self, params):
        if not self.c.magnetic:
            return jnp.zeros(self.shape)
        m = np.zeros((self.n_atoms, self.shape[-1]))
        m[self.species == FE, -1] = self.c.fe_moment
        return bj.stoner_shifts(m, float(params["stoner"][FE, 2]))

    @staticmethod
    def sane(aux):
        """Shell occupations within their capacity (s 2, p 6, d 10)."""
        n = np.asarray(aux["n_shell"])
        return bool(np.all(n > -0.05) and np.all(n < np.array([2, 6, 10]) + 0.05))

    def solve(self, params, positions=None, verbose=False):
        """(total energy, aux) at the given (default: current) positions."""
        pos = self.positions if positions is None else positions
        starts = ([self._x] if self._x is not None else []) + [self._start(params)]
        for x0 in starts:
            e, aux = self._potential(pos, self.cell, params, x0)
            res = float(aux["scf_residual"])
            if res < 1e-6 and np.isfinite(float(e)) and self.sane(aux):
                break
            if verbose:
                print(f"  {self.c.name}: start failed (residual {res:.1e})", flush=True)
        else:
            self._x = None
            n = np.asarray(aux["n_shell"])
            raise RuntimeError(f"{self.c.name}: no self-consistent state ({res:.2e}, "
                               f"occupations {n.min(axis=0)} .. {n.max(axis=0)})")
        self._x = aux["shifts"]
        self.stationarity = float(jnp.max(jnp.abs(self._gshift(aux["shifts"], pos,
                                                                  self.cell, params))))
        self.aux = aux
        return float(e), aux

    def forces(self, params, positions):
        """Forces (implicit differentiation of the potential condition)."""
        return self._forces(positions, self.cell, params, self._x)

    def relax(self, params, ftol=5e-3, steps=200, fixed=None, verbose=False):
        """Relax the atoms at fixed cell (FIRE), starting from the current
        positions; ``fixed`` atoms do not move.  Returns the energy and aux."""
        pos = self.positions
        keep = np.ones((self.n_atoms, 1))
        if fixed is not None:
            keep[list(fixed)] = 0.0
        v = jnp.zeros_like(pos)
        dt, dt_max, alpha, n_pos = 0.1, 0.4, 0.1, 0
        self.solve(params, pos)
        f = self.forces(params, pos) * keep
        self.n_steps = 0
        for _ in range(steps):
            if float(jnp.max(jnp.abs(f))) < ftol:
                break
            if float(jnp.vdot(f, v)) > 0:
                vn, fn = jnp.linalg.norm(v), jnp.linalg.norm(f)
                v = (1 - alpha) * v + alpha * f * vn / fn
                n_pos += 1
                if n_pos > 5:
                    dt, alpha = min(1.1 * dt, dt_max), 0.99 * alpha
            else:
                v, dt, alpha, n_pos = jnp.zeros_like(v), 0.5 * dt, 0.1, 0
            v = v + dt * f  # unit masses
            step = dt * v
            longest = float(jnp.max(jnp.linalg.norm(step, axis=1)))
            if longest > 0.1:  # no atom moves more than 0.1 A per step
                step, v = step * 0.1 / longest, v * 0.1 / longest
            pos = pos + step
            e, aux = self.solve(params, pos, verbose=verbose)
            f = self.forces(params, pos) * keep
            self.n_steps += 1
            if verbose:
                print(f" relax {self.n_steps} E {e:.6f} fmax {float(jnp.max(jnp.abs(f))):.4f} "
                      f"m {np.round(np.asarray(aux['moment'])[[1, 3, 16]], 3)}", flush=True)
        self.positions = pos
        self.max_force = float(jnp.max(jnp.abs(f)))
        return self.solve(params, pos)

    def shell_modes(self, n_shells=2, centre=None):
        """Breathing modes of the Fe shells around the carbon atom ``centre``
        (default: the last atom): unit displacements of every Fe atom of a
        shell along its direction from C.  Returns (k, N, 3)."""
        centre = self.n_atoms - 1 if centre is None else centre
        cell = np.asarray(self.cell)
        d = np.asarray(self.positions) - np.asarray(self.positions[centre])
        frac = d @ np.linalg.inv(cell)
        d = (frac - np.round(frac)) @ cell
        r = np.linalg.norm(d, axis=1)
        r[centre] = np.inf
        r[self.species == C] = np.inf
        shells = np.unique(np.round(r[np.isfinite(r)], 2))[:n_shells]
        modes = np.zeros((n_shells, self.n_atoms, 3))
        for k, rs in enumerate(shells):
            sel = np.abs(r - rs) < 0.01
            modes[k, sel] = d[sel] / r[sel, None]
        return modes

    def relax_shells(self, params, n_shells=2, gtol=2e-3, max_step=0.05, verbose=False):
        """Relax only the breathing modes of the first Fe shells around C
        (amplitudes in A, positive outward), starting from the current
        amplitudes.  Much cheaper than a full relaxation, which needs fresh
        levels after every large step; returns energy and aux."""
        if getattr(self, "_modes", None) is None:
            self._modes = jnp.asarray(self.shell_modes(n_shells))
            self._base = self.positions
            self.amplitudes = np.zeros(n_shells)
        modes, base = self._modes, self._base

        def grad(u):
            pos = base + jnp.tensordot(jnp.asarray(u), modes, 1)
            e, aux = self.solve(params, pos)
            g = -np.asarray(jnp.tensordot(modes, self.forces(params, pos), 2))
            if verbose:
                m = np.asarray(aux["moment"])
                print(f"  shells u {np.round(u, 4)} E {e:.5f} g {np.round(g, 3)} "
                      f"qC {float(aux['charge'][-1]):+.3f} m {np.round(m[[1, -1]], 2)} "
                      f"stat {self.stationarity:.1e}", flush=True)
            return e, g

        # gradient descent with Barzilai-Borwein steps, at most max_step A
        # per iteration: small moves keep every warm start close
        u = np.array(self.amplitudes, float)
        e, g = grad(u)
        step = 0.01
        for _ in range(60):
            if np.max(np.abs(g)) < gtol:
                break
            du = -step * g
            du *= min(1.0, max_step / max(np.max(np.abs(du)), 1e-12))
            e_new, g_new = grad(u + du)
            dg = g_new - g
            curv = float(du @ dg)
            step = float(du @ du) / curv if curv > 0 else 2.0 * step
            u, e, g = u + du, e_new, g_new
        self.amplitudes = u
        self.positions = base + jnp.tensordot(jnp.asarray(u), modes, 1)
        return self.solve(params)

    def stress(self, params):
        """Stress (3, 3) in GPa at the current positions, positive tensile."""
        self.solve(params)
        return np.asarray(self._stress(self.positions, self.cell, params, self._x)) * EV_A3_TO_GPA

    def volume(self):
        return float(abs(np.linalg.det(np.asarray(self.cell))))


def fe_moments(aux, species):
    return np.asarray(aux["moment"])[np.asarray(species) == FE]


__all__ = ["A_DIAMOND", "A_FE", "Calculator", "Cell", "bcc_fe", "bcc_supercell", "cementite",
           "diamond", "fe_moments", "fe_with_carbon", "iron_carbon"]
