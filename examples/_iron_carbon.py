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
    """Self-consistent energy, forces and stress of one cell.

    The parameters are an argument, so a fit does not recompile.  The onsite
    levels are solved by Newton's method with a cached Hessian (a chord
    method): the Hessian on the levels of the shells each atom has is built
    once and reused across relaxation steps and nearby parameters until the
    residual stops dropping fast.  Every iteration then costs one gradient,
    about 25 times less than a Newton-Krylov step on these cells.  The last
    levels and positions are kept as warm starts.
    """

    def __init__(self, c: Cell, skin=0.4):
        self.c = c
        self.species = np.asarray(c.species)
        self.cfg = bj.BOPConfig(spin="collinear" if c.magnetic else "none",
                                **IRON_CARBON_CONFIG)
        topo = bj.build_topology(c.positions, c.species, self.cfg.rcut, self.cfg.nmom,
                                 c.cell, rcut2=self.cfg.rcut2, skin=skin)
        energy = bj.make_energy_fn(topo, self.cfg, radial=tapered_exp)
        shape = energy.shift_shape
        own = np.array([[0, 0, 1], [1, 1, 0]], bool)[self.species]  # (N, shells s p d)
        mask = np.broadcast_to(own[:, None, :], shape) if c.magnetic else own
        free = np.flatnonzero(mask.ravel())
        size = int(np.prod(shape))
        self.n_free = len(free)

        def full(x):
            return jnp.zeros(size, x.dtype).at[free].set(x).reshape(shape)

        def e_of(x, pos, cell, params):
            return energy(full(x), pos, cell, params)

        def e_only(x, pos, cell, params):
            return e_of(x, pos, cell, params)[0]

        def strained(strain, x, pos, cell, params):
            deform = jnp.eye(3) + 0.5 * (strain + strain.T)
            return e_only(x, pos @ deform, cell @ deform, params)

        self._full = jax.jit(full)
        self._vg = jax.jit(jax.value_and_grad(e_of, has_aux=True))
        grad_x = jax.grad(e_only)

        def hessian(x, pos, cell, params):
            # columns in small batches: all tangents at once need ~20 GB
            def column(v):
                return jax.jvp(lambda y: grad_x(y, pos, cell, params), (x,), (v,))[1]
            return jax.lax.map(column, jnp.eye(len(x), dtype=x.dtype), batch_size=4)

        self._hess = jax.jit(hessian)
        self._dpos = jax.jit(jax.grad(e_only, argnums=1))
        self._dstrain = jax.jit(jax.grad(strained))
        self.cell = jnp.asarray(c.cell)
        self.positions = jnp.asarray(c.positions)
        self.n_atoms = len(c.positions)
        self._x = None  # last self-consistent levels (free entries)
        self._hinv = None  # cached inverse Hessian
        self.n_hessian = 0

    def _start(self, params):
        if not self.c.magnetic:
            return jnp.zeros(self.n_free)
        m = np.zeros((self.n_atoms, 3))
        m[self.species == FE, 2] = self.c.fe_moment
        full = np.asarray(bj.stoner_shifts(m, float(params["stoner"][FE, 2])))
        own = np.array([[0, 0, 1], [1, 1, 0]], bool)[self.species]
        return jnp.asarray(full[np.broadcast_to(own[:, None, :], full.shape)])

    def _new_hessian(self, x, pos, params):
        h = self._hess(x, pos, self.cell, params)
        # soft modes (|eigenvalue| < 1e-3 of the largest) make the step blow up
        self._hinv = jnp.linalg.pinv(0.5 * (h + h.T), rtol=1e-3, hermitian=True)
        self.n_hessian += 1

    def solve(self, params, positions=None, tol=1e-8, max_iter=60, verbose=False):
        """(total energy, aux) at the given (default: current) positions."""
        pos = self.positions if positions is None else positions
        x = self._x if self._x is not None else self._start(params)
        (e, aux), g = self._vg(x, pos, self.cell, params)
        if self._hinv is None:
            self._new_hessian(x, pos, params)
        fresh, stale = False, False
        for _ in range(max_iter):
            if float(jnp.max(jnp.abs(g))) < tol:
                break
            norm = float(jnp.linalg.norm(g))
            if stale and not fresh:
                self._new_hessian(x, pos, params)
                fresh = True
            step = -(self._hinv @ g)
            accepted = False
            for _ in range(8):  # backtracking: far from the solution Newton overshoots
                x_new = x + step
                (e_new, aux_new), g_new = self._vg(x_new, pos, self.cell, params)
                norm_new = float(jnp.linalg.norm(g_new))
                if np.isfinite(norm_new) and norm_new < 0.95 * norm:
                    accepted = True
                    break
                step = 0.5 * step
            if not accepted:
                if fresh:
                    break  # Newton itself fails
                stale = True
                continue
            if verbose:
                print(f"  scf |g| {norm:.3e} -> {norm_new:.3e}  fresh={fresh}", flush=True)
            # slow progress with an old Hessian: rebuild it at the next step
            stale = norm_new > 0.3 * norm
            fresh = False
            x, e, aux, g = x_new, e_new, aux_new, g_new
        self.residual = float(jnp.max(jnp.abs(g)))
        if self.residual > 1e-6:
            self._x, self._hinv = None, None
            raise RuntimeError(f"{self.c.name}: no self-consistent state ({self.residual:.2e})")
        self._x = x
        self.aux = aux
        return float(e), aux

    def forces(self, params, positions):
        """Forces at the last self-consistent levels (call ``solve`` first)."""
        return -self._dpos(self._x, positions, self.cell, params)

    def relax(self, params, ftol=5e-3, steps=200, fixed=None):
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
            self.solve(params, pos)
            f = self.forces(params, pos) * keep
            self.n_steps += 1
        self.positions = pos
        self.max_force = float(jnp.max(jnp.abs(f)))
        return self.solve(params, pos)

    def stress(self, params):
        """Stress (3, 3) in GPa at the current positions, positive tensile."""
        self.solve(params)
        d = self._dstrain(jnp.zeros((3, 3)), self._x, self.positions, self.cell, params)
        return np.asarray(d) / self.volume() * EV_A3_TO_GPA

    def volume(self):
        return float(abs(np.linalg.det(np.asarray(self.cell))))


def fe_moments(aux, species):
    return np.asarray(aux["moment"])[np.asarray(species) == FE]


__all__ = ["A_DIAMOND", "A_FE", "Calculator", "Cell", "bcc_fe", "bcc_supercell", "cementite",
           "diamond", "fe_moments", "fe_with_carbon", "iron_carbon"]
