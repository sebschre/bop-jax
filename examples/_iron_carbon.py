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


class _Chord:
    """Newton's method for stationary levels in the subspace ``levels = B x``,
    with a cached Hessian (a chord method) and a Levenberg-Marquardt fallback
    on |g|^2 as in ``bopjax.make_scf``.

    The Hessian is built once and reused across relaxation steps and nearby
    parameters until the residual stops dropping fast; every iteration then
    costs one gradient.
    """

    def __init__(self, energy, basis, shape):
        self.basis = jnp.asarray(basis)

        def e_of(x, pos, cell, params):
            return energy((self.basis @ x).reshape(shape), pos, cell, params)

        grad_x = jax.grad(lambda *a: e_of(*a)[0])

        def hessian(x, pos, cell, params):
            # columns in small batches: all tangents at once need ~20 GB
            def column(v):
                return jax.jvp(lambda y: grad_x(y, pos, cell, params), (x,), (v,))[1]
            return jax.lax.map(column, jnp.eye(len(x), dtype=x.dtype), batch_size=4)

        def columns(x, pos, cell, params, vs):
            """H @ vs.T for a few directions vs (k, n)."""
            def column(v):
                return jax.jvp(lambda y: grad_x(y, pos, cell, params), (x,), (v,))[1]
            return jax.lax.map(column, vs, batch_size=4).T

        self._vg = jax.jit(jax.value_and_grad(e_of, has_aux=True))
        self._hess = jax.jit(hessian)
        self.columns = jax.jit(columns)
        self.h = None
        self.n_hessian = 0

    def _new_hessian(self, x, pos, cell, params):
        h = self._hess(x, pos, cell, params)
        self.h = 0.5 * (h + h.T)
        self.hinv = jnp.linalg.pinv(self.h, rtol=1e-10, hermitian=True)
        self.normal = self.h @ self.h
        self.n_hessian += 1

    def __call__(self, x, pos, cell, params, tol, max_iter, verbose=False):
        (e, aux), g = self._vg(x, pos, cell, params)
        if self.h is None:
            self._new_hessian(x, pos, cell, params)
        fresh, stale = False, False
        for _ in range(max_iter):
            if float(jnp.max(jnp.abs(g))) < tol:
                break
            norm = float(jnp.linalg.norm(g))
            if stale and not fresh:
                self._new_hessian(x, pos, cell, params)
                fresh = True
            accepted, damping = False, 0.0
            scale = float(jnp.max(jnp.diag(self.normal)))
            for _ in range(10):
                if damping == 0.0:
                    step = -(self.hinv @ g)
                else:
                    step = -jnp.linalg.solve(self.normal + damping * jnp.eye(len(x)),
                                             self.h @ g)
                (e_new, aux_new), g_new = self._vg(x + step, pos, cell, params)
                norm_new = float(jnp.linalg.norm(g_new))
                if np.isfinite(norm_new) and norm_new < 0.95 * norm:
                    accepted = True
                    break
                damping = 1e-4 * scale if damping == 0.0 else 10.0 * damping
            if not accepted and fresh:
                # The residual lies along a soft mode (the C 2s level against
                # 2p): Newton's step there is far too long and |g|^2 has a
                # false minimum.  Search along the softest modes that carry it.
                w, v = jnp.linalg.eigh(self.h)
                proj = v.T @ g
                carry = jnp.abs(proj) > 0.2 * jnp.linalg.norm(g)
                order = jnp.argsort(jnp.where(carry, jnp.abs(w), jnp.inf))
                best = (norm, None)
                for k in [int(i) for i in order[:2] if bool(carry[int(i)])]:
                    direction = -jnp.sign(proj[k]) * v[:, k]
                    for t in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, -0.1, -0.4, -1.6):
                        (e_t, aux_t), g_t = self._vg(x + t * direction, pos, cell, params)
                        n_t = float(jnp.linalg.norm(g_t))
                        if np.isfinite(n_t) and n_t < best[0]:
                            best = (n_t, (t * direction, e_t, aux_t, g_t))
                if best[1] is not None and best[0] < 0.95 * norm:
                    step, e_new, aux_new, g_new = best[1]
                    norm_new = best[0]
                    accepted = True
            if not accepted:
                if fresh:
                    break
                stale = True
                continue
            if verbose:
                print(f"  scf {len(x)} |g| {norm:.3e} -> {norm_new:.3e}  fresh={fresh}",
                      flush=True)
            stale = norm_new > 0.3 * norm  # slow with an old Hessian: rebuild
            fresh = False
            x, e, aux, g = x + step, e_new, aux_new, g_new
        return x, e, aux, float(jnp.max(jnp.abs(g)))


class Calculator:
    """Self-consistent energy, forces and stress of one cell.

    The parameters are an argument, so a fit does not recompile.  The model
    has one onsite shift per atom (and spin), so the unknowns are the
    N (or 2N) shifts, solved by chord-Newton (``_Chord``) from the previous
    solution.  Cold starts go by blocks (``_block_start``); magnetic cells
    start from the non-magnetic levels plus a Stoner splitting on iron.
    """

    def __init__(self, c: Cell, skin=0.4):
        self.c = c
        self.species = np.asarray(c.species)
        self.cfg = bj.BOPConfig(spin="collinear" if c.magnetic else "none",
                                **IRON_CARBON_CONFIG)
        topo = bj.build_topology(c.positions, c.species, self.cfg.rcut, self.cfg.nmom,
                                 c.cell, rcut2=self.cfg.rcut2, skin=skin)
        energy = bj.make_energy_fn(topo, self.cfg, radial=tapered_exp)
        shape = energy.shift_shape  # (N, 1) or (N, 2, 1)
        size = int(np.prod(shape))
        self._solver = _Chord(energy, np.eye(size), shape)
        # which unknown belongs to a carbon atom (atom index is the leading axis)
        self._is_c = np.repeat(self.species == C, size // len(self.species))

        def e_x(x, pos, cell, params):
            return energy(x.reshape(shape), pos, cell, params)[0]

        def strained(strain, x, pos, cell, params):
            deform = jnp.eye(3) + 0.5 * (strain + strain.T)
            return e_x(x, pos @ deform, cell @ deform, params)

        self._dpos = jax.jit(jax.grad(e_x, argnums=1))
        self._dstrain = jax.jit(jax.grad(strained))
        self.cell = jnp.asarray(c.cell)
        self.positions = jnp.asarray(c.positions)
        self.n_atoms = len(c.positions)
        self._x = None  # last self-consistent shifts (flat)
        self._nm = None  # non-magnetic twin for cold starts of magnetic cells

    @property
    def n_hessian(self):
        return self._solver.n_hessian

    def solve(self, params, positions=None, tol=1e-8, max_iter=60, verbose=False):
        """(total energy, aux) at the given (default: current) positions."""
        pos = self.positions if positions is None else positions
        x, cold = self._x, self._x is None
        if cold:
            x = (self._from_nonmagnetic(params, pos, verbose) if self.c.magnetic
                 else self._block_start(params, pos, verbose))
        x, e, aux, res = self._solver(x, pos, self.cell, params, tol, max_iter, verbose)
        if (res > 1e-6 or not self.sane(aux)) and not cold:  # warm start failed
            self._x = None
            return self.solve(params, positions, tol, max_iter, verbose)
        self.residual = res
        if res > 1e-6 or not self.sane(aux):
            self._x = None
            n = np.asarray(aux["n_shell"])
            raise RuntimeError(f"{self.c.name}: no self-consistent state ({res:.2e}, "
                               f"occupations {n.min(axis=0)} .. {n.max(axis=0)})")
        self._x = x
        self.aux = aux
        return float(e), aux

    def _from_nonmagnetic(self, params, pos, verbose=False):
        """Cold start of a magnetic cell: the non-magnetic shifts plus the
        Stoner splitting of the starting moment on iron."""
        if self._nm is None:
            self._nm = Calculator(Cell(self.c.name + " NM", self.c.cell, self.c.positions,
                                       self.c.species, magnetic=False))
        self._nm.positions = jnp.asarray(pos)
        self._nm._x = None
        self._nm.solve(params, verbose=verbose)
        x_nm = np.asarray(self._nm._x)
        split = 0.5 * float(params["stoner"][FE, 2]) * self.c.fe_moment
        m = np.where(self.species == FE, split, 0.0)
        self._solver.h = None
        return jnp.asarray(np.stack([x_nm - m, x_nm + m], axis=1).ravel())

    @staticmethod
    def sane(aux):
        """Shell occupations within their capacity (s 2, p 6, d 10): outside
        it the expansion of the local density of states has broken down
        (far-shifted levels) and a stationary point there is spurious."""
        n = np.asarray(aux["n_shell"])
        return bool(np.all(n > -0.05) and np.all(n < np.array([2, 6, 10]) + 0.05))

    def _block_start(self, params, pos, verbose=False, cap=0.25):
        """Shifts from a cold start, by blocks.  Carbon can take or give most
        of an electron, which the Hubbard term answers with shifts of several
        eV; joint Newton steps from zero can wander into levels where the
        expansion of the C density of states breaks down and has spurious
        stationary points.  So: pick the carbon shift from a scan of sane
        states, converge the iron shifts with carbon fixed, take a capped,
        backtracked Newton step for carbon, and repeat until the joint solver
        can take over."""
        n = len(self._is_c)
        fe, cc = np.flatnonzero(~self._is_c), np.flatnonzero(self._is_c)
        if len(fe) == 0 or len(cc) == 0:  # one element: the joint solver alone
            self._solver.h = None
            return jnp.zeros(n)
        eye = np.eye(n)
        vg, cols = self._solver._vg, self._solver.columns

        def evaluate(x):
            (_, aux), g = vg(jnp.asarray(x), pos, self.cell, params)
            g = np.asarray(g)
            ok = self.sane(aux) and np.all(np.isfinite(g))
            return g, ok

        best = (np.inf, None)
        for shift in np.arange(-3.0, 3.01, 0.5):
            x = np.where(self._is_c, shift, 0.0)
            g, ok = evaluate(x)
            if ok and np.max(np.abs(g[cc])) < best[0]:
                best = (np.max(np.abs(g[cc])), x)
        if best[1] is None:
            raise RuntimeError(f"{self.c.name}: no sane starting levels")
        x = best[1]
        h_fe = None
        for outer in range(40):
            for it in range(15):  # iron block, cached Hessian, capped steps
                g, ok = evaluate(x)
                if np.max(np.abs(g[fe])) < 1e-4:
                    break
                if h_fe is None or it == 6:
                    h = np.asarray(cols(jnp.asarray(x), pos, self.cell, params,
                                        jnp.asarray(eye[fe])))[fe]
                    h_fe = np.linalg.pinv(0.5 * (h + h.T), rcond=1e-10)
                step = -h_fe @ g[fe]
                x[fe] += step * min(1.0, cap / max(np.max(np.abs(step)), 1e-12))
            if verbose:
                print(f"  block {outer}: |g_Fe| {np.max(np.abs(g[fe])):.2e} "
                      f"|g_C| {np.max(np.abs(g[cc])):.2e} sane={ok}", flush=True)
            if np.max(np.abs(g)) < 1e-3 and ok:
                break
            h = np.asarray(cols(jnp.asarray(x), pos, self.cell, params,
                                jnp.asarray(eye[cc])))[cc]
            step = -np.linalg.pinv(0.5 * (h + h.T), rcond=1e-8) @ g[cc]
            step *= min(1.0, cap / max(np.max(np.abs(step)), 1e-12))
            for _ in range(5):  # stay among sane states
                trial = x.copy()
                trial[cc] += step
                if evaluate(trial)[1]:
                    break
                step *= 0.5
            x = trial
        self._solver.h = None  # the joint solver builds its own Hessian here
        return jnp.asarray(x)

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
            self.solve(params, pos, tol=1e-7)
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
