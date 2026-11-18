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

    The parameters are an argument, so a fit does not recompile.  The onsite
    levels are found in two stages.  From a cold start they are first solved
    with one level per atom (iron: one per spin): all shells of an atom
    shifted together.  That removes a soft mode, the C 2s level against 2p
    (2s is nearly full and hardly responds), along which Newton overshoots.
    The levels of every shell are then converged from there, and from the
    previous solution on later calls.  Both stages are chord-Newton
    (``_Chord``).
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
        size = int(np.prod(shape))
        idx = np.arange(size).reshape(shape)
        tied, own = [], []  # columns of the two subspaces: lists of level indices
        for i, sp in enumerate(self.species):
            shells = [2] if sp == FE else [0, 1]
            if c.magnetic and sp == FE:
                tied += [idx[i, 0, shells], idx[i, 1, shells]]
            elif c.magnetic:
                tied.append(idx[i][:, shells].ravel())
            else:
                tied.append(idx[i, shells])
            for rows in tied[-2 if c.magnetic and sp == FE else -1:]:
                own += [[r] for r in rows]

        def basis(cols):
            b = np.zeros((size, len(cols)))
            for k, rows in enumerate(cols):
                b[rows, k] = 1.0
            return b

        b_tied, b_own = basis(tied), basis(own)
        self._own_is_c = np.array([self.species[np.unravel_index(rows[0], shape)[0]] == C
                                   for rows in own])
        # tied columns that belong to carbon atoms
        self._tied_is_c = np.array([self.species[np.unravel_index(rows[0], shape)[0]] == C
                                    for rows in tied], float)
        self._tied_to_own = b_own.T @ b_tied  # own levels of a tied solution
        self._tied = _Chord(energy, b_tied, shape)
        self._own = _Chord(energy, b_own, shape)
        self._b_own = jnp.asarray(b_own)

        def e_own(x, pos, cell, params):
            return energy((self._b_own @ x).reshape(shape), pos, cell, params)[0]

        def strained(strain, x, pos, cell, params):
            deform = jnp.eye(3) + 0.5 * (strain + strain.T)
            return e_own(x, pos @ deform, cell @ deform, params)

        self._dpos = jax.jit(jax.grad(e_own, argnums=1))
        self._dstrain = jax.jit(jax.grad(strained))
        self.cell = jnp.asarray(c.cell)
        self.positions = jnp.asarray(c.positions)
        self.n_atoms = len(c.positions)
        self._x = None  # last self-consistent levels (own subspace)

    @property
    def n_hessian(self):
        return self._tied.n_hessian + self._own.n_hessian

    def _start(self, params):
        """Tied levels of the starting moments (Stoner splitting on Fe d)."""
        x = []
        split = 0.5 * float(params["stoner"][FE, 2]) * self.c.fe_moment
        for sp in self.species:
            x += ([-split, split] if self.c.magnetic and sp == FE else [0.0])
        return jnp.asarray(x)

    def solve(self, params, positions=None, tol=1e-8, max_iter=60, verbose=False):
        """(total energy, aux) at the given (default: current) positions."""
        pos = self.positions if positions is None else positions
        x = self._x
        if x is None:
            x = self._block_start(params, pos, verbose)
        x, e, aux, res = self._own(x, pos, self.cell, params, tol, max_iter, verbose)
        self.residual = res
        if res > 1e-6:
            self._x = None
            raise RuntimeError(f"{self.c.name}: no self-consistent state ({res:.2e})")
        self._x = x
        self.aux = aux
        return float(e), aux

    def _block_start(self, params, pos, verbose=False, cap=0.5):
        """Levels from a cold start, by blocks.  Carbon can take or give most
        of an electron, which the Hubbard term answers with shifts of several
        eV; joint Newton steps from zero wander into levels where the C
        expansion diverges and |g| has false minima.  So: converge the iron
        levels with carbon fixed, take a Newton step of at most ``cap`` eV in
        the carbon block, and repeat until the joint solver can take over."""
        n = len(self._own_is_c)
        x = np.asarray(self._tied_to_own @ np.asarray(self._start(params)))
        fe, cc = np.flatnonzero(~self._own_is_c), np.flatnonzero(self._own_is_c)
        eye = np.eye(n)
        vg = self._own._vg
        h_fe = None
        for outer in range(40):
            for it in range(15):  # iron block, cached Hessian
                g = np.asarray(vg(jnp.asarray(x), pos, self.cell, params)[1])
                if np.max(np.abs(g[fe])) < 1e-4:
                    break
                if h_fe is None or it == 6:
                    h = np.asarray(self._own.columns(jnp.asarray(x), pos, self.cell, params,
                                                     jnp.asarray(eye[fe])))[fe]
                    h_fe = np.linalg.pinv(0.5 * (h + h.T), rcond=1e-10)
                step = -h_fe @ g[fe]
                x[fe] += step * min(1.0, cap / max(np.max(np.abs(step)), 1e-12))
            res = float(np.max(np.abs(g)))
            if verbose:
                print(f"  block {outer}: |g_Fe| {np.max(np.abs(g[fe])):.2e} "
                      f"|g_C| {np.max(np.abs(g[cc])):.2e}", flush=True)
            if res < 1e-3 or not np.all(np.isfinite(g)):
                break
            h = np.asarray(self._own.columns(jnp.asarray(x), pos, self.cell, params,
                                             jnp.asarray(eye[cc])))[cc]
            step = -np.linalg.pinv(0.5 * (h + h.T), rcond=1e-8) @ g[cc]
            x[cc] += step * min(1.0, cap / max(np.max(np.abs(step)), 1e-12))
        self._own.h = None  # the joint solver builds its own Hessian here
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
