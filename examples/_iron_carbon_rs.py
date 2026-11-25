"""Iron-carbon calculator on bopfox-rs (Rust, analytic forces).

The same model as ``examples/_iron_carbon.py`` (``bopjax.models.iron_carbon``,
``IRON_CARBON_CONFIG``, potential self-consistency), evaluated by the Rust
port instead of JAX: exact (implicit-differentiation) forces and stress of
Fe16C take about 10 s from a cold start instead of minutes, with no
compilation and a few hundred MB of memory, so every atom can be relaxed
inside a fit.  Needs the bindings (``cd bopfox-rs && maturin develop --release``)
and scipy.
"""

from __future__ import annotations

import numpy as np

import bopfox_rs
import bopjax as bj
from bopjax.models import FE, IRON_CARBON_CONFIG, tapered_exp_sum

from _iron_carbon import (A_DIAMOND, A_FE, EV_A3_TO_GPA, bcc_fe, bcc_supercell, cementite,  # noqa: F401
                          diamond, fe_with_carbon)


def site_symmetry(cell, positions, species, centre, tol=1e-6):
    """Point operations of the cell about atom ``centre``: the signed
    permutation matrices R (of the 48 cubic ones) that map every atom onto
    an atom of the same species (minimum image).  Returns a list of (R, perm)
    with ``positions[perm[i]] = c + R (positions[i] - c)`` (mod the cell)."""
    from itertools import permutations, product
    cell = np.asarray(cell, float)
    inv = np.linalg.inv(cell)
    c = positions[centre]
    ops = []
    for perm_axes in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            for k in range(3):
                r[k, perm_axes[k]] = signs[k]
            img = c + (positions - c) @ r.T
            perm = []
            for i, x in enumerate(img):
                d = positions - x
                d -= np.round(d @ inv) @ cell
                j = int(np.argmin(np.linalg.norm(d, axis=1)))
                if np.linalg.norm(d[j]) > tol or species[j] != species[i]:
                    break
                perm.append(j)
            else:
                ops.append((r, np.array(perm)))
    return ops


def make_potential(params, magnetic=True):
    cfg = bj.BOPConfig(spin="collinear" if magnetic else "none", **IRON_CARBON_CONFIG)
    return bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, params, tapered_exp_sum))


class RsCell:
    """One cell: self-consistent energy, exact forces and stress, and
    relaxation, warm-started from the previous shifts (and Anderson history).

    States whose self-consistency fails or whose shell occupations leave
    their capacity are retried from a Stoner start, then rejected.
    """

    def __init__(self, name, cell, positions, species, magnetic=True, fe_moment=2.4,
                 tol=1e-6, accept=1e-5, adjoint_tol=1e-7):
        self.name = name
        self.cell = np.asarray(cell, float)
        self.positions = np.asarray(positions, float)
        self.species = np.asarray(species)
        self.magnetic = magnetic
        self.fe_moment = fe_moment
        self.tol, self.accept, self.adjoint_tol = tol, accept, adjoint_tol
        self.evaluations = 0
        self.pot = None
        self._x = None
        self.n = len(self.positions)

    def set_params(self, params):
        self.params = params
        self.pot = make_potential(params, self.magnetic)

    def _start(self):
        if not self.magnetic:
            return np.zeros((self.n, 1, 3))
        m = np.zeros((self.n, 3))
        m[self.species == FE, 2] = self.fe_moment
        return np.asarray(bj.stoner_shifts(m, float(self.params["stoner"][FE, 2])))

    @staticmethod
    def sane(out):
        n = out["n_shell"]
        return bool(np.all(n > -0.05) and np.all(n < np.array([2, 6, 10]) + 0.05))

    def solve(self, positions=None):
        pos = self.positions if positions is None else np.asarray(positions, float)
        # full Anderson steps from a warm start; the cautious default mixing
        # (0.3) for cold starts and as the fallback
        tries = []
        if self._x is not None:
            tries += [(self._x, True, 1.0), (self._x, False, None)]
        tries += [(self._start(), False, None), (self._start(), False, 0.05)]
        for x0, reuse, mixing in tries:
            if not reuse:
                self.pot.clear_history()
            out = self.pot.scf(pos, self.species, self.cell, x0, condition="potential",
                               tol=self.tol, max_iter=400, reuse_history=reuse, mixing=mixing,
                               adjoint_tol=self.adjoint_tol)
            self.evaluations += out["evaluations"]
            if out["residual"] < self.accept and np.isfinite(out["energy"]) and self.sane(out):
                self._x = out["shifts"]
                self.out = out
                return out
        self._x = None
        raise RuntimeError(f"{self.name}: no self-consistent state (residual {out['residual']:.1e})")

    def ground_state(self, positions=None, moments=(1.6, 2.0, 2.4, 2.8, 3.2),
                     mixings=(None, 0.05)):
        """The lowest-energy self-consistent state at the given (default:
        current) positions, from the current state and cold starts with
        Fe moments ``moments`` (and each mixing).  Iron next to carbon has
        several self-consistent magnetic states, eV apart in Fe16C; a warm
        start keeps whichever it came from."""
        pos = self.positions if positions is None else np.asarray(positions, float)
        best = None
        if self._x is not None:
            try:
                best = self.solve(pos)
            except RuntimeError:
                pass
        if self.magnetic:
            keep = self.fe_moment
            for mom in moments:
                for mixing in mixings:
                    self.fe_moment = mom
                    self.pot.clear_history()
                    out = self.pot.scf(pos, self.species, self.cell, self._start(),
                                       condition="potential", tol=self.tol, max_iter=600,
                                       mixing=mixing, adjoint_tol=self.adjoint_tol)
                    self.evaluations += out["evaluations"]
                    if (out["residual"] < self.accept and np.isfinite(out["energy"])
                            and self.sane(out)
                            and (best is None or out["energy"] < best["energy"] - 1e-6)):
                        best = out
            self.fe_moment = keep
        if best is None:
            raise RuntimeError(f"{self.name}: no self-consistent state")
        self._x, self.out = best["shifts"], best
        self.pot.clear_history()
        return best

    def relax_ground(self, rounds=3, **kw):
        """Relax on the lowest-energy magnetic state: find it at the current
        positions, relax, and repeat while another state is lower at the
        relaxed positions.  Takes the arguments of ``relax``."""
        self.ground_state()
        for _ in range(rounds):
            out = self.relax(**kw)
            e = out["energy"]
            new = self.ground_state()
            if new["energy"] > e - 1e-4:
                self.ground_state_switches = getattr(self, "ground_state_switches", 0)
                return out
            self.ground_state_switches = getattr(self, "ground_state_switches", 0) + 1
        return self.relax(**kw)

    def volume(self):
        return float(abs(np.linalg.det(self.cell)))

    def stress(self, out=None):
        """Stress (3, 3) in GPa, positive tensile."""
        out = self.out if out is None else out
        return np.asarray(out["virial"]) / self.volume() * EV_A3_TO_GPA

    def relax(self, fixed=(), ftol=2e-3, maxiter=200, max_step=0.05, memory=12, verbose=False,
              fix_centre=False, symmetry=None, fixed_components=()):
        """Relax every atom but ``fixed`` at fixed cell on the exact forces:
        L-BFGS with no atom moving more than ``max_step`` A per step (small
        moves keep each warm start close), a step accepted when the energy
        does not rise, else halved with the memory cleared.  Stops when no
        force component exceeds ``ftol`` (eV/A).  Returns the results at the
        relaxed positions.  ``fix_centre`` also keeps the centre of the free
        atoms in place (their mean force is removed): with only a carbon atom
        held on a saddle site, the lattice would otherwise slide rigidly
        around it into a stable site.  ``symmetry`` (from ``site_symmetry``)
        averages every gradient over those operations, so that symmetric
        positions stay symmetric (a saddle point held by its symmetry).
        ``fixed_components`` holds single coordinates, (atom, axis): carbon's
        coordinate along a migration path, with the Fe centre held, gives the
        lowest energy in the plane of the saddle (the drag method)."""
        mask = np.ones((self.n, 3))
        mask[list(fixed)] = 0.0
        for a, k in fixed_components:
            mask[a, k] = 0.0
        lattice = (mask.sum(axis=1) == 3) & (self.species == FE)  # whose centre is held
        pos = self.positions.copy()
        out = self.solve(pos)

        def gradient(o):
            g_ = -o["forces"] * mask
            if symmetry:
                gs = np.zeros_like(g_)
                for r, perm in symmetry:
                    gs[perm] += g_ @ r.T
                g_ = gs / len(symmetry) * mask
            if fix_centre:
                g_[lattice] -= g_[lattice].mean(axis=0)
            return g_

        g = gradient(out)
        s_hist, y_hist = [], []
        scale = max_step
        self.n_steps = 0
        for _ in range(maxiter):
            if np.abs(g).max() < ftol:
                break
            # two-loop recursion
            q = g.ravel().copy()
            alphas = []
            for s, y in zip(reversed(s_hist), reversed(y_hist)):
                a = (s @ q) / (y @ s)
                alphas.append(a)
                q -= a * y
            if s_hist:
                s, y = s_hist[-1], y_hist[-1]
                q *= (s @ y) / (y @ y)
            else:
                q *= 0.01  # A^2/eV: a stiff first guess
            for (s, y), a in zip(zip(s_hist, y_hist), reversed(alphas)):
                b = (y @ q) / (y @ s)
                q += (a - b) * s
            step = -q.reshape(-1, 3) * mask
            if (step.ravel() @ g.ravel()) >= 0:  # not a descent direction
                s_hist, y_hist = [], []
                step = -0.01 * g
            longest = np.linalg.norm(step, axis=1).max()
            step *= min(1.0, scale / longest)
            try:
                new = self.solve(pos + step)
            except RuntimeError:  # no self-consistent state there: a bad step
                new = None
            if new is None or new["energy"] > out["energy"] + 1e-7:
                s_hist, y_hist = [], []
                scale = 0.5 * min(scale, longest)
                if scale < 1e-4:
                    break
                self.solve(pos)  # back to the warm start of pos
                continue
            g_new = gradient(new)
            s, y = step.ravel(), (g_new - g).ravel()
            if s @ y > 1e-12:
                s_hist.append(s)
                y_hist.append(y)
                s_hist, y_hist = s_hist[-memory:], y_hist[-memory:]
            pos, out, g = pos + step, new, g_new
            scale = min(max_step, 2.0 * scale)
            self.n_steps += 1
            if verbose:
                print(f"  {self.name} step {self.n_steps} E {out['energy']:.6f} fmax {np.abs(g).max():.4f} evals {self.evaluations}", flush=True)
        self.positions = pos
        self.max_force = float(np.abs(g).max())
        self.out = out
        return out
