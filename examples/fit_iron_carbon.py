"""Fit the Fe-C cross terms of the iron-carbon BOP (bopjax.models.iron_carbon)
on bopfox-rs, the Rust port with analytic forces (examples/_iron_carbon_rs.py).

Only the Fe-C parameters are fitted; Fe-Fe and C-C stay those of
``iron_egorov`` and ``CARBON_FIT``, as Egorov et al. fitted only the cross
terms of their Fe-Co BOP on fixed elemental models.  Each evaluation:

- relaxes every atom of the Fe16C cells and of cementite (at the experimental
  cell) on exact forces, with the potential self-consistency;
- puts every cell on its lowest-energy magnetic state at each iterate (iron
  next to carbon has several self-consistent states);
- takes the migration saddle as the lowest energy in the plane half way
  between two octahedral sites (carbon held along the path only), which
  agrees with the NEB of examples/neb_iron_carbon_rs.py.

One evaluation takes 0.5-1.5 minutes on four cores, warm-started.  See
docs/iron-carbon.md.

Run with:  uv run python examples/fit_iron_carbon.py [--quick] [--max-nfev N]
           [--start checkpoint.json] [--checkpoint best.json] [--free a,b,c]

Each residual is (model - target) / tolerance.  The fitted values are printed
in the form of ``bopjax.models.IRON_CARBON_FIT``.
"""

import argparse
import json
import sys
import time

import numpy as np
from _iron_carbon_rs import (
    A_FE,
    RsCell,
    bcc_fe,
    bcc_supercell,
    cementite,
    diamond,
    fe_with_carbon,
    site_symmetry,
)
from scipy.optimize import least_squares

from bopjax.models import FE, IRON_CARBON_FIT, iron_carbon

# target, tolerance
TARGETS = {
    # solution energy of C in an octahedral site of FM bcc Fe, relative to
    # graphite (Jiang and Carter, PRB 67, 214103 (2003): 0.74 eV); the model's
    # carbon reference is diamond, about 0.02 eV above graphite
    "E_sol oct": (0.72, 0.15),
    # migration barrier between octahedral sites: 0.86 eV (DFT, Jiang and
    # Carter), 0.87 eV (experiment)
    "E saddle - oct": (0.86, 0.10),
    # elastic dipole tensor of octahedral C, along and across its Fe-C-Fe axis
    # (DFT, Clouet et al.; values from memory, loose tolerance)
    "P axial": (8.0, 1.5),
    "P transverse": (3.4, 1.5),
    # cementite Fe3C at the experimental lattice: formation energy per atom
    # relative to FM bcc Fe and diamond, pressure, mean Fe moment
    "dH Fe3C": (0.03, 0.04),
    "p Fe3C": (0.0, 5.0),
    "m Fe3C": (1.95, 0.15),
    # electrons carbon gains in the octahedral site: DFT (Bader) gives C in
    # iron a gain of roughly half to one electron; this tight-binding charge
    # is not a Bader charge, so the target only keeps the transfer moderate
    "charge C oct": (0.3, 0.4),
}

# fitted parameters: name, lower and upper bound
BOUNDS = {
    # below about 1 eV (the C levels near the Fe d band) relaxed cementite
    # loses its Fe moments: 0.6 muB at e_c = 0.5 eV, 2.0 muB at 1.5 eV, a jump
    # the least-squares steps walk into and cannot climb out of
    "e_c": (1.3, 6.0),
    "sds": (-4.0, 0.0),
    "pds": (-5.0, 0.0),
    "pdp": (0.0, 3.0),
    "decay": (1.0, 7.0),
    "decay_pi": (1.0, 8.0),
    "pair": (0.05, 20.0),
    "pair_decay": (2.0, 14.0),
    "rho_fe": (0.0, 5.0),
    "rho_c": (0.0, 5.0),
    "hubbard_c": (2.0, 12.0),
    "core": (0.0, 3000.0),
    "core_r": (1.5, 1.95),
    # Gaussian Fe-C embedding densities, centred near the transverse
    # neighbours of the octahedral site: binding there without pushing
    "rho_gauss_fe": (0.0, 3.0),
    "rho_gauss_c": (0.0, 3.0),
    "rho_gauss_r": (1.7, 2.4),
    "rho_gauss_w": (0.15, 0.6),
}


# a relaxed Fe-C distance below this (A) counts as a collapse: penalised in
# the fit, and the next relaxation starts afresh
MIN_FE_C = 1.5


def min_fe_c(c):
    """Shortest Fe-C distance in a cell (minimum image, orthorhombic cells)."""
    box = np.diag(c.cell)
    fe, carbon = c.positions[c.species == FE], c.positions[c.species != FE]
    d = fe[:, None] - carbon[None]
    d -= np.round(d / box) * box
    return float(np.linalg.norm(d, axis=-1).min())


class Model:
    def __init__(self, checkpoint=None, ftol=1e-3, verbose=False):
        self.checkpoint = checkpoint
        self.ftol = ftol
        self.verbose = verbose
        self.oct = RsCell("oct", *fe_with_carbon("octahedral"))
        self.tet = RsCell("tet", *fe_with_carbon("tetrahedral"))
        self.cem = RsCell("cem", *cementite())
        # relaxations start from the ideal sites plus a small random
        # displacement, so that an unstable site is found (carbon collapsing
        # onto one Fe neighbour, as with an earlier fit's parameters)
        rng = np.random.default_rng(11)
        self.ideal = {c.name: c.positions.copy() for c in (self.oct, self.tet, self.cem)}
        tet = fe_with_carbon("tetrahedral")
        self.tet_ops = site_symmetry(tet[0], tet[1], np.asarray(tet[2]), 16)
        self.start = {k: v + (0.02 * rng.standard_normal(v.shape) if k == "oct" else 0.0)
                      for k, v in self.ideal.items()}
        # the migration saddle: carbon held at the tetrahedral site's y (half
        # way along the path from the octahedral site at y = a/2 to the one at
        # y = 0), free in that plane, the Fe free with their centre held.  The
        # tetrahedral point is a higher-order saddle (``tet_symmetric`` holds
        # carbon there instead); from it the relaxation can stall, so carbon
        # starts 0.06 a off it in the plane (the saddle is at about 0.12 a;
        # starts at 0.06 and 0.10 a reach the NEB barrier to 1e-4 eV)
        rng = np.random.default_rng(12)
        self.start["tet"] = self.ideal["tet"] + 0.02 * rng.standard_normal((17, 3))
        self.start["tet"][16] = self.ideal["tet"][16] + A_FE * np.array([0.06, 0.0, 0.06])
        p = iron_carbon()
        fe = RsCell("bcc Fe", *bcc_fe())
        fe.set_params(p)
        self.e_fe = fe.solve()["energy"] / 2
        dia = RsCell("diamond", *diamond(), magnetic=False)
        dia.set_params(p)
        self.e_c = dia.solve()["energy"] / 2
        host = RsCell("Fe16", *bcc_supercell())
        host.set_params(p)
        host.solve()
        self.host_stress = host.stress()  # zero up to the precision of A_FE
        self.species = np.asarray(cementite()[2])
        self.n_eval = 0
        self.best = (np.inf, None, None)
        self.ref = None  # relaxed state at the current iterate (see jacobian)
        self.last = None
        self.last_x = None
        self.ground_next = True  # the first evaluation finds the ground states
        # relax cementite (else: experimental positions, which can hide a
        # collapse of its Fe moments)
        self.relax_cementite = True
        # the symmetric tetrahedral site instead of the saddle in the plane
        # (C held, the Fe centre held, every step symmetrised over D2d)
        self.tet_symmetric = False

    def snapshot(self):
        return {c.name: (c.positions.copy(), None if c._x is None else c._x.copy())
                for c in (self.oct, self.tet, self.cem)}

    def restore(self, snap):
        for c in (self.oct, self.tet, self.cem):
            c.positions, x = snap[c.name][0].copy(), snap[c.name][1]
            c._x = None if x is None else x.copy()

    def properties(self, values, relax_cementite=True, reset=False, ground=False):
        """Fit quantities: Fe16C cells with every atom relaxed at fixed cell
        (the migration saddle in the plane between two octahedral sites);
        cementite relaxed, or at the experimental positions."""
        p = iron_carbon(**values)
        t0 = time.time()
        for c in (self.oct, self.tet, self.cem):
            c.set_params(p)
            if reset or min_fe_c(c) < MIN_FE_C:  # start afresh after a collapse
                c.positions = self.start[c.name].copy()
                c._x = None
        # with ``ground`` every cell is put on its lowest-energy magnetic state
        # (several cold starts; done at each iterate of the fit, the Jacobian
        # columns and trial steps follow that state by warm starts)
        relax = (lambda c_, **kw: c_.relax_ground(**kw)) if ground else (
            lambda c_, **kw: c_.relax(**kw))
        o = relax(self.oct, ftol=self.ftol)
        if self.tet_symmetric:
            t = relax(self.tet, fixed=(16,), ftol=self.ftol, fix_centre=True,
                      symmetry=self.tet_ops)
        else:  # carbon held only along the path, free in the plane of the saddle
            t = relax(self.tet, fixed_components=((16, 1),), ftol=self.ftol, fix_centre=True)
        if relax_cementite:
            c = relax(self.cem, ftol=self.ftol)
        else:
            self.cem.positions = self.ideal["cem"].copy()
            c = self.cem.ground_state() if ground else self.cem.solve()
        out = {}
        out["E_sol oct"] = o["energy"] - 16 * self.e_fe - self.e_c
        out["E saddle - oct"] = t["energy"] - o["energy"]
        dp = -(self.oct.stress(o) - self.host_stress) * self.oct.volume() / 160.21766
        out["P axial"] = dp[2, 2]
        out["P transverse"] = 0.5 * (dp[0, 0] + dp[1, 1])
        pos, box = self.oct.positions, np.diag(self.oct.cell)
        d = pos[:16] - pos[16]
        d -= np.round(d / box) * box
        r = np.linalg.norm(d, axis=1)
        nn = np.argsort(r)[:2]
        out["r axial"] = float(r[nn].mean())
        out["r transverse"] = float(np.sort(r)[2:6].mean())
        out["m Fe nn oct"] = float(o["moment"][nn].mean())
        out["m C oct"] = float(o["moment"][16])
        out["dH Fe3C"] = (c["energy"] - 12 * self.e_fe - 4 * self.e_c) / 16
        out["p Fe3C"] = -float(np.trace(self.cem.stress(c))) / 3
        out["m Fe3C"] = float(np.mean(c["moment"][self.species == FE]))
        out["charge C oct"] = float(o["charge"][16])
        out["charge C Fe3C"] = float(np.mean(c["charge"][12:]))
        out["fmax"] = max(self.oct.max_force, self.tet.max_force)
        out["min Fe-C"] = min(min_fe_c(self.oct), min_fe_c(self.tet))
        out["time"] = time.time() - t0
        return out

    def residuals(self, x, names):
        """Residuals at x.  Every evaluation starts from the relaxed state of
        the current iterate (``self.ref``), so that the residuals are a
        function of x alone: relaxations warm-started from wherever the
        previous evaluation ended reach slightly different minima, and a
        finite-difference Jacobian of those is noise."""
        values = dict(zip(names, x))
        if self.ref is not None:
            self.restore(self.ref)
        try:
            out = self.properties(dict(self.fixed, **values), ground=self.ground_next,
                                  relax_cementite=self.relax_cementite)
        except Exception as err:  # noqa: BLE001 (a failed evaluation is a bad point)
            print("failed:", err, file=sys.stderr, flush=True)
            for c in (self.oct, self.tet, self.cem):  # start the next one afresh
                c.positions = self.ideal[c.name].copy()
                c._x = None
            return np.full(len(TARGETS) + 1, 1e2)
        self.last = out
        r = np.array([(out[k] - t) / tol for k, (t, tol) in TARGETS.items()]
                     + [max(0.0, MIN_FE_C - out["min Fe-C"]) / 0.05])
        r = np.nan_to_num(r, nan=1e2, posinf=1e2, neginf=-1e2)
        self.n_eval += 1
        cost = float(r @ r)
        self.ground_next = False
        self.last_x, self.last_r = np.array(x, float), r
        self.last_state = self.snapshot()
        if self.verbose or cost < self.best[0]:
            mark = "*" if cost < self.best[0] else " "
            print(f"[{self.n_eval:4d}]{mark}cost {cost:8.3f} ({out['time']:.0f} s)  "
                  + " ".join(f"{k}={v:.4g}" for k, v in values.items()), flush=True)
            print("        " + " ".join(f"{k}={out[k]:.3f}" for k in TARGETS)
                  + f" | r_ax={out['r axial']:.3f} r_tr={out['r transverse']:.3f}"
                  f" minFeC={out['min Fe-C']:.3f}", flush=True)
        if cost < self.best[0]:
            self.best = (cost, dict(self.fixed, **values), out)
            if self.checkpoint:
                with open(self.checkpoint, "w") as fh:
                    json.dump({"cost": cost,
                               "values": {k: float(v) for k, v in self.best[1].items()},
                               "properties": {k: float(v) for k, v in out.items()}}, fh, indent=1)
        return r


def jacobian(model, lo, hi, step=1e-2):
    """Forward-difference Jacobian at the iterate x: every column relaxes
    from the relaxed state at x (``least_squares`` calls this right after
    evaluating the residuals at x)."""

    def jac(x, names):
        if model.last_x is None or not np.array_equal(model.last_x, x):
            model.residuals(x, names)
        model.ref = model.last_state
        model.ground_next = True  # the iterate on its lowest magnetic states
        r0 = model.residuals(x, names)  # from the new reference: consistent columns
        model.ref = model.last_state
        cols = []
        for j in range(len(x)):
            h = step * max(1.0, abs(x[j]))
            if x[j] + h > hi[j]:
                h = -h
            xj = np.array(x, float)
            xj[j] += h
            cols.append((model.residuals(xj, names) - r0) / h)
        model.restore(model.ref)  # leave the state of x itself
        model.last_x, model.last_r = np.array(x, float), r0
        model.last_state = model.ref
        return np.array(cols).T

    return jac


def report(model, values):
    """The fit quantities from scratch, on the lowest magnetic states, with
    cementite at the experimental positions and relaxed."""
    rows = [("Fe3C experimental", model.properties(values, relax_cementite=False, reset=True,
                                                   ground=True)),
            ("Fe3C relaxed", model.properties(values, relax_cementite=True, ground=True))]
    print(f"{'property':14s} " + " ".join(f"{n:>16s}" for n, _ in rows) + f" {'target':>9s}")
    for k, (t, _) in TARGETS.items():
        print(f"{k:14s} " + " ".join(f"{o[k]:16.4f}" for _, o in rows) + f" {t:9.4f}")
    for k in ("r axial", "r transverse", "min Fe-C", "m Fe nn oct", "m C oct", "charge C Fe3C",
              "fmax"):
        print(f"{k:14s} " + " ".join(f"{o[k]:16.4f}" for _, o in rows))
    for name, o in rows:
        cost = sum(((o[k] - t) / tol) ** 2 for k, (t, tol) in TARGETS.items())
        print(f"cost ({name}) {cost:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="evaluate the start only")
    ap.add_argument("--max-nfev", type=int, default=200)
    ap.add_argument("--checkpoint", help="JSON file the best point is written to")
    ap.add_argument("--start", help="JSON checkpoint to start from")
    ap.add_argument("--free", help="comma-separated parameters to fit (default: all of BOUNDS)")
    ap.add_argument("--set", help="JSON dict of fixed values overriding the start")
    ap.add_argument("--verbose", action="store_true", help="print every evaluation")
    ap.add_argument("--tet-symmetric", action="store_true",
                    help="the barrier from the symmetric tetrahedral site instead of the saddle"
                         " (a higher-order saddle for this model; for comparison)")
    ap.add_argument("--cementite-fixed", action="store_true",
                    help="cementite at the experimental positions (it can hide a collapse of"
                         " the Fe moments that relaxation shows)")
    args = ap.parse_args()

    start = dict(IRON_CARBON_FIT)
    if args.start:
        with open(args.start) as fh:
            start.update(json.load(fh)["values"])
    if args.set:
        start.update(json.loads(args.set))
    names = args.free.split(",") if args.free else list(BOUNDS)
    model = Model(checkpoint=args.checkpoint, verbose=args.verbose)
    model.fixed = {k: v for k, v in start.items() if k not in names}
    model.relax_cementite = not args.cementite_fixed
    if args.tet_symmetric:
        model.tet_symmetric = True
        model.start["tet"] = model.ideal["tet"].copy()
    x0 = np.array([start[k] for k in names])
    lo = np.array([BOUNDS[k][0] for k in names])
    hi = np.array([BOUNDS[k][1] for k in names])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    t = time.time()
    model.residuals(x0, names)
    print(f"one evaluation: {time.time() - t:.1f} s", flush=True)
    if not args.quick:
        model.last_x = None
        least_squares(model.residuals, x0, jac=jacobian(model, lo, hi), args=(names,),
                      bounds=(lo, hi), x_scale="jac", max_nfev=args.max_nfev)
    values = model.best[1]
    report(model, values)
    print("IRON_CARBON_FIT =", json.dumps({k: round(float(v), 5) for k, v in values.items()},
                                          indent=4))


if __name__ == "__main__":
    main()
