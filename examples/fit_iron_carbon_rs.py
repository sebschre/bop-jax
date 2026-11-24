"""Fit the Fe-C cross terms of the iron-carbon BOP on bopfox-rs (Rust).

The same model, targets and conventions as ``examples/fit_iron_carbon.py``,
evaluated with the Rust port (``examples/_iron_carbon_rs.py``): potential
self-consistency, exact implicit forces and stress, and **every atom of the
Fe16C cells relaxed** in each evaluation (the JAX fit could only afford the
breathing modes of two Fe shells).  One evaluation takes about two minutes on
four cores instead of 10-25 minutes, in one process of a few hundred MB.

Run with:  uv run python examples/fit_iron_carbon_rs.py [--quick] [--max-nfev N]
           [--start checkpoint.json] [--checkpoint best.json] [--free a,b,c]

Each residual is (model - target) / tolerance.  The fitted values are printed
in the form of ``bopjax.models.IRON_CARBON_FIT``.
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy.optimize import least_squares

from _iron_carbon_rs import (RsCell, bcc_fe, bcc_supercell, cementite, diamond,
                             fe_with_carbon)
from bopjax.models import FE, IRON_CARBON_FIT, iron_carbon
from fit_iron_carbon import TARGETS

# fitted parameters: name, lower and upper bound
BOUNDS = {
    "e_c": (-6.0, 6.0),
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
START_EXTRA = {"rho_gauss_fe": 0.0, "rho_gauss_c": 0.0, "rho_gauss_r": 2.0, "rho_gauss_w": 0.3}


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
        # displacement, so that an unstable site (carbon collapsing onto one
        # Fe neighbour, as with the JAX fit's parameters) is found
        rng = np.random.default_rng(11)
        self.ideal = {c.name: c.positions.copy() for c in (self.oct, self.tet, self.cem)}
        self.start = {k: v + (0.02 * rng.standard_normal(v.shape) if k != "cem" else 0.0)
                      for k, v in self.ideal.items()}
        self.start["tet"][16] = self.ideal["tet"][16]  # the saddle point: C on its site
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

    def properties(self, values, relax_cementite=False, reset=False):
        """Fit quantities: Fe16C cells with every atom relaxed at fixed cell
        (the tetrahedral C held on its site, the saddle point); cementite at
        the experimental positions unless ``relax_cementite``."""
        p = iron_carbon(**values)
        t0 = time.time()
        for c in (self.oct, self.tet, self.cem):
            c.set_params(p)
            if reset or min_fe_c(c) < MIN_FE_C:  # start afresh after a collapse
                c.positions = self.start[c.name].copy()
                c._x = None
        o = self.oct.relax(ftol=self.ftol)
        t = self.tet.relax(fixed=(16,), ftol=self.ftol)
        if relax_cementite:
            c = self.cem.relax(ftol=self.ftol)
        else:
            self.cem.positions = self.ideal["cem"].copy()
            c = self.cem.solve()
        out = {}
        out["E_sol oct"] = o["energy"] - 16 * self.e_fe - self.e_c
        out["E tet - oct"] = t["energy"] - o["energy"]
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
        values = dict(zip(names, x))
        try:
            out = self.properties(dict(self.fixed, **values))
        except Exception as err:  # noqa: BLE001 (a failed evaluation is a bad point)
            print("failed:", err, file=sys.stderr, flush=True)
            for c in (self.oct, self.tet, self.cem):  # start the next one afresh
                c.positions = self.ideal[c.name].copy()
                c._x = None
            return np.full(len(TARGETS) + 1, 1e2)
        r = np.array([(out[k] - t) / tol for k, (t, tol) in TARGETS.items()]
                     + [max(0.0, MIN_FE_C - out["min Fe-C"]) / 0.05])
        r = np.nan_to_num(r, nan=1e2, posinf=1e2, neginf=-1e2)
        self.n_eval += 1
        cost = float(r @ r)
        if self.verbose or cost < self.best[0]:
            mark = "*" if cost < self.best[0] else " "
            print(f"[{self.n_eval:4d}]{mark}cost {cost:8.3f} ({out['time']:.0f} s)  "
                  + " ".join(f"{k}={v:.4g}" for k, v in values.items()), flush=True)
            print("        " + " ".join(f"{k}={out[k]:.3f}" for k in TARGETS), flush=True)
        if cost < self.best[0]:
            self.best = (cost, dict(self.fixed, **values), out)
            if self.checkpoint:
                with open(self.checkpoint, "w") as fh:
                    json.dump({"cost": cost,
                               "values": {k: float(v) for k, v in self.best[1].items()},
                               "properties": {k: float(v) for k, v in out.items()}}, fh, indent=1)
        return r


def report(model, values, relax_cementite=True):
    out = model.properties(values, relax_cementite=False, reset=True)
    rows = [("Fe16C relaxed", out)]
    if relax_cementite:
        rows.append(("+ Fe3C relaxed", model.properties(values, relax_cementite=True)))
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
    args = ap.parse_args()

    start = dict(IRON_CARBON_FIT)
    for k, v in START_EXTRA.items():
        start.setdefault(k, v)
    if args.start:
        with open(args.start) as fh:
            start.update(json.load(fh)["values"])
    if args.set:
        start.update(json.loads(args.set))
    names = args.free.split(",") if args.free else list(BOUNDS)
    model = Model(checkpoint=args.checkpoint, verbose=args.verbose)
    model.fixed = {k: v for k, v in start.items() if k not in names}
    x0 = np.array([start[k] for k in names])
    lo = np.array([BOUNDS[k][0] for k in names])
    hi = np.array([BOUNDS[k][1] for k in names])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    t = time.time()
    model.residuals(x0, names)
    print(f"one evaluation: {time.time() - t:.1f} s", flush=True)
    if not args.quick:
        least_squares(model.residuals, x0, args=(names,), bounds=(lo, hi), x_scale="jac",
                      diff_step=1e-2, max_nfev=args.max_nfev)
    values = model.best[1]
    report(model, values)
    print("IRON_CARBON_FIT =", json.dumps({k: round(float(v), 5) for k, v in values.items()},
                                          indent=4))


if __name__ == "__main__":
    main()
