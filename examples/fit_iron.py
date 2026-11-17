"""Fit the magnetic d-band BOP for iron to structural, magnetic and elastic data.

Needs scipy (installed with the md extra:  uv sync --extra md).
Run with:  uv run python examples/fit_iron.py  [--quick]

The fitted values are printed in the form of ``bopjax.models.IRON_FIT``.
Each residual is (model - target) / tolerance; the targets are listed in
``TARGETS`` and documented in docs/iron.md.  About 20 minutes on 4 cores.
"""

import argparse
import json
import sys
import time

import numpy as np
from _iron import PHASES, Calculator, atomic_volume, iron, mean_abs_moment
from scipy.optimize import least_squares

from bopjax.models import IRON_FIT

# target, tolerance.  bcc FM: experiment at low temperature; energy and
# volume differences between phases: DFT-GGA (PBE), see docs/iron.md
TARGETS = {
    "a bcc FM": (2.855, 0.005),  # Angstrom
    "m bcc FM": (2.2, 0.05),  # Bohr magnetons
    "C11": (243.0, 8.0),  # GPa
    "C12": (138.0, 8.0),
    "C44": (122.0, 8.0),
    "dE hcp NM": (0.065, 0.010),  # eV/atom above bcc FM
    "dE fcc AFM": (0.100, 0.012),
    "dE fcc NM": (0.130, 0.012),
    "dE bcc NM": (0.480, 0.030),
    "V/V0 hcp NM": (0.90, 0.02),  # atomic volume relative to bcc FM
    "V/V0 fcc NM": (0.91, 0.02),
}

# fitted parameters: name, lower and upper bound
FREE = {
    "dds": (-2.0, -0.4),
    "ddp": (0.2, 1.5),
    "ddd": (-0.5, 0.1),
    "decay": (1.5, 8.0),
    "stoner": (0.5, 1.4),
    "n_d": (6.5, 7.5),
    "pair": (0.05, 5.0),
    "pair_decay": (3.0, 16.0),
    "embed": (0.0, 5.0),
    "density_decay": (1.0, 12.0),
}
FIT_PHASES = ("bcc FM", "bcc NM", "fcc AFM", "fcc NM", "hcp NM")


class Model:
    def __init__(self):
        self.calcs = {name: Calculator(PHASES[name]) for name in FIT_PHASES}
        self.n_eval = 0
        self.best = (np.inf, None)

    def properties(self, values):
        p = iron(**values)
        out = {}
        eq = {name: c.equilibrium(p) for name, c in self.calcs.items()}
        a0, e0, aux0 = eq["bcc FM"]
        v0 = atomic_volume(PHASES["bcc FM"], a0)
        out["a bcc FM"] = a0
        out["m bcc FM"] = mean_abs_moment(aux0)
        out["C11"], out["C12"], out["C44"] = self.calcs["bcc FM"].cubic_elastic(p, a0)
        for name, (a, e, aux) in eq.items():
            out[f"dE {name}"] = e - e0
            out[f"V/V0 {name}"] = atomic_volume(PHASES[name], a) / v0
            out[f"m {name}"] = mean_abs_moment(aux)
            # an equilibrium on the edge of the search range is no equilibrium
            rel = a / PHASES[name].a_guess - 1.0
            out[f"edge {name}"] = max(0.0, abs(rel) - 0.055)
        return out

    def residuals(self, x):
        values = dict(zip(FREE, x))
        try:
            out = self.properties(values)
        except Exception as err:  # noqa: BLE001 (a failed SCF is a bad point)
            print("failed:", err, file=sys.stderr)
            return np.full(len(TARGETS) + len(FIT_PHASES), 1e3)
        r = [(out[k] - t) / tol for k, (t, tol) in TARGETS.items()]
        r += [100.0 * out[f"edge {name}"] for name in FIT_PHASES]
        r = np.nan_to_num(np.array(r), nan=1e3, posinf=1e3, neginf=-1e3)
        self.n_eval += 1
        cost = float(r @ r)
        if cost < self.best[0]:
            self.best = (cost, values)
            print(f"[{self.n_eval:4d}] cost {cost:9.3f}  "
                  + " ".join(f"{k}={v:.4f}" for k, v in values.items()), flush=True)
        return r


def report(model, values):
    out = model.properties(values)
    print(f"{'property':14s} {'model':>9s} {'target':>9s}")
    for k, (t, _) in TARGETS.items():
        print(f"{k:14s} {out[k]:9.4f} {t:9.4f}")
    for name in FIT_PHASES:
        print(f"m {name:12s} {out[f'm {name}']:9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="evaluate the start only")
    ap.add_argument("--max-nfev", type=int, default=15)
    args = ap.parse_args()

    model = Model()
    start = dict(IRON_FIT)
    x0 = np.array([start[k] for k in FREE])
    lo, hi = np.array([b[0] for b in FREE.values()]), np.array([b[1] for b in FREE.values()])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    t = time.time()
    model.residuals(x0)
    print(f"one evaluation: {time.time() - t:.1f} s")
    if not args.quick:
        least_squares(model.residuals, x0, bounds=(lo, hi), x_scale="jac", diff_step=2e-3,
                      max_nfev=args.max_nfev)
    values = dict(start, **model.best[1])
    report(model, values)
    print("IRON_FIT =", json.dumps({k: round(float(v), 5) for k, v in values.items()},
                                   indent=4))


if __name__ == "__main__":
    main()
