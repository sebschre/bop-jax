"""Fit the sp-valent BOP for carbon to structural and elastic data.

Needs scipy (installed with the md extra:  uv sync --extra md).
Run with:  uv run python examples/fit_carbon.py  [--quick] [--max-nfev N]

The fitted values are printed in the form of ``bopjax.models.CARBON_FIT``.
Each residual is (model - target) / tolerance; the targets are listed in
``TARGETS`` and documented in docs/carbon.md.
"""

import argparse
import json
import sys
import time

import numpy as np
from _carbon import PHASES, Calculator, bond_length, carbon
from scipy.optimize import least_squares

from bopjax.models import CARBON_FIT

# target, tolerance.  Diamond and graphite: experiment; the other phases:
# rounded first-principles values, see docs/carbon.md
TARGETS = {
    "a diamond": (3.567, 0.005),  # Angstrom
    "C11": (1079.0, 30.0),  # GPa
    "C12": (124.0, 15.0),
    "C44": (578.0, 20.0),
    "d graphene": (1.42, 0.01),  # bond length, Angstrom
    "dE graphene": (0.03, 0.04),  # eV/atom above diamond
    "d chain": (1.28, 0.03),
    "dE chain": (1.2, 0.2),
    "dE sc": (2.6, 0.4),
    "dE bcc": (3.5, 0.5),
    "dE fcc": (4.0, 0.5),
}

# fitted parameters: name, lower and upper bound
FREE = {
    "e_s": (-12.0, -2.0),
    "sss": (-9.0, -1.0),
    "sps": (1.0, 9.0),
    "pps": (1.0, 10.0),
    "ppp": (-5.0, -0.2),
    "decay": (0.5, 6.0),
    "pair": (0.2, 30.0),
    "pair_decay": (1.0, 14.0),
    "embed": (0.0, 20.0),
    "density_decay": (0.0, 12.0),
}

FIT_PHASES = ("diamond", "graphene", "chain", "sc", "bcc", "fcc")
CHECK_PHASES = ("lonsdaleite",)


class Model:
    def __init__(self, phases=FIT_PHASES):
        self.calcs = {name: Calculator(PHASES[name]) for name in phases}
        self.n_eval = 0
        self.best = (np.inf, None)

    def properties(self, values):
        p = carbon(**values)
        out = {}
        eq = {name: c.equilibrium(p) for name, c in self.calcs.items()}
        a0, e0, _ = eq["diamond"]
        out["a diamond"] = a0
        out["C11"], out["C12"], out["C44"] = self.calcs["diamond"].cubic_elastic(p, a0)
        for name, (a, e, _) in eq.items():
            out[f"dE {name}"] = e - e0
            out[f"d {name}"] = bond_length(PHASES[name], a)
            # an equilibrium on the edge of the search range is no equilibrium
            rel = a / PHASES[name].a_guess - 1.0
            out[f"edge {name}"] = max(0.0, abs(rel) - 0.095)
        return out

    def residuals(self, x):
        values = dict(zip(FREE, x))
        try:
            out = self.properties(values)
        except Exception as err:  # noqa: BLE001 (a failed evaluation is a bad point)
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


def report(values):
    model = Model(FIT_PHASES + CHECK_PHASES)
    out = model.properties(values)
    print(f"{'property':14s} {'model':>9s} {'target':>9s}")
    for k, (t, _) in TARGETS.items():
        print(f"{k:14s} {out[k]:9.4f} {t:9.4f}")
    for name in FIT_PHASES[1:] + CHECK_PHASES:
        print(f"{name:12s} dE {out[f'dE {name}']:8.4f} eV  bond {out[f'd {name}']:.4f} A")
    cost = sum(((out[k] - t) / tol) ** 2 for k, (t, tol) in TARGETS.items())
    print(f"cost {cost:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="evaluate the start only")
    ap.add_argument("--max-nfev", type=int, default=30)
    args = ap.parse_args()

    model = Model()
    x0 = np.array([CARBON_FIT[k] for k in FREE])
    lo, hi = np.array([b[0] for b in FREE.values()]), np.array([b[1] for b in FREE.values()])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    t = time.time()
    model.residuals(x0)
    print(f"one evaluation: {time.time() - t:.1f} s")
    if not args.quick:
        least_squares(model.residuals, x0, bounds=(lo, hi), x_scale="jac", diff_step=2e-3,
                      max_nfev=args.max_nfev)
    values = dict(CARBON_FIT, **model.best[1])
    report(values)
    shown = {k: values[k] for k in FREE}
    print("CARBON_FIT =", json.dumps({k: round(float(v), 5) for k, v in shown.items()},
                                     indent=4))


if __name__ == "__main__":
    main()
