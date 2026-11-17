"""Fit the magnetic d-band BOP for iron to structural, magnetic and elastic data.

Needs scipy (installed with the md extra:  uv sync --extra md).
Run with:  uv run python examples/fit_iron.py  [--quick] [--basis d|sd|pd|spd]

The fitted values are printed in the form of ``bopjax.models.IRON_FIT``
(``IRON_SD_FIT`` for --basis sd).
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

from bopjax.models import IRON_FIT, IRON_SD_FIT, IRON_SP_DEFAULTS

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
# with an s and/or p shell: the d electron count is no longer a parameter
# (n_el = 8 valence electrons share the Fermi level), the onsite levels and
# the bond integrals with an s or p shell are fitted instead
FREE_SP = {
    "e_s": (-2.0, 6.0),
    "e_p": (2.0, 14.0),
    "decay_sp": (0.5, 6.0),
    "sss": (-2.5, -0.2),
    "sps": (0.2, 3.0),
    "pps": (0.2, 4.0),
    "ppp": (-1.5, 0.0),
    "sds": (-2.0, -0.1),
    "pds": (-2.0, -0.1),
    "pdp": (0.0, 1.0),
}
BASES = {"d": ("d",), "sd": ("s", "d"), "pd": ("p", "d"), "spd": ("s", "p", "d")}


def free_parameters(basis):
    if basis == ("d",):
        return dict(FREE)
    out = {k: b for k, b in FREE.items() if k != "n_d"}
    for k, b in FREE_SP.items():
        shells = k[2] if k.startswith("e_") else k[:2]  # e_s -> s, pds -> pd
        if k == "decay_sp" or all(sh in basis for sh in shells):
            out[k] = b
    return out


FIT_PHASES = ("bcc FM", "bcc NM", "fcc AFM", "fcc NM", "hcp NM")


class Model:
    def __init__(self, basis=("d",)):
        self.basis = basis
        self.free = free_parameters(basis)
        self.calcs = {name: Calculator(PHASES[name], basis=basis) for name in FIT_PHASES}
        self.n_eval = 0
        self.best = (np.inf, None)

    def properties(self, values):
        p = iron(basis=self.basis, **values)
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
        values = dict(zip(self.free, x))
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
    cost = sum(((out[k] - t) / tol) ** 2 for k, (t, tol) in TARGETS.items())
    print(f"cost {cost:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="evaluate the start only")
    ap.add_argument("--max-nfev", type=int, default=15)
    ap.add_argument("--basis", choices=BASES, default="d")
    args = ap.parse_args()

    basis = BASES[args.basis]
    model = Model(basis)
    free = model.free
    start = dict(IRON_FIT)
    if basis != ("d",):
        start.update(IRON_SP_DEFAULTS)
        if basis == ("s", "d"):
            start.update(IRON_SD_FIT)
    x0 = np.array([start[k] for k in free])
    lo, hi = np.array([b[0] for b in free.values()]), np.array([b[1] for b in free.values()])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    t = time.time()
    model.residuals(x0)
    print(f"one evaluation: {time.time() - t:.1f} s")
    if not args.quick:
        least_squares(model.residuals, x0, bounds=(lo, hi), x_scale="jac", diff_step=2e-3,
                      max_nfev=args.max_nfev)
    values = dict(start, **model.best[1])
    report(model, values)
    name = "IRON_FIT" if basis == ("d",) else f"IRON_{args.basis.upper()}_FIT"
    shown = values if basis == ("d",) else {k: values[k] for k in free}
    print(name, "=", json.dumps({k: round(float(v), 5) for k, v in shown.items()}, indent=4))


if __name__ == "__main__":
    main()
