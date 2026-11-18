"""Fit the Fe-C cross terms of the iron-carbon BOP (d iron, sp carbon).

Only the Fe-C parameters are fitted; Fe-Fe and C-C stay those of
``IRON_FIT`` and ``CARBON_FIT``.  Needs scipy (uv sync --extra md).
Run with:  uv run python examples/fit_iron_carbon.py  [--quick] [--max-nfev N]

Each residual is (model - target) / tolerance; targets in ``TARGETS``, see
docs/iron-carbon.md.  The fitted values are printed in the form of
``bopjax.models.IRON_CARBON_FIT``.
"""

import argparse
import json
import sys
import time

import numpy as np
from _iron_carbon import (Calculator, Cell, bcc_fe, bcc_supercell, cementite, diamond,
                          fe_moments, fe_with_carbon, iron_carbon)
from scipy.optimize import least_squares

from bopjax.models import IRON_CARBON_FIT

# target, tolerance
TARGETS = {
    # solution energy of C in an octahedral site of FM bcc Fe, relative to
    # graphite (Jiang and Carter, PRB 67, 214103 (2003): 0.74 eV); the model's
    # carbon reference is diamond, about 0.02 eV above graphite
    "E_sol oct": (0.72, 0.15),
    # tetrahedral site, the saddle point of octahedral-octahedral migration:
    # 0.86 eV (DFT, Jiang and Carter), 0.87 eV (experiment)
    "E tet - oct": (0.86, 0.10),
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
FREE = {
    "e_c": (-6.0, 6.0),
    "sds": (-4.0, 0.0),
    "pds": (-5.0, 0.0),
    "pdp": (0.0, 3.0),
    "decay": (1.0, 7.0),
    "pair": (0.05, 20.0),
    "pair_decay": (2.0, 14.0),
    "rho_fe": (0.0, 5.0),
    "rho_c": (0.0, 5.0),
    "hubbard_c": (2.0, 12.0),
}


class Model:
    def __init__(self):
        self.oct = Calculator(Cell("oct", *fe_with_carbon("octahedral")))
        self.tet = Calculator(Cell("tet", *fe_with_carbon("tetrahedral")))
        self.cem = Calculator(Cell("Fe3C", *cementite()))
        p = iron_carbon()
        self.e_fe = Calculator(Cell("bcc Fe", *bcc_fe())).solve(p)[0] / 2
        self.e_c = Calculator(Cell("diamond", *diamond(), magnetic=False)).solve(p)[0] / 2
        host = Calculator(Cell("Fe16", *bcc_supercell()))
        host.solve(p)
        self.host_stress = host.stress(p)  # zero up to the precision of A_FE
        self.n_eval = 0
        self.best = (np.inf, None)

    def properties(self, values, relax_cementite=True, full=False):
        """Fit quantities.  During the fit (``full`` False) only the
        breathing modes of the first two Fe shells around C are relaxed;
        the report relaxes every atom."""
        p = iron_carbon(**values)
        out = {}
        if full:
            e_oct, aux_oct = self.oct.relax(p)
            e_tet, _ = self.tet.relax(p, fixed=[16])  # the saddle: C stays on the site
        else:
            e_oct, aux_oct = self.oct.relax_shells(p)
            e_tet, _ = self.tet.relax_shells(p)
            out["u oct"], out["u tet"] = self.oct.amplitudes, self.tet.amplitudes
        out["E_sol oct"] = e_oct - 16 * self.e_fe - self.e_c
        out["E tet - oct"] = e_tet - e_oct
        dp = -(self.oct.stress(p) - self.host_stress) * self.oct.volume() / 160.21766
        out["P axial"] = dp[2, 2]
        out["P transverse"] = 0.5 * (dp[0, 0] + dp[1, 1])
        # moment of the two Fe neighbours of C at a/2, along its axis
        d = np.asarray(self.oct.positions[:16] - self.oct.positions[16])
        d -= np.round(d / np.diag(np.asarray(self.oct.cell))) * np.diag(np.asarray(self.oct.cell))
        nn = np.argsort(np.linalg.norm(d, axis=1))[:2]
        out["m Fe nn oct"] = float(np.asarray(aux_oct["moment"])[nn].mean())
        e_cem, aux_cem = (self.cem.relax(p) if relax_cementite else self.cem.solve(p))
        out["dH Fe3C"] = (e_cem - 12 * self.e_fe - 4 * self.e_c) / 16
        out["p Fe3C"] = -float(np.trace(self.cem.stress(p))) / 3
        out["m Fe3C"] = float(np.mean(fe_moments(aux_cem, self.cem.species)))
        out["charge C oct"] = float(aux_oct["charge"][16])
        out["charge C Fe3C"] = float(np.mean(np.asarray(aux_cem["charge"])[12:]))
        return out

    def residuals(self, x):
        values = dict(zip(FREE, x))
        try:
            # cementite at its experimental positions during the fit (relaxed
            # in the report): its relaxation costs as much as the rest together
            out = self.properties(values, relax_cementite=False)
        except Exception as err:  # noqa: BLE001 (a failed evaluation is a bad point)
            print("failed:", err, file=sys.stderr)
            return np.full(len(TARGETS), 1e2)
        r = np.array([(out[k] - t) / tol for k, (t, tol) in TARGETS.items()])
        r = np.nan_to_num(r, nan=1e2, posinf=1e2, neginf=-1e2)
        self.n_eval += 1
        cost = float(r @ r)
        if cost < self.best[0]:
            self.best = (cost, values)
            print(f"[{self.n_eval:4d}] cost {cost:9.3f}  "
                  + " ".join(f"{k}={v:.4f}" for k, v in values.items()), flush=True)
            print("       " + " ".join(f"{k}={out[k]:.3f}" for k in TARGETS), flush=True)
        return r


def report(model, values):
    out = model.properties(values, full=False)
    print("shell relaxation (as fitted): u oct", np.round(out["u oct"], 3),
          "u tet", np.round(out["u tet"], 3))
    for k, (t, _) in TARGETS.items():
        print(f"  {k:14s} {out[k]:9.4f} {t:9.4f}")
    print("full relaxation:")
    out = model.properties(values, full=True)
    print(f"{'property':14s} {'model':>9s} {'target':>9s}")
    for k, (t, _) in TARGETS.items():
        print(f"{k:14s} {out[k]:9.4f} {t:9.4f}")
    for k in ("m Fe nn oct", "charge C Fe3C"):
        print(f"{k:14s} {out[k]:9.4f}")
    cost = sum(((out[k] - t) / tol) ** 2 for k, (t, tol) in TARGETS.items())
    print(f"cost {cost:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="evaluate the start only")
    ap.add_argument("--max-nfev", type=int, default=40)
    args = ap.parse_args()

    model = Model()
    x0 = np.array([IRON_CARBON_FIT[k] for k in FREE])
    lo, hi = np.array([b[0] for b in FREE.values()]), np.array([b[1] for b in FREE.values()])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    t = time.time()
    model.residuals(x0)
    print(f"one evaluation: {time.time() - t:.1f} s", flush=True)
    if not args.quick:
        least_squares(model.residuals, x0, bounds=(lo, hi), x_scale="jac", diff_step=3e-3,
                      max_nfev=args.max_nfev)
    values = dict(IRON_CARBON_FIT, **model.best[1])
    report(model, values)
    shown = {k: values[k] for k in FREE}
    print("IRON_CARBON_FIT =", json.dumps({k: round(float(v), 5) for k, v in shown.items()},
                                          indent=4))


if __name__ == "__main__":
    main()
