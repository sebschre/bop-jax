"""Fit the Fe-C cross terms of the iron-carbon BOP (Egorov d iron, sp carbon).

Only the Fe-C parameters are fitted; Fe-Fe and C-C stay those of
``iron_egorov`` and ``CARBON_FIT``, as Egorov et al. fitted only the cross
terms of their Fe-Co BOP on fixed elemental models.  Self-consistency is the
potential condition (Anderson mixing); inside the fit, forces and stress are
partial derivatives at the converged shifts (no adjoint solve).  Needs scipy (uv sync --extra md).
Run with:  uv run python examples/fit_iron_carbon.py  [--quick] [--max-nfev N]

Each residual is (model - target) / tolerance; targets in ``TARGETS``, see
docs/iron-carbon.md.  The fitted values are printed in the form of
``bopjax.models.IRON_CARBON_FIT``.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import jax

# compiled functions are cached on disk, so the short-lived cell processes
# of the fit load them instead of compiling again
jax.config.update("jax_compilation_cache_dir",
                  os.environ.get("BOPJAX_JAX_CACHE", os.path.expanduser("~/.cache/bopjax/jax")))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import jax.numpy as jnp  # noqa: E402

import numpy as np  # noqa: E402
from _iron_carbon import (Calculator, Cell, bcc_fe, bcc_supercell, cementite, diamond,
                          fe_moments, fe_with_carbon, iron_carbon)

from bopjax.models import IRON_CARBON_CONFIG
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
    "decay_pi": (1.0, 8.0),  # decay of pdp alone
    "core": (0.0, 40.0),  # short-range Fe-C core repulsion: the axial neighbours
    "core_r": (1.5, 1.95),
}


CELLS = {"oct": lambda: fe_with_carbon("octahedral"), "tet": lambda: fe_with_carbon("tetrahedral"),
         "cem": cementite}


def cell_job(name, values, mode, state, out_path, exact):
    """One cell, one evaluation, in a process of its own (``--cell``): exact
    forces and stress of Fe16C take several GB to compile, and three cells
    kept in one process, or in three resident workers, ran out of 15 GB.
    The persistent compilation cache makes a fresh process cheap; warm
    starts (shifts, shell amplitudes, positions) go through ``state``."""
    config = dict(IRON_CARBON_CONFIG, batch=4)  # exact forces otherwise need > 15 GB
    calc = Calculator(Cell(name, *CELLS[name]()), config=config)
    calc.exact = exact
    if os.path.exists(state):
        st = np.load(state)
        calc._x = jnp.asarray(st["shifts"])
        if "amplitudes" in st and len(st["amplitudes"]):
            calc._modes = jnp.asarray(calc.shell_modes())
            calc._base = calc.positions
            calc.amplitudes = st["amplitudes"]
        calc.positions = jnp.asarray(st["positions"])
    p = iron_carbon(**values)
    gtol = 0.15 if exact else 2e-3  # exact gradients carry noise from the adjoint solve
    if mode == "relax":  # every atom (report); the tetrahedral C stays on its site
        e, aux = calc.relax(p, fixed=[16] if name == "tet" else None)
    elif name == "cem":
        e, aux = calc.solve(p)
    else:
        e, aux = calc.relax_shells(p, gtol=gtol)
    out = {"energy": e, "moment": np.asarray(aux["moment"]), "charge": np.asarray(aux["charge"]),
           "positions": np.asarray(calc.positions),
           "amplitudes": np.asarray(getattr(calc, "amplitudes", [])),
           "stationarity": calc.stationarity, "volume": calc.volume()}
    if name in ("oct", "cem"):
        out["stress"] = calc.stress(p)
    np.savez(out_path, **out)
    if mode == "fit":  # warm start for the next evaluation
        np.savez(state, shifts=np.asarray(calc._x), positions=out["positions"],
                 amplitudes=out["amplitudes"])


class Model:
    def __init__(self, exact=True, checkpoint=None,
                 workdir=os.path.expanduser("~/.cache/bopjax/fit_state")):
        # exact (implicit-differentiation) forces and stress, or partial
        # derivatives at fixed shifts: those are off by up to 0.76 eV/A at
        # the parameters of the first fit
        self.exact = exact
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.checkpoint = checkpoint
        p = iron_carbon()
        self.e_fe = Calculator(Cell("bcc Fe", *bcc_fe())).solve(p)[0] / 2
        self.e_c = Calculator(Cell("diamond", *diamond(), magnetic=False)).solve(p)[0] / 2
        host = Calculator(Cell("Fe16", *bcc_supercell()))
        host.exact = False  # perfect crystal: partial and exact derivatives agree
        host.solve(p)
        self.host_stress = host.stress(p)  # zero up to the precision of A_FE
        self.species = np.array(cementite()[2])
        self.n_eval = 0
        self.best = (np.inf, None)

    def _ask(self, name, values, mode="fit"):
        out = os.path.join(self.workdir, f"{name}_out.npz")
        if os.path.exists(out):
            os.remove(out)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--cell", name, "--mode", mode,
               "--values", json.dumps({k: (None if v is None else float(v))
                                       for k, v in values.items()}),
               "--state", os.path.join(self.workdir, f"{name}_state.npz"), "--out", out]
        if not self.exact:
            cmd.append("--partial")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(out):
            tail = (res.stderr or "").strip().splitlines()[-3:]
            raise RuntimeError(f"{name} job failed ({res.returncode}): {' | '.join(tail)}")
        with np.load(out) as d:
            return {k: d[k] for k in d.files}

    def properties(self, values, relax_cementite=False, full=False):
        """Fit quantities.  The fit relaxes the breathing modes of the first
        two Fe shells around C; with ``full`` every atom is relaxed.
        Cementite stays at the experimental positions (``relax_cementite``
        relaxes it)."""
        mode = "relax" if full else "fit"
        o = self._ask("oct", values, mode)
        t = self._ask("tet", values, mode)
        c = self._ask("cem", values, "relax" if relax_cementite else "fit")
        out = {"u oct": o["amplitudes"], "u tet": t["amplitudes"]}
        o = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in o.items()}
        t = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in t.items()}
        c = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in c.items()}
        out["E_sol oct"] = o["energy"] - 16 * self.e_fe - self.e_c
        out["E tet - oct"] = t["energy"] - o["energy"]
        dp = -(o["stress"] - self.host_stress) * o["volume"] / 160.21766
        out["P axial"] = dp[2, 2]
        out["P transverse"] = 0.5 * (dp[0, 0] + dp[1, 1])
        # moment of the two Fe neighbours of C at a/2, along its axis
        pos, box = o["positions"], np.diag(fe_with_carbon("octahedral")[0])
        d = pos[:16] - pos[16]
        d -= np.round(d / box) * box
        nn = np.argsort(np.linalg.norm(d, axis=1))[:2]
        out["m Fe nn oct"] = float(o["moment"][nn].mean())
        out["m C oct"] = float(o["moment"][16])
        out["dH Fe3C"] = (c["energy"] - 12 * self.e_fe - 4 * self.e_c) / 16
        out["p Fe3C"] = -float(np.trace(c["stress"])) / 3
        out["m Fe3C"] = float(np.mean(c["moment"][self.species == 0]))
        out["charge C oct"] = float(o["charge"][16])
        out["charge C Fe3C"] = float(np.mean(c["charge"][12:]))
        out["stationarity"] = max(o["stationarity"], t["stationarity"], c["stationarity"])
        return out

    def residuals(self, x):
        values = dict(zip(FREE, x))
        try:
            # cementite at its experimental positions during the fit
            out = self.properties(values)
        except Exception as err:  # noqa: BLE001 (a failed evaluation is a bad point)
            print("failed:", err, file=sys.stderr)
            return np.full(len(TARGETS), 1e2)
        r = np.array([(out[k] - t) / tol for k, (t, tol) in TARGETS.items()])
        r = np.nan_to_num(r, nan=1e2, posinf=1e2, neginf=-1e2)
        self.n_eval += 1
        cost = float(r @ r)
        if cost < self.best[0]:
            self.best = (cost, values)
            if self.checkpoint:  # survives a restart of the machine
                with open(self.checkpoint, "w") as fh:
                    json.dump({"cost": cost, "values": {k: float(v) for k, v in values.items()},
                               "properties": {k: float(np.asarray(out[k]).ravel()[0])
                                              if np.ndim(out[k]) else float(out[k])
                                              for k in TARGETS}}, fh, indent=1)
            print(f"[{self.n_eval:4d}] cost {cost:9.3f}  "
                  + " ".join(f"{k}={v:.4f}" for k, v in values.items()), flush=True)
            print("       " + " ".join(f"{k}={out[k]:.3f}" for k in TARGETS), flush=True)
        return r


def report(model, values, full=True):
    """Properties as fitted (shell relaxation) and, with ``full``, with every
    atom relaxed and cementite relaxed (exact forces: hours)."""
    rows = [("shells (fitted)", model.properties(values))]
    if full:
        rows.append(("all atoms", model.properties(values, relax_cementite=True, full=True)))
    print("u oct", np.round(rows[0][1]["u oct"], 3), "u tet", np.round(rows[0][1]["u tet"], 3))
    print(f"{'property':14s} " + " ".join(f"{n:>16s}" for n, _ in rows) + f" {'target':>9s}")
    for k, (t, _) in TARGETS.items():
        print(f"{k:14s} " + " ".join(f"{o[k]:16.4f}" for _, o in rows) + f" {t:9.4f}")
    for k in ("m Fe nn oct", "m C oct", "charge C Fe3C", "stationarity"):
        print(f"{k:14s} " + " ".join(f"{o[k]:16.4f}" for _, o in rows))
    for name, out in rows:
        cost = sum(((out[k] - t) / tol) ** 2 for k, (t, tol) in TARGETS.items())
        print(f"cost ({name}) {cost:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="evaluate the start only")
    ap.add_argument("--max-nfev", type=int, default=4,
                    help="least-squares iterations; each costs one evaluation per free "
                         "parameter and one more")
    ap.add_argument("--partial", action="store_true",
                    help="partial-derivative forces and stress (fast, inaccurate)")
    ap.add_argument("--checkpoint", help="JSON file the best point is written to")
    ap.add_argument("--start", help="JSON checkpoint to start from")
    ap.add_argument("--no-full-report", action="store_true",
                    help="report the fitted (shell-relaxed) properties only")
    ap.add_argument("--workdir", default=os.path.expanduser("~/.cache/bopjax/fit_state"),
                    help="directory for the warm starts of the cells")
    # internal: one cell, one evaluation (see cell_job)
    ap.add_argument("--cell", help=argparse.SUPPRESS)
    ap.add_argument("--mode", default="fit", help=argparse.SUPPRESS)
    ap.add_argument("--values", help=argparse.SUPPRESS)
    ap.add_argument("--state", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.cell:
        cell_job(args.cell, json.loads(args.values), args.mode, args.state, args.out,
                 exact=not args.partial)
        return

    model = Model(exact=not args.partial, checkpoint=args.checkpoint, workdir=args.workdir)
    start = dict(IRON_CARBON_FIT)
    if start.get("decay_pi") is None:
        start["decay_pi"] = start["decay"]
    if args.start:
        with open(args.start) as fh:
            start.update(json.load(fh)["values"])
    x0 = np.array([start[k] for k in FREE])
    lo, hi = np.array([b[0] for b in FREE.values()]), np.array([b[1] for b in FREE.values()])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    t = time.time()
    model.residuals(x0)
    print(f"one evaluation: {time.time() - t:.1f} s", flush=True)
    if not args.quick:
        least_squares(model.residuals, x0, bounds=(lo, hi), x_scale="jac", diff_step=3e-3,
                      max_nfev=args.max_nfev)
    values = dict(start, **model.best[1])
    report(model, values, full=not args.no_full_report)
    shown = {k: values[k] for k in FREE}
    print("IRON_CARBON_FIT =", json.dumps({k: round(float(v), 5) for k, v in shown.items()},
                                          indent=4))


if __name__ == "__main__":
    main()
