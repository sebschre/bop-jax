"""Structural, magnetic and elastic properties of the iron BOP.

Needs scipy (installed with the md extra:  uv sync --extra md); the figure
needs matplotlib.
Run with:  uv run python examples/iron_properties.py [--figure iron.png] [--basis d|sd]

Prints
  1. the phases ordered by energy, with lattice parameters, volume, moment
     and bulk modulus,
  2. the elastic constants of the ground state and of the next stable phase,
  3. the magnetic energies and the collapse of the moment under compression,
  4. the pressure of the bcc -> hcp transition.
"""

import argparse
import dataclasses

import numpy as np
from _iron import EV_A3_TO_GPA, PHASES, Calculator, atomic_volume, iron, mean_abs_moment
from scipy.optimize import brentq, curve_fit, minimize_scalar

BASES = {"d": ("d",), "sd": ("s", "d")}
BASIS = BASES["d"]
PARAMS = iron()


def calculator(name, a_guess=None):
    phase = PHASES[name]
    if a_guess is not None:
        phase = dataclasses.replace(phase, a_guess=a_guess)
    return Calculator(phase, basis=BASIS)


def equilibrium(name):
    """Calculator rebuilt around the equilibrium, lattice parameter(s), energy, aux."""
    calc = calculator(name)
    a, e, aux = calc.equilibrium(PARAMS)
    calc = calculator(name, a)  # neighbour lists centred on the equilibrium
    if name.startswith("hcp"):
        def at_ratio(ca):
            calc.extra = (ca,)
            return calc.equilibrium(PARAMS)[1]
        ca = minimize_scalar(at_ratio, bounds=(1.50, 1.72), method="bounded",
                             options={"xatol": 1e-4}).x
        calc.extra = (ca,)
    a, e, aux = calc.equilibrium(PARAMS)
    return calc, a, e, aux


def birch_murnaghan(v, e0, v0, b0, b1):
    x = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((x - 1) ** 3 * b1 + (x - 1) ** 2 * (6 - 4 * x))


def eos(calc, a, span=0.05, n=11):
    """Energy-volume curve around a, its Birch-Murnaghan fit and the moments."""
    scales = np.linspace(1 - span, 1 + span, n)
    vols, ens, moms = [], [], []
    for s in scales:
        e, aux = calc.energy(PARAMS, s * a)
        vols.append(atomic_volume(calc.phase, s * a) if not calc.extra else
                    abs(np.linalg.det(calc.phase.lattice(s * a, *calc.extra)[0])) / 2)
        ens.append(e)
        moms.append(mean_abs_moment(aux))
    vols, ens, moms = np.array(vols), np.array(ens), np.array(moms)
    ok = np.isfinite(ens)  # a point whose self-consistency failed is left out
    vols, ens, moms = vols[ok], ens[ok], moms[ok]
    i = np.argmin(ens)
    fit, _ = curve_fit(birch_murnaghan, vols, ens, p0=(ens[i], vols[i], 1.0, 5.0))
    return vols, ens, moms, fit


def volume(calc, a):
    cell, pos = calc.phase.lattice(a, *calc.extra)
    return abs(np.linalg.det(cell)) / len(pos)


def print_elastic(name, calc, a):
    hexagonal = name.startswith("hcp")
    c = calc.elastic_tensor(PARAMS, a, relax=hexagonal)
    if hexagonal:
        vals = dict(C11=0.5 * (c[0, 0] + c[1, 1]), C12=c[0, 1], C13=0.5 * (c[0, 2] + c[1, 2]),
                    C33=c[2, 2], C44=0.5 * (c[3, 3] + c[4, 4]), C66=c[5, 5])
        b = (2 * (vals["C11"] + vals["C12"]) + vals["C33"] + 4 * vals["C13"]) / 9  # Voigt
    else:
        vals = dict(C11=np.mean(np.diag(c)[:3]), C12=np.mean([c[0, 1], c[0, 2], c[1, 2]]),
                    C44=np.mean(np.diag(c)[3:]))
        vals["C'"] = 0.5 * (vals["C11"] - vals["C12"])
        b = (vals["C11"] + 2 * vals["C12"]) / 3
        vals["A"] = vals["C44"] / vals["C'"]
    vals["B"] = b
    print(f"  {name}: " + "  ".join(f"{k} = {v:6.1f}" if k != "A" else f"A = {v:.2f}"
                                    for k, v in vals.items()))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", help="write energy-volume and moment curves to this file")
    ap.add_argument("--basis", choices=BASES, default="d",
                    help="d (IRON_FIT) or sd (IRON_SD_FIT)")
    args = ap.parse_args()
    global BASIS, PARAMS
    BASIS = BASES[args.basis]
    PARAMS = iron(basis=BASIS)
    print(f"basis {args.basis}")

    print("1. Phases, ordered by energy (equilibrium of each; E relative to the lowest)")
    results = {}
    for name in PHASES:
        calc, a, e, aux = equilibrium(name)
        results[name] = dict(calc=calc, a=a, e=e, m=mean_abs_moment(aux))
    # a magnetic start that ends without moments is the non-magnetic phase again
    order = sorted(results, key=lambda k: results[k]["e"])
    e_min = results[order[0]]["e"]
    print(f"  {'phase':8s} {'a (A)':>7s} {'c/a':>6s} {'V (A^3)':>8s} {'dE (meV)':>9s} "
          f"{'|m| (muB)':>9s}  note")
    for name in order:
        r = results[name]
        calc = r["calc"]
        ca = f"{calc.extra[0]:6.3f}" if calc.extra else "     -"
        note = "collapsed to NM" if PHASES[name].moments and r["m"] < 0.05 else ""
        print(f"  {name:8s} {r['a']:7.4f} {ca} {volume(calc, r['a']):8.3f} "
              f"{1000 * (r['e'] - e_min):9.1f} {r['m']:9.3f}  {note}")
    # distinct phases: drop magnetic starts that collapsed onto another state
    distinct = [k for k in order if not (PHASES[k].moments and results[k]["m"] < 0.05)]
    ground, second = distinct[0], distinct[1]
    print(f"  ground state {ground}; next {second}, "
          f"{1000 * (results[second]['e'] - results[ground]['e']):.1f} meV/atom higher")

    print("\n2. Elastic constants (GPa)")
    for name in (ground, second):
        print_elastic(name, results[name]["calc"], results[name]["a"])
    print("  experiment, bcc Fe at 4 K: C11 = 243, C12 = 138, C44 = 122, C' = 52.5, "
          "B = 173, A = 2.32")

    print("\n3. Magnetism")
    for lattice in ("bcc", "fcc", "hcp"):
        nm = results[f"{lattice} NM"]["e"]
        parts = []
        for kind in ("FM", "AFM"):
            r = results[f"{lattice} {kind}"]
            parts.append(f"E({kind}) - E(NM) = {1000 * (r['e'] - nm):7.1f} meV, "
                         f"|m| = {r['m']:.2f}")
        print(f"  {lattice}: " + ";  ".join(parts))
    curves = {}
    for name in ("bcc FM", "bcc NM", "hcp NM", "fcc AFM", "fcc NM"):
        r = results[name]
        curves[name] = eos(r["calc"], r["a"], span=0.07 if name == "bcc FM" else 0.05)
    v, e, m, fit = curves["bcc FM"]
    print("  bcc FM moment against volume:")
    for vi, mi in zip(v[::2], m[::2]):
        print(f"    V = {vi:6.2f} A^3   |m| = {mi:.3f}")
    for name, (_, _, _, f) in curves.items():
        print(f"  {name:8s} Birch-Murnaghan: V0 = {f[1]:.3f} A^3, "
              f"B0 = {f[2] * EV_A3_TO_GPA:6.1f} GPa, B' = {f[3]:.2f}")

    print("\n4. bcc FM -> hcp NM under pressure (common tangent of the fitted E(V))")
    fb, fh = curves["bcc FM"][3], curves["hcp NM"][3]

    def enthalpy(f, p):  # p in eV/A^3
        vv = minimize_scalar(lambda x: birch_murnaghan(x, *f) + p * x,
                             bounds=(0.7 * f[1], 1.1 * f[1]), method="bounded").x
        return birch_murnaghan(vv, *f) + p * vv, vv

    try:
        pt = brentq(lambda p: enthalpy(fb, p)[0] - enthalpy(fh, p)[0], 0.0, 60 / EV_A3_TO_GPA)
        vb, vh = enthalpy(fb, pt)[1], enthalpy(fh, pt)[1]
        print(f"  transition at {pt * EV_A3_TO_GPA:.1f} GPa, volume {vb:.2f} -> {vh:.2f} A^3 "
              "(experiment: 13 GPa; GGA: about 10 GPa)")
    except ValueError:
        print("  no transition below 60 GPa")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for name, (v, e, m, f) in curves.items():
            vv = np.linspace(v.min(), v.max(), 200)
            (line,) = ax1.plot(vv, 1000 * (birch_murnaghan(vv, *f) - e_min), label=name)
            ax1.plot(v, 1000 * (e - e_min), "o", ms=3, color=line.get_color())
            if PHASES[name].moments:
                ax2.plot(v, m, "o-", ms=3, label=name, color=line.get_color())
        ax1.set(xlabel="volume per atom (A^3)", ylabel="E - E(bcc FM) (meV/atom)")
        ax2.set(xlabel="volume per atom (A^3)", ylabel="|moment| (muB)")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"\nfigure written to {args.figure}")


if __name__ == "__main__":
    main()
