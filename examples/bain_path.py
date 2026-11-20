"""Bain path bcc -> fcc of the iron BOP, ferromagnetic, antiferromagnetic and
non-magnetic.

Needs scipy (installed with the md extra:  uv sync --extra md); the figure
needs matplotlib.
Run with:  uv run python examples/bain_path.py [--basis d|sd] [--figure bain.png]
           [--out bain.npz] [--volumes 10.4 10.8 ...] [--ratios 0.85 0.9 ...]

The path is a body-centred tetragonal cell of two atoms, a x a x c with an
atom at the corner and one at the body centre, at fixed volume per atom
V = a^2 c / 2.  c/a = 1 is bcc and c/a = sqrt(2) is fcc (the fcc cube axes
are the c axis and the diagonals of the square base).  Three magnetic states
are followed along the path:

  FM   moments (+, +): bcc FM -> fcc FM
  AFM  moments (+, -): alternating (001) planes, so bcc AFM (B2 order) ->
       fcc AFM-I, the antiferromagnet of examples/_iron.py and docs/iron.md
  NM   no moments

Prints
  1. E(c/a) at each volume for each state, relative to bcc FM at its
     equilibrium, with the moments;
  2. for each volume the barrier along the path for the lowest state at each
     c/a, and where the magnetic order changes;
  3. the path at constant pressure: for each c/a and state the enthalpy
     H = E + P V minimised over the volume (a cubic fit of E(V) on the volume
     grid), at the pressures given by --pressures, with the barrier from bcc
     FM to fcc AFM.

Points whose self-consistency fails are left out (nan).  Each point keeps the
lowest of a warm start from the previous c/a and fresh starts from the
state's moments scaled by 1, 0.8, 0.6 and 0.3, as in examples/iron_properties.py.

With an s shell the self-consistency also has unphysical solutions: an s spin
channel almost full or empty, recursion coefficients b_inf two to five times
those of the physical states and the Fermi level several eV away.  The moments
expansion is outside its range there, and the energies can be hundreds of
meV/atom *below* the physical states (about -870 meV for "AFM" at 12.2 A^3).
Solutions with an s spin channel outside 0.05 to 0.9 electrons are rejected
(physical states along the path have 0.4 to 0.85).
"""

import argparse
import dataclasses

import numpy as np
from _iron import EV_A3_TO_GPA, Calculator, Phase, iron
from scipy.optimize import minimize_scalar

BASES = {"d": ("d",), "sd": ("s", "d")}
STATES = {"FM": (2.5, 2.5), "AFM": (2.0, -2.0), "NM": None}
FCC = np.sqrt(2.0)
S_MIN, S_MAX = 0.05, 0.9  # accepted electrons per s spin channel


def bct(v, r=1.0):
    """(cell, positions) of the two-atom body-centred tetragonal cell with
    volume v per atom and c/a = r."""
    a = (2.0 * v / r) ** (1.0 / 3.0)
    c = r * a
    return np.diag([a, a, c]), np.array([[0.0, 0.0, 0.0], [0.5 * a, 0.5 * a, 0.5 * c]])


class PathCalculator:
    """Calculators of one magnetic state along the path.

    The neighbour lists of a calculator hold for every cell that does not
    shorten any vector by more than the skin allows: with D the deformation
    from the reference cell, the lists are valid while
    rcut / sigma_min(D) <= rcut + skin.  A new calculator (and a compilation)
    is made when the path leaves that range.  A small skin keeps the clusters
    small: skin 0.6 costs about 6 times skin 0.2 per evaluation.
    """

    def __init__(self, state, basis, skin=0.2):
        self.state, self.basis, self.skin = state, basis, skin
        self.calcs = []  # (reference cell, calculator)
        self.last = None  # shifts of the previous point, for a warm start

    def _calculator(self, v, r):
        cell = bct(v, r)[0]
        for ref, calc in self.calcs:
            d = np.linalg.solve(ref, cell)  # cell = ref @ d, rows are vectors
            s_min = np.linalg.svd(d, compute_uv=False).min()
            rcut = max(calc.cfg.rcut, calc.cfg.rcut2 or 0.0)
            if rcut / s_min <= rcut + self.skin - 1e-9:
                return calc
        phase = Phase(self.state, lambda vol, rr=r: bct(vol, rr), v, STATES[self.state])
        calc = Calculator(phase, skin=self.skin, basis=self.basis)
        self.calcs.append((cell, calc))
        return calc

    def physical(self, aux):
        """False for the spurious solutions with an s spin channel almost full
        or empty (see the module docstring)."""
        if "s" not in self.basis or "n_spin" not in aux:
            return True
        n_s = np.asarray(aux["n_spin"])[..., self.basis.index("s")]  # (atoms, spin)
        return bool(np.all((n_s > S_MIN) & (n_s < S_MAX)))

    def energy(self, params, v, r):
        """(energy per atom, moments) of the lowest self-consistent state."""
        calc = self._calculator(v, r)
        calc.extra = (r,)
        phase = calc.phase
        best = None
        starts = [None] if phase.moments is None else [self.last, 1.0, 0.8, 0.6, 0.3]
        for start in starts:
            if phase.moments is not None:
                if isinstance(start, float):
                    calc.phase = dataclasses.replace(
                        phase, moments=tuple(start * m for m in STATES[self.state]))
                    calc._last = None
                else:  # warm start from the previous point (if any)
                    if start is None:
                        continue
                    calc._last = start
            e, aux = calc.energy(params, v)
            if calc.residual < 1e-6 and self.physical(aux) and (
                    best is None or e < best[0] - 1e-9):
                best = (e, aux)
        calc.phase = phase
        if best is None:
            return np.nan, np.full(2, np.nan)
        e, aux = best
        if phase.moments is not None:
            self.last = aux["shifts"]
            m = np.asarray(aux["moment"])
            m = m.sum(axis=-1) if m.ndim > 1 else m  # d + s (+ p) moment per atom
        else:
            m = np.zeros(2)
        return e, m


def label(m):
    """Magnetic order from the two moments."""
    if not np.all(np.isfinite(m)):
        return "--"
    if np.max(np.abs(m)) < 0.05:
        return "NM"
    return "FM" if m[0] * m[1] > 0 else "AFM"


def constant_pressure(volumes, energies, p):
    """min over V of E(V) + p V from a cubic fit, inside the volume grid."""
    ok = np.isfinite(energies)
    if ok.sum() < 4:
        return np.nan, np.nan
    fit = np.polyfit(volumes[ok], energies[ok], 3)
    res = minimize_scalar(lambda x: np.polyval(fit, x) + p * x,
                          bounds=(volumes[ok].min(), volumes[ok].max()), method="bounded")
    return res.fun, res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=BASES, default="sd",
                    help="d (IRON_FIT) or sd (IRON_SD_FIT, default)")
    ap.add_argument("--volumes", type=float, nargs="+",
                    default=[10.2, 10.6, 11.0, 11.4, 11.8, 12.2],
                    help="volumes per atom (A^3)")
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=list(np.round(np.arange(0.80, 1.601, 0.05), 3)) + [FCC],
                    help="c/a values (sqrt(2) = fcc is always added)")
    ap.add_argument("--pressures", type=float, nargs="+", default=[0.0, 10.0],
                    help="pressures (GPa) of the constant-pressure path")
    ap.add_argument("--figure", help="write the path to this file")
    ap.add_argument("--out", help="save energies and moments (npz)")
    args = ap.parse_args()

    basis = BASES[args.basis]
    params = iron(basis=basis)
    volumes = np.array(sorted(args.volumes))
    ratios = np.array(sorted(set(np.round(list(args.ratios) + [1.0, FCC], 6))))
    print(f"basis {args.basis}; {len(volumes)} volumes x {len(ratios)} c/a x {len(STATES)} states")

    # reference: bcc FM at its equilibrium volume
    ref = PathCalculator("FM", basis)
    e_ref = minimize_scalar(lambda v: ref.energy(params, v, 1.0)[0], bounds=(11.0, 12.4),
                            method="bounded", options={"xatol": 1e-3})
    v_ref = e_ref.x
    e_ref = ref.energy(params, v_ref, 1.0)[0]
    print(f"bcc FM equilibrium: V = {v_ref:.3f} A^3, E = {e_ref:.5f} eV/atom\n")

    energy = np.full((len(STATES), len(volumes), len(ratios)), np.nan)
    moment = np.full((len(STATES), len(volumes), len(ratios), 2), np.nan)
    for s, state in enumerate(STATES):
        path = PathCalculator(state, basis)
        for i, v in enumerate(volumes):
            path.last = None
            # start from bcc and walk both ways, so the warm start follows the branch
            j1 = int(np.argmin(np.abs(ratios - 1.0)))
            for j in list(range(j1, len(ratios))) + list(range(j1 - 1, -1, -1)):
                if j == j1 - 1:
                    path.last = None
                energy[s, i, j], moment[s, i, j] = path.energy(params, v, ratios[j])

    de = 1000 * (energy - e_ref)  # meV/atom
    j_bcc = int(np.argmin(np.abs(ratios - 1.0)))
    j_fcc = int(np.argmin(np.abs(ratios - FCC)))

    print("1. E - E(bcc FM, equilibrium) (meV/atom) and moments |m1|, |m2| (muB)")
    for i, v in enumerate(volumes):
        print(f"\n  V = {v:.2f} A^3")
        print(f"  {'c/a':>6s}" + "".join(f" {st:>9s} {'(m, order)':>14s}" for st in STATES))
        for j, r in enumerate(ratios):
            row = f"  {r:6.3f}"
            for s in range(len(STATES)):
                m = moment[s, i, j]
                row += f" {de[s, i, j]:9.1f} {np.mean(np.abs(m)):6.2f} {label(m):>7s}"
            tag = "  bcc" if j == j_bcc else "  fcc" if j == j_fcc else ""
            print(row + tag)

    print("\n2. Lowest state along the path at fixed volume (meV/atom)")
    print(f"  {'V':>6s} {'E bcc':>8s} {'E fcc':>8s} {'max between':>12s} {'at c/a':>7s} "
          f"{'barrier':>8s}  states along the path (barrier: above the higher end)")
    for i, v in enumerate(volumes):
        low = np.nanmin(de[:, i, :], axis=0)
        kind = [list(STATES)[k] if np.isfinite(low[j]) else "--"
                for j, k in enumerate(np.argmin(np.where(np.isfinite(de[:, i, :]),
                                                         de[:, i, :], np.inf), axis=0))]
        # the label of the state that is lowest: a start may have changed order
        kind = [label(moment[list(STATES).index(k), i, j]) if k != "--" else "--"
                for j, k in enumerate(kind)]
        seg = slice(j_bcc, j_fcc + 1)
        jm = j_bcc + int(np.nanargmax(low[seg]))
        barrier = low[jm] - max(low[j_bcc], low[j_fcc])
        changes = [kind[j_bcc]]
        for j in range(j_bcc + 1, j_fcc + 1):
            if kind[j] != changes[-1].split("@")[0]:
                changes.append(f"{kind[j]}@{ratios[j]:.2f}")
        print(f"  {v:6.2f} {low[j_bcc]:8.1f} {low[j_fcc]:8.1f} {low[jm]:12.1f} {ratios[jm]:7.3f} "
              f"{barrier:8.1f}  {' -> '.join(changes)}")

    print("\n3. Path at constant pressure: H - H(bcc FM) (meV/atom), V (A^3) minimised "
          "for each c/a and state")
    h_all = {}
    for p_gpa in args.pressures:
        p = p_gpa / EV_A3_TO_GPA
        h = np.full((len(STATES), len(ratios)), np.nan)
        vol = np.full_like(h, np.nan)
        for s in range(len(STATES)):
            for j in range(len(ratios)):
                h[s, j], vol[s, j] = constant_pressure(volumes, energy[s, :, j], p)
        h = 1000 * (h - h[0, j_bcc])
        h_all[p_gpa] = (h, vol)
        print(f"\n  P = {p_gpa:.1f} GPa (* volume at the edge of the grid: the minimum "
              "lies outside it, extend --volumes)")
        print(f"  {'c/a':>6s}" + "".join(f" {st:>8s} {'V':>6s} " for st in STATES))
        for j, r in enumerate(ratios):
            edge = np.isclose(vol[:, j], volumes.min(), atol=1e-3) | np.isclose(
                vol[:, j], volumes.max(), atol=1e-3)
            print(f"  {r:6.3f}" + "".join(f" {h[s, j]:8.1f} {vol[s, j]:6.2f}{'*' if edge[s] else ' '}"
                                          for s in range(len(STATES))))
        low = np.nanmin(h, axis=0)
        seg = slice(j_bcc, j_fcc + 1)
        jm = j_bcc + int(np.nanargmax(low[seg]))
        print(f"  bcc FM -> fcc {list(STATES)[int(np.nanargmin(h[:, j_fcc]))]}: "
              f"H(fcc) - H(bcc) = {low[j_fcc] - low[j_bcc]:.1f} meV/atom; highest point "
              f"{low[jm]:.1f} at c/a = {ratios[jm]:.3f}, barrier above the higher end "
              f"{low[jm] - max(low[j_bcc], low[j_fcc]):.1f} meV/atom")

    if args.out:
        np.savez(args.out, states=list(STATES), volumes=volumes, ratios=ratios,
                 energy=energy, moment=moment, e_ref=e_ref, v_ref=v_ref)
        print(f"\ndata written to {args.out}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        styles = {"FM": "-", "AFM": "--", "NM": ":"}
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(volumes)))
        for s, state in enumerate(STATES):
            for i, v in enumerate(volumes):
                kw = dict(color=colors[i], ls=styles[state])
                axes[0].plot(ratios, de[s, i], label=f"V = {v:.1f}" if s == 0 else None, **kw)
                if state != "NM":
                    axes[1].plot(ratios, np.mean(np.abs(moment[s, i]), axis=-1), **kw)
        for p_gpa, (h, _) in h_all.items():
            for s, state in enumerate(STATES):
                axes[2].plot(ratios, h[s], ls=styles[state], marker="o", ms=2,
                             label=f"{state}, {p_gpa:g} GPa")
        for ax in axes:
            ax.axvline(1.0, color="grey", lw=0.5)
            ax.axvline(FCC, color="grey", lw=0.5)
        axes[0].set(xlabel="c/a", ylabel="E - E(bcc FM) (meV/atom)",
                    title="fixed volume (colour: V in A^3)")
        axes[1].set(xlabel="c/a", ylabel="|moment| (muB)", title="moments")
        axes[2].set(xlabel="c/a", ylabel="H - H(bcc FM) (meV/atom)", title="constant pressure")
        handles = axes[0].get_legend_handles_labels()[0] + [
            plt.Line2D([], [], color="k", ls=styles[st], label=st) for st in STATES]
        axes[0].legend(handles=handles, fontsize=7, ncol=2)
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"figure written to {args.figure}")


if __name__ == "__main__":
    main()
