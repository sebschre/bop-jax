"""Figure of docs/carbon-dislocation.md from OUT/binding.json (written by
``examples/carbon_screw_rs.py OUT report``).

    uv run python examples/plot_carbon_screw.py OUT docs/figures/carbon_screw.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, png = sys.argv[1], sys.argv[2]
d = json.load(open(f"{out}/binding.json"))
r = {"oct_0": 0.43, "oct_1": 1.81, "oct_2": 2.42, "oct_3": 2.92, "oct_4": 3.35,
     "oct_5": 3.76, "oct_6": 4.10, "oct_7": 4.37, "hard_C_free": 1.19}
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)

for k, v in d["binding"].items():
    if k not in r:
        continue
    ax1.plot(r[k], v["E_b"], "o", ms=6, color=BLUE, mec="white", mew=1.5, zorder=3)
    label = "relaxed from prism" if k == "hard_C_free" else k
    ax1.annotate(label, (r[k], v["E_b"]), textcoords="offset points",
                 xytext=(-6, 3) if k == "hard_C_free" else (6, 3),
                 ha="right" if k == "hard_C_free" else "left",
                 fontsize=7.5, color=INK2)
ax1.axhline(0, color=INK2, lw=0.8)
ax1.set_xlabel("C distance from the easy core (A)")
ax1.set_ylabel("binding energy E_b (eV)")
ax1.set_title("C at a 1/2<111> screw: E_b < 0 attracts", fontsize=9, color=INK, loc="left")

styles = {"pure Fe": BLUE, "C, core moved away": ORANGE, "C, core moved past": AQUA}
for label, prof in d["drag"].items():
    f, e = prof["fraction"], [1e3 * x for x in prof["energy_per_b"]]
    ax2.plot(f, e, "-o", lw=2, ms=5, color=styles[label], mec="white", mew=1.2,
             label=label)
    ax2.annotate(label, (f[-1], e[-1]), textcoords="offset points", xytext=(4, 0),
                 fontsize=7.5, color=INK2, va="center")
ax2.set_xlabel("drag fraction f (core A -> next valley B)")
ax2.set_ylabel("E - E_A (meV per b)")
ax2.set_xlim(-0.03, 1.45)
ax2.set_title("Peierls drag, one C per 3 b of line", fontsize=9, color=INK, loc="left")
ax2.legend(frameon=False, fontsize=7.5, loc="upper left")
for ax in (ax1, ax2):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(png, facecolor="white")
print("wrote", png)
