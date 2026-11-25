"""Migration barrier of carbon in bcc iron with the iron-carbon BOP: NEB on
bopfox-rs from one octahedral site to the next, through the tetrahedral site.

Checks that the minimum energy path goes through the symmetric tetrahedral
saddle that ``fit_iron_carbon_rs.py`` fits to (E(tet) - E(oct)).
Fe16C, every atom relaxed, potential self-consistency with exact forces.

Run with:  uv run python examples/neb_iron_carbon_rs.py [checkpoint.json] [--images 5]
"""

import argparse
import json

import numpy as np

from _iron_carbon_rs import A_FE, RsCell, bcc_supercell, make_potential
from bopfox_rs.neb import image_forces, neb
from bopjax.models import C, FE, iron_carbon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", help="fit checkpoint (default IRON_CARBON_FIT)")
    ap.add_argument("--images", type=int, default=5, help="images between the end points")
    ap.add_argument("--fmax", type=float, default=0.02)
    args = ap.parse_args()
    values = json.load(open(args.checkpoint))["values"] if args.checkpoint else {}
    p = iron_carbon(**values)
    cell, pos, sp = bcc_supercell()
    sp = sp + [C]
    ends = []
    rng = np.random.default_rng(3)
    for frac in ([0.5, 0.5, 0.0], [0.5, 0.0, 0.0]):  # octahedral sites, axes z and x
        c = RsCell("oct", cell, np.vstack([pos, A_FE * np.array(frac)]), sp)
        c.set_params(p)
        c.positions = c.positions + 0.01 * rng.standard_normal(c.positions.shape)
        out = c.relax(ftol=2e-3)
        ends.append((c.positions.copy(), out))
        print(f"end point E {out['energy']:.5f}  fmax {c.max_force:.4f}", flush=True)
    a, b = ends[0][0], ends[1][0]
    d = b - a
    d -= np.round(d @ np.linalg.inv(cell)) @ cell  # minimum image
    n_img = args.images + 2
    band = np.array([a + t * d for t in np.linspace(0.0, 1.0, n_img)])
    m = np.zeros((len(sp), 3))
    m[np.array(sp) == FE, 2] = 2.4
    import bopjax as bj
    x0 = np.asarray(bj.stoner_shifts(m, float(p["stoner"][FE, 2])))
    pot = make_potential(p)
    ff = image_forces(pot, sp, cell, x0, n_img, tol=1e-6, condition="potential",
                      scf_options={"adjoint_tol": 1e-7})
    res = neb(ff, band, fmax=args.fmax, max_step=0.05, steps=400, verbose=True)
    e = res.energies - res.energies[0]
    print("converged", res.converged, "climbing", res.climbing)
    print("E - E(oct) along the band:", np.round(e, 4))
    print(f"barrier {e.max():.4f} eV")
    saddle = res.images[int(np.argmax(e))]
    print("C at the saddle (fractional):", np.round(saddle[-1] / A_FE % 2, 3))


if __name__ == "__main__":
    main()
