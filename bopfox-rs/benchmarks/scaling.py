"""bopfox-rs on the setup of bop-jax/benchmarks/scaling.py (same positions):
energy + force at zero shifts, and self-consistent energy + force from zero
shifts (tol 1e-8 in max |dE/dshift|, as there).  Times include building the
neighbour lists and clusters.  Run from the bop-jax root:
    uv run python bopfox-rs/benchmarks/scaling.py [n_max]
"""

import sys
import time

import numpy as np

import bopjax as bj
import bopfox_rs
from bopjax.models import canonical_d

sys.path.insert(0, "benchmarks")
from scaling import supercell  # noqa: E402


def best(fn, repeat=3):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), out


def main(n_max=4):
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged")
    pot = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, canonical_d(hubbard=8.0)))
    print("atoms  energy+force(s)  scf+force(s)  gradient evals  U/atom")
    for n in range(1, n_max + 1):
        pos, cell = supercell(n)
        species = np.zeros(len(pos), dtype=int)
        t_fixed, _ = best(lambda: pot.compute(pos, species, cell))
        t_scf, out = best(lambda: pot.scf(pos, species, cell, tol=1e-8))
        assert out["converged"]
        print(f"{len(pos):5d}  {t_fixed:15.3f}  {t_scf:12.3f}  {len(out['history']):14d}  "
              f"{out['energy'] / len(pos):.10f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
