"""Timing against system size: distorted fcc supercells, canonical d-band,
9 moments.  Run with:  uv run python benchmarks/scaling.py [n_max]

Prints, per size: neighbour/cluster construction, one energy + force
evaluation at fixed onsite levels, and a full self-consistent energy + force
evaluation started from zero shifts.  The first call of each compiled function
is excluded (compilation takes roughly 10 to 30 s per configuration).
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax.models import canonical_d

A = 3.5
BASE = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def supercell(n):
    grid = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    pos = A * (grid[:, None, :] + BASE[None, :, :]).reshape(-1, 3)
    pos = pos + np.random.default_rng(1).normal(scale=0.05, size=pos.shape)
    return pos, A * n * np.eye(3)


def best(fn, *args, repeat=3):
    jax.block_until_ready(fn(*args))  # compile
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return min(times)


def main(n_max=3):
    params = canonical_d(hubbard=8.0)
    print("atoms  cluster  topology(s)  energy+force(s)  scf+force(s)  U/atom")
    for n in range(1, n_max + 1):
        pos, cell = supercell(n)
        natom = len(pos)
        cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged")
        t0 = time.perf_counter()
        topo = bj.build_topology(pos, [0] * natom, cfg.rcut, cfg.nmom, cell)
        t_topo = time.perf_counter() - t0
        energy = bj.make_energy_fn(topo, cfg)
        p, c = jnp.array(pos), jnp.array(cell)
        zero = jnp.zeros(energy.shift_shape)

        fixed = jax.jit(jax.value_and_grad(lambda r: energy(zero, r, c, params)[0]))
        potential = bj.make_potential(energy, tol=1e-8)
        scf = jax.jit(jax.value_and_grad(lambda r: potential(r, c, params)[0]))
        t_fixed, t_scf = best(fixed, p), best(scf, p, repeat=1)
        u = float(scf(p)[0]) / natom
        print(f"{natom:5d}  {topo.cluster_sizes[-1]:7d}  {t_topo:11.2f}  {t_fixed:15.3f}  "
              f"{t_scf:12.2f}  {u:.10f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
