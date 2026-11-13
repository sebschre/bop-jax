"""Distorted fcc cell, canonical d-band model: self-consistent energy and forces.

Run with:  uv run python examples/fcc_distorted.py
The same case run through BOPfox gives U_binding/atom = -6.0112485448437
after 12 Broyden steps.
"""

import time

import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax.models import canonical_d

a = 3.5
cell = a * np.eye(3)
pos = a * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
pos[0] += [0.15, -0.08, 0.05]
pos[2] += [-0.05, 0.10, 0.02]

cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5)
topo = bj.build_topology(pos, [0, 0, 0, 0], cfg.rcut, cfg.nmom, cell)
print(f"cluster of {topo.cluster_sizes[-1]} atoms per atom, {topo.nbr_mask.shape[1]} neighbours")

energy = bj.make_energy_fn(topo, cfg)
scf = bj.make_scf(energy)
forces = bj.make_forces(energy)
params = canonical_d(hubbard=8.0)  # BOPfox default Jii
pos, cell = jnp.array(pos), jnp.array(cell)

result = scf(pos, cell, params, tol=1e-12, verbose=True)  # first call compiles
t0 = time.perf_counter()
result = scf(pos, cell, params, tol=1e-12)
print(f"\nNewton self-consistency: {len(result.history) - 1} steps, "
      f"{time.perf_counter() - t0:.3f} s after compilation")
print(f"U_binding/atom = {result.energy / 4:.13f}")
print("onsite shifts  =", np.array(result.shifts[:, 0]))
print("charges        =", np.array(result.aux["charge"]))
print("forces\n", np.array(forces(result.shifts, pos, cell, params)))
