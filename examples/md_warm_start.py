"""Molecular dynamics with warm-started self-consistency (54 tungsten atoms).

bopjax.md.nve is a velocity-Verlet integrator that keeps the onsite levels in
its state and extrapolates them to the next step, so the Newton solver usually
needs a single step.  Run with:  uv run python examples/md_warm_start.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax import md
from bopjax import pairwise as pw
from bopjax.hamiltonian import v0_bond
from bopjax.models import tungsten_v0

n, a = 3, 3.165
grid = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
positions = a * (grid[:, None, :] + np.array([[0, 0, 0], [0.5, 0.5, 0.5]])[None]).reshape(-1, 3)
cell = a * n * np.eye(3)

cfg = bj.BOPConfig(nmom=6, rcut=3.8, dcut=0.5, rcut2=3.8, dcut2=0.5, repulsion=pw.V0)
params = tungsten_v0()
dt = 1.0 / md.TIME_UNIT_FS  # 1 fs


def build(pos):
    """Neighbour lists and clusters are static, so they are rebuilt (and the
    step function recompiled) once an atom has moved more than skin / 2."""
    topo = bj.build_topology(np.asarray(pos), [0] * len(pos), cfg.rcut, cfg.nmom, cell,
                             rcut2=cfg.rcut2, skin=0.5)
    init, apply = md.nve(bj.make_energy_fn(topo, cfg, radial=v0_bond), cell, params, dt)
    return topo, init, jax.jit(apply)


topo, init, apply = build(positions)
state = init(jax.random.PRNGKey(0), jnp.array(positions), mass=183.84, kT=600 * md.KB_EV)
apply(state)  # compile

print(" step   E_tot (eV)      T (K)   Newton steps   ms/step")
t0 = time.perf_counter()
for step in range(1, 51):
    state = apply(state)
    if bj.needs_rebuild(state.position, topo):
        topo, _, apply = build(state.position)  # the state, with its levels, carries over
        print("       (neighbour lists rebuilt)")
    if step % 10 == 0:
        ms = 1e3 * (time.perf_counter() - t0) / step
        e_tot = float(state.energy + md.kinetic_energy(state))
        print(f"{step:5d}  {e_tot:12.6f}  {float(md.temperature(state)):8.1f}  "
              f"{int(state.scf_iterations):13d}  {ms:8.1f}")
