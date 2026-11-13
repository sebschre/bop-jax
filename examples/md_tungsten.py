"""Constant-energy molecular dynamics of bcc tungsten with JAX-MD.

Needs the optional dependency:  uv sync --extra md
Run with:  uv run python examples/md_tungsten.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import quantity, simulate

import bopjax as bj
from bopjax import md
from bopjax import pairwise as pw
from bopjax.hamiltonian import v0_bond
from bopjax.models import tungsten_v0

a = 3.165
cell = a * np.eye(3)
positions = a * np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
cfg = bj.BOPConfig(nmom=6, rcut=3.8, dcut=0.5, rcut2=3.8, dcut2=0.5, repulsion=pw.V0)
params = tungsten_v0()
_, shift = md.jaxmd_space()  # free space: bopjax handles the periodic images


def build(pos):
    """Neighbour lists are static, so the energy function is rebuilt on demand."""
    topo = bj.build_topology(np.asarray(pos), [0, 0], cfg.rcut, cfg.nmom, cell,
                             rcut2=cfg.rcut2, skin=0.3)
    energy = md.jaxmd_energy_fn(bj.make_energy_fn(topo, cfg, radial=v0_bond), cell, params)
    _, apply = simulate.nve(energy, shift, dt)
    return topo, jax.jit(energy), jax.jit(apply)


dt = 1.0 / md.TIME_UNIT_FS  # 1 fs
topo, energy, apply = build(positions)
init, _ = simulate.nve(energy, shift, dt)
state = init(jax.random.PRNGKey(0), jnp.array(positions), mass=183.84, kT=600 * md.KB_EV)

print(" step   E_pot (eV)    E_kin (eV)    E_tot (eV)    T (K)")
for step in range(101):
    if step % 10 == 0:
        e_pot = float(energy(state.position))
        e_kin = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
        temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass)) / md.KB_EV
        print(f"{step:5d}  {e_pot:12.6f}  {e_kin:12.6f}  {e_pot + e_kin:12.6f}  {temp:7.1f}")
    state = apply(state)
    if bj.needs_rebuild(state.position, topo):
        topo, energy, apply = build(state.position)
