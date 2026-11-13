"""Two species with charge transfer, intersite Coulomb, forces and stress.

Run with:  uv run python examples/alloy_stress.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from _cell import CELL, POSITIONS

import bopjax as bj


def channel(aa, bb, ab):
    return jnp.array([[[aa, 2.081, 5.0], [ab, 2.081, 5.0]], [[ab, 2.081, 5.0], [bb, 2.081, 5.0]]])


params = {
    "onsite": jnp.array([[0.0], [0.3]]),  # species 0 and 1
    "n0": jnp.array([[7.0], [4.8]]),
    "hubbard": jnp.array([8.0, 10.0]),
    "bond": {"dds": channel(-1.458, -1.458, -1.3), "ddp": channel(0.972, 0.972, 0.9),
             "ddd": channel(-0.243, -0.243, -0.2)},
    "taper": jnp.broadcast_to(jnp.array([0.0, 5.0]), (2, 2, 2)),  # Coulomb switched off at 5 A
}
cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, rcut2=5.0, bandwidth="averaged",
                   coulomb="frauenheim")
topo = bj.build_topology(POSITIONS, [0, 1, 0, 1], cfg.rcut, cfg.nmom, CELL, rcut2=cfg.rcut2)
energy = bj.make_energy_fn(topo, cfg)

# make_potential puts the self-consistency inside one compiled function, so
# forces and stress are ordinary gradients of it.
potential = jax.jit(bj.make_potential(energy))
pos, cell = jnp.array(POSITIONS), jnp.array(CELL)
e, aux = potential(pos, cell, params)
forces = -jax.grad(lambda p: potential(p, cell, params)[0])(pos)
stress = bj.make_stress(bj.make_potential(energy))(pos, cell, params)

print(f"U/atom         = {float(e) / 4:.10f}   (BOPfox: -5.0959957808)")
print("charges        =", np.array(aux["charge"]).round(6))
print("Coulomb / atom =", float(jnp.mean(aux["energy_coulomb"])))
print("forces (eV/A)\n", np.array(forces).round(6))
print("stress (eV/A^3)\n", np.array(stress).round(6))
