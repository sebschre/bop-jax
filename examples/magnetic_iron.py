"""Collinear and non-collinear magnetism for a canonical d-band "iron".

Run with:  uv run python examples/magnetic_iron.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from _cell import CELL, POSITIONS

import bopjax as bj
from bopjax.models import canonical_d

STONER = 0.95
params = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[STONER]]))
pos, cell = jnp.array(POSITIONS), jnp.array(CELL)


def solve(spin, start_moments, **extra):
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged", spin=spin)
    topo = bj.build_topology(POSITIONS, [0] * 4, cfg.rcut, cfg.nmom, CELL)
    energy = bj.make_energy_fn(topo, cfg)
    start = None
    if start_moments is not None:
        start = bj.stoner_shifts(np.asarray(start_moments, float)[:, None], STONER)
    par = dict(params, **extra)
    return energy, par, bj.make_scf(energy)(pos, cell, par, shifts0=start, tol=1e-11, max_iter=40)


_, _, nm = solve("none", None)
print(f"non-magnetic       U/atom = {nm.energy / 4:.8f}")
# The solution found depends on the starting moments.
for name, m0 in (("start ferro", [2, 2, 2, 2]), ("start antiferro", [2, 2, -2, -2])):
    _, _, res = solve("collinear", m0)
    print(f"{name:18s} U/atom = {res.energy / 4:.8f}  moments = {np.array(res.aux['moment']).round(4)}")

# Non-collinear: each atom has its own spin axis.  The axes are fixed inputs;
# the energy gradient with respect to them gives the magnetic torques.
axes = jnp.array([[2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [2.0, 1.0, 0.0], [-2.0, 0.5, 0.0]])
energy, par, res = solve("noncollinear", [2, 2, 2, 2], spin_axes=axes)
print(f"{'non-collinear':18s} U/atom = {res.energy / 4:.8f}  moments = {np.array(res.aux['moment']).round(4)}")

unit = axes / jnp.linalg.norm(axes, axis=1, keepdims=True)
grad = jax.grad(lambda e: energy(res.shifts, pos, cell, dict(par, spin_axes=e))[0])(unit)
print("torques -e x dE/de (eV):\n", np.array(-jnp.cross(unit, grad)).round(6))
