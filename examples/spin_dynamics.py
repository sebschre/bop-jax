"""Spin dynamics of a non-collinear canonical d-band "iron" (4 atoms).

The spin axes move under the torques of the self-consistent energy
(Landau-Lifshitz-Gilbert equation) while the atoms stay where they are:
first without damping, where the energy is conserved, then with damping,
where the energy falls at the rate the damping term dissipates it.

The damped run is short on purpose.  This model relaxes towards a ferromagnet
with a completely filled spin-up band (m = 3), and close to that saturation
the self-consistency solver fails (see "Caveats" in the README).

Run with:  uv run python examples/spin_dynamics.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from _cell import CELL, POSITIONS

import bopjax as bj
from bopjax import md, spin
from bopjax.models import canonical_d

STONER = 0.95
params = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[STONER]]))
cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged", spin="noncollinear")
topo = bj.build_topology(POSITIONS, [0] * 4, cfg.rcut, cfg.nmom, CELL)
energy_fn = bj.make_energy_fn(topo, cfg)

axes = jnp.array([[2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [2.0, 1.0, 0.0], [-2.0, 0.5, 0.0]])
# the starting levels select the magnetic solution
start = bj.stoner_shifts(np.full((4, 1), 2.0), STONER)


def run(title, dt_fs, steps, every, alpha=0.0):
    dt = dt_fs / md.TIME_UNIT_FS
    init, apply = spin.llg(energy_fn, CELL, params, dt, alpha=alpha)
    init, apply = jax.jit(init), jax.jit(apply)
    state = init(jnp.array(POSITIONS), axes, start)
    apply(state)  # compile
    e0, dissipated = float(state.energy), 0.0
    print(f"\n{title}\n t (fs)   E - E0 (eV)  -dissipated (eV)  max torque (eV)   |M| (mu_B)"
          "   Newton steps   s/step")
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        new = apply(state)
        dissipated += 0.5 * dt * float(spin.damping_power(state, alpha)
                                       + spin.damping_power(new, alpha))
        state = new
        if step % every == 0:
            print(f"{step * dt_fs:7.1f}  {float(state.energy) - e0:12.3e}  {-dissipated:16.3e}  "
                  f"{float(jnp.max(jnp.abs(spin.torque(state)))):15.6f}  "
                  f"{float(jnp.linalg.norm(spin.magnetisation(state))):11.4f}  "
                  f"{str(np.array(state.scf_iterations)):>13s}  "
                  f"{(time.perf_counter() - t0) / step:7.2f}")
    return state


run("precession, no damping", dt_fs=0.2, steps=20, every=5)
state = run("Gilbert damping alpha = 1", dt_fs=0.5, steps=16, every=4, alpha=1.0)
print("\naxes after damping:\n", np.array(state.axes).round(4))
print("moments (mu_B):", np.array(state.moment).round(4))
