# Molecular dynamics

```python
from jax_md import simulate
from bopjax import md

energy = md.jaxmd_energy_fn(bj.make_energy_fn(topo, cfg), cell, params)
_, shift = md.jaxmd_space()
init, apply = simulate.nve(energy, shift, dt=1.0 / md.TIME_UNIT_FS)
```

Periodic images are handled by bopjax, so positions must not be wrapped: use
the shift from `jaxmd_space()`. Neighbour lists are static; build the topology
with a `skin` and rebuild when `bj.needs_rebuild(positions, topo)` is true (see
`examples/md_tungsten.py`).

**Warm start.** Onsite levels change little between MD steps, so starting the
solver from the previous ones saves most of its work. A JAX-MD integrator
cannot carry them in its state, so `bopjax.md.nve` is a velocity-Verlet
integrator that does, and extrapolates them quadratically to the next step:

```python
init, apply = md.nve(energy_fn, cell, params, dt=1.0 / md.TIME_UNIT_FS)
state = init(key, positions, mass=183.84, kT=600 * md.KB_EV)
state = apply(state)          # state.scf_iterations: Newton steps taken
```

For 54 tungsten atoms this takes one Newton step per MD step instead of two
to three, 183 ms against 440 ms, with the same trajectory and energy
conservation (`examples/md_warm_start.py`). With JAX-MD's own integrators,
pass `shifts0=` to `apply_fn` and obtain it from `md.jaxmd_shifts_fn`, at the
price of a second, already converged solve.

**Fixed atoms.** `md.nve(..., mobile=mask)` holds the atoms outside the mask
in place (zero momentum, force ignored) while their onsite levels stay
self-consistent. `examples/screw_dislocation.py` uses it for a 1/2[111] screw
dislocation in a cylinder that is periodic along the line, with the outer
12 A fixed at the anisotropic (Stroh) elastic solution for the model's own
elastic constants. It relaxes the core and runs NVE dynamics, following the
core through the differential displacements. The fixed shell must be thick:
the free surface still exerts 0.2 eV/A on atoms 8 to 10 A below it.
