# Visualisation with OVITO

`bopjax.ovito.Trajectory` collects the frames of an `md`, `spin` or
`spinlattice` run and writes them as extended XYZ, which
[OVITO](https://www.ovito.org) opens directly. Spin moments (`moment * axes`),
forces, velocities and any further per-atom array become particle properties
that OVITO can draw as arrows. With the `ovito` Python module (the `viz`
extra) it also renders pictures and animations with the arrows set up:

```python
from bopjax import ovito as bov

traj = bov.Trajectory(cell, names=("Fe",))
for step in range(steps):
    state = apply(state)
    traj.append(state, time=step * dt_fs, orbital_moment=L)   # extra per-atom data
traj.write("run.xyz")                                          # NumPy only
traj.render("run.gif", arrows={"spin_moment": bov.SPIN, "orbital_moment": bov.ORBITAL},
            label="t = {time:.1f} fs", replicate=(2, 2, 2), exaggerate=10.0)
```

`exaggerate` scales the displacements from the first frame, `label` fills in
global quantities of each frame, and `pipeline()` returns the OVITO pipeline
for further modifiers. The GIFs in the README are made by OVITO's Tachyon
renderer (`renderer="opengl"` is about fifteen times faster) and assembled
with Pillow. `--from-xyz` renders an example again from its saved trajectory.

## Examples

Both animations in the [README](../README.md) come from examples that write
the trajectory to `docs/animations/*.xyz`, which OVITO opens, and render the
GIF next to it. Times are for a 4-core cloud machine, rendering included
(about one minute each).

- `examples/ovito_spin_orbit.py` (3 minutes): spin and orbital moments of a
  chain precessing into the easy axis of [spin-orbit coupling](spin-orbit.md).
- `examples/ovito_spin_lattice.py` (10 minutes): precessing spins pairing the
  planes of a bcc lattice by exchange striction
  ([spin-lattice dynamics](spin-lattice.md)).

## Rendering on a headless machine

The `ovito` module renders without a display, but from OVITO 3.13 on its
renderers need a Vulkan driver. Mesa's software driver is enough
(`apt install mesa-vulkan-drivers`); the GIFs in this repository were
rendered that way.
