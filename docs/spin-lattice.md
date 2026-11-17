# Spin-lattice dynamics

`bopjax.spinlattice` moves atoms and spin axes together. Both follow the same
self-consistent energy `E(R, e)`: its gradient with respect to the positions
is the force, which depends on the spins, and its gradient with respect to the
axes is the torque, which depends on the positions.

```python
from bopjax import md, spinlattice

init, apply = spinlattice.nve(energy_fn, cell, params, dt=0.25 / md.TIME_UNIT_FS)
state = init(key, positions, axes, mass=55.85, kT=300 * md.KB_EV, shifts0=shifts)
state = jax.jit(apply)(state)
state.position, state.momentum, state.axes, state.moment, state.energy
```

Without damping `E + E_kin` is conserved while energy moves between the
lattice and the spin directions. `alpha` and `kT` couple the spins to a bath
as in `bopjax.spin` (the atoms have no thermostat), and `mobile=` and `free=`
hold atoms and axes fixed. `spin.torque`, `spin.damping_power` and
`spin.magnetisation` accept the state.

**No spin-orbit coupling is needed for this, and the model has none.** The
coupling is exchange striction: the bond integrals depend on distance, and
with them the energy two atoms gain from aligning their moments. What is
missing without spin-orbit coupling is a preferred direction in space for the
spins. The energy does not change when all axes are rotated together, so:

- the torques sum to zero and the forces exert no torque on the lattice:
  energy is exchanged between spins and lattice, angular momentum is not. The
  magnetisation cannot relax into the lattice (no Einstein-de Haas effect, no
  demagnetisation by phonons), and there is no magnetic anisotropy;
- a collinear state has no torques whatever the atoms do. The lattice then
  reaches the spins only through the lengths of the moments. Transverse
  dynamics needs a non-collinear start or a spin bath (`kT > 0`).

[Spin-orbit coupling](spin-orbit.md) lifts these restrictions, but the
coupled integrator has not been run with it.

**Integrator.** Half kick, half drift, spin step, half drift, half kick: the
atoms follow velocity Verlet and the axes take one midpoint step at the
positions of the middle of the step. The midpoint rule is implicit and is
iterated `correctors` times (default 3); a step costs `correctors + 1`
self-consistent evaluations. One pass is an explicit second-order scheme at
the cost of `spin.llg`; three passes are close to the time-reversible limit,
whose energy error stays bounded instead of growing. The time step is set by
the precession. The solver inside takes plain Newton steps, so `shifts0`
should come from `make_scf` at the starting configuration.

`examples/spin_lattice.py` runs a bcc ferromagnet of four (001) planes (a toy:
canonical d band, 7 electrons, Stoner integral 0.6 eV, repulsion ~ r^-16,
moments of 2.37 Bohr magnetons) with the planes canted in pairs by +20 and
-20 degrees, 56 meV above the ferromagnet, on a perfect lattice at rest:

- The canting alone puts 0.054 eV/A on the atoms, which the ferromagnet does
  not: planes with parallel moments attract. At constant energy the planes
  oscillate about a paired arrangement (0.011 A peak to peak, period 105 fs),
  and the precession of the two groups of spins about their sum slows from
  0.404 to 0.396 rad/fs and back in step with the distance between the
  planes. The total energy changes by 2e-6 eV in 110 fs at a 0.25 fs step,
  against 0.3 meV that goes in and out of the kinetic energy.
- With `alpha = 0.1` the spins relax towards the ferromagnet within tens of
  femtoseconds, the forces vanish, and the lattice is left ringing about the
  undistorted positions with 0.0026 A (displacive excitation of a coherent
  phonon). Of the 56 meV the spins release, 0.08 meV ends in that vibration;
  the balance with the integral of `spin.damping_power` closes to 0.5 meV at
  the 0.5 fs step used.
- A ferromagnet on a lattice started at 300 K: the axes do not change in the
  last digit, while the moments move between 2.09 and 2.44 Bohr magnetons with
  the bond lengths.

The run takes about 7 minutes on an M1 Pro (0.45 s per step of four
evaluations; most of it is the dense Hessian of the self-consistency).

What is checked (`tests/test_spinlattice.py`): for a ring of spins whose
exchange depends on the bond length, conservation of energy, momentum and
total spin while energy is exchanged, second-order convergence, agreement
with `spin.integrator` for fixed atoms and with velocity Verlet for fixed
axes, and that collinear spins stay put on a moving lattice; for the BOP
energy, the exchange-striction forces against finite differences of the
self-consistent energy and energy conservation. Damping and thermal torques
are not tested in the coupled integrator beyond what `tests/test_spin.py`
covers for the shared equation of motion.

Limits: one time step for atoms and spins, set by the precession, so
picoseconds are expensive and the equilibration of spin and lattice
temperatures in a large cell is out of reach here; the total spin is conserved
only as far as the lengths of the moments stay constant, since those follow
adiabatically; the repulsion of a model must be steeper than the square of its
bond integrals or the lattice can be unstable (with the canonical r^-5 bond
integrals and a repulsion ~ r^-10 the bcc planes paired up spontaneously; in
the second-moment picture that case is exactly marginal).
