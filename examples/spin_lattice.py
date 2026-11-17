"""Coupled spin-lattice dynamics of a bcc d-band ferromagnet (4 atoms).

Atoms and spin axes move together under the two gradients of one
self-consistent energy E(R, e).  The cell is bcc with four (001) planes along
z, and the planes are canted in pairs: planes 0 and 1 by +20 degrees from z,
planes 2 and 3 by -20 degrees.  The lattice starts perfect and at rest.

1. Constant energy.  Planes with parallel moments attract each other more
   than planes with canted ones (exchange striction), so the canting puts
   forces on a lattice that has none when ferromagnetic.  The planes start to
   oscillate about a paired arrangement, while the two groups of spins precess
   about their sum at a rate that follows the distance between the planes.

2. Gilbert damping.  The spins relax to the ferromagnet within tens of
   femtoseconds, faster than the planes can follow, and the forces vanish:
   the lattice is left ringing about the undistorted positions (displacive
   excitation of a coherent phonon), and the energy released by the spins
   that did not go into the damping is in that vibration.

3. Collinear spins on a hot lattice.  Without spin-orbit coupling a collinear
   state has no torques, so the axes do not move at all; the atoms reach the
   spins only through the lengths of the moments, which follow the bond
   lengths.

The model is the canonical d band with 7 electrons, a Stoner integral of
0.6 eV and a pair repulsion ~ r^-16, at the lattice constant where the
ferromagnet (2.37 mu_B) has no pressure to speak of.  It is a toy: the numbers
show the mechanisms, not iron.

Run with:  uv run python examples/spin_lattice.py     (about 7 minutes)
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax import md, spin, spinlattice
from bopjax import pairwise as pw
from bopjax.models import canonical_d

STONER, A, MASS = 0.6, 2.8, 55.85
positions = A * np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1], [0.5, 0.5, 1.5]])
cell = A * np.diag([1.0, 1.0, 2.0])
# The repulsion must be steeper than the square of the bond integrals (r^-10),
# or the bcc lattice is unstable against the pairing of its planes.
params = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[STONER]]),
              repulsion={"pair": jnp.array([[[0.3, 16.0, 2.4]]]),
                         "density": jnp.zeros((1, 1, 3)), "embed": jnp.zeros((1, 2))})
cfg = bj.BOPConfig(nmom=6, rcut=3.6, dcut=0.5, rcut2=3.6, dcut2=0.5, bandwidth="averaged",
                   repulsion=pw.POWERLAW, spin="noncollinear")
topo = bj.build_topology(positions, [0] * 4, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2,
                         skin=0.3)
energy_fn = bj.make_energy_fn(topo, cfg)
solve = bj.make_scf(energy_fn)


def tilted(degrees):
    angle = np.radians(degrees)
    return jnp.array(np.stack([np.sin(angle), 0.0 * angle, np.cos(angle)], axis=1))


def levels(axes):
    """Self-consistent onsite levels of the magnetic solution: the starting
    point of the dynamics, found with the safeguarded solver."""
    return solve(jnp.array(positions), jnp.array(cell), dict(params, spin_axes=axes),
                 shifts0=bj.stoner_shifts(np.full((4, 1), 2.0), STONER))


def integrator(dt_fs, alpha=0.0):
    init, apply = spinlattice.nve(energy_fn, cell, params, dt_fs / md.TIME_UNIT_FS,
                                  alpha=alpha, tol=1e-5)
    return init, jax.jit(apply)


def pairing(state):
    """Change of the distance between planes 0 and 1 (Angstrom); the distance
    between planes 1 and 2 changes by the opposite amount."""
    return float(state.position[1, 2] - state.position[0, 2]) - 0.5 * A


def opening(state):
    """Angle between the spins of the two pairs of planes (degrees)."""
    return float(jnp.degrees(jnp.arccos(jnp.clip(state.axes[0] @ state.axes[2], -1.0, 1.0))))


canted, ferro = tilted([20.0, 20.0, -20.0, -20.0]), tilted([0.0, 0.0, 0.0, 0.0])
start, ground = levels(canted), levels(ferro)
print(f"ferromagnet: E = {ground.energy:.5f} eV, moments {np.array(ground.aux['moment']).round(3)}")
print(f"canted     : E = {start.energy:.5f} eV, moments {np.array(start.aux['moment']).round(3)}"
      f"   ({1e3 * (start.energy - ground.energy):.1f} meV above)")

# --- 1. constant energy ----------------------------------------------------------
dt_fs = 0.25
init, apply = integrator(dt_fs)
state = init(jax.random.PRNGKey(0), jnp.array(positions), canted, mass=MASS,
             shifts0=start.shifts)
print("\nforces along z on the perfect lattice (eV/A):", np.array(state.force[:, 2]).round(4))
print("sum of the torques (eV):", np.array(jnp.sum(spin.torque(state), axis=0)).round(10))
apply(state)  # compile

print("\n1. constant energy\n"
      " t (fs)  E_tot - E0 (eV)  E_kin (meV)  pairing (A)  precession (rad/fs)"
      "  opening (deg)  s/step")
e0, t0 = float(state.energy), time.perf_counter()
for step in range(1, 441):
    before = state
    state = apply(state)
    if step % 20 == 0:
        turn = jnp.arctan2(jnp.cross(before.axes[0], state.axes[0])[2],
                           before.axes[0, :2] @ state.axes[0, :2])
        print(f"{step * dt_fs:7.1f}  {float(state.energy + spinlattice.kinetic_energy(state)) - e0:15.2e}"
              f"  {1e3 * float(spinlattice.kinetic_energy(state)):11.4f}  {pairing(state):11.5f}"
              f"  {abs(float(turn)) / dt_fs:19.4f}  {opening(state):13.2f}"
              f"  {(time.perf_counter() - t0) / step:6.2f}")

# --- 2. Gilbert damping ----------------------------------------------------------
dt_fs, alpha = 0.5, 0.1
dt = dt_fs / md.TIME_UNIT_FS
init, apply = integrator(dt_fs, alpha)
state = init(jax.random.PRNGKey(0), jnp.array(positions), canted, mass=MASS,
             shifts0=start.shifts)
apply(state)

print(f"\n2. Gilbert damping alpha = {alpha}\n"
      " t (fs)  E_pot - E0 (eV)  dissipated (eV)  balance (eV)  E_kin (meV)  pairing (A)"
      "  opening (deg)")
dissipated = 0.0
for step in range(1, 401):
    new = apply(state)
    dissipated += 0.5 * dt * float(spin.damping_power(state, alpha)
                                   + spin.damping_power(new, alpha))
    state = new
    if step % 20 == 0:
        kinetic = float(spinlattice.kinetic_energy(state))
        print(f"{step * dt_fs:7.1f}  {float(state.energy) - e0:15.5f}  {dissipated:15.5f}"
              f"  {float(state.energy) + kinetic + dissipated - e0:12.2e}  {1e3 * kinetic:11.4f}"
              f"  {pairing(state):11.5f}  {opening(state):13.2f}")

# --- 3. collinear spins, atoms at 300 K ------------------------------------------
state = init(jax.random.PRNGKey(1), jnp.array(positions), ferro, mass=MASS,
             kT=300 * md.KB_EV, shifts0=ground.shifts)
print("\n3. ferromagnet, atoms started at 300 K\n"
      " t (fs)    T (K)  largest change of an axis  moments (mu_B)")
for step in range(1, 81):
    state = apply(state)
    if step % 10 == 0:
        print(f"{step * dt_fs:7.1f}  {float(spinlattice.temperature(state)):7.1f}"
              f"  {float(jnp.max(jnp.abs(state.axes - ferro))):25.1e}"
              f"  {np.array(state.moment).round(3)}")
