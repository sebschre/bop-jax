"""A 1/2<111> screw dislocation in bcc tungsten: relaxation and molecular dynamics.

The dislocation runs along z = [111] through a cylinder that is periodic along
z.  Atoms in the outer shell of the cylinder are held fixed at the anisotropic
elastic (Stroh) solution, which uses the lattice constant and the elastic
constants of the potential itself, so the boundary is consistent with the
model.  The core is first relaxed at 0 K (quick-min) and then followed in
constant-energy molecular dynamics, with the onsite levels warm-started.

The core position is the triangle of [111] atomic columns that the
differential displacements enclose a Burgers vector in.

What to expect with this parameter set (a0 = 3.232 A, C11 = 337, C12 = 257,
C44 = 121 GPa, so C' is soft and the anisotropy ratio is 3): the core starts
compact, relaxes by about 0.73 eV per b into a core spread on the (-110)
plane, and keeps that structure, in its initial site, through the molecular
dynamics; the total energy stays within a few meV with 2 fs steps.  Real
tungsten (DFT, and the BOP of Mrovec et al.) has a compact core; the
parameters in ``bopjax.models.tungsten_v0`` are not fitted to it.

705 atoms, 126 of them mobile.  About an hour on a 4-core cloud machine,
most of it in the relaxation.

Needs nothing beyond bopjax.  Run with:
    uv run python examples/screw_dislocation.py [trajectory.xyz]
The optional file receives extended-XYZ frames (for OVITO, ASE, ...): the
perfect cylinder, the elastic solution, the relaxed core, a snapshot every 10
MD steps and the time-averaged positions of the second half of the run.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from _dislocation import ROT, Columns, perfect_cylinder, stroh_displacement

import bopjax as bj
from bopjax import md
from bopjax import pairwise as pw
from bopjax.hamiltonian import v0_bond
from bopjax.models import tungsten_v0

RADIUS = 21.0  # cylinder radius (Angstrom)
# Thickness of the fixed outer shell.  The free surface of the cylinder still
# exerts 0.2 eV/A at 8-10 A depth in a perfect crystal (moments reach three
# hops, and self-consistency further), and less than 0.01 eV/A beyond 12 A.
FIXED = 12.0
LAYERS = 3  # periodic length along the line, in Burgers vectors
# The core first sits near a compact configuration (max |F| ~0.02 eV/A after
# 30 steps) and then spreads slowly on the (-110) plane, so do not stop early.
RELAX_STEPS, FMAX = 200, 0.005  # quick-min steps (5 fs each) and force tolerance (eV/A)
MD_STEPS = 100
DT = 2.0 / md.TIME_UNIT_FS  # 2 fs
T_START = 600.0  # K; about half goes into the potential energy
GPA = 160.21766208  # eV / A^3 in GPa

cfg = bj.BOPConfig(nmom=6, rcut=3.8, dcut=0.5, rcut2=3.8, dcut2=0.5, repulsion=pw.V0)
params = tungsten_v0()


# --- bulk properties of the model -------------------------------------------

def bulk_stress_fn():
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    topo = bj.build_topology(3.2 * frac, [0, 0], cfg.rcut, cfg.nmom, 3.2 * np.eye(3),
                             rcut2=cfg.rcut2, skin=0.5)
    stress = jax.jit(bj.make_stress(bj.make_potential(bj.make_energy_fn(topo, cfg,
                                                                        radial=v0_bond))))

    def sigma(a, strain=np.zeros((3, 3))):
        deform = np.eye(3) + strain
        return np.asarray(stress(jnp.array(a * frac @ deform), jnp.array(a * deform), params))

    return sigma


def lattice_and_elastic_constants():
    """Zero-pressure lattice constant, then C_ijkl from central differences of
    the stress (the stress itself is exact; its strain derivative needs the
    response of the onsite levels, which finite differences include)."""
    sigma = bulk_stress_fn()
    a0, a1 = 3.15, 3.20
    p0, p1 = np.trace(sigma(a0)), np.trace(sigma(a1))
    while abs(a1 - a0) > 1e-9:  # secant on the pressure
        a0, a1, p0 = a1, a1 - p1 * (a1 - a0) / (p1 - p0), p1
        p1 = np.trace(sigma(a1))
    h, c = 1e-4, np.zeros((3, 3, 3, 3))
    for k in range(3):
        for m in range(k, 3):
            e = np.zeros((3, 3))
            e[k, m] += h / 2
            e[m, k] += h / 2
            c[:, :, k, m] = c[:, :, m, k] = (sigma(a1, e) - sigma(a1, -e)) / (2 * h)
    return a1, c


def write_xyz(f, pos, cell, mobile, label):
    lattice = " ".join(f"{v:.6f}" for v in np.asarray(cell).ravel())
    f.write(f'{len(pos)}\nLattice="{lattice}" Properties=species:S:1:pos:R:3:fixed:I:1 '
            f'pbc="F F T" frame={label}\n')
    for r, m in zip(np.asarray(pos), mobile):
        f.write(f"W {r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {int(not m)}\n")


# --- main ----------------------------------------------------------------------

t_start = time.perf_counter()
a, c_cubic = lattice_and_elastic_constants()
b = a * np.sqrt(3) / 2
print(f"model: a0 = {a:.4f} A, C11 = {c_cubic[0, 0, 0, 0] * GPA:.1f}, "
      f"C12 = {c_cubic[0, 0, 1, 1] * GPA:.1f}, C44 = {c_cubic[1, 2, 1, 2] * GPA:.1f} GPa")

c_rot = np.einsum("ia,jb,kc,ld,abcd->ijkl", ROT, ROT, ROT, ROT, c_cubic)
u_elastic = stroh_displacement(c_rot, np.array([0.0, 0.0, b]))

length = LAYERS * b
ref = perfect_cylinder(a, RADIUS, length)
cols = Columns(ref, a, b)
centre = cols.easy_core_site()
positions = ref + u_elastic(ref[:, 0] - centre[0], ref[:, 1] - centre[1])
mobile = np.hypot(*(ref[:, :2] - centre).T) < RADIUS - FIXED
# lateral cell vectors longer than the cylinder plus the cutoff: periodic along z only
side = 2 * RADIUS + 2 * (cfg.rcut + 1.0)
cell = np.diag([side, side, length])
print(f"{len(ref)} atoms, {mobile.sum()} mobile, {LAYERS} b = {length:.3f} A along [111], "
      f"core site at ({centre[0]:.3f}, {centre[1]:.3f})")
u_inplane = u_elastic(ref[:, 0] - centre[0], ref[:, 1] - centre[1])[:, :2]
print(f"largest in-plane displacement of the anisotropic solution: "
      f"{np.abs(u_inplane).max():.3f} A")


def report_core(pos, label):
    sites, content = cols.core(np.asarray(pos)[:, 2] - ref[:, 2])
    desc = ", ".join(f"({x:+.2f}, {y:+.2f}) {q:+.2f} b" for (x, y), q in zip(sites - centre,
                                                                               content))
    print(f"{label}: core triangle(s) relative to the initial site: {desc or 'none'}")


def build(pos, dt=DT):
    # third neighbours are at 4.57 A, so a skin of 0.6 A adds no neighbours
    topo = bj.build_topology(np.asarray(pos), [0] * len(pos), cfg.rcut, cfg.nmom, cell,
                             rcut2=cfg.rcut2, skin=0.6)
    init, apply = md.nve(bj.make_energy_fn(topo, cfg, radial=v0_bond), cell, params, dt,
                         mobile=mobile)
    return topo, init, jax.jit(apply)


report_core(positions, "elastic solution")
traj = open(sys.argv[1], "w") if len(sys.argv) > 1 else None
if traj:
    write_xyz(traj, ref, cell, mobile, "perfect")
    write_xyz(traj, positions, cell, mobile, "elastic")
topo, init, apply = build(positions, dt=5.0 / md.TIME_UNIT_FS)

# quick-min: velocity Verlet, keeping only the velocity along the force
print("\nrelaxation (quick-min)\n step   E_pot (eV)    max |F| (eV/A)  Newton steps")
state = init(jax.random.PRNGKey(0), jnp.array(positions), mass=183.84, kT=0.0)
e_elastic = float(state.energy)
for step in range(1, RELAX_STEPS + 1):
    state = apply(state)
    f = state.force.ravel()
    power = jnp.dot(state.momentum.ravel(), f)
    momentum = jnp.where(power > 0, power * f / jnp.dot(f, f), 0.0).reshape(f.shape[0] // 3, 3)
    state = state._replace(momentum=momentum)
    if bj.needs_rebuild(state.position, topo):
        topo, init, apply = build(state.position, dt=5.0 / md.TIME_UNIT_FS)
    fmax = float(jnp.max(jnp.linalg.norm(state.force, axis=1)))
    if step % 10 == 0 or fmax < FMAX:
        print(f"{step:5d}  {float(state.energy):12.5f}  {fmax:12.4f}  "
              f"{int(state.scf_iterations):10d}")
    if fmax < FMAX:
        break
print(f"core relaxation energy: {float(state.energy) - e_elastic:.4f} eV "
      f"({(float(state.energy) - e_elastic) / LAYERS:.4f} eV per b)")
report_core(state.position, "relaxed")
relaxed = np.asarray(state.position)
if traj:
    write_xyz(traj, relaxed, cell, mobile, "relaxed")

# constant-energy molecular dynamics from the relaxed core, with fresh lists
topo, init, apply = build(relaxed)
state = init(jax.random.PRNGKey(1), state.position, mass=183.84, kT=T_START * md.KB_EV)
e0 = float(state.energy + md.kinetic_energy(state))
print(f"\nNVE, dt = {DT * md.TIME_UNIT_FS:.0f} fs\n"
      " step   E_tot - E_0 (eV)   T (K)   Newton steps   core triangle(s)")
t0, positions_sum = time.perf_counter(), np.zeros_like(relaxed)
for step in range(1, MD_STEPS + 1):
    state = apply(state)
    if bj.needs_rebuild(state.position, topo):
        topo, _, apply = build(state.position)
        print("       (neighbour lists rebuilt)")
    if step > MD_STEPS // 2:
        positions_sum += np.asarray(state.position)
    if step % 10 == 0:
        sites, _ = cols.core(np.asarray(state.position)[:, 2] - ref[:, 2])
        where = " ".join(f"({x:+.2f}, {y:+.2f})" for x, y in sites - centre)
        e_tot = float(state.energy + md.kinetic_energy(state))
        print(f"{step:5d}  {e_tot - e0:16.6f}  {float(md.temperature(state)):7.1f}  "
              f"{int(state.scf_iterations):11d}     {where}")
        if traj:
            write_xyz(traj, state.position, cell, mobile, f"md{step}")
print(f"{1e3 * (time.perf_counter() - t0) / MD_STEPS:.0f} ms per MD step")
average = positions_sum / (MD_STEPS - MD_STEPS // 2)
report_core(average, "time average, second half")
if traj:
    write_xyz(traj, average, cell, mobile, "average")
    traj.close()
print(f"total {time.perf_counter() - t_start:.0f} s")
