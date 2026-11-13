"""A 1/2<111> screw dislocation in bcc tungsten: relaxation and molecular dynamics.

The dislocation runs along z = [111] through a cylinder that is periodic along
z.  Atoms in the outer shell of the cylinder are held fixed at the anisotropic
elastic (Stroh) solution, which uses the lattice constant and the elastic
constants of the potential itself, so the boundary is consistent with the
model.  The core is first relaxed at 0 K (quick-min) and then followed in
constant-energy molecular dynamics, with the onsite levels warm-started.

The core position is the triangle of [111] atomic columns that the
differential displacements enclose a Burgers vector in.

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

import bopjax as bj
from bopjax import md
from bopjax import pairwise as pw
from bopjax.hamiltonian import v0_bond
from bopjax.models import tungsten_v0

RADIUS = 18.0  # cylinder radius (Angstrom)
FIXED = 7.6  # thickness of the fixed outer shell, two bond cutoffs
LAYERS = 3  # periodic length along the line, in Burgers vectors
RELAX_STEPS, MD_STEPS = 60, 100
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


# --- anisotropic elasticity --------------------------------------------------

def stroh_displacement(c, burgers):
    """Displacement field u(x, y) of a straight dislocation along z in an
    anisotropic medium with elastic tensor c (Stroh sextic formalism):

        u = (1/pi) Im sum_a A_a ln(x + p_a y) (L_a . b),   2 A_a . L_a = 1

    with the branch cut along the negative x axis."""
    q, r, t = c[:, 0, :, 0], c[:, 0, :, 1], c[:, 1, :, 1]
    ti = np.linalg.inv(t)
    n = np.block([[-ti @ r.T, ti], [r @ ti @ r.T - q, -r @ ti]])
    p, xi = np.linalg.eig(n)
    keep = np.argsort(-p.imag)[:3]  # the three roots in the upper half plane
    p, xi = p[keep], xi[:, keep]
    if np.min(np.abs(p[:, None] - p[None, :]) + 10 * np.eye(3)) < 1e-6:
        raise ValueError("degenerate Stroh roots (isotropic medium)")
    xi = xi / np.sqrt(2 * np.sum(xi[:3] * xi[3:], axis=0))
    a_mat, l_mat = xi[:3], xi[3:]
    coef = l_mat.T @ burgers

    def u(x, y):
        z = np.asarray(x)[..., None] + p * np.asarray(y)[..., None]
        return np.imag((np.log(z) * coef) @ a_mat.T) / np.pi

    def stress_function(x, y):  # the traction across a curve is its increment
        z = np.asarray(x)[..., None] + p * np.asarray(y)[..., None]
        return np.imag((np.log(z) * coef) @ l_mat.T) / np.pi

    # the closure failure is b and the core carries no net force
    th = np.array([-np.pi + 1e-12, np.pi - 1e-12])
    assert np.allclose(np.diff(u(np.cos(th), np.sin(th)), axis=0), burgers, atol=1e-8)
    assert np.allclose(np.diff(stress_function(np.cos(th), np.sin(th)), axis=0), 0, atol=1e-8)
    return u


# --- geometry and core analysis ----------------------------------------------

# x = [11-2], y = [-110] (normal of the (-110) glide plane), z = [111]
ROT = np.array([np.array([1, 1, -2]) / np.sqrt(6), np.array([-1, 1, 0]) / np.sqrt(2),
                np.array([1, 1, 1]) / np.sqrt(3)])


def perfect_cylinder(a, radius, length):
    n = int(np.ceil((radius + length) / a)) + 2
    g = np.stack(np.meshgrid(*[np.arange(-n, n + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    pts = a * np.concatenate([g, g + 0.5]) @ ROT.T
    rho = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[(rho < radius) & (pts[:, 2] > -1e-6) & (pts[:, 2] < length - 1e-6)]
    return pts[np.lexsort((pts[:, 2], np.hypot(pts[:, 0], pts[:, 1])))]


class Columns:
    """[111] atomic columns of the perfect cylinder and their triangles."""

    def __init__(self, ref, a, b):
        self.b = b
        key = np.round(ref[:, :2], 3)
        self.xy, self.of_atom = np.unique(key, axis=0, return_inverse=True)
        self.of_atom = self.of_atom.ravel()
        self.height = np.zeros(len(self.xy))
        self.height[self.of_atom] = np.mod(ref[:, 2], b)
        d = np.linalg.norm(self.xy[:, None] - self.xy[None], axis=-1)
        near = np.abs(d - a * np.sqrt(2 / 3)) < 0.1  # first neighbours in projection
        tri = set()
        for i in range(len(self.xy)):
            nb = np.nonzero(near[i])[0]
            for j in nb:
                for k in nb:
                    if j < k and near[j, k]:
                        tri.add(tuple(sorted((i, j, k))))
        tri = np.array(sorted(tri))
        # order counter-clockwise
        e1, e2 = self.xy[tri[:, 1]] - self.xy[tri[:, 0]], self.xy[tri[:, 2]] - self.xy[tri[:, 0]]
        cw = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 0
        tri[cw] = tri[cw][:, [0, 2, 1]]
        self.triangles = tri
        self.centroid = self.xy[tri].mean(axis=1)

    def easy_core_site(self):
        """Centroid of the triangle next to the axis whose column heights rise
        by b/3 counter-clockwise: the elastic field b theta / 2 pi reverses the
        helicity of that triangle, which is the compact (easy) core."""
        h = self.height[self.triangles] / self.b
        rising = np.all(np.isclose(np.mod(np.roll(h, -1, axis=1) - h, 1), 1 / 3, atol=1e-3),
                        axis=1)
        dist = np.linalg.norm(self.centroid, axis=1) + 1e3 * ~rising
        return self.centroid[np.argmin(dist)]

    def burgers_content(self, displacement_z):
        """Burgers vector (in units of b) enclosed by every triangle, from the
        differential displacements of the column-averaged u_z."""
        n = np.bincount(self.of_atom)
        u = np.bincount(self.of_atom, weights=displacement_z) / n
        t = self.triangles
        dd = u[np.roll(t, -1, axis=1)] - u[t]
        dd = dd - self.b * np.round(dd / self.b)  # differential displacement in [-b/2, b/2]
        return np.sum(dd, axis=1) / self.b

    def core(self, displacement_z):
        content = self.burgers_content(displacement_z)
        inside = np.abs(content) > 0.5
        return self.centroid[inside], content[inside]


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


def build(pos):
    topo = bj.build_topology(np.asarray(pos), [0] * len(pos), cfg.rcut, cfg.nmom, cell,
                             rcut2=cfg.rcut2, skin=0.4)
    init, apply = md.nve(bj.make_energy_fn(topo, cfg, radial=v0_bond), cell, params, DT,
                         mobile=mobile)
    return topo, init, jax.jit(apply)


report_core(positions, "elastic solution")
traj = open(sys.argv[1], "w") if len(sys.argv) > 1 else None
if traj:
    write_xyz(traj, ref, cell, mobile, "perfect")
    write_xyz(traj, positions, cell, mobile, "elastic")
topo, init, apply = build(positions)

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
        topo, init, apply = build(state.position)
    fmax = float(jnp.max(jnp.linalg.norm(state.force, axis=1)))
    if step % 10 == 0 or fmax < 0.01:
        print(f"{step:5d}  {float(state.energy):12.5f}  {fmax:12.4f}  "
              f"{int(state.scf_iterations):10d}")
    if fmax < 0.01:
        break
print(f"core relaxation energy: {float(state.energy) - e_elastic:.4f} eV "
      f"({(float(state.energy) - e_elastic) / LAYERS:.4f} eV per b)")
report_core(state.position, "relaxed")
relaxed = np.asarray(state.position)
if traj:
    write_xyz(traj, relaxed, cell, mobile, "relaxed")

# constant-energy molecular dynamics from the relaxed core
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
