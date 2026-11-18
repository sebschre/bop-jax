"""First-order effect of a magnetic field on the Peierls barrier of a 1/2<111>
screw dislocation in bcc iron.

For any configuration X the self-consistent energy in a field B is

    E(X, B) = E(X, 0) - mu_B B.M(X) + O(B^2),     M = sum_i (g/2 <sigma>_i + <L>_i),

and since minima and saddle points are stationary in the positions, the
barrier changes to first order by

    dE_P / dB = -mu_B [M(saddle) - M(minimum)]

without relaxing anything in the field (``bopjax.zeeman``).  This script

1. builds the iron model's lattice constant and elastic constants,
2. puts a screw dislocation along z = [111] into a cylinder that is periodic
   along the line with one Burgers vector, the outer shell held at the
   anisotropic elastic solution centred midway between two easy-core sites
   one a sqrt(2/3) apart along x = [11-2] (glide on the (-110) plane),
3. relaxes the two easy cores and finds the zero-field Peierls path between
   them with the climbing-image NEB (``bopjax.neb``), collinear magnetism,
4. evaluates the magnetic moment of every image, the first-order barrier
   shift and the first-order change of the Peierls stress (largest slope of
   the profile over b and the line length), and
5. checks the first-order barrier shift against self-consistent energies of
   the end point and the saddle in fields of +-B_CHECK along z.

With ``--xi`` (d-shell spin-orbit constant in eV, about 0.05 for Fe) the
moments of every image are re-evaluated non-collinearly with spin-orbit
coupling, all spin axes along the field, for fields along the line, the glide
direction and the glide-plane normal.  This adds the orbital moment and the
direction dependence; the magnetic anisotropy of the BOP is not converged
with the number of moments (docs/spin-orbit.md), so that part is only
indicative.

The magnetisation of the collinear model is along z (the line direction); it
is isotropic without spin-orbit coupling, so only the field component along
the magnetisation matters there.

The onsite levels are self-consistent in the mobile region and a buffer
around it (--free-buffer) and frozen at the bulk values further out.

Status: this script has not been run to completion.  On a 4-core cloud
machine (no GPU) the self-consistency of the 9-moment iron model in a
cylinder that is one b long is the obstacle: one energy and force of 61
atoms costs about 1 s at fixed levels, and the compiled Newton loop needs
16-20 s per step with Krylov steps; those converge only linearly and stall
near a residual of 1e-2 (without and with the frozen outer shell), while
undamped dense Newton steps diverge, and the damped solver of ``make_scf``
needs more than 15 GB for 122 unknowns.  Expect to need a GPU, a more robust
(damped, preconditioned) solver inside ``make_potential``, or a fully
periodic quadrupole cell without free surfaces.  The tools it uses
(``bopjax.zeeman``, ``bopjax.neb``) are tested on small systems in
tests/test_zeeman.py.

    uv run python examples/magnetoplasticity.py [--basis sd] [--xi 0.05] [--out profile.npz]
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from _dislocation import ROT, Columns, perfect_cylinder, stroh_displacement

import bopjax as bj
from bopjax import neb, zeeman
from bopjax.hamiltonian import v0_bond
from bopjax.models import IRON_CONFIG, iron
from bopjax.zeeman import G_SPIN, MU_B

GPA = 160.21766208  # eV / A^3 in GPa

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--basis", default="d", choices=("d", "sd"))
parser.add_argument("--radius", type=float, default=20.0, help="cylinder radius (A)")
parser.add_argument("--fixed", type=float, default=10.0, help="fixed outer shell (A)")
parser.add_argument("--condition", default="potential", choices=("potential", "stationary"),
                    help="self-consistency condition (scf.make_potential)")
parser.add_argument("--adjoint-tol", type=float, default=1e-4,
                    help="relative tolerance of the adjoint solve of the gradients "
                         "(--condition potential); 1e-4 gives forces to about 1e-6 eV/A")
parser.add_argument("--free-buffer", type=float, default=None,
                    help="freeze the onsite levels (at bulk values) of fixed atoms farther "
                         "than this from the mobile region (A); default: none frozen")
parser.add_argument("--solver", default="auto", choices=("auto", "dense", "krylov"),
                    help="linear solver of --condition stationary")
parser.add_argument("--images", type=int, default=5, help="images between the end points")
parser.add_argument("--fmax", type=float, default=0.02, help="NEB force tolerance (eV/A)")
parser.add_argument("--steps", type=int, default=400, help="NEB steps")
parser.add_argument("--check-field", type=float, default=100.0,
                    help="field (T) of the self-consistent check of the first-order shift")
parser.add_argument("--xi", type=float, default=0.0,
                    help="d spin-orbit constant (eV) for the orbital moments; 0 to skip")
parser.add_argument("--batch", type=int, default=16,
                    help="atoms whose moments are evaluated at once (memory)")
parser.add_argument("--ends", default=None,
                    help="file (.npz) with relaxed end points: read if present, else written")
parser.add_argument("--out", default=None, help="save the path and moments (.npz)")
args = parser.parse_args()

t_start = time.perf_counter()
basis = ("d",) if args.basis == "d" else ("s", "d")
cfg = bj.BOPConfig(**IRON_CONFIG, basis=basis, spin="collinear", batch=args.batch)
params = iron(basis=basis)
d_shell = np.array([sh == "d" for sh in basis], dtype=float)


def start_shifts(n, moment=2.3):
    return bj.stoner_shifts(moment * np.tile(d_shell, (n, 1)), params["stoner"][0])


# --- bulk: lattice constant and elastic constants of the model -----------------------

def lattice_and_elastic_constants():
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    topo = bj.build_topology(2.85 * frac, [0, 0], cfg.rcut, cfg.nmom, 2.85 * np.eye(3),
                             rcut2=cfg.rcut2, skin=0.5)
    potential = bj.make_potential(bj.make_energy_fn(topo, cfg, radial=v0_bond))
    stress = jax.jit(bj.make_stress(potential))
    s0 = start_shifts(2)

    def sigma(a, strain=np.zeros((3, 3))):
        deform = np.eye(3) + strain
        return np.asarray(stress(jnp.array(a * frac @ deform), jnp.array(a * deform),
                                 params, s0))

    a0, a1 = 2.84, 2.87
    p0, p1 = np.trace(sigma(a0)), np.trace(sigma(a1))
    while abs(a1 - a0) > 1e-9:
        a0, a1, p0 = a1, a1 - p1 * (a1 - a0) / (p1 - p0), p1
        p1 = np.trace(sigma(a1))
    h, c = 1e-4, np.zeros((3, 3, 3, 3))
    for k in range(3):
        for m in range(k, 3):
            e = np.zeros((3, 3))
            e[k, m] += h / 2
            e[m, k] += h / 2
            c[:, :, k, m] = c[:, :, m, k] = (sigma(a1, e) - sigma(a1, -e)) / (2 * h)
    bulk = bj.make_scf(bj.make_energy_fn(topo, cfg, radial=v0_bond))(
        jnp.array(a1 * frac), jnp.array(a1 * np.eye(3)), params, shifts0=s0, tol=1e-12)
    return a1, c, np.asarray(bulk.shifts[0])


a, c_cubic, bulk_shifts = lattice_and_elastic_constants()
b = a * np.sqrt(3) / 2
print(f"model ({args.basis}): a0 = {a:.4f} A, C11 = {c_cubic[0, 0, 0, 0] * GPA:.1f}, "
      f"C12 = {c_cubic[0, 0, 1, 1] * GPA:.1f}, C44 = {c_cubic[1, 2, 1, 2] * GPA:.1f} GPa")
c_rot = np.einsum("ia,jb,kc,ld,abcd->ijkl", ROT, ROT, ROT, ROT, c_cubic)
u_elastic = stroh_displacement(c_rot, np.array([0.0, 0.0, b]))

# --- the cylinder and the two easy-core sites ----------------------------------------

ref = perfect_cylinder(a, args.radius, b)
cols = Columns(ref, a, b)
site_a = cols.easy_core_site()
hop = a * np.sqrt(2.0 / 3.0)
site_b = site_a + np.array([hop, 0.0])
mid = 0.5 * (site_a + site_b)
rho = np.hypot(*(ref[:, :2] - mid).T)
mobile = rho < args.radius - args.fixed
# optionally, onsite levels frozen at the bulk values in the outer part of the
# fixed shell
free = (np.ones(len(ref), bool) if args.free_buffer is None
        else rho < args.radius - args.fixed + args.free_buffer)
side = 2 * args.radius + 2 * (cfg.rcut + 1.0)
cell = np.diag([side, side, b])
n_atoms = len(ref)
print(f"{n_atoms} atoms, {mobile.sum()} mobile, {free.sum()} with self-consistent levels, one b = {b:.4f} A along [111]; easy cores "
      f"at x = {site_a[0]:+.3f} and {site_b[0]:+.3f} A")


def dislocated(site):
    """Elastic solution: the mobile region centred on ``site``, the fixed shell
    on the midpoint so that both end points share it."""
    u_site = u_elastic(ref[:, 0] - site[0], ref[:, 1] - site[1])
    u_mid = u_elastic(ref[:, 0] - mid[0], ref[:, 1] - mid[1])
    return ref + np.where(mobile[:, None], u_site, u_mid)


def core_x(pos):
    """Position of the core along x: the Burgers-content weighted centroid of
    the triangles that enclose it."""
    sites, content = cols.core(np.asarray(pos)[:, 2] - ref[:, 2])
    if len(sites) == 0:
        return np.nan
    return float(np.sum(sites[:, 0] * np.abs(content)) / np.sum(np.abs(content)))


# --- a calculator per image: own neighbour list, warm-started onsite levels ----------

class Image:
    def __init__(self, positions):
        self.shifts = jnp.tile(jnp.asarray(bulk_shifts)[None], (n_atoms, 1, 1))
        self._build(positions)

    def _build(self, positions):
        self.topo = bj.build_topology(np.asarray(positions), [0] * n_atoms, cfg.rcut, cfg.nmom,
                                      cell, rcut2=cfg.rcut2, skin=0.6)
        self.energy_fn = bj.make_energy_fn(self.topo, cfg, radial=v0_bond)
        def value_grad(**options):
            potential = bj.make_potential(self.energy_fn, tol=1e-9, method=args.solver,
                                          free=None if free.all() else free,
                                          condition=args.condition,
                                          adjoint_tol=args.adjoint_tol, **options)
            return potential, jax.jit(jax.value_and_grad(
                lambda p, s0, par: potential(p, jnp.asarray(cell), par, s0), has_aux=True))

        potential, self._value_grad = value_grad()
        self._careful = None  # gentler mixing for a retry, compiled when first needed
        self._careful_options = (dict(mixing=0.03, max_iter=600)
                                 if args.condition == "potential" else dict(max_iter=60))
        self._make = value_grad
        self._moment = jax.jit(zeeman.make_scf_magnetic_moment(potential, self.energy_fn))

    def __call__(self, positions, field=None):
        if bj.needs_rebuild(positions, self.topo, cell, cfg.rcut, cfg.rcut2):
            print("   (neighbour lists rebuilt)", flush=True)
            self._build(positions)
        par = params if field is None else dict(params, magnetic_field=jnp.asarray(field))
        (e, aux), g = self._value_grad(jnp.asarray(positions), self.shifts, par)
        if float(aux["scf_residual"]) > 1e-6:
            if self._careful is None:
                self._careful = self._make(**self._careful_options)[1]
            first = float(aux["scf_residual"])
            (e, aux), g = self._careful(jnp.asarray(positions), self.shifts, par)
            print(f"   self-consistency residual {first:.1e}; retry with gentler mixing: "
                  f"{float(aux['scf_residual']):.1e}", flush=True)
        if field is None and float(aux["scf_residual"]) < 1e-6:
            self.shifts = aux["shifts"]  # warm start only from converged levels
        self.aux = aux
        return float(e), np.asarray(-g)

    def moment(self, positions):
        """Total magnetic moment per atom (N, 3) at B = 0, the field derivative
        of the self-consistent energy."""
        return np.asarray(self._moment(jnp.asarray(positions), jnp.asarray(cell), params,
                                       self.shifts)["total"])


# --- end points and the Peierls path ----------------------------------------------

ends = []
saved = {}
if args.ends and os.path.exists(args.ends):
    saved = dict(np.load(args.ends))
    print(f"end points read from {args.ends}")
for label, site in (("A", site_a), ("B", site_b)):
    if label in saved:
        pos = saved[label]
        calc = Image(pos)
    else:
        calc = Image(dislocated(site))
        t0 = time.perf_counter()
        pos, e, ok = neb.relax(calc, dislocated(site), mobile, fmax=0.01, steps=400,
                               verbose=True)
        print(f"end point {label}: E = {e:.5f} eV, core at x = {core_x(pos):+.3f} A, "
              f"{'converged' if ok else 'NOT converged'} ({time.perf_counter() - t0:.0f} s)")
        saved[label] = pos
        if args.ends:
            np.savez(args.ends, **saved)
    ends.append((pos, calc))
e_ends = [c(pos)[0] for pos, c in ends]
print(f"E(B) - E(A) = {1e3 * (e_ends[1] - e_ends[0]):+.3f} meV (zero by symmetry, up to the "
      f"shared boundary)")

p = args.images + 2
frac = np.linspace(0.0, 1.0, p)
band = (1 - frac)[:, None, None] * ends[0][0] + frac[:, None, None] * ends[1][0]
calcs = [ends[0][1]] + [Image(band[i]) for i in range(1, p - 1)] + [ends[1][1]]
for k, c in enumerate(calcs[1:-1], start=1):  # interpolate the onsite levels too
    c.shifts = (1 - frac[k]) * calcs[0].shifts + frac[k] * calcs[-1].shifts

t0 = time.perf_counter()
path = neb.neb(lambda i, x: calcs[i](x), band, mobile, spring=0.5, fmax=args.fmax,
               steps=args.steps, verbose=True)
print(f"NEB {'converged' if path.converged else 'NOT converged'} in {len(path.history)} "
      f"steps ({time.perf_counter() - t0:.0f} s), climbing image {path.climbing}")

# reaction coordinate: the arc length scaled to the distance between the cores
x = hop * path.path / path.path[-1]
slopes_x = path.slopes * path.path[-1] / hop  # dE/dx
e0 = path.energies - path.energies[0]
tau0, x_star = zeeman.peierls_stress(x, e0, b, b, slopes=slopes_x)
print("\n image   x (A)   core x (A)   E - E_A (meV)")
for i in range(p):
    print(f"{i:6d}  {x[i]:6.3f}   {core_x(path.images[i]) - site_a[0]:+9.3f}   "
          f"{1e3 * e0[i]:10.3f}")
print(f"Peierls barrier {1e3 * e0.max():.2f} meV per b; Peierls stress from the "
      f"zero-stress profile {tau0 * GPA * 1e3:.0f} MPa (steepest at x = {x_star:.3f} A)")

# --- magnetic moments along the path ------------------------------------------------

moments = np.zeros((p, n_atoms, 3))
for i, c in enumerate(calcs):
    c(path.images[i])  # converged levels at the final band
    moments[i] = c.moment(path.images[i])
m_cell = moments.sum(axis=1)[:, 2]
first = zeeman.first_order_barrier(path.energies, moments)
saddle = first["saddle"]
print("\n image   M_z (muB)   M_z - M_z(A) (muB)   largest |dm_i| (muB)")
for i in range(p):
    dm = moments[i, :, 2] - moments[0, :, 2]
    print(f"{i:6d}  {m_cell[i]:10.4f}   {m_cell[i] - m_cell[0]:+12.5f}   "
          f"{np.max(np.abs(dm)):+14.5f}")
grad_z = first["gradient"][2]
print(f"\nmoment change at the saddle: {first['delta_moment'][2]:+.5f} muB per b "
      f"(the whole cell, {mobile.sum()} mobile atoms)")
print(f"first-order barrier shift: dE_P/dB = {grad_z * 1e6:+.4f} micro-eV/T per b, "
      f"relative {grad_z / first['barrier']:+.2e} per T")

# Peierls stress in a field along z, first order in the profile
for field in (10.0, -10.0):
    tau_b, _ = zeeman.peierls_stress(
        x, e0 + zeeman.first_order_energies(moments, [0, 0, field]), b, b,
        slopes=slopes_x - MU_B * field * np.gradient(m_cell, x))
    print(f"Peierls stress at B_z = {field:+.0f} T: {tau_b * GPA * 1e3:.3f} MPa "
          f"(change {(tau_b - tau0) * GPA * 1e3:+.4f} MPa, "
          f"relative {(tau_b - tau0) / tau0:+.2e})")

# --- check against the self-consistent energy in a field ----------------------------

bc = args.check_field
shift = {}
for i in (0, saddle):
    e_plus = calcs[i](path.images[i], field=[0, 0, bc])[0]
    e_minus = calcs[i](path.images[i], field=[0, 0, -bc])[0]
    shift[i] = (e_plus, e_minus)
direct = ((shift[saddle][0] - shift[0][0]) - (shift[saddle][1] - shift[0][1])) / (2 * bc)
second = ((shift[saddle][0] - shift[0][0]) + (shift[saddle][1] - shift[0][1])
          - 2 * (path.energies[saddle] - path.energies[0])) / 2
print(f"\nself-consistent check at +-{bc:.0f} T (fixed positions): "
      f"dE_P/dB = {direct * 1e6:+.4f} micro-eV/T (first order {grad_z * 1e6:+.4f}); "
      f"second-order shift {second * 1e6:+.3f} micro-eV at {bc:.0f} T")

# --- spin-orbit coupling: orbital moments and direction dependence -------------------

soc = {}
if args.xi > 0:
    cfg_nc = bj.BOPConfig(**IRON_CONFIG, basis=basis, spin="noncollinear", spin_orbit=True,
                          batch=max(1, args.batch // 4))
    xi = jnp.array([args.xi * d_shell])
    directions = {"[111] line": [0, 0, 1.0], "[11-2] glide": [1.0, 0, 0],
                  "[-110] normal": [0, 1.0, 0]}
    print(f"\nnon-collinear with xi = {args.xi} eV on the d shell, axes along the field:")
    print(" direction        E_P (meV)   M_n(A)     M_L,n(A)   dM_n saddle   "
          "dE_P/dB (micro-eV/T)")
    for name, n in directions.items():
        n = np.asarray(n)
        e_dir, m_dir, l_dir = [], [], []
        for i in sorted({0, saddle}):
            topo = bj.build_topology(path.images[i], [0] * n_atoms, cfg_nc.rcut, cfg_nc.nmom,
                                     cell, rcut2=cfg_nc.rcut2, skin=0.6)
            energy_nc = bj.make_energy_fn(topo, cfg_nc, radial=v0_bond)
            par = dict(params, spin_orbit=xi, spin_axes=jnp.tile(jnp.asarray(n), (n_atoms, 1)))
            pot_nc = bj.make_potential(energy_nc, tol=1e-9, condition=args.condition)
            here = (jnp.asarray(path.images[i]), jnp.asarray(cell), par, calcs[i].shifts)
            e_nc, aux_nc = jax.jit(pot_nc)(*here)
            mm = jax.jit(zeeman.make_scf_magnetic_moment(pot_nc, energy_nc))(*here)
            e_dir.append(float(e_nc))
            m_dir.append(float(np.sum(np.asarray(mm["total"]) @ n)))
            l_dir.append(float(np.sum(np.asarray(mm["orbital"]) @ n)))
            if float(aux_nc["scf_residual"]) > 1e-6:
                print(f"   ({name}, image {i}: self-consistency not converged, "
                      f"residual {float(aux_nc['scf_residual']):.1e})")
        dm = m_dir[-1] - m_dir[0]
        soc[name] = dict(barrier=e_dir[-1] - e_dir[0], moment=m_dir, orbital=l_dir)
        print(f" {name:15s}  {1e3 * (e_dir[-1] - e_dir[0]):9.3f}  {m_dir[0]:9.3f}  "
              f"{l_dir[0]:9.4f}   {dm:+10.5f}   {-MU_B * dm * 1e6:+10.4f}")

if args.out:
    np.savez(args.out, images=path.images, energies=path.energies, x=x, slopes=slopes_x,
             moments=moments, mobile=mobile, a=a, b=b, g_spin=G_SPIN)
    print(f"path and moments written to {args.out}")
print(f"\ntotal {time.perf_counter() - t_start:.0f} s")
