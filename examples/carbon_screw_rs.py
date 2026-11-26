"""Carbon at a 1/2<111> screw dislocation in iron, with the iron-carbon BOP on
bopfox-rs (``IRON_CARBON_FIT``, potential self-consistency, exact forces).

Geometry: a cylinder of radius ``--radius`` around a [111] line, ``--layers``
Burgers vectors long and periodic along the line, so one carbon atom per
``layers`` b of line.  Atoms of the outer ``--fixed`` A stay at the
anisotropic elastic (Stroh) solution of the Egorov iron (a = 2.8394 A,
C11/C12/C44 = 272/140/85 GPa).  Every other atom is relaxed on the exact
forces (``RsCell.relax``) and the final state is checked against cold starts
for a lower magnetic state.

Jobs (each written to ``OUT/<job>.npz``, skipped when present):

  perfect        cylinder without dislocation
  perfect_C      ... with C at the octahedral site nearest the axis
  easy           relaxed easy core
  hard           hard core (unstable in pure Fe: it shows where it relaxes to)
  hard_C         hard core with C in the prismatic site at its centre
  oct_<k>        easy core with C at the k-th inequivalent octahedral site
  drag_<f>[_C]   Peierls drag, see ``--help`` of the drag subcommand

Binding energy:  E_b = [E(disl + C) - E(easy)] - [E(perfect_C) - E(perfect)],
negative when carbon is attracted.

    uv run python examples/carbon_screw_rs.py OUT sites           # list the sites
    uv run python examples/carbon_screw_rs.py OUT run JOB [JOB...]
    uv run python examples/carbon_screw_rs.py OUT report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from _dislocation import ROT, Columns, perfect_cylinder, stroh_displacement
from _iron_carbon_rs import A_FE, RsCell

from bopjax.models import FE, C, iron_carbon

GPA = 160.21766208
A = A_FE
B = A * np.sqrt(3) / 2
OCT_FRAC = np.array([[.5, .5, 0], [.5, 0, .5], [0, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]])
OCT_AXIS = np.array([2, 1, 0, 0, 1, 2])  # cube axis of the two close Fe neighbours


def elastic_field():
    c11, c12, c44 = 272.0 / GPA, 140.0 / GPA, 85.0 / GPA
    c = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            c[i, i, j, j] = c12
        c[i, i, i, i] = c11
    for i, j in ((0, 1), (1, 2), (0, 2)):
        c[i, j, i, j] = c[j, i, j, i] = c[i, j, j, i] = c[j, i, i, j] = c44
    c_rot = np.einsum("ia,jb,kc,ld,abcd->ijkl", ROT, ROT, ROT, ROT, c)
    return stroh_displacement(c_rot, np.array([0.0, 0.0, B])), c_rot


class Geometry:
    def __init__(self, radius, fixed, layers):
        self.radius, self.fixed, self.layers = radius, fixed, layers
        self.length = layers * B
        self.u_el, self.c_rot = elastic_field()
        self.ref = perfect_cylinder(A, radius, self.length)
        self.n = len(self.ref)
        self.cols = Columns(self.ref, A, B)
        self.easy = self.cols.easy_core_site()
        self.hard = self.hard_core_site()
        self.mobile = np.hypot(*self.ref[:, :2].T) < radius - fixed
        side = 2 * radius + 2 * (6.0 + 1.0)  # rcut2 = 6 A: isolated across the cylinder
        self.cell = np.diag([side, side, self.length])

    def hard_core_site(self):
        """Centroid, next to the easy site, of a triangle whose column heights
        fall by b/3 counter-clockwise: the elastic field brings its three
        columns to the same height (the hard core)."""
        cols = self.cols
        h = cols.height[cols.triangles] / B
        falling = np.all(np.isclose(np.mod(np.roll(h, -1, axis=1) - h, 1), 2 / 3, atol=1e-3),
                         axis=1)
        dist = np.linalg.norm(cols.centroid - self.easy, axis=1) + 1e3 * ~falling
        return cols.centroid[np.argmin(dist)]

    def dislocated(self, site):
        return self.ref + self.u_el(self.ref[:, 0] - site[0], self.ref[:, 1] - site[1])

    def fixed_atoms(self):
        return np.nonzero(~self.mobile)[0]

    def wrap_z(self, d):
        d = np.array(d, float)
        d[..., 2] -= self.length * np.round(d[..., 2] / self.length)
        return d

    # -- carbon sites --------------------------------------------------------

    def octahedral_sites(self, rmax):
        """Octahedral sites of the perfect cylinder within rmax of the easy
        core, one period b along the line (b in the middle of the cell)."""
        nmax = int(np.ceil((self.radius + self.length) / A)) + 2
        g = np.stack(np.meshgrid(*[np.arange(-nmax, nmax + 1)] * 3, indexing="ij"),
                     -1).reshape(-1, 3)
        pts, axis = [], []
        for f, ax in zip(OCT_FRAC, OCT_AXIS):
            p = A * (g + f) @ ROT.T
            keep = ((np.hypot(*(p[:, :2] - self.easy).T) < rmax) & (p[:, 2] >= B - 1e-6)
                    & (p[:, 2] < 2 * B - 1e-6))
            pts.append(p[keep])
            axis += [ax] * int(keep.sum())
        return np.concatenate(pts), np.array(axis)

    def place(self, site_ref, core, fe_pos):
        """Position of a perfect-lattice site in the dislocated (and relaxed)
        crystal: the elastic displacement of the core at ``core`` plus the mean
        relaxation of its six nearest Fe atoms."""
        el = self.dislocated(core)
        d = np.linalg.norm(self.wrap_z(self.ref - site_ref), axis=1)
        near = np.argsort(d)[:6]
        u = self.u_el(site_ref[0] - core[0], site_ref[1] - core[1])
        return site_ref + u + (fe_pos[near] - el[near]).mean(axis=0)

    def descriptor(self, x, fe_pos, k=24):
        d = np.sort(np.linalg.norm(self.wrap_z(fe_pos - x), axis=1))[:k]
        return np.round(d, 2)

    def inequivalent_sites(self, fe_pos, rmax):
        """Octahedral sites around the easy core, one of each set related by the
        core's symmetry (same sorted Fe distances at the unrelaxed positions)."""
        sites, axis = self.octahedral_sites(rmax)
        placed = np.array([self.place(s, self.easy, fe_pos) for s in sites])
        # classify on the elastic solution, which keeps the symmetry of the
        # core exactly (rounding the relaxed one splits classes)
        el = self.dislocated(self.easy)
        groups = {}
        for s, x, ax in zip(sites, placed, axis):
            key = tuple(self.descriptor(self.place(s, self.easy, el), el))
            groups.setdefault(key, []).append((s, x, ax))
        out = []
        for members in groups.values():
            s, x, ax = min(members, key=lambda m: np.linalg.norm(m[0][:2] - self.easy))
            out.append(dict(ref=s, pos=x, axis=int(ax), multiplicity=len(members),
                            r=float(np.linalg.norm(x[:2] - self.easy))))
        return sorted(out, key=lambda o: o["r"])

    def prismatic_site(self, fe_pos):
        """Centre of the trigonal prism of the hard core: the centroid of the
        three hard-core columns, half way between two of their atoms along z."""
        dxy = np.linalg.norm(self.cols.xy - self.hard, axis=1)
        three = np.argsort(dxy)[:3]
        atoms = [i for i in range(self.n) if self.cols.of_atom[i] in three]
        # the three atoms of one layer: those nearest along z to the first one
        z0 = fe_pos[atoms[0], 2]
        dz = fe_pos[atoms, 2] - z0
        dz -= self.length * np.round(dz / self.length)
        dz -= B * np.round(dz / B)  # the columns are at one height mod b
        return np.array([*self.hard, z0 + np.mean(dz) + B / 2])


# --- running jobs ------------------------------------------------------------

def load(out, job):
    path = os.path.join(out, job + ".npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k].item() if d[k].shape == () else d[k] for k in d.files}


def relax_cell(name, geo, pos, species, args, fixed=None, fixed_components=()):
    cell = RsCell(name, geo.cell, pos, species, tol=1e-6, accept=1e-5, adjoint_tol=1e-7)
    cell.set_params(iron_carbon())
    fixed = geo.fixed_atoms() if fixed is None else fixed
    t0 = time.time()
    part = os.path.join(args.out, name + ".part.npz")
    if os.path.exists(part):  # resume an interrupted relaxation
        cell.positions = np.load(part)["positions"]
        print(f"{name}: resuming from {part}", flush=True)
    for rnd in range(3):
        # chunks of steps; stop on the force tolerance, or once a chunk gains
        # less than `plateau` eV with forces below `fplateau` (the last steps
        # of a soft cylinder change the energy by 1e-4 eV per step)
        e_prev, steps = np.inf, 0
        while True:
            out = cell.relax(fixed=fixed, ftol=args.fmax, maxiter=args.chunk, verbose=True,
                             fixed_components=fixed_components)
            steps += cell.n_steps
            e = out["energy"]
            np.savez(part, positions=cell.positions)
            if (cell.max_force < args.fmax or steps >= args.steps or cell.n_steps == 0
                    or (e_prev - e < args.plateau and cell.max_force < args.fplateau)):
                break
            e_prev = e
        e = out["energy"]
        if not args.ground_state:
            break
        # a lower magnetic state at the relaxed positions?
        cell.fe_moment = 2.4
        best = cell.ground_state(moments=(2.0, 2.8), mixings=(None,))
        print(f"{name}: relaxed E {e:.5f}  fmax {cell.max_force:.4f}  "
              f"ground state {best['energy']:.5f}", flush=True)
        if best["energy"] > e - 1e-3:
            break
    return cell, out, time.time() - t0


def save(out_dir, job, cell, out, seconds, **extra):
    m = np.asarray(out["moment"]).reshape(cell.n, -1)
    np.savez(os.path.join(out_dir, job + ".npz"), positions=cell.positions,
             species=cell.species, energy=out["energy"], max_force=cell.max_force,
             moments=m, charge=np.asarray(out["charge"]), shifts=out["shifts"],
             seconds=seconds, **extra)
    part = os.path.join(out_dir, job + ".part.npz")
    if os.path.exists(part):
        os.remove(part)
    print(f"{job}: E = {out['energy']:.6f} eV, max force {cell.max_force:.4f} eV/A, "
          f"{seconds / 60:.1f} min", flush=True)


def run_job(job, geo, args):
    if load(args.out, job) is not None:
        print(f"{job}: done already")
        return
    fe = np.full(geo.n, FE)
    sp_c = np.append(fe, C)
    if job == "perfect":
        cell, out, s = relax_cell(job, geo, geo.ref, fe, args)
        save(args.out, job, cell, out, s)
    elif job == "easy":
        cell, out, s = relax_cell(job, geo, geo.dislocated(geo.easy), fe, args)
        save(args.out, job, cell, out, s)
    elif job == "hard":
        cell, out, s = relax_cell(job, geo, geo.dislocated(geo.hard), fe, args)
        save(args.out, job, cell, out, s)
    elif job == "perfect_C":
        base = require(args.out, "perfect")
        sites, axis = geo.octahedral_sites(rmax=2.0)
        k = np.argmin(np.hypot(*sites[:, :2].T))
        x = sites[k]
        cell, out, s = relax_cell(job, geo, np.vstack([base["positions"], x]), sp_c, args)
        save(args.out, job, cell, out, s, carbon_start=x)
    elif job == "hard_C":
        el = geo.dislocated(geo.hard)
        x = geo.prismatic_site(el)
        cell, out, s = relax_cell(job, geo, np.vstack([el, x]), sp_c, args)
        save(args.out, job, cell, out, s, carbon_start=x)
    elif job in ("hard_C_held", "hard_C_free"):
        # From a symmetric prism no self-consistent state is found (cold
        # starts); reach the centre by continuation from the relaxed
        # off-centre state, carbon moved along the line in 0.1 A steps.
        src = require(args.out, "offcentre_hardC" if job == "hard_C_held" else "hard_C_held")
        pos = np.asarray(src["positions"], float).copy()
        cell = RsCell(job, geo.cell, pos, sp_c, tol=1e-6, accept=1e-5, adjoint_tol=1e-7)
        cell.set_params(iron_carbon())
        cell._x = np.asarray(src["shifts"])
        n_c = geo.n
        if job == "hard_C_held":
            target = geo.prismatic_site(pos)[2]
            dz = target - pos[n_c, 2]
            dz -= B * np.round(dz / B)
            steps = int(np.ceil(abs(dz) / 0.1))
            z0 = pos[n_c, 2]
            k = 1
            while k <= steps:
                p = cell.positions.copy()
                p[n_c, 2] = z0 + dz * k / steps
                x_keep, out_keep = cell._x, getattr(cell, "out", None)
                try:
                    o = cell.solve(p)
                except RuntimeError:
                    cell._x, cell.out = x_keep, out_keep  # solve() drops the warm start
                    # no state at fixed Fe: relax the lattice at the last
                    # carbon position, carbon held along the line, and retry
                    if getattr(cell, "_relaxed_at", None) == k:
                        raise
                    cell._relaxed_at = k
                    print(f"{job}: relaxing at C {dz * (k - 1) / steps:+.3f} A", flush=True)
                    cell.relax(fixed=geo.fixed_atoms(), ftol=0.05, maxiter=40,
                               fixed_components=((n_c, 2),))
                    print(f"{job}: C held at {dz * (k - 1) / steps:+.3f} A, relaxed "
                          f"E {cell.out['energy']:.5f}", flush=True)
                    continue
                cell.positions = p
                print(f"{job}: C moved {dz * k / steps:+.3f} A, E {o['energy']:.5f}", flush=True)
                k += 1
            pos = cell.positions
            fixed_components = ((n_c, 2),)
        else:
            fixed_components = ()
        t0 = time.time()
        out = cell.relax(fixed=geo.fixed_atoms(), ftol=args.fmax, maxiter=args.steps,
                         verbose=True, fixed_components=fixed_components)
        save(args.out, job, cell, out, time.time() - t0)
    elif job.startswith("shear_"):
        shear(job[6:], geo, args)
    elif job in ("dragA", "dragB", "dragA_C", "dragB_C", "dragAp", "dragBp", "dragA_Cp",
                 "dragB_Cp"):
        drag_end(job, geo, args)
    elif job in ("dragpath", "dragpath_C", "dragpath_Cp"):
        drag_path("_C" in job, geo, args, "p" if job.endswith("p") else "")
    elif job.startswith("oct_"):
        k = int(job[4:])
        base = require(args.out, "easy")
        site = geo.inequivalent_sites(base["positions"], args.rmax)[k]
        x = site["pos"]
        cell, out, s = relax_cell(job, geo, np.vstack([base["positions"], x]), sp_c, args)
        save(args.out, job, cell, out, s, carbon_start=x, site_r=site["r"],
             site_axis=site["axis"], multiplicity=site["multiplicity"])
    else:
        raise SystemExit(f"unknown job {job}")


VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def shear_strain(c_rot, tau_gpa):
    """Homogeneous strain of a pure shear stress sigma_yz = tau: the resolved
    shear stress on the (-110) glide plane along b, no other stress component
    (the [111] frame couples sigma_yz to normal strains through C14)."""
    c6 = np.array([[c_rot[i, j, k, m] for (k, m) in VOIGT] for (i, j) in VOIGT])
    sig = np.zeros(6)
    sig[3] = tau_gpa / GPA
    e6 = np.linalg.solve(c6, sig)  # engineering shears
    eps = np.zeros((3, 3))
    for v, (i, j) in enumerate(VOIGT):
        eps[i, j] = eps[j, i] = e6[v] if i == j else e6[v] / 2
    return eps


def core_centre(geo, pos, cell_def):
    """Burgers-weighted centroid of the core triangles, the homogeneous strain
    removed first."""
    undeformed = pos[: geo.n] @ np.linalg.inv(cell_def).T
    sites, content = geo.cols.core(undeformed[:, 2] - geo.ref[:, 2])
    if not len(sites):
        return np.array([np.nan, np.nan])
    return (sites * content[:, None]).sum(0) / content.sum()


def shear(base, geo, args):
    """Raise the resolved shear stress in steps (``--taus``, GPa) on the relaxed
    configuration ``base`` and relax at each, every atom (the fixed shell
    included) carrying the homogeneous strain.  Stops once the core has moved
    by more than half the distance between easy sites (a sqrt(2/3) / 2)."""
    start = require(args.out, base)
    pos = np.asarray(start["positions"], float)
    species = np.asarray(start["species"])
    eps_old = np.zeros((3, 3))
    c0 = core_centre(geo, pos, np.eye(3))
    print(f"shear {base}: core at {np.round(c0, 3)}", flush=True)
    rows = []
    for tau in args.taus:
        job = f"shear_{base}_{tau:g}"
        done = load(args.out, job)
        eps = shear_strain(geo.c_rot, tau)
        f_new = np.eye(3) + eps
        if done is not None:
            pos, eps_old = np.asarray(done["positions"]), eps
            rows.append((tau, float(done["energy"]), done["core"]))
            continue
        rel = f_new @ np.linalg.inv(np.eye(3) + eps_old)
        pos = pos @ rel.T
        cell_def = geo.cell @ f_new.T
        g = Geometry.__new__(Geometry)
        g.__dict__.update(geo.__dict__)
        g.cell = cell_def
        cell, out, s = relax_cell(job, g, pos, species, args)
        pos, eps_old = cell.positions, eps
        core = core_centre(geo, pos, f_new)
        save(args.out, job, cell, out, s, core=core, tau=tau, strain=eps)
        moved = np.linalg.norm(core - c0)
        print(f"{job}: core at {np.round(core, 3)}, moved {moved:.3f} A", flush=True)
        rows.append((tau, float(out["energy"]), core))
        if not np.isfinite(moved) or moved > A * np.sqrt(2 / 3) / 2:
            break


# --- Peierls drag: the core moved one valley, the boundary at the midpoint -------

def drag_sites(geo, sign=-1):
    """Easy site A and its neighbour B one a sqrt(2/3) along x on the (-110)
    plane: along -x (sign -1, jobs without the suffix "p") away from carbon
    in its deepest bound state (+0.9, +0.7 A from A), along +x (suffix "p")
    past it."""
    a_site = geo.easy
    return a_site, a_site + sign * np.array([A * np.sqrt(2 / 3), 0.0])


def drag_elastic(geo, site, sign=-1):
    """Mobile region at the elastic solution of a core at ``site``, the fixed
    shell at that of the midpoint of A and B (as examples/peierls_drag.py)."""
    sa, sb = drag_sites(geo, sign)
    mid = 0.5 * (sa + sb)
    u_site = geo.u_el(geo.ref[:, 0] - site[0], geo.ref[:, 1] - site[1])
    u_mid = geo.u_el(geo.ref[:, 0] - mid[0], geo.ref[:, 1] - mid[1])
    return geo.ref + np.where(geo.mobile[:, None], u_site, u_mid)


def drag_end(job, geo, args):
    sign = 1 if job.endswith("p") else -1
    base = job[:-1] if sign == 1 else job
    sa, sb = drag_sites(geo, sign)
    fe = np.full(geo.n, FE)
    if base in ("dragA", "dragB"):
        pos = drag_elastic(geo, sa if base == "dragA" else sb, sign)
        cell, out, s = relax_cell(job, geo, pos, fe, args)
        save(args.out, job, cell, out, s)
        return
    # carbon where it binds most strongly next to A (hard_C_free), carried
    # along by the change of its six nearest Fe atoms
    bound = require(args.out, "hard_C_free")
    x = np.asarray(bound["positions"][-1], float)
    fe_a = np.asarray(bound["positions"][: geo.n])
    end = require(args.out, base[:-2] + ("p" if sign == 1 else ""))
    fe_end = np.asarray(end["positions"])
    near = np.argsort(np.linalg.norm(geo.wrap_z(fe_a - x), axis=1))[:6]
    x = x + (fe_end[near] - fe_a[near]).mean(axis=0)
    held = ()
    if base == "dragB_C":
        # carbon draws a core one valley away back to A without a barrier, so
        # B holds the z coordinates of the core columns at those of pure B
        sfx = "p" if sign == 1 else ""
        za = np.asarray(require(args.out, "dragA" + sfx)["positions"])[:, 2]
        dz = fe_end[:, 2] - za
        dz -= geo.length * np.round(dz / geo.length)
        held = tuple((int(i), 2) for i in np.nonzero((np.abs(dz) > 0.2) & geo.mobile)[0])
    cell, out, s = relax_cell(job, geo, np.vstack([fe_end, x]), np.append(fe, C), args,
                              fixed_components=held)
    save(args.out, job, cell, out, s, held=np.array([i for i, _ in held]))


def drag_path(with_c, geo, args, suffix=""):
    """The z coordinates of the core columns (those that differ by more than
    0.2 A between A and B) held at z_A + f (z_B - z_A), everything else
    relaxed (carbon included), continuing from the previous fraction."""
    tag = ("_C" if with_c else "") + suffix
    a, b = require(args.out, "dragA" + tag), require(args.out, "dragB" + tag)
    pa, pb = np.asarray(a["positions"], float), np.asarray(b["positions"], float)
    dz = pb[: geo.n, 2] - pa[: geo.n, 2]
    dz -= geo.length * np.round(dz / geo.length)
    core = np.nonzero((np.abs(dz) > 0.2) & geo.mobile)[0]
    print(f"drag{tag}: {len(core)} core atoms, E_B - E_A = "
          f"{float(b['energy']) - float(a['energy']):+.4f} eV", flush=True)
    pos = pa.copy()
    species = np.asarray(a["species"])
    for f in args.fractions:
        job = f"drag_{f:g}{tag}"
        done = load(args.out, job)
        if done is not None:
            pos = np.asarray(done["positions"])
            continue
        target = pa + f * (pb - pa)  # starting guess, then the held z set
        target[:, 2] = pa[:, 2]
        target[core, 2] = pa[core, 2] + f * dz[core]
        start = pos.copy()
        start[core] = target[core]
        cell, out, s = relax_cell(job, geo, start, species, args,
                                  fixed_components=tuple((int(i), 2) for i in core))
        pos = cell.positions
        save(args.out, job, cell, out, s, fraction=f, core_atoms=core)
        print(f"{job}: E - E_A = {out['energy'] - float(a['energy']):+.4f} eV "
              f"({(out['energy'] - float(a['energy'])) / geo.layers * 1e3:+.1f} meV/b)",
              flush=True)


def require(out_dir, job):
    d = load(out_dir, job)
    if d is None:
        raise SystemExit(f"run '{job}' first")
    return d


# --- analysis ----------------------------------------------------------------

def core_of(geo, pos):
    sites, content = geo.cols.core(pos[: geo.n, 2] - geo.ref[:, 2])
    return sites, content


def describe(geo, d):
    pos = d["positions"]
    txt = ""
    sites, content = core_of(geo, pos)
    if len(sites):
        cen = (sites * content[:, None]).sum(0) / content.sum()
        txt += f"core at ({cen[0]-geo.easy[0]:+.2f}, {cen[1]-geo.easy[1]:+.2f}) from easy"
        if np.linalg.norm(cen - geo.hard) < 0.3:
            txt += " [hard]"
        elif np.linalg.norm(cen - geo.easy) < 0.3:
            txt += " [easy]"
    if len(pos) > geo.n:
        c = pos[-1]
        dist = np.linalg.norm(geo.wrap_z(pos[: geo.n] - c), axis=1)
        nn = np.sort(dist)[:6]
        txt += (f"; C at r = {np.linalg.norm(c[:2] - geo.easy):.2f} A, Fe-C "
                + " ".join(f"{v:.2f}" for v in nn)
                + f"; m_C {d['moments'][-1].sum():+.2f}")
    return txt


def report(geo, args):
    e = {}
    for f in sorted(os.listdir(args.out)):
        if f.endswith(".npz") and not f.endswith(".part.npz"):
            e[f[:-4]] = load(args.out, f[:-4])
    if "perfect" in e and "perfect_C" in e:
        e_sol = e["perfect_C"]["energy"] - e["perfect"]["energy"]
        print(f"C in the perfect cylinder: E(perfect_C) - E(perfect) = {e_sol:.4f} eV "
              f"({describe(geo, e['perfect_C'])})")
    else:
        e_sol = None
    if "easy" in e and "perfect" in e:
        print(f"easy core: {describe(geo, e['easy'])}")
    if "hard" in e and "easy" in e:
        print(f"hard - easy (pure Fe): {(e['hard']['energy'] - e['easy']['energy']) / geo.layers * 1e3:.1f} meV/b "
              f"({describe(geo, e['hard'])})")
    rows = []
    for job, d in e.items():
        if (job.startswith("oct_") or job.startswith("hard_C") or job == "offcentre_hardC") and e_sol is not None and "easy" in e:
            eb = d["energy"] - e["easy"]["energy"] - e_sol
            rows.append((job, eb, d))
    rows.sort(key=lambda r: r[1])
    for job, eb, d in rows:
        start = (f"start r {float(d['site_r']):.2f} A, axis {'xyz'[int(d['site_axis'])]}, "
                 f"x{int(d['multiplicity'])}" if "site_r" in d else "prismatic site")
        print(f"{job:8s} E_b = {eb:+.3f} eV  fmax {float(d['max_force']):.3f}  [{start}]  "
              f"{describe(geo, d)}")
    drags = report_drag(geo, e)
    summary = {job: dict(E_b=float(eb), max_force=float(d["max_force"])) for job, eb, d in rows}
    with open(os.path.join(args.out, "binding.json"), "w") as fh:
        json.dump(dict(E_perfect_C_minus_perfect=e_sol, binding=summary, drag=drags), fh,
                  indent=1)


def report_drag(geo, e):
    """Drag profiles E(f) - E_A per b, and the stress of the steepest segment,
    tau = (dE/dx) / (b L): the 0 K Peierls (or depinning) stress of the
    straight segment along that path, an upper bound (constrained path,
    finite differences between the computed fractions)."""
    lam = A * np.sqrt(2 / 3)
    out = {}
    for tag, label in (("", "pure Fe"), ("_C", "C, core moved away"),
                       ("_Cp", "C, core moved past")):
        sfx = "p" if tag.endswith("p") else ""
        a, b = e.get("dragA" + tag), e.get("dragB" + tag)
        if a is None:
            continue
        pts = [(0.0, float(a["energy"]))]
        for job, d in e.items():
            if job.startswith("drag_") and job.endswith(tag) and (
                    job[5:-len(tag)] if tag else job[5:]).replace(".", "").isdigit():
                pts.append((float(d["fraction"]), float(d["energy"])))
        if b is not None:
            pts.append((1.0, float(b["energy"])))
        pts.sort()
        f = np.array([q[0] for q in pts])
        en = (np.array([q[1] for q in pts]) - pts[0][1]) / geo.layers  # eV per b
        if tag == "" and f[f < 1].max() <= 0.5 + 1e-9:  # mirror the symmetric profile
            en = en[f <= 0.5 + 1e-9]
            f = f[f <= 0.5 + 1e-9]
            f = np.concatenate([f, 1 - f[::-1][1:]])
            en = np.concatenate([en, en[::-1][1:]])
        slope = np.diff(en) / np.diff(f * lam)  # eV / A per b of line
        tau = slope / B**2 * GPA * 1e3  # MPa
        k = int(np.argmax(tau)) if len(tau) else 0
        print(f"drag, {label}: " + "  ".join(f"{x:.2f}:{y * 1e3:.1f}" for x, y in zip(f, en))
              + " meV/b" + (f"; steepest {f[k]:.2f}-{f[k + 1]:.2f}: tau = {tau[k]:.0f} MPa"
                             if len(tau) else ""))
        out[label] = dict(fraction=f.tolist(), energy_per_b=en.tolist(),
                          tau_max_MPa=float(tau[k]) if len(tau) else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("what", choices=["sites", "run", "report"])
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--radius", type=float, default=13.0)
    ap.add_argument("--fixed", type=float, default=6.0)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--rmax", type=float, default=4.5, help="octahedral sites within (A)")
    ap.add_argument("--fmax", type=float, default=5e-3)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=15, help="relaxation steps per checkpoint")
    ap.add_argument("--plateau", type=float, default=2e-3,
                    help="stop when a chunk gains less than this (eV) ...")
    ap.add_argument("--fplateau", type=float, default=0.2, help="... and fmax is below (eV/A)")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0],
                    help="resolved shear stresses of the shear jobs (GPa)")
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85],
                    help="fractions of the Peierls drag")
    ap.add_argument("--ground-state", action=argparse.BooleanOptionalAction, default=True,
                    help="check the relaxed state against cold magnetic starts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    geo = Geometry(args.radius, args.fixed, args.layers)
    print(f"{geo.n} Fe atoms, {geo.mobile.sum()} mobile, {geo.layers} b; easy core "
          f"{np.round(geo.easy, 3)}, hard core {np.round(geo.hard, 3)}", flush=True)
    if args.what == "sites":
        base = load(args.out, "easy")
        pos = base["positions"] if base is not None else geo.dislocated(geo.easy)
        for k, s in enumerate(geo.inequivalent_sites(pos, args.rmax)):
            print(f"oct_{k}: r = {s['r']:.2f} A, axis {'xyz'[s['axis']]}, "
                  f"multiplicity {s['multiplicity']}, Fe-C "
                  + " ".join(f"{v:.2f}" for v in geo.descriptor(s['pos'], pos)[:6]))
        print("prismatic site:", np.round(geo.prismatic_site(geo.dislocated(geo.hard)), 3))
    elif args.what == "run":
        for job in args.jobs:
            run_job(job, geo, args)
    else:
        report(geo, args)


if __name__ == "__main__":
    sys.exit(main())
