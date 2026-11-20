"""Peierls path of a 1/2<111> screw dislocation in iron by a drag calculation,
and the first-order field dependence of its barrier.

Model: the Fe part of the Fe-Co BOP of Egorov et al. (``models.iron_egorov``).
A cylinder one Burgers vector long, its outer shell held at the anisotropic
elastic solution centred midway between two easy-core sites A and B one
a sqrt(2/3) apart on the (-110) plane.

1. Relax A and B (the mobile region starts from the elastic solution centred
   on each site).
2. The columns whose position along the line differs between A and B by more
   than 0.2 A are the core columns; for each fraction f their z coordinates are
   fixed at z_A + f (z_B - z_A) and everything else is relaxed, continuing from
   the previous f.  The energy along f is an upper bound of the minimum energy
   path; the core moves from A's triangle through the intermediate one.
3. At every point, the magnetic moment of the cell from the field derivative of
   the self-consistent energy (``zeeman.make_scf_magnetic_moment``), and from it
   dE_P/dB = -mu_B [M(f = 0.5) - M(A)] and the change of the Peierls stress.
4. A self-consistent check of dE_P/dB at +-B_check.

Self-consistency: ``condition="potential"`` (Anderson mixing), which converges
in 30-50 steps here.  ``BOPConfig(remat=True)`` keeps a gradient at about 2 GB.
On a 4-core CPU a relaxation step takes about 2.5 minutes; the whole script
takes about 10 hours.  Every converged step is written to ``<out>/live.npz``
and every relaxed point to ``<out>/<label>.npz``; rerun to resume.

Results (docs/magnetic-field.md): barrier 47 meV/b at f = 0.5, moment of the
cell larger at the saddle by 0.098 muB, dE_P/dB = -5.67 micro-eV/T per b.

    uv run python examples/peierls_drag.py OUTDIR [--fractions 0.15 0.3 0.35 0.4 0.5]
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from _dislocation import ROT, Columns, perfect_cylinder, stroh_displacement

import bopjax as bj
from bopjax import neb, zeeman
from bopjax import pairwise as pw
from bopjax.models import IRON_EGOROV_CONFIG, iron_egorov
from bopjax.zeeman import MU_B

GPA = 160.21766208

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("out", help="directory for checkpoints and results")
parser.add_argument("--fractions", type=float, nargs="+", default=[0.15, 0.3, 0.35, 0.4, 0.5])
parser.add_argument("--radius", type=float, default=12.0)
parser.add_argument("--fixed", type=float, default=6.0)
parser.add_argument("--fmax", type=float, default=0.03, help="force tolerance (eV/A)")
parser.add_argument("--steps", type=int, default=80, help="relaxation steps per point")
parser.add_argument("--check-field", type=float, default=100.0)
parser.add_argument("--batch", type=int, default=4)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)


def path(name):
    return os.path.join(args.out, name)


# --- model and geometry ----------------------------------------------------------
a = 2.8395  # zero-pressure lattice constant of the model
b = a * np.sqrt(3) / 2
cfg = bj.BOPConfig(**dict(IRON_EGOROV_CONFIG, batch=args.batch, remat=True), spin="collinear")
params = iron_egorov()
c11, c12, c44 = 272.0 / GPA, 140.0 / GPA, 85.0 / GPA  # elastic constants of the model
c_cub = np.zeros((3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        c_cub[i, i, j, j] = c12
    c_cub[i, i, i, i] = c11
for i, j in ((0, 1), (1, 2), (0, 2)):
    c_cub[i, j, i, j] = c_cub[j, i, j, i] = c_cub[i, j, j, i] = c_cub[j, i, i, j] = c44
u_el = stroh_displacement(np.einsum("ia,jb,kc,ld,abcd->ijkl", ROT, ROT, ROT, ROT, c_cub),
                          np.array([0.0, 0.0, b]))
ref = perfect_cylinder(a, args.radius, b)
n = len(ref)
site_a = Columns(ref, a, b).easy_core_site()
site_b = site_a + np.array([a * np.sqrt(2 / 3), 0.0])
mid = 0.5 * (site_a + site_b)
mobile = np.hypot(*(ref[:, :2] - mid).T) < args.radius - args.fixed
cell = np.diag([2 * args.radius + 2 * (cfg.rcut2 + 1)] * 2 + [b])
C = jnp.asarray(cell)
u_mid = u_el(ref[:, 0] - mid[0], ref[:, 1] - mid[1])


def elastic(site):
    return ref + np.where(mobile[:, None], u_el(ref[:, 0] - site[0], ref[:, 1] - site[1]),
                          u_mid)


print(f"{n} atoms, {mobile.sum()} mobile; easy cores at {np.round(site_a, 3)} and "
      f"{np.round(site_b, 3)}", flush=True)


class Calc:
    """Self-consistent energy and forces with warm-started levels."""

    def __init__(self, pos, shifts):
        self.shifts = shifts
        self.build(pos)

    def build(self, pos):
        self.topo = bj.build_topology(np.asarray(pos), [0] * n, cfg.rcut, cfg.nmom, cell,
                                      rcut2=cfg.rcut2, skin=0.5)
        self.energy_fn = bj.make_energy_fn(self.topo, cfg, radial=pw.exp_sum)
        self.potential = bj.make_potential(self.energy_fn, tol=1e-9, condition="potential",
                                           adjoint_tol=1e-4)
        self.vg = jax.jit(jax.value_and_grad(
            lambda x, s, par: self.potential(x, C, par, s), has_aux=True))

    def __call__(self, pos, par=params):
        if bj.needs_rebuild(pos, self.topo, cell, cfg.rcut, cfg.rcut2):
            self.build(pos)
        (e, aux), g = self.vg(jnp.asarray(pos), self.shifts, par)
        self.res = float(aux["scf_residual"])
        self.m = np.asarray(aux["moment"])
        if self.res < 1e-6 and par is params:
            self.shifts = aux["shifts"]
        return float(e), -np.asarray(g)


done = json.load(open(path("done.json"))) if os.path.exists(path("done.json")) else {}
live = dict(np.load(path("live.npz"))) if os.path.exists(path("live.npz")) else None
if live is not None:
    calc = Calc(live["pos"], jnp.asarray(live["shifts"]))
else:
    topo = bj.build_topology(elastic(site_a), [0] * n, cfg.rcut, cfg.nmom, cell,
                             rcut2=cfg.rcut2, skin=0.5)
    en = bj.make_energy_fn(topo, cfg, radial=pw.exp_sum)
    start = bj.stoner_shifts(np.full((n, 1), 2.6), params["stoner"][0])
    _, aux = jax.jit(bj.make_potential(en, tol=1e-9, condition="potential"))(
        jnp.asarray(elastic(site_a)), C, params, start)
    calc = Calc(elastic(site_a), aux["shifts"])


def relax(label, pos, fixed_z=None):
    """Relax ``pos`` (core columns' z fixed if given), checkpointing every step."""
    if label in done:
        st = np.load(path(f"{label}.npz"))
        calc.shifts = jnp.asarray(st["shifts"])
        return st["pos"]
    if live is not None and str(live["label"]) == label:
        pos = live["pos"]
    step = [0]

    def force_fn(x):
        e, f = calc(x)
        if fixed_z is not None:
            f[fixed_z, 2] = 0.0
        step[0] += 1
        e_a = done.get("A", {}).get("E", np.nan)
        print(f"  {label} step {step[0]:3d}  E - E_A {1e3 * (e - e_a):8.2f} meV  max|F| "
              f"{np.max(np.linalg.norm(f[mobile], axis=1)):.4f}  scf residual {calc.res:.1e}",
              flush=True)
        if calc.res < 1e-6:
            np.savez(path("live.npz"), pos=np.asarray(x), shifts=np.asarray(calc.shifts),
                     label=label)
        return e, f

    pos, e, ok = neb.relax(force_fn, pos, mobile, fmax=args.fmax, steps=args.steps)
    calc(pos)
    done[label] = dict(E=e, converged=bool(ok), m0=float(calc.m[0]))
    np.savez(path(f"{label}.npz"), pos=pos, shifts=np.asarray(calc.shifts))
    json.dump(done, open(path("done.json"), "w"), indent=1)
    print(f"{label} relaxed ({'converged' if ok else 'NOT converged'}): "
          f"E - E_A {1e3 * (e - done.get('A', dict(E=e))['E']):.2f} meV", flush=True)
    return pos


pos_a = relax("A", elastic(site_a))
shifts_a = calc.shifts
pos_b = relax("B", elastic(site_b))
dz = pos_b[:, 2] - pos_a[:, 2]
dz -= b * np.round(dz / b)
core = np.abs(dz) > 0.2
print(f"E_B - E_A {1e3 * (done['B']['E'] - done['A']['E']):.2f} meV; core columns "
      f"{np.nonzero(core)[0]} move by {np.round(dz[core] / b, 3)} b", flush=True)

pos = pos_a
calc.shifts = shifts_a
labels = ["A"]
for f in args.fractions:
    label = f"f{f:.2f}"
    start = pos.copy()
    start[core, 2] = pos_a[core, 2] + f * dz[core]
    pos = relax(label, start, fixed_z=core)
    labels.append(label)

# --- magnetic moments along the path and dE_P/dB ---------------------------------
fracs = np.array([0.0] + list(args.fractions))
energies, moments, states = [], [], {}
for label in labels:
    st = np.load(path(f"{label}.npz"))
    x, s = jnp.asarray(st["pos"]), jnp.asarray(st["shifts"])
    topo = bj.build_topology(st["pos"], [0] * n, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2,
                             skin=0.5)
    en = bj.make_energy_fn(topo, cfg, radial=pw.exp_sum)
    pot = bj.make_potential(en, tol=1e-10, condition="potential", adjoint_tol=1e-8)
    e, aux = jax.jit(pot)(x, C, params, s)
    m = np.asarray(jax.jit(zeeman.make_scf_magnetic_moment(pot, en))(
        x, C, params, aux["shifts"])["total"])[:, 2]
    energies.append(float(e))
    moments.append(m)
    states[label] = (pot, x, aux["shifts"])
    print(f"{label:6s} E - E_A {1e3 * (float(e) - energies[0]):7.2f} meV  "
          f"M_cell {m.sum():.5f} muB", flush=True)
energies = np.array(energies) - energies[0]
moments = np.array(moments)
first = zeeman.first_order_barrier(energies, np.stack([np.zeros_like(moments), np.zeros_like(
    moments), moments], axis=-1))
print(f"\nbarrier at f = {fracs[first['saddle']]:.2f}: {1e3 * first['barrier']:.2f} meV/b; "
      f"M(saddle) - M(A) = {first['delta_moment'][2]:+.5f} muB per b; "
      f"dE_P/dB = {1e6 * first['gradient'][2]:+.4f} micro-eV/T per b "
      f"({first['gradient'][2] / first['barrier']:+.2e} per T)")

if np.isclose(fracs[-1], 0.5):  # mirror the half profile to a full period
    f_full = np.concatenate([fracs, 1 - fracs[-2::-1]])
    e_full = np.concatenate([energies, energies[-2::-1]])
    m_full = np.concatenate([moments.sum(1), moments.sum(1)[-2::-1]])
    x_full = a * np.sqrt(2 / 3) * f_full
    tau0, _ = zeeman.peierls_stress(x_full, e_full, b, b)
    tau1, _ = zeeman.peierls_stress(x_full, e_full - MU_B * 10.0 * (m_full - m_full[0]), b, b)
    print(f"Peierls stress {tau0 * GPA * 1e3:.0f} MPa; at B = 10 T along the magnetisation "
          f"{(tau1 - tau0) * GPA * 1e3:+.3f} MPa")

# --- self-consistent check -----------------------------------------------------------
bc = args.check_field
shifted = {}
for label in (labels[0], labels[first["saddle"]]):
    pot, x, s = states[label]
    shifted[label] = [float(jax.jit(pot)(x, C, dict(params, magnetic_field=jnp.array(
        [0.0, 0.0, sign * bc])), s)[0]) for sign in (1.0, -1.0)]
la, ls = labels[0], labels[first["saddle"]]
direct = ((shifted[ls][0] - shifted[la][0]) - (shifted[ls][1] - shifted[la][1])) / (2 * bc)
print(f"self-consistent check at +-{bc:.0f} T: dE_P/dB = {1e6 * direct:+.4f} micro-eV/T per b "
      f"(first order {1e6 * first['gradient'][2]:+.4f})")
np.savez(path("result.npz"), f=fracs, energies=energies, moments=moments,
         gradient=first["gradient"][2])
