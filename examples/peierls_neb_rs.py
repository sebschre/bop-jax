"""Free Peierls path of a 1/2<111> screw dislocation in iron (climbing-image
NEB on bopfox-rs) and the first-order field dependence of its barrier, with
and without spin-orbit coupling.

The unconstrained counterpart of ``examples/peierls_drag.py``: same model (the
Fe part of the Fe-Co BOP of Egorov et al., ``models.iron_egorov``), same
cylinder (one Burgers vector long, the outer shell held at the anisotropic
elastic solution centred midway between two easy-core sites A and B one
a sqrt(2/3) apart on the (-110) plane), same potential self-consistency, but
the path is a climbing-image NEB between the relaxed A and B with no
constraint on the core.

1. Relax A and B with FIRE (exact potential-condition forces from bopfox-rs).
2. Climbing-image NEB with ``--images`` images between them, every image with
   its own warm-started self-consistency (``bopfox_rs.neb``).
3. Collinear: the magnetic moment of every image (field derivative of the
   self-consistent energy), dE_P/dB = -mu_B [M(saddle) - M(A)], the Peierls
   stress and its change in 10 T, and a self-consistent check at
   +-``--check-field``.
4. Spin-orbit (``--xi``, eV, d shell; 0 skips it): every image of the band
   re-evaluated non-collinearly with xi L.S, all spin axes along the field,
   for fields along the line z = [111], the glide direction x = [11-2] and the
   glide-plane normal y = [-110]: barrier for each magnetisation direction,
   spin and orbital moments, dE_P/dB per direction, and a self-consistent
   check.  The positions are those of the collinear band; spin-orbit coupling
   changes the forces only slightly, and since the images are stationary in
   the positions this changes the energies and moments at second order in
   that force change.

Every stage is written to OUTDIR (``ends.npz``, ``band.npz``,
``collinear.json``, ``spin_orbit.json``); rerun to resume.

    uv run python examples/peierls_neb_rs.py OUTDIR [--images 7] [--xi 0.05]
"""

import argparse
import json
import os
import time

import numpy as np
from _dislocation import ROT, Columns, perfect_cylinder, stroh_displacement
from bopfox_rs import MU_B, Potential, model_from_bopjax, zeeman
from bopfox_rs.neb import image_forces, neb, relax

import bopjax as bj
from bopjax.models import IRON_EGOROV_CONFIG, iron_egorov

GPA = 160.21766208

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("out", help="directory for checkpoints and results")
parser.add_argument("--images", type=int, default=7, help="images between A and B")
parser.add_argument("--radius", type=float, default=12.0)
parser.add_argument("--fixed", type=float, default=6.0)
parser.add_argument("--fmax", type=float, default=0.01, help="force tolerance (eV/A)")
parser.add_argument("--steps", type=int, default=600, help="NEB steps")
parser.add_argument("--check-field", type=float, default=100.0,
                    help="field of the self-consistent check, collinear (T)")
parser.add_argument("--so-check-field", type=float, default=1.0,
                    help="field of the self-consistent check with spin-orbit coupling (T): "
                         "with fixed axes the response is strongly nonlinear, and at 100 T the "
                         "third-order terms are as large as the moment differences")
parser.add_argument("--xi", type=float, default=0.05, help="d-shell spin-orbit constant (eV)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)


def path(name):
    return os.path.join(args.out, name)


# --- model and geometry (as examples/peierls_drag.py) ----------------------------
a = 2.8395  # zero-pressure lattice constant of the model
b = a * np.sqrt(3) / 2
cfg = bj.BOPConfig(**IRON_EGOROV_CONFIG, spin="collinear")
params = iron_egorov()
model = model_from_bopjax(cfg, params, radial="exp_sum")
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
species = [0] * n
site_a = Columns(ref, a, b).easy_core_site()
site_b = site_a + np.array([a * np.sqrt(2 / 3), 0.0])
mid = 0.5 * (site_a + site_b)
mobile = np.hypot(*(ref[:, :2] - mid).T) < args.radius - args.fixed
cell = np.diag([2 * args.radius + 2 * (cfg.rcut2 + 1)] * 2 + [b])
u_mid = u_el(ref[:, 0] - mid[0], ref[:, 1] - mid[1])
SCF = dict(condition="potential", tol=1e-8, adjoint_tol=1e-7)


def elastic(site):
    return ref + np.where(mobile[:, None], u_el(ref[:, 0] - site[0], ref[:, 1] - site[1]),
                          u_mid)


print(f"{n} atoms, {mobile.sum()} mobile; easy cores at {np.round(site_a, 3)} and "
      f"{np.round(site_b, 3)}", flush=True)
start = np.asarray(bj.stoner_shifts(np.full((n, 1), 2.6), params["stoner"][0]))

# --- 1. end points --------------------------------------------------------------------
if os.path.exists(path("ends.npz")):
    st = np.load(path("ends.npz"))
    pos_a, pos_b, shifts_a = st["pos_a"], st["pos_b"], st["shifts_a"]
else:
    ends = []
    for label, site in (("A", site_a), ("B", site_b)):
        pot = Potential(model)
        state = {"shifts": start, "first": True, "step": 0}

        def force_fn(x):
            out = pot.scf(x, species, cell, state["shifts"], reuse_history=not state["first"],
                          **SCF)
            if not out["converged"]:
                out = pot.scf(x, species, cell, start, **SCF)
            state.update(shifts=out["shifts"], first=False, step=state["step"] + 1)
            if state["step"] % 10 == 0:
                print(f"  {label} step {state['step']:3d}  E {out['energy']:.6f}  max|F| "
                      f"{np.max(np.linalg.norm(out['forces'][mobile], axis=1)):.4f}", flush=True)
            return out["energy"], out["forces"]

        t0 = time.time()
        x, e, ok = relax(force_fn, elastic(site), mobile, fmax=args.fmax, steps=1000,
                         max_step=0.05)
        print(f"{label}: E {e:.6f} eV, {'converged' if ok else 'NOT converged'}, "
              f"{state['step']} steps, {time.time() - t0:.0f} s", flush=True)
        ends.append((x, e, state["shifts"]))
    (pos_a, e_a, shifts_a), (pos_b, e_b, _) = ends
    print(f"E_B - E_A = {1e3 * (e_b - e_a):+.3f} meV", flush=True)
    np.savez(path("ends.npz"), pos_a=pos_a, pos_b=pos_b, shifts_a=shifts_a)

# --- 2. climbing-image NEB --------------------------------------------------------------
n_img = args.images + 2
if os.path.exists(path("band.npz")):
    st = np.load(path("band.npz"))
    images, climbing = st["images"], int(st["climbing"])
else:
    band = np.array([pos_a + t * (pos_b - pos_a) for t in np.linspace(0, 1, n_img)])
    ff = image_forces(model, species, cell, shifts_a, n_img, tol=SCF["tol"],
                      condition=SCF["condition"],
                      scf_options={"adjoint_tol": SCF["adjoint_tol"]})
    t0 = time.time()
    res = neb(ff, band, mobile, fmax=args.fmax, max_step=0.05, steps=args.steps, verbose=True)
    print(f"NEB {'converged' if res.converged else 'NOT converged'} after "
          f"{len(res.history)} steps, {time.time() - t0:.0f} s; climbing image {res.climbing}",
          flush=True)
    images, climbing = res.images, res.climbing
    np.savez(path("band.npz"), images=images, climbing=climbing, energies=res.energies,
             path=res.path, slopes=res.slopes, converged=res.converged,
             history=np.array(res.history))

# reaction coordinate: projection of each image on B - A, as a glide distance
d_ab = (pos_b - pos_a).ravel()
frac = np.array([(img - pos_a).ravel() @ d_ab / (d_ab @ d_ab) for img in images])
x_glide = a * np.sqrt(2 / 3) * frac
dz = pos_b[:, 2] - pos_a[:, 2]
core = np.abs(dz) > 0.2


def analyse(pot, axes=None, direction=(0.0, 0.0, 1.0), label="", check=None):
    """Energies and moments along the band; first-order barrier shift along
    ``direction``; self-consistent check at the saddle."""
    d = np.asarray(direction, float)
    rows, outs = [], []
    shifts = shifts_a
    for k, img in enumerate(images):
        out = pot.scf(img, species, cell, shifts, condition="potential", tol=1e-10,
                      adjoint_tol=1e-9, axes=axes, magnetic_field=np.zeros(3))
        if not out["converged"]:
            raise RuntimeError(f"{label} image {k}: self-consistency not converged")
        shifts = out["shifts"]
        outs.append(out)
        rows.append(dict(f=float(frac[k]), E=float(out["energy"]),
                         M=float(out["magnetic_moment"].sum(0) @ d),
                         S=float(out["spin_moment"].sum(0) @ d),
                         L=float(out["orbital_moment"].sum(0) @ d),
                         core_moment=float(np.abs(out["moment"][core]).mean())))
    e = np.array([r["E"] for r in rows])
    e_abs = e.copy()
    e -= e[0]
    m = np.array([r["M"] for r in rows])
    first = zeeman.first_order_barrier(e, np.outer(m, d))
    s = first["saddle"]
    grad = float(first["gradient"] @ d)
    for r, ek in zip(rows, e):
        r["E"] = float(ek)
        print(f"  {label} f {r['f']:.3f}  E - E_A {1e3 * r['E']:7.2f} meV  M {r['M']:.4f}  "
              f"(S {r['S']:.4f}, L {r['L']:.4f}) muB  |m| core {r['core_moment']:.3f}",
              flush=True)
    # Peierls stress of the band and its change in 10 T along d (first order)
    tau0, _ = zeeman.peierls_stress(x_glide, e, b, b)
    tau10, _ = zeeman.peierls_stress(x_glide, e - MU_B * 10.0 * (m - m[0]), b, b)
    # self-consistent check at +-B along d, fixed positions
    bc = args.check_field if check is None else check
    shifted = []
    for k in (0, s):
        shifted.append([pot.scf(images[k], species, cell, outs[k]["shifts"],
                                condition="potential", tol=1e-10, adjoint_tol=1e-9, axes=axes,
                                magnetic_field=sign * bc * d)["energy"]
                        for sign in (1.0, -1.0)])
    direct = ((shifted[1][0] - shifted[0][0]) - (shifted[1][1] - shifted[0][1])) / (2 * bc)
    second = 0.5 * ((shifted[1][0] - shifted[0][0]) + (shifted[1][1] - shifted[0][1])) - e[s]
    result = dict(rows=rows, E_A=float(e_abs[0]), E_saddle=float(e_abs[s]), saddle=s, barrier=float(e[s]), delta_moment=float(m[s] - m[0]),
                  dEdB=grad, dEdB_check=float(direct), second_order_at_check=float(second),
                  peierls_stress_MPa=tau0 * GPA * 1e3,
                  peierls_stress_change_10T_MPa=(tau10 - tau0) * GPA * 1e3)
    print(f"{label}: barrier {1e3 * e[s]:.2f} meV/b at f = {frac[s]:.3f}; M(saddle) - M(A) = "
          f"{m[s] - m[0]:+.4f} muB; dE_P/dB = {1e6 * grad:+.4f} micro-eV/T per b "
          f"({grad / e[s]:+.2e} per T); check at +-{bc:g} T {1e6 * direct:+.4f}, second "
          f"order {1e6 * second:+.2f} micro-eV; Peierls stress {tau0 * GPA * 1e3:.0f} MPa, "
          f"{(tau10 - tau0) * GPA * 1e3:+.3f} MPa in 10 T", flush=True)
    return result


# --- 3. collinear field effect -----------------------------------------------------------
if not os.path.exists(path("collinear.json")):
    col = analyse(Potential(model), label="collinear")
    col["climbing"] = climbing
    json.dump(col, open(path("collinear.json"), "w"), indent=1)

# --- 4. spin-orbit coupling, three field directions -----------------------------------------
if args.xi > 0 and not os.path.exists(path("spin_orbit.json")):
    import jax.numpy as jnp

    cfg_so = bj.BOPConfig(**IRON_EGOROV_CONFIG, spin="noncollinear", spin_orbit=True)
    model_so = model_from_bopjax(cfg_so, dict(params, spin_orbit=jnp.array([[args.xi]])),
                                 radial="exp_sum")
    so = {"xi": args.xi}
    for name, d in (("line [111]", (0, 0, 1)), ("glide [11-2]", (1, 0, 0)),
                    ("normal [-110]", (0, 1, 0))):
        d = np.asarray(d, float)
        so[name] = analyse(Potential(model_so), axes=np.tile(d, (n, 1)), direction=d,
                           label=f"SO, B || {name}", check=args.so_check_field)
    json.dump(so, open(path("spin_orbit.json"), "w"), indent=1)
