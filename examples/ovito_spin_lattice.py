"""Animation of coupled spin-lattice dynamics with OVITO.

    uv run --extra viz python examples/ovito_spin_lattice.py     (about 10 minutes)

The first run of ``examples/spin_lattice.py``: a bcc d-band ferromagnet of four
(001) planes, canted in pairs by +20 and -20 degrees, on a perfect lattice at
rest and at constant energy.  The moments (red) of the two pairs precess about
their sum.  Planes with parallel moments attract each other more than planes
with canted ones, so the canting puts forces on the atoms and the planes
start to oscillate about a paired arrangement.  There is no spin-orbit
coupling here: the coupling is the dependence of the exchange on the distance.

The displacements are 0.006 A at most and are drawn 40 times larger.

Writes docs/animations/spin_lattice.xyz (open it in OVITO) and, if the ovito
module is installed, docs/animations/spin_lattice.gif.  With ``--from-xyz`` the
animation is rendered again from the saved trajectory without simulating.
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax import md, spinlattice
from bopjax import ovito as bov
from bopjax import pairwise as pw
from bopjax.models import canonical_d

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "animations"
STONER, A, MASS = 0.6, 2.8, 55.85
EXAGGERATE = 40.0


def simulate(dt_fs=0.25, steps=440, every=2):
    positions = A * np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1], [0.5, 0.5, 1.5]])
    cell = A * np.diag([1.0, 1.0, 2.0])
    params = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[STONER]]),
                  repulsion={"pair": jnp.array([[[0.3, 16.0, 2.4]]]),
                             "density": jnp.zeros((1, 1, 3)), "embed": jnp.zeros((1, 2))})
    cfg = bj.BOPConfig(nmom=6, rcut=3.6, dcut=0.5, rcut2=3.6, dcut2=0.5, bandwidth="averaged",
                       repulsion=pw.POWERLAW, spin="noncollinear")
    topo = bj.build_topology(positions, [0] * 4, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2,
                             skin=0.3)
    energy_fn = bj.make_energy_fn(topo, cfg)

    angle = np.radians([20.0, 20.0, -20.0, -20.0])
    canted = jnp.array(np.stack([np.sin(angle), 0.0 * angle, np.cos(angle)], axis=1))
    start = bj.make_scf(energy_fn)(
        jnp.array(positions), jnp.array(cell), dict(params, spin_axes=canted),
        shifts0=bj.stoner_shifts(np.full((4, 1), 2.0), STONER))

    init, apply = spinlattice.nve(energy_fn, cell, params, dt_fs / md.TIME_UNIT_FS, tol=1e-5)
    apply = jax.jit(apply)
    state = init(jax.random.PRNGKey(0), jnp.array(positions), canted, mass=MASS,
                 shifts0=start.shifts)

    traj = bov.Trajectory(cell, names=("Fe",))
    traj.append(state, time=0.0)
    e0 = float(state.energy)
    for step in range(1, steps + 1):
        state = apply(state)
        if step % every == 0:
            traj.append(state, time=step * dt_fs)
        if step % 40 == 0:
            total = float(state.energy + spinlattice.kinetic_energy(state))
            pairing = float(state.position[1, 2] - state.position[0, 2]) - 0.5 * A
            print(f"t = {step * dt_fs:6.1f} fs   E_tot - E0 = {total - e0:9.2e} eV   "
                  f"z1 - z0 - a/2 = {pairing:8.5f} A")
    return traj


def render(traj, path, **options):
    # the pairing of the planes, for the label: change of the distance between
    # planes 0 and 1 in milli-Angstrom
    for positions, _, attributes in traj.frames:
        attributes["pairing"] = 1e3 * (positions[1, 2] - positions[0, 2] - 0.5 * A)
    style = dict(
        size=(520, 300), fps=20, every=2, colors=48, replicate=(2, 1, 2), exaggerate=EXAGGERATE,
        arrows={"spin_moment": dict(bov.SPIN, scale=0.42, width=0.13)},
        legend={"spin_moment": "spin moments (planes canted by \u00b120\u00b0)"},
        label=("t = {time:5.1f} fs    pairing of the planes {pairing:+6.2f} m\u00c5"
               "    (displacements x %d)" % EXAGGERATE),
        camera_dir=(0.3, 1.0, 0.0), camera_up=(1.0, 0.0, 0.0), zoom=1.0, radius=0.3)
    style.update(options)
    return traj.render(path, **style)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--from-xyz" in sys.argv:
        trajectory = bov.Trajectory.read(OUT / "spin_lattice.xyz")
    else:
        trajectory = simulate()
        print("wrote", trajectory.write(OUT / "spin_lattice.xyz"))
    try:
        print("wrote", render(trajectory, str(OUT / "spin_lattice.gif")))
    except ImportError as error:
        print(error)
