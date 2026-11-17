"""Animation of spin precession driven by spin-orbit coupling, with OVITO.

    uv run --extra viz python examples/ovito_spin_orbit.py     (about 2 minutes)

Two ferromagnetic chains of the canonical d-band model of
``examples/spin_orbit_anisotropy.py`` (7 electrons, Stoner integral 0.6 eV,
spacing 2.2 A, 9 moments), one atom per periodic cell, so that every atom of a
chain does the same: a macrospin.  Both start with the moments 60 degrees away
from the chain axis.

* Without spin-orbit coupling (lower chain) the energy does not depend on the
  direction of a uniform magnetisation: there is no torque, no orbital moment,
  and the moments stay where they are.
* With ``xi = 0.1 eV`` (upper chain) the chain axis is the easy axis.  The
  spin moments (red) precess about it and, with Gilbert damping
  ``alpha = 0.1``, spiral into it.  The orbital moment (blue, drawn 3 times
  longer) appears with the spin-orbit term.  It is parallel to the spin for
  this more than half filled d band, but larger along the chain than across
  it (0.37 against 0.21 hbar), so it is not exactly parallel to the spin
  while the moments are tilted.  The easy axis is the direction of the larger
  orbital moment, as Bruno's relation between anisotropy and orbital moment
  anticipates.

The orbital moment comes from ``bopjax.make_orbital_moment`` as the
derivative of the energy with respect to a field coupling to ``L`` on each
atom.  It is an observable only: like the rest of the local-axis ansatz the
energy does not feed it back (see the README).

Writes docs/animations/spin_orbit.xyz and, if the ovito module is installed,
docs/animations/spin_orbit.gif.  ``--from-xyz`` renders again from the saved
trajectory.
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax import md, spin
from bopjax import ovito as bov
from bopjax.models import canonical_d

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "animations"
STONER, SPACING, XI, ALPHA, TILT = 0.6, 2.2, 0.1, 0.1, 60.0
WIDTH = 8.0  # cell across the chain
ORBITAL_SCALE = 3.0


def chain(xi):
    """One chain with spin-orbit constant ``xi``: its position, starting axes and
    self-consistent shifts, the energy function, cell, parameters, and a
    function of an LLG state that returns the orbital moment."""
    cell = jnp.diag(jnp.array([WIDTH, WIDTH, SPACING]))
    position = jnp.zeros((1, 3))
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged", spin="noncollinear",
                       spin_orbit=True)
    topo = bj.build_topology(position, [0], cfg.rcut, cfg.nmom, cell)
    energy_fn = bj.make_energy_fn(topo, cfg)
    params = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[STONER]]),
                  spin_orbit=jnp.array([[xi]]))
    tilt = np.radians(TILT)
    axes = jnp.array([[np.sin(tilt), 0.0, np.cos(tilt)]])
    start = bj.make_scf(energy_fn)(position, cell, dict(params, spin_axes=axes),
                                   shifts0=bj.stoner_shifts(np.full((1, 1), 2.0), STONER),
                                   tol=1e-12)
    assert start.converged
    orbital = jax.jit(lambda state: bj.make_orbital_moment(energy_fn)(
        state.shifts, position, cell, dict(params, spin_axes=state.axes)))
    return position, axes, start.shifts, energy_fn, cell, params, orbital


def simulate(dt_fs=4.0, steps=1500, every=8):
    runs = []
    for xi in (XI, 0.0):
        position, axes, shifts, energy_fn, cell, params, orbital = chain(xi)
        init, apply = spin.llg(energy_fn, cell, params, dt_fs / md.TIME_UNIT_FS, alpha=ALPHA)
        runs.append((jax.jit(apply), orbital, init(position, axes, shifts)))

    # both chains in one picture: the one with spin-orbit coupling above
    sites = np.array([[0.75 * WIDTH, 0.5 * WIDTH, 0.0], [0.25 * WIDTH, 0.5 * WIDTH, 0.0]])
    traj = bov.Trajectory(np.diag([WIDTH, WIDTH, SPACING]), species=[0, 1], names=("Fe", "Fe0"),
                          pbc=(False, False, True))
    states = [state for _, _, state in runs]
    e0 = float(states[0].energy)
    for step in range(steps + 1):
        if step > 0:
            states = [apply(state) for (apply, _, _), state in zip(runs, states)]
        if step % every:
            continue
        spins = np.concatenate([np.asarray(s.moment)[:, None] * np.asarray(s.axes) for s in states])
        orbit = np.concatenate([np.asarray(orbital(s)) for (_, orbital, _), s in zip(runs, states)])
        theta = float(np.degrees(np.arccos(np.clip(states[0].axes[0, 2], -1.0, 1.0))))
        traj.append(positions=sites, spin_moment=spins, orbital_moment=orbit,
                    time=step * dt_fs / 1000.0, theta=theta,
                    delta_e_mev=1e3 * (float(states[0].energy) - e0))
        if step % (25 * every) == 0:
            print(f"t = {step * dt_fs / 1000:5.2f} ps  angle to the chain {theta:5.1f} deg  "
                  f"E - E0 = {1e3 * (float(states[0].energy) - e0):7.3f} meV  "
                  f"m = {float(states[0].moment[0]):.3f}  L = {orbit[0].round(3)}  "
                  f"(xi = 0: axis {np.asarray(states[1].axes[0]).round(6)}, "
                  f"|L| = {np.linalg.norm(orbit[1]):.1e})")
    return traj


def render(traj, path, **options):
    style = dict(
        size=(520, 370), fps=20, colors=48, replicate=(1, 1, 7), show_cell=False,
        arrows={"spin_moment": dict(bov.SPIN, scale=1.4, width=0.09),
                "orbital_moment": dict(bov.ORBITAL, scale=ORBITAL_SCALE, width=0.24)},
        legend={"spin_moment": "spin moment",
                "orbital_moment": "orbital moment (x %g)" % ORBITAL_SCALE},
        label=("t = {time:4.2f} ps"
               "\nupper chain: \u03be = 0.1 eV, {theta:4.1f}\u00b0 from the chain axis"
               "\nlower chain: no spin-orbit coupling"),
        color=[(0.65, 0.65, 0.68), (0.80, 0.80, 0.82)], radius=0.35,
        camera_dir=(-0.4, 1.0, 0.4), camera_up=(1.0, 0.0, 0.0), zoom=0.92)
    style.update(options)
    return traj.render(path, **style)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--from-xyz" in sys.argv:
        trajectory = bov.Trajectory.read(OUT / "spin_orbit.xyz")
    else:
        trajectory = simulate()
        print("wrote", trajectory.write(OUT / "spin_orbit.xyz"))
    try:
        print("wrote", render(trajectory, str(OUT / "spin_orbit.gif")))
    except ImportError as error:
        print(error)
