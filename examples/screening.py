"""Effect of screening on the d-d bond integrals and on structural energies.

Run with:  uv run python examples/screening.py

A valence s orbital on the atoms around a bond screens its d-d integrals, so
they depend on the environment and no longer on the bond length alone.  The
example compares first and second neighbours of bcc with the first neighbours
of fcc and of a simple cubic lattice, and then the bcc-fcc energy difference
with and without screening.

The screening functions below are made up for illustration (power laws of a
plausible size); they are not a fitted potential.
"""

import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax import hamiltonian as hm
from bopjax.geometry import bond_vectors, reverse_slots
from bopjax.models import canonical_d

RCUT, DCUT = 3.7, 0.4
VOLUME = 15.8  # per atom, close to tungsten

params = canonical_d()


def channel(prefactor, power):
    return jnp.array([[[prefactor, 2.5, power]]])


params["screening"] = {
    "sds": channel(-0.5, 3.5),
    "sds_overlap": channel(0.07, 3.0),
    "dds_overlap": channel(-0.04, 4.0),
    "ddp_overlap": channel(0.03, 4.0),
    "ddd_overlap": channel(-0.01, 4.0),
}

half = 0.5 * np.ones((3, 3))
LATTICES = {  # primitive cells of unit volume
    "bcc": (half - np.eye(3)) * 2.0 ** (1 / 3),
    "fcc": (half - 0.5 * np.eye(3)) * 4.0 ** (1 / 3),
    "sc": np.eye(3),
}


def lattice(name, volume=VOLUME):
    return LATTICES[name] * volume ** (1 / 3), np.zeros((1, 3))


print("Screened bond integrals, beta~ / beta, at one volume per atom")
print(f"{'':6}{'shell':>6}{'bonds':>7}{'r (A)':>8}{'sigma':>9}{'pi':>9}{'delta':>9}")
for name in LATTICES:
    cell, pos = lattice(name)
    topo = bj.build_topology(pos, [0], RCUT, 2, cell)
    rij = bond_vectors(jnp.array(pos), jnp.array(cell), topo)
    scale, shift = hm.screening_factors(
        rij, topo.nbr_mask, topo.species, topo.nbr_origin, reverse_slots(topo),
        params["screening"], RCUT, DCUT)
    r = np.linalg.norm(np.asarray(rij), axis=-1)[0]
    beta = np.stack([hm.powerlaw(params["bond"][c][0, 0], r) for c in ("dds", "ddp", "ddd")],
                    axis=-1)
    ratio = np.asarray(scale[0]) + np.asarray(shift[0]) / beta
    for shell, dist in enumerate(np.unique(np.round(r[topo.nbr_mask[0]], 6)), start=1):
        sel = topo.nbr_mask[0] & (np.abs(r - dist) < 1e-5)
        s, p, d = ratio[sel].mean(axis=0)
        print(f"{name:6}{shell:>6}{sel.sum():>7}{dist:8.3f}{s:9.3f}{p:9.3f}{d:9.3f}")


def energy_per_atom(name, volume, screening):
    cfg = bj.BOPConfig(nmom=9, rcut=RCUT, dcut=DCUT, screening=screening)
    cell, pos = lattice(name, volume)
    topo = bj.build_topology(pos, [0], cfg.rcut, cfg.nmom, cell)
    energy = bj.make_energy_fn(topo, cfg)
    # one atom per cell: neutral by symmetry, no self-consistency needed
    return float(energy(jnp.zeros((1, 1)), jnp.array(pos), jnp.array(cell), params)[0])


print("\nBond energy per atom (eV), 9 moments")
print(f"{'V (A^3)':>8}{'bcc - fcc':>12}{'screened':>12}{'bcc':>10}{'screened':>10}")
for volume in (14.0, 15.8, 17.5):
    e = {(n, s): energy_per_atom(n, volume, s) for n in ("bcc", "fcc") for s in (False, True)}
    print(f"{volume:8.1f}{e['bcc', False] - e['fcc', False]:12.4f}"
          f"{e['bcc', True] - e['fcc', True]:12.4f}"
          f"{e['bcc', False]:10.4f}{e['bcc', True]:10.4f}")
