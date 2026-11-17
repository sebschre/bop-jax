"""Magnetic anisotropy from the onsite spin-orbit term, and how it depends on
the number of moments.

    uv run python examples/spin_orbit_anisotropy.py

Two ferromagnets of the canonical d-band model with 7 electrons and a Stoner
integral of 0.6 eV, chosen because their moments are not saturated:

* a linear chain (spacing 2.2 A): uniaxial, E = K sin^2(theta), K ~ xi^2
* bcc (a = 2.87 A): cubic, E = K1 (a1^2 a2^2 + a2^2 a3^2 + a3^2 a1^2), K1 ~ xi^4

These are the numbers quoted in the README.  They show that the symmetry and
the order in xi come out right, and that neither constant is converged with
the number of moments.
"""

import jax.numpy as jnp
import numpy as np

import bopjax as bj
from bopjax.models import canonical_d

STONER = 0.6
BASE = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[STONER]]))
START = bj.stoner_shifts(np.full((1, 1), 2.0), STONER)
POS = jnp.zeros((1, 3))


def solver(cell, nmom, rcut):
    cfg = bj.BOPConfig(nmom=nmom, rcut=rcut, dcut=0.5, bandwidth="averaged",
                       spin="noncollinear", spin_orbit=True)
    topo = bj.build_topology(POS, [0], cfg.rcut, nmom, cell)
    scf = bj.make_scf(bj.make_energy_fn(topo, cfg))

    def solve(axis, xi, shifts0=START):
        params = dict(BASE, spin_axes=jnp.array([axis], dtype=float),
                      spin_orbit=jnp.array([[xi]]))
        return scf(POS, cell, params, shifts0=shifts0, tol=1e-12, max_iter=40)

    return solve


def difference(solve, axis, reference, xi, shifts0):
    a, b = solve(axis, xi, shifts0), solve(reference, xi, shifts0)
    return a.energy - b.energy if a.converged and b.converged else float("nan")


XIS = (0.025, 0.05, 0.1)

print("linear chain: K / xi^2 in 1/eV, K = E(across) - E(along)")
print("nmom  moment " + "".join(f"  xi={xi:<6}" for xi in XIS))
chain = jnp.diag(jnp.array([12.0, 12.0, 2.2]))
for nmom in (4, 5, 6, 7, 8, 9):
    solve = solver(chain, nmom, 3.3)
    plain = solve([0, 0, 1], 0.0)
    row = [difference(solve, [1, 0, 0], [0, 0, 1], xi, plain.shifts) / xi**2 for xi in XIS]
    print(f"{nmom:4d}  {float(plain.aux['moment'][0]):6.3f} " + "".join(f"  {k:9.5f}" for k in row))

print("\nbcc: K1 / xi^4 in 1/eV^3, K1 = 4 [E(110) - E(001)]")
print("nmom  moment " + "".join(f"  xi={xi:<7}" for xi in XIS))
bcc = jnp.array(0.5 * 2.87 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]]))
for nmom in (6, 8, 9, 10, 11, 12):
    solve = solver(bcc, nmom, 3.4)
    plain = solve([0, 0, 1], 0.0)
    row = [4.0 * difference(solve, [1, 1, 0], [0, 0, 1], xi, plain.shifts) / xi**4 for xi in XIS]
    print(f"{nmom:4d}  {float(plain.aux['moment'][0]):6.3f} " + "".join(f"  {k:10.2e}" for k in row))
