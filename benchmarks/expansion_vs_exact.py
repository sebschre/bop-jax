"""Expansion settings against exact tight binding, for the iron-carbon model.

For each structure the band energy of the BOP expansion (non-magnetic, zero
onsite shifts) is compared with exact k-point diagonalisation of the same
Hamiltonian: the sum of the occupied eigenvalues.  Pair terms are identical
in both and drop out.  The figure of merit is how well energy *differences*
between structures come out (bulk iron, iron with carbon in either site,
diamond, graphene), so one fixed setting per species can be chosen that works
in bulk and around interstitials alike.

Run with:  uv run python benchmarks/expansion_vs_exact.py
"""

import dataclasses
import sys
import time

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples")
import bopjax as bj  # noqa: E402
from bopjax.geometry import bond_vectors  # noqa: E402
from bopjax.hamiltonian import bond_blocks, shell_slices  # noqa: E402
from bopjax.models import IRON_CARBON_CONFIG, iron_carbon, tapered_exp_sum  # noqa: E402
from _iron_carbon import bcc_fe, diamond, fe_with_carbon  # noqa: E402

FCC_PRIM = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]])


def fcc_fe(a=3.47):
    return a * FCC_PRIM, np.zeros((1, 3)), [0]


def graphene(a=2.4503):
    cell = np.array([[a, 0, 0], [-0.5 * a, 0.5 * np.sqrt(3) * a, 0], [0, 0, 8.0]])
    return cell, np.array([[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]]) @ cell, [1, 1]


STRUCTURES = {  # name: (cell, positions, species, k-mesh)
    "bcc Fe": bcc_fe() + ((14, 14, 14),),
    "fcc Fe": fcc_fe() + ((16, 16, 16),),
    "Fe16C oct": fe_with_carbon("octahedral") + ((5, 5, 5),),
    "Fe16C tet": fe_with_carbon("tetrahedral") + ((5, 5, 5),),
    "diamond": diamond() + ((12, 12, 12),),
    "graphene": graphene() + ((24, 24, 1),),
}

FE, C = IRON_CARBON_CONFIG["species"]
SETTINGS = {  # name: (iron settings, carbon settings)
    "current": (FE, C),
    "Fe jackson20+guard": (dataclasses.replace(FE, kernel="jackson", nexp=20,
                                               bandwidth_guard=True), C),
    "Fe jackson20+guard+osc": (dataclasses.replace(FE, kernel="jackson", nexp=20,
                                                   bandwidth_guard=True,
                                                   terminator="oscillation"), C),
    "Fe+C gershgorin jackson20": (
        dataclasses.replace(FE, kernel="jackson", nexp=20, bandwidth="gershgorin"),
        dataclasses.replace(C, bandwidth="gershgorin", bandwidth_guard=False)),
    "Fe+C jackson20+guard, C osc": (
        dataclasses.replace(FE, kernel="jackson", nexp=20, bandwidth_guard=True),
        dataclasses.replace(C, terminator="oscillation")),
}


def exact_band_energy(cell, pos, species, mesh, params, smearing=0.05):
    """Sum of occupied eigenvalues per atom, Fermi-Dirac smearing (eV)."""
    cfg = bj.BOPConfig(**IRON_CARBON_CONFIG)
    topo = bj.build_topology(pos, species, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    sp = np.asarray(species)
    rij = bond_vectors(jnp.asarray(pos), jnp.asarray(cell), topo)
    hblk = np.asarray(bond_blocks(rij, topo.nbr_mask, np.broadcast_to(sp[:, None],
                                  topo.nbr_origin.shape), sp[topo.nbr_origin],
                                  params["bond"], cfg.basis, cfg.rcut, cfg.dcut,
                                  tapered_exp_sum))
    rij = np.asarray(rij)
    slices, norb = shell_slices(cfg.basis)
    own = {0: [2], 1: [0, 1]}  # shells of Fe (d) and C (s, p)
    orbs = [[o for k in own[s] for o in range(*slices[k])] for s in sp]
    offsets = np.cumsum([0] + [len(o) for o in orbs])
    level = np.asarray(params["onsite"])  # (species, shell)
    shell_of = np.concatenate([[k] * (b - a) for k, (a, b) in enumerate(slices)])
    diag = np.concatenate([level[s][shell_of[o]] for s, o in zip(sp, orbs)])
    n_el = float(np.sum(np.asarray(params["n0"])[sp]))
    recip = 2 * np.pi * np.linalg.inv(cell).T
    grid = np.stack(np.meshgrid(*[(np.arange(m) + 0.5) / m - 0.5 for m in mesh],
                                indexing="ij"), -1).reshape(-1, 3)
    eigs = []
    for frac in grid:
        k = frac @ recip
        h = np.diag(diag).astype(complex)
        for i in range(len(sp)):
            for slot in np.flatnonzero(topo.nbr_mask[i]):
                j = topo.nbr_origin[i, slot]
                block = hblk[i, slot][np.ix_(orbs[i], orbs[j])]
                h[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]] += (
                    block * np.exp(1j * k @ rij[i, slot]))
        eigs.append(np.linalg.eigvalsh(0.5 * (h + h.conj().T)))
    eigs = np.concatenate(eigs)
    w = 2.0 / len(grid)  # spin degeneracy, k weight

    def count(ef):
        return w * np.sum(1.0 / (1.0 + np.exp(np.clip((eigs - ef) / smearing, -60, 60))))

    lo, hi = eigs.min() - 1, eigs.max() + 1
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if count(mid) < n_el else (lo, mid)
    ef = 0.5 * (lo + hi)
    occ = 1.0 / (1.0 + np.exp(np.clip((eigs - ef) / smearing, -60, 60)))
    return w * np.sum(occ * eigs) / len(sp)


def bop_band_energy(cell, pos, species, params, settings):
    cfg = bj.BOPConfig(**dict(IRON_CARBON_CONFIG, species=settings))
    topo = bj.build_topology(pos, species, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=tapered_exp_sum)
    _, aux = energy(jnp.zeros(energy.shift_shape), jnp.asarray(pos), jnp.asarray(cell), params)
    sp = np.asarray(species)
    e0 = np.asarray(params["onsite"])[sp]
    band = np.asarray(aux["energy_bond"]) + np.sum(e0 * np.asarray(aux["n_shell"]), axis=1)
    return float(np.sum(band)) / len(sp)


DIFFERENCES = {  # name: {structure: coefficient on its energy per atom}
    "fcc - bcc Fe": {"fcc Fe": 1, "bcc Fe": -1},
    "graphene - diamond": {"graphene": 1, "diamond": -1},
    "tet - oct (per cell)": {"Fe16C tet": 17, "Fe16C oct": -17},
    "C in Fe, oct (per cell)": {"Fe16C oct": 17, "bcc Fe": -16, "diamond": -1},
}


def main():
    params = iron_carbon()
    exact = {}
    for name, (cell, pos, sp, mesh) in STRUCTURES.items():
        t = time.time()
        exact[name] = exact_band_energy(cell, pos, sp, mesh, params)
        print(f"exact {name:10s} {exact[name]:10.4f} eV/atom  ({time.time() - t:.0f} s)",
              flush=True)
    for label, settings in SETTINGS.items():
        bop = {}
        for name, (cell, pos, sp, _) in STRUCTURES.items():
            try:
                bop[name] = bop_band_energy(cell, pos, sp, params, settings)
            except Exception as err:  # noqa: BLE001
                bop[name] = np.nan
                print(f"  {label} {name}: {err}", flush=True)
        print(f"\n{label}")
        for name in STRUCTURES:
            print(f"  {name:10s} BOP {bop[name]:10.4f}  exact {exact[name]:10.4f}  "
                  f"error {bop[name] - exact[name]:+.4f} eV/atom")
        for dname, terms in DIFFERENCES.items():
            b = sum(c * bop[s] for s, c in terms.items())
            x = sum(c * exact[s] for s, c in terms.items())
            print(f"  {dname:24s} BOP {b:+8.3f}  exact {x:+8.3f}  error {b - x:+.3f} eV")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
