# Iron

A magnetic d-band BOP for Fe (`bopjax.models.iron`, `IRON_FIT`, `IRON_CONFIG`):
d electrons only, collinear Stoner magnetism, exponential bond integrals
(`v0_bond` without its short-range factor), an exponential pair repulsion and
square-root embedding, 9 moments, cutoff 3.4 A (bcc first and second
neighbours, close-packed first neighbours).

**Status: the fit is not finished.** `IRON_FIT` holds the best point reached
so far; run `examples/fit_iron.py` to continue it from there, and
`examples/iron_properties.py` for the full report.

## Targets (`examples/fit_iron.py`)

| Property | Target | Tolerance | Source |
|---|---|---|---|
| a, bcc FM | 2.855 A | 0.005 | experiment, low temperature |
| moment, bcc FM | 2.2 muB | 0.05 | experiment |
| C11, C12, C44 | 243, 138, 122 GPa | 8 | experiment, 4 K (Rayne and Chandrasekhar, Phys. Rev. 122, 1714 (1961)) |
| E(hcp NM) - E(bcc FM) | 65 meV | 10 | DFT-GGA, rounded literature value |
| E(fcc AFM) - E(bcc FM) | 100 meV | 12 | DFT-GGA, rounded literature value |
| E(fcc NM) - E(bcc FM) | 130 meV | 12 | DFT-GGA, rounded literature value |
| E(bcc NM) - E(bcc FM) | 480 meV | 30 | DFT-GGA, rounded literature value |
| V(hcp NM), V(fcc NM) / V(bcc FM) | 0.90, 0.91 | 0.02 | DFT-GGA |

The DFT values are rounded figures from the literature (e.g. Mrovec et al.,
PRL 106, 246402 (2011)) and were not re-checked against the papers; replace
them with your own reference calculations where you have them.

## What `examples/iron_properties.py` reports

1. all nine phases (bcc, fcc, hcp; FM, AFM, NM) at their equilibrium, hcp
   with relaxed c/a, ordered by energy;
2. the elastic constants of the ground state and of the next stable phase
   (hcp with internal relaxation);
3. magnetic energies of each lattice and the moment of bcc FM against volume;
4. the bcc FM -> hcp NM transition pressure (experiment 13 GPa).

## Known limits

- fcc AFM and the magnetic hcp states collapse to non-magnetic solutions;
  DFT gives fcc AFM a moment of about 1-1.5 muB.
- With the starting parameters C44 was 44 GPa. Faster-decaying second-neighbour
  interactions raise it, and the fit has moved there.
- The moment is the d moment of the Stoner model; there is no sp polarisation
  or orbital moment.
