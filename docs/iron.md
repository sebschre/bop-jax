# Iron

A magnetic d-band BOP for Fe (`bopjax.models.iron`, `IRON_FIT`, `IRON_CONFIG`):
d electrons only, collinear Stoner magnetism, exponential bond integrals
(`v0_bond` without its short-range factor), an exponential pair repulsion and
square-root embedding, 9 moments, cutoff 3.4 A (bcc first and second
neighbours, close-packed first neighbours).

Parameters: `IRON_FIT`, fitted with `examples/fit_iron.py` (bounded least
squares, 10 parameters, about 2 hours on 4 cores). Full report:
`uv run python examples/iron_properties.py --figure iron.png` (about 15 minutes).

## Results

### Structural stability (each phase at its own equilibrium, hcp with relaxed c/a)

| Phase | a (A) | V (A^3) | E - E(bcc FM) (meV/atom) | moment (muB) | target (meV) |
|---|---|---|---|---|---|
| bcc FM | 2.854 | 11.62 | 0 | 2.43 | 0, 2.2 muB |
| hcp NM | 2.470, c/a 1.600 | 10.44 | **67** | 0 | 65 |
| fcc NM | 3.472 | 10.46 | 107 | 0 | 130 |
| fcc AFM | collapses to fcc NM at its equilibrium | | 107 | 0 | 100 |
| bcc AFM | 2.813 | 11.13 | 200 | 1.75 | |
| bcc NM | 2.773 | 10.67 | 309 | 0 | 480 |

The order matches DFT: bcc FM is the ground state, and hcp NM comes next at
67 meV/atom, then fcc, the bcc antiferromagnet and non-magnetic bcc. The
hcp FM and AFM and fcc FM starts collapse to the non-magnetic states.

### Elastic constants (GPa)

| | C11 | C12 | C44 | C' | A = C44/C' | B |
|---|---|---|---|---|---|---|
| bcc FM, model | 250.7 | 130.1 | 119.5 | 60.3 | 1.98 | 170.3 |
| bcc FM, experiment (4 K) | 243 | 138 | 122 | 52.5 | 2.32 | 173 |

hcp NM (next stable phase, atoms relaxed in every strained cell): C11 = 735,
C12 = 277, C13 = 279, C33 = 798, C44 = 127, C66 = 229, B = 438 GPa. These
were not fitted and are far too stiff: DFT gives a bulk modulus of
about 280-300 GPa for hcp Fe at this volume. The non-magnetic phases of
the model are generally too stiff (Birch-Murnaghan B0 of 400-410 GPa for
bcc NM, hcp NM and fcc NM).

### Magnetic stability

- bcc: FM is 309 meV/atom below NM, AFM (B2 order) only 109 meV below NM,
  so ferromagnetic order is clearly preferred (by 200 meV/atom).
- fcc and hcp: non-magnetic at their equilibrium volumes. The fcc AFM
  moment appears on expansion, above 10.5 A^3, and reaches 1.9 muB at
  12.1 A^3.
- bcc FM moment against volume: 1.65 muB at 9.4 A^3, 2.25 at 11.1, 2.63 at
  12.1, saturating at 3.1 (the d-band limit for 6.88 electrons) above 13 A^3.
- bcc FM -> hcp NM under pressure: **11.3 GPa** (common tangent), volume
  11.16 -> 10.18 A^3; experiment 13 GPa, GGA about 10 GPa.

![Energy-volume curves and moments of the iron model](figures/iron_eos.png)

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
- The moment (2.43 muB) and the magnetic energy of bcc (309 meV against
  480) pull against each other in the Stoner model: a larger Stoner
  integral raises both. The fit compromises between them.
- fcc AFM has no moment at its equilibrium, where DFT gives it about
  1-1.5 muB, so fcc AFM and fcc NM are the same state.
- The non-magnetic phases are too stiff (see above); only bcc FM elastic
  constants were fitted.
- A bcc FM equation of state over a wide volume range has a kink where the
  moment saturates; use the elastic B = 170 GPa, not the Birch-Murnaghan
  value.
- The moment is the d moment of the Stoner model; there is no sp polarisation
  or orbital moment.
