# Carbon

A non-magnetic sp-valent BOP for C (`bopjax.models.carbon`, `CARBON_FIT`,
`CARBON_CONFIG`): 2s and 2p shells, 4 valence electrons sharing one Fermi
level, one self-consistent onsite shift per atom for both shells
(`atomic_shifts`, see below), exponential bond integrals `sss`, `sps`, `pps`, `ppp` with a common
decay (`v0_bond` without its short-range factor), an exponential pair
repulsion and square-root embedding. 10 moments, a Chebyshev expansion of
length 20 with the Jackson kernel, cutoff 2.2 A with a 0.3 A taper (first
neighbours of every phase below, no second neighbours of diamond or graphene).

Parameters: `CARBON_FIT`, fitted with `examples/fit_carbon.py` (bounded least
squares, 10 parameters, about 10 s per evaluation on 4 cores). The first fit
(two runs, 287 + 109 evaluations) used one shift per shell; the present
parameters are a refit with atomic shifts started from it (172 evaluations,
stalled at cost 45.8 after about 140), made for the
[iron-carbon model](iron-carbon.md).

## Results

Each phase at its own equilibrium (one lattice parameter per phase; graphene
flat, the chain straight with equal bonds, lonsdaleite with ideal c/a and u).

| Property | Model | Target | Tolerance | first fit (shell shifts) |
|---|---|---|---|---|
| a diamond (A) | 3.568 | 3.567 | 0.005 | 3.565 |
| C11 diamond (GPa) | 970 | 1079 | 30 | 993 |
| C12 diamond (GPa) | 156 | 124 | 15 | 165 |
| C44 diamond (GPa), relaxed | 611 | 578 | 20 | 601 |
| bond graphene (A) | 1.404 | 1.42 | 0.01 | 1.408 |
| E graphene - E diamond (eV/atom) | +0.090 | +0.03 | 0.04 | +0.096 |
| bond chain (A) | 1.229 | 1.28 | 0.03 | 1.234 |
| E chain (eV/atom) | +0.69 | +1.2 | 0.2 | +0.77 |
| E sc (eV/atom) | +2.82 (bond 1.92 A) | +2.6 | 0.4 | +2.87 |
| E fcc (eV/atom) | **+5.45** (bond 2.08 A) | +4.0 | 0.5 | +3.92 |
| E bcc (eV/atom) | +4.32 (bond 1.99 A) | +3.5 | 0.5 | +4.41 |
| E lonsdaleite (eV/atom) | +0.012 | DFT about +0.025 | (not fitted) | +0.011 |
| cost (sum of squared residuals) | 45.8 | | | 31.9 |

Diamond is the ground state, with the right lattice parameter, bulk modulus
and C44 (with internal relaxation; the Kleinman relaxation is needed: the
unrelaxed C44 is 1237 GPa). Graphene comes 96 meV/atom above diamond,
close to degenerate as it should be but on the wrong side of the
experimental graphite-diamond difference: with a first-neighbour cutoff there
is no interlayer binding, so this compares with graphene, which experiment
puts about 30 meV above diamond. Lonsdaleite comes out slightly above diamond
(11 meV) without being fitted: the two have identical first and second
neighbour shells, so the difference comes from the eclipsed bonds and the
boat-shaped rings seen by the higher moments. The order of the dense phases,
sc < bcc < fcc, now has the right order but fcc 1.45 eV too high: with
atomic shifts the dense close-packed phase loses the s-p redistribution the
first fit had, and the refit did not recover it.

Electrons: diamond 2s1.23 2p2.77, graphene 2s1.17 2p2.83, the chain
2s1.12 2p2.88; sc, bcc and fcc fill the s shell more (1.56, 1.70, 1.96).

The fitted bond integrals at r0 = 1.54 A are `sss` = -2.91, `sps` = 7.17,
`pps` = 6.61, `ppp` = -0.78 eV, decaying as exp(-4.04 (r/r0 - 1)), with the
2s level 8.09 eV below 2p. Starting from the orthogonal tight-binding
integrals of Xu, Wang, Chan and Ho (J. Phys.: Condens. Matter 4, 6047
(1992)), the fit made the sigma bonds stronger and halved the pi integral
(`ppp`/`pps` = -0.12 against -0.28 there). The embedding density barely
decays across the first-neighbour range (`density_decay` = 0.49), so it acts
as a square root of the coordination: this is what lets three- and
two-fold coordinated carbon compete with diamond.

## One shift per atom (`atomic_shifts`)

Self-consistency adjusts onsite levels until the energy is stationary in
them. With one shift per shell, the shift of 2s relative to 2p is an
unknown of its own, and nothing but the truncated expansion fixes it: the
Hubbard term only sees the atomic charge. In the phases of carbon that
works (perfect diamond settles at -0.26/+0.26 eV, graphene at
-0.95/+0.95 eV, a distorted atom at -5.0/+0.6 eV), but for carbon in iron
the s-p shift has no stationary point at all for moderate Fe-C coupling
(the residual grows as 2s is pushed down). The model therefore uses
`BOPConfig(atomic_shifts=True)`: both shells of an atom move together and
only its charge is made self-consistent, as in self-consistent-charge
tight binding. With the first-fit parameters that raised diamond by
5 meV/atom and graphene by 31 meV/atom and cost 1.7 eV in fcc; the refit
above re-balances the rest.

## Targets (`examples/fit_carbon.py`)

| Property | Target | Tolerance | Source |
|---|---|---|---|
| a, diamond | 3.567 A | 0.005 | experiment, room temperature |
| C11, C12, C44, diamond | 1079, 124, 578 GPa | 30, 15, 20 | experiment (McSkimin and Andreatch, J. Appl. Phys. 43, 2944 (1972)) |
| bond length, graphene | 1.42 A | 0.01 | experiment (graphite, in plane) |
| E(graphene) - E(diamond) | +0.03 eV | 0.04 | graphite about 0.02 eV below diamond, interlayer binding about 0.05 eV |
| bond length, linear chain (cumulene) | 1.28 A | 0.03 | DFT, rounded |
| E(chain) - E(diamond) | +1.2 eV | 0.2 | DFT, rounded |
| E(sc), E(bcc), E(fcc) - E(diamond) | +2.6, +3.5, +4.0 eV | 0.4, 0.5, 0.5 | DFT, rounded |

The values for the chain and the cubic phases are rounded figures from memory
of the first-principles literature, with loose tolerances; they were not
re-checked against the papers (the network of the session that made this fit
could not reach them). The fit mainly asks that the order diamond ~ graphene
< chain < sc < bcc ~ fcc and the rough size of the gaps come out right.
Replace them with your own reference calculations where you have them.

## Settings that matter

- **Jackson kernel.** Diamond has a gap. Without a kernel the truncated
  expansion has negative density of states in it, the electron count is not
  monotonic in the Fermi level, and the Fermi level jumps between roots: the
  energy against lattice parameter has kinks and NaNs. The kernel keeps the
  density positive. Against exact tight binding (supercell diagonalisation
  with the starting parameters) the kernel costs about 0.6 eV/atom in the
  absolute diamond energy but follows the differences between phases. The
  constant terminator is no help here: with the averaged band estimate it
  diverges for diamond.
- **Cutoff.** A taper from 2.0 to 2.4 A let the second neighbours of
  compressed diamond into the cutoff and the fit collapsed diamond to the edge
  of its search range; 1.9 to 2.2 A keeps them out. The nearest neighbours of
  fcc (2.07 A) and the second neighbours of bcc lie in the taper, so the
  energies of those dense phases depend on the cutoff as much as on the
  bond integrals.
- **Moments.** The p-shell moments are averaged over px, py and pz, so in
  graphene the sigma and pi parts of the p density of states are not
  separated; the averaged density is exact in its moments, and 10 moments
  resolve it well enough for the energy differences above.

## Known limits

- Graphene 90 meV above diamond (target 30), and no interlayer binding: graphite
  is graphene.
- The chain is under-bound by 0.5 eV and its bond 0.05 A too short.
- C11 is 10 % low, C12 26 % high.
- fcc is 1.45 eV too high; fcc and bcc depend on the cutoff taper.
- Fixed shapes only: no relaxation of c/a, buckling, dimerisation of the
  chain (polyyne) or the dimer; no surfaces, defects or amorphous carbon were
  checked.
- Not magnetic; no charge transfer (every atom is equivalent in all fitted
  phases, so the Hubbard term does nothing there).
