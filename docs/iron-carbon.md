# Iron-carbon

A BOP for Fe-C (`bopjax.models.iron_carbon`, `IRON_CARBON_FIT`,
`IRON_CARBON_CONFIG`, `radial=models.tapered_exp_sum`) built from two
elemental models: iron is the Fe model of Egorov et al.
([Iron](iron.md#the-egorov-et-al-fe-model-iron_egorov-iron_egorov_config):
d-valent, Stoner integral 0.751 eV, 200-term Jackson expansion), carbon the
sp model of [Carbon](carbon.md) (`CARBON_FIT`). Fe-Fe and C-C interactions
are exactly those of the two models (to 1e-13 eV); only the Fe-C terms are
new. As Egorov et al. did for Fe-Co, the elemental models are kept fixed
and only the cross terms are fitted.

## Approach

Two lessons from the Egorov iron model and the
[potential self-consistency](formulation.md#potential-self-consistency)
decided the approach, after the attempts described under
[History](#history-what-did-not-work):

- **A moderate Stoner integral and weaker dd bonds.** The earlier iron
  models (`IRON_FIT`, `IRON_SD_FIT`: Stoner 1.16 eV, dd-sigma about twice
  Egorov's) put iron next to carbon close to saturation, where the
  self-consistency has nearly flat directions. Egorov's iron keeps bulk-like
  moments in defect cores, and next to carbon (below) too.
- **The potential condition** (shifts = dE_dc/dN, Anderson mixing) instead
  of the stationary condition. The BOP bond energy is not exactly
  variational in inhomogeneous cells; near saturation the stationary point
  moves far away and Newton's method stalls, which is what every
  stationary attempt for Fe16C ran into. With the potential condition the
  tetrahedral Fe16C cell, which never converged before, converges in 25
  iterations (about 1 minute including compilation). Each converged state
  records its stationarity, max |dE/dshift| (about 0.08 eV in Fe16C at the
  starting parameters), to which the energy is off to first order.

The Fe-C bond couples C 2s and 2p to Fe 3d (`sds`; `pds`, `pdp`): d-sp,
because the carbon model is sp-valent and carbides have a C 2s - Fe 3d sigma
bond. A d-p model would have meant dropping the s shell of the carbon model.

## What it took in the code

- **Per-species shells and expansions** (`BOPConfig(species=...)`,
  `SpeciesSettings`). The basis is the union (s, p, d); iron carries only
  d, carbon only s and p. Hopping into absent shells is switched off,
  absent shells hold no electrons, and moments are walked only from each
  atom's own orbitals (6 start vectors per atom instead of 9). Iron keeps
  9 moments without kernel, carbon 10 moments with a 20-term Jackson
  expansion; both share one Fermi level. With these settings pure iron and
  pure carbon cells give the energies, moments and occupations of the
  elemental models to 1e-13 eV, including distorted cells.
- **One functional form for all pairs, with per-pair ranges**
  (`models.tapered_exp_sum`, `models.FECARBON_REPULSION`). Bond integrals
  are sums of three stretched exponentials (Egorov's form; carbon's single
  exponentials are one term); the pair term is Egorov's pair and core
  repulsion; the embedding density is a Gaussian (Egorov) plus an
  exponential (carbon). Each carries its own cosine taper: C-C 1.9-2.2 A,
  Fe-C 2.5-3.0 A (bcc octahedral site: 1.43 and 2.02 A; cementite:
  1.9-2.1 A; the next Fe shell of the octahedral site, 3.19 A, is
  outside), while Fe-Fe keeps Egorov's global cutoffs (4.1/1.0 A bonds,
  6.0/0.5 A repulsion). The Yukawa repulsion acts between iron atoms only.
- **One onsite shift per atom for carbon** (`atomic_shifts`, globally in
  `CARBON_CONFIG` and per species in `SpeciesSettings`). With one shift per
  shell, the shift of C 2s against 2p has no restoring force (the Hubbard
  term only sees the atomic charge), and for carbon in iron it has no
  stationary point for moderate Fe-C coupling: the residual grows as 2s is
  pushed down. Both carbon shells now move together; iron keeps one shift
  per shell, as fitted. Carbon was refitted (see [Carbon](carbon.md)).
- **Carbon not spin-polarized** (`SpeciesSettings(spin_polarized=False)`).
  Carbon has no Stoner term, so the splitting of its spin-up and
  spin-down levels had no restoring force either: its moment flipped
  between solutions during relaxation. It now has one shift for both
  spins; a moment is still induced through its iron neighbours.
- **Bandwidth guard** (`bandwidth_guard`). Next to iron, the band estimate
  of C 2s became too narrow for the 20-term expansion, which then diverged
  (2s held 2.3 electrons at the tetrahedral site). The guard widens it
  where needed; carbon was refitted with it.

## Fitting (`examples/fit_iron_carbon.py`)

The cross terms are fitted on [bopfox-rs](../bopfox-rs/README.md#iron-carbon),
the Rust port with analytic forces:

- `examples/_iron_carbon_rs.py` is the calculator, `RsCell`;
- `examples/fit_iron_carbon.py` is the fit;
- `examples/neb_iron_carbon_rs.py` checks the migration path.

Fe16C self-consistency with exact forces takes about 2 s on four cores from a
warm start, so every atom is relaxed in every evaluation (0.5-1.5 minutes,
warm-started). bopjax runs the same model (`examples/_iron_carbon.py`,
`Calculator`), but takes minutes per solve, 6-7 minutes for the exact forces
or the stress of Fe16C, and 12 GB.

### Protocol

Each evaluation of the fit:

- **Self-consistency and forces.** Potential self-consistency (shifts =
  dE_dc/dN), with exact implicit forces and stress. Partial derivatives at
  the converged shifts are off by up to 0.76 eV/A, see [History](#history-what-did-not-work).
- **Relaxations.** Fe16C cells (2x2x2 bcc, a = 2.8394 A, the equilibrium of
  the Egorov iron) and cementite (experimental lattice, Wood et al., J. Appl.
  Cryst. 37, 2004), with every atom relaxed at fixed cell.
  - The octahedral cell starts from slightly displaced positions, so that an
    unstable site shows up in the cost.
  - A relaxed Fe-C distance below 1.5 A is penalised.
- **Lowest magnetic state.** Iron next to carbon has several
  self-consistent magnetic states, eV apart in Fe16C, and a warm start keeps
  whichever it came from. At each iterate, `RsCell.ground_state` tries cold
  starts with several Fe moments and keeps the lowest. The Jacobian columns
  and trial steps follow that state by warm starts.
- **Migration saddle.** The barrier is the lowest energy in the plane half
  way between two octahedral sites: carbon is held along the path only,
  with the Fe centre held. The saddle lies in that plane, as the two sites'
  mirror symmetry requires, but off the tetrahedral point. This agrees with
  NEB (0.281 against 0.282 eV, and 0.844 against 0.843 eV at the fitted
  parameters). The symmetric tetrahedral site is a higher-order saddle in
  this model (`--tet-symmetric` fits to it instead).
- **Deterministic residuals.** Every evaluation starts from the relaxed state
  of the current iterate. Relaxations warm-started from wherever the previous
  evaluation ended reach slightly different minima, and a finite-difference
  Jacobian of those is noise.
- **`e_c` bounded below at 1.3 eV.** Relaxed cementite loses its Fe moments
  once the carbon levels come near the Fe d band: 0.6 muB at `e_c` = 0.5 eV,
  2.0 muB at 1.5 eV. Least squares walks into that jump and cannot climb back
  out.

The dipole tensor is -V times the stress of the relaxed octahedral cell minus
that of pure Fe16. 17 cross parameters, bounded least squares.

### Targets

| Property | Target | Tolerance | Source |
|---|---|---|---|
| solution energy of C, octahedral site of FM bcc Fe | 0.72 eV relative to diamond | 0.15 | 0.74 eV relative to graphite, DFT (Jiang and Carter, PRB 67, 214103 (2003)); diamond is 0.02 eV above graphite |
| migration barrier, octahedral to octahedral | 0.86 eV | 0.10 | DFT (Jiang and Carter); experiment 0.87 eV |
| dipole tensor of octahedral C, along / across the Fe-C-Fe axis | 8.0 / 3.4 eV | 1.5 | DFT (Clouet et al.), from memory, not checked |
| formation energy of relaxed cementite Fe3C relative to FM bcc Fe and diamond | +0.03 eV/atom | 0.04 | DFT and calorimetry give a small positive value; rounded, from memory |
| pressure of relaxed cementite at the experimental lattice | 0 | 5 GPa | |
| mean Fe moment of relaxed cementite | 1.95 muB | 0.15 | DFT, rounded |
| electrons gained by octahedral C | +0.3 | 0.4 | DFT (Bader) gives a gain of about half to one electron; the tight-binding charge is not a Bader charge, so the target only keeps the transfer moderate |

The tolerances are loose: a 16-atom cell is not the dilute limit, and three
of the targets were not checked against the papers.

## Results (`IRON_CARBON_FIT`)

Recomputed from scratch on the lowest magnetic states; the barrier is from
the NEB (five images, climbing image, every atom free). The fit ended at a
cost of 0.71.

| Property | `IRON_CARBON_FIT` | Target |
|---|---|---|
| E_sol, octahedral, relative to diamond (eV) | 0.735 | 0.72 +- 0.15 |
| migration barrier, NEB (eV) | 0.843 | 0.86 +- 0.10 |
| dipole tensor, axial / transverse (eV) | 7.36 / 3.82 | 8.0 / 3.4 +- 1.5 |
| formation energy of Fe3C (eV/atom), relaxed / experimental positions | 0.026 / 0.028 | +0.03 +- 0.04 |
| pressure of Fe3C (GPa), relaxed / experimental positions | -0.9 / -2.5 | 0 +- 5 |
| mean Fe moment of Fe3C (muB), relaxed / experimental positions | 1.93 / 1.91 | 1.95 +- 0.15 |
| electrons gained by octahedral C | 0.07 | +0.3 +- 0.4 |

- **Octahedral site.** The two axial Fe neighbours relax from 1.42 to
  1.60 A, the four transverse ones from 2.01 to 2.03 A. The axial Fe keep
  2.03 muB (bulk 2.87), and carbon carries -0.23 muB, antiparallel to iron,
  as in DFT.
- **Migration.** The saddle sits 0.33 A off the tetrahedral site, in both
  directions across the path (carbon at (0.615, 0.25, 0.115) a against
  (0.5, 0.25, 0) a). The symmetric tetrahedral site is 1.82 eV above the
  octahedral one.
- **Dipole tensor.** The anisotropy P axial - P transverse is +3.5 eV (DFT
  +4.6): the right sign at about three quarters of the size. The trace (the
  volume change) is 15.0 eV, against 14.8.
  - The strong short-range core (125 eV below 1.70 A) pushes the axial
    neighbours.
  - Gaussian Fe-C embedding densities at 2.00 A (width 0.20 A) bind the
    transverse ones without pushing on them.
  - The octahedral relaxation behind these numbers stopped at 0.04 eV/A
    (see [Limitations](#limitations)), so they may move by about 0.1 eV.
- **Cementite** is ferromagnetic and marginally metastable, as in experiment
  and DFT, both at the experimental positions and relaxed.

What follows for applications:

| Property | Depends on | With this model |
|---|---|---|
| solution energy, migration barrier, cementite | energetics | close to the targets |
| lattice expansion with carbon content, carbon at edge dislocations | trace of P (pressure coupling) | about right |
| martensite tetragonality, Snoek relaxation, carbon at screw dislocations | anisotropy of P | right sign; coupling to shear about three quarters of DFT's (Snoek strength about 0.6) |

## Cross parameters

| e_c | sds | pds | pdp | decay | decay_pi | pair | pair_decay | rho_fe | rho_c | hubbard_c | core | core_r | rho_gauss_fe | rho_gauss_c | rho_gauss_r | rho_gauss_w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.30 | -0.258 | -0.473 | 0.627 | 4.35 | 4.15 | 1.87 | 4.28 | 0.328 | 0.937 | 3.32 | 125 | 1.704 | 0.103 | 0.295 | 2.004 | 0.204 |

Energies are in eV and lengths in A. Bond integrals are given at r0 =
1.95 A; decays are dimensionless; `rho_decay` is fixed at 3; every Fe-C term
tapers off between 2.5 and 3.0 A.

- **Levels.** `e_c` puts the carbon levels (2s and 2p together) against the
  Fe d level. It sits 1.3 eV above it, on its bound.
- **Bond integrals.** `sds`, `pds`, `pdp` are the C 2s - Fe 3d and C 2p - Fe
  3d bond integrals, with the decay `decay` of the sigma bonds and
  `decay_pi` of the pi bond.
- **Pair terms.** `pair` and `pair_decay` give the Fe-C pair term. `core`
  and `core_r` give the short-range core, `core (core_r - r)^3 / r
  exp(core_r - r)` below `core_r`, Egorov's pair form.
- **Embedding densities.** A C neighbour adds `rho_fe` to the embedding
  density of Fe, and an Fe neighbour adds `rho_c` to that of C, both
  exponentials. On top come the Gaussians `rho_gauss_fe` and `rho_gauss_c`,
  `amplitude * exp(-((r - rho_gauss_r) / rho_gauss_w)^2)`.
- **Hubbard term.** `hubbard_c` is the onsite Coulomb integral of carbon,
  which does nothing in pure carbon.

## Limitations

- **Dipole anisotropy about three quarters of DFT's** (+3.5 against
  +4.6 eV), with the trace right: couplings to shear are somewhat weak.
- **The migration saddle is off the tetrahedral site.** The barrier is
  right, but the symmetric site is a higher-order saddle 1.82 eV up.
  Path-dependent details (the saddle's dipole tensor, the attempt frequency)
  were not fitted.
- **Several magnetic states.** Iron next to carbon has several
  self-consistent magnetic states, eV apart in Fe16C. Take results on the
  lowest (`RsCell.ground_state`, `relax_ground`); a warm start keeps
  whichever state it came from.
- **Cementite near a collapse.** `e_c` is on its bound (1.3 eV). Relaxed
  cementite is magnetic there, but not below about 1 eV. Strained or
  defective carbides near that boundary have not been checked.
- **Relaxations can stop short.** A step that lands on another magnetic
  state can end a relaxation early, so check `RsCell.max_force`.
- **Not the dilute limit, and unchecked targets.** The cells are Fe16C, and
  three targets were taken from memory. arxiv.org was not reachable from the
  session in which the fit was made, so they are still unchecked.
- **Self-consistency and forces.** The model needs the potential
  self-consistency and exact forces. Partial derivatives at fixed shifts are
  first order in the stationarity residual.
- **Carbon's own expansion** is far from its exact tight binding in
  graphene - diamond; see [Carbon](carbon.md) and the benchmark below.

## History: what did not work

### Fits that relaxed only part of each cell

The first fits ran on bopjax and relaxed only the breathing modes of the
first two Fe shells around carbon. Full relaxations with exact forces took
more than 4 hours there.

- **A partial-force fit** (10 parameters, 98 + 80 evaluations) was off by
  up to 0.76 eV/A in its forces.
- **An exact-force refit** (13 parameters, 10-25 minutes per evaluation,
  cost 16.6) reached a solution energy of 0.72 eV, a barrier (at the
  tetrahedral site) of 0.75 eV and cementite at +0.034 eV/atom. Its
  parameters:

  | e_c | sds | pds | pdp | decay | decay_pi | pair | pair_decay | rho_fe | rho_c | hubbard_c | core | core_r |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 2.56 | -0.250 | -0.558 | 0.655 | 4.15 | 4.37 | 1.49 | 5.69 | 0.364 | 0.713 | 9.79 | 2.05 | 1.70 |

  Two things were wrong with it:
  - **The dipole tensor was reversed:** 4.3 eV axial and 6.9 eV transverse,
    an anisotropy of -2.6 eV. A strong core turned it the right way
    (30 eV: 7.1 and 6.1 eV) but raised the octahedral site, and the barrier
    dropped to 0.47 eV. What was missing was a term binding the transverse
    neighbours: the Gaussian embedding densities of the present fit.
  - **The octahedral site is not stable.** Relaxed in full on bopfox-rs,
    carbon moves 0.5 A along the axis onto one axial Fe, which comes 0.18 A
    to meet it. That leaves Fe-C at 1.09 A, the Fe moment at 0.75 muB, and
    the solution energy at -1.7 eV. Its weak core (2 eV) cannot stop the
    short-range Fe-C bonding.

### Saddle at the tetrahedral site

- **Lattice sliding.** With only carbon held on the tetrahedral site, the
  Fe lattice slides rigidly around it into an octahedral site.
- **Symmetry breaking.** With the Fe centre held too, the Fe still break the
  site's D2d symmetry, about 0.4 eV lower.
- **The symmetric site is not the saddle.** A fit that symmetrised every
  step (`site_symmetry`, `--tet-symmetric`) had 0.84 eV at the symmetric
  site, but its NEB barrier was 0.28 eV: the path went around a
  higher-order saddle. Hence the saddle in the plane.

### Cementite at experimental positions

With cementite at the experimental positions, a fit went to `e_c` = -0.9 eV:
the carbon levels sat in the Fe d band. Relaxed, its cementite was
non-magnetic (0.05 muB, -22 GPa), and every start converged to that state.
Hence cementite relaxed in the fit, and the bound on `e_c`.

### Numerics in bopfox-rs

Carbon 2s lies 11 eV below the iron d band, so carbon's moments reached 1e10
and the moments-to-recursion map lost about 9 digits. The analytic
dE/dshift of carbon jumped by up to 0.3 under perturbations of 1e-12; bopjax
and finite differences agreed with each other. The implicit forces were off
by up to 0.1 eV/A and relaxations stalled. The walks are now centred on each
orbital's own level ([bopfox-rs](../bopfox-rs/README.md#iron-carbon)).

### Stationary self-consistency on the earlier iron models

The model was first built on the d iron model (`IRON_FIT`) and then on the
sd model (`IRON_SD_FIT`), with the stationary condition, solved by
chord-Newton on the shifts and a series of cold-start strategies.
Non-magnetic Fe-C cells converged reliably; ferromagnetic cells with
carbon did not, on the earlier iron models:

- With the d iron model: the octahedral Fe16C cell converged at some
  parameters (C -0.12 to -0.35 muB antiparallel to Fe, axial Fe neighbours
  reduced to about 2.2-2.4 muB), but the tetrahedral cell, with C 1.60 A
  from four Fe, had no stationary magnetic state above 0.8 of the Stoner
  integral, even with a continuation in it. Holding carbon neutral
  (Hubbard 8 eV) pushed the Fe moments to 2.7-2.8 muB, close to
  saturation, where the levels along the majority shifts are not unique;
  a weaker carbon Hubbard term (2 eV) let carbon take about four electrons.
- With the sd iron model: the tetrahedral cell got closer (residual
  1.3e-3 against 0.09) but still failed, and accepting small residuals moved
  the failure elsewhere rather than removing it.

Chord-Newton with Levenberg-Marquardt, block cold starts, starts from the
non-magnetic levels, a Stoner continuation and acceptance of residuals along
flat modes were tried (in the git history of `examples/_iron_carbon.py`).
What remained was a property of the functional rather than of the solver:
the truncated expansion does not satisfy the band-structure theorem
exactly, so near-degenerate directions (saturated majority bands) keep a
residual that no choice of levels removes. This is the delta of the
[potential self-consistency](formulation.md#potential-self-consistency).

### Expansion settings against exact tight binding (earlier iron models)

`benchmarks/expansion_vs_exact.py` compares the band energy of the
expansion (non-magnetic, zero shifts) with exact k-point diagonalisation of
the same Hamiltonian, at the cross parameters of the first fits. Errors, BOP minus
exact (eV/atom, or eV per cell where marked):

| Setting (iron / carbon) | bcc Fe | Fe16C oct | tet - oct (cell) | C in Fe, oct (cell) | graphene - diamond |
|---|---|---|---|---|---|
| current: 9 moments, no kernel / Jackson 20 + guard | -0.03 | -0.14 | -0.34 | -3.2 | -2.4 |
| iron Jackson 20 + guard | -0.23 | -1.70 | -1.09 | -26.4 | -2.4 |
| iron Jackson 20 + guard + oscillation terminator | -0.25 | -1.45 | -1.00 | -21.8 | -2.4 |
| both Gershgorin bandwidth, Jackson 20 | -0.34 | -1.58 | -0.87 | -22.8 | -3.3 |
| iron Jackson 20 + guard / carbon oscillation terminator | -0.23 | -1.70 | -1.09 | -26.4 | -2.3 |

A kernel on iron, with or without the guard, the oscillation terminator or
the Gershgorin bandwidth, makes iron next to carbon far worse: Fe16C is
overbound by 1.5-1.7 eV/atom, the spurious overbinding the self-consistent
runs with these settings showed (a non-magnetic state about 2 eV/atom too
deep). Iron's present settings are the best of those tried; what is left
with them (0.14 eV/atom in Fe16C, 0.34 eV in tet - oct) is the truncation at
9-10 moments. Carbon's expansion is far from its own exact tight binding in
graphene - diamond (2.4 eV/atom) with the strong sigma integrals of the
refit; as a fitted BOP it still reproduces its targets.

