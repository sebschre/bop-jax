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

## Calculator (`examples/_iron_carbon.py`)

`Calculator` wraps `make_potential(condition="potential")` with warm starts,
rejects states with an occupation outside a shell, and records the
stationarity. Exact forces (implicit differentiation, one GMRES adjoint
solve) cost about 100-230 s and 12 GB for Fe16C with `remat` and
`batch=4`. Partial derivatives at the converged shifts (`exact = False`,
5 s) are first order in the stationarity and were off by up to 0.76 eV/A
at the parameters of the partial-force fit (see [History](#partial-force-fit)),
so the present fit and its report use exact forces and stress throughout.
The breathing modes of the first two Fe shells around C relax in a few
minutes.

## Results (`IRON_CARBON_FIT`)

Fitted with `examples/fit_iron_carbon.py` with exact forces and stress: 13
cross parameters, bounded least squares, starting from the partial-force fit
below; about 10-25 minutes per evaluation, each cell in its own process.
Cost 16.9 at the start, 16.6 at the end (17 evaluations). FM Fe16C cells;
"shells" relaxes the breathing modes of the first two Fe shells around C,
as in the fit; "all atoms" relaxes every atom at fixed cell. Cementite at
the experimental lattice ("shells": experimental positions; "all atoms":
relaxed).

| Property | Shells (fitted) | All atoms | Target |
|---|---|---|---|
| E_sol, octahedral, relative to diamond (eV) | 0.717 | | 0.72 +- 0.15 |
| E(tetrahedral) - E(octahedral) (eV) | 0.746 | | 0.86 +- 0.10 |
| dipole tensor, axial (eV) | **4.29** | | 8.0 +- 1.5 |
| dipole tensor, transverse (eV) | **6.92** | | 3.4 +- 1.5 |
| formation energy of Fe3C (eV/atom) | 0.034 | | +0.03 +- 0.04 |
| pressure of Fe3C at the experimental lattice (GPa) | -6.5 | | 0 +- 5 |
| mean Fe moment of Fe3C (muB) | 2.11 | | 1.95 +- 0.15 |
| electrons gained by octahedral C | -0.06 | | +0.3 +- 0.4 |

(The all-atom column is being computed.)

- **Energetics.** The solution energy of octahedral carbon (0.72 eV), the
  formation energy of cementite (+0.034 eV/atom, marginally metastable as in
  experiment and DFT) and its Fe moment are close to their targets; the
  octahedral site is the stable one.
- **Migration barrier.** The tetrahedral site is 0.75 eV above the
  octahedral one (target 0.86).
- **Dipole tensor: wrong anisotropy**, see [Limitations](#limitations).
- **Magnetism and charge.** Iron next to carbon keeps a sizeable moment
  (bulk 2.86 muB), cementite averages 2.11 muB per Fe, and octahedral carbon
  carries a small moment antiparallel to iron, as in DFT. Carbon loses
  about 0.06 electron, where DFT charge analyses give it a gain; the
  tight-binding charge is not a Bader charge, and the target was loose.
- **Self-consistency.** Every cell converges with the potential condition.
  Energies are off from the stationary ones to first order in the
  stationarity; differences between similar cells (octahedral and
  tetrahedral) should be less affected than absolute energies.

## Limitations

**The elastic dipole tensor of octahedral carbon is reversed.** Carbon on
the octahedral site has two Fe neighbours along one cube axis (1.43 A) and
four in the plane across it (2.02 A). In DFT it pushes the two axial
neighbours hardest: P axial 8.0 eV, P transverse 3.4 eV, an anisotropy
P axial - P transverse of +4.6 eV. This model gives 4.3 and 6.9 eV,
an anisotropy of -2.6 eV, with a trace (the volume change) of 18.1 eV
against 14.8. What follows from that:

| Property | Depends on | With this model |
|---|---|---|
| solution energy, migration barrier, cementite | energetics | usable (see Results) |
| lattice expansion with carbon content, carbon at edge dislocations, drift to tensile regions | trace of P (pressure coupling) | right sign, about 20 % too strong |
| martensite tetragonality (Zener ordering of C on one octahedral sublattice) | sign of the anisotropy | **wrong sign**: c/a < 1 instead of > 1 |
| Snoek relaxation, carbon redistribution under uniaxial stress | anisotropy, squared for the strength | **wrong site preference** under stress; Snoek strength about (2.6/4.6)^2 = 1/3 of the real one |
| carbon at screw dislocations (strain ageing, pinning) | anisotropy only (a screw has no pressure field) | **wrong sites** around the core, binding about half as strong |
| elastic carbon-carbon interactions (clustering, ordering) | anisotropy | unreliable |

Use the model for carbon energetics, diffusion barriers, cementite and
carbon near vacancies or edge dislocations; not for martensite
tetragonality, Snoek damping, carbon at screw dislocations or carbon
ordering.

**Why, and what was tried.** The anisotropy has to come from the two close
neighbours at 1.43 A being pushed away much harder than the four at 2.02 A.
The refit separated the decay of the pi bond (`decay_pi`) from the sigma
bonds and added a short-range Fe-C core repulsion
(Egorov's core form, `core (core_r - r)^3 / r exp(core_r - r)` below
`core_r` = 1.70 A: about 0.018 `core` at the octahedral 1.43 A, 20 times
less at the tetrahedral 1.59 A). A weak core (2 eV, this fit) leaves the anisotropy
reversed. A strong one (30 eV, a second exact-force run, three days, cost
24 against 16.6) turns it the right way (axial 7.1, transverse 6.1 eV) but
also raises the octahedral site against the tetrahedral one, whose four
neighbours at 1.59 A barely feel it: the barrier dropped to 0.47 eV and
the solution energy rose to 0.81 eV. The core already separates 1.43 A from
1.59 A; what it cannot do is push the axial neighbours without raising
the octahedral energy. Compensating that needs a term that lowers the
octahedral site at the same time, such as more Fe-C bonding to the four
transverse neighbours at 2.02 A (a separate decay or onset for them) or an
angular term; not tried.

**Other limitations.**
- The migration barrier, 0.75 eV, is about 0.1 eV low.
- Carbon loses charge to iron where DFT has it gain some.
- Fe16C is not the dilute limit, and three of the targets were taken from
  memory, not checked against the papers (see [Targets](#targets-examplesfit_iron_carbonpy)).
- The model needs the potential self-consistency and exact forces; see
  [Calculator](#calculator-examples_iron_carbonpy).
- The carbon model's own expansion is far from its exact tight binding in
  graphene - diamond; see [Carbon](carbon.md) and the benchmark below.

## History: what did not work

### Partial-force fit

The first fit on the Egorov iron used partial-derivative forces and stress
(10 parameters, no `decay_pi` or core; two runs, 98 + 80 evaluations, cost
15.0). Its octahedral solution energy (0.715 eV) and tet - oct difference
(0.79 eV) held up with exact forces, but at its parameters the exact force at
the octahedral geometry the fit relaxed to was 0.76 eV/A where the partial one
was about zero, and its dipole tensor (5.8 axial, 7.5 transverse, from the
partial stress) was already reversed. The present fit starts from it.

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
the same Hamiltonian, at the starting cross parameters. Errors, BOP minus
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

## Targets (`examples/fit_iron_carbon.py`)

| Property | Target | Tolerance | Source |
|---|---|---|---|
| solution energy of C, octahedral site of FM bcc Fe | 0.72 eV relative to diamond | 0.15 | 0.74 eV relative to graphite, DFT (Jiang and Carter, PRB 67, 214103 (2003)); diamond is 0.02 eV above graphite |
| E(tetrahedral) - E(octahedral) | 0.86 eV | 0.10 | the migration barrier, DFT (Jiang and Carter); experiment 0.87 eV |
| dipole tensor of octahedral C, along / across the Fe-C-Fe axis | 8.0 / 3.4 eV | 1.5 | DFT (Clouet et al.), from memory, not checked |
| formation energy of cementite Fe3C relative to FM bcc Fe and diamond | +0.03 eV/atom | 0.04 | DFT and calorimetry give a small positive value; rounded, from memory |
| pressure of cementite at the experimental lattice | 0 | 5 GPa | |
| mean Fe moment of cementite | 1.95 muB | 0.15 | DFT, rounded |

Fe16C cells (2x2x2 bcc, a = 2.8394 A, the equilibrium of the Egorov iron),
FM, the breathing modes of the first two Fe shells around C relaxed during
the fit (all atoms at fixed cell for the report; a full relaxation needs
fresh levels after almost every step and costs about an hour), C held on
the tetrahedral site (the saddle point). The dipole tensor is -V times the stress of the relaxed cell
minus that of pure Fe16. Cementite is at the experimental lattice and
positions (Wood et al., J. Appl. Cryst. 37, 2004) during the fit and
relaxed at fixed cell for the report. The tolerances are loose: a 16-atom
cell is not the dilute limit, and three of the targets were not checked
against the papers.

## Cross parameters

| e_c | sds | pds | pdp | decay | decay_pi | pair | pair_decay | rho_fe | rho_c | hubbard_c | core | core_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.56 | -0.250 | -0.558 | 0.655 | 4.15 | 4.37 | 1.49 | 5.69 | 0.364 | 0.713 | 9.79 | 2.05 | 1.70 |

(eV, at r0 = 1.95 A; decays dimensionless; rho_decay fixed at 3, Fe-C range
2.5-3.0 A; core_r in A.) The carbon levels sit 2.6 eV above the iron d
level, the C 2p - Fe 3d pi bond (`pdp`) is larger than the C 2s - Fe 3d
sigma bond, and the carbon Hubbard term (9.8 eV) keeps carbon near neutral.

What the parameters are:

`e_c` (C levels against Fe d), `sds`, `pds`, `pdp`, with the decay `decay`
of the sigma bonds and `decay_pi` of the pi bond, at r0 = 1.95 A; the Fe-C
pair term (`pair`, `pair_decay`) and a short-range core of Egorov's pair
form (amplitude `core`, onset `core_r`); the embedding densities a C
neighbour adds on Fe (`rho_fe`) and an Fe neighbour on C (`rho_c`),
decaying with `rho_decay` (fixed at 3); and the onsite Coulomb integral of
carbon (`hubbard_c`), which does nothing in pure carbon. 13 parameters,
bounded least squares. One more target keeps carbon's charge moderate
(+0.3 +- 0.4 electrons in the octahedral site).
