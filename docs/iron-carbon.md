# Iron-carbon

A BOP for Fe-C (`bopjax.models.iron_carbon`, `IRON_CARBON_FIT`,
`IRON_CARBON_CONFIG`) built from the two elemental models: iron is the
magnetic sd model of [Iron](iron.md) (`IRON_SD_FIT`), carbon the sp model of
[Carbon](carbon.md) (`CARBON_FIT`). Fe-Fe and C-C interactions are exactly
those of the two models; only the Fe-C terms are new.

**Status: the Fe-C cross parameters are not fitted.** The values in
`IRON_CARBON_FIT` are starting guesses. The machinery is in place and
tested, but magnetic self-consistency of carbon in iron is not robust
enough for a fit yet; see [Open problem](#open-problem-magnetic-self-consistency).

## Which coupling

The Fe-C bond couples C 2s and 2p to Fe 3d (`sds`; `pds`, `pdp`) and, with
the sd iron model, to Fe 4s (`sss`, `sps`). A d-p model (no C 2s) would
have meant dropping the s shell of the carbon model and refitting it; with
d-sp the carbon model stays sp-valent, and the C 2s - Fe 3d sigma bond,
which carbides have, is kept. The model was first built on the d iron
model (`IRON_FIT`, d-sp coupling only); it moved to the sd model because
without a 4s shell iron could only exchange charge with carbon through its
d band, and the magnetic self-consistency ran away (below).

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
- **Per-pair cutoffs** (`models.tapered_exp`, `models.TAPERED_V0`). Every
  bond integral, pair term and embedding density carries its own cosine
  taper: Fe-Fe 3.0-3.4 A, C-C 1.9-2.2 A, Fe-C 2.5-3.0 A (bcc octahedral
  site: 1.43 and 2.02 A; cementite: 1.9-2.1 A; the next Fe shell of the
  octahedral site, 3.19 A, is outside). The global cutoff is a hard step at
  3.4 A that no pair reaches.
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

## Self-consistency (`examples/_iron_carbon.py`)

Where it works, the 17-atom cells are solved by chord-Newton on the shifts (one dense
Hessian, built from Hessian-vector products in batches of four and reused
until convergence slows; Levenberg-Marquardt steps when Newton fails), from
the previous solution during relaxations and fits. Newton-Krylov on these
cells took about a minute per step. Cold starts are the delicate part:
carbon can take or give most of an electron, which the Hubbard term
answers with shifts of several eV, and joint Newton steps from zero land on
spurious stationary points (occupations outside the capacity of a shell,
energies hundreds of eV too high). The calculator therefore picks the
carbon shift from a scan over physically sane states, converges iron with
carbon fixed and carbon in capped steps, rejects any state with an
occupation outside its shell, and starts magnetic cells from the
non-magnetic levels plus a Stoner splitting, or by a continuation in the
Stoner integral. A cold non-magnetic Fe16C takes 1-4 minutes on 4 cores,
a magnetic one 5-20 minutes when it converges.

## Open problem: magnetic self-consistency

Non-magnetic Fe-C cells converge reliably, and pure iron and pure carbon
are unchanged. Ferromagnetic cells with carbon do not converge reliably
enough to fit:

- With the d iron model: the octahedral Fe16C cell converged at some
  parameters (C -0.12 to -0.35 muB antiparallel to Fe, axial Fe neighbours
  reduced to about 2.2-2.4 muB), but the tetrahedral cell, with C 1.60 A
  from four Fe, had no stationary magnetic state above 0.8 of the Stoner
  integral, even with a continuation in it. Holding carbon neutral
  (Hubbard 8 eV) pushed the Fe moments to 2.7-2.8 muB, close to
  saturation, where the levels along the majority shifts are not unique;
  a weaker carbon Hubbard term (2 eV) let carbon take about four electrons.
- With the sd iron model: the tetrahedral cell gets closer (residual
  1.3e-3 against 0.09) but still fails, and accepting small residuals moved
  the failure elsewhere rather than removing it.

The pieces that were tried are in `examples/_iron_carbon.py`: chord-Newton
with Levenberg-Marquardt, block cold starts, starts from the non-magnetic
levels, a Stoner continuation, rejection of states whose occupations exceed
their shells, and acceptance of residuals along flat modes. What remains
looks like a property of the functional rather than of the solver: the
truncated expansion does not satisfy the band-structure theorem exactly,
so near-degenerate directions (saturated majority bands, the charge of a
carbon atom whose 2p sits near the Fermi level) keep a residual that no
choice of levels removes.

Possible ways on, none tried yet:

- fit the cross terms with non-magnetic iron and use them as a starting
  point (the targets are ferromagnetic quantities, so only approximately);
- a kernel or bandwidth guard for iron next to carbon, as carbon needed;
  this changes the iron model and needs a refit of it;
- a larger cell (Fe54C), where fewer iron atoms are near carbon;
- local charge neutrality (a large Hubbard term on iron and carbon) with
  the moments constrained, as some transition-metal BOPs do.

## Targets (`examples/fit_iron_carbon.py`, prepared but not reached)

| Property | Target | Tolerance | Source |
|---|---|---|---|
| solution energy of C, octahedral site of FM bcc Fe | 0.72 eV relative to diamond | 0.15 | 0.74 eV relative to graphite, DFT (Jiang and Carter, PRB 67, 214103 (2003)); diamond is 0.02 eV above graphite |
| E(tetrahedral) - E(octahedral) | 0.86 eV | 0.10 | the migration barrier, DFT (Jiang and Carter); experiment 0.87 eV |
| dipole tensor of octahedral C, along / across the Fe-C-Fe axis | 8.0 / 3.4 eV | 1.5 | DFT (Clouet et al.), from memory, not checked |
| formation energy of cementite Fe3C relative to FM bcc Fe and diamond | +0.03 eV/atom | 0.04 | DFT and calorimetry give a small positive value; rounded, from memory |
| pressure of cementite at the experimental lattice | 0 | 5 GPa | |
| mean Fe moment of cementite | 1.95 muB | 0.15 | DFT, rounded |

Fe16C cells (2x2x2 bcc, a = 2.855 A, the equilibrium of `IRON_SD_FIT`),
FM, the breathing modes of the first two Fe shells around C relaxed during
the fit (all atoms at fixed cell for the report; a full relaxation needs
fresh levels after almost every step and costs about an hour), C held on
the tetrahedral site (the saddle point). The dipole tensor is -V times the stress of the relaxed cell
minus that of pure Fe16. Cementite is at the experimental lattice and
positions (Wood et al., J. Appl. Cryst. 37, 2004) during the fit and
relaxed at fixed cell for the report. The tolerances are loose: a 16-atom
cell is not the dilute limit, and three of the targets were not checked
against the papers.

## Cross parameters (to be fitted)

`e_c` (C levels against Fe d), `sss`, `sps`, `sds`, `pds`, `pdp` and their
common decay `decay` at r0 = 1.95 A, the Fe-C pair term (`pair`,
`pair_decay`), the embedding densities a C neighbour adds on Fe (`rho_fe`)
and an Fe neighbour on C (`rho_c`), decaying with `rho_decay` (fixed at 3),
and the onsite Coulomb integral of carbon (`hubbard_c`), which does
nothing in pure carbon. One more target keeps carbon's charge moderate
(+0.3 +- 0.4 electrons in the octahedral site).
