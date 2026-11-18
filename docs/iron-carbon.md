# Iron-carbon

A BOP for Fe-C (`bopjax.models.iron_carbon`, `IRON_CARBON_FIT`,
`IRON_CARBON_CONFIG`) built from the two elemental models: iron is the
magnetic d model of [Iron](iron.md) (`IRON_FIT`), carbon the sp model of
[Carbon](carbon.md) (`CARBON_FIT`). Fe-Fe and C-C interactions are exactly
those of the two models; only the Fe-C terms are new and fitted.

## Why d-sp

The iron model has only a d shell and the carbon model s and p shells, so
the Fe-C bond couples Fe 3d to C 2s and 2p: `sds` (sigma), `pds` and `pdp`
(sigma and pi). A d-p model (no C 2s) would have meant dropping the s shell
of the carbon model and refitting it; with d-sp the carbon model stays
sp-valent, and the C 2s - Fe 3d sigma bond, which the bonding of carbides
has, is kept.

## What it took in the code

- **Per-species shells and expansions** (`BOPConfig(species=...)`,
  `SpeciesSettings`). The basis is the union (s, p, d); iron carries only
  d, carbon only s and p. Hopping into absent shells is switched off,
  absent shells hold no electrons, and moments are walked only from each
  atom's own orbitals (5 start vectors per atom instead of 9). Iron keeps
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
- **One onsite shift per atom** (`atomic_shifts`). With one shift per
  shell, the shift of C 2s against 2p has no restoring force (the Hubbard
  term only sees the atomic charge), and for carbon in iron it has no
  stationary point for moderate Fe-C coupling: the residual grows as 2s is
  pushed down. Both carbon shells now move together. Iron has one shell,
  so nothing changes there; carbon was refitted (see [Carbon](carbon.md)).
- **Bandwidth guard** (`bandwidth_guard`). Next to iron, the band estimate
  of C 2s became too narrow for the 20-term expansion, which then diverged
  (2s held 2.3 electrons at the tetrahedral site). The guard widens it
  where needed; carbon was refitted with it.

## Self-consistency (`examples/_iron_carbon.py`)

The 17-atom cells are solved by chord-Newton on the shifts (one dense
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
non-magnetic levels plus a Stoner splitting. A cold magnetic Fe16C takes
about 5-7 minutes on 4 cores, a warm one seconds; a relaxation of Fe16C
from the ideal site about 50 FIRE steps.

## Results

(after the fit)

## Targets (`examples/fit_iron_carbon.py`)

| Property | Target | Tolerance | Source |
|---|---|---|---|
| solution energy of C, octahedral site of FM bcc Fe | 0.72 eV relative to diamond | 0.15 | 0.74 eV relative to graphite, DFT (Jiang and Carter, PRB 67, 214103 (2003)); diamond is 0.02 eV above graphite |
| E(tetrahedral) - E(octahedral) | 0.86 eV | 0.10 | the migration barrier, DFT (Jiang and Carter); experiment 0.87 eV |
| dipole tensor of octahedral C, along / across the Fe-C-Fe axis | 8.0 / 3.4 eV | 1.5 | DFT (Clouet et al.), from memory, not checked |
| formation energy of cementite Fe3C relative to FM bcc Fe and diamond | +0.03 eV/atom | 0.04 | DFT and calorimetry give a small positive value; rounded, from memory |
| pressure of cementite at the experimental lattice | 0 | 5 GPa | |
| mean Fe moment of cementite | 1.95 muB | 0.15 | DFT, rounded |

Fe16C cells (2x2x2 bcc, a = 2.854 A, the equilibrium of `IRON_FIT`), FM,
atoms relaxed at fixed cell, C held on the tetrahedral site (the saddle
point). The dipole tensor is -V times the stress of the relaxed cell
minus that of pure Fe16. Cementite is at the experimental lattice and
positions (Wood et al., J. Appl. Cryst. 37, 2004) during the fit and
relaxed at fixed cell for the report. The tolerances are loose: a 16-atom
cell is not the dilute limit, and three of the targets were not checked
against the papers.

## Fitted parameters

`e_c` (C levels against Fe d), `sds`, `pds`, `pdp` and their common decay
`decay` at r0 = 1.95 A, the Fe-C pair term (`pair`, `pair_decay`), and the
embedding densities a C neighbour adds on Fe (`rho_fe`) and an Fe
neighbour on C (`rho_c`), decaying with `rho_decay` (fixed at 3).
