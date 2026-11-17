# Validation against BOPfox

Distorted fcc cell, canonical d-band integrals, see `tests/`. "Energy" is the
binding energy per atom.

| Case | Compared | Agreement |
|---|---|---|
| perfect fcc, 6 moments | energy, Fermi level, a_inf, b_inf | 1e-13 |
| self-consistent, 7 bandwidth estimators | energy / charges | 1e-14 / 1e-12 |
| `global` band | energy, onsite levels, stress | 1e-14, 1e-9, 1e-8 |
| terminators and Jackson kernel (fixed levels) | energy | 1e-12 |
| dimer with repulsion and embedding | repulsive / bond energy | 1e-15 / 3e-9 |
| two species, onsite Coulomb | energy, charges, stress | 1e-15, 1e-8, 1e-8 |
| two species, intersite Coulomb | energy, charges, Coulomb energy | 1e-14, 1e-8, 1e-12 |
| collinear ferro- and antiferromagnet | energy, moments, levels, stress | 1e-14, 1e-3 (printed), 1e-8, 1e-8 |
| non-collinear (BOPfox rebuilt with `MAGNETISM = NCM`) | energy, levels, moments, forces | 2e-14, 1e-8, 1e-9, 2e-7 |
| forces, all cases | | 2e-7 to 3e-7 |

Forces agree with fourth-order finite differences of the self-consistent energy
to 1e-9, so the remaining force difference is on the BOPfox side.

## Where the two codes differ

- **Weighted and oscillation terminators with self-consistency.** Energies at
  fixed levels agree, but BOPfox resets the terminator's derivative terms to
  zero, so its self-consistent state differs slightly (charges by 1e-4).
- **`alphaweight`.** BOPfox gives the same result for any value; here it works.
- **Beer-Pettifor-Aoki.** BOPfox reads one element past its array (a zero),
  which makes the result depend on the energy origin, and it refuses
  self-consistency. `beerpettiforaoki` is the consistent estimator;
  `beerpettiforaoki_bopfox` reproduces BOPfox for comparison.
- **Stress with intersite Coulomb.** BOPfox leaves that term out; here the
  stress is the full strain derivative (checked by finite differences).
- **Magnetic torques.** BOPfox's printed torques do not match the derivative
  of its own energy (one atom prints exactly zero). Here torques are the
  energy gradient with respect to the spin axes, checked by finite differences.
- **Stoner energy with several shells.** Here `-1/4 I m^2` is summed per
  shell; BOPfox uses the summed integrals times the total moment squared. The
  two agree for a single shell, which is all that was compared.
