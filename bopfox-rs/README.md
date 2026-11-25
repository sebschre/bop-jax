# bopfox-rs

Analytic bond-order potentials in Rust, with analytic forces: a port of
[bopjax](../README.md) that keeps its formulation but replaces automatic
differentiation by the derivative route of BOPfox.

Done so far:
- **Phase 1:** energy, forces, virial and dE/dshifts at given onsite shifts.
- **Phase 2:** self-consistency on BOPfox's condition dE/dshifts = 0, from
  gradients only.
- **Then:**
  - bopjax's potential self-consistency condition, with implicit-function
    forces;
  - an ASE calculator with warm-started self-consistency;
  - non-collinear magnetism with optional spin-orbit coupling;
  - external magnetic fields with analytic magnetic moments;
  - spin dynamics and spin-lattice dynamics;
  - the remaining model terms: d-d screening, Yukawa repulsion, intersite
    Coulomb, electronic temperature, the Beer-Pettifor-Aoki bandwidths;
  - NEB and the first-order field effect on barriers;
  - a cell-list neighbour search;
  - the iron-carbon model (`bopjax.models.iron_carbon`): per-species shells
    and expansions, the bandwidth guard, tied and unpolarized shifts, the
    tapered bond integrals and repulsion (see [below](#iron-carbon)).

## Self-consistency

`Bop::scf` (Python: `Potential.scf`) solves dE/dshifts = 0 by Anderson
(Pulay) mixing of preconditioned gradient steps. It needs no Hessian and no
Hessian-vector products. On the linearised problem it converges like GMRES,
at one gradient evaluation per iteration (`src/scf.rs`):

- **Cheap gradient.** The gradient comes from a mode that skips the
  bond-block derivatives: one forward walk plus a backward pass that only
  forms the onsite derivatives.
- **Preconditioner.** The gradient is `g = chi (target - shifts) + delta`,
  so its Jacobian is close to `J0 = chi K chi - chi`, with `K` the onsite
  Hubbard and Stoner kernel. `chi` is modelled as
  `-(Q - q q^T / sigma)`, where:
  - `Q` is the *exact* response of each atom's occupations to its own levels
    at fixed Fermi level: one tangent (forward-mode) walk per shell,
    refreshed every `refresh` = 4 evaluations;
  - the rank-one term keeps the electron count fixed.

  `J0 step = -g` is solved exactly in O(N) (Sherman-Morrison), or by GMRES
  on the free levels when some atoms are frozen (`free`).
- **What the preconditioner captures.** Measured Jacobians showed that
  neither ingredient can be dropped:
  - the DOS at the Fermi level underestimates the local response by a third
    (interband terms), and the error enters `J0` squared;
  - with `J` = 8 eV the fixed-N coupling between atoms is as large as the
    diagonal;
  - with Stoner magnetism the spin block of `J0` can change sign.
- **Safeguards.** Steps are capped at `max_step` (0.5 eV per level), the
  best iterate is kept, and the mixing restarts with half the step when the
  gradient grows tenfold.
- **Uniform shifts.** A uniform shift of all levels leaves the energy
  unchanged, so it is projected out and the mean of the shifts stays at its
  start, as with bopjax's minimum-norm Newton steps.

Returned: the shifts, `converged`, the gradient history, and everything
`compute` returns at the converged shifts, forces included. These forces
are the total derivatives of the self-consistent energy (checked by finite
differences of the SCF energy to 1e-8).

### Potential condition

`ScfOptions::potential()` (Python: `scf(condition="potential")`) solves the
condition of `bopjax.make_potential(condition="potential")` instead:
shifts = K N(shifts). This is the usual charge and spin self-consistency of
tight binding. It stays well conditioned where moments saturate and the
stationary point is far away (clusters, surfaces).

- **Solver.** The same Anderson mixing, preconditioned by `(1 + K P)^{-1}`,
  the Newton step of the linearised condition, from the same response model.
  The defaults are mixing 0.3 and history 8: near saturation, full steps and
  long histories fail.
- **Forces.** The energy is not stationary there, so forces, virial and
  torques carry the response of the shifts. They are obtained by implicit
  differentiation: `dE/dR = dE/dR|_x - u^T dr/dR` with
  `(dr/dx)^T u = dE/dx`, `r = K N - x`.
  - The adjoint system is solved by GMRES, right-preconditioned with the
    transpose of the same preconditioner.
  - Each application of `(dr/dx)^T u = chi^T K u - u` is one backward pass
    seeded with the occupations instead of the energy (`seeded_backward`
    with `wbond = 0`). It reuses the stored forward walk and forms no bond
    derivatives.
  - `u^T dr/dR` costs one more pass, with the bond derivatives.
  - Typical: 10-20 operator applications.

### Warm starts

`Bop::scf_continued` (Python: `scf(reuse_history=True)`) starts the Anderson
mixing with the history of the previous solve. In molecular dynamics, with
linearly extrapolated starting shifts, 16-atom bcc iron at 600 K needs 8.6
evaluations per step at tol 1e-9 against 34 from a cold start, and 3.8 at
tol 1e-7. Starting from the previous shifts alone, without the history,
needs 21.

## ASE calculator

```python
from bopfox_rs.calculator import BOPCalculator
atoms.calc = BOPCalculator.from_bopjax(cfg, params, hamiltonian.v0_bond,
                                       condition="stationary", tol=1e-8)
```

- **Properties:** energy, free energy, per-atom energies, forces, stress,
  magnetic moments and charges of the self-consistent state, periodic or
  finite.
- **Warm starts:** the first calculation starts from the initial magnetic
  moments, as an exchange splitting -/+ I m / 2. Later ones start from the
  previous shifts, linearly extrapolated, and continue the Anderson history.
  The start is reset when the atoms change.
- **Non-collinear models:** the spin axes are the directions of vector
  initial moments (or `axes=`). `magmoms` are vectors, and
  `results["axes_gradient"]` holds dE/daxes; the torque on spin i is
  `-e_i x dE/de_i`.
- **Tests** (`tests/test_ase.py`):
  - forces and stress against ASE's numerical derivatives, for both
    conditions;
  - NVE energy conservation and the warm-start saving;
  - BFGS relaxation of a vacancy;
  - a non-collinear spin-orbit case with its magnetic anisotropy.

Install the calculator's dependency with `pip install "bopfox-rs[ase]"`, or
add `ase` to the environment.

## Non-collinear magnetism

`"spin": "noncollinear"` puts the spinor Hamiltonian of
`bopjax.bop.moments_noncollinear` on a complex walk (`src/spinor.rs`):

- **The walk.** Each atom has a local spin axis `e_i`
  (`Bop::with_axes`; Python `axes=`). The hopping is spin-diagonal, and the
  onsite block is `sum_s level_s P_s(e_i)`, plus `xi L.S` with
  `"spin_orbit": true` and `"spin_orbit_xi"` per species and shell. The
  moments of the local spin channels are projected from 2x2 moment
  matrices.
- **Derivatives.** The backward pass is the real one in complex arithmetic.
  It gives dE/dH_b for forces and virial, and the onsite cotangent, from
  which dE/dshifts and the onsite part of dE/daxes follow. The rest of
  dE/daxes comes from the projections. The SCF preconditioner's local
  response comes from a spinor tangent walk.
- **Fast path.** Without spin-orbit coupling the onsite blocks are
  orbital-diagonal, and the walk uses that.
- **Model scope.** As in bopjax, the local-axis ansatz holds: the exchange
  field of each atom stays along its axis.
- **Virial.** With spin-orbit coupling and fixed axes, a rigid rotation of
  the lattice alone changes the energy, so the virial is the symmetrised
  strain derivative.

## Magnetic fields

`magnetic_field` (Tesla, one vector or one per atom; Rust
`Bop::with_magnetic_field`) adds bopjax's Zeeman term
`-mu_B B.(L + g/2 sigma)`. `spin_field` (`h_i.sigma`) and `orbital_field`
(`b_i.L`), in eV, are its lower-level sources (`Bop::with_fields`).

- **Spin, along each local axis.** It moves the up and down levels by
  `-/+ h.e` and adds `(h.e) m` to the energy, so the shifts remain the
  internal potentials.
- **Spin, across the axis, and orbital** (non-collinear only). `h_perp.sigma`
  and `b.L` enter the spinor onsite block. Collinear models couple only
  `h_z`.
- **Moments.** The backward pass returns dE/dh and dE/db, so the moments
  `-(1/mu_B) dE/dB` come out analytically: `spin_moment` (g/2 <sigma>),
  `orbital_moment` (<L>) and `magnetic_moment`, as from
  `bopjax.make_magnetic_moment`. They are available at zero field as well.
- **Torques** include the field terms, and with the potential condition the
  moments include the response of the shifts.
- **Checks:**
  - finite differences of dE/dh, dE/db, forces, shifts and torques in a
    field, with and without spin-orbit coupling;
  - the self-consistent moment against finite differences of the
    self-consistent energy in the field, for both conditions;
  - bopjax at 10 T: energy, forces, torques, moments and self-consistent
    shifts.

## Spin dynamics

`bopfox_rs.spin` ports `bopjax.spin` and `bopjax.spinlattice` to work on the
Rust self-consistency. Units are femtoseconds and amu (bopjax's time unit is
`TIME_UNIT_FS` = 10.18 fs).

- **`SpinDynamics`:** Landau-Lifshitz-Gilbert dynamics of the spin axes at
  fixed positions. Heun steps are applied as rotations; there is Gilbert
  damping `alpha` and stochastic torques at temperature `kT`, the
  fluctuation-dissipation pair. It reports torques, damping power,
  effective fields and the magnetisation.
- **`SpinLatticeDynamics`:** atoms and axes together. Velocity Verlet for
  the atoms, with a midpoint step of the axes solved by `correctors`
  fixed-point passes. Without damping `E + E_kin` is conserved.
- **Evaluations:** each one is a self-consistent solve, warm-started from
  extrapolated shifts and the previous Anderson history, in fields too.
- **Unconverged evaluations:** `on_failure` ("warn", "ignore", "raise")
  says what happens when an evaluation misses `tol`. The stationary
  condition can lack an exact root. In a distorted, canted 4-atom iron cell
  with the 7-moment variant of the iron model, both bopfox-rs and bopjax's
  Newton solver stall at about 1e-4 after a few steps. bopjax carries on
  silently; here such evaluations are counted (`evaluate.failures`).
  `condition="potential"` has no such failures, and the 9-moment model
  converges throughout.
- **Checks** (`tests/test_spin.py`):
  - **bopjax trajectories:** the axes agree to 1e-11 after 2 fs at time
    steps of 0.1, 0.05 and 0.025 fs. The energy drift is identical too; it
    is the Heun scheme's and falls 5-7 times per halving of the step.
  - **Larmor precession:** a rigid ferromagnet precesses at
    `g mu_B B / hbar` about the field, at constant angle.
  - **Damping:** the energy falls at the damping power.
  - **Thermal equilibrium (slow test):** stochastic torques sample
    exp(-E / kT): `<cos theta>` follows the Langevin function.
  - **Spin-lattice:** the trajectory matches `bopjax.spinlattice.nve`, and
    `E + E_kin` is conserved.

## Screening, Coulomb, Yukawa, temperature, bandwidths

- **Screening of the d-d bond integrals** (`"screening"`, bopjax's
  `screening_factors`).
  - Each bond's d-d integrals get a scale and shift from sums over the other
    neighbours of its central atom, and the ring term of the reverse bond.
  - The three-body derivatives are analytic: the bond blocks take scale and
    shift as extra dual inputs, and their cotangents go back through the
    factor formula and every pair term to the bond vectors.
- **Intersite Coulomb** (`"coulomb": "frauenheim"`, `"taper"`).
  - Frauenheim integrals between the charges of the atoms on the second
    list, times bopjax's taper.
  - They enter the energy, dE/dN, the forces and virial. For the potential
    condition they also enter the target, the adjoint operator (the full
    kernel) and the explicit R-dependence of the target in the implicit
    forces.
  - **Taper caveat:** bopjax's taper is a proper switch only for an inner
    radius R1 = 0. For R1 > 0 its coefficients give values in the hundreds
    between R1 and R2. The port keeps the same coefficients for comparison;
    use R1 = 0.
- **Yukawa repulsion** (`"yukawa"`): the environment-dependent form of
  bopjax/Egorov et al. Forces come in two passes: the pair derivatives, then
  the derivatives of the decay constants.
- **Electronic temperature** (`"kT"`, `"smearing_nodes"`). The filling is
  averaged over the Fermi kernel (Gauss-Legendre nodes in the Fermi
  function), and the bond energy is the Mermin free energy.
- **Beer-Pettifor-Aoki bandwidths** (`beerpettiforaoki`,
  `beerpettiforaoki_bopfox`).
  - The extreme eigenvalues come from a Jacobi solve of the BPA matrix, and
    their derivatives from the eigenvectors (Hellmann-Feynman).
  - The centre comes from bisection plus two Newton steps in dual
    arithmetic, which gives the exact derivative of the root.
- **Checks:**
  - finite differences of all of them (with two species, s/p/d bases,
    periodic and finite, and the potential condition with intersite
    Coulomb);
  - bopjax: screening with no, collinear and non-collinear spin; Egorov
    iron with Yukawa; kT = 0.1 eV self-consistently; both BPA variants;
  - bopjax's two-species intersite-Coulomb reference: energy to 1e-10,
    Coulomb energy, charges.

## NEB and field effects on barriers

- **`bopfox_rs.neb`** is bopjax's climbing-image NEB and FIRE relaxation:
  the same algorithm, giving bit-identical bands for the same forces.
  `image_forces` builds the force function from a model, with one
  warm-started self-consistency per image (its own shifts and Anderson
  history). It retries without the history, and then from the starting
  shifts, before reporting a failure.
- **`bopfox_rs.zeeman`** ports `first_order_energies`,
  `first_order_barrier` and `peierls_stress`, using the analytic moments.
- **Checks:**
  - the Mueller-Brown saddle;
  - a vacancy hop in bcc iron. With only the hopping atom free, the saddle
    is at the midpoint. Some images of this hop, in so small a cell, are out
    of reach for the stationary condition, in bopjax too; the potential
    condition converges them.
  - the first-order field derivative of the barrier against the barrier
    recomputed in +-5 T.

## Self-consistency solver fallback

When the Anderson mixing stalls (two restarts in a row without a tenfold
gain), the solver continues with Newton-Krylov steps (`newton_fallback`):
- GMRES, right-preconditioned with the same model Jacobian;
- Jacobian-vector products by forward differences of the residual, one
  evaluation each;
- a line search on the residual.

It converges, for example, a canted 4-atom iron cell where the mixing
stalled at 7e-3 and bopjax's exact Newton reaches 2e-13. The preconditioner
uses the full per-atom response, including one local spin channel's
response to the other's level (non-zero with canted neighbours or spin-orbit
coupling).

## Iron-carbon

`bopjax.models.iron_carbon` with `IRON_CARBON_CONFIG` (`radial=models.tapered_exp_sum`)
converts with `model_from_bopjax` and runs with both self-consistency
conditions; the potential condition is the one the model is fitted with
(`examples/_iron_carbon_rs.py`, `examples/fit_iron_carbon.py`).

- **Per-species settings** (`"species"`, one entry per species with `basis`,
  `nmom`, `nexp`, `bandwidth`, `terminator`, `kernel`, `bandwidth_guard`,
  `spin_polarized`, `atomic_shifts`, as `bopjax.SpeciesSettings`):
  - bond blocks into shells a species lacks are zero, absent shells hold no
    states;
  - each atom's walk starts only from its own orbitals (Fe 5, C 4 of the 9
    s, p, d orbitals), which made Fe16C 2.7 times faster than walking from
    all of them;
  - moments are walked to the model's `nmom` for every atom, and each
    species expands the first `nmom + 1` of them with its own settings.
- **Bandwidth guard** (`"bandwidth_guard"`, bopjax `guarded_width`): the
  extreme eigenvalues of the terminated chain come from a Jacobi solve, and
  their exact first derivatives from the eigenvectors.
- **Tied shifts** (`atomic_shifts`, `spin_polarized = false` per species):
  the shift layout stays `[atom][spin][shell]`. The energy sees `Pi x`, with
  `Pi` the per-atom projector that averages tied entries and zeroes absent
  shells; gradients are `Pi g` and the potential target is `Pi K N`.
  - The SCF preconditioner uses `Pi Q Pi` and `Pi K Pi`; the adjoint of the
    implicit forces is exact, with the seed `K Pi u`.
  - Converged shifts of absent shells are zero; bopjax leaves them at the
    Hubbard potential. Neither enters the energy.
  - The stationary condition is not supported with tied shifts (its
    preconditioner inverts `Q`), nor are tied shifts with non-collinear spin.
- **Centred walks.** Each start column walks `H - c`, with c the level of
  its own orbital, and the expansion is evaluated at `ef - c`. The reported
  moments are converted back to those of `H`.
  - Why: carbon 2s lies 11 eV below the iron d band, so carbon's uncentred
    moments reached 1e10 and the moments-to-recursion map lost about 9
    digits.
  - The effect: its dE/dshift jumped by up to 0.3 under perturbations of
    1e-12, and the implicit forces were off by up to 0.1 eV/A.
  - Scope: shifting is exact for a symmetric Hamiltonian
    (`tests/centred_moments.rs`), so it is applied only then: collinear or
    no spin, no screening, same-shell channels symmetric in the species. The
    deliberately asymmetric test models keep the plain walk.
  - Now the forces of Fe-C cells match finite differences of the
    self-consistent energy.
- **Forms:** `"tapered_exp_sum"` bond integrals and the `"fecarbon"`
  repulsion (Egorov pair and core, Gaussian plus exponential density, each
  with its own cosine taper).
- **Agreement with bopjax** (`test_iron_carbon`; Fe16C, cementite, diamond):
  - diamond agrees to 4e-13 eV;
  - cells with iron next to carbon agree to about 1e-8 eV in energy and
    1e-5 eV/A in forces. Each code matches its own finite differences to
    about 4e-6 eV/A, and the energies carry about 1e-9 eV of rounding noise:
    the 200-term iron expansion amplifies differences in summation order
    once Fe-C hopping is on (pure Fe16 agrees to 2e-12 eV).
  - self-consistent energies of the Fe16C and cementite cells agree to
    1e-7 eV or better.
- **Cost** (4 cores): Fe16C takes 0.09 s per energy evaluation and about 2 s
  for self-consistency with exact forces and stress from a warm start.
  bopjax needs minutes per solve, 6-7 minutes for the exact forces or the
  stress, and 12 GB.

## How the derivatives are obtained

- **Moments → bond blocks** (the expensive part) are differentiated by hand
  (`src/moments.rs`). The forward walk builds `w_a = H^a |i d>` on the cluster
  of atom `i` and evaluates `mu_k = <w_a|w_b>`. The backward pass carries adjoint
  vectors back through the walk. Each application of a bond block `H_b`
  contributes `wbar_{a+1} w_a^T` to `dE/dH_b`. This is the BOPfox sum over the
  interference paths through bond `b`, organised so that no path is
  enumerated. It costs about two forward walks.
- **The Fermi level** enters through a Lagrange multiplier:
  `dE/dmu = dF/dmu - lambda dN/dmu` with `lambda = (dF/def) / (dN/def)`.
- **Small per-bond and per-atom chains** use forward-mode dual numbers on
  scalar code written once for both values and derivatives (`src/real.rs`).
  These are the Slater-Koster block of one bond, moments → recursion
  coefficients → terminator → Chebyshev moments → filling and bond energy of
  one atom, and the pair terms. Nothing is taped and nothing is compiled at
  run time.
- **Stationarity.** The energy is stationary in the shifts, so at
  self-consistency the forces returned here are total derivatives.

## Supported

| | |
|---|---|
| basis | any combination of s, p, d; several species |
| bond integrals | `powerlaw`, `v0` (BOPfox v0), `exp_sum` (Egorov), `tapered_exp_sum` (iron-carbon) |
| species | per-species shells and expansion settings, tied (atomic, unpolarized) shifts |
| expansion | any `nmom` ≤ 14, `nexp`, optional bandwidth guard; bandwidths `azero`, `lowest`, `highest`, `avgan_maxbn`, `global`, `averaged`, `findeminemax`, `gershgorin`; terminators `none`, `constantabn`, `weighted`, `oscillation`; Jackson kernel |
| magnetism | none, collinear (Stoner), non-collinear with optional onsite spin-orbit coupling |
| bond integrals, environment | screening of the d-d integrals (three-body) |
| electrostatics, repulsion | onsite Hubbard `J`, intersite Frauenheim Coulomb; `v0`, `powerlaw`, `egorov`, `fecarbon` pair + embedding, Yukawa |
| electrons | `kT > 0` (Mermin free energy); all bandwidth estimators including Beer-Pettifor-Aoki |
| paths | climbing-image NEB, FIRE, first-order field effect on barriers, Peierls stress |
| self-consistency | stationary (BOPfox) and potential conditions, frozen atoms, warm starts |
| fields | Zeeman term of an external field, spin and orbital field sources; analytic spin, orbital and total magnetic moments |
| dynamics | LLG spin dynamics (damping, thermal torques), coupled spin-lattice dynamics |
| outputs | energy (total and per atom), Fermi level, occupations, charges, moments, `a_inf`/`b_inf`, forces, virial `dE/dstrain`, `dE/dshifts`, `dE/daxes`, `dE/dh`, `dE/db` |
| interfaces | Rust, Python (`Potential`), ASE (`BOPCalculator`) |

Everything in bopjax's feature table is now ported. The exceptions are
bopjax's JAX-specific interfaces (JAX-MD, `jax.grad` with respect to model
parameters for fitting) and OVITO rendering, which works on any trajectory.
Fitting stays in bopjax, whose parameter dicts `model_from_bopjax` converts.

Neighbour lists are binned (linear) whenever the cell holds three bins of
width rcut per lattice direction, and always for clusters; 6912 atoms take
0.27 s.

## Validation

- `cargo test --release` (`tests/reference.rs`):
  - BOPfox reference numbers for the canonical d band: the perfect-fcc log to
    1e-11 eV/atom, and the distorted fcc at BOPfox's self-consistent shifts.
    There the energy agrees to 1e-10, charges to 1e-8, forces to 1e-6 and
    dE/dshift is below 1e-7.
  - Five-point finite differences of forces, dE/dshifts and virial for d, and
    for two-species spd magnetic models with repulsion, periodic and finite,
    over all terminators and estimators listed above.
- `uv run pytest bopfox-rs/tests` (from the bop-jax root, after building the
  bindings): agreement with bopjax to 1e-9 in energy, per-atom energies, Fermi
  level, forces, virial, dE/dshifts and moments. Models: canonical d (32
  atoms), magnetic iron with d and sd bases, sp carbon (Jackson kernel,
  `nexp` 20), Egorov iron (`exp_sum`, `nexp` 200, without the Yukawa term).
- Self-consistency (`tests/scf.rs` and the `scf` tests in Python):
  - **BOPfox:** BOPfox's self-consistent shifts to 1e-9, charges to 1e-9,
    energy to 1e-10 and forces to 1e-6, in 9 gradient evaluations (BOPfox
    needs 12 Broyden steps).
  - **bopjax's Newton solver:** shifts and forces to 1e-7 and energy to
    1e-9 per atom, for canonical d (32 atoms), magnetic iron with d and sd
    bases (16 atoms) and metallic sp carbon.
  - **Other checks:** frozen atoms; the cheap gradient mode agrees bit for
    bit with the full evaluation.
- Potential condition:
  - **Finite differences of the self-consistent energy:** forces to 1e-7,
    virial to 1e-6. The partial forces differ from these by more than 1e-5.
  - **bopjax `make_potential(condition="potential")`:** bulk iron, and a
    15-atom iron cluster with surface moments near 3 μB. On the cluster the
    shifts agree to 2e-11, energy to 7e-12 and forces to 2e-8, and the
    implicit term moves the forces by up to 4.1 eV/Å. bopfox-rs needs 120
    evaluations there, bopjax 3000.
- Non-collinear (`tests/noncollinear.rs`, Python `test_noncollinear`):
  - **Collinear limit:** parallel axes reproduce the collinear results to
    1e-12, with zero axis gradient.
  - **Finite differences:** forces, virial, dE/dshifts and dE/daxes, with
    and without spin-orbit coupling, for d iron and a two-species spd
    cluster. At self-consistency the axis gradient is the total derivative
    for both conditions.
  - **bopjax:** energy, forces, dE/dshifts and dE/daxes to 1e-8, and the
    self-consistent shifts to 1e-7, with and without spin-orbit
    coupling.

## Performance

Energy and forces at fixed shifts, canonical d band, 9 moments, distorted fcc
(`cargo run --release --example scaling 7` against bopjax on
`benchmarks/scaling.py`'s setup). Same 4-core cloud machine, times exclude
bopjax's 10-30 s compilation:

| atoms | bopfox-rs energy | bopfox-rs energy + force | bopjax energy | bopjax energy + force |
|---:|---:|---:|---:|---:|
| 32 | 0.005 s | 0.013 s | 0.071 s | 0.170 s |
| 108 | 0.013 s | 0.035 s | 0.133 s | 0.453 s |
| 256 | 0.027 s | 0.077 s | 0.146 s | 1.518 s |
| 500 | 0.050 s | 0.172 s | 0.381 s | 2.442 s |
| 1372 | 0.144 s | 0.516 s | | |

The neighbour list and clusters take 8 ms for 256 atoms (bopjax: 2.3 s).

Where the speed comes from:
- sparse clusters with no padding;
- compile-time orbital counts for the 5×5 (and 1, 4, 6, 9) block loops;
- the forward walk kept for the backward pass (up to 512 MB, recomputed
  beyond);
- clusters one hop shell smaller than bopjax's for odd `nmom`: 309 instead of
  561 members for 9 moments. `mu_nmom = <w_a|w_{a+1}>` only needs `w_{a+1}`
  inside `a` hops.

### Self-consistent energy and forces

Same setup, from zero shifts to max |dE/dshift| < 1e-8, times including
the neighbour lists. bopfox-rs: `uv run python bopfox-rs/benchmarks/scaling.py 6`;
bopjax: `benchmarks/scaling.py`, `make_potential`:

| atoms | bopfox-rs scf + force | gradient evaluations | bopjax scf + force |
|---:|---:|---:|---:|
| 32 | 0.14 s | 14 | 3.85 s |
| 108 | 0.48 s | 15 | 16.8 s |
| 256 | 1.25 s | 16 | 67.0 s |
| 500 | 2.81 s | 17 | |
| 864 | 5.89 s | 20 | |

The energies per atom agree to all ten printed digits, and the number of
evaluations hardly grows with size.

Magnetic bcc iron (`IRON_CONFIG`, distorted, from a Stoner-split start, same
tolerance):

| basis | atoms | bopfox-rs | evaluations | bopjax | energy difference |
|---|---:|---:|---:|---:|---:|
| d | 16 | 0.38 s | 28 | 15.2 s | 9e-14 eV |
| d | 54 | 1.96 s | 44 | 60.5 s | 3e-13 eV |
| sd | 16 | 1.01 s | 42 | 34.0 s | 4e-13 eV |
| sd | 54 | 4.98 s | 55 | 152 s | 1e-13 eV |

Iron needs more evaluations than the d band. The preconditioner leaves out the
inter-site exchange coupling, which the Anderson history (24 steps) has to
learn.

Non-collinear bcc iron (`IRON_CONFIG`, 16 atoms, distorted, random canted
axes, from a Stoner-split start; bopfox-rs times include the neighbour
lists, called from Python):

| spin-orbit | bopfox-rs energy + force | bopjax energy + force | bopfox-rs scf + force | evaluations | bopjax scf + force | energy difference |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.13 s | 0.86 s | 1.28 s | 25 | 164 s | 6e-14 eV |
| xi = 0.06 eV | 0.09 s | 1.00 s | 1.21 s | 24 | 140 s | 9e-14 eV |

**Caveat (a property of the condition, not of the solver).** Where the
density of states at the Fermi level nearly vanishes, as in distorted
diamond with the sp carbon model, the stationary point is set by the
non-variational part of the BOP energy. There are then several stationary
points, eV apart in the s-p splitting. bopjax's Newton solver stalls there;
bopfox-rs converges, but to whichever stationary point it reaches.

## Use

Rust:

```rust
use bopfox_rs::{Bop, Model, Topology};
let model = Model::from_json(&std::fs::read_to_string("model.json")?)?;
let topo = Topology::build(&pos, &species, cell.as_ref(), model.rcut, model.nmom, model.rcut2);
let bop = Bop::new(&model, &topo);
let res = bop.compute(&pos, &cell, &shifts, true); // at given shifts
let scf = bop.scf(&pos, &cell, None, &ScfOptions::default()); // self-consistent
```

Python, from the bop-jax root:

```bash
cd bopfox-rs && VIRTUAL_ENV=../.venv uvx maturin develop --release && cd ..
```

```python
import bopjax as bj, bopfox_rs
from bopjax import hamiltonian
from bopjax.models import IRON_CONFIG, iron

cfg = bj.BOPConfig(spin="collinear", **IRON_CONFIG)
pot = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, iron(), hamiltonian.v0_bond))
out = pot.compute(positions, species, cell, shifts)  # shifts: (N, nspin, nshell)
out["energy"], out["forces"], out["shift_gradient"]
scf = pot.scf(positions, species, cell, shifts0)   # self-consistent, forces included
scf["shifts"], scf["converged"], scf["energy"], scf["forces"]
pot.scf(positions, species, cell, shifts0, condition="potential")  # implicit forces

# non-collinear, with spin-orbit coupling
cfg = bj.BOPConfig(spin="noncollinear", spin_orbit=True, **IRON_CONFIG)
params = dict(iron(), spin_orbit=np.array([[0.06]]))
pot = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, params, hamiltonian.v0_bond))
out = pot.scf(positions, species, cell, shifts0, axes=axes)
out["axes_gradient"]
```

The model JSON uses the field names of `bopjax.BOPConfig` and the bopjax
parameter dictionaries (`src/model.rs`).
