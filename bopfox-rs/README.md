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
  - non-collinear magnetism with optional spin-orbit coupling.

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
| bond integrals | `powerlaw`, `v0` (BOPfox v0), `exp_sum` (Egorov) |
| expansion | any `nmom` ≤ 14, `nexp`; bandwidths `azero`, `lowest`, `highest`, `avgan_maxbn`, `global`, `averaged`, `findeminemax`, `gershgorin`; terminators `none`, `constantabn`, `weighted`, `oscillation`; Jackson kernel |
| magnetism | none, collinear (Stoner), non-collinear with optional onsite spin-orbit coupling |
| self-consistency | stationary (BOPfox) and potential conditions, frozen atoms, warm starts |
| electrostatics | onsite Hubbard `J` |
| repulsion | `v0`, `powerlaw`, `egorov` (pair + embedding) |
| outputs | energy (total and per atom), Fermi level, occupations, charges, moments, `a_inf`/`b_inf`, forces, virial `dE/dstrain`, `dE/dshifts`, `dE/daxes` |
| interfaces | Rust, Python (`Potential`), ASE (`BOPCalculator`) |

Not yet:
- external magnetic fields (Zeeman terms, the orbital moment);
- spin dynamics (the axis gradient it needs is there);
- screening, Yukawa repulsion, intersite Coulomb;
- `kT > 0`;
- the Beer-Pettifor-Aoki bandwidth;
- a cell-list neighbour search (the search is O(N²), still only 62 ms for
  1372 atoms).

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
