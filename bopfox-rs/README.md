# bopfox-rs

Analytic bond-order potentials in Rust, with analytic forces: a port of
[bopjax](../README.md) that keeps its formulation but replaces automatic
differentiation by the derivative route of BOPfox.

Phase 1 of the migration: energy, forces, virial and dE/dshifts at given onsite
shifts. Self-consistency (Broyden on dE/dshifts) is phase 2.

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

## Supported (phase 1)

| | |
|---|---|
| basis | any combination of s, p, d; several species |
| bond integrals | `powerlaw`, `v0` (BOPfox v0), `exp_sum` (Egorov) |
| expansion | any `nmom` ≤ 14, `nexp`; bandwidths `azero`, `lowest`, `highest`, `avgan_maxbn`, `global`, `averaged`, `findeminemax`, `gershgorin`; terminators `none`, `constantabn`, `weighted`, `oscillation`; Jackson kernel |
| magnetism | none, collinear (Stoner) |
| electrostatics | onsite Hubbard `J` |
| repulsion | `v0`, `powerlaw`, `egorov` (pair + embedding) |
| outputs | energy (total and per atom), Fermi level, occupations, charges, moments, `a_inf`/`b_inf`, forces, virial `dE/dstrain`, `dE/dshifts` |

Not yet: self-consistency, non-collinear spins, spin-orbit, magnetic fields,
screening, Yukawa repulsion, intersite Coulomb, `kT > 0`, Beer-Pettifor-Aoki
bandwidth, cell-list neighbour search (the search is O(N²), still only 62 ms
for 1372 atoms).

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

## Use

Rust:

```rust
use bopfox_rs::{Bop, Model, Topology};
let model = Model::from_json(&std::fs::read_to_string("model.json")?)?;
let topo = Topology::build(&pos, &species, cell.as_ref(), model.rcut, model.nmom, model.rcut2);
let res = Bop::new(&model, &topo).compute(&pos, &cell, &shifts, true);
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
```

The model JSON uses the field names of `bopjax.BOPConfig` and the bopjax
parameter dictionaries (`src/model.rs`).
