# bopjax

The analytic bond-order potential (BOP) of BOPfox written in JAX. The energy is
one differentiable function; forces, stress, magnetic torques and the
self-consistency Jacobian come from automatic differentiation instead of
hand-derived chains.

```bash
uv sync                 # add --extra md for the JAX-MD interface
uv run pytest           # about 3 minutes: every configuration is compiled once
uv run python examples/fcc_distorted.py
```

## Formulation

`energy(shifts, positions, cell, params)` evaluates

    E = sum U_bond + sum E0 (N - N0) + 1/2 sum_i J_ii q_i^2
        + 1/2 sum_{i != j} J_ij q_i q_j - 1/4 sum I m^2 + E_rep

where `shifts` are the changes of the onsite levels (per atom and shell, and per
spin channel when magnetic), `U_bond` and `N` come from the Chebyshev expansion
of the local density of states, and the Fermi level is fixed by the total
electron count.

- **Self-consistency** is the stationary point `dE/dshifts = 0`. This is the
  condition BOPfox solves (`BOhat - Nelec = 0`). It is solved by Newton's
  method with the exact Hessian `d2E/dshifts2` from `jacfwd(grad(E))`.
- **Forces, stress, torques, parameter gradients** are plain partial
  derivatives at that point; no response term is needed because the energy is
  stationary in the shifts.
- `make_potential` wraps the self-consistency into one compiled function
  `E(positions, cell, params)` whose `jax.grad` is the exact force.

## Features

| Feature | Options |
|---|---|
| Basis | s, p, d shells in any combination, several species |
| Bond integrals | power law, BOPfox `v0`; any `radial(p, r)` callable |
| Bandwidth | `azero`, `lowest`, `highest`, `avgan_maxbn`, `averaged`, `global`, `findeminemax`, `gershgorin`, `beerpettiforaoki` |
| Terminator | `none`/`constantabn`, `oscillation`, `weighted`; Jackson kernel |
| Magnetism | `collinear`, `noncollinear` (fixed local spin axes, torques from the gradient) |
| Coulomb | onsite `J_ii`; intersite `frauenheim` with taper |
| Repulsion | pair term plus embedding (`pairwise.V0`, `pairwise.POWERLAW`, or your own) |
| Stress | `make_stress`, strain derivative of the self-consistent energy |
| Molecular dynamics | `bopjax.md` for [JAX-MD](https://github.com/jax-md/jax-md) |

## Layout

| File | Content |
|---|---|
| `src/bopjax/geometry.py` | neighbour lists and per-atom local clusters (NumPy), bond vectors (JAX) |
| `src/bopjax/hamiltonian.py` | Slater-Koster blocks for s, p, d; radial functions; cutoff |
| `src/bopjax/expansion.py` | moments -> recursion coefficients -> band estimate, terminator, sigma |
| `src/bopjax/pairwise.py` | repulsion, embedding and Coulomb functions |
| `src/bopjax/bop.py` | moments (scalar and spinor), Fermi level, total energy |
| `src/bopjax/scf.py` | Newton self-consistency (dense or Hessian-free), compiled potential, forces, stress |
| `src/bopjax/md.py` | JAX-MD interface |
| `src/bopjax/models.py` | canonical d-band and tungsten `v0` parameters |

## Performance

Moments are evaluated on a local cluster around each atom (the atoms within
half the path length), so the cost is linear in the number of atoms. For more
than 16 unknowns the Newton step is solved with GMRES on Hessian-vector
products instead of a dense Hessian; the Hessian is well conditioned, so about
ten products per step are enough.

Distorted fcc supercells, canonical d-band, 9 moments, Apple M1 Pro. Times
exclude compilation. Reproduce with `benchmarks/scaling.py`.

| Atoms | bopjax, one energy + force | bopjax, self-consistent energy + force |
|---|---|---|
| 4 | 0.005 s | 0.06 s |
| 32 | 0.03 s | 1.2 s |
| 108 | 0.09 s | 3.6 s |
| 256 | 0.31 s | 14.9 s |

The self-consistent column starts from zero shifts and converges to 1e-12 in
three Newton steps.

Compilation takes 10 to 30 s per configuration, and building the clusters is
done in Python (1.5 s for 256 atoms).

## Molecular dynamics

```python
from jax_md import simulate
from bopjax import md

energy = md.jaxmd_energy_fn(bj.make_energy_fn(topo, cfg), cell, params)
_, shift = md.jaxmd_space()
init, apply = simulate.nve(energy, shift, dt=1.0 / md.TIME_UNIT_FS)
```

Periodic images are handled by bopjax, so positions must not be wrapped: use
the shift from `jaxmd_space()`. Neighbour lists are static; build the topology
with a `skin` and rebuild when `bj.needs_rebuild(positions, topo)` is true (see
`examples/md_tungsten.py`).

## Validation against BOPfox

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

### Where the two codes differ

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

## Not implemented

- screening
- constrained-moment options of BOPfox (`scfFixMagCom`, `scfFixMagDiff`)
- bond integral and repulsion forms other than those listed; reading BOPfox
  `atoms.bx` / `bonds.bx` / structure files
- warm-starting self-consistency from the previous MD step (each step starts
  from zero shifts)
- dynamic neighbour lists inside a compiled MD loop (rebuilt from Python)

## Caveats

- Bond integrals for mixed shells (sp, sd, pd) are checked against the
  Slater-Koster table, for hermiticity and rotation invariance, but have not
  been compared with BOPfox.
- A strong ferromagnet with a completely filled spin-up band has extra flat
  directions; the solver handles them, but the onsite levels are then not
  unique. Non-collinear cases close to saturation can fail to converge.
- Timings are for one machine and one model; GPUs have not been tried.
