# bopjax

Prototype of the analytic bond-order potential (BOP) of BOPfox written in JAX.
The energy is one differentiable function; forces and the self-consistency
Jacobian come from automatic differentiation instead of hand-derived chains.

```bash
uv sync
uv run pytest
uv run python examples/fcc_distorted.py
```

## Formulation

`energy(shifts, positions, cell, params)` evaluates

    E = sum_i U_bond,i + sum_ia E0_ia (N_ia - N0_ia) + 1/2 sum_i J_i q_i^2

where `shifts` are the changes of the onsite levels (one per atom and shell),
`U_bond` and `N` come from the Chebyshev expansion of the local density of
states, and the Fermi level is fixed by the total electron count.

- **Self-consistency** is the stationary point `dE/dshifts = 0`. This is the
  same condition BOPfox solves (`BOhat - Nelec = 0`). It is solved by Newton's
  method with the exact Hessian `d2E/dshifts2` from `jacfwd(grad(E))`.
- **Forces** are `-dE/dpositions` at fixed shifts. No response term is needed
  because the energy is stationary in the shifts.
- **Parameter gradients** `dE/dparams` are available the same way.

## Layout

| File | Content |
|---|---|
| `src/bopjax/geometry.py` | neighbour list and supercell index map (NumPy), bond vectors (JAX) |
| `src/bopjax/hamiltonian.py` | Slater-Koster blocks for s, p, d; radial functions; cutoff |
| `src/bopjax/expansion.py` | moments -> recursion coefficients -> sigma, response functions |
| `src/bopjax/bop.py` | moments by repeated application of H; Fermi level; total energy |
| `src/bopjax/scf.py` | Newton self-consistency and forces |
| `src/bopjax/models.py` | canonical d-band parameters |

## Validation against BOPfox

Canonical d-band model (`W_LIM_I`, 6 moments), see `tests/test_bopfox_reference.py`.

| Case | Quantity | Agreement |
|---|---|---|
| perfect fcc | binding energy per atom | 1e-13 |
| perfect fcc | Fermi level, a_inf, b_inf | all printed digits |
| distorted fcc, self-consistent | binding energy per atom | 1e-12 |
| distorted fcc, self-consistent | onsite levels, charges | 1e-10 |
| distorted fcc, self-consistent | forces | 3e-7 |

For the distorted cell BOPfox needs 12 Broyden steps; Newton needs 3. The JAX
forces agree with a fourth-order finite difference of the self-consistent
energy to 1e-9, so the remaining force difference is on the BOPfox side.

## Not implemented

- magnetism (collinear or non-collinear)
- pair repulsion, embedding terms, intersite Coulomb `J_ij`, screening
- bandwidth estimates other than `lowest`, `highest`, `averaged`; terminators
  other than the constant one
- reading BOPfox `atoms.bx` / `bonds.bx` / structure files
- stress
- linear scaling: start vectors currently span the whole supercell, and the
  Hessian is dense

Bond integrals for mixed shells (sp, sd, pd) are checked against the
Slater-Koster table, for hermiticity and for rotation invariance, but have not
been compared with BOPfox. With the `lowest` bandwidth estimate the expansion
diverges when onsite levels of different shells are far apart compared with the
hopping, exactly as it would in BOPfox.
