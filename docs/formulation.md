# Formulation and code layout

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

## Layout

| File | Content |
|---|---|
| `src/bopjax/geometry.py` | neighbour lists and per-atom local clusters (NumPy), bond vectors (JAX) |
| `src/bopjax/hamiltonian.py` | Slater-Koster blocks for s, p, d; angular momentum matrices; radial functions; cutoff; screening |
| `src/bopjax/expansion.py` | moments -> recursion coefficients -> band estimate, terminator, sigma |
| `src/bopjax/pairwise.py` | repulsion, embedding and Coulomb functions |
| `src/bopjax/bop.py` | moments (scalar and spinor), Fermi level, total energy |
| `src/bopjax/scf.py` | Newton self-consistency (dense or Hessian-free), compiled potential, forces, stress |
| `src/bopjax/md.py` | JAX-MD interface |
| `src/bopjax/spin.py` | spin dynamics of the non-collinear spin axes |
| `src/bopjax/spinlattice.py` | coupled dynamics of atoms and spin axes |
| `src/bopjax/ovito.py` | trajectories for OVITO: extended XYZ and rendering |
| `src/bopjax/models.py` | canonical d-band and tungsten `v0` parameters |
