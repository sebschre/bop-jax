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

### Potential self-consistency

`make_potential(..., condition="potential")` uses another condition: the
shifts equal the potentials their occupations imply,

    shifts = dE_dc / dN,   E_dc = 1/2 sum J_ii q^2 + 1/2 sum J_ij q_i q_j - 1/4 sum I m^2

(`energy.potential_target`, from `energy.double_counting` of
`aux["occupations"]`). This is the charge and spin self-consistency of
ordinary tight-binding codes, and it is solved by Anderson mixing.

Why both: the gradient of the energy is `g = dE/dshifts = chi (target - shifts) + delta`
with `chi = dN/dshifts`. For exact tight binding `delta = 0` and the two
conditions coincide; for the BOP, `delta` is the amount by which the bond
energy of the truncated moment expansion is not variational. It vanishes in
a perfect crystal (bcc iron: both conditions agree to 1e-12) but not in
inhomogeneous systems. Where moments approach saturation, `chi` nearly
vanishes, so the stationary point needs large changes of the shifts to
cancel `delta`; Newton's method then stalls or diverges. In a 61-atom iron
cylinder one Burgers vector long (`IRON_FIT`, 9 moments, surface moments
2.9-3.0 muB), Krylov-Newton steps stalled at max |g| ~ 1e-2 after 25 steps
(over 7 minutes), exact Newton steps from the eigendecomposition of the
Hessian (6 negative eigenvalues, condition number 1e4) and their
Levenberg-Marquardt damping stalled at the same level, while Anderson mixing
(mixing 0.1, 8 previous steps) reaches max |target - shifts| = 2e-10 in 37
iterations (27 s including compilation), with interior moments 2.40-2.49 muB (bulk 2.43).
At that point max |g| = 2.2e-2: the stationary point is not nearby.

The energy is not stationary in the shifts under this condition, so
gradients include the response of the shifts. They come from implicit
differentiation: with `r(shifts, theta) = target - shifts = 0`, the
cotangent of the shifts is carried back by one adjoint solve
`(dr/dshifts)^T u = shifts_bar` (GMRES with vector-Jacobian products) and
`theta_bar = -(dr/dtheta)^T u`. `dr/dshifts = K chi - 1` stays well
conditioned where `chi` vanishes. Forces, stress, parameter gradients and the
field derivatives of `zeeman.make_scf_magnetic_moment` all go through it
(`tests/test_scf.py`: forces and a Stoner-parameter gradient against finite
differences in a distorted cell with nearly saturated moments; in the iron
cylinder a force agrees with finite differences to 4e-8 eV/A).

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
