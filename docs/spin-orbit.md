# Spin-orbit coupling

`BOPConfig(spin="noncollinear", spin_orbit=True)` adds the onsite term
`xi L.S` to the spinor Hamiltonian, with `params["spin_orbit"]` of shape
(nspecies, nshell) holding `xi` in eV (`L` in units of hbar, `S = sigma / 2`;
the `j = l + 1/2` and `j = l - 1/2` levels of a free atom are
`xi (2l + 1) / 2` apart). The onsite block of the moment walk becomes a
matrix in (orbital, spin); the hopping stays diagonal in spin. With the flag
off the code path is the previous one and the BOPfox comparisons are
unchanged. With the flag on, an energy with its gradients at fixed onsite
levels costs 5 to 8 % more (the 4-atom non-collinear cell of the
[validation](validation.md), `xi` = 0.05 eV, 4-core cloud machine: 28 against
30 ms at 6 moments, 158 against 165 ms at 9). A self-consistent evaluation
is dominated by the Newton steps, whose number changes either way (3.0
against 4.4 s at 6 moments, 7.1 against 6.9 s at 9).

The term ties the spin axes to the lattice. The energy is no longer invariant
under a rotation of all axes, only under a common rotation of axes and
positions, so the torques on the axes no longer sum to zero: their sum is
minus the mechanical torque `sum r x F` on the atoms.

**Orbital moment.** `bj.make_orbital_moment(energy_fn)` returns
`orbital_moment(shifts, positions, cell, params)`, the expectation value
`<L>` of each atom in units of hbar. It is the derivative of the energy with
respect to a field `b_i` coupled by an onsite term `b_i.L`
(`params["orbital_field"]`, normally absent) at `b = 0`. At self-consistent
shifts this is the derivative of the self-consistent energy, as for the
forces. It is an observable only: the energy does not depend on it. For the
chain of the anisotropy tests below at `xi` = 0.05 eV it is 0.195 hbar along the chain and 0.111 hbar
across it, parallel to the spin for 7 electrons and antiparallel for 3 (Hund's
third rule), first order in `xi`, and largest along the easy axis.

BOPfox has no spin-orbit coupling, so the checks are internal
(`tests/test_spin_orbit.py`):

- `[Lx, Ly] = i Lz` and `L^2 = l(l+1)` for p and d in the real orbitals of
  the Slater-Koster blocks, and `exp(-i phi n.L)` carries the p-d hopping
  block of a bond into that of the rotated bond, which fixes orbital order
  and sign conventions.
- The moments up to the seventh of a p-d trimer with tilted axes and
  different `xi` on the two shells equal the diagonal elements of powers of
  an explicitly built 48 x 48 spinor matrix to 1e-11.
- A ferromagnetic chain (canonical d band, 7 electrons, Stoner integral
  0.6 eV, spacing 2.2 A, 9 moments, moment 1.29) is isotropic without
  spin-orbit coupling to 1e-11 eV. With it the self-consistent energy is
  `K sin^2(theta)` in the angle to the chain (half the barrier at 45 degrees
  to 0.5 %, no dependence on the direction normal to the chain), with
  `K / xi^2` = 0.114, 0.115, 0.116 per eV at `xi` = 0.0125, 0.025, 0.05 eV:
  second order. At `xi` = 0.05 eV, `K` = 0.29 meV per atom, easy axis along
  the chain.
- For a free trimer with tilted axes and `xi` = 0.1 eV the torques from
  `jax.grad` sum to 4e-3 eV (to 1e-12 without spin-orbit coupling), agree
  with finite differences of the self-consistent energy to 1e-8, and the sum
  is cancelled by `sum r x dE/dr` to 1e-11.
- The orbital moment is zero without spin-orbit coupling, equals the
  derivative of the self-consistent energy in a field `b.L` (finite
  differences, to 1e-7), and turns with a common rotation of lattice and
  spins.

Limits, all from `examples/spin_orbit_anisotropy.py`:

- **The anisotropy constants are not converged with the number of moments.**
  For the chain, `K / xi^2` is zero at 4 moments (the first term that
  depends on the direction is in the fifth moment), then -0.010, about 0,
  +0.014, +0.003 and +0.115 per eV at 5 to 9 moments; at 6 to 8 moments it
  does not follow `xi^2` between 0.025 and 0.1 eV. At 12 and 16 moments this
  chain is saturated and the self-consistency with `xi > 0` mostly fails
  (see [Caveats](limitations.md#caveats)). The tests therefore establish the symmetry and the order
  in `xi`, not a value.
- **Cubic bulk anisotropy is not resolved.** For bcc (a = 2.87 A, same
  model, moment 2.6) the fourth-order constant `K1` is zero to round-off
  (1e-13 eV) up to 8 moments. From 9 moments on it is non-zero and has the
  cubic form exactly (`K1` from [110], [111] and a general direction agree
  to four digits), but it is not converged: `K1 / xi^4` is -1e-2 per eV^3 at
  9 moments, +6e-2 to +1e-1 at 10, and changes sign with `xi` at 11 and 12.
  It follows `xi^4` only within a factor of two at 9 moments (-1.3e-2,
  -1.0e-2, -5e-3 at `xi` = 0.025, 0.05, 0.1 eV) and not at all beyond, partly
  because the self-consistent moment does not vary smoothly with `xi`
  (2.68, 2.63, 2.70 at 0.05, 0.1, 0.2 eV). At `xi` = 0.05 eV these values
  are 6e-8 to 2e-6 eV per atom. No cubic
  anisotropy from this model should be used.
- The local-axis (Stoner) ansatz is kept as it is, and it is only consistent
  with spin-orbit coupling to zeroth order in `xi`: the exchange field of an
  atom stays along its axis and the moment is the spin moment projected on
  it, while spin-orbit coupling induces a spin density perpendicular to the
  axis and an orbital moment, both of first order in `xi`. Neither is fed
  back, and the Stoner energy uses the projected moment only. The orbital
  moment can be evaluated as an observable (below).
  The axis is therefore a constrained direction of the exchange field, not
  the direction of the self-consistent magnetisation.
- For the same reason the angular momentum balance of a spin-lattice run
  would be between `hbar m e / 2` and the atoms only; the orbital angular
  momentum of the electrons is not tracked.
- The expansion is per shell and local spin channel, not per `j` multiplet:
  the moments are averaged over the real orbitals of a shell. This keeps
  the rotational invariance (the torque balance above) but is one more
  reason for the slow convergence.
