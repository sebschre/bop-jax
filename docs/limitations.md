# Limitations

## Not implemented

- environment-dependent (screened Yukawa) repulsion; screening of integrals
  other than d-d
- constrained-moment options of BOPfox (`scfFixMagCom`, `scfFixMagDiff`)
- bond integral and repulsion forms other than those in the
  [feature table](../README.md#features); reading BOPfox
  `atoms.bx` / `bonds.bx` / structure files
- spin-orbit coupling beyond the local-axis ansatz: the orbital moment and
  the spin density perpendicular to the axes are of first order in `xi` and
  are not fed back into the energy (see [Spin-orbit coupling](spin-orbit.md));
  spin-lattice dynamics with spin-orbit coupling, and with it the exchange of
  angular momentum between spins and lattice, has not been run
- longitudinal fluctuations of the moments; a thermostat for the atoms in
  spin-lattice dynamics
- an external field couples to the spin along each local axis self-consistently,
  but its perpendicular part and the orbital term only through the spinor
  Hamiltonian, within the local-axis ansatz ([Magnetic field](magnetic-field.md));
  there is no vector potential (no orbital effect of the field on the hopping)
- thermostats and barostats with warm start (only constant-energy dynamics
  carries the onsite levels; use JAX-MD's integrators otherwise)
- dynamic neighbour lists inside a compiled MD loop (rebuilt from Python)

## Caveats

- The iron-carbon model needs the potential self-consistency and, with it,
  exact (implicit-differentiation) forces. Fe16C has several self-consistent
  magnetic states, so take results on the lowest one. With every atom
  relaxed, the earlier bopjax fit (`IRON_CARBON_FIT_JAX`) collapses: carbon
  moves onto one Fe neighbour. The present fit (on bopfox-rs) gets the
  dipole anisotropy of octahedral carbon right (+4.7 eV, DFT +4.6), but the
  trace is about 30 % too large, so pressure couplings are too strong. See
  [Iron-carbon](iron-carbon.md#limitations).
- `bandwidth_guard` and `atomic_shifts` change a model's energies; a model
  fitted without them needs a refit with them, as carbon had.

- Bond integrals for mixed shells (sp, sd, pd) are checked against the
  Slater-Koster table, for hermiticity and rotation invariance, but have not
  been compared with BOPfox.
- A strong ferromagnet with a completely filled spin-up band has extra flat
  directions; the solver handles them, but the onsite levels are then not
  unique. Where moments come close to saturation in inhomogeneous systems
  (surfaces), the stationary point can be out of reach of Newton's method;
  `make_potential(condition="potential")` converges there, to a slightly
  different state (see [Formulation](formulation.md#potential-self-consistency)). Non-collinear cases close to saturation can fail to converge, and
  with spin-orbit coupling saturated ferromagnets (dimer, chain and fcc with
  7 electrons were tried) mostly do not converge at all.
- Timings are for the machine named with them (an Apple M1 Pro or a 4-core
  cloud machine) and for one model.
