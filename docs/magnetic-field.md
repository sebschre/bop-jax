# Magnetic field

## Zeeman term

`params["magnetic_field"]` (Tesla, shape (3,) or (N, 3)) adds

    H_Z = -mu_B B.(L + g/2 sigma),   g = 2.00231930436

to the onsite Hamiltonian of every atom, in collinear and non-collinear
calculations. Moments are counted as everywhere in bopjax, with the
magnetisation along `m e` (the orbital magnetic moment in Bohr magnetons is
`<L>` where the spin one is `m e`), so the field favours a magnetic moment

    M_i = g/2 <sigma>_i + <L>_i

along B. Time reversal maps this onto the physical convention (electron
moment antiparallel to its angular momentum) without changing any energy.

How the term enters the functional:

- **Spin, along the local axis.** It acts like a free-atom level: the spin-up
  and spin-down levels of every shell move by `-/+ h.e` with
  `h = -mu_B g/2 B`, and `(h.e) m` is added to the energy
  (`aux["energy_zeeman"]`). The shifts therefore remain the internal Stoner
  and Coulomb potentials, and zero-field shifts are a good starting guess.
  This part is self-consistent: the moment grows in the field (paraprocess).
- **Spin, perpendicular to the local axis** (non-collinear only). The matrix
  `h_perp.sigma` enters the spinor Hamiltonian like `xi L.S`. Its first-order
  energy is the transverse spin density that spin-orbit coupling or
  non-collinear neighbours induce; it is not fed back into the Stoner term
  (local-axis ansatz, see [Spin-orbit coupling](spin-orbit.md)). A field
  across the axes exerts the torque of a rigid moment, `-mu_B g/2 m e x B`.
- **Orbital** (non-collinear only). `b.L` with `b = -mu_B B` enters the
  spinor Hamiltonian. With spin-orbit coupling it gives the first-order
  energy `-mu_B B.<L>`; without, the orbital moment is quenched at B = 0 and
  the term only adds a Van Vleck orbital moment linear in B. In collinear
  calculations the orbital term is left out and only the z component of B
  couples.

There is no vector potential, so the field does not act on the hopping
(no Landau levels, no orbital diamagnetism of itinerant electrons).

Two lower-level sources are also accepted, in eV and per atom:
`params["spin_field"]` (`h_i.sigma`) and `params["orbital_field"]`
(`b_i.L`); `make_magnetic_moment` and `make_orbital_moment` differentiate
with respect to them.

## Magnetic moments

`bj.make_magnetic_moment(energy_fn)` returns
`magnetic_moment(shifts, positions, cell, params)`, a dict of (N, 3) arrays in
Bohr magnetons: `"spin"` (`g/2 <sigma>`), `"orbital"` (`<L>`) and `"total"`,
each `-(1/mu_B) dE/dB_i` through that part of the Zeeman term. At
self-consistent shifts these are derivatives of the self-consistent energy
(Hellmann-Feynman), as for the forces.

Checks (`tests/test_zeeman.py`):

- a zero field changes nothing; collinear `"spin"` is `g/2 m` along z, and
  agrees with central differences of the self-consistent energy in
  `B_z = +-5 T` to 1e-6 muB;
- a non-collinear calculation with axes along z and a spin field along z is
  the collinear one to 1e-11 eV; the full field adds a positive Van Vleck
  orbital moment;
- without spin-orbit coupling, a common rotation of axes and field changes
  nothing, and a transverse field gives the rigid-moment torque;
- for the ferromagnetic chain with spin-orbit coupling (`xi` = 0.05 eV, axis
  tilted 45 degrees), all three components of `"total"` agree with
  differences of the self-consistent energy in `+-2 T`, the orbital part is
  `make_orbital_moment`, and the spin has a transverse component (zero
  without spin-orbit coupling).

For the d-band iron model (`IRON_FIT`, bcc, 2.43 muB), the self-consistent
energy in 100 T along the moment is 0.3 % below the first-order value
`-mu_B B.M`, and the moment grows by 1.3e-4 muB per Tesla per atom (the
high-field susceptibility of the model).

## First-order field effect on a barrier

For any configuration X, `E(X, B) = E(X, 0) - mu_B B.M(X) + O(B^2)`. Minima
and saddle points are stationary in the positions, so the barrier between
them changes to first order by

    dE_P / dB = -mu_B [M(saddle) - M(minimum)]

with nothing relaxed in the field. `bopjax.zeeman` provides:

| Function | Result |
|---|---|
| `first_order_energies(moments, field)` | `-mu_B B.(M_p - M_0)` for every image of a path |
| `first_order_barrier(energies, moments)` | saddle index, barrier, `M(saddle) - M(minimum)` and `dE_P/dB` (eV/T) |
| `peierls_stress(x, energies, b, length, slopes)` | largest slope of the profile over `b length` (cubic Hermite interpolation), i.e. the resolved shear stress the line can resist |

The Peierls stress from the zero-stress profile is an estimate of the 0 K
critical resolved shear stress; the profile is not recomputed under load.

## Minimum energy paths

`bopjax.neb.neb(force_fn, images, mobile, ...)` is a climbing-image nudged
elastic band (improved tangents) relaxed with FIRE;
`force_fn(index, positions) -> (energy, forces)` lets each image keep its
own neighbour list and warm-started onsite levels. `bopjax.neb.relax` is the
same FIRE minimiser for one configuration. The band returns arc lengths and
the slopes `dE/ds` from the forces along the tangents, for interpolating the
profile. Checked on an analytic curved valley (saddle position to 1e-3,
energy to 1e-6).

## Screw dislocation in iron

`examples/magnetoplasticity.py` applies this to a 1/2[111] screw dislocation
of the iron model: a cylinder periodic along the line with one Burgers vector
and a fixed shell at the anisotropic elastic solution, two relaxed easy cores
one `a sqrt(2/3)` apart on the (-110) glide plane, the NEB path between them,
the moments of every image, the first-order barrier shift and Peierls stress
change, and a self-consistent check of the first-order shift at +-100 T. With
`--xi` the end point and saddle are recomputed non-collinearly with
spin-orbit coupling for fields along the line, the glide direction and the
plane normal (orbital moments and direction dependence; the anisotropy is
not converged with the number of moments, see
[Spin-orbit coupling](spin-orbit.md)).

The onsite levels are self-consistent in the mobile region and a buffer
around it, and frozen at the bulk values in the rest of the fixed shell
(`make_potential(..., free=mask)`): the surface moments of the cylinder reach
2.95-3.01 muB, close to the d-band limit, which leaves the self-consistency
with nearly flat directions.

**Status: no converged Peierls path yet.** Run on a 4-core cloud machine
without GPU, 12 A cylinder (91 atoms, 25 mobile), `--condition potential`:

- The stationary self-consistency (`condition="stationary"`) does not
  converge in the cylinder: near-saturated surface moments leave it with
  nearly flat directions (Krylov and exact Newton steps stall near
  max |dE/dshift| = 1e-2; see [Formulation](formulation.md#potential-self-consistency)).
  The potential self-consistency converges there (37 Anderson steps for the
  perfect 61-atom cylinder).
- Both relaxed end points are easy cores at the expected sites (core at
  x = -1.165 and +1.165 A, energies equal to 0.008 meV), about 45 FIRE steps
  each at about 60 s per self-consistent energy and force.
- Along the NEB band the barrier falls from 156 meV (linear interpolation) to
  41 meV after 7 steps, but from step 7 on the self-consistency of the middle
  image fails: the residual stalls at 9e-3 from warm, cold and
  reduced-moment starts. It sits on the atom at the centre of the core,
  whose moment has dropped to 1.42 muB (bulk 2.43), in its spin channel
  only. Scanning that atom's exchange splitting, `I m(Delta) - Delta` stays
  negative and touches its maximum near the current state with
  `I dm/dDelta ~ 1`: the core moment is at a Stoner fold, where the
  high-moment branch followed from the easy core ends. No other branch
  was found from starts with 0 to 60 % of the moment. The configuration is
  written to `<out>_failed_scf.npz` for analysis.

What was tried at that configuration:

- An electronic temperature (`BOPConfig(kT=...)`, Fermi smearing with the
  Mermin free energy) does not help: the residual stays at 9.5e-3, 1.1e-2
  and 1.6e-2 for kT = 0.025, 0.05 and 0.1 eV. The local DOS of the core
  atom is positive everywhere, so the expansion is not the cause.
- Scanning the core atom's splitting from -2 to +2.4 eV, the environment
  alone drives its moment antiparallel (-0.37 muB at zero splitting), and
  started from a reversed moment the potential condition converges
  (m = -2.19 muB) at an energy 0.58 eV lower. Flipping any core atom of
  the relaxed easy core also "lowers" the energy by 0.36 eV, and one atom
  in perfect bulk by 0.25 eV. These are artefacts: at such states the BOP
  energy is far from stationary in the shifts (0.85 eV in bulk), and the
  potential-condition energy is wrong to first order (see
  [Formulation](formulation.md#potential-self-consistency)).

So past the fold of the core moment neither self-consistency condition gives
a trustworthy energy for this model. Options: check whether the fold
survives more moments or a Jackson kernel (both change the fitted model),
the sd model (`IRON_SD_FIT`), or a constrained core moment; or relax the
band images perpendicular to the path before the fold is reached, in case
it is an artefact of unrelaxed intermediate images.
