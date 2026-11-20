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

**The fold is a property of the model, not of the band.** A drag
calculation (the z coordinates of the five core columns that move between
the end points fixed at `(1 - f) A + f B`, every other mobile atom relaxed
to 0.02 eV/A, each f continued from the previous relaxed positions and
levels) shows it in relaxed images and under both conditions:

| f | potential condition: E - E_A (meV), m_0 (muB) | stationary core (10 atoms, shell levels frozen): E - E_A (meV), m_0 (muB) |
|---|---|---|
| 0.1 | 9.9, 2.26 | |
| 0.2 | 28.4, 2.13 | (m_0 = 1.99 at the same positions) |
| 0.25 | | 33.3, 1.90 |
| 0.3 | relaxation drives m_0 from 1.99 to 1.47, then no solution | 44.7, 1.74 |
| 0.35 | | relaxation drives m_0 from 1.69 to 1.29 (Newton needs 16 steps), then no solution (residual 0.1 after 30 steps, m_0 -> 0.84) |

E_A is that of the same condition at A (the two differ by 19 meV). At fixed
f the relaxation lowers the energy by reducing the moment of the central
column (atom 0, the column the core passes), with forces of 0.02-0.08 eV/A,
until the self-consistent branch ends; the stationary condition follows it
somewhat further (to f = 0.35) than the potential condition (f = 0.3).
Within this d-band model the moment of that column collapses as the core
leaves the easy site, at a turning point of the self-consistency. The
stationary condition also gives systematically lower core moments than the
potential condition (by 0.1-0.15 muB at f = 0.1-0.2) and core energies lower
by about 19 meV.

**The sd model (`IRON_SD_FIT`) shows the same collapse, earlier.** Same
drag, stationary condition on the 10 core atoms, end point A relaxed with
the sd model from the d-model geometry (moment of the central column 1.98
muB at A, against 2.21 in the d model):

| f | E - E_A (meV) | m_0 (muB) |
|---|---|---|
| 0.15 | 5.3 (relaxed to 0.03 eV/A) | 1.72 |
| 0.3 | first steps 58 -> 47 -> 33 while m_0 falls 1.39 -> 1.32 -> 1.04 (Newton 8, 8, 22 steps), then no solution (residual 0.05 after 30 steps, jumping to a reversed moment) | |

So the collapse of the central column's moment is not specific to the d
basis: with the s shell it starts from a lower moment and sets in between
f = 0.15 and 0.3, where it is between 0.3 and 0.35 in the d model.

**The Egorov et al. Fe model does not collapse.** `models.iron_egorov` (the Fe
part of the Fe-Co BOP of Egorov et al., arXiv:2208.12973: DFT-projected,
then optimised bond integrals with a 4.1 A cutoff, I = 0.751 eV, Jackson
kernel with 200 moments, Yukawa repulsion; reproduces the published bcc Fe
volume, moment and elastic constants to 1-5 %). Its own end points A and B
(equal to 0.01 meV), its own column targets (central column -0.33 b, two
pairs of neighbours +0.16 b and -0.09 b), potential condition (36-52
Anderson steps), forces to 0.03 eV/A:

| f | E - E_A (meV) | m_0 (muB) | core triangle |
|---|---|---|---|
| 0 (A) | 0 | 2.84 | A |
| 0.15 | 10.7 | 2.84 | A |
| 0.3 | 31.9 | 2.85 | A |
| 0.35 | 37.6 | 2.86 | |
| 0.4 | 42.9 | 2.86 | intermediate |
| 0.5 | 47.0 | 2.86 | intermediate |
| 1 (B) | 0.0 | 2.84 | B |

The core moves from A through the intermediate triangle towards B, the
moment of the central column stays within 1.5 % of bulk (2.875 muB) all the
way, and max |dE/dshift| stays at 2.6e-2 (no sign of a non-variational
state). The barrier at the midpoint, 47 meV/b (an upper bound for the
minimum energy path), is within the DFT range of 35-58 meV/b and close to
the ~40 meV/b of Mrovec et al. (PRL 106, 246402 (2011)), whose screw core
moments also change by less than 1 %.

Compared with `IRON_FIT`, the Egorov model has a Stoner integral of 0.751
against 1.165 eV, a dd-sigma bond integral about half as strong (-0.64
against -1.15 eV at the nearest-neighbour distance, -1.00 against -1.88 eV
at 2.2 A), and a moment that hardly reacts to the core (1.3 % below bulk at
the relaxed easy core, against 9 % in the d model and 14 % in the sd model).
The collapse in `IRON_FIT` and `IRON_SD_FIT` is therefore an artefact of
those fits (large I with strong, steep bond integrals put the core near the
Stoner threshold), not a property of d-band BOPs for iron. A first run of
this drag with the column targets of the d model left the core in place
and only strained it (233 meV at f = 0.5): the drag targets must come from
the model's own end points.

For the first-order field effect, the Egorov model's moments along the path
give `dE_P/dB` directly: the moment of the central column changes by only
+0.02 muB between the easy core and the midpoint, so the magnetoplastic
effect stays of the small, bulk-like size estimated at the outset.

For `IRON_FIT`, the Peierls path passes a magnetic transition of the core,
and neither solver follows it through the turning point. Ways forward:
continuation through the turning point (arc-length continuation, or a
constrained moment of atom 0 as an extra path parameter, `E(f, m_0)`), which
also gives the size of the moment change that enters `dE_P/dB`; and checking
whether it survives more moments, and how it compares with DFT for the core
moments of a moving screw dislocation, before reading anything physical into
it.
