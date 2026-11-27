# bopjax

A fresh JAX implementation of the analytic bond-order-potential (BOP)
methodology implemented by BOPfox. The energy is one differentiable function;
forces, stress, magnetic torques and the self-consistency Jacobian come from
automatic differentiation instead of hand-derived chains. Spin axes are
dynamical variables like positions: spins precess, relax and move the atoms,
with or without spin-orbit coupling.

## Spins and atoms in motion

Rendered with [OVITO](https://www.ovito.org) through `bopjax.ovito`
([visualisation](docs/visualisation.md)).

**Spin-orbit coupling** (`examples/ovito_spin_orbit.py`). Two
ferromagnetic chains of the [anisotropy model](docs/spin-orbit.md) (canonical d band,
7 electrons, `xi` = 0.1 eV in the upper chain, none in the lower), one atom
per periodic cell, start with the moments 60 degrees from the chain. With
spin-orbit coupling the chain is the easy axis: the spin moments precess
about it and, with Gilbert damping `alpha = 0.1`, spiral into it, releasing
0.85 meV per atom in 6 ps (`K sin^2 60` with `K` = 1.13 meV). The orbital
moment (blue, drawn three times longer) follows the spin but is larger along
the chain (0.37 hbar) than across it (0.21 hbar). Without spin-orbit coupling
there is no torque and no orbital moment, and nothing moves.

![Spin and orbital moments of a chain precessing into its easy axis](docs/animations/spin_orbit.gif)

**Spin-lattice coupling** (`examples/ovito_spin_lattice.py`). The
constant-energy run of `examples/spin_lattice.py`: four (001) planes of a bcc
ferromagnet, canted in pairs by +20 and -20 degrees, on a perfect lattice at
rest. The two groups of spins precess about their sum (16 fs per turn), and
exchange striction pairs the planes: they oscillate by 0.011 A peak to peak
with a period of 105 fs, drawn 40 times larger. There is no spin-orbit coupling in this
model.

![Precessing spins pulling the planes of a bcc lattice together](docs/animations/spin_lattice.gif)

## Install

As a dependency of another project, from a tagged release:

```bash
pip install "bopjax @ git+https://github.com/sebschre/bop-jax@v0.5.0"
# with the JAX-MD interface
pip install "bopjax[md] @ git+https://github.com/sebschre/bop-jax@v0.5.0"
```

or in `pyproject.toml`:

```toml
dependencies = ["bopjax @ git+https://github.com/sebschre/bop-jax@v0.5.0"]
```

Each [release](https://github.com/sebschre/bop-jax/releases) also carries a
wheel that `pip install <url>` accepts. Python 3.11 to 3.13.

To work on the code:

```bash
uv sync                 # --extra md: JAX-MD interface, --extra viz: OVITO rendering
uv run pytest           # about 15 minutes on 4 cores: every configuration is compiled once
uv run python examples/fcc_distorted.py
```

## Quick start

The distorted fcc cell of the validation, made self-consistent
(`examples/fcc_distorted.py`):

```python
import jax.numpy as jnp
import bopjax as bj
from bopjax.models import canonical_d

a = 3.5
cell = a * jnp.eye(3)
positions = a * jnp.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
positions = positions.at[0].add(jnp.array([0.15, -0.08, 0.05]))
positions = positions.at[2].add(jnp.array([-0.05, 0.10, 0.02]))

cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5)            # canonical d band, 6 moments
topo = bj.build_topology(positions, [0] * 4, cfg.rcut, cfg.nmom, cell)
energy = bj.make_energy_fn(topo, cfg)                      # E(shifts, positions, cell, params)
params = canonical_d(hubbard=8.0)

result = bj.make_scf(energy)(positions, cell, params, tol=1e-12)
forces = bj.make_forces(energy)(result.shifts, positions, cell, params)
print(result.energy / 4, result.aux["charge"], forces)
```

This prints -6.0112485448437 eV per atom, the BOPfox value to all 13 digits.
`bj.make_potential` puts the self-consistency inside one compiled function
`E(positions, cell, params)`, whose `jax.grad` is the force; the
[formulation](docs/formulation.md) explains why no response term is needed.

## Features

| Feature | Options |
|---|---|
| Basis | s, p, d shells in any combination, several species, each with its own shells and expansion settings (`BOPConfig(species=...)`) |
| Bond integrals | power law, BOPfox `v0`; any `radial(p, r)` callable |
| Bandwidth | `azero`, `lowest`, `highest`, `avgan_maxbn`, `averaged`, `global`, `findeminemax`, `gershgorin`, `beerpettiforaoki`; optional guard against a diverging expansion (`bandwidth_guard`) |
| Self-consistency | one onsite shift per shell, or per atom (`atomic_shifts`) |
| Terminator | `none`/`constantabn`, `oscillation`, `weighted`; Jackson kernel |
| Magnetism | `collinear`, `noncollinear` (prescribed local spin axes, torques from the gradient) |
| [Spin-orbit coupling](docs/spin-orbit.md) | onsite `xi L.S` for p and d shells in non-collinear calculations (`BOPConfig(spin_orbit=True)`); no reference code to compare with |
| [Magnetic field](docs/magnetic-field.md) | Zeeman term `-mu_B B.(L + g/2 sigma)` (`params["magnetic_field"]`), magnetic moments as field derivatives, first-order field shift of barriers and Peierls stress (`bopjax.zeeman`) |
| Coulomb | onsite `J_ii`; intersite `frauenheim` with taper |
| [Screening](docs/screening.md) | environment-dependent d-d bond integrals (`BOPConfig(screening=True)`), not yet compared with BOPfox |
| Repulsion | pair term plus embedding (`pairwise.V0`, `pairwise.POWERLAW`, `pairwise.EGOROV`, or your own); environment-dependent Yukawa repulsion (`BOPConfig(yukawa=True)`) |
| [Self-consistency](docs/formulation.md) | stationary point by Newton's method (BOPfox), or potentials consistent with charges and moments by Anderson mixing with implicit-function gradients (`make_potential(condition="potential")`); a subset of atoms (`free=`); electronic temperature with the Mermin free energy (`BOPConfig(kT=...)`); `remat=True` for memory |
| [Stress](docs/formulation.md) | `make_stress`, strain derivative of the self-consistent energy |
| [Molecular dynamics](docs/molecular-dynamics.md) | `bopjax.md` for [JAX-MD](https://github.com/jax-md/jax-md) |
| [Spin dynamics](docs/spin-dynamics.md) | `bopjax.spin`: Landau-Lifshitz-Gilbert precession, damping and thermal torques for the non-collinear spin axes |
| [Spin-lattice dynamics](docs/spin-lattice.md) | `bopjax.spinlattice`: atoms and spin axes moving together under the forces and torques of one energy |
| [Minimum energy paths](docs/magnetic-field.md#minimum-energy-paths) | `bopjax.neb`: climbing-image nudged elastic band and FIRE relaxation |
| [Visualisation](docs/visualisation.md) | `bopjax.ovito`: trajectories with spin, orbital and other per-atom vectors as extended XYZ for [OVITO](https://www.ovito.org), and rendered pictures and animations through its Python module |

## Rust port

[`bopfox-rs/`](bopfox-rs/README.md) ports the energy to Rust with analytic
derivatives (the BOPfox route) instead of automatic differentiation. Phase 1
covers energy, forces, virial and dE/dshifts at given shifts for s/p/d bases,
collinear magnetism and the pair repulsions. Phase 2 adds self-consistency
from gradients only, by preconditioned Anderson mixing. Since then it also
has bopjax's potential condition (with implicit-function forces),
non-collinear magnetism with spin-orbit coupling, external magnetic fields,
spin and spin-lattice dynamics, d-d screening, intersite Coulomb, Yukawa
repulsion, electronic temperature, all bandwidth estimators, NEB, and an ASE
calculator with warm-started self-consistency: everything in the feature
table except the JAX-specific interfaces.

It matches bopjax to 1e-9. It is 13-20 times faster for energy and forces at
given shifts, and 30-50 times faster self-consistently (256-atom d band:
1.3 s against 67 s).

## Documentation

| Page | Content |
|---|---|
| [Formulation](docs/formulation.md) | the energy functional, self-consistency, code layout |
| [Performance](docs/performance.md) | scaling, timings, GPUs and single precision |
| [Molecular dynamics](docs/molecular-dynamics.md) | JAX-MD interface, warm-started NVE, fixed atoms, screw dislocation |
| [Spin dynamics](docs/spin-dynamics.md) | Landau-Lifshitz-Gilbert dynamics of the spin axes |
| [Spin-orbit coupling](docs/spin-orbit.md) | onsite `xi L.S`, magnetic anisotropy, orbital moment, its limits |
| [Spin-lattice dynamics](docs/spin-lattice.md) | atoms and spin axes moving together |
| [Magnetic field](docs/magnetic-field.md) | Zeeman term, magnetic moments, first-order field effect on the Peierls barrier of a screw dislocation in iron (drag and free NEB, with and without spin-orbit coupling) |
| [Visualisation](docs/visualisation.md) | trajectories and animations with OVITO |
| [Validation](docs/validation.md) | agreement with BOPfox, and where the two codes differ |
| [Screening](docs/screening.md) | environment-dependent d-d bond integrals |
| [Iron](docs/iron.md) | magnetic d-band iron models: the fit of this repository, and the Fe model of Egorov et al. used for dislocations |
| [Carbon](docs/carbon.md) | sp-valent carbon model: fit, diamond, graphene and dense phases, elastic constants |
| [Iron-carbon](docs/iron-carbon.md) | Fe-C on the Egorov iron and sp carbon models: cross terms fitted on bopfox-rs with every atom relaxed, C in bcc Fe, cementite; limitations (several magnetic states, saddle off the tetrahedral site) |
| [Carbon at a screw dislocation](docs/carbon-dislocation.md) | Fe-C BOP on bopfox-rs: carbon binding energies around a 1/2<111> screw, no hard-core reconstruction, Peierls drag with and without carbon |
| [Limitations](docs/limitations.md) | what is not implemented, caveats |

## References

For the analytic BOP methodology and its implementation in BOPfox, see:

- R. Drautz, T. Hammerschmidt, M. Cak, and D. G. Pettifor, “Bond-order
  potentials: Derivation and parameterization for refractory elements,”
  *Modelling and Simulation in Materials Science and Engineering* 23 (2015),
  074004. [doi:10.1088/0965-0393/23/7/074004](https://doi.org/10.1088/0965-0393/23/7/074004)
- T. Hammerschmidt et al., “BOPfox program for tight-binding and analytic
  bond-order potential calculations,” [arXiv:1803.07491](https://arxiv.org/abs/1803.07491),
  [doi:10.1016/j.cpc.2018.08.013](https://doi.org/10.1016/j.cpc.2018.08.013).

See the [BOPfox website](https://bopfox.de/) for more information about the
reference program and its contributors.
