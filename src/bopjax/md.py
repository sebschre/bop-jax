"""Molecular dynamics interface for JAX-MD (https://github.com/jax-md/jax-md).

JAX-MD integrators take ``energy_fn(R, **kwargs) -> scalar`` and a ``shift_fn``
that moves positions.  ``jaxmd_energy_fn`` provides the first, with the onsite
levels made self-consistent inside the compiled function, and ``jaxmd_space``
the second.

Periodic boundaries are handled by bopjax itself through the lattice
translations stored in the topology, so positions must NOT be wrapped back
into the cell: use the free-space shift returned by ``jaxmd_space``.

Units: with positions in Angstrom, energies in eV and masses in atomic mass
units, JAX-MD's time unit is ``TIME_UNIT_FS`` femtoseconds.

Neighbour lists are static.  Build the topology with a ``skin`` and call
``bopjax.needs_rebuild(positions, topo)`` every few steps; when it returns
True, rebuild the topology and the energy function.
"""

from __future__ import annotations

import jax.numpy as jnp

from .scf import make_potential

TIME_UNIT_FS = 10.180505710  # Angstrom * sqrt(amu / eV) in femtoseconds
KB_EV = 8.617333262e-5  # Boltzmann constant in eV / K


def jaxmd_energy_fn(energy_fn, cell, params, self_consistent=True, tol=1e-9,
                    max_iter=20):
    """Wrap a bopjax energy function for JAX-MD.

    energy_fn : result of ``bopjax.make_energy_fn``
    cell      : (3, 3) lattice vectors or None; a ``box`` keyword passed by
                JAX-MD at call time overrides it
    params    : model parameters; a ``params`` keyword overrides them

    The returned function is differentiable: JAX-MD obtains forces from
    ``jax.grad``, which is exact because the self-consistent energy is
    stationary in the onsite levels.
    """
    potential = make_potential(energy_fn, tol=tol, max_iter=max_iter,
                               self_consistent=self_consistent)
    cell = None if cell is None else jnp.asarray(cell)

    def energy(positions, **kwargs):
        box = kwargs.get("box", cell)
        return potential(positions, box, kwargs.get("params", params))[0]

    return energy


def jaxmd_space():
    """``(displacement_fn, shift_fn)`` to use with bopjax energies: free space,
    so that atoms are never wrapped across the cell boundary."""
    from jax_md import space

    return space.free()
