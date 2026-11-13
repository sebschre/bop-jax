"""Neighbour topology (NumPy, not differentiated) and bond vectors (JAX).

Moments are sums over closed hopping paths, so a periodic cell must not let a
path wrap around and meet itself.  We therefore work on a supercell that is
large enough that no closed path of ``n_hops`` hops can wrap.  Every supercell
atom is a translation of a unit-cell atom, so Hamiltonian blocks are only ever
computed for the unit cell and gathered by index.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Topology:
    """Static connectivity.  Integer arrays only; positions enter separately."""

    n_atoms: int  # atoms in the unit cell
    n_super: int  # atoms in the supercell
    reps: tuple[int, int, int]
    species: np.ndarray  # (N,)
    nbr_origin: np.ndarray  # (N, K) unit-cell index of each neighbour
    nbr_cell: np.ndarray  # (N, K, 3) lattice translation of each neighbour
    nbr_mask: np.ndarray  # (N, K) False for padding
    nbr_super: np.ndarray  # (M, K) supercell index of each neighbour
    origin: np.ndarray  # (M,) unit-cell index of each supercell atom
    # second, longer-ranged list for repulsion, embedding and Coulomb terms
    nbr2_origin: np.ndarray | None = None  # (N, K2)
    nbr2_cell: np.ndarray | None = None  # (N, K2, 3)
    nbr2_mask: np.ndarray | None = None  # (N, K2)
    reference_positions: np.ndarray | None = None  # positions the lists were built for
    skin: float = 0.0


def _heights(cell: np.ndarray) -> np.ndarray:
    """Distance between opposite faces of the cell along each lattice vector."""
    vol = abs(np.linalg.det(cell))
    cross = np.array(
        [
            np.cross(cell[1], cell[2]),
            np.cross(cell[2], cell[0]),
            np.cross(cell[0], cell[1]),
        ]
    )
    return vol / np.linalg.norm(cross, axis=1)


def _neighbours(pos, cell, rcut):
    """Padded neighbour arrays (origin, lattice translation, mask) within rcut."""
    n = len(pos)
    if cell is None:
        shifts = np.zeros((1, 3), dtype=int)
        shift_vecs = np.zeros((1, 3))
    else:
        nc = [int(np.ceil(rcut / hd)) for hd in _heights(cell)]
        grids = np.meshgrid(*[np.arange(-c, c + 1) for c in nc], indexing="ij")
        shifts = np.stack([g.ravel() for g in grids], axis=1)
        shift_vecs = shifts @ cell

    # displacement from unit-cell atom i to atom j translated by shift c
    d = pos[None, :, None, :] + shift_vecs[None, None, :, :] - pos[:, None, None, :]
    dist = np.linalg.norm(d, axis=-1)
    is_nbr = (dist < rcut) & (dist > 1e-8)

    k_max = max(int(is_nbr.reshape(n, -1).sum(axis=1).max()), 1)
    origin = np.tile(np.arange(n)[:, None], (1, k_max))
    trans = np.zeros((n, k_max, 3), dtype=int)
    mask = np.zeros((n, k_max), dtype=bool)
    for i in range(n):
        jj, cc = np.nonzero(is_nbr[i])
        origin[i, : len(jj)] = jj
        trans[i, : len(jj)] = shifts[cc]
        mask[i, : len(jj)] = True
    return origin, trans, mask


def build_topology(positions, species, rcut, n_hops, cell=None, rcut2=None,
                   skin=0.0) -> Topology:
    """Neighbour lists plus the supercell index map.

    positions : (N, 3) Cartesian
    cell      : (3, 3) rows are lattice vectors, or None for a finite cluster
    rcut      : cutoff of the bond integrals
    n_hops    : longest closed path that will be walked (the number of moments)
    rcut2     : cutoff of the pairwise terms (repulsion, embedding, Coulomb)
    skin      : extra list radius, so the lists stay valid while no atom has
                moved by more than skin / 2 (see ``needs_rebuild``)
    """
    pos = np.asarray(positions, dtype=float)
    species = np.asarray(species, dtype=int)
    n = len(pos)
    cell = None if cell is None else np.asarray(cell, dtype=float)
    r_list = rcut + skin

    if cell is None:
        reps = (1, 1, 1)
    else:
        # a closed path of n_hops hops spans at most n_hops * r_list
        reps = tuple(int(np.floor(n_hops * r_list / hd)) + 1 for hd in _heights(cell))

    nbr_origin, nbr_cell, nbr_mask = _neighbours(pos, cell, r_list)
    k_max = nbr_mask.shape[1]
    second = (None, None, None) if rcut2 is None else _neighbours(pos, cell, rcut2 + skin)

    # supercell atom m = t * N + i, with t the raveled translation index
    reps_arr = np.array(reps)
    tg = np.meshgrid(*[np.arange(r) for r in reps], indexing="ij")
    trans = np.stack([g.ravel() for g in tg], axis=1)  # (T, 3)
    n_t = len(trans)
    t_nbr = (trans[:, None, None, :] + nbr_cell[None, :, :, :]) % reps_arr
    t_idx = np.ravel_multi_index(np.moveaxis(t_nbr, -1, 0), reps)  # (T, N, K)
    nbr_super = t_idx * n + nbr_origin[None, :, :]
    self_idx = (np.arange(n_t)[:, None] * n + np.arange(n)[None, :])[:, :, None]
    nbr_super = np.where(nbr_mask[None, :, :], nbr_super, self_idx)

    return Topology(
        n_atoms=n,
        n_super=n_t * n,
        reps=reps,
        species=species,
        nbr_origin=nbr_origin,
        nbr_cell=nbr_cell,
        nbr_mask=nbr_mask,
        nbr_super=nbr_super.reshape(n_t * n, k_max),
        origin=np.tile(np.arange(n), n_t),
        nbr2_origin=second[0],
        nbr2_cell=second[1],
        nbr2_mask=second[2],
        reference_positions=pos.copy(),
        skin=skin,
    )


def needs_rebuild(positions, topo: Topology) -> bool:
    """True once some atom has moved more than skin / 2 since the lists were built."""
    moved = np.linalg.norm(np.asarray(positions) - topo.reference_positions, axis=1)
    return bool(moved.max() > 0.5 * topo.skin)


def _vectors(positions, cell, origin, trans, mask):
    rij = positions[origin] - positions[:, None, :]
    if cell is not None:
        rij = rij + jnp.asarray(trans, dtype=positions.dtype) @ cell
    dummy = jnp.array([1.0, 0.0, 0.0], dtype=positions.dtype)
    return jnp.where(mask[:, :, None], rij, dummy)


def bond_vectors(positions, cell, topo: Topology):
    """Differentiable bond vectors, shape (N, K, 3).  Padded slots get a dummy
    unit vector so that norms and their gradients stay finite."""
    return _vectors(positions, cell, topo.nbr_origin, topo.nbr_cell, topo.nbr_mask)


def pair_vectors(positions, cell, topo: Topology):
    """Vectors of the second neighbour list, shape (N, K2, 3)."""
    return _vectors(positions, cell, topo.nbr2_origin, topo.nbr2_cell, topo.nbr2_mask)
