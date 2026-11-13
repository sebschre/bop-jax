"""Neighbour topology (NumPy, not differentiated) and bond vectors (JAX).

Moments are sums over closed hopping paths that start and end on one atom, so
each atom only needs its own cluster: the atoms it can reach in half the path
length.  Clusters are built from the neighbour graph with explicit lattice
translations, which treats periodic cells and finite clusters alike and makes
the cost linear in the number of atoms.  Every cluster member is a translation
of a unit-cell atom, so Hamiltonian blocks are only ever computed for the unit
cell and gathered by index.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Topology:
    """Static connectivity.  Integer arrays only; positions enter separately."""

    n_atoms: int  # atoms in the unit cell
    species: np.ndarray  # (N,)
    nbr_origin: np.ndarray  # (N, K) unit-cell index of each neighbour
    nbr_cell: np.ndarray  # (N, K, 3) lattice translation of each neighbour
    nbr_mask: np.ndarray  # (N, K) False for padding
    # Local cluster of each atom, members ordered by hop distance.  Slots
    # [0, cluster_sizes[h]) hold every atom within h hops (plus padding).
    cluster_origin: np.ndarray  # (N, C) unit-cell index of each member
    cluster_nbr: np.ndarray  # (N, C, K) slot of each member's neighbours; C = outside
    cluster_sizes: tuple[int, ...]  # length n_shells + 1
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


def _clusters(nbr_origin, nbr_cell, nbr_mask, n_shells):
    """Breadth-first clusters around every atom, padded to common shell sizes."""
    n, k_max = nbr_mask.shape
    valid = [np.nonzero(nbr_mask[i])[0] for i in range(n)]
    shells_all = []
    for i in range(n):
        seen = {(i, 0, 0, 0)}
        shells = [[(i, 0, 0, 0)]]
        for _ in range(n_shells):
            new = []
            for j, tx, ty, tz in shells[-1]:
                for k in valid[j]:
                    c = nbr_cell[j, k]
                    key = (int(nbr_origin[j, k]), tx + int(c[0]), ty + int(c[1]), tz + int(c[2]))
                    if key not in seen:
                        seen.add(key)
                        new.append(key)
            shells.append(new)
        shells_all.append(shells)

    widths = [max(len(sh[h]) for sh in shells_all) for h in range(n_shells + 1)]
    offsets = np.concatenate([[0], np.cumsum(widths)])
    size = int(offsets[-1])
    origin = np.zeros((n, size), dtype=np.int32)
    nbr = np.full((n, size, k_max), size, dtype=np.int32)
    for i, shells in enumerate(shells_all):
        slot = {}
        for h, members in enumerate(shells):
            for m, key in enumerate(members):
                slot[key] = int(offsets[h]) + m
        for (j, tx, ty, tz), c in slot.items():
            origin[i, c] = j
            for k in valid[j]:
                t = nbr_cell[j, k]
                key = (int(nbr_origin[j, k]), tx + int(t[0]), ty + int(t[1]), tz + int(t[2]))
                nbr[i, c, k] = slot.get(key, size)
    return origin, nbr, tuple(int(o) for o in offsets[1:])


def build_topology(positions, species, rcut, n_hops, cell=None, rcut2=None,
                   skin=0.0) -> Topology:
    """Neighbour lists plus the local cluster of every atom.

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
    cell = None if cell is None else np.asarray(cell, dtype=float)

    nbr_origin, nbr_cell, nbr_mask = _neighbours(pos, cell, rcut + skin)
    second = (None, None, None) if rcut2 is None else _neighbours(pos, cell, rcut2 + skin)
    # a closed path of n_hops hops never gets further than half of them away
    cl_origin, cl_nbr, cl_sizes = _clusters(nbr_origin, nbr_cell, nbr_mask, (n_hops + 1) // 2)

    return Topology(
        n_atoms=len(pos),
        species=species,
        nbr_origin=nbr_origin,
        nbr_cell=nbr_cell,
        nbr_mask=nbr_mask,
        cluster_origin=cl_origin,
        cluster_nbr=cl_nbr,
        cluster_sizes=cl_sizes,
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
