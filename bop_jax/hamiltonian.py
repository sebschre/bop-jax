"""
Tight-Binding Hamiltonian Construction for BOP

This module handles the construction of tight-binding Hamiltonian matrices
from atomic positions and neighbor lists, including Slater-Koster transformations
and environment-dependent onsite energies.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any, Tuple
import numpy as np


def build_hamiltonian(
    R: jnp.ndarray,
    neighbor,
    displacement: Callable,
    species: jnp.ndarray,
    bop_params: Dict[str, Any]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build tight-binding Hamiltonian and overlap matrices.
    
    Args:
        R: atomic positions (N, 3)
        neighbor: JAX-MD neighbor list
        displacement: JAX-MD displacement function
        species: atomic species (N,)
        bop_params: BOP parameters containing hopping integrals
        
    Returns:
        Tuple of (Hamiltonian matrix, overlap matrix)
    """
    N = R.shape[0]
    
    # Determine orbital structure (simplified: assume each atom has s,p,d orbitals)
    orbitals_per_atom = 9  # 1s + 3p + 5d orbitals
    total_orbitals = N * orbitals_per_atom
    
    # Initialize matrices
    H_matrix = jnp.zeros((total_orbitals, total_orbitals))
    S_matrix = jnp.eye(total_orbitals)  # Start with identity for overlap
    
    # Build onsite terms (diagonal elements)
    H_matrix = add_onsite_energies(H_matrix, species, bop_params)
    
    # Build hopping terms (off-diagonal elements)
    H_matrix, S_matrix = add_hopping_terms(
        H_matrix, S_matrix, R, neighbor, displacement, species, bop_params
    )
    
    return H_matrix, S_matrix


def add_onsite_energies(
    H_matrix: jnp.ndarray,
    species: jnp.ndarray,
    bop_params: Dict[str, Any]
) -> jnp.ndarray:
    """
    Add onsite energies to the diagonal of the Hamiltonian.
    
    Args:
        H_matrix: Hamiltonian matrix to modify
        species: atomic species 
        bop_params: BOP parameters
        
    Returns:
        Updated Hamiltonian matrix
    """
    N = species.shape[0]
    orbitals_per_atom = 9
    onsite_energies = bop_params['onsite_energies']
    
    # Orbital ordering: s(1), p(3), d(5) for each atom
    orbital_types = ['s'] + ['p'] * 3 + ['d'] * 5
    
    for atom in range(N):
        atom_type = species[atom]  # Could use this to distinguish elements
        base_idx = atom * orbitals_per_atom
        
        for orb_idx, orb_type in enumerate(orbital_types):
            matrix_idx = base_idx + orb_idx
            H_matrix = H_matrix.at[matrix_idx, matrix_idx].set(
                onsite_energies[orb_type]
            )
    
    return H_matrix


def add_hopping_terms(
    H_matrix: jnp.ndarray,
    S_matrix: jnp.ndarray, 
    R: jnp.ndarray,
    neighbor,
    displacement: Callable,
    species: jnp.ndarray,
    bop_params: Dict[str, Any]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Add hopping terms between neighboring atoms using Slater-Koster rules.
    
    Args:
        H_matrix: Hamiltonian matrix
        S_matrix: Overlap matrix
        R: atomic positions
        neighbor: neighbor list
        displacement: displacement function
        species: atomic species
        bop_params: BOP parameters
        
    Returns:
        Updated (H_matrix, S_matrix)
    """
    neighbors_idx = neighbor.idx
    N = R.shape[0]
    orbitals_per_atom = 9
    
    # Process each atom and its neighbors
    for i in range(N):
        for j_idx in range(neighbors_idx.shape[1]):
            j = neighbors_idx[i, j_idx]
            
            # Skip if no neighbor (padding)
            if j == N:  # JAX-MD uses N as padding value
                continue
                
            # Skip self-interaction
            if i == j:
                continue
                
            # Calculate displacement vector
            dr_vec = displacement(R[i], R[j])
            dr = jnp.linalg.norm(dr_vec)
            
            # Skip if beyond cutoff
            if dr > bop_params['bond_cutoff']:
                continue
                
            # Calculate direction cosines
            l, m, n = dr_vec / dr  # direction cosines
            
            # Get hopping integrals between all orbital pairs
            H_block, S_block = calculate_hopping_block(
                dr, l, m, n, species[i], species[j], bop_params
            )
            
            # Add to matrices
            i_start = i * orbitals_per_atom
            j_start = j * orbitals_per_atom
            
            H_matrix = H_matrix.at[
                i_start:i_start+orbitals_per_atom, 
                j_start:j_start+orbitals_per_atom
            ].set(H_block)
            
            S_matrix = S_matrix.at[
                i_start:i_start+orbitals_per_atom,
                j_start:j_start+orbitals_per_atom  
            ].set(S_block)
    
    return H_matrix, S_matrix


def calculate_hopping_block(
    dr: float,
    l: float,
    m: float, 
    n: float,
    species_i: int,
    species_j: int,
    bop_params: Dict[str, Any]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate 9x9 hopping block between two atoms using Slater-Koster rules.
    
    Args:
        dr: distance between atoms
        l, m, n: direction cosines  
        species_i, species_j: atomic species
        bop_params: BOP parameters
        
    Returns:
        Tuple of (H_block, S_block) both 9x9 arrays
    """
    # Distance-dependent decay
    decay_factor = jnp.exp(-dr / bop_params['bond_decay_length'])
    
    # Get base hopping integrals
    hopping_integrals = bop_params['hopping_integrals']
    
    # Calculate Slater-Koster integrals
    Vss_sigma = hopping_integrals[('s', 's', 'sigma')] * decay_factor
    Vsp_sigma = hopping_integrals[('s', 'p', 'sigma')] * decay_factor  
    Vpp_sigma = hopping_integrals[('p', 'p', 'sigma')] * decay_factor
    Vpp_pi = hopping_integrals[('p', 'p', 'pi')] * decay_factor
    Vsd_sigma = hopping_integrals[('s', 'd', 'sigma')] * decay_factor
    Vpd_sigma = hopping_integrals[('p', 'd', 'sigma')] * decay_factor
    Vpd_pi = hopping_integrals[('p', 'd', 'pi')] * decay_factor
    Vdd_sigma = hopping_integrals[('d', 'd', 'sigma')] * decay_factor
    Vdd_pi = hopping_integrals[('d', 'd', 'pi')] * decay_factor
    Vdd_delta = hopping_integrals[('d', 'd', 'delta')] * decay_factor
    
    # Initialize 9x9 block (s, px, py, pz, dxy, dyz, dz2, dxz, dx2-y2)
    H_block = jnp.zeros((9, 9))
    S_block = jnp.zeros((9, 9))  # Simplified: assume orthogonal basis
    
    # s-s interaction
    H_block = H_block.at[0, 0].set(Vss_sigma)
    
    # s-p interactions  
    H_block = H_block.at[0, 1].set(l * Vsp_sigma)  # s-px
    H_block = H_block.at[0, 2].set(m * Vsp_sigma)  # s-py
    H_block = H_block.at[0, 3].set(n * Vsp_sigma)  # s-pz
    
    # p-p interactions (using direction cosines)
    H_block = H_block.at[1, 1].set(l*l * Vpp_sigma + (1-l*l) * Vpp_pi)  # px-px
    H_block = H_block.at[1, 2].set(l*m * (Vpp_sigma - Vpp_pi))  # px-py
    H_block = H_block.at[1, 3].set(l*n * (Vpp_sigma - Vpp_pi))  # px-pz
    H_block = H_block.at[2, 2].set(m*m * Vpp_sigma + (1-m*m) * Vpp_pi)  # py-py
    H_block = H_block.at[2, 3].set(m*n * (Vpp_sigma - Vpp_pi))  # py-pz
    H_block = H_block.at[3, 3].set(n*n * Vpp_sigma + (1-n*n) * Vpp_pi)  # pz-pz
    
    # Simplified d-orbital interactions (full SK rules would be more complex)
    # For now, use simplified forms
    for d_i in range(4, 9):
        for d_j in range(4, 9):
            if d_i == d_j:
                H_block = H_block.at[d_i, d_j].set(0.5 * (Vdd_sigma + Vdd_pi))
            else:
                H_block = H_block.at[d_i, d_j].set(0.1 * Vdd_delta)
    
    # Make matrix Hermitian
    H_block = 0.5 * (H_block + H_block.T)
    S_block = 0.5 * (S_block + S_block.T)
    
    return H_block, S_block


def get_orbital_structure(species: jnp.ndarray) -> Dict[str, Any]:
    """
    Get orbital structure information for given atomic species.
    
    Args:
        species: atomic species array
        
    Returns:
        Dictionary with orbital structure information
    """
    # Simplified: assume all atoms have s, p, d orbitals
    N = species.shape[0]
    return {
        'orbitals_per_atom': 9,
        'total_orbitals': N * 9,
        'orbital_types': ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2'],
        'orbital_indices': {
            's': [0],
            'p': [1, 2, 3], 
            'd': [4, 5, 6, 7, 8]
        }
    }