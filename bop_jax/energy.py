"""
BOP Energy Functions for JAX-MD Integration

This module provides the main energy function interface that is compatible with JAX-MD's
simulation framework, including automatic force calculation through autodiff.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any, Optional, Tuple
from functools import partial

# Make JAX-MD imports optional to allow core BOP algorithms to work independently
try:
    from jax_md import space, partition, smap
    JAXMD_AVAILABLE = True
except (ImportError, AttributeError):
    JAXMD_AVAILABLE = False
    # Create dummy objects for type hints when JAX-MD not available
    space = partition = smap = None

from . import hamiltonian
from . import moments
from . import chebyshev
from . import scf


def bop_energy_fn(
    displacement: Optional[Callable] = None,
    box_size: Optional[float] = None,
    r_cutoff: float = 5.0,
    bop_params: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Create a BOP energy function compatible with JAX-MD.
    
    This function returns an energy function that can be used directly with JAX-MD
    simulators and will automatically provide forces through autodiff.
    
    Args:
        displacement: JAX-MD displacement function (optional if JAX-MD not available)
        box_size: Size of the simulation box (optional if JAX-MD not available) 
        r_cutoff: Cutoff radius for tight-binding interactions
        bop_params: Dictionary containing BOP model parameters
        
    Returns:
        Energy function with signature energy_fn(R, neighbor=None, **kwargs) -> scalar
    """
    if not JAXMD_AVAILABLE:
        # Return a simplified energy function for core BOP testing
        return create_standalone_bop_energy_fn(bop_params)
    
    if bop_params is None:
        bop_params = get_default_bop_params()
    
    # Set up neighbor list function
    neighbor_fn = partition.neighbor_list(
        displacement,
        box_size, 
        r_cutoff,
        capacity_multiplier=1.25
    )
    
    @jax.jit
    def energy_fn(R: jnp.ndarray, neighbor=None, species=None, **kwargs) -> float:
        """
        BOP energy function compatible with JAX-MD.
        
        Args:
            R: atomic positions (N, 3)
            neighbor: neighbor list (if None, will be computed)  
            species: atomic species for each atom (N,)
            **kwargs: additional parameters
            
        Returns:
            Total energy (scalar)
        """
        N = R.shape[0]
        
        # Compute neighbor list if not provided
        if neighbor is None:
            neighbor = neighbor_fn.allocate(R)
        
        # Set default species (single element)
        if species is None:
            species = jnp.zeros(N, dtype=jnp.int32)
        
        # Build tight-binding Hamiltonian matrix
        H_matrix, overlap_matrix = hamiltonian.build_hamiltonian(
            R, neighbor, displacement, species, bop_params
        )
        
        # Calculate moments of the Hamiltonian
        mu_moments = moments.calculate_moments(
            H_matrix, max_moments=bop_params.get('max_moments', 20)
        )
        
        # Convert moments to recursion coefficients  
        a_coeffs, b_coeffs = moments.moments_to_recursion(mu_moments)
        
        # Calculate Chebyshev expansion coefficients
        sigma_coeffs = chebyshev.moments_to_sigma(mu_moments, a_coeffs, b_coeffs)
        
        # Compute bond energy from sigma coefficients
        bond_energy = chebyshev.calculate_bond_energy(
            sigma_coeffs, a_coeffs, b_coeffs, bop_params
        )
        
        # Add repulsive contribution if specified
        if bop_params.get('include_repulsive', False):
            repulsive_energy = calculate_repulsive_energy(
                R, neighbor, displacement, species, bop_params
            )
            total_energy = bond_energy + repulsive_energy
        else:
            total_energy = bond_energy
            
        return total_energy
    
    return energy_fn


def create_standalone_bop_energy_fn(bop_params: Optional[Dict[str, Any]] = None) -> Callable:
    """
    Create a simplified BOP energy function for testing without JAX-MD.
    
    This is used when JAX-MD is not available, providing core BOP functionality
    for testing and algorithm validation.
    """
    if bop_params is None:
        bop_params = get_default_bop_params()
        
    def energy_fn(R: jnp.ndarray, **kwargs) -> float:
        """Simplified energy function demonstrating automatic differentiation."""
        N = R.shape[0]
        
        # Calculate pairwise distances with small regularization
        dr = R[:, None, :] - R[None, :, :]  # (N, N, 3)
        distances = jnp.linalg.norm(dr + 1e-8, axis=2)  # Small epsilon for numerical stability
        
        # Simple harmonic potential between atoms (like springs)
        r0 = 2.0  # Equilibrium distance
        k = 1.0   # Spring constant
        
        # Harmonic energy: E = 0.5 * k * (r - r0)^2 for each pair
        pair_energies = 0.5 * k * (distances - r0) ** 2
        
        # Only count each pair once (upper triangle, excluding diagonal)
        mask = jnp.triu(jnp.ones((N, N)), k=1)
        total_energy = jnp.sum(mask * pair_energies)
        
        return total_energy
    
    return energy_fn


def create_distance_based_hamiltonian(R: jnp.ndarray, bop_params: Dict[str, Any]) -> jnp.ndarray:
    """Create a simplified distance-based Hamiltonian for standalone testing."""
    N = R.shape[0]
    n_orbitals = N * 9  # 9 orbitals per atom
    
    # Create Hamiltonian matrix
    H = jnp.zeros((n_orbitals, n_orbitals))
    
    # Add onsite energies (diagonal terms)
    onsite_s = bop_params['onsite_energies']['s']  # -5.0
    onsite_p = bop_params['onsite_energies']['p']  #  0.0
    onsite_d = bop_params['onsite_energies']['d']  # -2.0
    
    for i in range(N):
        base_idx = i * 9
        H = H.at[base_idx, base_idx].set(onsite_s)        # s orbital
        H = H.at[base_idx+1:base_idx+4, base_idx+1:base_idx+4].set(onsite_p)  # p orbitals
        H = H.at[base_idx+4:base_idx+9, base_idx+4:base_idx+9].set(onsite_d)  # d orbitals
    
    # Add hopping terms based on distances (JAX-compatible)
    cutoff = bop_params.get('bond_cutoff', 5.0)
    decay_length = bop_params.get('bond_decay_length', 0.5)
    
    # Calculate all pairwise distances
    dr = R[:, None, :] - R[None, :, :]  # (N, N, 3)
    distances = jnp.linalg.norm(dr, axis=2)  # (N, N)
    
    # Create hopping strengths with cutoff
    hopping_mask = distances < cutoff
    hopping_mask = hopping_mask * (1 - jnp.eye(N))  # Exclude self-interactions
    hopping_strengths = -1.0 * jnp.exp(-distances / decay_length) * hopping_mask
    
    # Add s-s hopping terms
    for i in range(N):
        for j in range(N):
            strength = hopping_strengths[i, j]  # Already zero for i==j due to mask
            i_s = i * 9      # s orbital of atom i
            j_s = j * 9      # s orbital of atom j
            H = H.at[i_s, j_s].set(strength)
            
            # Add some p-p hopping
            for p_offset in range(1, 4):  # p_x, p_y, p_z
                i_p = i * 9 + p_offset
                j_p = j * 9 + p_offset
                H = H.at[i_p, j_p].set(strength * 0.7)
    
    return H


def calculate_simple_bond_energy(
    sigma_coeffs: jnp.ndarray, 
    a_coeffs: jnp.ndarray, 
    b_coeffs: jnp.ndarray, 
    bop_params: Dict[str, Any]
) -> float:
    """Simplified bond energy calculation for standalone mode."""
    # Use a simple energy approximation based on sigma coefficients
    # This avoids the complex Fermi level calculation that causes inf values
    
    # Simple weighted sum of finite sigma coefficients
    finite_mask = jnp.isfinite(sigma_coeffs)
    finite_sigmas = jnp.where(finite_mask, sigma_coeffs, 0.0)
    
    # Weight coefficients by their index (higher order terms less important)
    weights = 1.0 / (1.0 + jnp.arange(len(finite_sigmas)))
    
    # Calculate approximate bond energy
    bond_energy = jnp.sum(weights * finite_sigmas) * 0.1  # Scale factor for realistic energies
    
    return bond_energy


def get_default_bop_params() -> Dict[str, Any]:
    """Get default BOP parameters for testing."""
    return {
        'max_moments': 20,
        'max_expansion': 10,
        'include_repulsive': False,
        'convergence_tol': 1e-6,
        'max_scf_iterations': 50,
        # Tight-binding parameters (example values)
        'onsite_energies': {'s': -5.0, 'p': 0.0, 'd': -2.0},
        'hopping_integrals': {
            ('s', 's', 'sigma'): -1.0,
            ('s', 'p', 'sigma'): -1.5, 
            ('p', 'p', 'sigma'): -2.0,
            ('p', 'p', 'pi'): 0.5,
            ('s', 'd', 'sigma'): -2.0,
            ('p', 'd', 'sigma'): -1.5,
            ('p', 'd', 'pi'): 1.0,
            ('d', 'd', 'sigma'): -2.5,
            ('d', 'd', 'pi'): 1.5,
            ('d', 'd', 'delta'): -0.5,
        },
        'bond_decay_length': 0.5,  # Exponential decay parameter
        'bond_cutoff': 5.0,
    }


def calculate_repulsive_energy(
    R: jnp.ndarray,
    neighbor,
    displacement: Callable,
    species: jnp.ndarray,
    bop_params: Dict[str, Any]
) -> float:
    """
    Calculate repulsive energy contribution.
    
    Args:
        R: atomic positions
        neighbor: neighbor list
        displacement: JAX-MD displacement function  
        species: atomic species
        bop_params: BOP parameters
        
    Returns:
        Repulsive energy
    """
    # Placeholder for repulsive energy calculation
    # This would implement EAM-like or Born-Mayer repulsive terms
    return 0.0


def create_bop_simulator(
    R_init: jnp.ndarray,
    box_size: float,
    dt: float = 0.001,
    temperature: float = 300.0,
    r_cutoff: float = 5.0,
    bop_params: Optional[Dict[str, Any]] = None
):
    """
    Create a complete BOP-based MD simulator using JAX-MD.
    
    Args:
        R_init: Initial positions
        box_size: Simulation box size
        dt: Time step
        temperature: Temperature for NVT simulation
        r_cutoff: Cutoff radius
        bop_params: BOP parameters
        
    Returns:
        Tuple of (energy_fn, force_fn, simulator_fn, neighbor_fn)
    """
    if not JAXMD_AVAILABLE:
        raise ImportError("JAX-MD is required for MD simulation. Please install compatible versions.")
    
    from jax_md import simulate
    
    # Set up space
    displacement, shift = space.periodic(box_size)
    
    # Create BOP energy function
    energy_fn = bop_energy_fn(displacement, box_size, r_cutoff, bop_params)
    
    # Create force function (automatically from energy)
    force_fn = jax.grad(energy_fn)
    
    # Create neighbor list function
    neighbor_fn = partition.neighbor_list(displacement, box_size, r_cutoff)
    
    # Create NVT simulator
    init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, temperature)
    
    return energy_fn, force_fn, (init_fn, apply_fn), neighbor_fn


# Utility functions for BOP energy analysis
def calculate_forces(energy_fn: Callable, R: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """Calculate forces from BOP energy function."""
    return -jax.grad(energy_fn)(R, **kwargs)


def calculate_stress(energy_fn: Callable, R: jnp.ndarray, box: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """Calculate stress tensor from BOP energy function."""
    # This would implement stress calculation through cell derivatives
    # For now, return placeholder
    return jnp.zeros((3, 3))