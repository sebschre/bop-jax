"""
Moment Calculation and Continued Fraction Conversion for BOP

This module implements the core BOP algorithm components:
1. Calculation of moments μₙ = Tr[H^n] from the Hamiltonian matrix
2. Conversion of moments to recursion coefficients via continued fraction
3. JAX autodiff integration for automatic derivative calculation

Based on the original BOPfox implementation in bopfct.p.f and util.p.f
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial
import numpy as np


@partial(jax.jit, static_argnums=(1, 2))
def calculate_moments(
    H_matrix: jnp.ndarray,
    max_moments: int = 20,
    use_trace_method: bool = True
) -> jnp.ndarray:
    """
    Calculate moments μₙ = Tr[H^n] of the Hamiltonian matrix.
    
    This is the core calculation in BOP that characterizes the local density
    of states. The moments are calculated either by matrix powers and trace
    or by eigenvalue decomposition.
    
    Args:
        H_matrix: Hamiltonian matrix (M, M)
        max_moments: maximum moment order to calculate
        use_trace_method: if True, use Tr[H^n], else use eigenvalue method
        
    Returns:
        moments array of shape (max_moments + 1,) containing μ₀, μ₁, ..., μₙ
    """
    if use_trace_method:
        return _calculate_moments_trace(H_matrix, max_moments)
    else:
        return _calculate_moments_eigenvalues(H_matrix, max_moments)


@partial(jax.jit, static_argnums=(1,))
def _calculate_moments_trace(H_matrix: jnp.ndarray, max_moments: int) -> jnp.ndarray:
    """Calculate moments using matrix powers and trace."""
    moments = jnp.zeros(max_moments + 1)
    
    # μ₀ = Tr[I] = number of orbitals
    moments = moments.at[0].set(jnp.trace(jnp.eye(H_matrix.shape[0])))
    
    # μ₁ = Tr[H]  
    H_power = H_matrix.copy()
    moments = moments.at[1].set(jnp.trace(H_power))
    
    # μₙ = Tr[H^n] for n ≥ 2
    for n in range(2, max_moments + 1):
        H_power = jnp.dot(H_power, H_matrix)
        moments = moments.at[n].set(jnp.trace(H_power))
    
    return moments


@partial(jax.jit, static_argnums=(1,))
def _calculate_moments_eigenvalues(H_matrix: jnp.ndarray, max_moments: int) -> jnp.ndarray:
    """
    Calculate moments using eigenvalue decomposition.
    
    More numerically stable for large matrices but requires eigenvalue computation.
    μₙ = Σᵢ λᵢⁿ where λᵢ are eigenvalues of H.
    """
    eigenvalues = jnp.linalg.eigvals(H_matrix)
    moments = jnp.zeros(max_moments + 1)
    
    for n in range(max_moments + 1):
        moments = moments.at[n].set(jnp.sum(eigenvalues**n))
    
    return moments


@jax.jit
def moments_to_recursion(mu_moments: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert moments to recursion coefficients using continued fraction expansion.
    
    This implements the core algorithm from BOPfox util.p.f:momtorec subroutine.
    The recursion coefficients aₙ and bₙ define the continued fraction:
    
    G(z) = 1/(z - a₀ - b₁²/(z - a₁ - b₂²/(z - a₂ - ...)))
    
    Args:
        mu_moments: moments array μ₀, μ₁, μ₂, ...
        
    Returns:
        Tuple of (a_coeffs, b_coeffs) recursion coefficient arrays
    """
    n_moments = len(mu_moments)
    max_recursion = n_moments - 1
    
    # Initialize coefficient arrays
    a_coeffs = jnp.zeros(max_recursion)
    b_coeffs = jnp.zeros(max_recursion)
    
    # Initialize working arrays for the algorithm
    # Based on BOPfox implementation
    alpha = jnp.zeros((max_recursion, max_recursion))
    beta = jnp.zeros((max_recursion, max_recursion))
    
    # Set initial conditions
    # alpha[0, k] corresponds to the initial moment sequence
    for k in range(max_recursion):
        alpha = alpha.at[0, k].set(mu_moments[k])
    
    # Main recursion algorithm
    for i in range(max_recursion - 1):
        if i == 0:
            # First iteration: a₀ = μ₁/μ₀
            a_coeffs = a_coeffs.at[0].set(alpha[0, 1] / alpha[0, 0])
            
            # Calculate β values for first iteration
            for k in range(max_recursion - 1):
                beta = beta.at[0, k].set(
                    alpha[0, k+2] - a_coeffs[0] * alpha[0, k+1]
                )
        else:
            # Subsequent iterations
            # Calculate new α values
            for k in range(max_recursion - i - 1):
                alpha = alpha.at[i, k].set(
                    beta[i-1, k+1] / beta[i-1, k] - 
                    alpha[i-1, k+1] / alpha[i-1, k]
                )
            
            # Calculate aᵢ
            a_coeffs = a_coeffs.at[i].set(alpha[i, 1] / alpha[i, 0])
            
            # Calculate βᵢ values
            for k in range(max_recursion - i - 2):
                beta = beta.at[i, k].set(
                    alpha[i, k+2] - a_coeffs[i] * alpha[i, k+1]
                )
        
        # Calculate bᵢ₊₁ from βᵢ
        if i < max_recursion - 2:
            b_coeffs = b_coeffs.at[i+1].set(
                jnp.sqrt(jnp.abs(beta[i, 1] / beta[i, 0]))
            )
    
    # Handle the final coefficient
    if max_recursion >= 2:
        final_idx = max_recursion - 2
        a_coeffs = a_coeffs.at[final_idx].set(alpha[final_idx, 1] / alpha[final_idx, 0])
    
    return a_coeffs, b_coeffs


@jax.jit
def calculate_asymptotic_coefficients(
    a_coeffs: jnp.ndarray,
    b_coeffs: jnp.ndarray
) -> Tuple[float, float]:
    """
    Calculate asymptotic values a∞ and b∞ from recursion coefficients.
    
    These are important for the bond energy calculation and represent
    the limiting behavior of the continued fraction.
    
    Args:
        a_coeffs: recursion coefficients aₙ
        b_coeffs: recursion coefficients bₙ
        
    Returns:
        Tuple of (a_inf, b_inf) asymptotic coefficients
    """
    n = len(a_coeffs)
    
    # Simple extrapolation: use the last few values
    # More sophisticated methods could be implemented
    if n >= 3:
        a_inf = a_coeffs[-1]  # Could use extrapolation
        b_inf = b_coeffs[-1]  # Could use extrapolation
    else:
        a_inf = a_coeffs[-1] if len(a_coeffs) > 0 else 0.0
        b_inf = b_coeffs[-1] if len(b_coeffs) > 0 else 1.0
        
    return a_inf, b_inf


@jax.jit
def normalize_recursion_coefficients(
    a_coeffs: jnp.ndarray,
    b_coeffs: jnp.ndarray,
    a_inf: float,
    b_inf: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Normalize recursion coefficients as done in BOPfox.
    
    The normalized coefficients are used in the Chebyshev expansion.
    This corresponds to the anhx_orbspin and bnhx_orbspin arrays in BOPfox.
    
    Args:
        a_coeffs: original recursion coefficients aₙ
        b_coeffs: original recursion coefficients bₙ  
        a_inf: asymptotic value a∞
        b_inf: asymptotic value b∞
        
    Returns:
        Tuple of (normalized_a, normalized_b) coefficient arrays
    """
    # Normalization: subtract asymptotic values and scale
    normalized_a = (a_coeffs - a_inf) / (2.0 * b_inf)
    normalized_b = b_coeffs / (2.0 * b_inf)
    
    return normalized_a, normalized_b


def calculate_moment_derivatives(
    H_matrix: jnp.ndarray,
    max_moments: int,
    perturbation_indices: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Calculate derivatives of moments with respect to Hamiltonian elements.
    
    This leverages JAX's automatic differentiation to compute derivatives
    that were calculated analytically in the original BOPfox implementation.
    
    Args:
        H_matrix: Hamiltonian matrix
        max_moments: maximum moment order
        perturbation_indices: indices of H elements to differentiate w.r.t.
        
    Returns:
        Array of moment derivatives
    """
    def moments_fn(H):
        return calculate_moments(H, max_moments)
    
    if perturbation_indices is None:
        # Calculate full Jacobian
        return jax.jacfwd(moments_fn)(H_matrix)
    else:
        # Calculate derivatives for specific elements only
        def partial_moments_fn(params):
            H_modified = H_matrix.at[perturbation_indices].set(params)
            return calculate_moments(H_modified, max_moments)
        
        H_elements = H_matrix[perturbation_indices]
        return jax.jacfwd(partial_moments_fn)(H_elements)


# Test and utility functions

def test_moments_consistency(H_matrix: jnp.ndarray, max_moments: int = 10) -> bool:
    """
    Test consistency between trace and eigenvalue methods for moment calculation.
    
    Args:
        H_matrix: test Hamiltonian matrix
        max_moments: number of moments to test
        
    Returns:
        True if methods agree within tolerance
    """
    moments_trace = _calculate_moments_trace(H_matrix, max_moments)
    moments_eig = _calculate_moments_eigenvalues(H_matrix, max_moments)
    
    relative_error = jnp.abs(moments_trace - moments_eig) / (jnp.abs(moments_eig) + 1e-12)
    max_error = jnp.max(relative_error)
    
    return max_error < 1e-10


def create_test_hamiltonian(n: int, seed: int = 42) -> jnp.ndarray:
    """Create a random Hermitian matrix for testing."""
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (n, n))
    return 0.5 * (A + A.T)  # Make Hermitian