"""
Chebyshev Polynomial Expansion for Local Density of States

This module implements the Chebyshev polynomial expansion of the local density
of states (LDOS) and related functions, corresponding to the bopcheb.p.f 
module in the original BOPfox implementation.

The key functions include:
- Conversion from moments to sigma expansion coefficients  
- Calculation of bond energy from sigma coefficients
- Density of states evaluation using Chebyshev polynomials
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial
import math


@partial(jax.jit, static_argnums=(3,))
def moments_to_sigma(
    mu_moments: jnp.ndarray,
    a_coeffs: jnp.ndarray, 
    b_coeffs: jnp.ndarray,
    max_expansion: int = 10
) -> jnp.ndarray:
    """
    Convert moments to Chebyshev expansion coefficients (sigma coefficients).
    
    This implements the transformation from recursion coefficients to the
    sigma expansion coefficients used in the bond energy calculation.
    Based on the BOPfox bopcheb module.
    
    Args:
        mu_moments: moment array μₙ
        a_coeffs: recursion coefficients aₙ  
        b_coeffs: recursion coefficients bₙ
        max_expansion: maximum order for sigma expansion
        
    Returns:
        sigma coefficients array σₙ
    """
    n_coeffs = min(len(a_coeffs), max_expansion)
    sigma_coeffs = jnp.zeros(max_expansion + 1)
    
    # Calculate asymptotic values
    a_inf = a_coeffs[-1] if len(a_coeffs) > 0 else 0.0
    b_inf = b_coeffs[-1] if len(b_coeffs) > 0 else 1.0
    
    # Normalize coefficients for Chebyshev expansion
    # Map from [-2b_inf, 2b_inf] interval to [-1, 1]
    norm_a = (a_coeffs - a_inf) / (2.0 * b_inf)
    norm_b = b_coeffs / (2.0 * b_inf)
    
    # Calculate sigma coefficients using orthogonality relations
    # This is a simplified version - full implementation would use
    # the complete Chebyshev orthogonality relations
    for n in range(min(n_coeffs, max_expansion + 1)):
        # Sigma coefficient calculation based on moments and recursion coefficients
        # σₙ represents the weight of the n-th Chebyshev polynomial in the expansion
        if n == 0:
            sigma_coeffs = sigma_coeffs.at[0].set(mu_moments[0] / (2.0 * b_inf))
        elif n < len(norm_a):
            # Use recursion relation with normalized coefficients  
            sigma_coeffs = sigma_coeffs.at[n].set(
                _calculate_sigma_coefficient(n, mu_moments, norm_a, norm_b, b_inf)
            )
    
    return sigma_coeffs


@partial(jax.jit, static_argnums=(0,))
def _calculate_sigma_coefficient(
    n: int,
    mu_moments: jnp.ndarray,
    norm_a: jnp.ndarray,
    norm_b: jnp.ndarray, 
    b_inf: float
) -> float:
    """
    Calculate individual sigma coefficient using Chebyshev orthogonality.
    
    This is a simplified implementation of the complex orthogonality
    relations used in the original BOPfox code.
    """
    if n >= len(mu_moments):
        return 0.0
    
    # Simplified calculation - in full implementation this would use
    # the complete Chebyshev polynomial orthogonality relations
    weight = 1.0 / (math.pi * b_inf)
    
    if n < len(norm_a):
        # Include effects of recursion coefficients  
        coeff_factor = 1.0 - norm_a[n]**2 - norm_b[n]**2 if n < len(norm_b) else 1.0
        weight *= jnp.sqrt(jnp.abs(coeff_factor))
    
    return weight * mu_moments[n] if n < len(mu_moments) else 0.0


@jax.jit
def calculate_bond_energy(
    sigma_coeffs: jnp.ndarray,
    a_coeffs: jnp.ndarray,
    b_coeffs: jnp.ndarray, 
    bop_params: dict
) -> float:
    """
    Calculate bond energy from sigma expansion coefficients.
    
    This implements the core BOP energy formula:
    E_bond = 10 * b_inf * Σₙ σₙ [χₙ₊₂(φ_F) - γ₀ χₙ₊₁(φ_F) + χₙ(φ_F)]
    
    where χₙ are the Chebyshev response functions.
    
    Args:
        sigma_coeffs: Chebyshev expansion coefficients σₙ
        a_coeffs: recursion coefficients aₙ
        b_coeffs: recursion coefficients bₙ  
        bop_params: BOP parameters including Fermi level
        
    Returns:
        Bond energy contribution
    """
    # Get asymptotic coefficients
    a_inf = a_coeffs[-1] if len(a_coeffs) > 0 else 0.0
    b_inf = b_coeffs[-1] if len(b_coeffs) > 0 else 1.0
    
    # Get Fermi level (could be determined self-consistently)
    fermi_level = bop_params.get('fermi_level', 0.0)
    
    # Calculate normalized Fermi level  
    phi_F = (fermi_level - a_inf) / (2.0 * b_inf)
    
    # Clamp to valid range [-1, 1] for Chebyshev polynomials
    phi_F = jnp.clip(phi_F, -0.99, 0.99)
    
    # Calculate Chebyshev response functions χₙ(φ_F)
    max_order = len(sigma_coeffs) + 2
    chi_values = calculate_chebyshev_response_functions(phi_F, max_order)
    
    # BOP energy formula with γ₀ parameter  
    gamma_0 = bop_params.get('gamma_0', 1.0)  # Termination parameter
    bond_energy = 0.0
    
    # Number of d-orbitals (10 for full d-shell)
    orbital_factor = bop_params.get('orbital_factor', 10.0)
    
    for n in range(len(sigma_coeffs)):
        if n + 2 < len(chi_values):
            energy_contribution = (
                chi_values[n + 2] - gamma_0 * chi_values[n + 1] + chi_values[n]
            )
            bond_energy += sigma_coeffs[n] * energy_contribution
    
    return orbital_factor * b_inf * bond_energy


@partial(jax.jit, static_argnums=(1,))
def calculate_chebyshev_response_functions(phi_F: float, max_order: int) -> jnp.ndarray:
    """
    Calculate Chebyshev response functions χₙ(φ_F).
    
    These are the integrated Chebyshev polynomials that appear in the 
    BOP energy expression. For φ_F in [-1, 1]:
    
    χ₁(φ_F) = 1 - φ_F/π + sin(2φ_F)/(2π)  
    χₙ(φ_F) = recursive relations involving Chebyshev polynomials
    
    Args:
        phi_F: normalized Fermi level in [-1, 1]
        max_order: maximum order of response functions to calculate
        
    Returns:
        Array of response function values χ₀, χ₁, ..., χₙ
    """
    chi_values = jnp.zeros(max_order)
    
    # Calculate the angle for Chebyshev evaluation
    # φ_F = cos(θ) maps [-1,1] to [π,0]  
    theta = jnp.arccos(jnp.clip(phi_F, -0.9999, 0.9999))
    
    # χ₀(φ_F) = 1 (normalization)
    chi_values = chi_values.at[0].set(1.0)
    
    if max_order > 1:
        # χ₁(φ_F) = 1 - θ/π  
        chi_values = chi_values.at[1].set(1.0 - theta / math.pi)
    
    if max_order > 2:
        # χ₂(φ_F) and higher orders using recursion
        # These involve integrals of Chebyshev polynomials
        for n in range(2, max_order):
            chi_n = _calculate_chi_n(n, theta, phi_F)
            chi_values = chi_values.at[n].set(chi_n)
    
    return chi_values


@partial(jax.jit, static_argnums=(0,))
def _calculate_chi_n(n: int, theta: float, phi_F: float) -> float:
    """
    Calculate individual Chebyshev response function χₙ.
    
    Uses the recursive relations for integrated Chebyshev polynomials.
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return 1.0 - theta / math.pi
    else:
        # For n ≥ 2, use the integral relations
        # χₙ = (sin(nθ) - n*sin((n-1)θ)) / (π*n*(n-1)) + correction terms
        sin_n_theta = jnp.sin(n * theta)
        sin_n1_theta = jnp.sin((n - 1) * theta)
        
        # Avoid division by zero
        if n == 1:
            return 1.0 - theta / math.pi
        
        main_term = (sin_n_theta - n * sin_n1_theta) / (math.pi * n * (n - 1))
        
        # Add corrections for proper normalization
        if n % 2 == 0:  # Even n
            correction = 0.0
        else:  # Odd n  
            correction = -2.0 / (math.pi * (n * n - 1))
        
        return main_term + correction


@jax.jit
def calculate_local_density_of_states(
    energy: float,
    sigma_coeffs: jnp.ndarray,
    a_inf: float,
    b_inf: float
) -> float:
    """
    Calculate local density of states at given energy.
    
    LDOS(E) = (2/π) * √(1-ε²) * Σₙ σₙ Pₙ(ε)
    where ε = (E - a_inf)/(2*b_inf) and Pₙ are Chebyshev polynomials.
    
    Args:
        energy: energy at which to evaluate LDOS
        sigma_coeffs: Chebyshev expansion coefficients
        a_inf: asymptotic recursion coefficient
        b_inf: asymptotic recursion coefficient  
        
    Returns:
        Local density of states value
    """
    # Normalize energy to [-1, 1] interval
    epsilon = (energy - a_inf) / (2.0 * b_inf)
    epsilon = jnp.clip(epsilon, -0.9999, 0.9999)
    
    # Calculate weight factor √(1-ε²)
    weight = jnp.sqrt(1.0 - epsilon**2)
    
    # Calculate Chebyshev polynomials Pₙ(ε)
    max_order = len(sigma_coeffs)
    chebyshev_polys = calculate_chebyshev_polynomials(epsilon, max_order)
    
    # Sum the expansion
    ldos_sum = jnp.sum(sigma_coeffs * chebyshev_polys)
    
    # Final LDOS with normalization
    ldos = (2.0 / math.pi) * weight * ldos_sum
    
    return ldos


@partial(jax.jit, static_argnums=(1,))
def calculate_chebyshev_polynomials(x: float, max_order: int) -> jnp.ndarray:
    """
    Calculate Chebyshev polynomials of the first kind Pₙ(x).
    
    Uses the recurrence relation:
    P₀(x) = 1, P₁(x) = x, Pₙ₊₁(x) = 2x*Pₙ(x) - Pₙ₋₁(x)
    
    Args:
        x: evaluation point in [-1, 1]
        max_order: maximum polynomial order
        
    Returns:
        Array of polynomial values P₀(x), P₁(x), ..., Pₙ(x)
    """
    if max_order == 0:
        return jnp.array([1.0])
    
    polys = jnp.zeros(max_order)
    polys = polys.at[0].set(1.0)  # P₀(x) = 1
    
    if max_order > 1:
        polys = polys.at[1].set(x)  # P₁(x) = x
        
        # Recurrence relation for n ≥ 2
        for n in range(2, max_order):
            polys = polys.at[n].set(2.0 * x * polys[n-1] - polys[n-2])
    
    return polys


# Utility functions for analysis and testing

def calculate_total_electrons(
    sigma_coeffs: jnp.ndarray,
    a_inf: float, 
    b_inf: float,
    fermi_level: float,
    orbitals_per_atom: int = 9
) -> float:
    """
    Calculate total number of electrons per atom at given Fermi level.
    
    N(E_F) = ∫_{-∞}^{E_F} LDOS(E) dE = orbitals_per_atom * Σₙ σₙ χₙ₊₁(φ_F)
    """
    phi_F = (fermi_level - a_inf) / (2.0 * b_inf)
    phi_F = jnp.clip(phi_F, -0.99, 0.99)
    
    max_order = len(sigma_coeffs) + 1
    chi_values = calculate_chebyshev_response_functions(phi_F, max_order)
    
    electron_sum = jnp.sum(sigma_coeffs * chi_values[1:len(sigma_coeffs)+1])
    
    return orbitals_per_atom * electron_sum