"""
Self-Consistent Field (SCF) Framework for BOP Calculations

This module implements various SCF algorithms for solving the BOP equations
self-consistently, leveraging JAX's automatic differentiation for improved
convergence through second-order methods.

Key features:
- Multiple mixing schemes (linear, Broyden, DIIS)
- Automatic Fermi level determination
- Newton-Raphson SCF using automatic Hessians
- Integration with JAX optimization algorithms
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Dict, Any, Tuple, Optional
import functools


class SCFState:
    """Container for SCF iteration state."""
    
    def __init__(self, 
                 onsite_energies: jnp.ndarray,
                 fermi_level: float,
                 electron_count: jnp.ndarray,
                 iteration: int = 0,
                 converged: bool = False):
        self.onsite_energies = onsite_energies
        self.fermi_level = fermi_level  
        self.electron_count = electron_count
        self.iteration = iteration
        self.converged = converged


@jax.jit
def linear_mixing_scf(
    energy_fn: Callable,
    initial_onsites: jnp.ndarray,
    target_electrons: jnp.ndarray,
    mixing_parameter: float = 0.1,
    max_iterations: int = 50,
    tolerance: float = 1e-6
) -> SCFState:
    """
    Perform SCF iterations using simple linear mixing.
    
    This is the most basic SCF algorithm, corresponding to the original
    BOPfox implementation but with automatic differentiation.
    
    Args:
        energy_fn: function that returns (energy, new_onsites, fermi_level, electrons)
        initial_onsites: starting onsite energy values
        target_electrons: target electron count per orbital
        mixing_parameter: mixing coefficient (0 < α < 1)
        max_iterations: maximum SCF steps
        tolerance: convergence threshold
        
    Returns:
        Final SCF state
    """
    
    def scf_step(state: SCFState) -> SCFState:
        """Single SCF iteration with linear mixing."""
        # Calculate new quantities from current onsite energies
        energy, new_onsites, fermi_level, electrons = energy_fn(state.onsite_energies)
        
        # Linear mixing: onsite_new = (1-α)*onsite_old + α*onsite_calculated
        mixed_onsites = (1.0 - mixing_parameter) * state.onsite_energies + \
                       mixing_parameter * new_onsites
        
        # Check convergence  
        onsite_change = jnp.max(jnp.abs(mixed_onsites - state.onsite_energies))
        electron_error = jnp.max(jnp.abs(electrons - target_electrons))
        
        converged = (onsite_change < tolerance) and (electron_error < tolerance)
        
        return SCFState(
            onsite_energies=mixed_onsites,
            fermi_level=fermi_level,
            electron_count=electrons, 
            iteration=state.iteration + 1,
            converged=converged
        )
    
    # Initial state
    initial_state = SCFState(initial_onsites, 0.0, target_electrons)
    
    # SCF loop using lax.while_loop for JIT compatibility
    def continue_condition(state):
        return jnp.logical_and(state.iteration < max_iterations, 
                              jnp.logical_not(state.converged))
    
    final_state = lax.while_loop(continue_condition, scf_step, initial_state)
    
    return final_state


@jax.jit  
def newton_raphson_scf(
    residual_fn: Callable,
    initial_onsites: jnp.ndarray,
    max_iterations: int = 20,
    tolerance: float = 1e-8
) -> SCFState:
    """
    Perform SCF using Newton-Raphson method with automatic Hessian.
    
    This is a significant improvement over the original BOPfox implementation,
    enabled by JAX's automatic differentiation. The Hessian (second derivatives
    of energy w.r.t. onsite levels) is computed automatically.
    
    Args:
        residual_fn: function that computes SCF residual vector
        initial_onsites: starting onsite energies
        max_iterations: maximum Newton steps
        tolerance: convergence tolerance
        
    Returns:
        Converged SCF state
    """
    
    def newton_step(state: SCFState) -> SCFState:
        """Single Newton-Raphson step with automatic Hessian."""
        onsites = state.onsite_energies
        
        # Calculate residual and Jacobian using autodiff
        residual = residual_fn(onsites)
        jacobian = jax.jacfwd(residual_fn)(onsites)
        
        # Newton update: x_new = x_old - J^(-1) * residual
        try:
            delta = jnp.linalg.solve(jacobian, residual)
            new_onsites = onsites - delta
            
            # Check convergence
            residual_norm = jnp.linalg.norm(residual)
            converged = residual_norm < tolerance
            
            # Calculate updated properties for the state
            # (This would typically involve calling the energy function)
            fermi_level = 0.0  # Placeholder - would be calculated properly
            electrons = jnp.zeros_like(onsites)  # Placeholder
            
            return SCFState(
                onsite_energies=new_onsites,
                fermi_level=fermi_level,
                electron_count=electrons,
                iteration=state.iteration + 1, 
                converged=converged
            )
            
        except jnp.linalg.LinAlgError:
            # If Jacobian is singular, fall back to steepest descent
            gradient_step = 0.01 * residual
            new_onsites = onsites - gradient_step
            
            return SCFState(
                onsite_energies=new_onsites,
                fermi_level=state.fermi_level,
                electron_count=state.electron_count,
                iteration=state.iteration + 1,
                converged=False
            )
    
    initial_state = SCFState(initial_onsites, 0.0, jnp.zeros_like(initial_onsites))
    
    def continue_condition(state):
        return jnp.logical_and(state.iteration < max_iterations,
                              jnp.logical_not(state.converged))
    
    return lax.while_loop(continue_condition, newton_step, initial_state)


class BroydenMixer:
    """
    Broyden mixing for SCF convergence acceleration.
    
    This implements the quasi-Newton Broyden method for improved SCF convergence,
    maintaining a history of previous iterations to build an approximate inverse Jacobian.
    """
    
    def __init__(self, mixing_parameter: float = 0.1, history_length: int = 5):
        self.mixing_parameter = mixing_parameter
        self.history_length = history_length
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def mix(self, 
            onsite_old: jnp.ndarray,
            onsite_new: jnp.ndarray, 
            residual_history: jnp.ndarray,
            onsite_history: jnp.ndarray,
            iteration: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply Broyden mixing to improve SCF convergence.
        
        Args:
            onsite_old: previous onsite energies
            onsite_new: new onsite energies from current iteration  
            residual_history: history of SCF residuals
            onsite_history: history of onsite energies
            iteration: current iteration number
            
        Returns:
            Tuple of (mixed_onsites, updated_residual_history, updated_onsite_history)
        """
        residual = onsite_new - onsite_old
        
        if iteration == 0:
            # First iteration: simple linear mixing
            mixed = onsite_old + self.mixing_parameter * residual
            new_res_hist = residual_history.at[0].set(residual)
            new_on_hist = onsite_history.at[0].set(onsite_old)
            
        else:
            # Broyden mixing using history
            n_history = min(iteration, self.history_length)
            
            # Build matrices for Broyden update
            delta_res = residual_history[:n_history] - residual
            delta_ons = onsite_history[:n_history] - onsite_old
            
            # Broyden formula (simplified version)
            try:
                # Solve the linear system for Broyden coefficients
                A_matrix = delta_res @ delta_res.T + 1e-12 * jnp.eye(n_history)
                b_vector = delta_res @ residual
                coefficients = jnp.linalg.solve(A_matrix, b_vector)
                
                # Apply Broyden correction
                correction = jnp.sum(coefficients[:, None] * delta_ons, axis=0)
                mixed = onsite_old + self.mixing_parameter * (residual + correction)
                
            except:
                # Fall back to linear mixing if Broyden fails  
                mixed = onsite_old + self.mixing_parameter * residual
            
            # Update history
            hist_idx = iteration % self.history_length
            new_res_hist = residual_history.at[hist_idx].set(residual)
            new_on_hist = onsite_history.at[hist_idx].set(onsite_old)
        
        return mixed, new_res_hist, new_on_hist


@jax.jit
def determine_fermi_level(
    ldos_fn: Callable[[float], float],
    target_electrons: float,
    energy_range: Tuple[float, float] = (-10.0, 10.0),
    tolerance: float = 1e-8,
    max_iterations: int = 50
) -> float:
    """
    Determine Fermi level to achieve target electron count.
    
    Uses bisection method to find the Fermi energy such that the
    integrated density of states equals the target electron count.
    
    Args:
        ldos_fn: function that returns integrated electrons up to given energy
        target_electrons: desired electron count
        energy_range: search range for Fermi level
        tolerance: convergence tolerance  
        max_iterations: maximum bisection iterations
        
    Returns:
        Fermi level energy
    """
    
    def bisection_step(state):
        """Single step of bisection algorithm."""
        e_low, e_high, iteration = state
        e_mid = 0.5 * (e_low + e_high)
        
        electrons_mid = ldos_fn(e_mid)
        
        # Update bounds based on electron count
        new_low = jnp.where(electrons_mid < target_electrons, e_mid, e_low)
        new_high = jnp.where(electrons_mid < target_electrons, e_high, e_mid)
        
        return (new_low, new_high, iteration + 1)
    
    def continue_condition(state):
        e_low, e_high, iteration = state
        energy_diff = jnp.abs(e_high - e_low)
        return jnp.logical_and(iteration < max_iterations, energy_diff > tolerance)
    
    initial_state = (energy_range[0], energy_range[1], 0)
    final_low, final_high, _ = lax.while_loop(continue_condition, bisection_step, initial_state)
    
    return 0.5 * (final_low + final_high)


def create_scf_energy_function(
    bop_energy_fn: Callable,
    R: jnp.ndarray,
    neighbor,
    species: jnp.ndarray,
    bop_params: Dict[str, Any]
) -> Callable:
    """
    Create SCF energy function that includes onsite level updates.
    
    This creates a function that takes onsite energies as input and returns
    the BOP energy along with updated onsite levels and electron counts.
    
    Args:
        bop_energy_fn: base BOP energy function
        R: atomic positions
        neighbor: neighbor list
        species: atomic species
        bop_params: BOP parameters
        
    Returns:
        SCF energy function with signature f(onsites) -> (energy, new_onsites, fermi, electrons)
    """
    
    @jax.jit
    def scf_energy(onsite_energies: jnp.ndarray) -> Tuple[float, jnp.ndarray, float, jnp.ndarray]:
        """
        Evaluate energy and update SCF variables.
        
        Args:
            onsite_energies: current onsite energy values
            
        Returns:
            Tuple of (total_energy, new_onsite_energies, fermi_level, electron_counts)
        """
        # Update BOP parameters with current onsite energies
        updated_params = bop_params.copy()
        # Here we would update the onsite energies in the parameter dict
        # This is a simplified version
        
        # Calculate BOP energy
        energy = bop_energy_fn(R, neighbor=neighbor, species=species, **updated_params)
        
        # Calculate new onsite energies based on local environment
        # This would involve the full BOP self-consistency equations
        # For now, return placeholder values
        new_onsites = onsite_energies  # Placeholder
        fermi_level = 0.0  # Would be determined self-consistently
        electrons = jnp.ones_like(onsite_energies)  # Placeholder
        
        return energy, new_onsites, fermi_level, electrons
    
    return scf_energy


# Utility functions for SCF analysis

def analyze_scf_convergence(scf_state: SCFState) -> Dict[str, Any]:
    """Analyze SCF convergence properties."""
    return {
        'converged': scf_state.converged,
        'iterations': scf_state.iteration,
        'final_fermi_level': scf_state.fermi_level,
        'onsite_energy_range': (jnp.min(scf_state.onsite_energies), 
                               jnp.max(scf_state.onsite_energies)),
        'total_electrons': jnp.sum(scf_state.electron_count)
    }


def adaptive_mixing_parameter(iteration: int, residual_norm: float) -> float:
    """Calculate adaptive mixing parameter based on SCF history."""
    base_mixing = 0.1
    
    # Reduce mixing if residual is large (poor convergence)
    if residual_norm > 1.0:
        return base_mixing * 0.5
    elif residual_norm < 0.1:
        # Increase mixing if converging well
        return min(base_mixing * 2.0, 0.5)
    else:
        return base_mixing