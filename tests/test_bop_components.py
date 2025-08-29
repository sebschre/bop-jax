"""
Test Suite for BOP-JAX Components

This module contains unit tests for all major components of the BOP-JAX
implementation, ensuring correctness and consistency with the theoretical
expectations of the BOP method.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space, partition
import pytest

# Import BOP modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bop_jax import energy, hamiltonian, moments, chebyshev, scf


class TestHamiltonian:
    """Test Hamiltonian construction module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple 2-atom system
        self.R = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.species = jnp.array([0, 0])
        self.box_size = 10.0
        self.displacement, self.shift = space.periodic(self.box_size)
        self.r_cutoff = 5.0
        
        # Create neighbor list
        neighbor_fn = partition.neighbor_list(self.displacement, self.box_size, self.r_cutoff)
        self.neighbor = neighbor_fn.allocate(self.R)
        
        # Default BOP parameters
        self.bop_params = energy.get_default_bop_params()
    
    def test_hamiltonian_shape(self):
        """Test that Hamiltonian has correct dimensions."""
        H_matrix, S_matrix = hamiltonian.build_hamiltonian(
            self.R, self.neighbor, self.displacement, self.species, self.bop_params
        )
        
        N = self.R.shape[0]
        orbitals_per_atom = 9
        expected_size = N * orbitals_per_atom
        
        assert H_matrix.shape == (expected_size, expected_size)
        assert S_matrix.shape == (expected_size, expected_size)
    
    def test_hamiltonian_hermiticity(self):
        """Test that Hamiltonian is Hermitian.""" 
        H_matrix, S_matrix = hamiltonian.build_hamiltonian(
            self.R, self.neighbor, self.displacement, self.species, self.bop_params
        )
        
        # Check Hermiticity (H† = H)
        H_hermitian_error = jnp.max(jnp.abs(H_matrix - H_matrix.T))
        assert H_hermitian_error < 1e-12, f"Hamiltonian not Hermitian: max error = {H_hermitian_error}"
    
    def test_onsite_energies(self):
        """Test that onsite energies are set correctly."""
        H_matrix, S_matrix = hamiltonian.build_hamiltonian(
            self.R, self.neighbor, self.displacement, self.species, self.bop_params
        )
        
        # Check diagonal elements match expected onsite energies
        onsite_energies = self.bop_params['onsite_energies']
        
        # First atom, s orbital (index 0)
        assert jnp.abs(H_matrix[0, 0] - onsite_energies['s']) < 1e-12
        
        # First atom, p orbitals (indices 1-3)  
        for i in range(1, 4):
            assert jnp.abs(H_matrix[i, i] - onsite_energies['p']) < 1e-12
        
        # First atom, d orbitals (indices 4-8)
        for i in range(4, 9):
            assert jnp.abs(H_matrix[i, i] - onsite_energies['d']) < 1e-12


class TestMoments:
    """Test moment calculation module."""
    
    def setup_method(self):
        """Create test Hamiltonian matrices."""
        # Create a simple test Hamiltonian
        n = 4
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (n, n))
        self.H_test = 0.5 * (A + A.T)  # Make Hermitian
        
        # Known eigenvalues for verification
        self.eigenvals = jnp.linalg.eigvals(self.H_test)
    
    def test_moment_calculation_methods(self):
        """Test that trace and eigenvalue methods give same results."""
        max_moments = 8
        
        moments_trace = moments._calculate_moments_trace(self.H_test, max_moments)
        moments_eig = moments._calculate_moments_eigenvalues(self.H_test, max_moments)
        
        # Should agree within numerical precision
        max_error = jnp.max(jnp.abs(moments_trace - moments_eig))
        assert max_error < 1e-10, f"Moment calculation methods disagree: max error = {max_error}"
    
    def test_moment_properties(self):
        """Test basic properties of calculated moments."""
        max_moments = 6
        mu_moments = moments.calculate_moments(self.H_test, max_moments)
        
        # μ₀ should equal the matrix dimension (trace of identity)
        assert jnp.abs(mu_moments[0] - self.H_test.shape[0]) < 1e-12
        
        # μ₁ should equal trace of H
        assert jnp.abs(mu_moments[1] - jnp.trace(self.H_test)) < 1e-12
        
        # Verify using eigenvalues: μₙ = Σᵢ λᵢⁿ
        for n in range(min(max_moments + 1, len(mu_moments))):
            expected = jnp.sum(self.eigenvals**n)
            error = jnp.abs(mu_moments[n] - expected)
            assert error < 1e-10, f"Moment μ_{n} incorrect: error = {error}"
    
    def test_recursion_coefficients(self):
        """Test moment to recursion coefficient conversion."""
        max_moments = 10
        mu_moments = moments.calculate_moments(self.H_test, max_moments)
        
        a_coeffs, b_coeffs = moments.moments_to_recursion(mu_moments)
        
        # Check that we get reasonable coefficients
        assert len(a_coeffs) == max_moments
        assert len(b_coeffs) == max_moments
        
        # b coefficients should be real and positive (for stable continued fraction)
        assert jnp.all(b_coeffs >= 0), "b coefficients should be non-negative"
        
        # a coefficients should be real
        assert jnp.all(jnp.isreal(a_coeffs)), "a coefficients should be real"
    
    def test_moment_derivatives(self):
        """Test automatic differentiation of moments."""
        max_moments = 4
        
        # Calculate moments and their derivatives
        def moments_fn(H):
            return moments.calculate_moments(H, max_moments)
        
        # Get Jacobian using autodiff
        jacobian = jax.jacfwd(moments_fn)(self.H_test)
        
        # Check Jacobian shape: (n_moments, H_rows, H_cols)
        expected_shape = (max_moments + 1, self.H_test.shape[0], self.H_test.shape[1])
        assert jacobian.shape == expected_shape
        
        # Check that derivatives are finite
        assert jnp.all(jnp.isfinite(jacobian)), "Moment derivatives contain non-finite values"


class TestChebyshev:
    """Test Chebyshev polynomial expansion module."""
    
    def setup_method(self):
        """Set up test data."""
        # Create test moments and recursion coefficients
        self.mu_moments = jnp.array([4.0, 0.0, 2.0, 0.0, 1.5, 0.0, 1.0])
        self.a_coeffs = jnp.array([0.0, 0.5, 0.0, 0.2, 0.0, 0.1])
        self.b_coeffs = jnp.array([0.0, 1.0, 0.8, 0.6, 0.5, 0.4])
        
        self.bop_params = {
            'fermi_level': 0.0,
            'gamma_0': 1.0,
            'orbital_factor': 10.0
        }
    
    def test_chebyshev_polynomials(self):
        """Test Chebyshev polynomial calculation."""
        x_test = 0.5
        max_order = 5
        
        polys = chebyshev.calculate_chebyshev_polynomials(x_test, max_order)
        
        # Check known values
        assert jnp.abs(polys[0] - 1.0) < 1e-12, "P₀(x) should equal 1"
        assert jnp.abs(polys[1] - x_test) < 1e-12, "P₁(x) should equal x"
        
        # Check recurrence relation: Pₙ₊₁(x) = 2x*Pₙ(x) - Pₙ₋₁(x)
        for n in range(2, max_order):
            expected = 2.0 * x_test * polys[n-1] - polys[n-2]
            assert jnp.abs(polys[n] - expected) < 1e-12, f"Recurrence relation failed for P_{n}"
    
    def test_sigma_coefficients(self):
        """Test sigma coefficient calculation."""
        max_expansion = 4
        sigma_coeffs = chebyshev.moments_to_sigma(
            self.mu_moments, self.a_coeffs, self.b_coeffs, max_expansion
        )
        
        assert len(sigma_coeffs) == max_expansion + 1
        assert jnp.all(jnp.isfinite(sigma_coeffs)), "Sigma coefficients contain non-finite values"
    
    def test_bond_energy_calculation(self):
        """Test bond energy calculation from sigma coefficients."""
        max_expansion = 4
        sigma_coeffs = chebyshev.moments_to_sigma(
            self.mu_moments, self.a_coeffs, self.b_coeffs, max_expansion
        )
        
        bond_energy = chebyshev.calculate_bond_energy(
            sigma_coeffs, self.a_coeffs, self.b_coeffs, self.bop_params
        )
        
        assert jnp.isfinite(bond_energy), "Bond energy should be finite"
        assert isinstance(bond_energy, (float, jnp.floating)), "Bond energy should be scalar"
    
    def test_local_density_of_states(self):
        """Test LDOS calculation."""
        max_expansion = 4
        sigma_coeffs = chebyshev.moments_to_sigma(
            self.mu_moments, self.a_coeffs, self.b_coeffs, max_expansion
        )
        
        a_inf = self.a_coeffs[-1]
        b_inf = self.b_coeffs[-1]
        test_energy = 0.0
        
        ldos = chebyshev.calculate_local_density_of_states(
            test_energy, sigma_coeffs, a_inf, b_inf
        )
        
        assert jnp.isfinite(ldos), "LDOS should be finite"
        assert ldos >= 0, "LDOS should be non-negative"


class TestSCF:
    """Test SCF framework."""
    
    def setup_method(self):
        """Set up SCF test."""
        self.n_onsites = 8
        self.initial_onsites = jnp.zeros(self.n_onsites)
        self.target_electrons = jnp.ones(self.n_onsites) * 1.0
    
    def test_scf_state_creation(self):
        """Test SCF state object."""
        state = scf.SCFState(
            self.initial_onsites, 0.0, self.target_electrons
        )
        
        assert jnp.array_equal(state.onsite_energies, self.initial_onsites)
        assert state.fermi_level == 0.0
        assert jnp.array_equal(state.electron_count, self.target_electrons)
        assert state.iteration == 0
        assert state.converged == False
    
    def test_fermi_level_determination(self):
        """Test Fermi level finder."""
        # Create a simple LDOS function for testing
        def test_ldos_fn(energy):
            # Simple linear function: N(E) = E + 5
            return jnp.clip(energy + 5.0, 0, 10)
        
        target_electrons = 3.0
        fermi_level = scf.determine_fermi_level(
            test_ldos_fn, target_electrons, energy_range=(-5.0, 5.0)
        )
        
        # Should find E_F = -2 since N(-2) = -2 + 5 = 3
        assert jnp.abs(fermi_level - (-2.0)) < 1e-6
    
    def test_broyden_mixer(self):
        """Test Broyden mixing algorithm.""" 
        mixer = scf.BroydenMixer(mixing_parameter=0.1, history_length=3)
        
        onsite_old = jnp.array([1.0, 2.0, 3.0])
        onsite_new = jnp.array([1.1, 2.1, 3.1])
        
        # Initialize history arrays
        n_sites = len(onsite_old)
        residual_history = jnp.zeros((mixer.history_length, n_sites))
        onsite_history = jnp.zeros((mixer.history_length, n_sites))
        
        mixed, new_res_hist, new_on_hist = mixer.mix(
            onsite_old, onsite_new, residual_history, onsite_history, iteration=0
        )
        
        # First iteration should be linear mixing
        expected = onsite_old + mixer.mixing_parameter * (onsite_new - onsite_old)
        assert jnp.allclose(mixed, expected), "First Broyden iteration should be linear mixing"


class TestIntegration:
    """Test full BOP energy function integration."""
    
    def setup_method(self):
        """Set up integration test."""
        # Create simple test system
        self.R = jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        self.species = jnp.array([0, 0])
        self.box_size = 10.0
        self.displacement, self.shift = space.periodic(self.box_size)
        self.r_cutoff = 5.0
        
        # Create BOP energy function
        self.bop_energy_fn = energy.bop_energy_fn(
            self.displacement, self.box_size, self.r_cutoff
        )
    
    def test_energy_function_call(self):
        """Test that energy function executes without error."""
        total_energy = self.bop_energy_fn(self.R, species=self.species)
        
        assert jnp.isfinite(total_energy), "Energy should be finite"
        assert isinstance(total_energy, (float, jnp.floating)), "Energy should be scalar"
    
    def test_force_calculation(self):
        """Test automatic force calculation."""
        # Calculate forces using autodiff
        force_fn = jax.grad(self.bop_energy_fn, argnums=0)
        forces = force_fn(self.R, species=self.species)
        
        assert forces.shape == self.R.shape, "Forces should have same shape as positions"
        assert jnp.all(jnp.isfinite(forces)), "All force components should be finite"
    
    def test_energy_force_consistency(self):
        """Test that forces are consistent with energy gradients."""
        # Calculate energy and forces
        energy = self.bop_energy_fn(self.R, species=self.species)
        forces = -jax.grad(self.bop_energy_fn, argnums=0)(self.R, species=self.species)
        
        # Test force is gradient of energy by finite differences
        h = 1e-6
        R_plus = self.R.at[0, 0].add(h)
        energy_plus = self.bop_energy_fn(R_plus, species=self.species)
        
        # Numerical derivative
        numerical_force = -(energy_plus - energy) / h
        analytical_force = forces[0, 0]
        
        error = jnp.abs(numerical_force - analytical_force)
        assert error < 1e-5, f"Force-energy consistency error: {error}"
    
    def test_translation_invariance(self):
        """Test that energy is invariant under translation."""
        # Calculate energy of original system
        energy1 = self.bop_energy_fn(self.R, species=self.species)
        
        # Translate entire system
        translation = jnp.array([1.0, 2.0, 3.0])
        R_translated = self.R + translation
        energy2 = self.bop_energy_fn(R_translated, species=self.species)
        
        # Energies should be identical (within numerical precision)
        energy_diff = jnp.abs(energy1 - energy2)
        assert energy_diff < 1e-12, f"Energy not translation invariant: diff = {energy_diff}"


def run_all_tests():
    """Run all tests and report results."""
    test_classes = [
        TestHamiltonian,
        TestMoments, 
        TestChebyshev,
        TestSCF,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    print("Running BOP-JAX Test Suite")
    print("=" * 50)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Run setup if it exists
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                getattr(test_instance, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)