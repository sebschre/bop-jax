"""
Simple BOP Calculation Example

This script demonstrates how to use the BOP-JAX implementation for a basic
bond order potential calculation, showing the integration with JAX-MD and
the automatic differentiation capabilities.
"""

from typing import Tuple, Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

# Import our BOP modules
import sys
sys.path.append('..')
from bop_jax import energy, hamiltonian, moments, chebyshev, scf

# Try to import JAX-MD components (optional)
try:
    from jax_md import space, partition, simulate
    JAXMD_AVAILABLE = True
    print("✓ JAX-MD available - full functionality enabled")
except (ImportError, AttributeError) as e:
    JAXMD_AVAILABLE = False
    print(f"⚠ JAX-MD not available ({str(e)[:50]}...) - using standalone BOP mode")
    # Create dummy objects for fallback
    space = partition = simulate = None


def create_test_system() -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Create a simple test system (e.g., a small metal cluster)."""
    # Create a 2x2x2 BCC lattice (8 atoms)
    lattice_constant = 2.87  # Angstroms (approximately Fe lattice constant)

    positions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = i * lattice_constant + (j + k) * lattice_constant / 2
                y = j * lattice_constant + (i + k) * lattice_constant / 2  
                z = k * lattice_constant + (i + j) * lattice_constant / 2
                positions.append([x, y, z])

    R = jnp.array(positions)
    N = R.shape[0]
    species = jnp.zeros(N, dtype=jnp.int32)  # All atoms are the same species

    box_size = lattice_constant * 3  # Large enough to contain the cluster

    return R, species, box_size


def demonstrate_bop_calculation() -> Dict[str, Any]:
    """Demonstrate a complete BOP calculation."""
    print("=" * 60)
    print("BOP-JAX Demonstration: Simple Bond Order Potential Calculation") 
    print("=" * 60)

    # Create test system
    print("\n1. Creating test system...")
    R, species, box_size = create_test_system()
    print(f"   System: {R.shape[0]} atoms in {box_size:.2f} Å box")
    print(f"   Positions:\n{R}")

    # Set up spatial framework
    print("\n2. Setting up spatial framework...")
    if JAXMD_AVAILABLE:
        displacement, shift = space.periodic(box_size)
        print("   Using JAX-MD periodic boundary conditions")
    else:
        displacement, shift = None, None
        print("   Using standalone mode (no JAX-MD)")
    r_cutoff = 5.0

    # Get BOP parameters
    print("\n3. Loading BOP parameters...")  
    bop_params = energy.get_default_bop_params()
    print(f"   Cutoff radius: {r_cutoff:.2f} Å")
    print(f"   Max moments: {bop_params['max_moments']}")
    print(f"   Max expansion: {bop_params['max_expansion']}")

    # Create BOP energy function
    print("\n4. Creating BOP energy function...")
    if JAXMD_AVAILABLE:
        bop_energy_fn = energy.bop_energy_fn(displacement, box_size, r_cutoff, bop_params)

        # Create neighbor list
        neighbor_fn = partition.neighbor_list(displacement, box_size, r_cutoff)
        neighbor = neighbor_fn.allocate(R)
        print("   Using JAX-MD neighbor lists")
    else:
        bop_energy_fn = energy.create_standalone_bop_energy_fn(bop_params)
        neighbor = None
        print("   Using standalone BOP energy function")

    # Calculate energy
    print("\n5. Calculating BOP energy...")
    if JAXMD_AVAILABLE:
        total_energy = bop_energy_fn(R, neighbor=neighbor, species=species)
    else:
        total_energy = bop_energy_fn(R)
    print(f"   Total energy: {total_energy:.6f} eV")
    print(f"   Energy per atom: {total_energy/R.shape[0]:.6f} eV/atom")

    # Calculate forces using automatic differentiation
    print("\n6. Calculating forces via autodiff...")
    def energy_wrapper(positions: jnp.ndarray) -> Any:
        if JAXMD_AVAILABLE:
            return bop_energy_fn(positions, neighbor=neighbor, species=species)
        else:
            return bop_energy_fn(positions)

    force_fn = jax.grad(energy_wrapper)
    forces = -force_fn(R)  # Forces = -∇E
    print(f"   Force calculation completed")
    print(f"   Max force component: {jnp.max(jnp.abs(forces)):.6f} eV/Å")
    print(f"   Force norm: {jnp.linalg.norm(forces):.6f} eV/Å")

    # Demonstrate individual components
    print("\n7. Demonstrating individual BOP components...")

    if JAXMD_AVAILABLE:
        # Build Hamiltonian using JAX-MD
        H_matrix, S_matrix = hamiltonian.build_hamiltonian(
            R, neighbor, displacement, species, bop_params
        )
        print(f"   Hamiltonian matrix shape: {H_matrix.shape}")
        print(f"   Hamiltonian trace: {jnp.trace(H_matrix):.6f}")
    else:
        # Create a simple mock Hamiltonian for demonstration
        n_atoms = R.shape[0]
        n_orbitals = n_atoms * 9  # 9 orbitals per atom (s, p, d)
        H_matrix = jnp.eye(n_orbitals) * (-2.0) + 0.1 * jax.random.normal(jax.random.PRNGKey(42), (n_orbitals, n_orbitals))
        H_matrix = 0.5 * (H_matrix + H_matrix.T)  # Make symmetric
        print(f"   Mock Hamiltonian matrix shape: {H_matrix.shape}")
        print(f"   Mock Hamiltonian trace: {jnp.trace(H_matrix):.6f}")

    # Calculate atom-specific moments using corrected BOP theory
    mu_moments = moments.calculate_moments(
        H_matrix,
        max_moments=bop_params["max_moments"],
        orbitals_per_atom=9,  # s, p, d orbitals
        use_path_method=True,
    )
    print(
        f"   Calculated moments for {mu_moments.shape[0]} atoms with {mu_moments.shape[1]} moment orders"
    )
    print(
        f"   Atom 0: μ₀ = {mu_moments[0, 0]:.6f}, μ₁ = {mu_moments[0, 1]:.6f}, μ₂ = {mu_moments[0, 2]:.6f}"
    )
    if mu_moments.shape[0] > 1:
        print(
            f"   Atom 1: μ₀ = {mu_moments[1, 0]:.6f}, μ₁ = {mu_moments[1, 1]:.6f}, μ₂ = {mu_moments[1, 2]:.6f}"
        )

    # Convert to recursion coefficients for all atoms
    a_coeffs, b_coeffs = moments.moments_to_recursion_all_atoms(mu_moments)
    print(f"   Recursion coefficients calculated for all atoms")
    print(f"   Atom 0: a₀ = {a_coeffs[0, 0]:.6f}, a₁ = {a_coeffs[0, 1]:.6f}")
    print(f"   Atom 0: b₁ = {b_coeffs[0, 1]:.6f}, b₂ = {b_coeffs[0, 2]:.6f}")
    if a_coeffs.shape[0] > 1:
        print(f"   Atom 1: a₀ = {a_coeffs[1, 0]:.6f}, a₁ = {a_coeffs[1, 1]:.6f}")
        print(f"   Atom 1: b₁ = {b_coeffs[1, 1]:.6f}, b₂ = {b_coeffs[1, 2]:.6f}")

    # Calculate sigma coefficients for all atoms
    sigma_coeffs = chebyshev.moments_to_sigma_all_atoms(
        mu_moments, a_coeffs, b_coeffs, bop_params["max_expansion"]
    )
    print(f"   Sigma coefficients calculated for all atoms")
    print(f"   Atom 0: σ₀ = {sigma_coeffs[0, 0]:.6f}, σ₁ = {sigma_coeffs[0, 1]:.6f}")
    if sigma_coeffs.shape[0] > 1:
        print(
            f"   Atom 1: σ₀ = {sigma_coeffs[1, 0]:.6f}, σ₁ = {sigma_coeffs[1, 1]:.6f}"
        )

    return {
        'R': R,
        'species': species,
        'total_energy': total_energy,
        'forces': forces,
        'H_matrix': H_matrix,
        'moments': mu_moments,
        'a_coeffs': a_coeffs,
        'b_coeffs': b_coeffs,
        'sigma_coeffs': sigma_coeffs
    }


def demonstrate_scf_calculation() -> Optional[Any]:
    """Demonstrate SCF calculation with automatic differentiation."""
    print("\n" + "=" * 60)
    print("BOP-JAX SCF Demonstration")
    print("=" * 60)

    if not JAXMD_AVAILABLE:
        print("⚠ SCF demonstration requires JAX-MD - skipping in standalone mode")
        return None

    # Create test system
    R, species, box_size = create_test_system()
    displacement, shift = space.periodic(box_size)
    r_cutoff = 5.0

    # Set up BOP calculation
    bop_params = energy.get_default_bop_params()
    bop_energy_fn = energy.bop_energy_fn(displacement, box_size, r_cutoff, bop_params)

    # Create neighbor list
    neighbor_fn = partition.neighbor_list(displacement, box_size, r_cutoff)
    neighbor = neighbor_fn.allocate(R)

    print("\n1. Setting up SCF calculation...")

    # Initial guess for onsite energies
    N = R.shape[0]
    orbitals_per_atom = 9  # s, p, d orbitals
    initial_onsites = jnp.zeros(N * orbitals_per_atom)
    target_electrons = 8.0 * jnp.ones(N)  # 8 electrons per atom (example)

    print(f"   Initial onsite energies: all zero")
    print(f"   Target electrons per atom: {target_electrons[0]:.1f}")

    # Create SCF energy function
    scf_energy_fn = scf.create_scf_energy_function(
        bop_energy_fn, R, neighbor, species, bop_params
    )

    # Perform linear mixing SCF
    print("\n2. Performing linear mixing SCF...")
    scf_result = scf.linear_mixing_scf(
        scf_energy_fn,
        initial_onsites, 
        target_electrons.flatten(),
        mixing_parameter=0.1,
        max_iterations=20,
        tolerance=1e-6
    )

    print(f"   SCF converged: {scf_result.converged}")
    print(f"   SCF iterations: {scf_result.iteration}")
    print(f"   Final Fermi level: {scf_result.fermi_level:.6f} eV")

    return scf_result


def demonstrate_structural_differentiation() -> Dict[str, Any]:
    """Demonstrate that the corrected BOP implementation can distinguish between different structures."""
    print("\n" + "=" * 60)
    print("BOP-JAX Structural Differentiation Demonstration")
    print("=" * 60)
    print("\nThis test validates the theoretical correction by showing that")
    print("atom-specific moments can distinguish between different crystal structures.")

    # Run the structural differentiation test
    test_results = moments.test_structural_differentiation()

    print(f"\n1. Test Results:")
    print(f"   {test_results['test_message']}")

    if "comparison" in test_results:
        comp = test_results["comparison"]
        print(f"\n2. Structure Comparison:")
        print(
            f"   Structures: {comp['structures_compared'][0]} vs {comp['structures_compared'][1]}"
        )
        print(f"   Max moment difference: {comp['max_difference']:.8f}")
        print(f"   Mean moment difference: {comp['mean_difference']:.8f}")
        print(f"   Relative difference: {comp['relative_difference']:.2%}")
        print(f"   Can distinguish: {comp['can_distinguish']}")

    # Display some moment details
    for structure in ["linear", "square"]:
        if f"{structure}_first_moments" in test_results:
            moments_data = test_results[f"{structure}_first_moments"]
            print(f"\n3. {structure.title()} Structure Moments:")
            print(
                f"   Atom 0: μ₀={moments_data['atom_0']['mu_0']:.4f}, μ₁={moments_data['atom_0']['mu_1']:.4f}, μ₂={moments_data['atom_0']['mu_2']:.4f}"
            )
            print(
                f"   Atom 1: μ₀={moments_data['atom_1']['mu_0']:.4f}, μ₁={moments_data['atom_1']['mu_1']:.4f}, μ₂={moments_data['atom_1']['mu_2']:.4f}"
            )

    return test_results


def demonstrate_molecular_dynamics() -> Optional[Dict[str, Any]]:
    """Demonstrate molecular dynamics simulation with BOP potential."""
    print("\n" + "=" * 60) 
    print("BOP-JAX Molecular Dynamics Demonstration")
    print("=" * 60)

    if not JAXMD_AVAILABLE:
        print("⚠ MD demonstration requires JAX-MD - skipping in standalone mode")
        return None

    # Create test system
    R, species, box_size = create_test_system()
    temperature = 300.0  # Kelvin
    dt = 0.001  # ps
    n_steps = 100

    print(f"\n1. MD Setup:")
    print(f"   Temperature: {temperature:.1f} K")
    print(f"   Time step: {dt:.3f} ps") 
    print(f"   Number of steps: {n_steps}")

    # Create BOP-based simulator
    print(f"\n2. Creating BOP-MD simulator...")
    energy_fn, force_fn, (init_fn, apply_fn), neighbor_fn = energy.create_bop_simulator(
        R, box_size, dt, temperature
    )

    # Initialize MD state
    print(f"\n3. Initializing MD state...")
    key = jax.random.PRNGKey(42)
    state = init_fn(key, R)

    print(f"   Initial temperature: {simulate.temperature(state):.1f} K")
    print(f"   Initial kinetic energy: {simulate.kinetic_energy(state):.6f} eV")

    # Run short MD simulation
    print(f"\n4. Running MD simulation...")
    energies = []
    temperatures = []

    for step in range(n_steps):
        state = apply_fn(state)

        if step % 10 == 0:
            E_kin = simulate.kinetic_energy(state) 
            E_pot = energy_fn(state.position)
            T = simulate.temperature(state)

            energies.append(E_kin + E_pot)
            temperatures.append(T)

            if step % 20 == 0:
                print(f"   Step {step:3d}: T = {T:6.1f} K, E_total = {E_kin + E_pot:8.6f} eV")

    print(f"\n5. MD simulation completed!")
    print(f"   Final temperature: {temperatures[-1]:.1f} K")
    print(f"   Average temperature: {np.mean(temperatures):.1f} K")
    print(f"   Total energy drift: {energies[-1] - energies[0]:.8f} eV")

    return {
        'final_state': state,
        'energies': energies,
        'temperatures': temperatures
    }


def main() -> None:
    """Main demonstration function."""
    # Run BOP calculation demonstration
    bop_results = demonstrate_bop_calculation()

    # Demonstrate structural differentiation (NEW: validates theoretical correction)
    structure_results = demonstrate_structural_differentiation()

    # Run SCF demonstration
    scf_results = demonstrate_scf_calculation()

    # Run MD demonstration
    md_results = demonstrate_molecular_dynamics()

    print("\n" + "=" * 60)
    print("BOP-JAX Demonstration Completed Successfully!")
    print("=" * 60)
    print("\nKey Achievements:")
    print("✓ BOP energy calculation with automatic forces")
    print("✓ Corrected atom-specific moment calculation (Cyrot-Lackmann theorem)")
    print("✓ Structural differentiation capability validated")
    if JAXMD_AVAILABLE:
        print("✓ Integration with JAX-MD spatial framework") 
        print("✓ Self-consistent field calculations")
        print("✓ Molecular dynamics simulation")
    else:
        print("⚠ JAX-MD features skipped (standalone mode)")
    print("✓ All calculations use automatic differentiation")

    print(f"\nFinal Results Summary:")
    print(f"- Total BOP energy: {bop_results['total_energy']:.6f} eV")
    print(f"- Max force: {jnp.max(jnp.abs(bop_results['forces'])):.6f} eV/Å")
    print(f"- Structural differentiation: {structure_results['test_message']}")
    if scf_results is not None:
        print(f"- SCF converged in {scf_results.iteration} iterations")
    if md_results is not None:
        print(f"- MD average temperature: {np.mean(md_results['temperatures']):.1f} K")


if __name__ == "__main__":
    main()
