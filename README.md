# BOP-JAX: Bond Order Potential with JAX and Automatic Differentiation

A modern implementation of the Bond Order Potential (BOP) method using JAX and JAX-MD, enabling automatic differentiation and GPU acceleration for molecular dynamics simulations.

## Overview

This project migrates the Fortran-based BOPfox implementation to a Python/JAX framework, providing:

- **Automatic differentiation**: No manual force derivation needed
- **Second-order derivatives**: Enable advanced SCF optimization methods  
- **JAX-MD integration**: Leverage optimized spatial algorithms and MD simulators
- **GPU acceleration**: Scale to larger systems with hardware acceleration
- **Modern Python ecosystem**: Clean, maintainable, and extensible codebase

## Key Features

### 🚀 **Automatic Forces and Derivatives**
- Forces calculated via `jax.grad(energy_fn)` - no analytical derivation required
- Second derivatives `∂²E/∂εᵢ∂εⱼ` for Newton-Raphson SCF methods
- Full gradient information for advanced optimization algorithms

### 🔧 **JAX-MD Integration**  
- Uses JAX-MD's optimized neighbor lists and spatial partitioning
- Standard energy function interface compatible with all JAX-MD simulators
- Built-in support for periodic boundary conditions and various cell geometries

### ⚡ **High Performance**
- JIT compilation for optimal performance
- GPU/TPU acceleration through JAX
- Vectorized operations and efficient memory usage
- Batched calculations for multiple configurations

### 🧮 **Advanced SCF Methods**
- Linear mixing, Broyden mixing, and Newton-Raphson SCF
- Automatic Fermi level determination  
- Leverages autodiff for improved convergence

## Installation

### Using Poetry (Recommended)

```bash
# Install Poetry (if you don't have it)
curl -sSL https://install.python-poetry.org | python3 -

# Clone and install
git clone <repository-url>

# Install with Poetry (handles all version conflicts automatically)
poetry install

# Activate environment and test
poetry shell
poetry run bop-test minimal

# Optional: Add GPU support
poetry run pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Using pip (Manual)

```bash
# Clone the repository
git clone <repository-url>

# Install compatible versions manually
pip install jax==0.4.20 jaxlib==0.4.20 jax-md==0.2.8
pip install numpy scipy matplotlib

# Test installation
python examples/minimal_test.py
```

## Quick Start

### Poetry Environment
```bash
# Activate Poetry shell
poetry shell

# Run built-in examples
poetry run bop-test minimal  # Test core algorithms
poetry run bop-test example  # Full BOP calculation
```

### Python API
```python
import jax
import jax.numpy as jnp
from jax_md import space
from bop_jax import energy

# Create atomic system
R = jnp.array([[0.0, 0.0, 0.0],    # Atomic positions
               [1.5, 0.0, 0.0],
               [0.0, 1.5, 0.0]])
species = jnp.array([0, 0, 0])      # Atomic species
box_size = 10.0

# Set up JAX-MD spatial framework  
displacement, shift = space.periodic(box_size)

# Create BOP energy function
bop_energy_fn = energy.bop_energy_fn(displacement, box_size, r_cutoff=5.0)

# Calculate energy and forces (automatic differentiation!)
total_energy = bop_energy_fn(R, species=species)
forces = -jax.grad(bop_energy_fn)(R, species=species)

print(f"Total energy: {total_energy:.6f} eV")
print(f"Max force: {jnp.max(jnp.abs(forces)):.6f} eV/Å")
```

## Core Components

### 1. Energy Module (`bop_jax.energy`)
Main interface compatible with JAX-MD simulators:
- `bop_energy_fn()`: Creates BOP energy function  
- `create_bop_simulator()`: Complete MD simulator setup
- Automatic force calculation through autodiff

### 2. Hamiltonian Construction (`bop_jax.hamiltonian`)  
Tight-binding matrix construction:
- Uses JAX-MD neighbor lists for connectivity
- Slater-Koster bond integrals with distance dependence
- Environment-dependent onsite energies

### 3. Moment Calculations (`bop_jax.moments`)
Core BOP algorithm implementation:
- Moment calculation: `μₙ = Tr[H^n]` 
- Continued fraction conversion to recursion coefficients
- Automatic differentiation for all intermediate steps

### 4. Chebyshev Expansion (`bop_jax.chebyshev`)
Local density of states expansion:
- Sigma coefficient calculation from moments
- Bond energy evaluation using Chebyshev polynomials  
- Density of states and electron counting

### 5. SCF Framework (`bop_jax.scf`)
Self-consistent field algorithms:
- Linear mixing, Broyden mixing, Newton-Raphson methods
- Automatic Fermi level determination
- Leverages autodiff for improved convergence

## Algorithm Overview

The BOP method calculates bond energies through these steps:

1. **Hamiltonian Construction**: Build tight-binding matrix from atomic positions
2. **Moment Calculation**: Compute `μₙ = Tr[H^n]` characterizing local density of states  
3. **Continued Fraction**: Convert moments to recursion coefficients `aₙ, bₙ`
4. **Chebyshev Expansion**: Expand LDOS using Chebyshev polynomials → sigma coefficients
5. **Bond Energy**: Calculate energy from sigma coefficients and response functions
6. **SCF Loop**: Iterate until self-consistent onsite energies are achieved

## Examples

### Basic BOP Calculation
```python
# See examples/simple_bop_calculation.py for complete demonstration
python examples/simple_bop_calculation.py
```

### Molecular Dynamics Simulation
```python
from bop_jax.energy import create_bop_simulator
from jax_md import simulate

# Create BOP-based simulator
energy_fn, force_fn, (init_fn, apply_fn), neighbor_fn = create_bop_simulator(
    R_init, box_size, dt=0.001, temperature=300.0
)

# Run MD simulation  
key = jax.random.PRNGKey(42)
state = init_fn(key, R_init)

for step in range(1000):
    state = apply_fn(state)
    
    if step % 100 == 0:
        print(f"Step {step}: T = {simulate.temperature(state):.1f} K")
```

### Advanced SCF with Newton-Raphson
```python
from bop_jax import scf

# Create SCF residual function
def residual_fn(onsite_energies):
    # Return SCF residual vector
    return computed_onsites - target_onsites

# Perform Newton-Raphson SCF (uses automatic Hessian)
scf_result = scf.newton_raphson_scf(residual_fn, initial_onsites)
print(f"Converged in {scf_result.iteration} iterations")
```

## Comparison with BOPfox

| Feature | BOPfox (Fortran) | BOP-JAX |
|---------|------------------|---------|
| **Forces** | Manual analytical derivation | Automatic via `jax.grad()` |
| **Second Derivatives** | Not implemented | Automatic via `jax.hessian()` |  
| **SCF Methods** | Linear mixing only | Linear, Broyden, Newton-Raphson |
| **Performance** | CPU-optimized Fortran | GPU/TPU acceleration |
| **Integration** | Standalone code | JAX-MD ecosystem |
| **Development** | Complex derivative maintenance | Automatic differentiation |

## Performance Benefits

- **No Manual Derivatives**: Eliminates error-prone analytical force derivation
- **Advanced SCF**: Second-order methods enabled by automatic Hessians  
- **GPU Scaling**: Seamless acceleration for larger systems
- **Vectorization**: Efficient batch processing of multiple configurations
- **JIT Compilation**: Near-C performance with Python flexibility

## Development Status

### ✅ Completed
- [x] Project structure and JAX-MD integration
- [x] Core BOP algorithm implementation (moments, continued fractions, Chebyshev)  
- [x] Hamiltonian construction with neighbor lists
- [x] SCF framework with multiple mixing schemes
- [x] Automatic differentiation integration
- [x] Example calculations and documentation

### 🚧 In Progress  
- [ ] Validation against BOPfox reference calculations
- [ ] Performance optimization and GPU benchmarking
- [ ] Extended parameter sets for different elements
- [ ] Integration with experimental data fitting workflows

### 🎯 Future Plans
- [ ] Multi-element BOP parameterizations
- [ ] Neural network potential integration  
- [ ] Large-scale molecular dynamics applications
- [ ] Quantum mechanical property calculations

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bop_jax,
  title={BOP-JAX: Bond Order Potential with JAX and Automatic Differentiation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/bop-jax}
}
```

## References

1. **Original BOP Method**: Pettifor, D. G., et al. "Bond-order potentials: theory and implementation" *Phys. Rev. B* **59**, 8487 (1999)

2. **JAX**: Bradbury, J., et al. "JAX: composable transformations of Python+NumPy programs" (2018)

3. **JAX-MD**: Schoenholz, S. S. & Cubuk, E. D. "JAX, M.D.: A Framework for Differentiable Physics" *Advances in Neural Information Processing Systems* **33** (2020)

4. **BOPfox**: Original Fortran implementation by various authors

---

**Note**: This implementation focuses on the core BOP algorithm with automatic differentiation. For production calculations, parameter validation and comparison with experimental/DFT data is essential.