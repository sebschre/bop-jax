# BOP-JAX Quick Start Guide

## 🚀 **Get Started in 3 Steps**

### Step 1: Install Poetry & BOP-JAX
```bash
# Install Poetry (if you don't have it)
curl -sSL https://install.python-poetry.org | python3 -

# Navigate to project and install
cd /Users/sebastian/dev/bop_autodiff
poetry install

# Activate Poetry environment
poetry shell

# Optional: Add GPU support (if you have CUDA)
poetry run pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Step 2: Test Core BOP Algorithms
```bash
# Test core BOP algorithms (no JAX-MD required)
poetry run bop-test minimal
```

### Step 3: Run Full Example
```bash
# Test complete BOP calculation with JAX-MD integration
poetry run bop-test example
```

## 📋 **What You Get**

✅ **Automatic Forces**: No manual derivative calculation needed  
✅ **Advanced SCF**: Newton-Raphson with automatic Hessians  
✅ **JAX-MD Integration**: Optimized neighbor lists and MD simulators  
✅ **GPU Ready**: Seamless acceleration for larger systems  
✅ **Clean Dependencies**: Poetry handles all version conflicts automatically

## 🛠 **Development Commands**

```bash
# Install with development tools
poetry install --with dev,test

# Run tests
poetry run pytest

# Format code  
poetry run black bop_jax/

# Type checking
poetry run mypy bop_jax/

# Run specific examples
poetry run python examples/simple_bop_calculation.py
poetry run python examples/minimal_test.py
```

## 🐛 **Troubleshooting**

### If Poetry install fails:
```bash
# Update Poetry
poetry self update

# Clear cache and retry
poetry cache clear --all pypi
poetry install
```

### If JAX-MD import fails:
The `poetry install` should handle this automatically, but if issues persist:
```bash
# Poetry will install compatible versions automatically
poetry show  # See installed versions
```

### If tests fail:
```bash
# Run minimal test first (doesn't require JAX-MD)
poetry run bop-test minimal

# Check installed packages
poetry run bop-test version
```

## 📁 **Project Structure**

```
bop-jax/
├── pyproject.toml          # Poetry configuration
├── bop_jax/                # Main package
│   ├── energy.py           # JAX-MD integration
│   ├── hamiltonian.py      # Tight-binding matrices
│   ├── moments.py          # Core BOP algorithm  
│   ├── chebyshev.py        # LDOS expansion
│   └── scf.py             # Self-consistent field
├── examples/               # Usage examples
├── tests/                 # Test suite
└── README.md              # Full documentation
```

## 🎯 **Next Steps**

1. **Explore Examples**: Check `examples/` for BOP calculations
2. **Run Tests**: Use `poetry run pytest` for validation
3. **Customize Parameters**: Modify BOP parameters in your calculations
4. **Scale Up**: Try larger systems with GPU acceleration
5. **Integrate**: Use with your own MD workflows

## 📚 **Key Features Demo**

### Automatic Forces
```python
import jax
from bop_jax.energy import bop_energy_fn

# Create energy function
energy_fn = bop_energy_fn(...)

# Forces computed automatically!
forces = -jax.grad(energy_fn)(positions)
```

### Advanced SCF
```python
from bop_jax.scf import newton_raphson_scf

# Newton-Raphson with automatic Hessians
scf_result = newton_raphson_scf(residual_fn, initial_onsites)
```

### JAX-MD Integration
```python
from bop_jax.energy import create_bop_simulator

# Complete MD setup in one line
energy_fn, force_fn, simulator, neighbor_fn = create_bop_simulator(
    R_init, box_size, dt=0.001, temperature=300.0
)
```

## 🎉 **Success!**

You now have a modern, automatically differentiable Bond Order Potential implementation that:

- **Eliminates manual force derivation** 
- **Enables advanced SCF methods** with 2nd-order convergence
- **Integrates seamlessly** with the JAX-MD ecosystem
- **Scales to GPU/TPU** for larger systems
- **Provides clean, maintainable code** in modern Python

Enjoy exploring the power of automatic differentiation for materials simulations!