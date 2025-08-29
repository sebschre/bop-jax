# Installation Guide for BOP-JAX

BOP-JAX now uses **Poetry** for dependency management, which automatically resolves version conflicts and provides reproducible environments.

## Quick Installation (Poetry - Recommended)

```bash
# Step 1: Install Poetry (if you don't have it)
curl -sSL https://install.python-poetry.org | python3 -

# Step 2: Clone and install BOP-JAX
poetry install

# Step 3: Activate the Poetry environment
poetry shell

# Step 4: Test the installation
poetry run bop-test minimal
```

## Alternative: Manual Pip Installation

If you prefer pip (not recommended due to version conflicts):

```bash
# First, activate your virtual environment
source venv/bin/activate  # or however you activate your venv

# Uninstall existing versions to avoid conflicts
pip uninstall jax jaxlib jax-md -y

# Install specific compatible versions
pip install jax==0.4.20 jaxlib==0.4.20
pip install jax-md==0.2.8

# Install other requirements
pip install numpy scipy matplotlib
```

## Poetry Installation Methods

### Option A: Basic Installation
```bash
# Install core dependencies only
poetry install --without dev,test
```

### Option B: Development Installation  
```bash
# Install with development tools (testing, linting, etc.)
poetry install --with dev,test
```

### Option C: With Optional Dependencies
```bash
# Install with visualization support
poetry install --extras viz

# Install all optional dependencies  
poetry install --extras all
```

### Option D: GPU Support
```bash
# First install CPU version
poetry install

# Then manually add GPU support (Poetry doesn't handle jax[cuda] syntax well)
poetry run pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Testing Your Installation

### Quick Test
```bash
# Test core functionality (works without JAX-MD)
poetry run bop-test minimal
```

### Full Test
```bash
# Test complete BOP calculation with JAX-MD
poetry run bop-test example
```

### Run Test Suite
```bash
# Run comprehensive tests
poetry run bop-test test
# Or use pytest directly
poetry run pytest
```

## Development Workflow

### Setup Development Environment
```bash
# Install with all development tools
poetry install --with dev,test

# Install pre-commit hooks (optional)
poetry run pre-commit install

# Activate Poetry shell
poetry shell
```

### Common Commands
```bash
# Run tests
poetry run pytest

# Format code
poetry run black bop_jax/

# Type checking
poetry run mypy bop_jax/

# Run example scripts
poetry run python examples/simple_bop_calculation.py
```

## Version Compatibility Matrix

| JAX Version | JAX-MD Version | Status |
|-------------|---------------|---------|
| 0.4.20      | 0.2.8         | ✅ Recommended |
| 0.4.25+     | 0.2.8         | ⚠️ May have issues |
| 0.3.x       | 0.2.8         | ❌ Not compatible |

## Troubleshooting

### Issue: `tree_map` Import Error
**Cause**: JAX version mismatch  
**Solution**: Install exact versions as shown above

### Issue: `jax-md` Installation Fails
**Solution**: Try installing without dependencies first:
```bash
pip install jax-md==0.2.8 --no-deps
```

### Issue: JAX CPU vs GPU Version
**For CPU only** (recommended for testing):
```bash
pip install jax==0.4.20 jaxlib==0.4.20
```

**For GPU support** (if you have CUDA):
```bash
pip install jax[cuda]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue: M1/M2 Mac Compatibility
**For Apple Silicon Macs**:
```bash
pip install jax==0.4.20 jaxlib==0.4.20
# JAX should work fine on Apple Silicon
```

## Verification Steps

After installation, verify everything works:

1. **Test JAX**:
   ```python
   import jax
   import jax.numpy as jnp
   print(f"JAX version: {jax.__version__}")
   ```

2. **Test JAX-MD**:
   ```python
   from jax_md import space
   print("JAX-MD imported successfully!")
   ```

3. **Test BOP-JAX**:
   ```bash
   python examples/minimal_test.py
   ```

## Expected Output

When `minimal_test.py` runs successfully, you should see:
```
✓ JAX 0.4.20 imported successfully
✓ NumPy 1.24.3 imported successfully
...
✓ JAX-MD imported successfully
✓ Standalone BOP calculation completed successfully!
```

## Next Steps

Once installation is successful:

1. Run the full example: `python examples/simple_bop_calculation.py`
2. Explore the test suite: `python tests/test_bop_components.py`  
3. Start using BOP-JAX in your own projects!

## Getting Help

If you still encounter issues:

1. Check that you're using the exact versions specified
2. Try creating a fresh virtual environment  
3. Run `python examples/minimal_test.py` to isolate the issue
4. Check the error messages carefully - they often indicate exactly what's missing

The core BOP implementation doesn't strictly require JAX-MD and can work with just JAX + NumPy if needed.