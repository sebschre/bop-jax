"""
Compatibility utilities for different JAX versions.

This module handles version differences in JAX and JAX-MD to ensure
our BOP implementation works across different environments.
"""

import jax
import jax.numpy as jnp

# Handle tree_map location changes in different JAX versions
try:
    from jax import tree_map
except ImportError:
    try:
        from jax.tree_util import tree_map
    except ImportError:
        # For very old versions, provide a basic implementation
        def tree_map(f, tree):
            if isinstance(tree, dict):
                return {k: tree_map(f, v) for k, v in tree.items()}
            elif isinstance(tree, (list, tuple)):
                return type(tree)(tree_map(f, item) for item in tree)
            else:
                return f(tree)

# Handle other potential compatibility issues
def check_jax_version():
    """Check if JAX version is compatible."""
    jax_version = jax.__version__
    major, minor = map(int, jax_version.split('.')[:2])
    
    if major == 0 and minor < 4:
        print(f"Warning: JAX version {jax_version} is older than recommended (0.4.x)")
        print("Some features may not work correctly.")
        return False
    
    return True

def safe_jit(func, **kwargs):
    """Safely apply JIT compilation with fallback."""
    try:
        return jax.jit(func, **kwargs)
    except Exception as e:
        print(f"Warning: JIT compilation failed ({e}), falling back to regular function")
        return func

# Export compatibility functions
__all__ = ['tree_map', 'check_jax_version', 'safe_jit']