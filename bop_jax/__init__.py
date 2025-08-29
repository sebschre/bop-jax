"""
Bond Order Potential (BOP) implementation using JAX and JAX-MD.

This package provides a JAX-based implementation of the Bond Order Potential method,
leveraging automatic differentiation and integration with the JAX-MD molecular dynamics library.
"""

__version__ = "0.1.0"

# Import core modules first (no JAX-MD dependencies)
from . import moments 
from . import chebyshev
from . import scf

# Import modules that may require JAX-MD (optional imports handled internally)
from . import hamiltonian
from . import energy

__all__ = ["moments", "chebyshev", "scf", "hamiltonian", "energy"]