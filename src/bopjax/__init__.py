"""Analytic bond-order potentials in JAX."""

import jax

jax.config.update("jax_enable_x64", True)  # moments -> recursion needs doubles

from .bop import BOPConfig, make_energy_fn, moments  # noqa: E402
from .geometry import Topology, build_topology  # noqa: E402
from .scf import SCFResult, make_forces, make_scf  # noqa: E402

__all__ = [
    "BOPConfig",
    "SCFResult",
    "Topology",
    "build_topology",
    "make_energy_fn",
    "make_forces",
    "make_scf",
    "moments",
]
