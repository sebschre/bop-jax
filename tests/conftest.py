"""Release compiled functions between test modules.

Every jitted function keeps its compiled executable; over the whole suite they
add up to several GB, which can exhaust memory on small machines (LLVM then
aborts the process).  Clearing JAX's caches after each module keeps the peak
at that of the largest module.
"""

import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_functions():
    yield
    jax.clear_caches()
    gc.collect()
