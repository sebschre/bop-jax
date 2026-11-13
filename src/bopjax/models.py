"""Ready-made parameter sets."""

from __future__ import annotations

import jax.numpy as jnp


def canonical_d(n_electrons=4.8, onsite=0.0, hubbard=0.0):
    """Canonical d-band model, dd(sigma:pi:delta) = -6:4:-1 with (r0/r)^5.

    This is the "canonicaltb" W_LIM_I entry shipped with BOPfox (BOP0 of
    Mrovec et al., PRB 75, 104119).  Arguments may be scalars (one species) or
    sequences (one value per species; all pairs then share the bond integrals).
    """
    n_el = jnp.atleast_1d(jnp.asarray(n_electrons, dtype=float))
    nsp = n_el.shape[0]
    on = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(onsite, dtype=float)), (nsp,))
    hub = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(hubbard, dtype=float)), (nsp,))

    def channel(prefactor):
        return jnp.broadcast_to(jnp.array([prefactor, 2.081, 5.0]), (nsp, nsp, 3))

    return {
        "onsite": on[:, None],
        "n0": n_el[:, None],
        "hubbard": hub,
        "bond": {"dds": channel(-1.458), "ddp": channel(0.972), "ddd": channel(-0.243)},
    }


def tungsten_v0():
    """d-band tungsten with pair repulsion and embedding (BOPfox "v0" W_LIM_I,
    after Lim).  Use with ``radial=bopjax.hamiltonian.v0_bond`` and
    ``BOPConfig(repulsion=bopjax.pairwise.V0, rcut2=...)``."""

    def channel(prefactor):
        return jnp.array([[[prefactor, 4.2, 100.0, 1.0, 2.73664]]])

    return {
        "onsite": jnp.zeros((1, 1)),
        "n0": jnp.array([[4.8]]),
        "hubbard": jnp.array([8.0]),
        "bond": {"dds": channel(-1.458), "ddp": channel(0.972), "ddd": channel(-0.243)},
        "repulsion": {
            "pair": jnp.array([[[7.94383, 6.43110, 2.73664]]]),
            "density": jnp.array([[[3.39726, 5.66484, 2.73664]]]),
            "embed": jnp.array([[-1.0, 1.0]]),
        },
    }
