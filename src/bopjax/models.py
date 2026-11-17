"""Ready-made parameter sets."""

from __future__ import annotations

import jax.numpy as jnp

from . import pairwise as pw


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


# fitted parameters of the iron model (examples/fit_iron.py, docs/iron.md)
IRON_FIT = {
    "dds": -1.14744, "ddp": 0.51229, "ddd": -0.03851, "decay": 4.38348, "n_d": 6.88178,
    "stoner": 1.16487, "pair": 0.64199, "pair_decay": 9.98837, "embed": 1.51835,
    "density_decay": 2.79649, "r0": 2.48,
}


def iron(**values):
    """Magnetic d-band iron: collinear Stoner magnetism, exponential bond
    integrals, pair repulsion and square-root embedding.

    Use with ``BOPConfig(**IRON_CONFIG, spin=...)`` and
    ``radial=hamiltonian.v0_bond``.  Keyword arguments override entries of
    ``IRON_FIT``:

      dds, ddp, ddd   bond integrals dd(sigma, pi, delta) at r0 (eV); each is
                      ``dd_tau * exp(-decay (r / r0 - 1))`` (the v0 form with
                      its short-range factor switched off)
      n_d             d electrons per atom
      stoner          Stoner integral I (eV); E_mag = -I m^2 / 4
      pair, pair_decay           pair term ``pair * exp(-pair_decay (r / r0 - 1))``
      embed, density_decay       embedding ``-embed * sqrt(rho)``,
                                 ``rho = sum_j exp(-density_decay (r_ij / r0 - 1))``
    """
    v = dict(IRON_FIT, **values)
    r0, decay = v["r0"], v["decay"]

    def channel(prefactor):
        return jnp.array([[[prefactor, decay, 100.0, 1.0, r0]]])

    return {
        "onsite": jnp.zeros((1, 1)),
        "n0": jnp.array([[v["n_d"]]]),
        "hubbard": jnp.array([8.0]),
        "stoner": jnp.array([[v["stoner"]]]),
        "bond": {"dds": channel(v["dds"]), "ddp": channel(v["ddp"]), "ddd": channel(v["ddd"])},
        "repulsion": {
            "pair": jnp.array([[[v["pair"], v["pair_decay"], r0]]]),
            "density": jnp.array([[[1.0, v["density_decay"], r0]]]),
            "embed": jnp.array([[-v["embed"], 0.5]]),
        },
    }


# BOPConfig settings of the iron model (pass radial=hamiltonian.v0_bond)
IRON_CONFIG = dict(nmom=9, bandwidth="averaged", rcut=3.4, dcut=0.4, rcut2=3.4, dcut2=0.4,
                   repulsion=pw.V0)
