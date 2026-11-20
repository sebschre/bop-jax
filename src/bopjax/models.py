"""Ready-made parameter sets."""

from __future__ import annotations

import jax.numpy as jnp

from . import pairwise as pw
from .hamiltonian import CHANNELS


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


# defaults of the s and p parameters of the iron model with an sp shell
# (``iron(basis=...)``); IRON_SD_FIT overrides them for basis ("s", "d")
IRON_SP_DEFAULTS = {
    "e_s": 0.5, "e_p": 7.0, "n_el": 8.0, "decay_sp": 3.0,
    "sss": -0.5, "sps": 1.0, "pps": 1.5, "ppp": -0.3, "sds": -0.2, "pds": -0.3, "pdp": 0.15,
}


def iron(basis=("d",), **values):
    """Magnetic iron: collinear Stoner magnetism on the d shell, exponential
    bond integrals, pair repulsion and square-root embedding.

    Use with ``BOPConfig(**IRON_CONFIG, basis=basis, spin=...)`` and
    ``radial=hamiltonian.v0_bond``.  Keyword arguments override entries of
    ``IRON_FIT`` (d basis), ``IRON_SD_FIT`` (s and d basis) or
    ``IRON_SP_DEFAULTS``:

      dds, ddp, ddd   bond integrals dd(sigma, pi, delta) at r0 (eV); each is
                      ``dd_tau * exp(-decay (r / r0 - 1))`` (the v0 form with
                      its short-range factor switched off)
      n_d             d electrons per atom (d basis only)
      stoner          Stoner integral I of the d shell (eV); E_mag = -I m^2 / 4
      pair, pair_decay           pair term ``pair * exp(-pair_decay (r / r0 - 1))``
      embed, density_decay       embedding ``-embed * sqrt(rho)``,
                                 ``rho = sum_j exp(-density_decay (r_ij / r0 - 1))``

    With an s and/or p shell in ``basis`` (onsite d level 0):

      e_s, e_p        onsite levels of the s and p shells (eV)
      n_el            valence electrons per atom (8 for 3d6 4s2), shared
                      between the shells by the common Fermi level
      sss, sps, pps, ppp, sds, pds, pdp
                      bond integrals at r0 of the channels with an s or p
                      shell, all decaying as ``exp(-decay_sp (r / r0 - 1))``
    """
    basis = tuple(basis)
    if "d" not in basis:
        raise ValueError("the iron model needs a d shell")
    v = dict(IRON_FIT)
    if basis != ("d",):
        v.update(IRON_SP_DEFAULTS)
        if basis == ("s", "d"):
            v.update(IRON_SD_FIT)
    v.update(values)
    r0 = v["r0"]

    def channel(prefactor, decay):
        return jnp.array([[[prefactor, decay, 100.0, 1.0, r0]]])

    bond = {c: channel(v[c], v["decay"]) for c in ("dds", "ddp", "ddd")}
    if basis == ("d",):
        onsite, n0 = [0.0], [v["n_d"]]
    else:
        for i, la in enumerate(basis):
            for lb in basis[i:]:
                if (la, lb) != ("d", "d"):
                    for c in CHANNELS[(la, lb)]:
                        bond[c] = channel(v[c], v["decay_sp"])
        level = {"s": v["e_s"], "p": v["e_p"], "d": 0.0}
        # free-atom reference 3d(n-1) 4s1: only fixes the zero of the energy
        ref = {"s": 1.0, "p": 0.0, "d": v["n_el"] - 1.0}
        if "s" not in basis:
            ref = {"p": 1.0, "d": v["n_el"] - 1.0}
        onsite, n0 = [level[sh] for sh in basis], [ref[sh] for sh in basis]
    stoner = [v["stoner"] if sh == "d" else 0.0 for sh in basis]
    return {
        "onsite": jnp.array([onsite]),
        "n0": jnp.array([n0]),
        "hubbard": jnp.array([8.0]),
        "stoner": jnp.array([stoner]),
        "bond": bond,
        "repulsion": {
            "pair": jnp.array([[[v["pair"], v["pair_decay"], r0]]]),
            "density": jnp.array([[[1.0, v["density_decay"], r0]]]),
            "embed": jnp.array([[-v["embed"], 0.5]]),
        },
    }


# fitted parameters of the sd iron model (examples/fit_iron.py --basis sd)
IRON_SD_FIT = {
    "dds": -1.12595, "ddp": 0.5211, "ddd": -0.04472, "decay": 4.6483, "stoner": 1.1603,
    "pair": 0.62759, "pair_decay": 11.17315, "embed": 1.39903, "density_decay": 2.16347,
    "e_s": 0.74816, "decay_sp": 2.85527, "sss": -0.29816, "sds": -0.26058,
}

# BOPConfig settings of the iron model (pass radial=hamiltonian.v0_bond)
IRON_CONFIG = dict(nmom=9, bandwidth="averaged", rcut=3.4, dcut=0.4, rcut2=3.4, dcut2=0.4,
                   repulsion=pw.V0)


# fitted parameters of the sp carbon model (examples/fit_carbon.py, docs/carbon.md)
CARBON_FIT = {
    "e_s": -7.8627, "sss": -2.87213, "sps": 7.17265, "pps": 6.62997, "ppp": -0.78207,
    "decay": 4.03771, "pair": 19.44275, "pair_decay": 4.85353, "embed": 4.86397,
    "density_decay": 0.50549, "r0": 1.54,
}


def carbon(**values):
    """Non-magnetic sp-valent carbon: s and p shells, exponential bond
    integrals, pair repulsion and square-root embedding.

    Use with ``BOPConfig(**CARBON_CONFIG)`` and ``radial=hamiltonian.v0_bond``.
    Keyword arguments override entries of ``CARBON_FIT``:

      e_s                 onsite level of the 2s shell relative to 2p (eV)
      sss, sps, pps, ppp  bond integrals at r0 (eV), each
                          ``beta * exp(-decay (r / r0 - 1))``
      pair, pair_decay    pair term ``pair * exp(-pair_decay (r / r0 - 1))``
      embed, density_decay  embedding ``-embed * sqrt(rho)``,
                          ``rho = sum_j exp(-density_decay (r_ij / r0 - 1))``

    Four valence electrons share one Fermi level; the free-atom reference
    2s2 2p2 only fixes the zero of the energy.
    """
    v = dict(CARBON_FIT)
    v.update(values)
    r0 = v["r0"]

    def channel(prefactor):
        return jnp.array([[[prefactor, v["decay"], 100.0, 1.0, r0]]])

    return {
        "onsite": jnp.array([[v["e_s"], 0.0]]),
        "n0": jnp.array([[2.0, 2.0]]),
        "hubbard": jnp.array([8.0]),
        "bond": {c: channel(v[c]) for c in ("sss", "sps", "pps", "ppp")},
        "repulsion": {
            "pair": jnp.array([[[v["pair"], v["pair_decay"], r0]]]),
            "density": jnp.array([[[1.0, v["density_decay"], r0]]]),
            "embed": jnp.array([[-v["embed"], 0.5]]),
        },
    }


# BOPConfig settings of the carbon model (pass radial=hamiltonian.v0_bond).  The
# Jackson kernel keeps the density of states positive: without it the Fermi
# level wanders in the gap of diamond and the energy is not smooth.
CARBON_CONFIG = dict(nmom=10, nexp=20, kernel="jackson", bandwidth="averaged",
                     basis=("s", "p"), rcut=2.2, dcut=0.3, rcut2=2.2, dcut2=0.3,
                     repulsion=pw.V0)


# Fe part of the magnetic Fe-Co BOP of Egorov, Subramanyam, Yuan, Drautz and
# Hammerschmidt, arXiv:2208.12973 (Supplemental Material, Table 1): d-valent,
# bond integrals as sums of exponentials, square-root embedding of Gaussian
# densities, pair, core and environment-dependent Yukawa repulsion
IRON_EGOROV = {
    "N": 6.98, "stoner": 0.751,
    "dds": [(-20.10578, 1.68256, 0.83602), (-3.09201, 0.38260, 2.44521),
            (3.20957, 0.29985, 6.02389)],
    "ddp": [(29.59495, 1.47219, 1.27945), (13.96316, 0.94631, 1.76772)],
    "ddd": [(-10.14975, 1.90305, 1.19157), (-42.05214, 2.58244, 1.07279)],
    "embed": (4.36232, 0.26944, 0.03008),
    "pair": (155.20896, 1.28421, 1.60531), "core": (1.5, 100.0),
    "yukawa": {"B": 1.24913, "r_core": 1.06402, "lambda0": 1.99623, "C": 113.85788,
               "upsilon": 1.78242, "m": 1.18487},
}

# BOPConfig settings of the Egorov Fe model (pass radial=pairwise.exp_sum): 9 moments,
# Jackson kernel with 200 expansion coefficients and a constant terminator;
# cutoffs 4.1 (1.0) A for the bond integrals, 6.0 (0.5) A for embedding and repulsion.
# The paper does not name the bandwidth estimate; "findeminemax" reproduces its bcc Fe
# values (V 11.45 A^3, 2.875 muB, B 184, C11 272, C12 140, C44 85 GPa; paper 11.45,
# 2.88, 185, 269, 147, 90) and keeps the 200 Chebyshev moments bounded, which the
# "averaged" estimate does not (split-off chain states outside the band grow without
# bound in the expansion).
IRON_EGOROV_CONFIG = dict(nmom=9, nexp=200, kernel="jackson", terminator="constantabn",
                          bandwidth="findeminemax", rcut=4.1, dcut=1.0, rcut2=6.0, dcut2=0.5,
                          repulsion=pw.EGOROV, yukawa=True)


def iron_egorov(hubbard=8.0, **values):
    """Parameters of the Egorov et al. Fe model (``IRON_EGOROV``, overridable by
    keyword).  Use with ``BOPConfig(**IRON_EGOROV_CONFIG, spin=...)`` and
    ``radial=pairwise.exp_sum``.  ``hubbard`` (not part of the published model)
    keeps atoms close to charge neutrality in inhomogeneous structures."""
    v = {k: val for k, val in IRON_EGOROV.items()}
    v.update(values)

    def channel(terms):
        flat = [x for t in terms for x in t]
        flat += [0.0, 1.0, 1.0] * (3 - len(terms))  # pad to three terms
        return jnp.array([[flat]])

    y = v["yukawa"]
    return {
        "onsite": jnp.array([[0.0]]),
        "n0": jnp.array([[v["N"]]]),
        "hubbard": jnp.array([hubbard]),
        "stoner": jnp.array([[v["stoner"]]]),
        "bond": {c: channel(v[c]) for c in ("dds", "ddp", "ddd")},
        "repulsion": {
            "pair": jnp.array([[[*v["pair"], *v["core"]]]]),
            # E_emb = -p1 sqrt(sum_j exp(-p2 (r - p3)^2)): with p1 inside the square
            # root, as Eqs. 6-7 of the paper read, bcc Fe binds by -4.9 eV/atom at
            # a = 3.05 A; with p1 in front it binds by -8.30 eV/atom at 2.83 A, the
            # DFT energy the model was fitted to (-8.31 eV/atom, a = 2.84 A)
            "density": jnp.array([[[1.0, v["embed"][1], v["embed"][2]]]]),
            "embed": jnp.array([[-v["embed"][0], 0.5]]),
        },
        "yukawa": {"B": jnp.array([[y["B"]]]), "r_core": jnp.array([[y["r_core"]]]),
                   "C": jnp.array([[y["C"]]]), "upsilon": jnp.array([[y["upsilon"]]]),
                   "lambda0": jnp.array([y["lambda0"]]), "m": jnp.array([y["m"]])},
    }
