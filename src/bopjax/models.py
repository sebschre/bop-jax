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


# fitted parameters of the sp carbon model with atomic shifts (examples/fit_carbon.py,
# docs/carbon.md)
CARBON_FIT = {
    "e_s": -8.08885, "sss": -2.91351, "sps": 7.17174, "pps": 6.60702, "ppp": -0.77806,
    "decay": 4.04441, "pair": 19.41433, "pair_decay": 4.85776, "embed": 4.80143,
    "density_decay": 0.48509, "r0": 1.54,
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
                     repulsion=pw.V0, atomic_shifts=True, bandwidth_guard=True)


# --- iron-carbon: the d iron model and the sp carbon model with d-sp cross terms ---


def _taper(r, rc, dc):
    """1 below rc - dc, 0 above rc, half cosine in between; rc and dc may be
    traced (the same curve as ``hamiltonian.cosine_cutoff``)."""
    x = jnp.clip((r - rc + dc) / dc, 0.0, 1.0)
    return jnp.where(r < rc, 0.5 * (1.0 + jnp.cos(jnp.pi * x)), 0.0)


def tapered_exp(p, r):
    """``p[0] * exp(-p[1] (r / p[2] - 1))`` switched off between ``p[3] - p[4]``
    and ``p[3]``: an exponential with a cutoff of its own, so that each pair
    of species can have its own range under one global cutoff."""
    return p[0] * jnp.exp(-p[1] * (r / p[2] - 1.0)) * _taper(r, p[3], p[4])


# repulsion and embedding density with per-pair cutoffs (rows as tapered_exp)
TAPERED_V0 = pw.Repulsion(pair=tapered_exp, density=tapered_exp, embed=pw.V0.embed)


# fitted cross parameters of the iron-carbon model (examples/fit_iron_carbon.py,
# docs/iron-carbon.md)
IRON_CARBON_FIT = {
    "e_c": 2.15, "sds": -0.5, "pds": -0.8, "pdp": 0.4, "decay": 3.0,
    "pair": 1.0, "pair_decay": 6.0, "rho_fe": 1.0, "rho_c": 1.0, "rho_decay": 3.0,
    "r0": 1.95, "rcut": 3.0, "dcut": 0.5,
}

FE, C = 0, 1  # species indices of the iron-carbon model


def iron_carbon(iron_values=None, carbon_values=None, **values):
    """Iron (species 0, the d model ``IRON_FIT``) and carbon (species 1, the
    sp model ``CARBON_FIT``) with d-sp bonds between them.

    Use with ``BOPConfig(**IRON_CARBON_CONFIG, spin=...)`` and
    ``radial=models.tapered_exp``.  Fe-Fe and C-C interactions are exactly
    those of ``iron()`` and ``carbon()``, with their own cutoffs;
    ``iron_values`` and ``carbon_values`` override their parameters.
    Keyword arguments override entries of ``IRON_CARBON_FIT``, the Fe-C terms:

      e_c                 shift of the carbon levels (2s and 2p together)
                          relative to the iron d level (eV)
      sds, pds, pdp       bond integrals C(2s, 2p) - Fe(3d) at r0 (eV), each
                          ``beta * exp(-decay (r / r0 - 1))``
      pair, pair_decay    Fe-C pair term ``pair * exp(-pair_decay (r / r0 - 1))``
      rho_fe, rho_c, rho_decay
                          embedding density that a C neighbour adds on Fe
                          (``rho_fe``) and an Fe neighbour adds on C
                          (``rho_c``), both ``* exp(-rho_decay (r / r0 - 1))``
      rcut, dcut          range of every Fe-C term (taper from rcut - dcut)
    """
    fe = dict(IRON_FIT, **(iron_values or {}))
    cv = dict(CARBON_FIT, **(carbon_values or {}))
    v = dict(IRON_CARBON_FIT, **values)
    fe_cut = (IRON_CONFIG["rcut"], IRON_CONFIG["dcut"])
    c_cut = (CARBON_CONFIG["rcut"], CARBON_CONFIG["dcut"])
    x_cut = (v["rcut"], v["dcut"])

    def row(amp, decay, r0, cut):
        return [amp, decay, r0, cut[0], cut[1]]

    zero = row(0.0, 1.0, 1.0, (1.0, 0.5))

    def table(entries):
        """(2, 2, 5) parameter array from {(a, b): row}."""
        t = [[zero, zero], [zero, zero]]
        for (a, b), r in entries.items():
            t[a][b] = r
        return jnp.array(t)

    bond = {}
    for c in ("dds", "ddp", "ddd"):
        bond[c] = table({(FE, FE): row(fe[c], fe["decay"], fe["r0"], fe_cut)})
    for c in ("sss", "sps", "pps", "ppp"):
        bond[c] = table({(C, C): row(cv[c], cv["decay"], cv["r0"], c_cut)})
    # mixed channels: entry [a, b] has the lower shell (s or p) on a, d on b
    for c in ("sds", "pds", "pdp"):
        bond[c] = table({(C, FE): row(v[c], v["decay"], v["r0"], x_cut)})
    e_c = v["e_c"]
    return {
        # shells (s, p, d); iron has only d, carbon only s and p
        "onsite": jnp.array([[0.0, 0.0, 0.0], [cv["e_s"] + e_c, e_c, 0.0]]),
        "n0": jnp.array([[0.0, 0.0, fe["n_d"]], [2.0, 2.0, 0.0]]),
        "hubbard": jnp.array([8.0, 8.0]),
        "stoner": jnp.array([[0.0, 0.0, fe["stoner"]], [0.0, 0.0, 0.0]]),
        "bond": bond,
        "repulsion": {
            "pair": table({
                (FE, FE): row(fe["pair"], fe["pair_decay"], fe["r0"], fe_cut),
                (C, C): row(cv["pair"], cv["pair_decay"], cv["r0"], c_cut),
                (FE, C): row(v["pair"], v["pair_decay"], v["r0"], x_cut),
                (C, FE): row(v["pair"], v["pair_decay"], v["r0"], x_cut)}),
            # density[a, b]: what a neighbour of species b adds on an atom of a
            "density": table({
                (FE, FE): row(1.0, fe["density_decay"], fe["r0"], fe_cut),
                (C, C): row(1.0, cv["density_decay"], cv["r0"], c_cut),
                (FE, C): row(v["rho_fe"], v["rho_decay"], v["r0"], x_cut),
                (C, FE): row(v["rho_c"], v["rho_decay"], v["r0"], x_cut)}),
            "embed": jnp.array([[-fe["embed"], 0.5], [-cv["embed"], 0.5]]),
        },
    }


def _iron_carbon_config():
    from .bop import SpeciesSettings

    fe = SpeciesSettings(basis=("d",), nmom=IRON_CONFIG["nmom"],
                         bandwidth=IRON_CONFIG["bandwidth"], kernel="none")
    c = SpeciesSettings(basis=CARBON_CONFIG["basis"], nmom=CARBON_CONFIG["nmom"],
                        nexp=CARBON_CONFIG["nexp"], bandwidth=CARBON_CONFIG["bandwidth"],
                        kernel=CARBON_CONFIG["kernel"],
                        bandwidth_guard=CARBON_CONFIG["bandwidth_guard"])
    rmax = max(IRON_CONFIG["rcut"], CARBON_CONFIG["rcut"], 3.0)
    # the global cutoff is a hard step at the longest range; every pair is
    # tapered inside its own radial function
    return dict(basis=("s", "p", "d"), species=(fe, c),
                nmom=max(IRON_CONFIG["nmom"], CARBON_CONFIG["nmom"]),
                rcut=rmax, dcut=0.0, rcut2=rmax, dcut2=0.0, repulsion=TAPERED_V0,
                atomic_shifts=True)


# BOPConfig settings of the iron-carbon model (pass radial=models.tapered_exp);
# Fe-C terms must not reach beyond rcut
IRON_CARBON_CONFIG = _iron_carbon_config()
