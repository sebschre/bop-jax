"""Ready-made parameter sets."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
    "e_s": -11.89778, "sss": -3.19329, "sps": 8.999, "pps": 8.08017, "ppp": -0.2274,
    "decay": 4.29598, "pair": 23.65792, "pair_decay": 5.11383, "embed": 4.44614,
    "density_decay": 0.00046, "r0": 1.54,
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



# --- iron-carbon: the d iron model and the sp carbon model with d-sp cross terms ---


def _taper(r, rc, dc):
    """1 below rc - dc, 0 above rc, half cosine in between; rc and dc may be
    traced (the same curve as ``hamiltonian.cosine_cutoff``)."""
    x = jnp.clip((r - rc + dc) / dc, 0.0, 1.0)
    return jnp.where(r < rc, 0.5 * (1.0 + jnp.cos(jnp.pi * x)), 0.0)


def tapered_exp_sum(p, r):
    """Sum of three stretched exponentials (``pairwise.exp_sum``, p[:9]) switched
    off between ``p[9] - p[10]`` and ``p[9]``: bond integrals of every pair of
    species in one form, each with its own range."""
    return pw.exp_sum(p[:9], r) * _taper(r, p[9], p[10])


def _tapered_pair(p, r):
    """Egorov pair and core repulsion (p[:5]) with a taper (p[5], p[6])."""
    return pw.EGOROV.pair(p[:5], r) * _taper(r, p[5], p[6])


def _tapered_density(p, r):
    """Gaussian p0 exp(-p1 (r - p2)^2) plus exponential p3 exp(-p4 r), with a
    taper (p[5], p[6]): Egorov's iron density and the carbon density in one form."""
    return ((p[0] * jnp.exp(-p[1] * (r - p[2]) ** 2) + p[3] * jnp.exp(-p[4] * r))
            * _taper(r, p[5], p[6]))


# repulsion of the iron-carbon model: pair, core and embedding per pair of species
FECARBON_REPULSION = pw.Repulsion(pair=_tapered_pair, density=_tapered_density,
                                  embed=pw.V0.embed)

NO_TAPER = (100.0, 1.0)  # leaves a pair to the global cosine cutoff


# fitted Fe-C cross terms (examples/fit_iron_carbon.py on bopfox-rs,
# docs/iron-carbon.md): every atom of the Fe16C cells relaxed on exact forces,
# each cell on its lowest-energy magnetic state, cementite relaxed too, the
# migration barrier from the saddle in the plane between two octahedral
# sites (it agrees with NEB); Gaussian Fe-C embedding densities and a strong
# short-range core.
IRON_CARBON_FIT = {
    "e_c": 1.30483, "sds": -0.25827, "pds": -0.47268, "pdp": 0.62705, "decay": 4.34798,
    "pair": 1.86696, "pair_decay": 4.28099, "rho_fe": 0.32776, "rho_c": 0.93649,
    "rho_decay": 3.0, "r0": 1.95, "rcut": 3.0, "dcut": 0.5, "hubbard_c": 3.32112,
    "decay_pi": 4.15288, "core": 125.01467, "core_r": 1.70398,
    "rho_gauss_fe": 0.10332, "rho_gauss_c": 0.29459, "rho_gauss_r": 2.00367,
    "rho_gauss_w": 0.2038,
}

FE, C = 0, 1  # species indices of the iron-carbon model


def _exp_term(beta, decay, r0):
    """beta exp(-decay (r / r0 - 1)) as one stretched-exponential term."""
    return [beta * float(np.exp(decay)) if not hasattr(beta, "shape") else beta * jnp.exp(decay),
            decay / r0, 1.0]


def iron_carbon(iron_values=None, carbon_values=None, **values):
    """Iron (species 0, the d model of Egorov et al., ``iron_egorov``) and carbon
    (species 1, the sp model ``CARBON_FIT``) with Fe 3d - C 2s, 2p bonds.

    Use with ``BOPConfig(**IRON_CARBON_CONFIG, spin=...)`` and
    ``radial=models.tapered_exp_sum``.  Fe-Fe and C-C interactions are exactly
    those of ``iron_egorov()`` and ``carbon()``; ``iron_values`` and
    ``carbon_values`` override their parameters.  Keyword arguments override
    entries of ``IRON_CARBON_FIT``, the Fe-C terms:

      e_c                 shift of the carbon levels (2s and 2p together)
                          against the iron d level (eV)
      sds, pds, pdp       C 2s - Fe 3d and C 2p - Fe 3d bond integrals at r0,
                          each ``beta * exp(-decay (r / r0 - 1))`` (eV)
      decay_pi            decay of ``pdp`` alone (None: ``decay``)
      pair, pair_decay    Fe-C pair term ``pair * exp(-pair_decay (r / r0 - 1))``
      core, core_r        short-range Fe-C core repulsion
                          ``core (core_r - r)^3 / r exp(core_r - r)`` for
                          r < core_r (Egorov's core form); with core_r
                          between the two neighbour distances of the
                          octahedral site (1.43 and 2.02 A) it acts on the
                          axial neighbours only
      rho_fe, rho_c, rho_decay
                          embedding density that a C neighbour adds on Fe
                          (``rho_fe``) and an Fe neighbour adds on C
                          (``rho_c``), both ``* exp(-rho_decay (r / r0 - 1))``
      rho_gauss_fe, rho_gauss_c, rho_gauss_r, rho_gauss_w
                          Gaussian embedding densities a C neighbour adds on
                          Fe and an Fe neighbour adds on C,
                          ``amplitude * exp(-((r - rho_gauss_r) / rho_gauss_w)^2)``
                          (default amplitude 0: none); centred near the
                          transverse neighbours of the octahedral site
                          (2.02 A) it binds them without pushing on them
      rcut, dcut          range of every Fe-C term (taper from rcut - dcut)
      hubbard_c           onsite Coulomb integral of carbon (eV); it does
                          nothing in pure carbon (no charge transfer)

    The Yukawa repulsion of iron acts between iron atoms only: carbon neither
    feels it nor screens it.
    """
    fe = dict(IRON_EGOROV, **(iron_values or {}))
    cv = dict(CARBON_FIT, **(carbon_values or {}))
    v = dict(IRON_CARBON_FIT, **values)
    x_cut = (v["rcut"], v["dcut"])
    c_cut = (CARBON_CONFIG["rcut"], CARBON_CONFIG["dcut"])
    zero_bond = [0.0, 1.0, 1.0] * 3 + list(NO_TAPER)

    def bond_row(terms, cut):
        flat = [x for t in terms for x in t] + [0.0, 1.0, 1.0] * (3 - len(terms))
        return flat + list(cut)

    def table(entries, zero):
        t = [[zero, zero], [zero, zero]]
        for (a, b), row in entries.items():
            t[a][b] = row
        return jnp.array(t)

    def c_term(c):
        return [_exp_term(cv[c], cv["decay"], cv["r0"])]

    def x_term(c):
        decay = v["decay_pi"] if (c == "pdp" and v.get("decay_pi") is not None) else v["decay"]
        return [_exp_term(v[c], decay, v["r0"])]

    # mixed channels: entry [a, b] has the lower shell on a, the higher on b
    bond = {c: table({(FE, FE): bond_row(fe[c], NO_TAPER)}, zero_bond)
            for c in ("dds", "ddp", "ddd")}
    for c in ("sss", "sps", "pps", "ppp"):
        bond[c] = table({(C, C): bond_row(c_term(c), c_cut)}, zero_bond)
    for c in ("sds", "pds", "pdp"):
        bond[c] = table({(C, FE): bond_row(x_term(c), x_cut)}, zero_bond)

    zero7 = [0.0, 1.0, 1.0, 0.0, 0.0] + list(NO_TAPER)
    c_pair = _exp_term(cv["pair"], cv["pair_decay"], cv["r0"])
    x_pair = _exp_term(v["pair"], v["pair_decay"], v["r0"])
    pair = table({
        (FE, FE): [*fe["pair"], *fe["core"], *NO_TAPER],
        (C, C): [*c_pair, 0.0, 0.0, *c_cut],
        (FE, C): [*x_pair, v["core_r"], v["core"], *x_cut],
        (C, FE): [*x_pair, v["core_r"], v["core"], *x_cut]}, zero7)
    c_rho = _exp_term(1.0, cv["density_decay"], cv["r0"])[:2]
    g_width = 1.0 / v.get("rho_gauss_w", 0.3) ** 2
    g_r = v.get("rho_gauss_r", 2.0)
    density = table({
        (FE, FE): [1.0, fe["embed"][1], fe["embed"][2], 0.0, 0.0, *NO_TAPER],
        (C, C): [0.0, 1.0, 0.0, *c_rho, *c_cut],
        (FE, C): [v.get("rho_gauss_fe", 0.0), g_width, g_r,
                  *_exp_term(v["rho_fe"], v["rho_decay"], v["r0"])[:2], *x_cut],
        (C, FE): [v.get("rho_gauss_c", 0.0), g_width, g_r,
                  *_exp_term(v["rho_c"], v["rho_decay"], v["r0"])[:2], *x_cut]},
        zero7)
    y = fe["yukawa"]

    def fe_only(value):
        return jnp.array([[value, 0.0], [0.0, 0.0]])

    e_c = v["e_c"]
    return {
        # shells (s, p, d); iron has only d, carbon only s and p
        "onsite": jnp.array([[0.0, 0.0, 0.0], [cv["e_s"] + e_c, e_c, 0.0]]),
        "n0": jnp.array([[0.0, 0.0, fe["N"]], [2.0, 2.0, 0.0]]),
        "hubbard": jnp.array([8.0, v["hubbard_c"]]),
        "stoner": jnp.array([[0.0, 0.0, fe["stoner"]], [0.0, 0.0, 0.0]]),
        "bond": bond,
        "repulsion": {"pair": pair, "density": density,
                      "embed": jnp.array([[-fe["embed"][0], 0.5], [-cv["embed"], 0.5]])},
        "yukawa": {"B": fe_only(y["B"]), "r_core": fe_only(y["r_core"]),
                   "C": fe_only(y["C"]), "upsilon": fe_only(y["upsilon"]),
                   "lambda0": jnp.array([y["lambda0"], 0.0]),
                   "m": jnp.array([y["m"], 1.0])},
    }


def _iron_carbon_config():
    from .bop import SpeciesSettings

    e = IRON_EGOROV_CONFIG
    fe = SpeciesSettings(basis=("d",), nmom=e["nmom"], nexp=e["nexp"], kernel=e["kernel"],
                         terminator=e["terminator"], bandwidth=e["bandwidth"])
    c = SpeciesSettings(basis=CARBON_CONFIG["basis"], nmom=CARBON_CONFIG["nmom"],
                        nexp=CARBON_CONFIG["nexp"], bandwidth=CARBON_CONFIG["bandwidth"],
                        kernel=CARBON_CONFIG["kernel"], terminator="none",
                        bandwidth_guard=CARBON_CONFIG["bandwidth_guard"],
                        spin_polarized=False, atomic_shifts=True)
    # global cutoffs are those of the iron model; carbon and Fe-C pairs taper
    # off inside their own functions, well inside the global ones
    return dict(basis=("s", "p", "d"), species=(fe, c),
                nmom=max(e["nmom"], CARBON_CONFIG["nmom"]),
                rcut=e["rcut"], dcut=e["dcut"], rcut2=e["rcut2"], dcut2=e["dcut2"],
                repulsion=FECARBON_REPULSION, yukawa=True)


# BOPConfig settings of the iron-carbon model (pass radial=models.tapered_exp_sum)
IRON_CARBON_CONFIG = _iron_carbon_config()
