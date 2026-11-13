"""Orthogonal two-centre tight-binding blocks for s, p and d orbitals.

Slater-Koster blocks are written as projectors that are polynomial in the bond
direction ``u``, so no local bond frame is needed and nothing is singular for
bonds along a Cartesian axis.  d orbitals are represented by an orthonormal
basis of symmetric traceless tensors ``B_m`` (order: xy, yz, zx, x2-y2, 3z2-r2).

Channel names: "sss", "sps", "pps", "ppp", "sds", "pds", "pdp", "dds", "ddp",
"ddd" (two shells, then sigma/pi/delta).  For a mixed channel such as "sps" the
parameter array entry ``[a, b]`` means the lower shell on species a and the
higher shell on species b.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

L_OF = {"s": 0, "p": 1, "d": 2}
CHANNELS = {
    ("s", "s"): ("sss",),
    ("s", "p"): ("sps",),
    ("s", "d"): ("sds",),
    ("p", "p"): ("pps", "ppp"),
    ("p", "d"): ("pds", "pdp"),
    ("d", "d"): ("dds", "ddp", "ddd"),
}


def _d_tensors() -> np.ndarray:
    e = np.eye(3)
    ex, ey, ez = e

    def sym(a, b):
        return (np.outer(a, b) + np.outer(b, a)) / np.sqrt(2.0)

    xy, yz, zx = sym(ex, ey), sym(ey, ez), sym(ez, ex)
    x2y2 = (np.outer(ex, ex) - np.outer(ey, ey)) / np.sqrt(2.0)
    z2 = (2 * np.outer(ez, ez) - np.outer(ex, ex) - np.outer(ey, ey)) / np.sqrt(6.0)
    return np.stack([xy, yz, zx, x2y2, z2])


D_TENSORS = _d_tensors()


def shell_slices(basis):
    """Orbital index range of each shell, and the total number of orbitals."""
    out, start = [], 0
    for sh in basis:
        n = 2 * L_OF[sh] + 1
        out.append((start, start + n))
        start += n
    return out, start


def _block_upper(la, lb, u, v):
    """Block <la on atom i | H | lb on atom j> with la <= lb, u pointing i -> j."""
    b = jnp.asarray(D_TENSORS, dtype=u.dtype)
    if (la, lb) == ("s", "s"):
        return v["sss"].reshape(1, 1)
    if (la, lb) == ("s", "p"):
        return (v["sps"] * u)[None, :]
    if (la, lb) == ("p", "p"):
        eye = jnp.eye(3, dtype=u.dtype)
        return v["ppp"] * eye + (v["pps"] - v["ppp"]) * jnp.outer(u, u)
    bu = b @ u  # (5, 3)
    q = bu @ u  # (5,)   u.B_m.u
    sig = jnp.sqrt(1.5) * q  # overlap of B_m with the sigma tensor
    if (la, lb) == ("s", "d"):
        return (v["sds"] * sig)[None, :]
    if (la, lb) == ("p", "d"):
        pi = jnp.sqrt(2.0) * (bu.T - jnp.outer(u, q))  # (3, 5)
        return v["pds"] * jnp.outer(u, sig) + v["pdp"] * pi
    if (la, lb) == ("d", "d"):
        p_sig = jnp.outer(sig, sig)
        p_pi = 2.0 * (bu @ bu.T - jnp.outer(q, q))
        eye = jnp.eye(5, dtype=u.dtype)
        return (
            v["ddd"] * eye
            + (v["ddp"] - v["ddd"]) * p_pi
            + (v["dds"] - v["ddd"]) * p_sig
        )
    raise ValueError(f"unsupported shell pair {la}{lb}")


def powerlaw(p, r):
    """Canonical-type bond integral  p[0] * (p[1] / r) ** p[2]."""
    return p[0] * (p[1] / r) ** p[2]


def cosine_cutoff(r, rcut, dcut):
    """1 below rcut - dcut, 0 above rcut, half-cosine in between (BOPfox form)."""
    step = jnp.where(r < rcut, 1.0, 0.0)
    if dcut <= 0.0:
        return step
    arg = jnp.pi * (r - rcut + dcut) / dcut
    smooth = 0.5 * (1.0 + jnp.cos(jnp.clip(arg, 0.0, jnp.pi)))
    return jnp.where(r < rcut - dcut, 1.0, smooth) * step


def bond_blocks(rij, mask, species_i, species_j, bond_params, basis, rcut, dcut,
                radial=powerlaw, dd_screen=None):
    """Hopping blocks for every neighbour slot, shape (N, K, norb, norb).

    Entry [i, k, a, b] is <orbital a on atom i | H | orbital b on neighbour k>.
    ``dd_screen`` is the pair (scale, shift) from ``screening_factors``; the
    d-d integrals then become ``scale * beta + shift``.
    """
    if dd_screen is None:
        scale = jnp.ones(rij.shape[:2] + (3,), dtype=rij.dtype)
        dd_screen = (scale, jnp.zeros_like(scale))

    def one(rvec, m, si, sj, scale, shift):
        r = jnp.linalg.norm(rvec)
        u = rvec / r
        fc = cosine_cutoff(r, rcut, dcut) * m
        rows = []
        for la in basis:
            row = []
            for lb in basis:
                if L_OF[la] <= L_OF[lb]:
                    v = {c: radial(bond_params[c][si, sj], r)
                         for c in CHANNELS[(la, lb)]}
                    if (la, lb) == ("d", "d"):
                        v = {c: v[c] * scale[t] + shift[t]
                             for t, c in enumerate(CHANNELS[("d", "d")])}
                    row.append(_block_upper(la, lb, u, v))
                else:
                    # <la_i|H|lb_j> = <lb_j|H|la_i>^T, seen from j the bond is -u
                    v = {c: radial(bond_params[c][sj, si], r)
                         for c in CHANNELS[(lb, la)]}
                    row.append(_block_upper(lb, la, -u, v).T)
            rows.append(row)
        return fc * jnp.block(rows)

    return jax.vmap(jax.vmap(one))(rij, mask.astype(rij.dtype), species_i, species_j,
                                   *dd_screen)


# --- screening of the d-d bond integrals ------------------------------------------

SCREENING_KEYS = ("sds", "sds_overlap", "dds_overlap", "ddp_overlap", "ddd_overlap")
# (1 + delta_{tau,sigma}) / 4: one half from symmetrising, and one half for pi
# and delta from averaging the two degenerate orbitals around the bond axis
_SCREEN_WEIGHT = np.array([0.5, 0.25, 0.25])


def screening_terms(rij, mask, species, nbr_origin, nbr_reverse, screen_params,
                    rcut, dcut, radial=powerlaw):
    """Ingredients of the screening function for every bond slot.

    A valence s orbital on each atom k around the bond i-j is coupled to the d
    orbitals by an sd-sigma bond integral and overlap.  With the bond as
    quantisation axis, the s orbital on k sees the d orbital of type tau on i
    through the Slater-Koster factor g_tau of the angle between i->j and i->k:

        g_sigma = (3 n^2 - 1) / 2,  g_pi = sqrt(3) n l,  g_delta = sqrt(3)/2 l^2

    (n, l: cosine and sine of that angle).  Returns four (N, K, 3) arrays, the
    last axis being sigma, pi, delta:

      interference  c1 * beta = w sum_k [beta_sd(ik) O_sd(kj) + O_sd(ik) beta_sd(kj)] g_i g_j
      mu2           1/2 (mu2_i + mu2_j),  mu2_i = O_tau(ij)^2 + 2 w sum_k O_sd(ik)^2 g_i^2
      mu3           4 w O_tau(ij) sum_k O_sd(ik) O_sd(kj) g_i g_j
      overlap       O_tau(ij)

    with w = (1 + delta_{tau,sigma}) / 4 and k running over all atoms other
    than i and j.  All radial functions carry the bond-integral cutoff, so
    only neighbours of i (and of j for mu2_j) contribute.
    """
    dtype = rij.dtype
    n_slots = rij.shape[1]
    rad = jnp.vectorize(radial, signature="(p),()->()")
    maskf = jnp.asarray(mask, dtype=dtype)
    sp_i = species[:, None]
    sp_j = species[nbr_origin]

    r = jnp.linalg.norm(rij, axis=-1)  # (N, K)
    u = rij / r[..., None]
    fc = cosine_cutoff(r, rcut, dcut) * maskf
    # s orbital on the neighbour, d orbital on the central atom
    o_sd = fc * rad(screen_params["sds_overlap"][sp_j, sp_i], r)
    b_sd = fc * rad(screen_params["sds"][sp_j, sp_i], r)
    o_dd = jnp.stack([fc * rad(screen_params[c + "_overlap"][sp_i, sp_j], r)
                      for c in CHANNELS[("d", "d")]], axis=-1)  # (N, K, 3)

    # pairs of slots: a is the bond i-j, b the screening atom k
    pair = (mask[:, :, None] & mask[:, None, :]
            & ~np.eye(n_slots, dtype=bool)[None])  # (N, K, K)
    pairf = jnp.asarray(pair, dtype=dtype)
    dummy = jnp.array([1.0, 0.0, 0.0], dtype=dtype)
    r_jk = jnp.where(pair[..., None], rij[:, None, :, :] - rij[:, :, None, :], dummy)
    d_jk = jnp.linalg.norm(r_jk, axis=-1)  # (N, K, K)
    z_i = jnp.einsum("nax,nbx->nab", u, rij)  # component of i->k along the bond
    z_j = z_i - r[:, :, None]
    d_ik = r[:, None, :]
    n_i, n_j = z_i / d_ik, z_j / d_jk
    rho2 = d_ik**2 - z_i**2  # squared distance of k from the bond axis
    ll = rho2 / (d_ik * d_jk)  # sin_i * sin_j, smooth for k on the axis

    g_ij = jnp.stack([0.25 * (3 * n_i**2 - 1) * (3 * n_j**2 - 1),
                      3.0 * n_i * n_j * ll,
                      0.75 * ll**2], axis=-1)  # (N, K, K, 3)
    l2 = rho2 / d_ik**2
    g_ii = jnp.stack([0.25 * (3 * n_i**2 - 1) ** 2,
                      3.0 * n_i**2 * l2,
                      0.75 * l2**2], axis=-1)

    # sd functions between k and j: s on k, d on j
    sp_k = sp_j[:, None, :]
    sp_jj = sp_j[:, :, None]
    fc_jk = cosine_cutoff(d_jk, rcut, dcut) * pairf
    o_kj = fc_jk * rad(screen_params["sds_overlap"][sp_k, sp_jj], d_jk)
    b_kj = fc_jk * rad(screen_params["sds"][sp_k, sp_jj], d_jk)
    o_ik, b_ik = o_sd[:, None, :], b_sd[:, None, :]

    w = jnp.asarray(_SCREEN_WEIGHT, dtype=dtype)
    interference = w * jnp.einsum("nab,nabt->nat", b_ik * o_kj + o_ik * b_kj, g_ij)
    mu3 = 4.0 * w * o_dd * jnp.einsum("nab,nabt->nat", o_ik * o_kj, g_ij)
    ring = 2.0 * w * jnp.einsum("nab,nabt->nat", pairf * o_ik**2, g_ii)
    ring_j = ring[nbr_origin, nbr_reverse]  # the same sum around atom j
    mu2 = o_dd**2 + 0.5 * (ring + ring_j)
    return interference, mu2, mu3, o_dd


def screening_factors(rij, mask, species, nbr_origin, nbr_reverse, screen_params,
                      rcut, dcut, radial=powerlaw):
    """Environment dependence of the d-d bond integrals (Nguyen-Manh, Pettifor
    and Vitek, PRL 85, 4136 (2000)):

        beta~_tau = beta_tau (1 - S_tau),
        S_tau = (c1 - mu2 + mu3) / (1 + O_tau^2 - 2 mu2)

    Returns (scale, shift), each (N, K, 3), such that
    ``beta~ = scale * beta + shift``; this form needs no division by beta.
    """
    interference, mu2, mu3, o_dd = screening_terms(
        rij, mask, species, nbr_origin, nbr_reverse, screen_params, rcut, dcut,
        radial)
    den = 1.0 + o_dd**2 - 2.0 * mu2
    return 1.0 - (mu3 - mu2) / den, -interference / den


def v0_bond(p, r):
    """BOPfox "v0" bond integral (Lim): p = [prefactor, decay, mu, power, r0]."""
    x = r / p[4]
    return p[0] * jnp.exp(-p[1] * (x - 1.0)) * (1.0 - jnp.exp(-p[2] * x ** p[3]))
