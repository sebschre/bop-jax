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
                radial=powerlaw):
    """Hopping blocks for every neighbour slot, shape (N, K, norb, norb).

    Entry [i, k, a, b] is <orbital a on atom i | H | orbital b on neighbour k>.
    """

    def one(rvec, m, si, sj):
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
                    row.append(_block_upper(la, lb, u, v))
                else:
                    # <la_i|H|lb_j> = <lb_j|H|la_i>^T, seen from j the bond is -u
                    v = {c: radial(bond_params[c][sj, si], r)
                         for c in CHANNELS[(lb, la)]}
                    row.append(_block_upper(lb, la, -u, v).T)
            rows.append(row)
        return fc * jnp.block(rows)

    return jax.vmap(jax.vmap(one))(rij, mask.astype(rij.dtype), species_i, species_j)


def v0_bond(p, r):
    """BOPfox "v0" bond integral (Lim): p = [prefactor, decay, mu, power, r0]."""
    x = r / p[4]
    return p[0] * jnp.exp(-p[1] * (x - 1.0)) * (1.0 - jnp.exp(-p[2] * x ** p[3]))
