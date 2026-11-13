"""Moments -> recursion coefficients -> Chebyshev expansion of the local DOS.

Follows BOPfox (``momtorec``, ``BOP_GetBandwidth``, ``BOP_MomToSigma``,
``BOP_GetSigmaRecursive``, ``BOPCHEB_GetChi``) but contains no derivative code:
every derivative is obtained from JAX.  All functions act on one atom/shell and
are vmapped by the caller.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def n_recursion(nmom: int) -> tuple[int, int]:
    """Highest a_n and b_n that ``nmom`` moments determine."""
    if nmom % 2:
        return (nmom - 1) // 2, (nmom - 1) // 2
    return nmom // 2 - 1, nmom // 2


def recursion_coefficients(mu, nmom: int):
    """a_0..a_amax and b_0..b_bmax (b_0 = 0) from moments mu_0..mu_nmom.

    Horsfield et al., Mat. Sci. Eng. B 37, 219 (1996): build the orthogonal
    polynomials P_n = sum_j c^n_j x^j of the local DOS.
    """
    amax, bmax = n_recursion(nmom)
    size = bmax + 1
    # hankel[j, l] = mu_{j+l+1}; entries past nmom are never weighted
    idx = np.add.outer(np.arange(size), np.arange(size)) + 1
    hankel = jnp.where(idx <= nmom, mu[np.minimum(idx, nmom)], 0.0)

    c_prev2 = jnp.zeros(size, dtype=mu.dtype)
    c_prev = jnp.zeros(size, dtype=mu.dtype).at[0].set(1.0)
    a = [mu[1]]
    b = [jnp.zeros((), dtype=mu.dtype)]
    for n in range(1, bmax + 1):
        shifted = jnp.concatenate([jnp.zeros(1, dtype=mu.dtype), c_prev[:-1]])
        c_tilde = shifted - a[n - 1] * c_prev - b[n - 1] * c_prev2
        bn = jnp.sqrt(c_tilde @ hankel @ c_prev)
        c = c_tilde / bn
        b.append(bn)
        if n <= amax:
            a.append(c @ hankel @ c)
        c_prev2, c_prev = c_prev, c
    return jnp.stack(a), jnp.stack(b)


def _bpa_edges(a, b, centre):
    """Extreme eigenvalues of the Beer-Pettifor-Aoki matrix for a trial centre."""
    nrec = a.shape[0] - 1
    diag = 0.5 * (a - centre)
    diag = diag.at[0].set(a[0] - centre).at[nrec].set(a[nrec] - centre)
    off = 0.5 * b[1 : nrec + 1]
    off = off.at[0].set(b[1] / jnp.sqrt(2.0)).at[nrec - 1].set(b[nrec] / jnp.sqrt(2.0))
    ev = jnp.linalg.eigvalsh(jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1))
    return ev[0], ev[-1]


def _beer_pettifor_aoki(a, b):
    """Band centre for which the BPA matrix has symmetric band edges.

    Bisection without gradients followed by two differentiable Newton steps,
    as for the Fermi level.
    """
    def mismatch(centre, a_, b_):
        lo, hi = _bpa_edges(a_, b_, centre)
        return hi + lo

    af, bf = jax.lax.stop_gradient(a), jax.lax.stop_gradient(b)
    width = 4.0 * jnp.max(bf) + jnp.max(af) - jnp.min(af) + 1.0
    lo, hi = jnp.min(af) - width, jnp.max(af) + width

    def bisect(_, bracket):
        lo_, hi_ = bracket
        mid = 0.5 * (lo_ + hi_)
        positive = mismatch(mid, af, bf) > 0.0  # mismatch decreases with centre
        return jnp.where(positive, mid, lo_), jnp.where(positive, hi_, mid)

    lo, hi = jax.lax.fori_loop(0, 80, bisect, (lo, hi))
    centre = jax.lax.stop_gradient(0.5 * (lo + hi))
    for _ in range(2):
        f, df = jax.value_and_grad(mismatch)(centre, a, b)
        centre = centre - f / df
    lo_e, hi_e = _bpa_edges(a, b, centre)
    return centre, 0.5 * (jnp.abs(hi_e) + jnp.abs(lo_e))


BANDWIDTHS = ("azero", "lowest", "highest", "avgan_maxbn", "global", "averaged",
              "beerpettiforaoki", "beerpettiforaoki_bopfox", "findeminemax",
              "gershgorin")


def band_centre_and_width(a, b, nmom: int, bandwidth: str = "lowest",
                          global_ab=(0.0, 1.0)):
    """a_inf and b_inf; the band is [a_inf - 2 b_inf, a_inf + 2 b_inf].

    Same estimators as BOPfox ``BOP_GetBandwidth``.
    """
    amax, bmax = n_recursion(nmom)
    if bandwidth == "azero":
        return a[0], b[1]
    if bandwidth == "lowest":
        return (a[0] if nmom == 2 else a[1]), b[1]
    if bandwidth == "highest":
        return a[amax], b[bmax]
    if bandwidth == "avgan_maxbn":
        return jnp.mean(a), b[bmax]
    if bandwidth == "global":
        return (jnp.asarray(global_ab[0], dtype=a.dtype),
                jnp.asarray(global_ab[1], dtype=a.dtype))
    if bandwidth == "averaged":
        return jnp.mean(a), jnp.sqrt(jnp.mean(b[1:] ** 2))
    if bandwidth == "findeminemax":
        e_min = jnp.min(a) - 2.0 * jnp.max(b[1:])
        e_max = jnp.max(a) + 2.0 * jnp.max(b[1:])
        return 0.5 * (e_max + e_min), 0.25 * (e_max - e_min)
    if bandwidth in ("gershgorin", "gershogorin"):
        i = np.arange(bmax)
        e_min = jnp.min(a[i] - b[i] - b[i + 1])
        e_max = jnp.max(a[i] + b[i] + b[i + 1])
        return 0.5 * (e_max + e_min), 0.25 * (e_max - e_min)
    if bandwidth == "beerpettiforaoki":
        return _beer_pettifor_aoki(a, b[: amax + 1])
    if bandwidth == "beerpettiforaoki_bopfox":
        # What BOPfox evaluates: it drops the last a_n when amax == bmax and then
        # reads one element past the array, which holds zero.  Kept only so the
        # two codes can be compared; the zero breaks invariance under a uniform
        # shift of the energy scale.
        nrec = (amax - 1 if amax == bmax else amax) + 1
        a_pad = jnp.concatenate([a[:nrec], jnp.zeros(1, dtype=a.dtype)])
        return _beer_pettifor_aoki(a_pad, b[: nrec + 1])
    raise ValueError(f"unknown bandwidth estimate {bandwidth!r}")


def jackson_kernel(nexp: int) -> np.ndarray:
    n_t = nexp + 1
    n = np.arange(nexp + 3)
    g = ((n_t - n + 1) * np.cos(np.pi * n / (n_t + 1))
         + np.sin(np.pi * n / (n_t + 1)) / np.tan(np.pi / (n_t + 1))) / (n_t + 1)
    kern = np.zeros(nexp + 1)
    kern[:] = g[1 : nexp + 2] / g[1]
    return kern


TERMINATORS = ("none", "constantabn", "oscillation", "weighted")


def terminated_chain(a, b, ainf, binf, nmom: int, nrec: int, terminator: str = "none",
                     alpha_weight: float = 1.0):
    """Scaled recursion coefficients a_hat, b_hat for chain sites 0..nrec.

    Coefficients fixed by the moments are used as they are; the rest follow
    the chosen terminator (BOPfox ``BOP_MomToSigma``).
    """
    amax, bmax = n_recursion(nmom)
    a_hat = jnp.zeros(nrec + 1, dtype=a.dtype)
    b_hat = jnp.full(nrec + 1, 0.5, dtype=a.dtype)
    a_ex = (a - ainf) / (2.0 * binf)
    b_ex = b / (2.0 * binf)
    idx = np.arange(nrec + 1)

    if terminator == "weighted":
        wa = 1.0 / (alpha_weight * (amax - np.arange(amax + 1)) + 1.0)
        wb = 1.0 / (alpha_weight * (bmax - np.arange(bmax + 1)) + 1.0)
        a_hat = a_hat + (jnp.sum(wa * a) / np.sum(wa) - ainf) / (2.0 * binf)
        b_hat = jnp.full(nrec + 1, 1.0, dtype=a.dtype) * (
            jnp.sum(wb * b) / np.sum(wb)) / (2.0 * binf)
    elif terminator == "oscillation":
        even, odd = (bmax, bmax - 1) if bmax % 2 == 0 else (bmax - 1, bmax - 2)
        if odd - 2 < 0:
            raise ValueError("the oscillation terminator needs at least 8 moments")
        # BOPfox pads the exact coefficients with the constant terminator first
        a_pad = jnp.zeros(bmax + 1, dtype=a.dtype).at[: amax + 1].set(a_ex)
        a_even = 0.5 * (a_pad[even] + a_pad[even - 2])
        a_odd = 0.5 * (a_pad[odd] + a_pad[odd - 2])
        b_even = 0.5 * (b_ex[even] + b_ex[even - 2])
        b_odd = 0.5 * (b_ex[odd] + b_ex[odd - 2])
        a_hat = jnp.where(idx % 2 == 0, a_even, a_odd)
        b_hat = jnp.where(idx % 2 == 0, b_even, b_odd)
    elif terminator not in ("none", "constantabn"):
        raise ValueError(f"unknown terminator {terminator!r}")

    na, nb = min(amax, nrec) + 1, min(bmax, nrec) + 1
    return a_hat.at[:na].set(a_ex[:na]), b_hat.at[:nb].set(b_ex[:nb])


def sigma_coefficients(a, b, ainf, binf, nmom: int, nexp: int, kernel,
                       terminator: str = "none", alpha_weight: float = 1.0):
    """Chebyshev (2nd kind) moments sigma_0..sigma_nexp of the scaled DOS."""
    _, nrec = n_recursion(nexp)
    nrec = max(nrec, 1)
    a_hat, b_hat = terminated_chain(a, b, ainf, binf, nmom, nrec, terminator, alpha_weight)

    chain = jnp.diag(a_hat) + jnp.diag(b_hat[1:], 1) + jnp.diag(b_hat[1:], -1)
    k_prev = jnp.zeros(nrec + 1, dtype=a.dtype)
    k = jnp.zeros(nrec + 1, dtype=a.dtype).at[0].set(1.0)
    sig = [k[0]]
    for _ in range(nexp):
        k_prev, k = k, 2.0 * chain @ k - k_prev
        sig.append(k[0])
    return jnp.stack(sig) * jnp.asarray(kernel, dtype=a.dtype)


def response_functions(nchi: int, cos_phi):
    """Reduced response functions chi_0..chi_nchi at the Fermi phase."""
    c = jnp.clip(cos_phi, -1.0 + 1e-12, 1.0 - 1e-12)
    phi = jnp.arccos(c)
    n = np.arange(2, nchi + 1)
    chi_hi = (jnp.sin((n + 1) * phi) / (n + 1) - jnp.sin((n - 1) * phi) / (n - 1)) / jnp.pi
    chi0 = jnp.zeros((), dtype=phi.dtype)
    chi1 = 1.0 - (phi - 0.5 * jnp.sin(2.0 * phi)) / jnp.pi
    return jnp.concatenate([jnp.stack([chi0, chi1]), chi_hi])


def filling_and_bond_energy(sigma, ainf, binf, ef, gamma0):
    """Band filling fraction (0..1) and bond energy per orbital and spin.

    The bond energy is measured from the centre of gravity of the local DOS,
    i.e. it is the integral of (E - mu_1) n(E) up to the Fermi level.
    """
    nexp = sigma.shape[0] - 1
    chi = response_functions(nexp + 2, (ef - ainf) / (2.0 * binf))
    fill = jnp.sum(sigma * chi[1 : nexp + 2])
    bond = binf * jnp.sum(sigma * (chi[2:] - gamma0 * chi[1 : nexp + 2] + chi[:nexp + 1]))
    return fill, bond
