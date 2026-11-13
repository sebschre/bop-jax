"""Moments -> recursion coefficients -> Chebyshev expansion of the local DOS.

Follows BOPfox (``momtorec``, ``BOP_GetBandwidth``, ``BOP_MomToSigma``,
``BOP_GetSigmaRecursive``, ``BOPCHEB_GetChi``) but contains no derivative code:
every derivative is obtained from JAX.  All functions act on one atom/shell and
are vmapped by the caller.
"""

from __future__ import annotations

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


def band_centre_and_width(a, b, nmom: int, bandwidth: str = "lowest"):
    """a_inf and b_inf; the band is [a_inf - 2 b_inf, a_inf + 2 b_inf]."""
    if bandwidth == "lowest":
        return (a[0] if nmom == 2 else a[1]), b[1]
    if bandwidth == "highest":
        return a[-1], b[-1]
    if bandwidth == "averaged":
        return jnp.mean(a), jnp.sqrt(jnp.mean(b[1:] ** 2))
    raise ValueError(f"unknown bandwidth estimate {bandwidth!r}")


def jackson_kernel(nexp: int) -> np.ndarray:
    n_t = nexp + 1
    n = np.arange(nexp + 3)
    g = ((n_t - n + 1) * np.cos(np.pi * n / (n_t + 1))
         + np.sin(np.pi * n / (n_t + 1)) / np.tan(np.pi / (n_t + 1))) / (n_t + 1)
    kern = np.zeros(nexp + 1)
    kern[:] = g[1 : nexp + 2] / g[1]
    return kern


def sigma_coefficients(a, b, ainf, binf, nmom: int, nexp: int, kernel):
    """Chebyshev (2nd kind) moments sigma_0..sigma_nexp of the scaled DOS.

    Recursion coefficients beyond those fixed by the moments are replaced by the
    constant terminator a = a_inf, b = b_inf.
    """
    amax, bmax = n_recursion(nmom)
    _, nrec = n_recursion(nexp)
    nrec = max(nrec, 1)
    a_hat = jnp.zeros(nrec + 1, dtype=a.dtype)
    b_hat = jnp.full(nrec + 1, 0.5, dtype=a.dtype)
    na, nb = min(amax, nrec) + 1, min(bmax, nrec) + 1
    a_hat = a_hat.at[:na].set((a[:na] - ainf) / (2.0 * binf))
    b_hat = b_hat.at[:nb].set(b[:nb] / (2.0 * binf))

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
