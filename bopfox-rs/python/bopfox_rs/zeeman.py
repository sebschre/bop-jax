"""First-order effect of an external field on energy barriers, and the
Peierls stress of an energy profile: a port of the post-processing functions
of ``bopjax.zeeman``.

For any configuration X the self-consistent energy in a field is
``E(X, B) = E(X, 0) - mu_B B.M(X) + O(B^2)``, and since minima and saddle
points are stationary in the positions, the first-order shift of a barrier
needs no relaxation in the field:
``dE_barrier/dB = -mu_B [M(saddle) - M(minimum)]`` (eV / T), with M the
total moment of the cell.  The moments come from ``Potential.scf`` /
``compute`` (``magnetic_moment``, -(1/mu_B) dE/dB, at B = 0 or any field).
"""

from __future__ import annotations

import numpy as np

from . import G_SPIN, MU_B  # noqa: F401

__all__ = ["first_order_energies", "first_order_barrier", "peierls_stress"]


def first_order_energies(moments, field):
    """First-order Zeeman energy of each image of a path relative to the first,
    ``-mu_B B.(M_p - M_0)`` in eV, shape (P,).

    moments : (P, 3) total moments of each image, or (P, N, 3) per atom
              (summed over atoms), in Bohr magnetons, at B = 0
    field   : (3,) external field in Tesla
    """
    m = np.asarray(moments, dtype=float)
    if m.ndim == 3:
        m = m.sum(axis=1)
    return -MU_B * (m - m[0]) @ np.asarray(field, dtype=float)


def first_order_barrier(energies, moments, initial=0):
    """First-order field dependence of the barrier of a path.

    energies : (P,) zero-field energies of the images
    moments  : (P, 3) or (P, N, 3) zero-field magnetic moments (Bohr magnetons)
    initial  : index of the minimum the barrier is measured from

    Returns a dict with the saddle index (highest image), the barrier
    ``E[saddle] - E[initial]`` (eV), the moment change
    ``delta_moment = M[saddle] - M[initial]`` (Bohr magnetons, (3,)) and
    ``gradient = -mu_B delta_moment``, the derivative of the barrier with
    respect to B (eV / T, (3,)): the barrier in a field B is
    ``barrier + gradient.B`` to first order.
    """
    e = np.asarray(energies, dtype=float)
    m = np.asarray(moments, dtype=float)
    if m.ndim == 3:
        m = m.sum(axis=1)
    saddle = int(np.argmax(e))
    delta = m[saddle] - m[initial]
    return {"saddle": saddle, "barrier": float(e[saddle] - e[initial]),
            "delta_moment": delta, "gradient": -MU_B * delta}


def _hermite_slope(x, y, dy, samples):
    """Derivative of the piecewise cubic Hermite interpolant on a fine grid."""
    out_x, out_s = [], []
    for k in range(len(x) - 1):
        h = x[k + 1] - x[k]
        t = np.linspace(0.0, 1.0, samples, endpoint=k == len(x) - 2)
        # derivatives of the Hermite basis functions with respect to t
        d00, d10 = 6 * t**2 - 6 * t, 3 * t**2 - 4 * t + 1
        d01, d11 = -6 * t**2 + 6 * t, 3 * t**2 - 2 * t
        out_s.append((d00 * y[k] + d10 * h * dy[k] + d01 * y[k + 1] + d11 * h * dy[k + 1]) / h)
        out_x.append(x[k] + h * t)
    return np.concatenate(out_x), np.concatenate(out_s)


def peierls_stress(x, energies, burgers, length, slopes=None, samples=200):
    """Largest resolved shear stress a line can resist along an energy profile.

    x        : (P,) position of the dislocation along the glide direction (A)
    energies : (P,) energy of each image (eV) for a line of ``length`` (A)
    burgers  : Burgers vector length b (A)
    slopes   : (P,) dE/dx at the images (eV / A); finite differences if None

    The resolved shear stress tau does work tau b per unit length and unit
    glide distance, so the line moves once tau b length exceeds dE/dx
    everywhere: tau_P = max dE/dx / (b length).  The profile is interpolated
    by cubic Hermite polynomials.  Returns ``(tau_P in eV / A^3, x at the
    steepest point)``; multiply by 160.21766208 for GPa.
    """
    x = np.asarray(x, dtype=float)
    e = np.asarray(energies, dtype=float)
    dy = np.gradient(e, x) if slopes is None else np.asarray(slopes, dtype=float)
    grid, slope = _hermite_slope(x, e, dy, samples)
    k = int(np.argmax(slope))
    return float(slope[k] / (burgers * length)), float(grid[k])
