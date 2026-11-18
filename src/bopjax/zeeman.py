"""External magnetic field: Zeeman term, magnetic moments and first-order
field effects on energy barriers.

``params["magnetic_field"]`` (Tesla, shape (3,) or (N, 3)) adds

    H_Z = -mu_B B.(L + g/2 sigma)

to the onsite Hamiltonian of every atom (see ``bop.make_energy_fn``).  Moments
are counted as everywhere in bopjax: the magnetisation is along ``m e`` and
the orbital magnetic moment in Bohr magnetons is ``<L>`` where the spin one is
``m e``.  The field therefore favours a magnetic moment

    M_i = g/2 <sigma>_i + <L>_i        (Bohr magnetons)

along B.  Time reversal maps this convention onto the physical one (electron
moment antiparallel to its angular momentum) without changing any energy.

Magnetic moment as a field derivative
-------------------------------------
``make_magnetic_moment`` returns ``M_i = -(1/mu_B) dE/dB_i``, evaluated as the
partial derivative at fixed onsite levels.  At self-consistent levels this is
the derivative of the self-consistent energy (Hellmann-Feynman), as for the
forces.  The spin part has two pieces: ``m e`` along the local axis, and in
non-collinear calculations the spin density perpendicular to the axes that
spin-orbit coupling or non-collinear neighbours induce.  The orbital part is
non-zero at B = 0 only with spin-orbit coupling.

First-order field effect on a barrier
-------------------------------------
For any configuration X the self-consistent energy in a field is

    E(X, B) = E(X, 0) - mu_B B.M(X) + O(B^2),

and because a minimum and a saddle point are stationary in the positions, the
first-order shift of a barrier between them needs no relaxation in the field:

    dE_barrier / dB = -mu_B [M(saddle) - M(minimum)]      (eV / T)

with M the total moment of the cell.  ``first_order_energies`` and
``first_order_barrier`` evaluate this along a path (e.g. from
``bopjax.neb``), and ``peierls_stress`` turns an energy profile into the
largest resolved shear stress it can resist.

The second-order terms (the susceptibility, i.e. the change of M with B, and
magnetostriction) are left out.  For a saturated ferromagnet the moment
changes by a small fraction of a percent at laboratory fields (the high-field
susceptibility), so the first-order term dominates there; compare with the
self-consistent energy in a field to check this for a given model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .bop import G_SPIN, MU_B

__all__ = ["G_SPIN", "MU_B", "make_magnetic_moment", "first_order_energies",
           "first_order_barrier", "peierls_stress"]


def make_magnetic_moment(energy_fn):
    """Return ``magnetic_moment(shifts, positions, cell, params) -> dict`` with
    the spin, orbital and total magnetic moments of each atom, (N, 3) each, in
    Bohr magnetons:

      "spin"    g/2 <sigma> = -(1/mu_B) dE/dB through the spin term
      "orbital" <L>         = -(1/mu_B) dE/dB through the orbital term
      "total"   their sum, -(1/mu_B) dE/dB

    evaluated at the field in ``params`` (zero if absent).  At self-consistent
    ``shifts`` these are derivatives of the self-consistent energy.  In
    collinear calculations only the z component of the spin moment is
    non-zero and there is no orbital moment.
    """
    cfg = energy_fn.config
    if not cfg.magnetic:
        raise ValueError("the magnetic moment needs a magnetic calculation (cfg.spin)")
    n = energy_fn.topology.n_atoms
    noncollinear = cfg.spin == "noncollinear"

    def magnetic_moment(shifts, positions, cell, params):
        dtype = jnp.asarray(positions).dtype
        zero = jnp.zeros((n, 3), dtype=dtype)

        def energy(h, b):
            here = dict(params)
            here["spin_field"] = params["spin_field"] + h if "spin_field" in params else h
            if noncollinear:
                here["orbital_field"] = (params["orbital_field"] + b
                                         if "orbital_field" in params else b)
            return energy_fn(shifts, positions, cell, here)[0]

        d_sigma, d_l = jax.grad(energy, argnums=(0, 1))(zero, zero)
        # h = -mu_B g/2 B and b = -mu_B B, so -(1/mu_B) dE/dB = g/2 dE/dh + dE/db
        spin = 0.5 * G_SPIN * d_sigma
        if not noncollinear:
            spin = spin * jnp.array([0.0, 0.0, 1.0], dtype=dtype)
        return {"spin": spin, "orbital": d_l, "total": spin + d_l}

    return magnetic_moment


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
