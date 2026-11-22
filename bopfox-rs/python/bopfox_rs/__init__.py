"""Analytic bond-order potentials in Rust (energies and analytic forces).

``Potential(model)`` takes a model as a JSON string or a dict (see
``model_from_bopjax`` to convert a bopjax configuration and parameter set).
``bopfox_rs.calculator.BOPCalculator`` is an ASE calculator (needs ``ase``).
"""

from __future__ import annotations

import json

import numpy as np

from ._core import Potential as _Potential

__all__ = ["Potential", "model_from_bopjax"]


class Potential:
    def __init__(self, model):
        self.model = model if isinstance(model, dict) else json.loads(model)
        self._core = _Potential(json.dumps(self.model))
        self.shifts_per_atom = self._core.shifts_per_atom

    def clear_history(self):
        """Forget the Anderson history kept for ``scf(reuse_history=True)``."""
        self._core.clear_history()

    def compute(self, positions, species, cell=None, shifts=None, gradients=True, axes=None,
                magnetic_field=None, spin_field=None, orbital_field=None):
        """Dict of numpy arrays: energy, energy_atom, fermi, n_shell, charge,
        moments, a_inf, b_inf (and moment when magnetic); with ``gradients``
        also forces (-dE/dpositions), virial (dE/dstrain) and shift_gradient
        (dE/dshifts, shaped like the shifts: (N, nspin, nshell)).  For
        ``spin="noncollinear"``, ``axes`` (N, 3) are the local spin axes
        (default z) and ``axes_gradient`` is dE/daxes.

        Fields (magnetic models): ``magnetic_field`` in Tesla, (3,) or (N, 3),
        adds the Zeeman term ``-mu_B B.(L + g/2 sigma)`` (the orbital part in
        noncollinear models only); ``spin_field`` (eV, ``h_i.sigma``) and
        ``orbital_field`` (eV, ``b_i.L``) are its lower-level sources, as in
        bopjax.  With gradients the result holds the magnetic moments in
        Bohr magnetons, ``-(1/mu_B) dE/dB``: ``spin_moment`` (g/2 <sigma>),
        ``orbital_moment`` (<L>) and ``magnetic_moment``, (N, 3) each, as
        ``bopjax.make_magnetic_moment``."""
        pos = np.asarray(positions, dtype=float).tolist()
        sp = [int(s) for s in np.asarray(species).ravel()]
        c = None if cell is None else np.asarray(cell, dtype=float).tolist()
        s = None if shifts is None else np.asarray(shifts, dtype=float).ravel().tolist()
        n = len(pos)
        out = _arrays(self._core.compute(pos, sp, c, s, gradients, _axes(axes),
                                         _field(spin_field, n), _field(orbital_field, n),
                                         _field(magnetic_field, n)))
        return _moments(out)

    def scf(self, positions, species, cell=None, shifts0=None, tol=1e-10, max_iter=200,
            mixing=None, history=None, max_step=0.5, free=None, refresh=4,
            condition="stationary", adjoint_tol=1e-10, reuse_history=False, axes=None,
            magnetic_field=None, spin_field=None, orbital_field=None):
        """Self-consistent onsite shifts by preconditioned Anderson mixing,
        from ``shifts0`` (default zero).  ``condition`` is "stationary"
        (dE/dshifts = 0, BOPfox; ``tol`` on max |dE/dshift|) or "potential"
        (shifts = K N, as ``bopjax.make_potential(condition="potential")``;
        ``tol`` on max |K N - shifts|, forces by implicit differentiation with
        an adjoint solve to ``adjoint_tol``).  ``mixing`` and ``history``
        default to 1.0 and 24 (stationary) or 0.3 and 8 (potential).
        ``reuse_history`` starts the Anderson mixing with the history of the
        previous ``scf`` call of this object (a nearby configuration, as in
        molecular dynamics; ``clear_history()`` forgets it).
        Returns the dict of ``compute`` at the converged shifts plus
        ``shifts`` (N, nspin, nshell), ``converged``, ``history`` (the
        convergence measure after each evaluation of the Anderson mixing and
        after each Newton step of its fallback), ``residual``,
        ``adjoint_iterations`` and ``evaluations`` (all residual evaluations).  ``free`` (N,) bools
        restricts the self-consistency to those atoms; ``refresh`` is the number
        of gradient evaluations between updates of the preconditioner."""
        pos = np.asarray(positions, dtype=float).tolist()
        sp = [int(s) for s in np.asarray(species).ravel()]
        c = None if cell is None else np.asarray(cell, dtype=float).tolist()
        s = None if shifts0 is None else np.asarray(shifts0, dtype=float).ravel().tolist()
        fr = None if free is None else [bool(f) for f in np.asarray(free).ravel()]
        out = _arrays(self._core.scf(pos, sp, c, s, tol, max_iter, mixing, history, max_step, fr,
                                       refresh, condition, adjoint_tol, reuse_history,
                                       _axes(axes), _field(spin_field, len(pos)),
                                       _field(orbital_field, len(pos)),
                                       _field(magnetic_field, len(pos))))
        out["shifts"] = out["shifts"].reshape(len(pos), -1, len(self.model["basis"]))
        return _moments(out)


G_SPIN = 2.00231930436  # electron spin g-factor
MU_B = 5.7883818060e-5  # Bohr magneton in eV / T


def _field(field, n):
    if field is None:
        return None
    return np.broadcast_to(np.asarray(field, dtype=float), (n, 3)).tolist()


def _moments(out):
    """Magnetic moments -(1/mu_B) dE/dB from the field-source gradients."""
    if "spin_field_gradient" in out:
        out["spin_moment"] = 0.5 * G_SPIN * out["spin_field_gradient"]
        orbital = out.get("orbital_field_gradient", np.zeros_like(out["spin_moment"]))
        out["orbital_moment"] = orbital
        out["magnetic_moment"] = out["spin_moment"] + orbital
    return out


def _axes(axes):
    return None if axes is None else np.asarray(axes, dtype=float).tolist()


def _arrays(out):
    return {k: (v if np.isscalar(v) or isinstance(v, bool) else np.asarray(v))
            for k, v in out.items()}


def _tolist(x):
    return np.asarray(x, dtype=float).tolist()


def model_from_bopjax(cfg, params, radial="powerlaw") -> dict:
    """Model dict from a ``bopjax.BOPConfig`` and a bopjax parameter dict.

    For ``spin="noncollinear"`` the spin axes are not part of the model:
    pass ``params["spin_axes"]`` as ``axes`` to ``compute`` / ``scf``; fields
    (``params["magnetic_field"]`` etc.) likewise go to ``compute`` / ``scf``.

    ``radial`` names the bond-integral form: "powerlaw"
    (``bopjax.hamiltonian.powerlaw``), "v0" (``v0_bond``) or "exp_sum"
    (``pairwise.exp_sum``); a function is accepted and mapped by name.
    """
    if callable(radial):
        radial = {"powerlaw": "powerlaw", "v0_bond": "v0", "exp_sum": "exp_sum"}[radial.__name__]
    unsupported = [name for name, on in [
        ("screening", cfg.screening), ("yukawa", cfg.yukawa), ("kT > 0", cfg.kT > 0),
        ("coulomb != diagonal", cfg.coulomb != "diagonal"),
        ("Beer-Pettifor-Aoki bandwidth", cfg.bandwidth.startswith("beerpettiforaoki"))] if on]
    if unsupported:
        raise NotImplementedError("not in bopfox-rs yet: " + ", ".join(unsupported))
    model = {
        "nmom": cfg.nmom, "nexp": cfg.nexp, "bandwidth": cfg.bandwidth,
        "global_ab": list(cfg.global_ab), "terminator": cfg.terminator,
        "alpha_weight": cfg.alpha_weight, "kernel": cfg.kernel,
        "rcut": cfg.rcut, "dcut": cfg.dcut, "rcut2": cfg.rcut2, "dcut2": cfg.dcut2,
        "basis": list(cfg.basis), "spin": cfg.spin, "radial": radial,
        "onsite": _tolist(params["onsite"]), "n0": _tolist(params["n0"]),
        "hubbard": _tolist(params["hubbard"]),
        "bond": {k: _tolist(v) for k, v in params["bond"].items()},
    }
    if cfg.magnetic:
        model["stoner"] = _tolist(params["stoner"])
    if cfg.spin_orbit:
        model["spin_orbit"] = True
        model["spin_orbit_xi"] = _tolist(params["spin_orbit"])
    if cfg.repulsion is not None:
        from bopjax import pairwise as pw
        form = {id(pw.V0): "v0", id(pw.POWERLAW): "powerlaw", id(pw.EGOROV): "egorov"}.get(
            id(cfg.repulsion))
        if form is None:
            raise NotImplementedError("unknown repulsion form")
        rp = params["repulsion"]
        model["repulsion"] = {"form": form, "pair": _tolist(rp["pair"]),
                              "density": _tolist(rp.get("density", [])),
                              "embed": _tolist(rp.get("embed", []))}
    return model
