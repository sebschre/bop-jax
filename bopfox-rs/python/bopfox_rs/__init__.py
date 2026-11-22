"""Analytic bond-order potentials in Rust (energies and analytic forces).

``Potential(model)`` takes a model as a JSON string or a dict (see
``model_from_bopjax`` to convert a bopjax configuration and parameter set).
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

    def compute(self, positions, species, cell=None, shifts=None, gradients=True):
        """Dict of numpy arrays: energy, energy_atom, fermi, n_shell, charge,
        moments, a_inf, b_inf (and moment when magnetic); with ``gradients``
        also forces (-dE/dpositions), virial (dE/dstrain) and shift_gradient
        (dE/dshifts, shaped like the shifts: (N, nspin, nshell))."""
        pos = np.asarray(positions, dtype=float).tolist()
        sp = [int(s) for s in np.asarray(species).ravel()]
        c = None if cell is None else np.asarray(cell, dtype=float).tolist()
        s = None if shifts is None else np.asarray(shifts, dtype=float).ravel().tolist()
        out = self._core.compute(pos, sp, c, s, gradients)
        return {k: (v if np.isscalar(v) else np.asarray(v)) for k, v in out.items()}


def _tolist(x):
    return np.asarray(x, dtype=float).tolist()


def model_from_bopjax(cfg, params, radial="powerlaw") -> dict:
    """Model dict from a ``bopjax.BOPConfig`` and a bopjax parameter dict.

    ``radial`` names the bond-integral form: "powerlaw"
    (``bopjax.hamiltonian.powerlaw``), "v0" (``v0_bond``) or "exp_sum"
    (``pairwise.exp_sum``); a function is accepted and mapped by name.
    """
    if callable(radial):
        radial = {"powerlaw": "powerlaw", "v0_bond": "v0", "exp_sum": "exp_sum"}[radial.__name__]
    unsupported = [name for name, on in [
        ("noncollinear spin", cfg.spin == "noncollinear"), ("spin_orbit", cfg.spin_orbit),
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
