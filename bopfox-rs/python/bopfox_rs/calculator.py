"""ASE calculator for bopfox-rs, with warm-started self-consistency.

    from bopfox_rs.calculator import BOPCalculator
    calc = BOPCalculator.from_bopjax(cfg, params, hamiltonian.v0_bond)
    atoms.calc = calc

Every calculation solves the self-consistency (``condition`` "stationary",
BOPfox's dE/dshifts = 0, or "potential", shifts = K N) and returns energy,
forces and stress of the self-consistent state.  The shifts of the previous
calculation are the starting point of the next one (``warm_start``); with
``extrapolation="linear"`` the last two are extrapolated, which suits
molecular dynamics, and the Anderson mixing continues with the history of
the previous solve (it already describes how the shifts respond).  The first calculation, or one after the number or kind
of atoms changed, starts from the initial magnetic moments of the atoms
(an exchange splitting -/+ I m / 2 of the levels).
"""

from __future__ import annotations

import json

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from . import Potential, model_from_bopjax


class SCFNotConverged(RuntimeError):
    pass


class BOPCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress",
                              "magmom", "magmoms", "charges"]

    def __init__(self, model, species=None, condition="stationary", tol=1e-8,
                 warm_start=True, extrapolation="linear", scf_options=None,
                 raise_on_failure=True, axes=None, magnetic_field=None, **kwargs):
        """
        model         bopfox-rs model (dict, JSON string or ``Potential``)
        species       chemical symbol -> species index of the model; may be
                      omitted for a one-species model
        condition     "stationary" or "potential"
        tol           self-consistency tolerance (eV)
        warm_start    start from the shifts of the previous calculation
        extrapolation "none" or "linear" (from the last two calculations)
        scf_options   further keyword arguments of ``Potential.scf``
        axes          (N, 3) local spin axes of a noncollinear model; by
                      default the directions of the initial magnetic moments
                      (vectors), or z
        magnetic_field  external field in Tesla, (3,) or (N, 3): Zeeman term
                      -mu_B B.(L + g/2 sigma) (magnetic models; the orbital
                      part in noncollinear ones).  ``results`` then also hold
                      ``spin_moment``, ``orbital_moment`` and
                      ``magnetic_moment`` (N, 3), -(1/mu_B) dE/dB in Bohr
                      magnetons (these are there at zero field too).

        For a noncollinear model ``magmoms`` are vectors along the axes and
        ``results["axes_gradient"]`` holds dE/d(axes) (eV per unit of the
        axis vector; the torque on spin i is ``-e_i x dE/de_i``).
        """
        super().__init__(**kwargs)
        self.pot = model if isinstance(model, Potential) else Potential(model)
        self.species = species
        self.condition = condition
        self.tol = tol
        self.warm_start = warm_start
        if extrapolation not in ("none", "linear"):
            raise ValueError("extrapolation must be 'none' or 'linear'")
        self.extrapolation = extrapolation
        self.scf_options = dict(scf_options or {})
        self.raise_on_failure = raise_on_failure
        self.axes = None if axes is None else np.asarray(axes, dtype=float)
        self.noncollinear = self.pot.model.get("spin", "none") == "noncollinear"
        self.magnetic_field = (None if magnetic_field is None
                               else np.asarray(magnetic_field, dtype=float))
        self._previous = []  # shifts of the last calculations, newest last
        self._key = None
        self.scf_info = {}  # evaluations, residual, converged of the last calculation

    @classmethod
    def from_bopjax(cls, cfg, params, radial="powerlaw", **kwargs):
        """Calculator for a ``bopjax.BOPConfig`` and parameter dict."""
        return cls(model_from_bopjax(cfg, params, radial), **kwargs)

    def reset(self):
        super().reset()
        self._previous = []
        self.pot.clear_history()

    def _species(self, atoms):
        if self.species is None:
            if len(self.pot.model["onsite"]) != 1:
                raise ValueError("pass species={symbol: index} for a multi-species model")
            return np.zeros(len(atoms), dtype=int)
        return np.array([self.species[s] for s in atoms.get_chemical_symbols()])

    def _initial_shifts(self, atoms):
        model = self.pot.model
        n, nsh = len(atoms), len(model["basis"])
        if model.get("spin", "none") == "none":
            return np.zeros((n, 1, nsh))
        shifts = np.zeros((n, 2, nsh))
        stoner = np.asarray(model["stoner"], dtype=float)[self._species(atoms)]  # (n, nshell)
        moments = atoms.get_initial_magnetic_moments()
        if moments.ndim > 1:
            if not self.noncollinear:
                raise ValueError("collinear model: initial magnetic moments must be scalars")
            moments = np.linalg.norm(moments, axis=1)
        shell = np.argmax(stoner, axis=1)  # the moment goes on the most magnetic shell
        split = 0.5 * stoner[np.arange(n), shell] * moments
        shifts[np.arange(n), 0, shell] = -split
        shifts[np.arange(n), 1, shell] = split
        return shifts

    def _axes(self, atoms):
        if not self.noncollinear:
            return None
        if self.axes is not None:
            return self.axes
        m = atoms.get_initial_magnetic_moments()
        axes = np.tile([0.0, 0.0, 1.0], (len(atoms), 1))
        if m.ndim > 1:
            norm = np.linalg.norm(m, axis=1)
            axes[norm > 0] = m[norm > 0] / norm[norm > 0, None]
        return axes

    def _start(self, atoms):
        key = (tuple(atoms.numbers),)
        if key != self._key or not self.warm_start:
            self._previous = []
            self._key = key
            self.pot.clear_history()
        if not self._previous:
            return self._initial_shifts(atoms)
        if self.extrapolation == "linear" and len(self._previous) == 2:
            return 2.0 * self._previous[1] - self._previous[0]
        return self._previous[-1]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        pbc = atoms.pbc
        if pbc.any() and not pbc.all():
            raise NotImplementedError("mixed periodic and open boundaries")
        cell = np.asarray(atoms.cell) if pbc.all() else None
        shifts0 = self._start(atoms)
        out = self.pot.scf(atoms.positions, self._species(atoms), cell, shifts0, tol=self.tol,
                           condition=self.condition, reuse_history=bool(self._previous),
                           axes=self._axes(atoms), magnetic_field=self.magnetic_field,
                           **self.scf_options)
        self.scf_info = {"evaluations": len(out["history"]), "residual": out["residual"],
                         "converged": out["converged"],
                         "adjoint_iterations": out["adjoint_iterations"]}
        if not out["converged"]:
            self._previous = []
            self.pot.clear_history()
            if self.raise_on_failure:
                raise SCFNotConverged(f"self-consistency not converged: {self.scf_info}")
        else:
            self._previous = (self._previous + [out["shifts"]])[-2:]
        self.results["energy"] = self.results["free_energy"] = out["energy"]
        self.results["energies"] = out["energy_atom"]
        self.results["forces"] = out["forces"]
        self.results["charges"] = out["charge"]
        if self.noncollinear:
            axes = self._axes(atoms)
            e = axes / np.linalg.norm(axes, axis=1, keepdims=True)
            self.results["magmoms"] = out["moment"][:, None] * e
            self.results["magmom"] = self.results["magmoms"].sum(axis=0)
            self.results["axes_gradient"] = out["axes_gradient"]
        elif "moment" in out:
            self.results["magmoms"] = out["moment"]
            self.results["magmom"] = float(np.sum(out["moment"]))
        else:
            self.results["magmoms"] = np.zeros(len(atoms))
            self.results["magmom"] = 0.0
        for key in ("spin_moment", "orbital_moment", "magnetic_moment"):
            if key in out:
                self.results[key] = out[key]
        if cell is not None:
            stress = out["virial"] / atoms.get_volume()
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)

    def todict(self):
        return {"model": json.dumps(self.pot.model), "condition": self.condition, "tol": self.tol}
