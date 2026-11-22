//! Python bindings (feature "python"): `bopfox_rs._core.Potential`.

use crate::{Bop, Condition, MixingHistory, Model, ScfOptions, Topology};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Mutex;

#[pyclass(module = "bopfox_rs._core")]
struct Potential {
    model: Model,
    /// Anderson history of the last `scf` call
    mixing: Mutex<Option<MixingHistory>>,
}

#[pymethods]
impl Potential {
    #[new]
    fn new(model_json: &str) -> PyResult<Self> {
        Ok(Potential {
            model: Model::from_json(model_json).map_err(PyValueError::new_err)?,
            mixing: Mutex::new(None),
        })
    }

    /// Forget the Anderson history kept for `scf(..., reuse_history=True)`.
    fn clear_history(&self) {
        *self.mixing.lock().unwrap() = None;
    }

    /// Shift entries per atom (nspin * nshell).
    #[getter]
    fn shifts_per_atom(&self) -> usize {
        self.model.nspin() * self.model.basis.len()
    }

    /// Energy (and with `gradients` forces, virial, dE/dshifts) for positions
    /// (N, 3), species (N,), cell (3, 3) rows = lattice vectors or None, and
    /// onsite shifts flattened as [atom][spin][shell] (default zero).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (positions, species, cell=None, shifts=None, gradients=true, axes=None,
                        spin_field=None, orbital_field=None, magnetic_field=None))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        positions: Vec<[f64; 3]>,
        species: Vec<usize>,
        cell: Option<[[f64; 3]; 3]>,
        shifts: Option<Vec<f64>>,
        gradients: bool,
        axes: Option<Vec<[f64; 3]>>,
        spin_field: Option<Vec<[f64; 3]>>,
        orbital_field: Option<Vec<[f64; 3]>>,
        magnetic_field: Option<Vec<[f64; 3]>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.check(&positions, &species)?;
        if axes.as_ref().is_some_and(|a| a.len() != positions.len()) {
            return Err(PyValueError::new_err("axes needs one entry per atom"));
        }
        self.check_fields(
            positions.len(),
            &spin_field,
            &orbital_field,
            &magnetic_field,
        )?;
        let n_shift = positions.len() * self.shifts_per_atom();
        let shifts = shifts.unwrap_or_else(|| vec![0.0; n_shift]);
        if shifts.len() != n_shift {
            return Err(PyValueError::new_err(format!("expected {n_shift} shifts")));
        }
        let m = &self.model;
        let r = py.allow_threads(|| {
            let topo =
                Topology::build(&positions, &species, cell.as_ref(), m.rcut, m.nmom, m.rcut2);
            let bop = with_fields(
                with_axes(Bop::new(m, &topo), &axes),
                &spin_field,
                &orbital_field,
                &magnetic_field,
            );
            bop.compute(&positions, &cell, &shifts, gradients)
        });
        results_dict(py, r, gradients)
    }

    /// Self-consistent shifts (dE/dshifts = 0) from `shifts0` (default zero),
    /// and everything at them, forces included.  `free` (N,) bools restricts
    /// the self-consistency to those atoms.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (positions, species, cell=None, shifts0=None, tol=1e-10, max_iter=200,
                        mixing=None, history=None, max_step=0.5, free=None, refresh=4,
                        condition="stationary", adjoint_tol=1e-10, reuse_history=false, axes=None,
                        spin_field=None, orbital_field=None, magnetic_field=None))]
    fn scf<'py>(
        &self,
        py: Python<'py>,
        positions: Vec<[f64; 3]>,
        species: Vec<usize>,
        cell: Option<[[f64; 3]; 3]>,
        shifts0: Option<Vec<f64>>,
        tol: f64,
        max_iter: usize,
        mixing: Option<f64>,
        history: Option<usize>,
        max_step: f64,
        free: Option<Vec<bool>>,
        refresh: usize,
        condition: &str,
        adjoint_tol: f64,
        reuse_history: bool,
        axes: Option<Vec<[f64; 3]>>,
        spin_field: Option<Vec<[f64; 3]>>,
        orbital_field: Option<Vec<[f64; 3]>>,
        magnetic_field: Option<Vec<[f64; 3]>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let condition = match condition {
            "stationary" => Condition::Stationary,
            "potential" => Condition::Potential,
            c => return Err(PyValueError::new_err(format!("unknown condition {c:?}"))),
        };
        self.check(&positions, &species)?;
        if axes.as_ref().is_some_and(|a| a.len() != positions.len()) {
            return Err(PyValueError::new_err("axes needs one entry per atom"));
        }
        self.check_fields(
            positions.len(),
            &spin_field,
            &orbital_field,
            &magnetic_field,
        )?;
        let n_shift = positions.len() * self.shifts_per_atom();
        if shifts0.as_ref().is_some_and(|s| s.len() != n_shift) {
            return Err(PyValueError::new_err(format!("expected {n_shift} shifts")));
        }
        if free.as_ref().is_some_and(|f| f.len() != positions.len()) {
            return Err(PyValueError::new_err("free needs one entry per atom"));
        }
        let base = match condition {
            Condition::Stationary => ScfOptions::default(),
            Condition::Potential => ScfOptions::potential(),
        };
        let opt = ScfOptions {
            tol,
            max_iter,
            mixing: mixing.unwrap_or(base.mixing),
            history: history.unwrap_or(base.history),
            max_step,
            free,
            refresh,
            condition,
            adjoint_tol,
            newton_fallback: true,
        };
        let m = &self.model;
        let previous = if reuse_history {
            self.mixing.lock().unwrap().clone()
        } else {
            None
        };
        let r = py.allow_threads(|| {
            let topo =
                Topology::build(&positions, &species, cell.as_ref(), m.rcut, m.nmom, m.rcut2);
            with_fields(
                with_axes(Bop::new(m, &topo), &axes),
                &spin_field,
                &orbital_field,
                &magnetic_field,
            )
            .scf_continued(
                &positions,
                &cell,
                shifts0.as_deref(),
                &opt,
                previous.as_ref(),
            )
        });
        *self.mixing.lock().unwrap() = Some(r.mixing.clone());
        let d = results_dict(py, r.results, true)?;
        d.set_item("shifts", r.shifts)?;
        d.set_item("converged", r.converged)?;
        d.set_item("history", r.history)?;
        d.set_item("residual", r.residual)?;
        d.set_item("adjoint_iterations", r.adjoint_iterations)?;
        d.set_item("evaluations", r.evaluations)?;
        Ok(d)
    }
}

fn with_fields<'a>(
    bop: Bop<'a>,
    spin: &Option<Vec<[f64; 3]>>,
    orbital: &Option<Vec<[f64; 3]>>,
    field: &Option<Vec<[f64; 3]>>,
) -> Bop<'a> {
    let bop = bop.with_fields(spin.as_deref(), orbital.as_deref());
    match field {
        Some(f) => bop.with_magnetic_field(f),
        None => bop,
    }
}

fn with_axes<'a>(bop: Bop<'a>, axes: &Option<Vec<[f64; 3]>>) -> Bop<'a> {
    match axes {
        Some(a) => bop.with_axes(a),
        None => bop,
    }
}

impl Potential {
    fn check_fields(
        &self,
        n: usize,
        spin: &Option<Vec<[f64; 3]>>,
        orbital: &Option<Vec<[f64; 3]>>,
        field: &Option<Vec<[f64; 3]>>,
    ) -> PyResult<()> {
        for f in [spin, orbital, field].into_iter().flatten() {
            if f.len() != n {
                return Err(PyValueError::new_err("fields need one vector per atom"));
            }
        }
        if (spin.is_some() || field.is_some()) && self.model.nspin() != 2 {
            return Err(PyValueError::new_err("a field needs a magnetic model"));
        }
        if orbital.is_some() && self.model.spin != crate::model::Spin::Noncollinear {
            return Err(PyValueError::new_err(
                "an orbital field needs spin = noncollinear",
            ));
        }
        Ok(())
    }

    fn check(&self, positions: &[[f64; 3]], species: &[usize]) -> PyResult<()> {
        if species.len() != positions.len() {
            return Err(PyValueError::new_err(
                "species and positions differ in length",
            ));
        }
        if species.iter().any(|&s| s >= self.model.nspecies()) {
            return Err(PyValueError::new_err("species index out of range"));
        }
        Ok(())
    }
}

fn results_dict(py: Python<'_>, r: crate::Results, gradients: bool) -> PyResult<Bound<'_, PyDict>> {
    {
        let d = PyDict::new(py);
        d.set_item("energy", r.energy)?;
        d.set_item("energy_atom", r.energy_atom)?;
        d.set_item("fermi", r.fermi)?;
        d.set_item("n_shell", r.n_shell)?;
        d.set_item("charge", r.charge)?;
        d.set_item("moments", r.moments)?;
        d.set_item("a_inf", r.a_inf)?;
        d.set_item("b_inf", r.b_inf)?;
        if !r.moment.is_empty() {
            d.set_item("moment", r.moment)?;
        }
        if gradients {
            d.set_item("forces", r.forces)?;
            d.set_item("virial", r.virial)?;
            d.set_item("shift_gradient", r.shift_gradient)?;
        }
        d.set_item("dos", r.dos)?;
        if !r.axes_gradient.is_empty() {
            d.set_item("axes_gradient", r.axes_gradient)?;
        }
        if !r.spin_field_gradient.is_empty() {
            d.set_item("spin_field_gradient", r.spin_field_gradient)?;
        }
        if !r.orbital_field_gradient.is_empty() {
            d.set_item("orbital_field_gradient", r.orbital_field_gradient)?;
        }
        d.set_item("occupations", r.occupations)?;
        if !r.coulomb_potential.is_empty() {
            d.set_item("coulomb_potential", r.coulomb_potential)?;
        }
        Ok(d)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Potential>()?;
    Ok(())
}
