//! Python bindings (feature "python"): `bopfox_rs._core.Potential`.

use crate::{Bop, Model, Topology};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(module = "bopfox_rs._core")]
struct Potential {
    model: Model,
}

#[pymethods]
impl Potential {
    #[new]
    fn new(model_json: &str) -> PyResult<Self> {
        Ok(Potential {
            model: Model::from_json(model_json).map_err(PyValueError::new_err)?,
        })
    }

    /// Shift entries per atom (nspin * nshell).
    #[getter]
    fn shifts_per_atom(&self) -> usize {
        self.model.nspin() * self.model.basis.len()
    }

    /// Energy (and with `gradients` forces, virial, dE/dshifts) for positions
    /// (N, 3), species (N,), cell (3, 3) rows = lattice vectors or None, and
    /// onsite shifts flattened as [atom][spin][shell] (default zero).
    #[pyo3(signature = (positions, species, cell=None, shifts=None, gradients=true))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        positions: Vec<[f64; 3]>,
        species: Vec<usize>,
        cell: Option<[[f64; 3]; 3]>,
        shifts: Option<Vec<f64>>,
        gradients: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if species.len() != positions.len() {
            return Err(PyValueError::new_err(
                "species and positions differ in length",
            ));
        }
        if let Some(&s) = species.iter().max() {
            if s >= self.model.nspecies() {
                return Err(PyValueError::new_err("species index out of range"));
            }
        }
        let n_shift = positions.len() * self.shifts_per_atom();
        let shifts = shifts.unwrap_or_else(|| vec![0.0; n_shift]);
        if shifts.len() != n_shift {
            return Err(PyValueError::new_err(format!("expected {n_shift} shifts")));
        }
        let m = &self.model;
        let r = py.allow_threads(|| {
            let topo =
                Topology::build(&positions, &species, cell.as_ref(), m.rcut, m.nmom, m.rcut2);
            Bop::new(m, &topo).compute(&positions, &cell, &shifts, gradients)
        });
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
        Ok(d)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Potential>()?;
    Ok(())
}
