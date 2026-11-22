//! Model description: settings of the expansion and parameters per species.
//! Serialised as JSON; the field names follow `bopjax.BOPConfig` and the
//! bopjax parameter dictionaries.

use crate::hamiltonian::{BondParams, ChannelParams, Radial, Shell};
use crate::real::Real;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bandwidth {
    Azero,
    Lowest,
    Highest,
    AvganMaxbn,
    Global,
    Averaged,
    Findeminemax,
    Gershgorin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Terminator {
    None,
    Constantabn,
    Oscillation,
    Weighted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kernel {
    None,
    Jackson,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Spin {
    None,
    Collinear,
    /// local spin axes per atom (`Bop::with_axes`), spinor Hamiltonian
    Noncollinear,
}

/// Functional form of the pair repulsion and embedding (bopjax.pairwise).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepulsionForm {
    /// exponential pair term and density, power-law embedding (BOPfox "v0")
    V0,
    /// pair term `p0 (p2 / r)^p1` only
    Powerlaw,
    /// Egorov et al.: exponential + core pair term, Gaussian densities
    Egorov,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Repulsion {
    pub form: RepulsionForm,
    pub pair: ChannelParams,
    #[serde(default)]
    pub density: ChannelParams,
    #[serde(default)]
    pub embed: Vec<Vec<f64>>, // [species][npar]
}

impl Repulsion {
    pub fn pair<T: Real>(&self, si: usize, sj: usize, r: T) -> T {
        let p = &self.pair[si][sj];
        match self.form {
            RepulsionForm::V0 => (-(r / p[2] - 1.0) * p[1]).exp() * p[0],
            RepulsionForm::Powerlaw => (r.powf(-1.0) * p[2]).powf(p[1]) * p[0],
            RepulsionForm::Egorov => {
                let mut e = (-r.powf(p[2]) * p[1]).exp() * p[0];
                if r.val() < p[3] {
                    let d = -r + p[3];
                    e += d * d * d / r * d.exp() * p[4];
                }
                e
            }
        }
    }

    pub fn has_embedding(&self) -> bool {
        self.form != RepulsionForm::Powerlaw
    }

    pub fn density<T: Real>(&self, si: usize, sj: usize, r: T) -> T {
        let p = &self.density[si][sj];
        match self.form {
            RepulsionForm::V0 => (-(r / p[2] - 1.0) * p[1]).exp() * p[0],
            RepulsionForm::Powerlaw => T::zero(),
            RepulsionForm::Egorov => {
                let d = r - p[2];
                (-(d * d) * p[1]).exp() * p[0]
            }
        }
    }

    pub fn embed<T: Real>(&self, s: usize, rho: T) -> T {
        if self.form == RepulsionForm::Powerlaw || rho.val() <= 0.0 {
            return T::zero();
        }
        let p = &self.embed[s];
        rho.powf(p[1]) * p[0]
    }
}

fn default_bandwidth() -> Bandwidth {
    Bandwidth::Lowest
}
fn default_terminator() -> Terminator {
    Terminator::None
}
fn default_kernel() -> Kernel {
    Kernel::None
}
fn default_spin() -> Spin {
    Spin::None
}
fn default_radial() -> Radial {
    Radial::Powerlaw
}
fn default_one() -> f64 {
    1.0
}
fn default_global() -> (f64, f64) {
    (0.0, 1.0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    // --- settings (bopjax.BOPConfig) ---
    pub nmom: usize,
    #[serde(default)]
    pub nexp: Option<usize>,
    #[serde(default = "default_bandwidth")]
    pub bandwidth: Bandwidth,
    #[serde(default = "default_global")]
    pub global_ab: (f64, f64),
    #[serde(default = "default_terminator")]
    pub terminator: Terminator,
    #[serde(default = "default_one")]
    pub alpha_weight: f64,
    #[serde(default = "default_kernel")]
    pub kernel: Kernel,
    pub rcut: f64,
    #[serde(default)]
    pub dcut: f64,
    #[serde(default)]
    pub rcut2: Option<f64>,
    #[serde(default)]
    pub dcut2: f64,
    pub basis: Vec<Shell>,
    #[serde(default = "default_spin")]
    pub spin: Spin,
    #[serde(default = "default_radial")]
    pub radial: Radial,
    // --- parameters, indexed by species ---
    pub onsite: Vec<Vec<f64>>, // [species][shell]
    pub n0: Vec<Vec<f64>>,     // [species][shell]
    pub hubbard: Vec<f64>,     // [species]
    #[serde(default)]
    pub stoner: Vec<Vec<f64>>, // [species][shell]
    /// onsite spin-orbit coupling xi L.S (spin = noncollinear only)
    #[serde(default)]
    pub spin_orbit: bool,
    /// spin-orbit constants xi [species][shell]
    #[serde(default)]
    pub spin_orbit_xi: Vec<Vec<f64>>,
    pub bond: BTreeMap<String, ChannelParams>,
    #[serde(default)]
    pub repulsion: Option<Repulsion>,
}

impl BondParams for Model {
    fn channel(&self, name: &str) -> &ChannelParams {
        self.bond
            .get(name)
            .unwrap_or_else(|| panic!("missing bond integral channel {name:?}"))
    }
}

impl Model {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let m: Model = serde_json::from_str(text).map_err(|e| e.to_string())?;
        m.validate()?;
        Ok(m)
    }

    pub fn n_expansion(&self) -> usize {
        self.nexp.map_or(self.nmom, |n| n.max(self.nmom))
    }

    pub fn nspin(&self) -> usize {
        match self.spin {
            Spin::None => 1,
            Spin::Collinear | Spin::Noncollinear => 2,
        }
    }

    pub fn nspecies(&self) -> usize {
        self.onsite.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.nmom < 2 {
            return Err("nmom must be at least 2".into());
        }
        if self.nmom + 2 > crate::expansion::MAX_DERIV {
            return Err(format!(
                "nmom up to {} supported",
                crate::expansion::MAX_DERIV - 2
            ));
        }
        let ns = self.nspecies();
        if self.n0.len() != ns || self.hubbard.len() != ns {
            return Err("onsite, n0 and hubbard need one entry per species".into());
        }
        if self.spin != Spin::None && self.stoner.len() != ns {
            return Err("magnetism needs stoner parameters".into());
        }
        if self.spin_orbit && self.spin != Spin::Noncollinear {
            return Err("spin-orbit coupling needs spin = noncollinear".into());
        }
        if self.spin_orbit && self.spin_orbit_xi.len() != ns {
            return Err("spin-orbit coupling needs spin_orbit_xi per species".into());
        }
        if self.repulsion.is_some() && self.rcut2.is_none() {
            return Err("repulsion needs rcut2".into());
        }
        for (i, la) in self.basis.iter().enumerate() {
            for lb in &self.basis[i..] {
                let (a, b) = if la.l() <= lb.l() {
                    (*la, *lb)
                } else {
                    (*lb, *la)
                };
                for c in crate::hamiltonian::channels(a, b) {
                    if !self.bond.contains_key(*c) {
                        return Err(format!("missing bond integral channel {c:?}"));
                    }
                }
            }
        }
        Ok(())
    }
}
