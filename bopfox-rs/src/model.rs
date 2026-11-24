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
    Beerpettiforaoki,
    #[serde(rename = "beerpettiforaoki_bopfox")]
    BeerpettiforaokiBopfox,
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
    /// iron-carbon (`bopjax.models.FECARBON_REPULSION`): Egorov pair and core
    /// term times a cosine taper `p[5], p[6]`; density Gaussian plus
    /// exponential `p0 exp(-p1 (r - p2)^2) + p3 exp(-p4 r)` times a taper
    /// `p[5], p[6]`; power-law embedding
    Fecarbon,
}

/// 1 below `rc - dc`, 0 above `rc`, half cosine in between (`models._taper`).
fn taper<T: Real>(r: T, rc: f64, dc: f64) -> T {
    crate::hamiltonian::cosine_cutoff(r, rc, dc)
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
            RepulsionForm::Egorov => egorov_pair(p, r),
            RepulsionForm::Fecarbon => egorov_pair(p, r) * taper(r, p[5], p[6]),
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
            RepulsionForm::Fecarbon => {
                let d = r - p[2];
                let mut rho = T::zero();
                if p[0] != 0.0 {
                    rho += (-(d * d) * p[1]).exp() * p[0];
                }
                if p[3] != 0.0 {
                    rho += (-r * p[4]).exp() * p[3];
                }
                rho * taper(r, p[5], p[6])
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

fn egorov_pair<T: Real>(p: &[f64], r: T) -> T {
    let mut e = (-r.powf(p[2]) * p[1]).exp() * p[0];
    if r.val() < p[3] {
        let d = -r + p[3];
        e += d * d * d / r * d.exp() * p[4];
    }
    e
}

/// Shells and expansion settings of one species (`bopjax.SpeciesSettings`):
/// `basis` is the subset of the model's basis this species carries (no
/// hopping into the others, no electrons in them); entries left out take
/// the model's values.  `nmom` may not exceed the model's, which sets how far
/// the moments are walked.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeciesSettings {
    pub basis: Vec<Shell>,
    #[serde(default)]
    pub nmom: Option<usize>,
    #[serde(default)]
    pub nexp: Option<usize>,
    #[serde(default)]
    pub bandwidth: Option<Bandwidth>,
    #[serde(default)]
    pub terminator: Option<Terminator>,
    #[serde(default)]
    pub kernel: Option<Kernel>,
    #[serde(default)]
    pub bandwidth_guard: Option<bool>,
    /// false: one onsite shift for both spin channels (magnetic models)
    #[serde(default = "default_true")]
    pub spin_polarized: bool,
    /// one onsite shift for all shells of the species
    #[serde(default)]
    pub atomic_shifts: bool,
}

fn default_true() -> bool {
    true
}

/// Coulomb interaction between the charges of different atoms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Coulomb {
    /// onsite Hubbard J only
    Diagonal,
    /// Frauenheim integrals between atoms with a polynomial taper
    Frauenheim,
}

/// Environment-dependent Yukawa repulsion (Znam et al.; form of Egorov et al.):
/// `E_i = 1/2 sum_j B / r exp(-lambda_ij (r - 2 r_core))`,
/// `lambda_ij = (lambda_i + lambda_j) / 2`,
/// `lambda_i = lambda0 + (sum_l C exp(-upsilon r_il))^(1/m)`, all with the
/// cutoff of the second list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Yukawa {
    #[serde(rename = "B")]
    pub b: Vec<Vec<f64>>,
    pub r_core: Vec<Vec<f64>>,
    #[serde(rename = "C")]
    pub c: Vec<Vec<f64>>,
    pub upsilon: Vec<Vec<f64>>,
    pub lambda0: Vec<f64>,
    pub m: Vec<f64>,
}

/// Parameters of the d-d screening (`bopjax` `params["screening"]`): the
/// sd-sigma bond integral and overlaps; for "sds*" the entry [a][b] has the
/// s orbital on species a.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScreeningParams {
    pub sds: ChannelParams,
    pub sds_overlap: ChannelParams,
    pub dds_overlap: ChannelParams,
    pub ddp_overlap: ChannelParams,
    pub ddd_overlap: ChannelParams,
}

fn default_coulomb() -> Coulomb {
    Coulomb::Diagonal
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
fn default_nodes() -> usize {
    24
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
    /// widen b_inf where the scaled recursion chain has eigenvalues outside
    /// [-1, 1] (`bopjax.expansion.guarded_width`)
    #[serde(default)]
    pub bandwidth_guard: bool,
    /// per-species shells and expansion settings (None: every species has
    /// the whole basis and the settings above)
    #[serde(default)]
    pub species: Option<Vec<SpeciesSettings>>,
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
    /// electronic temperature (eV); the energy is then the Mermin free energy
    #[serde(default, rename = "kT")]
    pub kt: f64,
    /// quadrature nodes of the Fermi kernel (kT > 0)
    #[serde(default = "default_nodes")]
    pub smearing_nodes: usize,
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
    #[serde(default)]
    pub yukawa: Option<Yukawa>,
    /// screening of the d-d bond integrals
    #[serde(default)]
    pub screening: Option<ScreeningParams>,
    /// radial form of the screening integrals (default: `radial`)
    #[serde(default)]
    pub screening_radial: Option<Radial>,
    #[serde(default = "default_coulomb")]
    pub coulomb: Coulomb,
    /// taper R1, R2 of the intersite Coulomb integrals [species][species]
    #[serde(default)]
    pub taper: Vec<Vec<[f64; 2]>>,
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

    /// The expansion settings of species `s` as a model (nmom, nexp,
    /// bandwidth, terminator, kernel and guard; everything else as `self`).
    pub fn species_model(&self, s: usize) -> Model {
        let mut m = self.clone();
        m.species = None;
        if let Some(st) = self.species.as_ref().map(|v| &v[s]) {
            m.nmom = st.nmom.unwrap_or(self.nmom);
            m.nexp = match (st.nexp, st.nmom) {
                (Some(n), _) => Some(n),
                (None, None) => self.nexp,
                (None, Some(_)) => None,
            };
            m.bandwidth = st.bandwidth.unwrap_or(self.bandwidth);
            m.terminator = st.terminator.unwrap_or(self.terminator);
            m.kernel = st.kernel.unwrap_or(self.kernel);
            m.bandwidth_guard = st.bandwidth_guard.unwrap_or(self.bandwidth_guard);
        }
        m
    }

    /// Whether species `s` carries shell `k` of the basis.
    pub fn has_shell(&self, s: usize, k: usize) -> bool {
        match &self.species {
            None => true,
            Some(v) => v[s].basis.contains(&self.basis[k]),
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
        if self.kt < 0.0 {
            return Err("kT must not be negative".into());
        }
        if (self.repulsion.is_some() || self.yukawa.is_some() || self.coulomb != Coulomb::Diagonal)
            && self.rcut2.is_none()
        {
            return Err("repulsion, Yukawa and intersite Coulomb need rcut2".into());
        }
        if let Some(st) = &self.species {
            if st.len() < ns {
                return Err("species settings needed for every species".into());
            }
            for x in st {
                if !x.basis.iter().all(|b| self.basis.contains(b)) {
                    return Err("a species' basis must be part of the model basis".into());
                }
                if x.nmom.is_some_and(|n| n > self.nmom) {
                    return Err("a species' nmom may not exceed the model's".into());
                }
            }
            let ties = st.iter().any(|x| x.atomic_shifts || !x.spin_polarized);
            if ties && self.spin == Spin::Noncollinear {
                return Err("tied shifts are not supported with noncollinear spin".into());
            }
        }
        if self.screening.is_some() && !self.basis.contains(&Shell::D) {
            return Err("screening acts on d-d bond integrals; the basis has no d shell".into());
        }
        if self.coulomb == Coulomb::Frauenheim && self.taper.len() != ns {
            return Err("intersite Coulomb needs taper [species][species][R1, R2]".into());
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
