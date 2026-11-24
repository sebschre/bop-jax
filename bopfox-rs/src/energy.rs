//! Total energy, its gradient with respect to the onsite shifts, forces and
//! virial of the analytic BOP (`bopjax.bop.make_energy_fn`, non-magnetic and
//! collinear).
//!
//!   E = sum U_bond + sum E0 (N - N0) + 1/2 sum J q^2 - 1/4 sum I m^2 + E_rep
//!
//! The Fermi level is fixed by the total electron count.  Its response is
//! handled by a Lagrange multiplier: with `F(mu, ef)` the energy at a given
//! Fermi level, `dE/dmu = dF/dmu - lambda dN/dmu`, `lambda = (dF/def) / (dN/def)`.
//! At self-consistent shifts the energy is stationary in them, so the forces
//! are the partial derivatives returned here.

use crate::complex::{pauli, projector_derivatives, spin_projectors, C64, ZERO};
use crate::expansion::{self, local_derivs, Local};
use crate::geometry::Topology;
use crate::hamiltonian::{angular_momentum, bond_block, cosine_cutoff, shell_slices};
use crate::model::Coulomb;
use crate::model::Model;
use crate::moments::{backward, moments, moments_tangent, walk, HamiltonianData, Walk};
use crate::real::{Dual, Real};
use crate::spinor::{self, SpinorData, SpinorWalk};
use rayon::prelude::*;

pub type Cell = Option<[[f64; 3]; 3]>;

/// Bohr magneton in eV / T
pub const MU_B: f64 = 5.7883818060e-5;
/// electron spin g-factor
pub const G_SPIN: f64 = 2.00231930436;

/// Which derivatives [`Bop::compute_mode`] evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// energy only
    Energy,
    /// energy and dE/dshifts (what the self-consistency needs; about half
    /// the cost of `Full`)
    ShiftGradient,
    /// energy, dE/dshifts, forces and virial
    Full,
    /// as `ShiftGradient`, plus `local_response` (for the self-consistency
    /// preconditioner)
    ScfStep,
    /// energy and `local_response`, no gradients (potential self-consistency)
    Response,
}

/// Walk vectors (f64 entries) kept between the forward and the backward pass;
/// above this they are recomputed (512 MB).
pub const WALK_BUDGET: usize = 1 << 26;

#[derive(Clone, Debug, Default)]
pub struct Results {
    pub energy: f64,
    pub energy_atom: Vec<f64>,
    pub fermi: f64,
    /// electrons per atom and shell (summed over spin)
    pub n_shell: Vec<Vec<f64>>,
    /// electrons per atom, spin and shell
    pub occupations: Vec<Vec<Vec<f64>>>,
    pub charge: Vec<f64>,
    /// magnetic moment per atom (collinear only)
    pub moment: Vec<f64>,
    /// shell moments [atom][spin][shell][k]
    pub moments: Vec<Vec<Vec<Vec<f64>>>>,
    pub a_inf: Vec<Vec<Vec<f64>>>,
    pub b_inf: Vec<Vec<Vec<f64>>>,
    /// intersite Coulomb potential `dE_coul/dq_i` per atom (frauenheim)
    pub coulomb_potential: Vec<f64>,
    /// local density of states at the Fermi level, dN/d(ef) [atom][spin][shell]
    pub dos: Vec<Vec<Vec<f64>>>,
    /// dN_{i s k} / d(level_{i s' k'}) at fixed Fermi level, the response of
    /// an atom's occupations to its own levels, [atom][s][k][s' * nshell + k'] (ScfStep)
    pub local_response: Vec<Vec<Vec<Vec<f64>>>>,
    /// dE/d(spin axes) [atom] (noncollinear; the axes need not be normalised,
    /// so the component along an axis is zero)
    pub axes_gradient: Vec<[f64; 3]>,
    /// dE/dh of the spin field source `h_i.sigma` [atom] (magnetic): `<sigma>`;
    /// the spin magnetic moment is `g/2` times it
    pub spin_field_gradient: Vec<[f64; 3]>,
    /// dE/db of the orbital field source `b_i.L` [atom] (noncollinear):
    /// the orbital moment `<L>`
    pub orbital_field_gradient: Vec<[f64; 3]>,
    /// dE/dshift [atom][spin][shell]; zero at self-consistency
    pub shift_gradient: Vec<Vec<Vec<f64>>>,
    /// -dE/dpositions
    pub forces: Vec<[f64; 3]>,
    /// dE/dstrain (symmetric 3x3); stress = virial / volume
    pub virial: [[f64; 3]; 3],
}

/// Static data of one model and topology.
pub struct Bop<'a> {
    pub model: &'a Model,
    pub topo: &'a Topology,
    norb: usize,
    nshell: usize,
    nspin: usize,
    slices: Vec<(usize, usize)>,
    orb_shell: Vec<usize>,
    degeneracy: Vec<f64>,
    /// expansion settings per species (`Model::species_model`) and kernels
    sp_models: Vec<Model>,
    sp_kernels: Vec<Vec<f64>>,
    /// [species][shell]: whether the species carries the shell
    has_shell: Vec<Vec<bool>>,
    /// [species][orbital] 1 or 0 (bond blocks into absent shells vanish)
    orb_on: Vec<Vec<f64>>,
    /// per species, the projector (nspin * nshell)^2 of the shifts onto the
    /// independent ones: shells and spins tied together (`atomic_shifts`,
    /// `spin_polarized = false`) share their mean, absent shells get none;
    /// None without per-species settings (identity)
    tie: Option<Vec<Vec<f64>>>,
    /// [species] the start orbitals of its walks (its own orbitals; empty:
    /// all orbitals)
    starts: Vec<Vec<usize>>,
    smear: Option<expansion::Smearing>,
    noncollinear: bool,
    /// local spin axes (noncollinear), default all along z
    axes: Vec<[f64; 3]>,
    /// onsite spin-orbit block per species, `M x M` (noncollinear)
    spin_orbit: Vec<Vec<C64>>,
    /// sources `h_i` of `h_i.sigma` (eV) per atom, empty for none
    spin_field: Vec<[f64; 3]>,
    /// sources `b_i` of `b_i.L` (eV) per atom (noncollinear), empty for none
    orbital_field: Vec<[f64; 3]>,
}

impl<'a> Bop<'a> {
    /// Sets the local spin axes (noncollinear; need not be normalised).
    /// Adds onsite field sources: `h_i.sigma` (`spin`, eV) and `b_i.L`
    /// (`orbital`, eV, noncollinear only), one per atom.  In collinear
    /// calculations only the z component of `h` couples.
    pub fn with_fields(mut self, spin: Option<&[[f64; 3]]>, orbital: Option<&[[f64; 3]]>) -> Self {
        let n = self.topo.n_atoms;
        let add = |dst: &mut Vec<[f64; 3]>, src: &[[f64; 3]]| {
            assert_eq!(src.len(), n, "one field vector per atom");
            if dst.is_empty() {
                *dst = vec![[0.0; 3]; n];
            }
            for (d, s) in dst.iter_mut().zip(src) {
                for x in 0..3 {
                    d[x] += s[x];
                }
            }
        };
        if let Some(h) = spin {
            add(&mut self.spin_field, h);
        }
        if let Some(b) = orbital {
            assert!(
                self.noncollinear,
                "an orbital field needs spin = noncollinear"
            );
            add(&mut self.orbital_field, b);
        }
        self
    }

    /// Adds the Zeeman term `-mu_B B.(L + g/2 sigma)` of an external field
    /// `B` (Tesla, one vector per atom): `h = -mu_B g/2 B`, `b = -mu_B B`.  The
    /// orbital term only in noncollinear calculations.
    pub fn with_magnetic_field(self, field: &[[f64; 3]]) -> Self {
        assert!(
            self.model.nspin() == 2,
            "a magnetic field needs a magnetic calculation"
        );
        let h: Vec<[f64; 3]> = field
            .iter()
            .map(|b| b.map(|v| -0.5 * G_SPIN * MU_B * v))
            .collect();
        let b: Vec<[f64; 3]> = field.iter().map(|b| b.map(|v| -MU_B * v)).collect();
        let nc = self.noncollinear;
        self.with_fields(Some(&h), nc.then_some(&b[..]))
    }

    /// The spin field along the local axis of atom i (collinear: along z).
    fn h_par(&self, i: usize) -> f64 {
        if self.spin_field.is_empty() {
            return 0.0;
        }
        let h = self.spin_field[i];
        if self.noncollinear {
            let e = unit(self.axes[i]);
            h[0] * e[0] + h[1] * e[1] + h[2] * e[2]
        } else {
            h[2]
        }
    }

    pub fn with_axes(mut self, axes: &[[f64; 3]]) -> Self {
        assert_eq!(axes.len(), self.topo.n_atoms, "one spin axis per atom");
        self.axes = axes.to_vec();
        self
    }

    fn spin_orbit_blocks(model: &Model, slices: &[(usize, usize)], norb: usize) -> Vec<Vec<C64>> {
        if !model.spin_orbit {
            return Vec::new();
        }
        let m = 2 * norb;
        let sigma = pauli();
        model
            .spin_orbit_xi
            .iter()
            .map(|xi| {
                let mut o = vec![ZERO; m * m];
                for (k, (&sh, &(a0, a1))) in model.basis.iter().zip(slices).enumerate() {
                    let l = angular_momentum(sh);
                    let n = a1 - a0;
                    for a in 0..n {
                        for b in 0..n {
                            for t in 0..2 {
                                for u in 0..2 {
                                    let mut v = ZERO;
                                    for x in 0..3 {
                                        v += l[x][a * n + b] * sigma[x][t * 2 + u];
                                    }
                                    o[((a0 + a) * 2 + t) * m + (a0 + b) * 2 + u] +=
                                        v * (0.5 * xi[k]);
                                }
                            }
                        }
                    }
                }
                o
            })
            .collect()
    }

    pub fn new(model: &'a Model, topo: &'a Topology) -> Self {
        let (slices, norb) = shell_slices(&model.basis);
        let nspin = model.nspin();
        let orb_shell = slices
            .iter()
            .enumerate()
            .flat_map(|(k, &(a, b))| std::iter::repeat_n(k, b - a))
            .collect();
        let degeneracy: Vec<f64> = model
            .basis
            .iter()
            .map(|s| (2.0 / nspin as f64) * s.norb() as f64)
            .collect();
        Bop {
            model,
            topo,
            norb,
            nshell: model.basis.len(),
            nspin,
            slices,
            orb_shell,
            degeneracy,
            sp_models: (0..model.nspecies())
                .map(|s| model.species_model(s))
                .collect(),
            sp_kernels: (0..model.nspecies())
                .map(|s| expansion::kernel(&model.species_model(s)))
                .collect(),
            has_shell: (0..model.nspecies())
                .map(|s| {
                    (0..model.basis.len())
                        .map(|k| model.has_shell(s, k))
                        .collect()
                })
                .collect(),
            orb_on: (0..model.nspecies())
                .map(|s| {
                    model
                        .basis
                        .iter()
                        .enumerate()
                        .flat_map(|(k, sh)| {
                            std::iter::repeat_n(
                                f64::from(u8::from(model.has_shell(s, k))),
                                sh.norb(),
                            )
                        })
                        .collect()
                })
                .collect(),
            tie: Self::tie_projectors(model, nspin),
            starts: (0..model.nspecies())
                .map(|s| {
                    if model.species.is_none() {
                        return Vec::new();
                    }
                    let (sl, _) = shell_slices(&model.basis);
                    (0..model.basis.len())
                        .filter(|&k| model.has_shell(s, k))
                        .flat_map(|k| sl[k].0..sl[k].1)
                        .collect()
                })
                .collect(),
            smear: expansion::smearing(model),
            noncollinear: model.spin == crate::model::Spin::Noncollinear,
            axes: vec![[0.0, 0.0, 1.0]; topo.n_atoms],
            spin_orbit: Self::spin_orbit_blocks(model, &slices_for_so(model), norb),
            spin_field: Vec::new(),
            orbital_field: Vec::new(),
        }
    }

    fn tie_projectors(model: &Model, nspin: usize) -> Option<Vec<Vec<f64>>> {
        let st = model.species.as_ref()?;
        let nsh = model.basis.len();
        let nc = nspin * nsh;
        Some(
            (0..model.nspecies())
                .map(|sp| {
                    let key = |c: usize| -> Option<(usize, usize)> {
                        let (s, k) = (c / nsh, c % nsh);
                        if !model.has_shell(sp, k) {
                            return None;
                        }
                        let kk = if st[sp].atomic_shifts { usize::MAX } else { k };
                        let ss = if !st[sp].spin_polarized {
                            usize::MAX
                        } else {
                            s
                        };
                        Some((kk, ss))
                    };
                    let mut pm = vec![0.0; nc * nc];
                    for c in 0..nc {
                        let Some(kc) = key(c) else { continue };
                        let members: Vec<usize> = (0..nc).filter(|&d| key(d) == Some(kc)).collect();
                        for &d in &members {
                            pm[c * nc + d] = 1.0 / members.len() as f64;
                        }
                    }
                    pm
                })
                .collect(),
        )
    }

    /// The shifts (or a gradient with respect to them: the projector is
    /// symmetric) projected onto the independent shifts; identity without
    /// per-species settings.
    pub fn project_shifts(&self, x: &[f64]) -> Vec<f64> {
        let Some(tie) = &self.tie else {
            return x.to_vec();
        };
        let nc = self.nspin * self.nshell;
        let mut out = vec![0.0; x.len()];
        for i in 0..self.topo.n_atoms {
            let pm = &tie[self.topo.species[i]];
            for c in 0..nc {
                out[i * nc + c] = (0..nc).map(|d| pm[c * nc + d] * x[i * nc + d]).sum();
            }
        }
        out
    }

    /// The tie projector of atom i, or None (identity).
    pub(crate) fn tie_of(&self, i: usize) -> Option<&[f64]> {
        self.tie.as_ref().map(|t| &t[self.topo.species[i]][..])
    }

    pub fn shift_len(&self) -> usize {
        self.topo.n_atoms * self.nspin * self.nshell
    }

    /// Hopping blocks and their derivatives with respect to the bond vector,
    /// for every bond: (`n_bonds * norb^2`, `n_bonds * 3 * norb^2`).
    /// Components of the derivative of a bond block: the bond vector, and
    /// with screening the scale and shift of the three d-d integrals.
    fn dh_width(&self) -> usize {
        if self.model.screening.is_some() {
            9
        } else {
            3
        }
    }

    #[allow(clippy::type_complexity)]
    fn blocks(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        with_deriv: bool,
    ) -> (Vec<f64>, Vec<f64>, Option<Screened>) {
        let nl = &self.topo.nbr;
        let nn = self.norb * self.norb;
        let owners = nl.owners();
        let m = self.model;
        let sp = &self.topo.species;
        let screened = m.screening.as_ref().map(|_| {
            let vectors: Vec<[f64; 3]> = (0..nl.n_bonds())
                .map(|b| nl.vector(b, owners[b], pos, cell.as_ref()))
                .collect();
            let sc = crate::screening::compute(m, nl, sp, &vectors);
            Screened { sc, vectors }
        });
        let nd = self.dh_width();
        let per_bond: Vec<(Vec<f64>, Vec<f64>)> = (0..nl.n_bonds())
            .into_par_iter()
            .map(|b| {
                let i = owners[b];
                let r = nl.vector(b, i, pos, cell.as_ref());
                let (si, sj) = (sp[i], sp[nl.j[b]]);
                let factors = screened.as_ref().map(|s| s.sc.factors[b]);
                if with_deriv && factors.is_none() {
                    let rd = [0, 1, 2].map(|x| Dual::<3>::var(r[x], x));
                    let mut out = vec![Dual::<3>::zero(); nn];
                    bond_block(
                        rd, si, sj, &m.basis, m, m.radial, m.rcut, m.dcut, None, &mut out,
                    );
                    let v = out.iter().map(|x| x.v).collect();
                    let mut d = vec![0.0; 3 * nn];
                    for x in 0..3 {
                        for e in 0..nn {
                            d[x * nn + e] = out[e].d[x];
                        }
                    }
                    (v, d)
                } else if with_deriv {
                    // derivatives with respect to the bond vector (0..3) and
                    // the d-d scale (3..6) and shift (6..9) of the screening
                    type D9 = Dual<9>;
                    let rd = [0, 1, 2].map(|x| D9::var(r[x], x));
                    let dd = factors.map(|(sc, sh)| {
                        (
                            [0, 1, 2].map(|t| D9::var(sc[t], 3 + t)),
                            [0, 1, 2].map(|t| D9::var(sh[t], 6 + t)),
                        )
                    });
                    let mut out = vec![D9::zero(); nn];
                    bond_block(
                        rd, si, sj, &m.basis, m, m.radial, m.rcut, m.dcut, dd, &mut out,
                    );
                    let v = out.iter().map(|x| x.v).collect();
                    let mut d = vec![0.0; nd * nn];
                    for x in 0..nd {
                        for e in 0..nn {
                            d[x * nn + e] = out[e].d[x];
                        }
                    }
                    (v, d)
                } else {
                    let mut out = vec![0.0; nn];
                    bond_block(
                        r, si, sj, &m.basis, m, m.radial, m.rcut, m.dcut, factors, &mut out,
                    );
                    (out, Vec::new())
                }
            })
            .collect();
        let per_bond: Vec<(Vec<f64>, Vec<f64>)> = if m.species.is_none() {
            per_bond
        } else {
            per_bond
                .into_iter()
                .enumerate()
                .map(|(b, (mut v, mut d))| {
                    let (oi, oj) = (&self.orb_on[sp[owners[b]]], &self.orb_on[sp[nl.j[b]]]);
                    for al in 0..self.norb {
                        for be in 0..self.norb {
                            if oi[al] * oj[be] == 0.0 {
                                let e = al * self.norb + be;
                                v[e] = 0.0;
                                for x in 0..d.len() / nn {
                                    d[x * nn + e] = 0.0;
                                }
                            }
                        }
                    }
                    (v, d)
                })
                .collect()
        };
        let mut h = Vec::with_capacity(nl.n_bonds() * nn);
        let mut dh = Vec::with_capacity(if with_deriv {
            nl.n_bonds() * nd * nn
        } else {
            0
        });
        for (v, d) in per_bond {
            h.extend(v);
            dh.extend(d);
        }
        (h, dh, screened)
    }

    /// Gradients with respect to the bond vectors from the contracted block
    /// derivatives (`[0..3]` bond vector, `[3..9]` screening factors).
    fn bond_vector_gradient(&self, fw: &Forward, gbond: &[[f64; 9]]) -> Vec<[f64; 3]> {
        let mut g: Vec<[f64; 3]> = gbond.iter().map(|x| [x[0], x[1], x[2]]).collect();
        if let Some(s) = &fw.screened {
            let sbar: Vec<[f64; 6]> = gbond
                .iter()
                .map(|x| [x[3], x[4], x[5], x[6], x[7], x[8]])
                .collect();
            let extra = crate::screening::backward(
                self.model,
                &self.topo.nbr,
                &self.topo.species,
                &s.vectors,
                &s.sc,
                &sbar,
            );
            for (a, b) in g.iter_mut().zip(extra) {
                for x in 0..3 {
                    a[x] += b[x];
                }
            }
        }
        g
    }

    /// Onsite levels per spin channel, `[spin][atom * norb + orbital]`.
    fn onsite_levels(&self, shifts: &[f64]) -> Vec<Vec<f64>> {
        let (n, ns, nsh) = (self.topo.n_atoms, self.nspin, self.nshell);
        let projected = self.project_shifts(shifts);
        let shifts = &projected[..];
        (0..ns)
            .map(|s| {
                // the spin field along the axis moves spin up down and spin down up
                let sign = if s == 0 { 1.0 } else { -1.0 };
                let zeeman: Vec<f64> = (0..n)
                    .map(|i| if ns == 2 { sign * self.h_par(i) } else { 0.0 })
                    .collect();
                let mut lev = vec![0.0; n * self.norb];
                for i in 0..n {
                    let e0 = &self.model.onsite[self.topo.species[i]];
                    for al in 0..self.norb {
                        let k = self.orb_shell[al];
                        lev[i * self.norb + al] =
                            e0[k] + shifts[(i * ns + s) * nsh + k] + zeeman[i];
                    }
                }
                lev
            })
            .collect()
    }

    /// Entries of the walk vectors of all atoms and spins.
    fn walk_size(&self) -> usize {
        let nn = self.norb * self.norb;
        let top = |c: &crate::geometry::Cluster, a: usize| c.size_le[a.min(c.size_le.len() - 1)];
        self.topo
            .clusters
            .iter()
            .map(|c| {
                (0..=self.model.nmom.div_ceil(2))
                    .map(|a| top(c, a))
                    .sum::<usize>()
                    * nn
            })
            .sum::<usize>()
            * self.nspin
    }

    /// Shell moments `[atom][spin][shell][k]`, and the walks `[atom][spin]`
    /// if `keep`.
    #[allow(clippy::type_complexity)]
    fn shell_moments(
        &self,
        h: &[f64],
        levels: &[Vec<f64>],
        keep: bool,
    ) -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<Walk>>) {
        let nmom = self.model.nmom;
        let per_atom: Vec<(Vec<Vec<Vec<f64>>>, Vec<Walk>)> = (0..self.topo.n_atoms)
            .into_par_iter()
            .map(|i| {
                let mut walks = Vec::new();
                let mu = (0..self.nspin)
                    .map(|s| {
                        let hd = HamiltonianData {
                            norb: self.norb,
                            hblk: h,
                            onsite: &levels[s],
                            starts: &self.starts[self.topo.species[i]],
                        };
                        let (wk, mu) = moments(&self.topo.clusters[i], &hd, nmom);
                        if keep {
                            walks.push(wk);
                        }
                        self.slices
                            .iter()
                            .map(|&(a, b)| {
                                (0..=nmom)
                                    .map(|k| {
                                        (a..b).map(|d| mu[d * (nmom + 1) + k]).sum::<f64>()
                                            / (b - a) as f64
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                (mu, walks)
            })
            .collect();
        per_atom.into_iter().unzip()
    }

    fn fermi_level(&self, locals: &[Vec<Vec<Local>>], n_total: f64) -> f64 {
        let excess = |ef: f64| -> f64 {
            let mut s = 0.0;
            for at in locals {
                for sp in at {
                    for (k, l) in sp.iter().enumerate() {
                        s += self.degeneracy[k] * l.fill(ef, self.smear.as_ref());
                    }
                }
            }
            s - n_total
        };
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for l in locals.iter().flatten().flatten() {
            lo = lo.min(l.ainf - 2.0 * l.binf - 40.0 * self.model.kt);
            hi = hi.max(l.ainf + 2.0 * l.binf + 40.0 * self.model.kt);
        }
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if excess(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let mut ef = 0.5 * (lo + hi);
        for _ in 0..2 {
            let mut dos = 0.0;
            let mut f = -n_total;
            for at in locals {
                for sp in at {
                    for (k, l) in sp.iter().enumerate() {
                        let (fd, _) = expansion::filling_bond(
                            &l.sigma
                                .iter()
                                .map(|&x| Dual::<1>::cst(x))
                                .collect::<Vec<_>>(),
                            Dual::cst(l.ainf),
                            Dual::cst(l.binf),
                            Dual::<1>::var(ef, 0),
                            Dual::cst(l.gamma0),
                            self.smear.as_ref(),
                        );
                        f += self.degeneracy[k] * fd.v;
                        dos += self.degeneracy[k] * fd.d[0];
                    }
                }
            }
            if dos > 1e-14 {
                ef -= f / dos;
            }
        }
        ef
    }

    /// Energy and (with `gradients`) shift gradient, forces and virial at the
    /// given positions and onsite shifts (`[atom][spin][shell]` flattened).
    pub fn compute(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        shifts: &[f64],
        gradients: bool,
    ) -> Results {
        let mode = if gradients { Mode::Full } else { Mode::Energy };
        self.compute_mode(pos, cell, shifts, mode)
    }

    /// The forward pass: bond blocks (and their derivatives with
    /// `with_derivatives`), moments, local expansions, Fermi level and the
    /// derivatives of filling and bond energy.  Kept walks let later backward
    /// passes reuse it.
    pub fn forward(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        shifts: &[f64],
        with_derivatives: bool,
        keep: bool,
    ) -> Forward {
        let m = self.model;
        let topo = self.topo;
        let n = topo.n_atoms;
        assert_eq!(
            shifts.len(),
            self.shift_len(),
            "shifts must have n_atoms * nspin * nshell entries"
        );
        let (h, dh, screened) = self.blocks(pos, cell, with_derivatives);
        let levels = self.onsite_levels(shifts);
        let keep = keep && self.walk_size() <= WALK_BUDGET;
        let (mut proj, mut onsite_c, mut swalks) = (Vec::new(), Vec::new(), Vec::new());
        let (mu, walks) = if self.noncollinear {
            proj = self
                .axes
                .iter()
                .map(|a| spin_projectors(unit(*a)))
                .collect();
            onsite_c = self.onsite_blocks(&levels, &proj);
            let (mu, sw) = self.shell_moments_nc(&h, &onsite_c, &proj, keep);
            swalks = sw;
            (mu, Vec::new())
        } else {
            self.shell_moments(&h, &levels, keep)
        };
        let locals: Vec<Vec<Vec<Local>>> = mu
            .par_iter()
            .enumerate()
            .map(|(i, at)| {
                let s_i = topo.species[i];
                let (ms, kern) = (&self.sp_models[s_i], &self.sp_kernels[s_i]);
                at.iter()
                    .map(|sp| {
                        sp.iter()
                            .enumerate()
                            .map(|(k, x)| {
                                if self.has_shell[s_i][k] {
                                    Local::new(x, ms, kern)
                                } else {
                                    Local::absent()
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let n_total: f64 = (0..n)
            .map(|i| m.n0[topo.species[i]].iter().sum::<f64>())
            .sum();
        let ef = self.fermi_level(&locals, n_total);

        // filling and bond energy with derivatives, per atom, spin, shell
        let derivs: Vec<Vec<Vec<expansion::LocalDerivs>>> = mu
            .par_iter()
            .enumerate()
            .map(|(i, at)| {
                let s_i = topo.species[i];
                let (ms, kern) = (&self.sp_models[s_i], &self.sp_kernels[s_i]);
                at.iter()
                    .map(|sp| {
                        sp.iter()
                            .enumerate()
                            .map(|(k, x)| {
                                if self.has_shell[s_i][k] {
                                    local_derivs(x, ef, ms, kern, self.smear.as_ref())
                                } else {
                                    expansion::LocalDerivs::absent()
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let dndef: f64 = derivs
            .iter()
            .flat_map(|at| {
                at.iter().flat_map(|sp| {
                    sp.iter()
                        .enumerate()
                        .map(|(k, d)| self.degeneracy[k] * d.fill.d[m.nmom + 1])
                })
            })
            .sum();
        Forward {
            screened,
            coulomb: self.coulomb_integrals(pos, cell),
            h,
            dh,
            levels,
            keep,
            walks,
            swalks,
            onsite_c,
            proj,
            mu,
            locals,
            derivs,
            ef,
            dndef,
        }
    }

    /// As [`Bop::compute`], with the derivatives selected by `mode`.
    pub fn compute_mode(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        shifts: &[f64],
        mode: Mode,
    ) -> Results {
        let forces = mode == Mode::Full;
        let gradients = matches!(mode, Mode::ShiftGradient | Mode::Full | Mode::ScfStep);
        let response = matches!(mode, Mode::ScfStep | Mode::Response);
        let fw = self.forward(pos, cell, shifts, forces, gradients || response);
        let (mut res, p) = self.assemble(&fw, response);
        let n = self.topo.n_atoms;
        let mut forces_grad = vec![[0.0; 3]; n]; // dE/dpos
        let mut virial = [[0.0; 3]; 3];
        self.add_pair_terms(
            pos,
            cell,
            forces,
            &mut res.energy_atom,
            &mut forces_grad,
            &mut virial,
        );
        if forces && !fw.coulomb.is_empty() {
            let (g, w) = self.coulomb_gradient(&fw, pos, cell, &res.charge, &res.charge);
            for i in 0..n {
                for x in 0..3 {
                    forces_grad[i][x] += g[i][x];
                }
            }
            add3(&mut virial, &w);
        }
        res.energy = res.energy_atom.iter().sum();
        if gradients {
            let mut gr = self.seeded_backward(&fw, pos, cell, 1.0, &p, forces);
            self.add_explicit_zeeman(&res.moment, &mut gr);
            res.shift_gradient = self.unflatten(&gr.shifts);
            res.axes_gradient = gr.axes;
            res.spin_field_gradient = gr.spin_field;
            res.orbital_field_gradient = gr.orbital_field;
            if forces {
                for i in 0..n {
                    for x in 0..3 {
                        forces_grad[i][x] += gr.pos[i][x];
                    }
                }
                add3(&mut virial, &gr.virial);
            }
        }
        if forces {
            res.forces = forces_grad.iter().map(|g| [-g[0], -g[1], -g[2]]).collect();
            res.virial = symmetrised(virial);
        }
        res.moments = fw.mu;
        res
    }

    /// `[atom][spin][shell]` from the flat layout of the shifts.
    pub fn unflatten(&self, flat: &[f64]) -> Vec<Vec<Vec<f64>>> {
        let (ns, nsh) = (self.nspin, self.nshell);
        (0..self.topo.n_atoms)
            .map(|i| {
                (0..ns)
                    .map(|s| (0..nsh).map(|k| flat[(i * ns + s) * nsh + k]).collect())
                    .collect()
            })
            .collect()
    }

    /// Band part of the results (no repulsion, no derivatives) and dF/dN
    /// per atom, spin and shell.
    #[allow(clippy::type_complexity)]
    pub(crate) fn assemble(&self, fw: &Forward, response: bool) -> (Results, Vec<Vec<Vec<f64>>>) {
        let m = self.model;
        let topo = self.topo;
        let (n, ns, nsh, norb, nmom) = (topo.n_atoms, self.nspin, self.nshell, self.norb, m.nmom);
        let (h, levels, keep, walks, derivs, locals, ef) = (
            &fw.h, &fw.levels, fw.keep, &fw.walks, &fw.derivs, &fw.locals, fw.ef,
        );
        let mut res = Results {
            fermi: ef,
            ..Default::default()
        };
        // dF/dN per atom, spin, shell
        let mut p = vec![vec![vec![0.0; nsh]; ns]; n];
        for i in 0..n {
            let s_i = topo.species[i];
            let mut nspin_shell = vec![vec![0.0; nsh]; ns];
            let mut e_bond = 0.0;
            for s in 0..ns {
                for k in 0..nsh {
                    let g = self.degeneracy[k];
                    nspin_shell[s][k] = g * derivs[i][s][k].fill.v;
                    e_bond += g * derivs[i][s][k].bond.v;
                }
            }
            let n_shell: Vec<f64> = (0..nsh)
                .map(|k| (0..ns).map(|s| nspin_shell[s][k]).sum())
                .collect();
            let charge: f64 = (0..nsh).map(|k| n_shell[k] - m.n0[s_i][k]).sum();
            let e0 = &m.onsite[s_i];
            let jii = m.hubbard[s_i];
            let mut e_atom = e_bond
                + (0..nsh)
                    .map(|k| e0[k] * (n_shell[k] - m.n0[s_i][k]))
                    .sum::<f64>()
                + 0.5 * jii * charge * charge;
            let mut mom = 0.0;
            for s in 0..ns {
                for k in 0..nsh {
                    p[i][s][k] = e0[k] + jii * charge;
                }
            }
            if ns == 2 {
                for k in 0..nsh {
                    let mk = nspin_shell[0][k] - nspin_shell[1][k];
                    let ist = m.stoner[s_i][k];
                    e_atom -= 0.25 * ist * mk * mk;
                    p[i][0][k] -= 0.5 * ist * mk;
                    p[i][1][k] += 0.5 * ist * mk;
                    mom += mk;
                }
                // (h.e) m: the spin field along the axis, as a free-atom level
                let hp = self.h_par(i);
                e_atom += hp * mom;
                for k in 0..nsh {
                    p[i][0][k] += hp;
                    p[i][1][k] -= hp;
                }
            }
            res.dos.push(
                (0..ns)
                    .map(|s| {
                        (0..nsh)
                            .map(|k| self.degeneracy[k] * derivs[i][s][k].fill.d[nmom + 1])
                            .collect()
                    })
                    .collect(),
            );
            res.energy_atom.push(e_atom);
            res.n_shell.push(n_shell);
            res.occupations.push(nspin_shell);
            res.charge.push(charge);
            res.moment.push(mom);
            res.a_inf.push(
                locals[i]
                    .iter()
                    .map(|sp| sp.iter().map(|l| l.ainf).collect())
                    .collect(),
            );
            res.b_inf.push(
                locals[i]
                    .iter()
                    .map(|sp| sp.iter().map(|l| l.binf).collect())
                    .collect(),
            );
        }
        if !fw.coulomb.is_empty() {
            // E_coul = 1/2 sum over the second list of q_i J_ij q_j
            let nl = topo.nbr2.as_ref().unwrap();
            let mut v = vec![0.0; n];
            for i in 0..n {
                for b in nl.bonds(i) {
                    let (j, jij) = (nl.j[b], fw.coulomb[b].0);
                    v[i] += 0.5 * jij * res.charge[j];
                    v[j] += 0.5 * jij * res.charge[i];
                    res.energy_atom[i] += 0.5 * res.charge[i] * jij * res.charge[j];
                }
            }
            for i in 0..n {
                for s in 0..ns {
                    for k in 0..nsh {
                        p[i][s][k] += v[i];
                    }
                }
            }
            res.coulomb_potential = v;
        }
        if ns == 1 {
            res.moment.clear();
        }
        if response && self.noncollinear {
            res.local_response = self.local_response_nc(fw);
        } else if response {
            res.local_response = (0..n)
                .into_par_iter()
                .map(|i| {
                    (0..ns)
                        .map(|s| {
                            let hd = HamiltonianData {
                                norb,
                                hblk: h,
                                onsite: &levels[s],
                                starts: &self.starts[self.topo.species[i]],
                            };
                            let cl = &topo.clusters[i];
                            let recomputed;
                            let wk = if keep {
                                &walks[i][s]
                            } else {
                                recomputed = walk(cl, &hd, nmom);
                                &recomputed
                            };
                            // collinear: no response across the spin channels
                            let mut resp = vec![vec![0.0; ns * nsh]; nsh];
                            for (k2, &orb) in self.slices.iter().enumerate() {
                                let dmu = moments_tangent(cl, &hd, nmom, wk, i, orb);
                                for (k, &(a, b)) in self.slices.iter().enumerate() {
                                    let fd = &derivs[i][s][k].fill.d;
                                    let mut v = 0.0;
                                    for d in a..b {
                                        for q in 0..=nmom {
                                            v += fd[q] * dmu[d * (nmom + 1) + q];
                                        }
                                    }
                                    resp[k][s * nsh + k2] = self.degeneracy[k] * v / (b - a) as f64;
                                }
                            }
                            resp
                        })
                        .collect()
                })
                .collect();
        }
        (res, p)
    }

    /// Gradient of `wbond * U_bond + sum p N` (with the electron count held
    /// fixed by the Fermi level) with respect to the shifts (flat) and, with
    /// `forces`, the positions (dE/dpos) and strain.  With `wbond = 1` and
    /// `p = dF/dN` this is the energy gradient; with `wbond = 0` it is the
    /// adjoint `chi^T p` of the occupations, which the potential
    /// self-consistency needs.
    pub fn seeded_backward(
        &self,
        fw: &Forward,
        pos: &[[f64; 3]],
        cell: &Cell,
        wbond: f64,
        p: &[Vec<Vec<f64>>],
        forces: bool,
    ) -> Grad {
        let lambda = self.fermi_multiplier(fw, wbond, p);
        if self.noncollinear {
            return self.seeded_backward_nc(fw, pos, cell, wbond, p, lambda, forces);
        }
        let (shifts, pos_grad, virial) =
            self.seeded_backward_collinear(fw, pos, cell, wbond, p, lambda, forces);
        let nsh = self.nshell;
        // the spin field along z moves the levels of the two channels apart
        let spin_field = if self.nspin == 2 {
            (0..self.topo.n_atoms)
                .map(|i| {
                    let g: f64 = (0..nsh)
                        .map(|k| shifts[i * 2 * nsh + k] - shifts[(i * 2 + 1) * nsh + k])
                        .sum();
                    [0.0, 0.0, g]
                })
                .collect()
        } else {
            Vec::new()
        };
        Grad {
            shifts: self.project_shifts(&shifts),
            pos: pos_grad,
            virial,
            axes: Vec::new(),
            spin_field,
            orbital_field: Vec::new(),
        }
    }

    /// Lagrange multiplier of the electron count for the seed `(wbond, p)`.
    fn fermi_multiplier(&self, fw: &Forward, wbond: f64, p: &[Vec<Vec<f64>>]) -> f64 {
        let nmom = self.model.nmom;
        let mut dfdef = 0.0;
        for (i, at) in fw.derivs.iter().enumerate() {
            for (s, sp) in at.iter().enumerate() {
                for (k, d) in sp.iter().enumerate() {
                    dfdef += self.degeneracy[k]
                        * (wbond * d.bond.d[nmom + 1] + p[i][s][k] * d.fill.d[nmom + 1]);
                }
            }
        }
        if fw.dndef.abs() > 0.0 {
            dfdef / fw.dndef
        } else {
            0.0
        }
    }

    /// dE/dmu of every start orbital of atom i, spin s: `[d * (nmom+1) + k]`.
    fn moment_seed(
        &self,
        fw: &Forward,
        i: usize,
        s: usize,
        wbond: f64,
        p: &[Vec<Vec<f64>>],
        lambda: f64,
    ) -> Vec<f64> {
        let nmom = self.model.nmom;
        let mut gmu = vec![0.0; self.norb * (nmom + 1)];
        for d in 0..self.norb {
            let k = self.orb_shell[d];
            let (a, b) = self.slices[k];
            let w = self.degeneracy[k] / (b - a) as f64;
            let dv = &fw.derivs[i][s][k];
            for q in 0..=nmom {
                gmu[d * (nmom + 1) + q] =
                    w * (wbond * dv.bond.d[q] + (p[i][s][k] - lambda) * dv.fill.d[q]);
            }
        }
        gmu
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn seeded_backward_collinear(
        &self,
        fw: &Forward,
        pos: &[[f64; 3]],
        cell: &Cell,
        wbond: f64,
        p: &[Vec<Vec<f64>>],
        lambda: f64,
        forces: bool,
    ) -> (Vec<f64>, Vec<[f64; 3]>, [[f64; 3]; 3]) {
        let m = self.model;
        let topo = self.topo;
        let (n, ns, nsh, norb, nmom) = (topo.n_atoms, self.nspin, self.nshell, self.norb, m.nmom);
        let (h, dh, levels, keep, walks, derivs) =
            (&fw.h, &fw.dh, &fw.levels, fw.keep, &fw.walks, &fw.derivs);
        assert!(
            !forces || !dh.is_empty(),
            "forward pass without bond-block derivatives"
        );
        let _ = (derivs, nsh);
        let mut forces_grad = vec![[0.0; 3]; n];
        let mut virial = [[0.0; 3]; 3];
        {
            // dE/dmu per atom, spin, start orbital, k
            let nn = norb * norb;
            let nb = topo.nbr.n_bonds();
            let owners = topo.nbr.owners();
            let chunk = n.div_ceil(rayon::current_num_threads() * 4).max(1);
            let nd = self.dh_width();
            let partial: Vec<(Vec<[f64; 9]>, Vec<f64>)> = (0..n)
                .collect::<Vec<_>>()
                .par_chunks(chunk)
                .map(|atoms| {
                    let mut gbond = vec![[0.0; 9]; if forces { nb } else { 0 }];
                    let mut obar = vec![0.0; ns * n * norb];
                    for &i in atoms {
                        for s in 0..ns {
                            let gmu = self.moment_seed(fw, i, s, wbond, p, lambda);
                            let hd = HamiltonianData {
                                norb,
                                hblk: h,
                                onsite: &levels[s],
                                starts: &self.starts[self.topo.species[i]],
                            };
                            let cl = &topo.clusters[i];
                            let recomputed;
                            let wk = if keep {
                                &walks[i][s]
                            } else {
                                recomputed = walk(cl, &hd, nmom);
                                &recomputed
                            };
                            let ob = &mut obar[s * n * norb..(s + 1) * n * norb];
                            let mut contract = |b: usize, hbar: &[f64]| {
                                let dblk = &dh[b * nd * nn..(b + 1) * nd * nn];
                                for x in 0..nd {
                                    let mut acc = 0.0;
                                    for e in 0..nn {
                                        acc += hbar[e] * dblk[x * nn + e];
                                    }
                                    gbond[b][x] += acc;
                                }
                            };
                            let bonds = if forces { Some(&mut contract) } else { None };
                            backward(cl, &hd, nmom, wk, &gmu, ob, bonds);
                        }
                    }
                    (gbond, obar)
                })
                .collect();
            let mut gbond9 = vec![[0.0; 9]; if forces { nb } else { 0 }];
            let mut obar = vec![0.0; ns * n * norb];
            for (g, o) in partial {
                for b in 0..g.len() {
                    for x in 0..nd {
                        gbond9[b][x] += g[b][x];
                    }
                }
                for (t, v) in obar.iter_mut().zip(o) {
                    *t += v;
                }
            }
            let gbond = if forces {
                self.bond_vector_gradient(fw, &gbond9)
            } else {
                Vec::new()
            };
            for b in 0..gbond.len() {
                let i = owners[b];
                let j = topo.nbr.j[b];
                let r = topo.nbr.vector(b, i, pos, cell.as_ref());
                for x in 0..3 {
                    forces_grad[j][x] += gbond[b][x];
                    forces_grad[i][x] -= gbond[b][x];
                    for y in 0..3 {
                        virial[x][y] += gbond[b][x] * r[y];
                    }
                }
            }
            let mut sg = vec![0.0; n * ns * nsh];
            for i in 0..n {
                for s in 0..ns {
                    for (k, &(a, b)) in self.slices.iter().enumerate() {
                        sg[(i * ns + s) * nsh + k] =
                            (a..b).map(|al| obar[(s * n + i) * norb + al]).sum();
                    }
                }
            }
            (sg, forces_grad, virial)
        }
    }

    /// Repulsion and embedding energy per atom, dE/dpos and virial.
    pub(crate) fn repulsion(
        &self,
        rep: &crate::model::Repulsion,
        pos: &[[f64; 3]],
        cell: &Cell,
        gradients: bool,
    ) -> (Vec<f64>, Vec<[f64; 3]>, [[f64; 3]; 3]) {
        let m = self.model;
        let nl = self
            .topo
            .nbr2
            .as_ref()
            .expect("topology built without rcut2");
        let sp = &self.topo.species;
        let (rcut2, dcut2) = (m.rcut2.unwrap(), m.dcut2);
        let n = self.topo.n_atoms;
        type D1 = Dual<1>;
        // per atom: energy and dE/dr of each of its bonds
        let per_atom: Vec<(f64, Vec<f64>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut rho = D1::zero();
                let mut pair = 0.0;
                let mut dpair = Vec::new();
                let mut drho = Vec::new();
                for b in nl.bonds(i) {
                    let r = nl.vector(b, i, pos, cell.as_ref());
                    let rr = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
                    let rd = D1::var(rr, 0);
                    let fc = cosine_cutoff(rd, rcut2, dcut2);
                    let sj = sp[nl.j[b]];
                    let pe = fc * rep.pair(sp[i], sj, rd) * 0.5;
                    pair += pe.v;
                    dpair.push(pe.d[0]);
                    if rep.has_embedding() {
                        let de = fc * rep.density(sp[i], sj, rd);
                        rho.v += de.v;
                        drho.push(de.d[0]);
                    } else {
                        drho.push(0.0);
                    }
                }
                let emb = rep.embed(sp[i], D1::var(rho.v, 0));
                let de_drho = emb.d[0];
                let g: Vec<f64> = dpair
                    .iter()
                    .zip(&drho)
                    .map(|(a, b)| a + de_drho * b)
                    .collect();
                (emb.v + pair, g)
            })
            .collect();
        let mut grad = vec![[0.0; 3]; n];
        let mut vir = [[0.0; 3]; 3];
        if gradients {
            for (i, (_, g)) in per_atom.iter().enumerate() {
                for (q, b) in nl.bonds(i).enumerate() {
                    let r = nl.vector(b, i, pos, cell.as_ref());
                    let rr = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
                    let j = nl.j[b];
                    for x in 0..3 {
                        let gx = g[q] * r[x] / rr;
                        grad[j][x] += gx;
                        grad[i][x] -= gx;
                        for y in 0..3 {
                            vir[x][y] += gx * r[y];
                        }
                    }
                }
            }
        }
        (per_atom.into_iter().map(|(e, _)| e).collect(), grad, vir)
    }
}

pub(crate) fn symmetrised(mut v: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    for x in 0..3 {
        for y in 0..x {
            let a = 0.5 * (v[x][y] + v[y][x]);
            v[x][y] = a;
            v[y][x] = a;
        }
    }
    v
}

/// Result of [`Bop::forward`].
/// Gradient from [`Bop::seeded_backward`].
#[derive(Clone, Debug, Default)]
pub struct Grad {
    /// d/dshifts, flat like the shifts
    pub shifts: Vec<f64>,
    /// d/dpositions (with `forces`)
    pub pos: Vec<[f64; 3]>,
    /// d/dstrain (with `forces`)
    pub virial: [[f64; 3]; 3],
    /// d/d(spin axes) (noncollinear)
    pub axes: Vec<[f64; 3]>,
    /// d/dh of the spin field sources through the levels and the onsite
    /// blocks (magnetic models)
    pub spin_field: Vec<[f64; 3]>,
    /// d/db of the orbital field sources (noncollinear)
    pub orbital_field: Vec<[f64; 3]>,
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

fn slices_for_so(model: &Model) -> Vec<(usize, usize)> {
    shell_slices(&model.basis).0
}

/// Screening data of a forward pass: the factors and the bond vectors.
pub struct Screened {
    sc: crate::screening::Screening,
    vectors: Vec<[f64; 3]>,
}

pub struct Forward {
    screened: Option<Screened>,
    /// intersite Coulomb integral and its radial derivative per entry of the
    /// second neighbour list (empty unless coulomb = frauenheim)
    coulomb: Vec<(f64, f64)>,
    h: Vec<f64>,
    dh: Vec<f64>,
    levels: Vec<Vec<f64>>,
    keep: bool,
    walks: Vec<Vec<Walk>>,
    swalks: Vec<SpinorWalk>,
    onsite_c: Vec<C64>,
    proj: Vec<[[C64; 4]; 2]>,
    pub mu: Vec<Vec<Vec<Vec<f64>>>>,
    locals: Vec<Vec<Vec<Local>>>,
    derivs: Vec<Vec<Vec<expansion::LocalDerivs>>>,
    pub ef: f64,
    dndef: f64,
}

pub(crate) fn add3(a: &mut [[f64; 3]; 3], b: &[[f64; 3]; 3]) {
    for x in 0..3 {
        for y in 0..3 {
            a[x][y] += b[x][y];
        }
    }
}

// --- non-collinear magnetism ---------------------------------------------------

impl<'a> Bop<'a> {
    /// Onsite blocks `O_i = sum_s level_{i s} (1 x P_s) + xi L.S`, `n * M * M`.
    fn onsite_blocks(&self, levels: &[Vec<f64>], proj: &[[[C64; 4]; 2]]) -> Vec<C64> {
        let (n, norb) = (self.topo.n_atoms, self.norb);
        let m = 2 * norb;
        let mut out = vec![ZERO; n * m * m];
        for i in 0..n {
            let o = &mut out[i * m * m..(i + 1) * m * m];
            for a in 0..norb {
                for t in 0..2 {
                    for u in 0..2 {
                        let mut v = ZERO;
                        for s in 0..2 {
                            v += proj[i][s][t * 2 + u] * levels[s][i * norb + a];
                        }
                        o[(a * 2 + t) * m + a * 2 + u] = v;
                    }
                }
            }
            if !self.spin_orbit.is_empty() {
                for (x, y) in o.iter_mut().zip(&self.spin_orbit[self.topo.species[i]]) {
                    *x += *y;
                }
            }
            if !self.spin_field.is_empty() {
                // h_perp.sigma, the part across the axis (the part along it is in the levels)
                let (h, e) = (self.spin_field[i], unit(self.axes[i]));
                let hp = self.h_par(i);
                let perp = [h[0] - hp * e[0], h[1] - hp * e[1], h[2] - hp * e[2]];
                let sigma = pauli();
                for a in 0..norb {
                    for t in 0..2 {
                        for u in 0..2 {
                            let mut v = ZERO;
                            for x in 0..3 {
                                v += sigma[x][t * 2 + u] * perp[x];
                            }
                            o[(a * 2 + t) * m + a * 2 + u] += v;
                        }
                    }
                }
            }
            if !self.orbital_field.is_empty() {
                let b = self.orbital_field[i];
                for (&sh, &(a0, a1)) in self.model.basis.iter().zip(&self.slices) {
                    let l = angular_momentum(sh);
                    let nl = a1 - a0;
                    for a in 0..nl {
                        for c in 0..nl {
                            let mut v = ZERO;
                            for x in 0..3 {
                                v += l[x][a * nl + c] * b[x];
                            }
                            for t in 0..2 {
                                o[((a0 + a) * 2 + t) * m + (a0 + c) * 2 + t] += v;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn spinor_data<'b>(&'b self, h: &'b [f64], onsite: &'b [C64]) -> SpinorData<'b> {
        SpinorData {
            norb: self.norb,
            hblk: h,
            onsite,
            orbital_diagonal: self.spin_orbit.is_empty() && self.orbital_field.is_empty(),
        }
    }

    /// `Re tr(P m)` for one 2x2 moment matrix at `off`: `Re sum_uv P[v][u] m[u][v]`.
    fn project(p: &[C64; 4], m: &[C64], off: usize) -> f64 {
        let mut v = 0.0;
        for u in 0..2 {
            for w in 0..2 {
                v += (p[w * 2 + u] * m[off + u * 2 + w]).re;
            }
        }
        v
    }

    #[allow(clippy::type_complexity)]
    fn shell_moments_nc(
        &self,
        h: &[f64],
        onsite: &[C64],
        proj: &[[[C64; 4]; 2]],
        keep: bool,
    ) -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<SpinorWalk>) {
        let nmom = self.model.nmom;
        let sd = self.spinor_data(h, onsite);
        let per_atom: Vec<(Vec<Vec<Vec<f64>>>, Option<SpinorWalk>)> = (0..self.topo.n_atoms)
            .into_par_iter()
            .map(|i| {
                let cl = &self.topo.clusters[i];
                let wk = spinor::walk(cl, &sd, nmom);
                let mm = spinor::moment_matrices(cl, &wk, self.norb, nmom);
                let mu = (0..2)
                    .map(|s| {
                        self.slices
                            .iter()
                            .map(|&(a, b)| {
                                (0..=nmom)
                                    .map(|k| {
                                        (a..b)
                                            .map(|d| {
                                                Self::project(
                                                    &proj[i][s],
                                                    &mm,
                                                    (d * (nmom + 1) + k) * 4,
                                                )
                                            })
                                            .sum::<f64>()
                                            / (b - a) as f64
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                (mu, keep.then_some(wk))
            })
            .collect();
        let (mu, walks): (Vec<_>, Vec<_>) = per_atom.into_iter().unzip();
        (mu, walks.into_iter().flatten().collect())
    }

    fn spinor_walk_of<'b>(
        &self,
        fw: &'b Forward,
        i: usize,
        sd: &SpinorData,
        store: &'b mut Option<SpinorWalk>,
    ) -> &'b SpinorWalk {
        if fw.keep {
            &fw.swalks[i]
        } else {
            *store = Some(spinor::walk(&self.topo.clusters[i], sd, self.model.nmom));
            store.as_ref().unwrap()
        }
    }

    /// `local_response` of the noncollinear walk: the response of the local
    /// spin channels of each atom to their own levels.
    fn local_response_nc(&self, fw: &Forward) -> Vec<Vec<Vec<Vec<f64>>>> {
        let (nsh, norb, nmom) = (self.nshell, self.norb, self.model.nmom);
        let m = 2 * norb;
        let sd = self.spinor_data(&fw.h, &fw.onsite_c);
        (0..self.topo.n_atoms)
            .into_par_iter()
            .map(|i| {
                let cl = &self.topo.clusters[i];
                let mut store = None;
                let wk = self.spinor_walk_of(fw, i, &sd, &mut store);
                // the level of channel (s2, k2) moves the occupations of both
                // local spin channels: with canted neighbours or spin-orbit
                // coupling the channels are not independent
                let mut resp = vec![vec![vec![0.0; 2 * nsh]; nsh]; 2];
                for s2 in 0..2 {
                    for (k2, &(a0, a1)) in self.slices.iter().enumerate() {
                        let mut d_o = vec![ZERO; m * m];
                        for a in a0..a1 {
                            for t in 0..2 {
                                for u in 0..2 {
                                    d_o[(a * 2 + t) * m + a * 2 + u] = fw.proj[i][s2][t * 2 + u];
                                }
                            }
                        }
                        let dm = spinor::moment_matrices_tangent(cl, &sd, nmom, wk, i, &d_o);
                        for s in 0..2 {
                            for (k, &(a, b)) in self.slices.iter().enumerate() {
                                let fd = &fw.derivs[i][s][k].fill.d;
                                let mut v = 0.0;
                                for d in a..b {
                                    for q in 0..=nmom {
                                        v += fd[q]
                                            * Self::project(
                                                &fw.proj[i][s],
                                                &dm,
                                                (d * (nmom + 1) + q) * 4,
                                            );
                                    }
                                }
                                resp[s][k][s2 * nsh + k2] = self.degeneracy[k] * v / (b - a) as f64;
                            }
                        }
                    }
                }
                resp
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn seeded_backward_nc(
        &self,
        fw: &Forward,
        pos: &[[f64; 3]],
        cell: &Cell,
        wbond: f64,
        p: &[Vec<Vec<f64>>],
        lambda: f64,
        forces: bool,
    ) -> Grad {
        let topo = self.topo;
        let (n, nsh, norb, nmom) = (topo.n_atoms, self.nshell, self.norb, self.model.nmom);
        let m = 2 * norb;
        let mm = m * m;
        let nn = norb * norb;
        let nb = topo.nbr.n_bonds();
        let dh = &fw.dh;
        assert!(
            !forces || !dh.is_empty(),
            "forward pass without bond-block derivatives"
        );
        let sd = self.spinor_data(&fw.h, &fw.onsite_c);
        let chunk = n.div_ceil(rayon::current_num_threads() * 4).max(1);
        let nd = self.dh_width();
        type Part = (Vec<[f64; 9]>, Vec<C64>, Vec<(usize, [[C64; 4]; 2])>);
        let partial: Vec<Part> = (0..n)
            .collect::<Vec<_>>()
            .par_chunks(chunk)
            .map(|atoms| {
                let mut gbond = vec![[0.0; 9]; if forces { nb } else { 0 }];
                let mut obar = vec![ZERO; n * mm];
                let mut mproj = Vec::new();
                for &i in atoms {
                    let gmu: Vec<Vec<f64>> = (0..2)
                        .map(|s| self.moment_seed(fw, i, s, wbond, p, lambda))
                        .collect();
                    let cl = &topo.clusters[i];
                    let mut store = None;
                    let wk = self.spinor_walk_of(fw, i, &sd, &mut store);
                    // C_k[d] = sum_s gmu[s][d][k] P_s, and the projection part
                    let mut cmat = vec![ZERO; norb * (nmom + 1) * 4];
                    for d in 0..norb {
                        for q in 0..=nmom {
                            for e in 0..4 {
                                let mut v = ZERO;
                                for s in 0..2 {
                                    v += fw.proj[i][s][e] * gmu[s][d * (nmom + 1) + q];
                                }
                                cmat[(d * (nmom + 1) + q) * 4 + e] = v;
                            }
                        }
                    }
                    let mmat = spinor::moment_matrices(cl, wk, norb, nmom);
                    let mut ms = [[ZERO; 4]; 2];
                    for s in 0..2 {
                        for d in 0..norb {
                            for q in 0..=nmom {
                                let g = gmu[s][d * (nmom + 1) + q];
                                for e in 0..4 {
                                    ms[s][e] += mmat[(d * (nmom + 1) + q) * 4 + e] * g;
                                }
                            }
                        }
                    }
                    mproj.push((i, ms));
                    let mut contract = |b: usize, hbar: &[f64]| {
                        let dblk = &dh[b * nd * nn..(b + 1) * nd * nn];
                        for x in 0..nd {
                            let mut acc = 0.0;
                            for e in 0..nn {
                                acc += hbar[e] * dblk[x * nn + e];
                            }
                            gbond[b][x] += acc;
                        }
                    };
                    let bonds = if forces { Some(&mut contract) } else { None };
                    spinor::backward(cl, &sd, nmom, wk, &cmat, &mut obar, bonds);
                }
                (gbond, obar, mproj)
            })
            .collect();
        let mut gbond9 = vec![[0.0; 9]; if forces { nb } else { 0 }];
        let mut obar = vec![ZERO; n * mm];
        let mut ms = vec![[[ZERO; 4]; 2]; n];
        for (g, o, mp) in partial {
            for b in 0..g.len() {
                for x in 0..nd {
                    gbond9[b][x] += g[b][x];
                }
            }
            for (t, v) in obar.iter_mut().zip(o) {
                *t += v;
            }
            for (i, v) in mp {
                ms[i] = v;
            }
        }
        let mut grad = Grad {
            shifts: vec![0.0; n * 2 * nsh],
            pos: vec![[0.0; 3]; n],
            ..Default::default()
        };
        let gbond = if forces {
            self.bond_vector_gradient(fw, &gbond9)
        } else {
            Vec::new()
        };
        let owners = topo.nbr.owners();
        for b in 0..gbond.len() {
            let (i, j) = (owners[b], topo.nbr.j[b]);
            let r = topo.nbr.vector(b, i, pos, cell.as_ref());
            for x in 0..3 {
                grad.pos[j][x] += gbond[b][x];
                grad.pos[i][x] -= gbond[b][x];
                for y in 0..3 {
                    grad.virial[x][y] += gbond[b][x] * r[y];
                }
            }
        }
        // Re sum_tu conj(Obar[(a t),(a u)]) X[t][u]
        let pair = |i: usize, a: usize, x: &[C64; 4]| -> f64 {
            let o = &obar[i * mm..(i + 1) * mm];
            let mut v = 0.0;
            for t in 0..2 {
                for u in 0..2 {
                    v += o[(a * 2 + t) * m + a * 2 + u].dot_re(x[t * 2 + u]);
                }
            }
            v
        };
        grad.axes = vec![[0.0; 3]; n];
        for i in 0..n {
            for s in 0..2 {
                for (k, &(a0, a1)) in self.slices.iter().enumerate() {
                    grad.shifts[(i * 2 + s) * nsh + k] =
                        (a0..a1).map(|a| pair(i, a, &fw.proj[i][s])).sum();
                }
            }
            let dp = projector_derivatives(self.axes[i]);
            for x in 0..3 {
                let mut v = 0.0;
                for s in 0..2 {
                    for a in 0..norb {
                        v += fw.levels[s][i * norb + a] * pair(i, a, &dp[x][s]);
                    }
                    v += Self::project(&dp[x][s], &ms[i][s], 0);
                }
                grad.axes[i][x] = v;
            }
        }
        // field sources: dE/dh_par through the levels, dE/dh_perp and dE/db
        // through the onsite blocks
        let sigma = pauli();
        let orbital = !(self.spin_orbit.is_empty() && self.orbital_field.is_empty());
        grad.spin_field = vec![[0.0; 3]; n];
        grad.orbital_field = vec![[0.0; 3]; n];
        let lmat: Vec<[Vec<C64>; 3]> = self
            .model
            .basis
            .iter()
            .map(|&sh| angular_momentum(sh))
            .collect();
        for i in 0..n {
            let e = unit(self.axes[i]);
            let g_par: f64 = (0..nsh)
                .map(|k| grad.shifts[i * 2 * nsh + k] - grad.shifts[(i * 2 + 1) * nsh + k])
                .sum();
            let mut sv = [0.0; 3];
            for x in 0..3 {
                sv[x] = (0..norb).map(|a| pair(i, a, &sigma[x])).sum();
            }
            let es = e[0] * sv[0] + e[1] * sv[1] + e[2] * sv[2];
            for x in 0..3 {
                grad.spin_field[i][x] = g_par * e[x] + sv[x] - es * e[x];
            }
            if orbital {
                let o = &obar[i * mm..(i + 1) * mm];
                for (k, &(a0, a1)) in self.slices.iter().enumerate() {
                    let nl = a1 - a0;
                    for x in 0..3 {
                        let mut v = 0.0;
                        for a in 0..nl {
                            for c in 0..nl {
                                for t in 0..2 {
                                    v += o[((a0 + a) * 2 + t) * m + (a0 + c) * 2 + t]
                                        .dot_re(lmat[k][x][a * nl + c]);
                                }
                            }
                        }
                        grad.orbital_field[i][x] += v;
                    }
                }
            }
            if !self.spin_field.is_empty() {
                // the levels hold h.e and the onsite block h - (h.e) e, both
                // functions of the axis: de/da = (1 - e e^T) / |a|
                let h = self.spin_field[i];
                let hp = self.h_par(i);
                let na =
                    (self.axes[i][0].powi(2) + self.axes[i][1].powi(2) + self.axes[i][2].powi(2))
                        .sqrt();
                let proj = |v: [f64; 3]| -> [f64; 3] {
                    let ev = e[0] * v[0] + e[1] * v[1] + e[2] * v[2];
                    [
                        (v[0] - ev * e[0]) / na,
                        (v[1] - ev * e[1]) / na,
                        (v[2] - ev * e[2]) / na,
                    ]
                };
                let (ph, ps) = (proj(h), proj(sv));
                for x in 0..3 {
                    grad.axes[i][x] += (g_par - es) * ph[x] - hp * ps[x];
                }
            }
        }
        grad
    }

    /// The explicit Zeeman energy `sum_i (h_i.e_i) m_i` (collinear: `h_z m`)
    /// at fixed occupations, added to an energy gradient.
    pub(crate) fn add_explicit_zeeman(&self, moments: &[f64], grad: &mut Grad) {
        if self.spin_field.is_empty() || moments.is_empty() {
            return;
        }
        for i in 0..self.topo.n_atoms {
            let m = moments[i];
            if self.noncollinear {
                let e = unit(self.axes[i]);
                let a = self.axes[i];
                let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                let h = self.spin_field[i];
                let hp = self.h_par(i);
                for x in 0..3 {
                    grad.spin_field[i][x] += m * e[x];
                    grad.axes[i][x] += m * (h[x] - hp * e[x]) / na;
                }
            } else {
                grad.spin_field[i][2] += m;
            }
        }
    }
}

// --- pair terms: repulsion, Yukawa, intersite Coulomb ------------------------------

impl<'a> Bop<'a> {
    /// Repulsion, embedding and Yukawa energies per atom added to `energy`,
    /// their dE/dpos to `grad` and virial to `virial` (with `forces`).
    pub(crate) fn add_pair_terms(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        forces: bool,
        energy: &mut [f64],
        grad: &mut [[f64; 3]],
        virial: &mut [[f64; 3]; 3],
    ) {
        let mut add = |(e, g, w): (Vec<f64>, Vec<[f64; 3]>, [[f64; 3]; 3])| {
            for i in 0..energy.len() {
                energy[i] += e[i];
                for x in 0..3 {
                    grad[i][x] += g[i][x];
                }
            }
            add3(virial, &w);
        };
        if let Some(rep) = &self.model.repulsion {
            add(self.repulsion(rep, pos, cell, forces));
        }
        if let Some(y) = &self.model.yukawa {
            add(self.yukawa(y, pos, cell, forces));
        }
    }

    /// Turns dE/dr of every entry of the second list into dE/dpos and virial.
    fn radial_to_cartesian(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        de_dr: &[f64],
    ) -> (Vec<[f64; 3]>, [[f64; 3]; 3]) {
        let nl = self.topo.nbr2.as_ref().unwrap();
        let n = self.topo.n_atoms;
        let mut grad = vec![[0.0; 3]; n];
        let mut vir = [[0.0; 3]; 3];
        for i in 0..n {
            for b in nl.bonds(i) {
                let r = nl.vector(b, i, pos, cell.as_ref());
                let rr = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
                let j = nl.j[b];
                for x in 0..3 {
                    let gx = de_dr[b] * r[x] / rr;
                    grad[j][x] += gx;
                    grad[i][x] -= gx;
                    for y in 0..3 {
                        vir[x][y] += gx * r[y];
                    }
                }
            }
        }
        (grad, vir)
    }

    /// Yukawa repulsion per atom, and with `gradients` its dE/dpos and virial.
    fn yukawa(
        &self,
        y: &crate::model::Yukawa,
        pos: &[[f64; 3]],
        cell: &Cell,
        gradients: bool,
    ) -> (Vec<f64>, Vec<[f64; 3]>, [[f64; 3]; 3]) {
        type D1 = Dual<1>;
        let m = self.model;
        let nl = self
            .topo
            .nbr2
            .as_ref()
            .expect("topology built without rcut2");
        let sp = &self.topo.species;
        let n = self.topo.n_atoms;
        let (rcut2, dcut2) = (m.rcut2.unwrap(), m.dcut2);
        let dist = |b: usize, i: usize| {
            let r = nl.vector(b, i, pos, cell.as_ref());
            (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt()
        };
        // environment and lambda per atom, with d(lambda_i)/d(env_i)
        let mut lam = vec![0.0; n];
        let mut dlam_denv = vec![0.0; n];
        for i in 0..n {
            let mut env = 0.0;
            for b in nl.bonds(i) {
                let r = D1::cst(dist(b, i));
                let s = (sp[i], sp[nl.j[b]]);
                env += (cosine_cutoff(r, rcut2, dcut2)
                    * (-r * y.upsilon[s.0][s.1]).exp()
                    * y.c[s.0][s.1])
                    .v;
            }
            lam[i] = y.lambda0[sp[i]];
            if env > 0.0 {
                let e = D1::var(env, 0).powf(1.0 / y.m[sp[i]]);
                lam[i] += e.v;
                dlam_denv[i] = e.d[0];
            }
        }
        let mut energy = vec![0.0; n];
        let mut de_dr = vec![0.0; nl.n_bonds()];
        let mut de_dlam = vec![0.0; n];
        for i in 0..n {
            for b in nl.bonds(i) {
                let j = nl.j[b];
                let s = (sp[i], sp[j]);
                // f(r, lambda_ij) with derivatives in r and lambda_ij
                let r = Dual::<2>::var(dist(b, i), 0);
                let l = Dual::<2>::var(0.5 * (lam[i] + lam[j]), 1);
                let f = cosine_cutoff(r, rcut2, dcut2)
                    * r.powf(-1.0)
                    * y.b[s.0][s.1]
                    * (-(r - 2.0 * y.r_core[s.0][s.1]) * l).exp()
                    * 0.5;
                energy[i] += f.v;
                de_dr[b] += f.d[0];
                de_dlam[i] += 0.5 * f.d[1];
                de_dlam[j] += 0.5 * f.d[1];
            }
        }
        if !gradients {
            return (energy, vec![[0.0; 3]; n], [[0.0; 3]; 3]);
        }
        // lambda_i depends on the distances of i's entries through env_i
        for i in 0..n {
            if dlam_denv[i] == 0.0 {
                continue;
            }
            for b in nl.bonds(i) {
                let r = D1::var(dist(b, i), 0);
                let s = (sp[i], sp[nl.j[b]]);
                let t = cosine_cutoff(r, rcut2, dcut2)
                    * (-r * y.upsilon[s.0][s.1]).exp()
                    * y.c[s.0][s.1];
                de_dr[b] += de_dlam[i] * dlam_denv[i] * t.d[0];
            }
        }
        let (g, w) = self.radial_to_cartesian(pos, cell, &de_dr);
        (energy, g, w)
    }

    /// Intersite Coulomb integrals `J_ij` and `dJ_ij/dr` per entry of the
    /// second list: Frauenheim integral times the taper, zero beyond rcut2.
    fn coulomb_integrals(&self, pos: &[[f64; 3]], cell: &Cell) -> Vec<(f64, f64)> {
        let m = self.model;
        if m.coulomb != Coulomb::Frauenheim {
            return Vec::new();
        }
        let nl = self
            .topo
            .nbr2
            .as_ref()
            .expect("topology built without rcut2");
        let sp = &self.topo.species;
        let rcut2 = m.rcut2.unwrap();
        let mut out = vec![(0.0, 0.0); nl.n_bonds()];
        for i in 0..self.topo.n_atoms {
            for b in nl.bonds(i) {
                let rv = nl.vector(b, i, pos, cell.as_ref());
                let r = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
                if r >= rcut2 {
                    continue;
                }
                let (si, sj) = (sp[i], sp[nl.j[b]]);
                let [r1, r2] = m.taper[si][sj];
                let rd = Dual::<1>::var(r, 0);
                let jij = crate::pairwise::taper(r1, r2, rd)
                    * crate::pairwise::frauenheim(m.hubbard[si], m.hubbard[sj], rd);
                out[b] = (jij.v, jij.d[0]);
            }
        }
        out
    }

    /// Gradient of `1/2 sum over the second list of left_i J_ij right_j`
    /// with respect to positions and strain (the J_ij at fixed charges).
    pub(crate) fn coulomb_gradient(
        &self,
        fw: &Forward,
        pos: &[[f64; 3]],
        cell: &Cell,
        left: &[f64],
        right: &[f64],
    ) -> (Vec<[f64; 3]>, [[f64; 3]; 3]) {
        let nl = self.topo.nbr2.as_ref().unwrap();
        let mut de_dr = vec![0.0; nl.n_bonds()];
        for i in 0..self.topo.n_atoms {
            for b in nl.bonds(i) {
                de_dr[b] = 0.5 * fw.coulomb[b].1 * left[i] * right[nl.j[b]];
            }
        }
        self.radial_to_cartesian(pos, cell, &de_dr)
    }

    /// `sum_j dV_i/dq_j u_j`, the intersite part of the Coulomb kernel applied
    /// to per-atom values (`V_i = dE_coul/dq_i`); empty without it.
    pub(crate) fn coulomb_apply(&self, fw: &Forward, u: &[f64]) -> Vec<f64> {
        if fw.coulomb.is_empty() {
            return Vec::new();
        }
        let nl = self.topo.nbr2.as_ref().unwrap();
        let mut out = vec![0.0; self.topo.n_atoms];
        for i in 0..self.topo.n_atoms {
            for b in nl.bonds(i) {
                let (j, jij) = (nl.j[b], fw.coulomb[b].0);
                out[i] += 0.5 * jij * u[j];
                out[j] += 0.5 * jij * u[i];
            }
        }
        out
    }

    /// Whether the forward pass holds intersite Coulomb integrals.
    pub(crate) fn has_coulomb(fw: &Forward) -> bool {
        !fw.coulomb.is_empty()
    }
}
