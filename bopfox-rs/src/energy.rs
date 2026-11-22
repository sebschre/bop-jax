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

use crate::expansion::{self, local_derivs, Local};
use crate::geometry::Topology;
use crate::hamiltonian::{bond_block, cosine_cutoff, shell_slices};
use crate::model::Model;
use crate::moments::{backward, moments, moments_tangent, walk, HamiltonianData, Walk};
use crate::real::{Dual, Real};
use rayon::prelude::*;

pub type Cell = Option<[[f64; 3]; 3]>;

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
    pub charge: Vec<f64>,
    /// magnetic moment per atom (collinear only)
    pub moment: Vec<f64>,
    /// shell moments [atom][spin][shell][k]
    pub moments: Vec<Vec<Vec<Vec<f64>>>>,
    pub a_inf: Vec<Vec<Vec<f64>>>,
    pub b_inf: Vec<Vec<Vec<f64>>>,
    /// local density of states at the Fermi level, dN/d(ef) [atom][spin][shell]
    pub dos: Vec<Vec<Vec<f64>>>,
    /// dN_{i s k} / d(level_{i s k'}) at fixed Fermi level, the response of
    /// an atom's occupations to its own levels, [atom][spin][k][k'] (ScfStep)
    pub local_response: Vec<Vec<Vec<Vec<f64>>>>,
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
    kernel: Vec<f64>,
}

impl<'a> Bop<'a> {
    pub fn new(model: &'a Model, topo: &'a Topology) -> Self {
        let (slices, norb) = shell_slices(&model.basis);
        let nspin = model.nspin();
        let orb_shell = slices
            .iter()
            .enumerate()
            .flat_map(|(k, &(a, b))| std::iter::repeat_n(k, b - a))
            .collect();
        let degeneracy = model
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
            kernel: expansion::kernel(model),
        }
    }

    pub fn shift_len(&self) -> usize {
        self.topo.n_atoms * self.nspin * self.nshell
    }

    /// Hopping blocks and their derivatives with respect to the bond vector,
    /// for every bond: (`n_bonds * norb^2`, `n_bonds * 3 * norb^2`).
    fn blocks(&self, pos: &[[f64; 3]], cell: &Cell, with_deriv: bool) -> (Vec<f64>, Vec<f64>) {
        let nl = &self.topo.nbr;
        let nn = self.norb * self.norb;
        let owners = nl.owners();
        let m = self.model;
        let sp = &self.topo.species;
        let per_bond: Vec<(Vec<f64>, Vec<f64>)> = (0..nl.n_bonds())
            .into_par_iter()
            .map(|b| {
                let i = owners[b];
                let r = nl.vector(b, i, pos, cell.as_ref());
                let (si, sj) = (sp[i], sp[nl.j[b]]);
                if with_deriv {
                    let rd = [
                        Dual::<3>::var(r[0], 0),
                        Dual::<3>::var(r[1], 1),
                        Dual::<3>::var(r[2], 2),
                    ];
                    let mut out = vec![Dual::<3>::zero(); nn];
                    bond_block(rd, si, sj, &m.basis, m, m.radial, m.rcut, m.dcut, &mut out);
                    let v = out.iter().map(|x| x.v).collect();
                    let mut d = vec![0.0; 3 * nn];
                    for x in 0..3 {
                        for e in 0..nn {
                            d[x * nn + e] = out[e].d[x];
                        }
                    }
                    (v, d)
                } else {
                    let mut out = vec![0.0; nn];
                    bond_block(r, si, sj, &m.basis, m, m.radial, m.rcut, m.dcut, &mut out);
                    (out, Vec::new())
                }
            })
            .collect();
        let mut h = Vec::with_capacity(nl.n_bonds() * nn);
        let mut dh = Vec::with_capacity(if with_deriv { nl.n_bonds() * 3 * nn } else { 0 });
        for (v, d) in per_bond {
            h.extend(v);
            dh.extend(d);
        }
        (h, dh)
    }

    /// Onsite levels per spin channel, `[spin][atom * norb + orbital]`.
    fn onsite_levels(&self, shifts: &[f64]) -> Vec<Vec<f64>> {
        let (n, ns, nsh) = (self.topo.n_atoms, self.nspin, self.nshell);
        (0..ns)
            .map(|s| {
                let mut lev = vec![0.0; n * self.norb];
                for i in 0..n {
                    let e0 = &self.model.onsite[self.topo.species[i]];
                    for al in 0..self.norb {
                        let k = self.orb_shell[al];
                        lev[i * self.norb + al] = e0[k] + shifts[(i * ns + s) * nsh + k];
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
                        s += self.degeneracy[k] * l.fill(ef);
                    }
                }
            }
            s - n_total
        };
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for l in locals.iter().flatten().flatten() {
            lo = lo.min(l.ainf - 2.0 * l.binf);
            hi = hi.max(l.ainf + 2.0 * l.binf);
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
                        let (fd, _) = expansion::filling_and_bond_energy(
                            &l.sigma
                                .iter()
                                .map(|&x| Dual::<1>::cst(x))
                                .collect::<Vec<_>>(),
                            Dual::cst(l.ainf),
                            Dual::cst(l.binf),
                            Dual::<1>::var(ef, 0),
                            Dual::cst(l.gamma0),
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

    /// As [`Bop::compute`], with the derivatives selected by `mode`.
    pub fn compute_mode(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        shifts: &[f64],
        mode: Mode,
    ) -> Results {
        let forces = mode == Mode::Full;
        let gradients = mode != Mode::Energy;
        let m = self.model;
        let topo = self.topo;
        let (n, ns, nsh, norb, nmom) = (topo.n_atoms, self.nspin, self.nshell, self.norb, m.nmom);
        assert_eq!(
            shifts.len(),
            self.shift_len(),
            "shifts must have n_atoms * nspin * nshell entries"
        );
        let (h, dh) = self.blocks(pos, cell, forces);
        let levels = self.onsite_levels(shifts);
        let keep = gradients && self.walk_size() <= WALK_BUDGET;
        let (mu, walks) = self.shell_moments(&h, &levels, keep);

        let locals: Vec<Vec<Vec<Local>>> = mu
            .par_iter()
            .map(|at| {
                at.iter()
                    .map(|sp| sp.iter().map(|x| Local::new(x, m, &self.kernel)).collect())
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
            .map(|at| {
                at.iter()
                    .map(|sp| {
                        sp.iter()
                            .map(|x| local_derivs(x, ef, m, &self.kernel))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let mut res = Results {
            fermi: ef,
            ..Default::default()
        };
        let mut dfdef = 0.0;
        let mut dndef = 0.0;
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
            }
            for s in 0..ns {
                for k in 0..nsh {
                    let g = self.degeneracy[k];
                    let d = &derivs[i][s][k];
                    dfdef += g * (d.bond.d[nmom + 1] + p[i][s][k] * d.fill.d[nmom + 1]);
                    dndef += g * d.fill.d[nmom + 1];
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
        if ns == 1 {
            res.moment.clear();
        }
        if mode == Mode::ScfStep {
            res.local_response = (0..n)
                .into_par_iter()
                .map(|i| {
                    (0..ns)
                        .map(|s| {
                            let hd = HamiltonianData {
                                norb,
                                hblk: &h,
                                onsite: &levels[s],
                            };
                            let cl = &topo.clusters[i];
                            let recomputed;
                            let wk = if keep {
                                &walks[i][s]
                            } else {
                                recomputed = walk(cl, &hd, nmom);
                                &recomputed
                            };
                            let mut resp = vec![vec![0.0; nsh]; nsh];
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
                                    resp[k][k2] = self.degeneracy[k] * v / (b - a) as f64;
                                }
                            }
                            resp
                        })
                        .collect()
                })
                .collect();
        }
        let lambda = if dndef.abs() > 0.0 {
            dfdef / dndef
        } else {
            0.0
        };

        let mut forces_grad = vec![[0.0; 3]; n]; // dE/dpos
        let mut virial = [[0.0; 3]; 3];
        if let Some(rep) = &m.repulsion {
            let (e_rep, g_rep, w_rep) = self.repulsion(rep, pos, cell, forces);
            for i in 0..n {
                res.energy_atom[i] += e_rep[i];
            }
            for i in 0..n {
                for x in 0..3 {
                    forces_grad[i][x] += g_rep[i][x];
                }
            }
            add3(&mut virial, &w_rep);
        }
        res.energy = res.energy_atom.iter().sum();

        if gradients {
            // dE/dmu per atom, spin, start orbital, k
            let nn = norb * norb;
            let nb = topo.nbr.n_bonds();
            let owners = topo.nbr.owners();
            let chunk = n.div_ceil(rayon::current_num_threads() * 4).max(1);
            let partial: Vec<(Vec<[f64; 3]>, Vec<f64>)> = (0..n)
                .collect::<Vec<_>>()
                .par_chunks(chunk)
                .map(|atoms| {
                    let mut gbond = vec![[0.0; 3]; if forces { nb } else { 0 }];
                    let mut obar = vec![0.0; ns * n * norb];
                    for &i in atoms {
                        for s in 0..ns {
                            let mut gmu = vec![0.0; norb * (nmom + 1)];
                            for d in 0..norb {
                                let k = self.orb_shell[d];
                                let (a, b) = self.slices[k];
                                let w = self.degeneracy[k] / (b - a) as f64;
                                let dv = &derivs[i][s][k];
                                for q in 0..=nmom {
                                    gmu[d * (nmom + 1) + q] =
                                        w * (dv.bond.d[q] + (p[i][s][k] - lambda) * dv.fill.d[q]);
                                }
                            }
                            let hd = HamiltonianData {
                                norb,
                                hblk: &h,
                                onsite: &levels[s],
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
                                let dblk = &dh[b * 3 * nn..(b + 1) * 3 * nn];
                                for x in 0..3 {
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
            let mut gbond = vec![[0.0; 3]; if forces { nb } else { 0 }];
            let mut obar = vec![0.0; ns * n * norb];
            for (g, o) in partial {
                for b in 0..g.len() {
                    for x in 0..3 {
                        gbond[b][x] += g[b][x];
                    }
                }
                for (t, v) in obar.iter_mut().zip(o) {
                    *t += v;
                }
            }
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
            res.shift_gradient = (0..n)
                .map(|i| {
                    (0..ns)
                        .map(|s| {
                            self.slices
                                .iter()
                                .map(|&(a, b)| (a..b).map(|al| obar[(s * n + i) * norb + al]).sum())
                                .collect()
                        })
                        .collect()
                })
                .collect();
        }
        if forces {
            res.forces = forces_grad.iter().map(|g| [-g[0], -g[1], -g[2]]).collect();
            for x in 0..3 {
                for y in 0..x {
                    let v = 0.5 * (virial[x][y] + virial[y][x]);
                    virial[x][y] = v;
                    virial[y][x] = v;
                }
            }
            res.virial = virial;
        }
        res.moments = mu;
        res
    }

    /// Repulsion and embedding energy per atom, dE/dpos and virial.
    fn repulsion(
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

fn add3(a: &mut [[f64; 3]; 3], b: &[[f64; 3]; 3]) {
    for x in 0..3 {
        for y in 0..3 {
            a[x][y] += b[x][y];
        }
    }
}
