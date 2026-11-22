//! Environment-dependent screening of the d-d bond integrals (Nguyen-Manh,
//! Pettifor and Vitek, PRL 85, 4136 (2000); `bopjax.hamiltonian.screening_*`):
//!
//!   beta~_tau = scale_tau * beta_tau + shift_tau,
//!   scale = 1 - (mu3 - mu2) / den,   shift = -c1 beta / den,
//!   den = 1 + O_tau^2 - 2 mu2,
//!
//! with the per-bond sums over the other neighbours k of the central atom i
//! (see bopjax for the terms): the interference `c1 beta`, `mu3 = O_tau *
//! mu3core`, and `mu2 = O_tau^2 + (ring_ij + ring_ji) / 2`, where `ring_ji`
//! is the ring term of the reverse bond around atom j.
//!
//! Derivatives: the bond blocks take (scale, shift) as extra dual inputs, so
//! the backward pass yields dE/dscale and dE/dshift per bond.  These go back
//! through the factor formula (dual numbers on its 15 inputs) to the per-bond
//! terms, and from there through every pair term to the two bond vectors it
//! depends on (dual numbers on 6 components).  Everything ends as gradients
//! with respect to bond vectors of the first neighbour list, like the
//! hopping itself, so forces and virial follow as for the blocks.

use crate::geometry::NeighbourList;
use crate::hamiltonian::{cosine_cutoff, Radial};
use crate::model::{Model, ScreeningParams};
use crate::real::{Dual, Real};

const WEIGHT: [f64; 3] = [0.5, 0.25, 0.25];

/// Per-bond sums of the screening terms.
#[derive(Clone, Copy, Debug, Default)]
pub struct Terms {
    pub o_dd: [f64; 3],
    pub interference: [f64; 3],
    pub mu3core: [f64; 3],
    pub ring: [f64; 3],
}

struct Ctx<'a> {
    p: &'a ScreeningParams,
    radial: Radial,
    rcut: f64,
    dcut: f64,
}

impl Ctx<'_> {
    fn rad<T: Real>(&self, params: &[Vec<Vec<f64>>], a: usize, b: usize, r: T) -> T {
        cosine_cutoff(r, self.rcut, self.dcut) * self.radial.eval(&params[a][b], r)
    }
}

fn norm<T: Real>(v: [T; 3]) -> T {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// d-d overlaps of bond i -> j.
fn o_dd<T: Real>(c: &Ctx, r_ia: [T; 3], si: usize, sj: usize) -> [T; 3] {
    let r = norm(r_ia);
    [
        c.rad(&c.p.dds_overlap, si, sj, r),
        c.rad(&c.p.ddp_overlap, si, sj, r),
        c.rad(&c.p.ddd_overlap, si, sj, r),
    ]
}

/// Contributions of the screening atom k (bond i -> k, `r_ib`) to the terms
/// of bond i -> j (`r_ia`): interference, mu3core, ring.
#[allow(clippy::type_complexity)]
fn pair<T: Real>(
    c: &Ctx,
    r_ia: [T; 3],
    r_ib: [T; 3],
    si: usize,
    sj: usize,
    sk: usize,
) -> ([T; 3], [T; 3], [T; 3]) {
    let r = norm(r_ia);
    let u = [r_ia[0] / r, r_ia[1] / r, r_ia[2] / r];
    let r_jk = [r_ib[0] - r_ia[0], r_ib[1] - r_ia[1], r_ib[2] - r_ia[2]];
    let d_jk = norm(r_jk);
    let d_ik = norm(r_ib);
    let z_i = u[0] * r_ib[0] + u[1] * r_ib[1] + u[2] * r_ib[2];
    let z_j = z_i - r;
    let (n_i, n_j) = (z_i / d_ik, z_j / d_jk);
    let rho2 = d_ik * d_ik - z_i * z_i;
    let ll = rho2 / (d_ik * d_jk);
    let l2 = rho2 / (d_ik * d_ik);
    let si3 = n_i * n_i * 3.0 - 1.0;
    let sj3 = n_j * n_j * 3.0 - 1.0;
    let g_ij = [si3 * sj3 * 0.25, n_i * n_j * ll * 3.0, ll * ll * 0.75];
    let g_ii = [si3 * si3 * 0.25, n_i * n_i * l2 * 3.0, l2 * l2 * 0.75];
    // s orbital on k, d orbital on i (o_ik, b_ik) and on j (o_kj, b_kj)
    let o_ik = c.rad(&c.p.sds_overlap, sk, si, d_ik);
    let b_ik = c.rad(&c.p.sds, sk, si, d_ik);
    let o_kj = c.rad(&c.p.sds_overlap, sk, sj, d_jk);
    let b_kj = c.rad(&c.p.sds, sk, sj, d_jk);
    let mut out = ([T::zero(); 3], [T::zero(); 3], [T::zero(); 3]);
    for t in 0..3 {
        let w = WEIGHT[t];
        out.0[t] = (b_ik * o_kj + o_ik * b_kj) * g_ij[t] * w;
        out.1[t] = o_ik * o_kj * g_ij[t] * (4.0 * w);
        out.2[t] = o_ik * o_ik * g_ii[t] * (2.0 * w);
    }
    out
}

/// (scale, shift) from the terms of a bond and the ring term of its reverse.
fn factor<T: Real>(
    o: [T; 3],
    interference: [T; 3],
    mu3core: [T; 3],
    ring: [T; 3],
    ring_j: [T; 3],
) -> ([T; 3], [T; 3]) {
    let (mut scale, mut shift) = ([T::zero(); 3], [T::zero(); 3]);
    for t in 0..3 {
        let mu3 = o[t] * mu3core[t];
        let mu2 = o[t] * o[t] + (ring[t] + ring_j[t]) * 0.5;
        let den = o[t] * o[t] - mu2 * 2.0 + 1.0;
        scale[t] = T::cst(1.0) - (mu3 - mu2) / den;
        shift[t] = -interference[t] / den;
    }
    (scale, shift)
}

/// The reverse bond (j -> i, opposite translation) of every bond.
pub fn reverse_bonds(nl: &NeighbourList) -> Vec<usize> {
    let owners = nl.owners();
    (0..nl.n_bonds())
        .map(|b| {
            let (i, j, t) = (owners[b], nl.j[b], nl.t[b]);
            nl.bonds(j)
                .find(|&c| nl.j[c] == i && nl.t[c] == [-t[0], -t[1], -t[2]])
                .expect("neighbour list is not symmetric")
        })
        .collect()
}

pub struct Screening {
    pub rev: Vec<usize>,
    pub terms: Vec<Terms>,
    /// (scale, shift) of every bond
    pub factors: Vec<([f64; 3], [f64; 3])>,
}

fn ctx(m: &Model) -> Ctx<'_> {
    Ctx {
        p: m.screening.as_ref().expect("no screening parameters"),
        radial: m.screening_radial.unwrap_or(m.radial),
        rcut: m.rcut,
        dcut: m.dcut,
    }
}

/// Screening terms and factors for all bonds; `vectors[b]` is bond b.
pub fn compute(
    m: &Model,
    nl: &NeighbourList,
    species: &[usize],
    vectors: &[[f64; 3]],
) -> Screening {
    let c = ctx(m);
    let owners = nl.owners();
    let rev = reverse_bonds(nl);
    let mut terms = vec![Terms::default(); nl.n_bonds()];
    for (a, t) in terms.iter_mut().enumerate() {
        let i = owners[a];
        let (si, sj) = (species[i], species[nl.j[a]]);
        t.o_dd = o_dd(&c, vectors[a], si, sj);
        for b in nl.bonds(i) {
            if b == a {
                continue;
            }
            let (x, y, z) = pair(&c, vectors[a], vectors[b], si, sj, species[nl.j[b]]);
            for k in 0..3 {
                t.interference[k] += x[k];
                t.mu3core[k] += y[k];
                t.ring[k] += z[k];
            }
        }
    }
    let factors = (0..nl.n_bonds())
        .map(|b| {
            let t = &terms[b];
            factor(
                t.o_dd,
                t.interference,
                t.mu3core,
                t.ring,
                terms[rev[b]].ring,
            )
        })
        .collect();
    Screening {
        rev,
        terms,
        factors,
    }
}

/// Gradients with respect to the bond vectors from `sbar[b]` = dE/d(scale,
/// shift) of every bond (`[scale_0..2, shift_0..2]`).
pub fn backward(
    m: &Model,
    nl: &NeighbourList,
    species: &[usize],
    vectors: &[[f64; 3]],
    sc: &Screening,
    sbar: &[[f64; 6]],
) -> Vec<[f64; 3]> {
    let c = ctx(m);
    let owners = nl.owners();
    let nb = nl.n_bonds();
    // cotangents of the per-bond terms
    let mut bar = vec![Terms::default(); nb];
    for b in 0..nb {
        if sbar[b].iter().all(|&v| v == 0.0) {
            continue;
        }
        type D = Dual<15>;
        let t = &sc.terms[b];
        let tj = &sc.terms[sc.rev[b]];
        let var = |v: [f64; 3], off: usize| -> [D; 3] { [0, 1, 2].map(|k| D::var(v[k], off + k)) };
        let (scale, shift) = factor(
            var(t.o_dd, 0),
            var(t.interference, 3),
            var(t.mu3core, 6),
            var(t.ring, 9),
            var(tj.ring, 12),
        );
        let mut g = [0.0; 15];
        for k in 0..3 {
            for q in 0..15 {
                g[q] += sbar[b][k] * scale[k].d[q] + sbar[b][3 + k] * shift[k].d[q];
            }
        }
        for k in 0..3 {
            bar[b].o_dd[k] += g[k];
            bar[b].interference[k] += g[3 + k];
            bar[b].mu3core[k] += g[6 + k];
            bar[b].ring[k] += g[9 + k];
            bar[sc.rev[b]].ring[k] += g[12 + k];
        }
    }
    let mut grad = vec![[0.0; 3]; nb];
    for a in 0..nb {
        let ba = bar[a];
        let i = owners[a];
        let (si, sj) = (species[i], species[nl.j[a]]);
        let ra3 = [0, 1, 2].map(|x| Dual::<3>::var(vectors[a][x], x));
        let o = o_dd(&c, ra3, si, sj);
        for k in 0..3 {
            for x in 0..3 {
                grad[a][x] += ba.o_dd[k] * o[k].d[x];
            }
        }
        if ba
            .interference
            .iter()
            .chain(&ba.mu3core)
            .chain(&ba.ring)
            .all(|&v| v == 0.0)
        {
            continue;
        }
        for b in nl.bonds(i) {
            if b == a {
                continue;
            }
            type D = Dual<6>;
            let ra = [0, 1, 2].map(|x| D::var(vectors[a][x], x));
            let rb = [0, 1, 2].map(|x| D::var(vectors[b][x], 3 + x));
            let (x, y, z) = pair(&c, ra, rb, si, sj, species[nl.j[b]]);
            let mut s = D::zero();
            for k in 0..3 {
                s += x[k] * ba.interference[k] + y[k] * ba.mu3core[k] + z[k] * ba.ring[k];
            }
            for q in 0..3 {
                grad[a][q] += s.d[q];
                grad[b][q] += s.d[3 + q];
            }
        }
    }
    grad
}
