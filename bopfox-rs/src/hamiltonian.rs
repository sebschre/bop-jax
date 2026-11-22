//! Orthogonal two-centre Slater-Koster blocks for s, p and d orbitals, radial
//! functions and the cutoff, as in `bopjax.hamiltonian`.
//!
//! The blocks are written as projectors that are polynomial in the bond
//! direction `u`, so no bond frame is needed.  d orbitals are an orthonormal
//! basis of symmetric traceless tensors (order xy, yz, zx, x2-y2, 3z2-r2).
//! Everything is generic over [`Real`]: evaluated with `Dual<3>` on the bond
//! vector it yields the exact derivative of every matrix element.

use crate::real::Real;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Shell {
    #[serde(rename = "s")]
    S,
    #[serde(rename = "p")]
    P,
    #[serde(rename = "d")]
    D,
}

impl Shell {
    pub fn l(self) -> usize {
        match self {
            Shell::S => 0,
            Shell::P => 1,
            Shell::D => 2,
        }
    }
    pub fn norb(self) -> usize {
        2 * self.l() + 1
    }
}

/// Bond-integral channels of a shell pair (lower shell first).
pub fn channels(la: Shell, lb: Shell) -> &'static [&'static str] {
    match (la, lb) {
        (Shell::S, Shell::S) => &["sss"],
        (Shell::S, Shell::P) => &["sps"],
        (Shell::S, Shell::D) => &["sds"],
        (Shell::P, Shell::P) => &["pps", "ppp"],
        (Shell::P, Shell::D) => &["pds", "pdp"],
        (Shell::D, Shell::D) => &["dds", "ddp", "ddd"],
        _ => panic!("channels: shells must be ordered"),
    }
}

/// Orbital index range of each shell and the number of orbitals.
pub fn shell_slices(basis: &[Shell]) -> (Vec<(usize, usize)>, usize) {
    let mut out = Vec::new();
    let mut start = 0;
    for sh in basis {
        out.push((start, start + sh.norb()));
        start += sh.norb();
    }
    (out, start)
}

const S2: f64 = std::f64::consts::SQRT_2;

/// The five d tensors B_m (3x3 each).
pub fn d_tensors() -> [[[f64; 3]; 3]; 5] {
    let r2 = 1.0 / S2;
    let r6 = 1.0 / 6f64.sqrt();
    [
        [[0.0, r2, 0.0], [r2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, r2], [0.0, r2, 0.0]],
        [[0.0, 0.0, r2], [0.0, 0.0, 0.0], [r2, 0.0, 0.0]],
        [[r2, 0.0, 0.0], [0.0, -r2, 0.0], [0.0, 0.0, 0.0]],
        [[-r6, 0.0, 0.0], [0.0, -r6, 0.0], [0.0, 0.0, 2.0 * r6]],
    ]
}

/// Radial dependence of the bond integrals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Radial {
    /// `p0 (p1 / r)^p2`
    Powerlaw,
    /// BOPfox "v0" (Lim): `p0 exp(-p1 (x - 1)) (1 - exp(-p2 x^p3))`, `x = r / p4`
    V0,
    /// `sum_k c_k exp(-lambda_k r^n_k)`, p = [c1, l1, n1, c2, l2, n2, ...]
    ExpSum,
}

impl Radial {
    pub fn eval<T: Real>(self, p: &[f64], r: T) -> T {
        match self {
            Radial::Powerlaw => (r.powf(-1.0) * p[1]).powf(p[2]) * p[0],
            Radial::V0 => {
                let x = r / p[4];
                (-(x - 1.0) * p[1]).exp() * (T::cst(1.0) - (-x.powf(p[3]) * p[2]).exp()) * p[0]
            }
            Radial::ExpSum => {
                let mut s = T::zero();
                for t in p.chunks(3) {
                    if t[0] != 0.0 {
                        s += (-r.powf(t[2]) * t[1]).exp() * t[0];
                    }
                }
                s
            }
        }
    }
}

/// 1 below `rcut - dcut`, 0 above `rcut`, half-cosine in between (BOPfox form).
pub fn cosine_cutoff<T: Real>(r: T, rcut: f64, dcut: f64) -> T {
    let rv = r.val();
    if rv >= rcut {
        return T::zero();
    }
    if dcut <= 0.0 || rv < rcut - dcut {
        return T::cst(1.0);
    }
    let arg = (r - rcut + dcut) * (PI / dcut);
    (arg.cos() + 1.0) * 0.5
}

/// Upper block `<la on i | H | lb on j>` with `l(la) <= l(lb)`, `u` from i to j,
/// written into `out` (row-major `na x nb`).  `v` holds the channel values in
/// the order of [`channels`].
fn block_upper<T: Real>(la: Shell, lb: Shell, u: [T; 3], v: &[T], out: &mut [T]) {
    let z = T::zero();
    match (la, lb) {
        (Shell::S, Shell::S) => out[0] = v[0],
        (Shell::S, Shell::P) => {
            for x in 0..3 {
                out[x] = v[0] * u[x];
            }
        }
        (Shell::P, Shell::P) => {
            let (vs, vp) = (v[0], v[1]);
            for x in 0..3 {
                for y in 0..3 {
                    let e = if x == y { vp } else { z };
                    out[3 * x + y] = e + (vs - vp) * u[x] * u[y];
                }
            }
        }
        _ => {
            let b = d_tensors();
            // bu[m][x] = (B_m u)_x, q[m] = u.B_m.u
            let mut bu = [[z; 3]; 5];
            let mut q = [z; 5];
            for m in 0..5 {
                for x in 0..3 {
                    let mut s = z;
                    for y in 0..3 {
                        if b[m][x][y] != 0.0 {
                            s += u[y] * b[m][x][y];
                        }
                    }
                    bu[m][x] = s;
                }
                q[m] = bu[m][0] * u[0] + bu[m][1] * u[1] + bu[m][2] * u[2];
            }
            let s15 = 1.5f64.sqrt();
            let sig: Vec<T> = q.iter().map(|&x| x * s15).collect();
            match (la, lb) {
                (Shell::S, Shell::D) => {
                    for m in 0..5 {
                        out[m] = v[0] * sig[m];
                    }
                }
                (Shell::P, Shell::D) => {
                    for x in 0..3 {
                        for m in 0..5 {
                            let pi = (bu[m][x] - u[x] * q[m]) * S2;
                            out[5 * x + m] = v[0] * u[x] * sig[m] + v[1] * pi;
                        }
                    }
                }
                (Shell::D, Shell::D) => {
                    let (vs, vp, vd) = (v[0], v[1], v[2]);
                    for m in 0..5 {
                        for n in 0..5 {
                            let bb =
                                bu[m][0] * bu[n][0] + bu[m][1] * bu[n][1] + bu[m][2] * bu[n][2];
                            let p_pi = (bb - q[m] * q[n]) * 2.0;
                            let p_sig = sig[m] * sig[n];
                            let e = if m == n { vd } else { z };
                            out[5 * m + n] = e + (vp - vd) * p_pi + (vs - vd) * p_sig;
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Parameters of one bond-integral channel for all species pairs.
pub type ChannelParams = Vec<Vec<Vec<f64>>>; // [species_a][species_b][npar]

/// Bond-integral parameter lookup: `bond[c]` for channel name `c`.
pub trait BondParams {
    fn channel(&self, name: &str) -> &ChannelParams;
}

/// Hopping block `<orbital a on i | H | orbital b on j>` (row-major
/// `norb x norb`) for bond vector `rvec` from i to j, cutoff included.
/// `dd_screen` is the (scale, shift) of the three d-d channels, or None.
#[allow(clippy::too_many_arguments)]
pub fn bond_block<T: Real, P: BondParams + ?Sized>(
    rvec: [T; 3],
    si: usize,
    sj: usize,
    basis: &[Shell],
    params: &P,
    radial: Radial,
    rcut: f64,
    dcut: f64,
    dd_screen: Option<([T; 3], [T; 3])>,
    out: &mut [T],
) {
    let (slices, norb) = shell_slices(basis);
    let r = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
    let u = [rvec[0] / r, rvec[1] / r, rvec[2] / r];
    let mu = [-u[0], -u[1], -u[2]];
    let fc = cosine_cutoff(r, rcut, dcut);
    let mut tmp = [T::zero(); 25];
    for (ia, &la) in basis.iter().enumerate() {
        for (ib, &lb) in basis.iter().enumerate() {
            let (a0, a1) = slices[ia];
            let (b0, b1) = slices[ib];
            let (na, nb) = (a1 - a0, b1 - b0);
            if la.l() <= lb.l() {
                let mut v: Vec<T> = channels(la, lb)
                    .iter()
                    .map(|c| radial.eval(&params.channel(c)[si][sj], r))
                    .collect();
                if let (Some((scale, shift)), Shell::D, Shell::D) = (dd_screen, la, lb) {
                    for t in 0..3 {
                        v[t] = v[t] * scale[t] + shift[t];
                    }
                }
                block_upper(la, lb, u, &v, &mut tmp);
                for x in 0..na {
                    for y in 0..nb {
                        out[(a0 + x) * norb + b0 + y] = fc * tmp[x * nb + y];
                    }
                }
            } else {
                // <la_i|H|lb_j> = <lb_j|H|la_i>^T, seen from j the bond is -u
                let v: Vec<T> = channels(lb, la)
                    .iter()
                    .map(|c| radial.eval(&params.channel(c)[sj][si], r))
                    .collect();
                block_upper(lb, la, mu, &v, &mut tmp);
                for x in 0..na {
                    for y in 0..nb {
                        out[(a0 + x) * norb + b0 + y] = fc * tmp[y * na + x];
                    }
                }
            }
        }
    }
}

/// Orbital angular momentum (3 matrices `n x n`, row-major) in units of hbar,
/// in the real orbitals and order of the Slater-Koster blocks (as
/// `bopjax.hamiltonian.angular_momentum`): imaginary and antisymmetric.
pub fn angular_momentum(shell: Shell) -> [Vec<crate::complex::C64>; 3] {
    use crate::complex::C64;
    let mut eps = [[[0.0f64; 3]; 3]; 3];
    for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
        eps[a][b][c] = 1.0;
        eps[a][c][b] = -1.0;
    }
    let n = shell.norb();
    let mut out: [Vec<C64>; 3] = [
        vec![C64::default(); n * n],
        vec![C64::default(); n * n],
        vec![C64::default(); n * n],
    ];
    match shell {
        Shell::S => {}
        Shell::P => {
            for k in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        out[k][a * 3 + b] = C64::new(0.0, -eps[k][a][b]);
                    }
                }
            }
        }
        Shell::D => {
            let d = d_tensors();
            for k in 0..3 {
                for m in 0..5 {
                    for nn in 0..5 {
                        // -i sum_ij D_m[i][j] (eps_k D_n + D_n eps_k^T)[i][j]
                        let mut v = 0.0;
                        for i in 0..3 {
                            for j in 0..3 {
                                let mut t = 0.0;
                                for l in 0..3 {
                                    t += eps[k][i][l] * d[nn][l][j] + eps[k][j][l] * d[nn][i][l];
                                }
                                v += d[m][i][j] * t;
                            }
                        }
                        out[k][m * 5 + nn] = C64::new(0.0, -v);
                    }
                }
            }
        }
    }
    out
}
