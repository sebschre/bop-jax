//! Moments -> recursion coefficients -> Chebyshev expansion of the local DOS,
//! for one atom, shell and spin channel (`bopjax.expansion`, after BOPfox
//! `momtorec`, `BOP_GetBandwidth`, `BOP_MomToSigma`, `BOPCHEB_GetChi`).
//!
//! Generic over [`Real`]: with `Dual<MAX_DERIV>` seeded on the moments and the
//! Fermi level it returns the exact partial derivatives of the band filling
//! and the bond energy, which feed the hand-written backward pass through the
//! moments.

use crate::model::{Bandwidth, Kernel, Model, Terminator};
use crate::real::{rmax, rmin, Dual, Real};
use std::f64::consts::PI;

/// Number of directional derivatives of the local chain: moments 0..nmom and
/// the Fermi level.
pub const MAX_DERIV: usize = 16;
pub type D = Dual<MAX_DERIV>;

/// Highest a_n and b_n that `nmom` moments determine.
pub fn n_recursion(nmom: usize) -> (usize, usize) {
    if nmom % 2 == 1 {
        ((nmom - 1) / 2, (nmom - 1) / 2)
    } else {
        (nmom / 2 - 1, nmom / 2)
    }
}

/// a_0..a_amax and b_0..b_bmax (b_0 = 0) from moments mu_0..mu_nmom
/// (Horsfield et al., Mat. Sci. Eng. B 37, 219 (1996)).
pub fn recursion_coefficients<T: Real>(mu: &[T], nmom: usize) -> (Vec<T>, Vec<T>) {
    let (amax, bmax) = n_recursion(nmom);
    let size = bmax + 1;
    let hankel = |j: usize, l: usize| -> T {
        let idx = j + l + 1;
        if idx <= nmom {
            mu[idx]
        } else {
            T::zero()
        }
    };
    let quad = |x: &[T], y: &[T]| -> T {
        let mut s = T::zero();
        for j in 0..size {
            for l in 0..size {
                s += x[j] * hankel(j, l) * y[l];
            }
        }
        s
    };
    let mut c_prev2 = vec![T::zero(); size];
    let mut c_prev = vec![T::zero(); size];
    c_prev[0] = T::cst(1.0);
    let mut a = vec![mu[1]];
    let mut b = vec![T::zero()];
    for n in 1..=bmax {
        let mut c_tilde = vec![T::zero(); size];
        for j in 0..size {
            let shifted = if j > 0 { c_prev[j - 1] } else { T::zero() };
            c_tilde[j] = shifted - a[n - 1] * c_prev[j] - b[n - 1] * c_prev2[j];
        }
        let bn = quad(&c_tilde, &c_prev).sqrt();
        let c: Vec<T> = c_tilde.iter().map(|&x| x / bn).collect();
        b.push(bn);
        if n <= amax {
            a.push(quad(&c, &c));
        }
        c_prev2 = std::mem::replace(&mut c_prev, c);
    }
    (a, b)
}

fn mean<T: Real>(x: &[T]) -> T {
    let mut s = T::zero();
    for &v in x {
        s += v;
    }
    s / x.len() as f64
}

/// a_inf and b_inf; the band is [a_inf - 2 b_inf, a_inf + 2 b_inf].
pub fn band_centre_and_width<T: Real>(a: &[T], b: &[T], m: &Model) -> (T, T) {
    let (amax, bmax) = n_recursion(m.nmom);
    match m.bandwidth {
        Bandwidth::Azero => (a[0], b[1]),
        Bandwidth::Lowest => (if m.nmom == 2 { a[0] } else { a[1] }, b[1]),
        Bandwidth::Highest => (a[amax], b[bmax]),
        Bandwidth::AvganMaxbn => (mean(a), b[bmax]),
        Bandwidth::Global => (T::cst(m.global_ab.0), T::cst(m.global_ab.1)),
        Bandwidth::Averaged => {
            let sq: Vec<T> = b[1..].iter().map(|&x| x * x).collect();
            (mean(a), mean(&sq).sqrt())
        }
        Bandwidth::Findeminemax => {
            let amin = a.iter().copied().reduce(rmin).unwrap();
            let amx = a.iter().copied().reduce(rmax).unwrap();
            let bmx = b[1..].iter().copied().reduce(rmax).unwrap();
            let e_min = amin - bmx * 2.0;
            let e_max = amx + bmx * 2.0;
            ((e_max + e_min) * 0.5, (e_max - e_min) * 0.25)
        }
        Bandwidth::Beerpettiforaoki => beer_pettifor_aoki(&a[..=amax], &b[..=amax]),
        Bandwidth::BeerpettiforaokiBopfox => {
            // What BOPfox evaluates: it drops the last a_n when amax == bmax and
            // reads one element past the array, which holds zero (bopjax keeps
            // this only for comparisons; it breaks invariance under a uniform
            // shift of the energy scale).
            let nrec = (if amax == bmax { amax - 1 } else { amax }) + 1;
            let mut a_pad = a[..nrec].to_vec();
            a_pad.push(T::zero());
            beer_pettifor_aoki(&a_pad, &b[..=nrec])
        }
        Bandwidth::Gershgorin => {
            let lo = (0..bmax)
                .map(|i| a[i] - b[i] - b[i + 1])
                .reduce(rmin)
                .unwrap();
            let hi = (0..bmax)
                .map(|i| a[i] + b[i] + b[i + 1])
                .reduce(rmax)
                .unwrap();
            ((hi + lo) * 0.5, (hi - lo) * 0.25)
        }
    }
}

/// Eigenvalues and eigenvectors of a small symmetric matrix (cyclic Jacobi),
/// row-major; eigenvector k is column k of the returned matrix.
fn jacobi_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j].powi(2))
            .sum();
        let scale: f64 = (0..n * n).map(|q| a[q].powi(2)).sum();
        if off <= 1e-30 * scale.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// Lowest and highest eigenvalue of the symmetric tridiagonal matrix with
/// `diag` and `off`, with exact first derivatives: the eigenvectors come from
/// the values, and `lambda = v^T A v` carries the derivatives
/// (Hellmann-Feynman).  Also returns the eigenvectors.
fn extreme_eigenvalues<T: Real>(diag: &[T], off: &[T]) -> (T, T, Vec<f64>, Vec<f64>) {
    let n = diag.len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = diag[i].val();
        if i + 1 < n {
            a[i * n + i + 1] = off[i].val();
            a[(i + 1) * n + i] = off[i].val();
        }
    }
    let (ev, vec) = jacobi_eigen(a, n);
    let (mut lo, mut hi) = (0, 0);
    for k in 0..n {
        if ev[k] < ev[lo] {
            lo = k;
        }
        if ev[k] > ev[hi] {
            hi = k;
        }
    }
    let column = |k: usize| -> Vec<f64> { (0..n).map(|i| vec[i * n + k]).collect() };
    let rayleigh = |v: &[f64]| -> T {
        let mut s = T::zero();
        for i in 0..n {
            s += diag[i] * (v[i] * v[i]);
            if i + 1 < n {
                s += off[i] * (2.0 * v[i] * v[i + 1]);
            }
        }
        s
    };
    let (vl, vh) = (column(lo), column(hi));
    (rayleigh(&vl), rayleigh(&vh), vl, vh)
}

/// The Beer-Pettifor-Aoki matrix for a trial centre: its diagonal and
/// off-diagonal, and the derivative of the diagonal with respect to the centre.
fn bpa_matrix<T: Real>(a: &[T], b: &[T], centre: T) -> (Vec<T>, Vec<T>, Vec<f64>) {
    let nrec = a.len() - 1;
    let mut diag: Vec<T> = a.iter().map(|&x| (x - centre) * 0.5).collect();
    let mut ddiag = vec![-0.5; nrec + 1];
    diag[0] = a[0] - centre;
    diag[nrec] = a[nrec] - centre;
    ddiag[0] = -1.0;
    ddiag[nrec] = -1.0;
    let mut off: Vec<T> = b[1..=nrec].iter().map(|&x| x * 0.5).collect();
    let r2 = std::f64::consts::SQRT_2;
    off[0] = b[1] / r2;
    off[nrec - 1] = b[nrec] / r2;
    (diag, off, ddiag)
}

/// Band centre for which the BPA matrix has symmetric band edges, and the
/// half width (bopjax `_beer_pettifor_aoki`): bisection on the values, then
/// two Newton steps in `T`, which give the exact first derivative of the
/// root.
fn beer_pettifor_aoki<T: Real>(a: &[T], b: &[T]) -> (T, T) {
    let mismatch = |c: T| -> (T, f64) {
        let (diag, off, ddiag) = bpa_matrix(a, b, c);
        let (lo, hi, vl, vh) = extreme_eigenvalues(&diag, &off);
        let slope: f64 = (0..diag.len())
            .map(|i| ddiag[i] * (vl[i] * vl[i] + vh[i] * vh[i]))
            .sum();
        (hi + lo, slope)
    };
    let av: Vec<f64> = a.iter().map(|x| x.val()).collect();
    let bv: Vec<f64> = b.iter().map(|x| x.val()).collect();
    let amin = av.iter().copied().fold(f64::INFINITY, f64::min);
    let amax = av.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let bmax = bv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let width = 4.0 * bmax + amax - amin + 1.0;
    let (mut lo, mut hi) = (amin - width, amax + width);
    let consts: (Vec<f64>, Vec<f64>) = (av, bv);
    let value = |c: f64| {
        let (diag, off, _) = bpa_matrix(&consts.0, &consts.1, c);
        let (l, h, _, _) = extreme_eigenvalues(&diag, &off);
        h + l
    };
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if value(mid) > 0.0 {
            lo = mid; // the mismatch decreases with the centre
        } else {
            hi = mid;
        }
    }
    let mut centre = T::cst(0.5 * (lo + hi));
    for _ in 0..2 {
        let (f, df) = mismatch(centre);
        if df != 0.0 {
            centre -= f / df;
        }
    }
    let (diag, off, _) = bpa_matrix(a, b, centre);
    let (l, h, _, _) = extreme_eigenvalues(&diag, &off);
    let abs = |x: T| if x.val() < 0.0 { -x } else { x };
    (centre, (abs(h) + abs(l)) * 0.5)
}

pub fn jackson_kernel(nexp: usize) -> Vec<f64> {
    let n_t = (nexp + 1) as f64;
    let g = |n: f64| {
        ((n_t - n + 1.0) * (PI * n / (n_t + 1.0)).cos()
            + (PI * n / (n_t + 1.0)).sin() / (PI / (n_t + 1.0)).tan())
            / (n_t + 1.0)
    };
    let g1 = g(1.0);
    (0..=nexp).map(|k| g((k + 1) as f64) / g1).collect()
}

pub fn kernel(m: &Model) -> Vec<f64> {
    let nexp = m.n_expansion();
    match m.kernel {
        Kernel::None => vec![1.0; nexp + 1],
        Kernel::Jackson => jackson_kernel(nexp),
    }
}

/// Scaled recursion coefficients for chain sites 0..nrec (BOPfox
/// `BOP_MomToSigma`): exact where the moments fix them, terminated beyond.
fn terminated_chain<T: Real>(
    a: &[T],
    b: &[T],
    ainf: T,
    binf: T,
    m: &Model,
    nrec: usize,
) -> (Vec<T>, Vec<T>) {
    let (amax, bmax) = n_recursion(m.nmom);
    let two_b = binf * 2.0;
    let a_ex: Vec<T> = a.iter().map(|&x| (x - ainf) / two_b).collect();
    let b_ex: Vec<T> = b.iter().map(|&x| x / two_b).collect();
    let mut a_hat = vec![T::zero(); nrec + 1];
    let mut b_hat = vec![T::cst(0.5); nrec + 1];
    match m.terminator {
        Terminator::None | Terminator::Constantabn => {}
        Terminator::Weighted => {
            let al = m.alpha_weight;
            let (mut sa, mut swa) = (T::zero(), 0.0);
            for (k, &x) in a.iter().enumerate().take(amax + 1) {
                let w = 1.0 / (al * (amax - k) as f64 + 1.0);
                sa += x * w;
                swa += w;
            }
            let (mut sb, mut swb) = (T::zero(), 0.0);
            for (k, &x) in b.iter().enumerate().take(bmax + 1) {
                let w = 1.0 / (al * (bmax - k) as f64 + 1.0);
                sb += x * w;
                swb += w;
            }
            let av = (sa / swa - ainf) / two_b;
            let bv = sb / swb / two_b;
            a_hat.iter_mut().for_each(|x| *x = av);
            b_hat.iter_mut().for_each(|x| *x = bv);
        }
        Terminator::Oscillation => {
            let (even, odd) = if bmax % 2 == 0 {
                (bmax, bmax - 1)
            } else {
                (bmax - 1, bmax - 2)
            };
            assert!(
                odd >= 2,
                "the oscillation terminator needs at least 8 moments"
            );
            let mut a_pad = vec![T::zero(); bmax + 1];
            a_pad[..=amax].copy_from_slice(&a_ex[..=amax]);
            let a_even = (a_pad[even] + a_pad[even - 2]) * 0.5;
            let a_odd = (a_pad[odd] + a_pad[odd - 2]) * 0.5;
            let b_even = (b_ex[even] + b_ex[even - 2]) * 0.5;
            let b_odd = (b_ex[odd] + b_ex[odd - 2]) * 0.5;
            for i in 0..=nrec {
                a_hat[i] = if i % 2 == 0 { a_even } else { a_odd };
                b_hat[i] = if i % 2 == 0 { b_even } else { b_odd };
            }
        }
    }
    let na = amax.min(nrec) + 1;
    let nb = bmax.min(nrec) + 1;
    a_hat[..na].copy_from_slice(&a_ex[..na]);
    b_hat[..nb].copy_from_slice(&b_ex[..nb]);
    (a_hat, b_hat)
}

/// b_inf, widened where needed so that the scaled recursion chain has its
/// spectrum inside [-1, 1] (`bopjax.expansion.guarded_width`: three passes,
/// a smooth max(1, reach) of width 0.02).  The extreme eigenvalues carry
/// exact first derivatives (Rayleigh quotients of the eigenvectors).
fn guarded_width<T: Real>(a: &[T], b: &[T], ainf: T, binf: T, m: &Model, nrec: usize) -> T {
    let smooth = 0.02;
    let abs = |x: T| if x.val() < 0.0 { -x } else { x };
    let mut binf = binf;
    for _ in 0..3 {
        let (a_hat, b_hat) = terminated_chain(a, b, ainf, binf, m, nrec);
        let (lo, hi, _, _) = extreme_eigenvalues(&a_hat, &b_hat[1..]);
        let reach = if abs(hi).val() >= abs(lo).val() {
            abs(hi)
        } else {
            abs(lo)
        };
        let over = reach - 1.0;
        binf *= (over + (over * over + smooth * smooth).sqrt()) * 0.5 + 1.0;
    }
    binf
}

/// Band centre, half width and Chebyshev moments sigma_0..sigma_nexp of one
/// local DOS from its moments.
pub fn local_expansion<T: Real>(mu: &[T], m: &Model, kern: &[f64]) -> (T, T, Vec<T>) {
    let (a, b) = recursion_coefficients(mu, m.nmom);
    let (ainf, binf) = band_centre_and_width(&a, &b, m);
    let nexp = m.n_expansion();
    let nrec = n_recursion(nexp).1.max(1);
    let binf = if m.bandwidth_guard {
        guarded_width(&a, &b, ainf, binf, m, nrec)
    } else {
        binf
    };
    let (a_hat, b_hat) = terminated_chain(&a, &b, ainf, binf, m, nrec);
    let mut k_prev = vec![T::zero(); nrec + 1];
    let mut k = vec![T::zero(); nrec + 1];
    k[0] = T::cst(1.0);
    let mut sig = vec![k[0] * kern[0]];
    for n in 1..=nexp {
        let mut k_new = vec![T::zero(); nrec + 1];
        for i in 0..=nrec {
            let mut s = a_hat[i] * k[i];
            if i > 0 {
                s += b_hat[i] * k[i - 1];
            }
            if i < nrec {
                s += b_hat[i + 1] * k[i + 1];
            }
            k_new[i] = s * 2.0 - k_prev[i];
        }
        k_prev = std::mem::replace(&mut k, k_new);
        sig.push(k[0] * kern[n]);
    }
    (ainf, binf, sig)
}

/// Reduced response functions chi_0..chi_nchi at the Fermi phase.
fn response_functions<T: Real>(nchi: usize, cos_phi: T) -> Vec<T> {
    let lim = 1.0 - 1e-12;
    let c = if cos_phi.val() > lim {
        T::cst(lim)
    } else if cos_phi.val() < -lim {
        T::cst(-lim)
    } else {
        cos_phi
    };
    let phi = c.acos();
    let mut chi = Vec::with_capacity(nchi + 1);
    chi.push(T::zero());
    chi.push(T::cst(1.0) - (phi - (phi * 2.0).sin() * 0.5) / PI);
    for n in 2..=nchi {
        let nf = n as f64;
        chi.push(
            ((phi * (nf + 1.0)).sin() / (nf + 1.0) - (phi * (nf - 1.0)).sin() / (nf - 1.0)) / PI,
        );
    }
    chi
}

/// Band filling fraction (0..1) and bond energy per orbital and spin,
/// measured from the centre of gravity of the local DOS.
pub fn filling_and_bond_energy<T: Real>(sigma: &[T], ainf: T, binf: T, ef: T, gamma0: T) -> (T, T) {
    let nexp = sigma.len() - 1;
    let chi = response_functions(nexp + 2, (ef - ainf) / (binf * 2.0));
    let mut fill = T::zero();
    let mut bond = T::zero();
    for n in 0..=nexp {
        fill += sigma[n] * chi[n + 1];
        bond += sigma[n] * (chi[n + 2] - gamma0 * chi[n + 1] + chi[n]);
    }
    (fill, bond * binf)
}

/// Gauss-Legendre nodes and weights on [-1, 1].
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        // Tricomi's initial guess, then Newton on P_n
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let pn = if n == 0 {
                1.0
            } else if n == 1 {
                z
            } else {
                p1
            };
            let pm = if n == 1 { 1.0 } else { p0 };
            let dp = n as f64 * (z * pn - pm) / (z * z - 1.0);
            let dz = pn / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                let dp = {
                    let (mut q0, mut q1) = (1.0, z);
                    for k in 2..=n {
                        let q2 = ((2 * k - 1) as f64 * z * q1 - (k - 1) as f64 * q0) / k as f64;
                        q0 = q1;
                        q1 = q2;
                    }
                    let (qn, qm) = if n == 1 { (z, 1.0) } else { (q1, q0) };
                    n as f64 * (z * qn - qm) / (z * z - 1.0)
                };
                w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
                break;
            }
        }
        x[i] = z;
    }
    (x, w)
}

/// Energy offsets and weights for averages over the Fermi kernel
/// `-df/dx` at temperature `kT` (bopjax `fermi_nodes`): Gauss-Legendre in
/// `u = f(x)`, `x = kT ln(1/u - 1)`; the weights sum to 1.
#[derive(Clone, Debug)]
pub struct Smearing {
    pub offsets: Vec<f64>,
    pub weights: Vec<f64>,
    pub kt: f64,
}

pub fn smearing(m: &Model) -> Option<Smearing> {
    if m.kt <= 0.0 {
        return None;
    }
    let (x, w) = gauss_legendre(m.smearing_nodes);
    let offsets = x.iter().map(|&xi| {
        let u = 0.5 * (xi + 1.0);
        m.kt * (1.0 / u - 1.0).ln()
    });
    Some(Smearing {
        offsets: offsets.collect(),
        weights: w.iter().map(|v| 0.5 * v).collect(),
        kt: m.kt,
    })
}

/// Filling and bond energy at the Fermi level, at an electronic temperature
/// with `smear`: the T = 0 filling averaged over the Fermi kernel, and the
/// Mermin free energy `int w(x) [U_0(ef + x) - x N_0(ef + x)] dx` measured from
/// the centre of gravity (bopjax `smeared_filling_and_bond_energy`).
pub fn filling_bond<T: Real>(
    sigma: &[T],
    ainf: T,
    binf: T,
    ef: T,
    gamma0: T,
    smear: Option<&Smearing>,
) -> (T, T) {
    match smear {
        None => filling_and_bond_energy(sigma, ainf, binf, ef, gamma0),
        Some(s) => {
            let (mut fill, mut bond) = (T::zero(), T::zero());
            for (&x, &w) in s.offsets.iter().zip(&s.weights) {
                let (f0, b0) = filling_and_bond_energy(sigma, ainf, binf, ef + x, gamma0);
                fill += f0 * w;
                bond += (b0 - f0 * x) * w;
            }
            (fill, bond)
        }
    }
}

/// Values of the local expansion; enough to locate the Fermi level.
#[derive(Clone, Debug)]
pub struct Local {
    pub ainf: f64,
    pub binf: f64,
    pub sigma: Vec<f64>,
    pub gamma0: f64,
}

impl Local {
    /// An absent shell: no states (the band only keeps the Fermi-level
    /// bracket finite, as in bopjax).
    pub fn absent() -> Self {
        Local {
            ainf: 0.0,
            binf: 1.0,
            sigma: vec![0.0],
            gamma0: 0.0,
        }
    }

    pub fn new(mu: &[f64], m: &Model, kern: &[f64]) -> Self {
        let (ainf, binf, sigma) = local_expansion(&mu[..=m.nmom], m, kern);
        let gamma0 = sigma[1] / kern[1];
        Local {
            ainf,
            binf,
            sigma,
            gamma0,
        }
    }
    pub fn fill(&self, ef: f64, smear: Option<&Smearing>) -> f64 {
        filling_bond(&self.sigma, self.ainf, self.binf, ef, self.gamma0, smear).0
    }
}

/// Filling and bond energy with their derivatives with respect to the
/// moments mu_0..mu_nmom (entries 0..=nmom) and the Fermi level (entry nmom+1).
pub struct LocalDerivs {
    pub fill: D,
    pub bond: D,
}

impl LocalDerivs {
    /// An absent shell: no electrons, no energy.
    pub fn absent() -> Self {
        LocalDerivs {
            fill: D::zero(),
            bond: D::zero(),
        }
    }
}

///
/// `mu` holds the moments up to the walk's `nmom` (that of the whole model);
/// the expansion of `m` (one species) may use fewer.  The derivative layout
/// is always that of `mu`: entry `mu.len()` is the Fermi level.
pub fn local_derivs(
    mu: &[f64],
    ef: f64,
    m: &Model,
    kern: &[f64],
    smear: Option<&Smearing>,
) -> LocalDerivs {
    let nmom = mu.len() - 1;
    let mud: Vec<D> = mu.iter().enumerate().map(|(i, &x)| D::var(x, i)).collect();
    let efd = D::var(ef, nmom + 1);
    let (ainf, binf, sigma) = local_expansion(&mud[..=m.nmom], m, kern);
    let gamma0 = sigma[1] / kern[1];
    let (fill, bond) = filling_bond(&sigma, ainf, binf, efd, gamma0, smear);
    LocalDerivs { fill, bond }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_integrates_polynomials() {
        let (x, w) = gauss_legendre(24);
        assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-13);
        // exact for degree <= 47
        let i: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(46)).sum();
        assert!((i - 2.0 / 47.0).abs() < 1e-13, "{i}");
    }
}
