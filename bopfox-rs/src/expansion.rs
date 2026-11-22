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

/// Band centre, half width and Chebyshev moments sigma_0..sigma_nexp of one
/// local DOS from its moments.
pub fn local_expansion<T: Real>(mu: &[T], m: &Model, kern: &[f64]) -> (T, T, Vec<T>) {
    let (a, b) = recursion_coefficients(mu, m.nmom);
    let (ainf, binf) = band_centre_and_width(&a, &b, m);
    let nexp = m.n_expansion();
    let nrec = n_recursion(nexp).1.max(1);
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

/// Values of the local expansion; enough to locate the Fermi level.
#[derive(Clone, Debug)]
pub struct Local {
    pub ainf: f64,
    pub binf: f64,
    pub sigma: Vec<f64>,
    pub gamma0: f64,
}

impl Local {
    pub fn new(mu: &[f64], m: &Model, kern: &[f64]) -> Self {
        let (ainf, binf, sigma) = local_expansion(mu, m, kern);
        let gamma0 = sigma[1] / kern[1];
        Local {
            ainf,
            binf,
            sigma,
            gamma0,
        }
    }
    pub fn fill(&self, ef: f64) -> f64 {
        filling_and_bond_energy(&self.sigma, self.ainf, self.binf, ef, self.gamma0).0
    }
}

/// Filling and bond energy with their derivatives with respect to the
/// moments mu_0..mu_nmom (entries 0..=nmom) and the Fermi level (entry nmom+1).
pub struct LocalDerivs {
    pub fill: D,
    pub bond: D,
}

pub fn local_derivs(mu: &[f64], ef: f64, m: &Model, kern: &[f64]) -> LocalDerivs {
    let nmom = m.nmom;
    let mud: Vec<D> = mu.iter().enumerate().map(|(i, &x)| D::var(x, i)).collect();
    let efd = D::var(ef, nmom + 1);
    let (ainf, binf, sigma) = local_expansion(&mud, m, kern);
    let gamma0 = sigma[1] / kern[1];
    let (fill, bond) = filling_and_bond_energy(&sigma, ainf, binf, efd, gamma0);
    LocalDerivs { fill, bond }
}
