//! A scalar trait over `f64` and forward-mode dual numbers.
//!
//! The expensive parts of the BOP (moments of the Hamiltonian and their
//! derivatives with respect to the bond integrals) are differentiated by hand,
//! see `moments.rs`.  The small per-bond and per-atom functions (Slater-Koster
//! blocks of one bond, the moments -> recursion -> Chebyshev chain of one atom,
//! a pair term) are written once, generic over [`Real`], and evaluated with
//! [`Dual`] numbers when their exact derivatives are needed.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    fn cst(x: f64) -> Self;
    fn val(self) -> f64;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, p: f64) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn acos(self) -> Self;
    /// `self ** p` with a differentiable exponent (`self > 0`).
    fn powr(self, p: Self) -> Self {
        (self.ln() * p).exp()
    }
    fn zero() -> Self {
        Self::cst(0.0)
    }
}

impl Real for f64 {
    #[inline]
    fn cst(x: f64) -> Self {
        x
    }
    #[inline]
    fn val(self) -> f64 {
        self
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        f64::powf(self, p)
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
}

/// Value and `N` directional derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<const N: usize> {
    pub v: f64,
    pub d: [f64; N],
}

impl<const N: usize> Dual<N> {
    #[inline]
    pub fn var(v: f64, i: usize) -> Self {
        let mut d = [0.0; N];
        d[i] = 1.0;
        Dual { v, d }
    }
    #[inline]
    fn chain(self, f: f64, df: f64) -> Self {
        let mut d = self.d;
        for x in d.iter_mut() {
            *x *= df;
        }
        Dual { v: f, d }
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, o: Self) -> Self {
        self.v += o.v;
        for i in 0..N {
            self.d[i] += o.d[i];
        }
        self
    }
}
impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: Self) -> Self {
        self.v -= o.v;
        for i in 0..N {
            self.d[i] -= o.d[i];
        }
        self
    }
}
impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        let mut d = [0.0; N];
        for i in 0..N {
            d[i] = self.d[i] * o.v + self.v * o.d[i];
        }
        Dual { v: self.v * o.v, d }
    }
}
impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let q = self.v * inv;
        let mut d = [0.0; N];
        for i in 0..N {
            d[i] = (self.d[i] - q * o.d[i]) * inv;
        }
        Dual { v: q, d }
    }
}
impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        self.v = -self.v;
        for x in self.d.iter_mut() {
            *x = -*x;
        }
        self
    }
}
impl<const N: usize> AddAssign for Dual<N> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl<const N: usize> SubAssign for Dual<N> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl<const N: usize> MulAssign for Dual<N> {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, o: f64) -> Self {
        self.v += o;
        self
    }
}
impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: f64) -> Self {
        self.v -= o;
        self
    }
}
impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(mut self, o: f64) -> Self {
        self.v *= o;
        for x in self.d.iter_mut() {
            *x *= o;
        }
        self
    }
}
impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: f64) -> Self {
        self * (1.0 / o)
    }
}

impl<const N: usize> Real for Dual<N> {
    #[inline]
    fn cst(x: f64) -> Self {
        Dual { v: x, d: [0.0; N] }
    }
    #[inline]
    fn val(self) -> f64 {
        self.v
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s)
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        self.chain(self.v.ln(), 1.0 / self.v)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        let f = self.v.powf(p);
        let df = if p == 0.0 {
            0.0
        } else {
            p * self.v.powf(p - 1.0)
        };
        self.chain(f, df)
    }
    #[inline]
    fn sin(self) -> Self {
        self.chain(self.v.sin(), self.v.cos())
    }
    #[inline]
    fn cos(self) -> Self {
        self.chain(self.v.cos(), -self.v.sin())
    }
    #[inline]
    fn acos(self) -> Self {
        self.chain(self.v.acos(), -1.0 / (1.0 - self.v * self.v).sqrt())
    }
}

/// Larger of two values; the derivative follows the chosen argument.
#[inline]
pub fn rmax<T: Real>(a: T, b: T) -> T {
    if b.val() > a.val() {
        b
    } else {
        a
    }
}

/// Smaller of two values; the derivative follows the chosen argument.
#[inline]
pub fn rmin<T: Real>(a: T, b: T) -> T {
    if b.val() < a.val() {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_derivatives() {
        let x = Dual::<2>::var(0.7, 0);
        let y = Dual::<2>::var(1.3, 1);
        let f = (x * y).sin() / y.sqrt() + x.powf(3.0) * y.exp() - (x / y).acos();
        let h = 1e-6;
        let g = |a: f64, b: f64| (a * b).sin() / b.sqrt() + a.powi(3) * b.exp() - (a / b).acos();
        let dx = (g(0.7 + h, 1.3) - g(0.7 - h, 1.3)) / (2.0 * h);
        let dy = (g(0.7, 1.3 + h) - g(0.7, 1.3 - h)) / (2.0 * h);
        assert!((f.d[0] - dx).abs() < 1e-8);
        assert!((f.d[1] - dy).abs() < 1e-8);
    }
}
