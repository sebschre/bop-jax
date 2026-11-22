//! A minimal complex number for the spinor walk of non-collinear magnetism.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };
pub const I: C64 = C64 { re: 0.0, im: 1.0 };

impl C64 {
    #[inline(always)]
    pub fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }
    #[inline(always)]
    pub fn conj(self) -> Self {
        C64 {
            re: self.re,
            im: -self.im,
        }
    }
    /// Re(conj(self) * o)
    #[inline(always)]
    pub fn dot_re(self, o: C64) -> f64 {
        self.re * o.re + self.im * o.im
    }
}

impl Add for C64 {
    type Output = C64;
    #[inline(always)]
    fn add(self, o: C64) -> C64 {
        C64 {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}
impl Sub for C64 {
    type Output = C64;
    #[inline(always)]
    fn sub(self, o: C64) -> C64 {
        C64 {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}
impl Mul for C64 {
    type Output = C64;
    #[inline(always)]
    fn mul(self, o: C64) -> C64 {
        C64 {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}
impl Mul<f64> for C64 {
    type Output = C64;
    #[inline(always)]
    fn mul(self, o: f64) -> C64 {
        C64 {
            re: self.re * o,
            im: self.im * o,
        }
    }
}
impl Neg for C64 {
    type Output = C64;
    #[inline(always)]
    fn neg(self) -> C64 {
        C64 {
            re: -self.re,
            im: -self.im,
        }
    }
}
impl AddAssign for C64 {
    #[inline(always)]
    fn add_assign(&mut self, o: C64) {
        self.re += o.re;
        self.im += o.im;
    }
}
impl SubAssign for C64 {
    #[inline(always)]
    fn sub_assign(&mut self, o: C64) {
        self.re -= o.re;
        self.im -= o.im;
    }
}
impl MulAssign<f64> for C64 {
    #[inline(always)]
    fn mul_assign(&mut self, o: f64) {
        self.re *= o;
        self.im *= o;
    }
}

/// Pauli matrices sigma_x, sigma_y, sigma_z, row-major 2x2.
pub fn pauli() -> [[C64; 4]; 3] {
    [
        [ZERO, ONE, ONE, ZERO],
        [ZERO, -I, I, ZERO],
        [ONE, ZERO, ZERO, -ONE],
    ]
}

/// Projectors on spin up and down along `e` (unit vector), 2x2 row-major.
pub fn spin_projectors(e: [f64; 3]) -> [[C64; 4]; 2] {
    let p = pauli();
    let mut out = [[ZERO; 4]; 2];
    for (s, sign) in [(0usize, 1.0), (1, -1.0)] {
        for q in 0..4 {
            let mut v = if q == 0 || q == 3 { ONE * 0.5 } else { ZERO };
            for x in 0..3 {
                v += p[x][q] * (0.5 * sign * e[x]);
            }
            out[s][q] = v;
        }
    }
    out
}

/// d P_s / d a_x for the axis `a` (not necessarily normalised): `[x][s]`.
pub fn projector_derivatives(a: [f64; 3]) -> [[[C64; 4]; 2]; 3] {
    let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let e = [a[0] / norm, a[1] / norm, a[2] / norm];
    let p = pauli();
    let mut out = [[[ZERO; 4]; 2]; 3];
    for x in 0..3 {
        for (s, sign) in [(0usize, 1.0), (1, -1.0)] {
            for q in 0..4 {
                let mut v = ZERO;
                for y in 0..3 {
                    let de = ((x == y) as u8 as f64 - e[x] * e[y]) / norm;
                    v += p[y][q] * (0.5 * sign * de);
                }
                out[x][s][q] = v;
            }
        }
    }
    out
}
