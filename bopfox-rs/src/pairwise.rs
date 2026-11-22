//! Coulomb integrals between atoms (`bopjax.pairwise`: BOPfox `buildhop`),
//! generic over [`Real`] for their exact radial derivatives.

use crate::real::Real;

pub const COULOMB_CONSTANT: f64 = 14.4; // eV Angstrom

/// Seventh-order polynomial switching the Coulomb integral off at `r2`
/// (bopjax `pairwise.taper`, coefficients as there).  With `r1 = 0` it is the
/// usual switch from 1 at r = 0 to 0 at `r2`; for `r1 > 0` bopjax's
/// coefficients do not give 1 at `r1` (they reach hundreds between `r1` and
/// `r2`), so use `r1 = 0`.
pub fn taper<T: Real>(r1: f64, r2: f64, r: T) -> T {
    if r.val() > r2 {
        return T::zero();
    }
    let d = (r2 - r1).powi(7);
    let t = [
        (-35.0 * r1.powi(3) * r2.powi(4)
            + 21.0 * r1.powi(2) * r2.powi(5)
            + 7.0 * r1 * r2.powi(6)
            + r2.powi(7))
            / d,
        140.0 * r1.powi(3) * r2.powi(3) / d,
        -210.0 * (r1.powi(3) * r2.powi(2) + r1.powi(2) * r2.powi(3)) / d,
        140.0 * (r1.powi(3) * r2 + 3.0 * r1.powi(2) * r2.powi(2) + r1 * r2.powi(3)) / d,
        -35.0 * (r1.powi(3) + 9.0 * r1.powi(2) * r2 + r2.powi(3)) / d,
        84.0 * (r1.powi(2) + 3.0 * r1 * r2 + r2.powi(2)) / d,
        -70.0 * (r1 + r2) / d,
        20.0 / d,
    ];
    let mut p = T::cst(t[7]);
    for k in (0..7).rev() {
        p = p * r + t[k];
    }
    p
}

/// Screened Coulomb integral between two exponential charge clouds whose
/// widths are set by the onsite integrals `j_i`, `j_j`.
pub fn frauenheim<T: Real>(j_i: f64, j_j: f64, r: T) -> T {
    let k = COULOMB_CONSTANT;
    let (a, b) = (3.2 * j_i, 3.2 * j_j);
    let kr = r.powf(-1.0) * k;
    if (a - b).abs() < 1e-12 {
        let x = r / k;
        return kr
            - (-x * b).exp()
                * (kr + 11.0 / 16.0 * b + x * (3.0 / 16.0 * b * b) + x * x * (b.powi(3) / 48.0));
    }
    let c = a * a - b * b;
    // bopjax: k/r - e^{-a r/k} (b^4 a / (2 c^2) - k b^4 (b^2 - 3a^2) / (c^3 r))
    //             - e^{-b r/k} (a^4 b / (2 c^2) + k a^4 (a^2 - 3b^2) / (c^3 r))
    let x = r / k;
    kr - (-x * a).exp()
        * (T::cst(0.5 * b.powi(4) * a / (c * c))
            - kr * (b.powi(4) * (b * b - 3.0 * a * a) / c.powi(3)))
        - (-x * b).exp()
            * (T::cst(0.5 * a.powi(4) * b / (c * c))
                + kr * (a.powi(4) * (a * a - 3.0 * b * b) / c.powi(3)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taper_switches_off_with_zero_inner_radius() {
        // as in bopjax: for R1 = 0 the polynomial is the usual seventh-order
        // switch from 1 at r = 0 to 0 at R2 (for R1 > 0 bopjax's coefficients
        // do not give 1 at R1; kept identical for comparison)
        assert!((taper(0.0, 5.0, 0.0) - 1.0).abs() < 1e-14);
        assert!(taper(0.0, 5.0, 5.0).abs() < 1e-12);
        assert!(taper(0.0, 5.0, 6.0) == 0.0);
        let x: f64 = 0.3;
        let smooth =
            1.0 - 35.0 * x.powi(4) + 84.0 * x.powi(5) - 70.0 * x.powi(6) + 20.0 * x.powi(7);
        assert!((taper(0.0, 5.0, 1.5) - smooth).abs() < 1e-12);
    }
}
