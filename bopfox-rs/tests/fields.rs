#![allow(clippy::needless_range_loop)]
//! External fields: finite differences of dE/dh, dE/db and of forces, shifts
//! and axis gradients in a field; the self-consistent moment as the field
//! derivative of the self-consistent energy.

use bopfox_rs::{Bop, Cell, Model, ScfOptions, Topology, G_SPIN, MU_B};
use serde_json::json;

fn iron(spin: &str, xi: f64) -> Model {
    let ch = |p: f64| json!([[[p, 4.38348, 100.0, 1.0, 2.48]]]);
    serde_json::from_value(json!({
        "nmom": 7, "bandwidth": "averaged", "rcut": 3.4, "dcut": 0.4, "rcut2": 3.4, "dcut2": 0.4,
        "basis": ["d"], "spin": spin, "radial": "v0",
        "spin_orbit": xi != 0.0, "spin_orbit_xi": [[xi]],
        "onsite": [[0.0]], "n0": [[6.88178]], "hubbard": [8.0], "stoner": [[1.16487]],
        "bond": {"dds": ch(-1.14744), "ddp": ch(0.51229), "ddd": ch(-0.03851)},
        "repulsion": {"form": "v0", "pair": [[[0.64199, 9.98837, 2.48]]],
                      "density": [[[1.0, 2.79649, 2.48]]], "embed": [[-1.51835, 0.5]]}
    }))
    .unwrap()
}

fn setup() -> (Vec<[f64; 3]>, Cell, Vec<f64>) {
    let a = 2.83;
    let pos = vec![
        [0.05, -0.03, 0.02],
        [0.5 * a, 0.5 * a + 0.06, 0.5 * a - 0.04],
    ];
    let x = (0..2)
        .flat_map(|i| [-1.2 - 0.01 * i as f64, 1.2 + 0.02 * i as f64])
        .collect();
    (pos, Some([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]), x)
}

fn d5(f: impl Fn(f64) -> f64) -> f64 {
    let h = 1e-5;
    (-f(2.0 * h) + 8.0 * f(h) - 8.0 * f(-h) + f(-2.0 * h)) / (12.0 * h)
}

const H: [[f64; 3]; 2] = [[0.02, -0.01, 0.03], [-0.015, 0.025, 0.01]];
const B: [[f64; 3]; 2] = [[0.01, 0.02, -0.015], [0.03, -0.01, 0.02]];

/// All derivatives of a model in the fields `h` (and `b`) by finite differences.
fn check(m: &Model, axes: Option<&[[f64; 3]]>, orbital: bool) {
    let (pos, cell, x) = setup();
    let topo = Topology::build(&pos, &[0; 2], cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    let make = |h: &[[f64; 3]], b: &[[f64; 3]], a: Option<&[[f64; 3]]>| {
        let mut bop = Bop::new(m, &topo);
        if let Some(a) = a {
            bop = bop.with_axes(a);
        }
        bop.with_fields(Some(h), orbital.then_some(b))
    };
    let r = make(&H, &B, axes).compute(&pos, &cell, &x, true);
    let tol = |v: f64| 1e-7 * (1.0 + v.abs());
    for i in 0..2 {
        for c in 0..3 {
            let fd = d5(|t| {
                let mut h = H;
                h[i][c] += t;
                make(&h, &B, axes).compute(&pos, &cell, &x, false).energy
            });
            let g = r.spin_field_gradient[i][c];
            assert!((g - fd).abs() < tol(fd), "dE/dh {i} {c}: {g} vs {fd}");
            if orbital {
                let fd = d5(|t| {
                    let mut b = B;
                    b[i][c] += t;
                    make(&H, &b, axes).compute(&pos, &cell, &x, false).energy
                });
                let g = r.orbital_field_gradient[i][c];
                assert!((g - fd).abs() < tol(fd), "dE/db {i} {c}: {g} vs {fd}");
            }
            let fd = -d5(|t| {
                let mut p = pos.clone();
                p[i][c] += t;
                make(&H, &B, axes).compute(&p, &cell, &x, false).energy
            });
            assert!((r.forces[i][c] - fd).abs() < tol(fd), "force {i} {c}");
            if let Some(a0) = axes {
                let fd = d5(|t| {
                    let mut a = a0.to_vec();
                    a[i][c] += t;
                    make(&H, &B, Some(&a))
                        .compute(&pos, &cell, &x, false)
                        .energy
                });
                let g = r.axes_gradient[i][c];
                assert!((g - fd).abs() < tol(fd), "axis {i} {c}: {g} vs {fd}");
            }
        }
    }
    for q in 0..x.len() {
        let fd = d5(|t| {
            let mut y = x.clone();
            y[q] += t;
            make(&H, &B, axes).compute(&pos, &cell, &y, false).energy
        });
        let g = r.shift_gradient[q / 2][q % 2][0];
        assert!((g - fd).abs() < tol(fd), "shift {q}: {g} vs {fd}");
    }
}

#[test]
fn collinear_field() {
    check(&iron("collinear", 0.0), None, false);
}

#[test]
fn collinear_field_couples_only_along_z() {
    let (pos, cell, x) = setup();
    let m = iron("collinear", 0.0);
    let topo = Topology::build(&pos, &[0; 2], cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    let e0 = Bop::new(&m, &topo).compute(&pos, &cell, &x, false).energy;
    let e1 = Bop::new(&m, &topo)
        .with_fields(Some(&[[0.1, -0.2, 0.0]; 2]), None)
        .compute(&pos, &cell, &x, false)
        .energy;
    assert_eq!(e0, e1);
}

#[test]
fn noncollinear_spin_field() {
    check(
        &iron("noncollinear", 0.0),
        Some(&[[0.3, -0.2, 1.0], [-0.4, 0.5, 0.8]]),
        false,
    );
}

#[test]
fn noncollinear_orbital_field_without_spin_orbit() {
    check(
        &iron("noncollinear", 0.0),
        Some(&[[0.3, -0.2, 1.0], [-0.4, 0.5, 0.8]]),
        true,
    );
}

#[test]
fn noncollinear_fields_with_spin_orbit() {
    check(
        &iron("noncollinear", 0.07),
        Some(&[[0.3, -0.2, 1.0], [-0.4, 0.5, 0.8]]),
        true,
    );
}

/// At self-consistency `-(1/mu_B) dE/dB = g/2 dE/dh + dE/db` is the total
/// derivative of the self-consistent energy with respect to the field, for
/// both conditions (the potential one with the implicit term).
#[test]
fn self_consistent_moment_is_field_derivative() {
    let (pos, cell, x) = setup();
    let m = iron("noncollinear", 0.07);
    let topo = Topology::build(&pos, &[0; 2], cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    let axes = [[0.3, -0.2, 1.0], [-0.4, 0.5, 0.8]];
    let field = [[2.0, -1.0, 5.0]; 2]; // Tesla
    for opt in [
        ScfOptions {
            tol: 1e-12,
            ..Default::default()
        },
        ScfOptions {
            tol: 1e-12,
            ..ScfOptions::potential()
        },
    ] {
        let run = |f: &[[f64; 3]], s0: &[f64]| {
            let r = Bop::new(&m, &topo)
                .with_axes(&axes)
                .with_magnetic_field(f)
                .scf(&pos, &cell, Some(s0), &opt);
            assert!(r.converged);
            r
        };
        let r = run(&field, &x);
        for (i, c) in [(0usize, 2usize), (1, 0)] {
            let moment = 0.5 * G_SPIN * r.results.spin_field_gradient[i][c]
                + r.results.orbital_field_gradient[i][c];
            let e = |t: f64| {
                let mut f = field;
                f[i][c] += t;
                run(&f, &r.shifts).results.energy
            };
            let h = 1.0; // Tesla
            let fd = -(-e(2.0 * h) + 8.0 * e(h) - 8.0 * e(-h) + e(-2.0 * h)) / (12.0 * h) / MU_B;
            assert!(
                (moment - fd).abs() < 1e-5,
                "{:?} moment {i} {c}: {moment} vs {fd}",
                opt.condition
            );
        }
    }
}
