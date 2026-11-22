#![allow(clippy::needless_range_loop)]
//! Non-collinear magnetism: the collinear limit, and finite differences of
//! every derivative (positions, strain, shifts, spin axes) with and without
//! spin-orbit coupling.

use bopfox_rs::{Bop, Cell, Model, Topology};
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

fn cell_atoms() -> (Vec<[f64; 3]>, Cell) {
    let a = 2.83;
    let pos = vec![
        [0.05, -0.03, 0.02],
        [0.5 * a, 0.5 * a + 0.06, 0.5 * a - 0.04],
    ];
    (pos, Some([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]))
}

fn split(n: usize, s: f64) -> Vec<f64> {
    (0..n)
        .flat_map(|i| [-s - 0.01 * i as f64, s + 0.02 * i as f64])
        .collect()
}

#[test]
fn collinear_limit_is_exact() {
    let (pos, cell) = cell_atoms();
    let n = pos.len();
    let (mc, mn) = (iron("collinear", 0.0), iron("noncollinear", 0.0));
    let topo = Topology::build(&pos, &[0; 2], cell.as_ref(), mc.rcut, mc.nmom, mc.rcut2);
    let x = split(n, 1.2);
    let a = Bop::new(&mc, &topo).compute(&pos, &cell, &x, true);
    let b = Bop::new(&mn, &topo)
        .with_axes(&[[0.0, 0.0, 2.0]; 2])
        .compute(&pos, &cell, &x, true);
    assert!(
        (a.energy - b.energy).abs() < 1e-12,
        "{} {}",
        a.energy,
        b.energy
    );
    for i in 0..n {
        for s in 0..2 {
            assert!((a.shift_gradient[i][s][0] - b.shift_gradient[i][s][0]).abs() < 1e-11);
        }
        for x in 0..3 {
            assert!((a.forces[i][x] - b.forces[i][x]).abs() < 1e-11);
            // rotating an axis of a collinear state costs nothing to first order
            assert!(b.axes_gradient[i][x].abs() < 1e-11, "{:?}", b.axes_gradient);
        }
    }
}

fn d5(f: impl Fn(f64) -> f64) -> f64 {
    let h = 1e-5;
    (-f(2.0 * h) + 8.0 * f(h) - 8.0 * f(-h) + f(-2.0 * h)) / (12.0 * h)
}

fn check(
    m: &Model,
    pos: &[[f64; 3]],
    species: &[usize],
    cell: &Cell,
    axes: &[[f64; 3]],
    x: &[f64],
) {
    let topo = Topology::build(pos, species, cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    let bop = Bop::new(m, &topo).with_axes(axes);
    let r = bop.compute(pos, cell, x, true);
    let tol = |v: f64| 1e-7 * (1.0 + v.abs());
    let n = pos.len();
    for i in 0..n {
        for c in 0..3 {
            let fd = -d5(|t| {
                let mut p = pos.to_vec();
                p[i][c] += t;
                bop.compute(&p, cell, x, false).energy
            });
            assert!(
                (r.forces[i][c] - fd).abs() < tol(fd),
                "force {i} {c}: {} {fd}",
                r.forces[i][c]
            );
            let fd = d5(|t| {
                let mut a = axes.to_vec();
                a[i][c] += t;
                Bop::new(m, &topo)
                    .with_axes(&a)
                    .compute(pos, cell, x, false)
                    .energy
            });
            assert!(
                (r.axes_gradient[i][c] - fd).abs() < tol(fd),
                "axis {i} {c}: {} {fd}",
                r.axes_gradient[i][c]
            );
        }
    }
    for q in 0..x.len() {
        let fd = d5(|t| {
            let mut y = x.to_vec();
            y[q] += t;
            bop.compute(pos, cell, &y, false).energy
        });
        let nsh = r.shift_gradient[0][0].len();
        let g = r.shift_gradient[q / (2 * nsh)][(q / nsh) % 2][q % nsh];
        assert!((g - fd).abs() < tol(fd), "shift {q}: {g} {fd}");
    }
    if let Some(c) = cell {
        let fd = d5(|t| {
            let def = |v: [f64; 3]| [v[0], v[1] + 0.5 * t * v[2], v[2] + 0.5 * t * v[1]];
            let p: Vec<[f64; 3]> = pos.iter().map(|&v| def(v)).collect();
            bop.compute(&p, &Some([def(c[0]), def(c[1]), def(c[2])]), x, false)
                .energy
        });
        // symmetric strain eps_yz = eps_zy = t / 2: dE/dt is the symmetrised
        // virial (with spin-orbit coupling and fixed spin axes a rotation of
        // the lattice alone changes the energy, so the virial is symmetrised)
        let w = r.virial[2][1];
        assert!((w - fd).abs() < tol(fd) * 10.0, "virial {w} {fd}");
    }
}

#[test]
fn noncollinear_derivatives_without_spin_orbit() {
    let (pos, cell) = cell_atoms();
    let axes = [[0.3, -0.2, 1.0], [-0.4, 0.5, 0.8]];
    check(
        &iron("noncollinear", 0.0),
        &pos,
        &[0; 2],
        &cell,
        &axes,
        &split(2, 1.2),
    );
}

#[test]
fn noncollinear_derivatives_with_spin_orbit() {
    let (pos, cell) = cell_atoms();
    let axes = [[0.3, -0.2, 1.0], [-0.4, 0.5, 0.8]];
    check(
        &iron("noncollinear", 0.08),
        &pos,
        &[0; 2],
        &cell,
        &axes,
        &split(2, 1.2),
    );
}

#[test]
fn spd_two_species_cluster_with_spin_orbit() {
    let mut bond = serde_json::Map::new();
    let names = [
        "sss", "sps", "sds", "pps", "ppp", "pds", "pdp", "dds", "ddp", "ddd",
    ];
    for (k, c) in names.iter().enumerate() {
        let pre = [-0.5, 1.0, -0.3, 1.5, -0.3, -0.4, 0.2, -1.1, 0.5, -0.05][k];
        let row = |a: f64| json!([pre * a, 3.0 + 0.1 * k as f64, 3.0, 2.0, 2.5]);
        bond.insert(
            c.to_string(),
            json!([[row(1.0), row(0.9)], [row(1.1), row(0.8)]]),
        );
    }
    let m: Model = serde_json::from_value(json!({
        "nmom": 5, "rcut": 3.4, "dcut": 0.4, "basis": ["s", "p", "d"], "radial": "v0",
        "spin": "noncollinear", "spin_orbit": true, "spin_orbit_xi": [[0.0, 0.2, 0.07], [0.0, 0.1, 0.05]],
        "bandwidth": "averaged",
        "onsite": [[0.5, 6.0, 0.0], [1.0, 7.0, 0.3]], "n0": [[1.0, 0.0, 6.5], [1.0, 0.2, 7.0]],
        "hubbard": [8.0, 6.0], "stoner": [[0.0, 0.0, 1.1], [0.1, 0.0, 0.9]], "bond": bond
    }))
    .unwrap();
    let pos = vec![
        [0.0, 0.0, 0.0],
        [1.45, 1.4, 0.05],
        [1.4, -0.1, 1.5],
        [-0.05, 1.42, 1.38],
    ];
    let axes = [
        [0.0, 0.0, 1.0],
        [0.5, 0.1, 0.9],
        [-0.3, 0.8, 0.4],
        [0.2, -0.6, -0.7],
    ];
    let x: Vec<f64> = (0..4 * 2 * 3)
        .map(|q| if (q / 3) % 2 == 0 { -0.4 } else { 0.4 } + 0.01 * (q % 7) as f64)
        .collect();
    check(&m, &pos, &[0, 1, 0, 1], &None, &axes, &x);
}

/// At self-consistency the axis gradient is the total derivative of the
/// self-consistent energy, for both conditions (with the implicit term for
/// the potential condition).
#[test]
fn self_consistent_axis_gradient_is_total_derivative() {
    use bopfox_rs::ScfOptions;
    let (pos, cell) = cell_atoms();
    let m = iron("noncollinear", 0.08);
    let topo = Topology::build(&pos, &[0; 2], cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    let axes = [[0.3, -0.2, 1.0], [-0.4, 0.5, 0.8]];
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
        let r = Bop::new(&m, &topo)
            .with_axes(&axes)
            .scf(&pos, &cell, Some(&split(2, 1.2)), &opt);
        assert!(r.converged, "{:?}", opt.condition);
        for (i, c) in [(0usize, 0usize), (1, 1)] {
            let e = |t: f64| {
                let mut a = axes;
                a[i][c] += t;
                let s = Bop::new(&m, &topo)
                    .with_axes(&a)
                    .scf(&pos, &cell, Some(&r.shifts), &opt);
                assert!(s.converged);
                s.results.energy
            };
            let h = 1e-4;
            let fd = (-e(2.0 * h) + 8.0 * e(h) - 8.0 * e(-h) + e(-2.0 * h)) / (12.0 * h);
            let g = r.results.axes_gradient[i][c];
            assert!(
                (g - fd).abs() < 1e-7,
                "{:?} axis {i} {c}: {g} vs {fd}",
                opt.condition
            );
        }
    }
}
