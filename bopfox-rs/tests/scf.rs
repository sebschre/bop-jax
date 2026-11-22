#![allow(clippy::needless_range_loop)]

//! Self-consistency against BOPfox and by stationarity.

use bopfox_rs::{Bop, Cell, Model, ScfOptions, Topology};
use serde_json::json;

const A: f64 = 3.5;

fn distorted_fcc() -> Vec<[f64; 3]> {
    let mut p = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.5 * A, 0.5 * A],
        [0.5 * A, 0.0, 0.5 * A],
        [0.5 * A, 0.5 * A, 0.0],
    ];
    for x in 0..3 {
        p[0][x] += [0.15, -0.08, 0.05][x];
        p[2][x] += [-0.05, 0.10, 0.02][x];
    }
    p
}

fn cubic(a: f64) -> Cell {
    Some([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
}

fn canonical_d(nmom: usize) -> Model {
    let ch = |p: f64| json!([[[p, 2.081, 5.0]]]);
    serde_json::from_value(json!({
        "nmom": nmom, "rcut": 3.3, "dcut": 0.5, "basis": ["d"],
        "onsite": [[0.0]], "n0": [[4.8]], "hubbard": [8.0],
        "bond": {"dds": ch(-1.458), "ddp": ch(0.972), "ddd": ch(-0.243)}
    }))
    .unwrap()
}

#[test]
fn scf_matches_bopfox() {
    let m = canonical_d(6);
    let pos = distorted_fcc();
    let cell = cubic(A);
    let topo = Topology::build(&pos, &[0; 4], cell.as_ref(), m.rcut, m.nmom, None);
    let bop = Bop::new(&m, &topo);
    let r = bop.scf(
        &pos,
        &cell,
        None,
        &ScfOptions {
            tol: 1e-12,
            ..Default::default()
        },
    );
    println!("history {:?}", r.history);
    assert!(r.converged);
    assert!(r.history.len() <= 15, "{} evaluations", r.history.len());
    assert!((r.results.energy / 4.0 + 6.011_248_544_843_719).abs() < 1e-10);
    let shifts = [
        -3.2427167768e-3,
        4.1529574320e-3,
        -1.3968614396e-3,
        4.8662078441e-4,
    ];
    let charge = [
        2.001241198650e-4,
        -5.18733079637e-5,
        2.22429422658e-5,
        -1.704937541867e-4,
    ];
    for i in 0..4 {
        assert!((r.shifts[i] - shifts[i]).abs() < 1e-9, "{:?}", r.shifts);
        assert!((r.results.charge[i] - charge[i]).abs() < 1e-9);
    }
    let f_bopfox = [
        [4.9859884934, -2.4231962888, 1.2439578230],
        [0.8291859687, -0.3280010053, -0.7088860647],
        [-3.5410851762, 2.7658615584, -0.2853256569],
        [-2.2740892253, -0.0146642081, -0.2497461152],
    ];
    for i in 0..4 {
        for x in 0..3 {
            assert!((r.results.forces[i][x] - f_bopfox[i][x]).abs() < 1e-6);
        }
    }
}

/// At self-consistency the force is the total derivative of the SCF energy.
#[test]
fn forces_are_total_derivative_of_scf_energy() {
    let m = canonical_d(9);
    let pos = distorted_fcc();
    let cell = cubic(A);
    let topo = Topology::build(&pos, &[0; 4], cell.as_ref(), m.rcut, m.nmom, None);
    let bop = Bop::new(&m, &topo);
    let opt = ScfOptions {
        tol: 1e-12,
        ..Default::default()
    };
    let r = bop.scf(&pos, &cell, None, &opt);
    assert!(r.converged);
    let e = |d: f64| {
        let mut p = pos.clone();
        p[0][0] += d;
        let s = bop.scf(&p, &cell, Some(&r.shifts), &opt);
        if !s.converged {
            println!("inner {:?}", s.history);
        }
        assert!(s.converged);
        s.results.energy
    };
    let h = 1e-3;
    let fd = -(-e(2.0 * h) + 8.0 * e(h) - 8.0 * e(-h) + e(-2.0 * h)) / (12.0 * h);
    assert!(
        (r.results.forces[0][0] - fd).abs() < 1e-8,
        "{} vs {}",
        r.results.forces[0][0],
        fd
    );
}

/// Magnetic iron (bopjax IRON_FIT, d basis): bcc ferromagnet from a Stoner
/// split start, and a supercell with one reversed moment.
fn iron() -> Model {
    let ch = |p: f64| json!([[[p, 4.38348, 100.0, 1.0, 2.48]]]);
    serde_json::from_value(json!({
        "nmom": 9, "bandwidth": "averaged", "rcut": 3.4, "dcut": 0.4, "rcut2": 3.4, "dcut2": 0.4,
        "basis": ["d"], "spin": "collinear", "radial": "v0",
        "onsite": [[0.0]], "n0": [[6.88178]], "hubbard": [8.0], "stoner": [[1.16487]],
        "bond": {"dds": ch(-1.14744), "ddp": ch(0.51229), "ddd": ch(-0.03851)},
        "repulsion": {"form": "v0", "pair": [[[0.64199, 9.98837, 2.48]]],
                      "density": [[[1.0, 2.79649, 2.48]]], "embed": [[-1.51835, 0.5]]}
    }))
    .unwrap()
}

fn bcc(a: f64, n: usize, noise: f64) -> (Vec<[f64; 3]>, Cell) {
    let mut pos = Vec::new();
    let mut seed = 3u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64 / u64::MAX as f64 - 0.5) * 2.0 * noise
    };
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for b in [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]] {
                    pos.push([
                        a * (i as f64 + b[0]) + rnd(),
                        a * (j as f64 + b[1]) + rnd(),
                        a * (k as f64 + b[2]) + rnd(),
                    ]);
                }
            }
        }
    }
    let l = a * n as f64;
    (pos, Some([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]))
}

#[test]
fn magnetic_iron_converges() {
    let m = iron();
    let (pos, cell) = bcc(2.83, 2, 0.05);
    let n = pos.len();
    let topo = Topology::build(&pos, &vec![0; n], cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    let bop = Bop::new(&m, &topo);
    let split = |rev: usize| -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                let s = if i == rev { -1.3 } else { 1.3 };
                [-s, s]
            })
            .collect()
    };
    let r = bop.scf(
        &pos,
        &cell,
        Some(&split(usize::MAX)),
        &ScfOptions::default(),
    );
    println!("FM history {:?}", r.history);
    assert!(r.converged);
    let mean: f64 = r.results.moment.iter().sum::<f64>() / n as f64;
    assert!(mean > 2.0 && mean < 3.0, "moment {mean}");
    // stationarity: the energy is a saddle point but its gradient vanishes
    assert!(r
        .results
        .shift_gradient
        .iter()
        .flatten()
        .flatten()
        .all(|g| g.abs() < 1e-10));
}

/// The self-consistency's cheap gradient mode is bit-for-bit the gradient of
/// the full evaluation.
#[test]
fn gradient_modes_agree_exactly() {
    let m = iron();
    let (pos, cell) = bcc(2.83, 2, 0.05);
    let n = pos.len();
    let topo = Topology::build(&pos, &vec![0; n], cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    let bop = Bop::new(&m, &topo);
    let x: Vec<f64> = (0..2 * n)
        .map(|q| if q % 2 == 0 { -1.2 } else { 1.2 } + 0.01 * (q % 5) as f64)
        .collect();
    let a = bop.compute_mode(&pos, &cell, &x, bopfox_rs::Mode::Full);
    let b = bop.compute_mode(&pos, &cell, &x, bopfox_rs::Mode::ScfStep);
    assert_eq!(a.energy, b.energy);
    assert_eq!(a.shift_gradient, b.shift_gradient);
}

/// `free` solves only for some atoms; the others keep their starting shifts.
#[test]
fn frozen_atoms_keep_their_shifts() {
    let m = iron();
    let (pos, cell) = bcc(2.83, 2, 0.05);
    let n = pos.len();
    let topo = Topology::build(&pos, &vec![0; n], cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    let bop = Bop::new(&m, &topo);
    let x0: Vec<f64> = (0..2 * n)
        .map(|q| if q % 2 == 0 { -1.25 } else { 1.25 })
        .collect();
    let free: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
    let opt = ScfOptions {
        free: Some(free.clone()),
        ..Default::default()
    };
    let r = bop.scf(&pos, &cell, Some(&x0), &opt);
    assert!(r.converged);
    for i in 0..n {
        for s in 0..2 {
            if free[i] {
                assert!(r.results.shift_gradient[i][s][0].abs() < 1e-10);
            } else {
                assert_eq!(r.shifts[2 * i + s], x0[2 * i + s]);
            }
        }
    }
}
