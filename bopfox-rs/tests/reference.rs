#![allow(clippy::needless_range_loop)]

//! Comparison with BOPfox (numbers of tests/test_bopfox_reference.py) and
//! finite-difference checks of every analytic derivative.

use bopfox_rs::{Bop, Cell, Model, Topology};
use serde_json::json;

const A: f64 = 3.5;

fn fcc() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.5 * A, 0.5 * A],
        [0.5 * A, 0.0, 0.5 * A],
        [0.5 * A, 0.5 * A, 0.0],
    ]
}

fn cubic(a: f64) -> Cell {
    Some([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
}

/// Canonical d band (bopjax.models.canonical_d).
fn canonical_d(nmom: usize, rcut: f64, dcut: f64, hubbard: f64) -> Model {
    let ch = |p: f64| json!([[[p, 2.081, 5.0]]]);
    serde_json::from_value(json!({
        "nmom": nmom, "rcut": rcut, "dcut": dcut, "basis": ["d"],
        "onsite": [[0.0]], "n0": [[4.8]], "hubbard": [hubbard],
        "bond": {"dds": ch(-1.458), "ddp": ch(0.972), "ddd": ch(-0.243)}
    }))
    .unwrap()
}

fn run(
    m: &Model,
    pos: &[[f64; 3]],
    species: &[usize],
    cell: &Cell,
    shifts: &[f64],
    grad: bool,
) -> bopfox_rs::Results {
    let topo = Topology::build(pos, species, cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    Bop::new(m, &topo).compute(pos, cell, shifts, grad)
}

#[test]
fn perfect_fcc_matches_bopfox_log() {
    let m = canonical_d(6, 3.01, 0.0, 0.0);
    let r = run(&m, &fcc(), &[0; 4], &cubic(A), &[0.0; 4], false);
    assert!(
        (r.energy / 4.0 + 5.82994578275954).abs() < 1e-11,
        "{}",
        r.energy / 4.0
    );
    assert!((r.fermi - 0.02525345).abs() < 1e-8);
    assert!((r.a_inf[0][0][0] + 0.31753865).abs() < 1e-8);
    assert!((r.b_inf[0][0][0] - 1.32390275).abs() < 1e-8);
    assert!((r.moments[0][0][0][2] - 1.75271848053838).abs() < 1e-12);
    for n in &r.n_shell {
        assert!((n[0] - 4.8).abs() < 1e-10);
    }
}

fn distorted() -> Vec<[f64; 3]> {
    let mut p = fcc();
    for x in 0..3 {
        p[0][x] += [0.15, -0.08, 0.05][x];
        p[2][x] += [-0.05, 0.10, 0.02][x];
    }
    p
}

#[test]
fn forces_at_bopfox_scf_shifts_match_bopfox() {
    let m = canonical_d(6, 3.3, 0.5, 8.0);
    let shifts = [
        -3.2427167768e-3,
        4.1529574320e-3,
        -1.3968614396e-3,
        4.8662078441e-4,
    ];
    let r = run(&m, &distorted(), &[0; 4], &cubic(A), &shifts, true);
    assert!(
        (r.energy / 4.0 + 6.011_248_544_843_719).abs() < 1e-10,
        "{}",
        r.energy / 4.0
    );
    let charge = [
        2.001241198650e-4,
        -5.18733079637e-5,
        2.22429422658e-5,
        -1.704937541867e-4,
    ];
    for i in 0..4 {
        assert!((r.charge[i] - charge[i]).abs() < 1e-8);
        assert!(
            r.shift_gradient[i][0][0].abs() < 1e-7,
            "{:?}",
            r.shift_gradient
        );
    }
    let f_bopfox = [
        [4.9859884934, -2.4231962888, 1.2439578230],
        [0.8291859687, -0.3280010053, -0.7088860647],
        [-3.5410851762, 2.7658615584, -0.2853256569],
        [-2.2740892253, -0.0146642081, -0.2497461152],
    ];
    for i in 0..4 {
        for x in 0..3 {
            assert!(
                (r.forces[i][x] - f_bopfox[i][x]).abs() < 1e-6,
                "{:?}",
                r.forces
            );
        }
    }
    let total: f64 = (0..3)
        .map(|x| r.forces.iter().map(|f| f[x]).sum::<f64>().abs())
        .sum();
    assert!(total < 1e-12);
}

/// Five-point central difference of f at 0.
fn d5(f: impl Fn(f64) -> f64) -> f64 {
    let h = 1e-4;
    (-f(2.0 * h) + 8.0 * f(h) - 8.0 * f(-h) + f(-2.0 * h)) / (12.0 * h)
}

/// Checks forces, shift gradient and virial against central differences.
fn check_fd(m: &Model, pos: &[[f64; 3]], species: &[usize], cell: &Cell, shifts: &[f64], tol: f64) {
    let topo = Topology::build(pos, species, cell.as_ref(), m.rcut, m.nmom, m.rcut2);
    let bop = Bop::new(m, &topo);
    let r = bop.compute(pos, cell, shifts, true);
    let e = |p: &[[f64; 3]], c: &Cell, s: &[f64]| bop.compute(p, c, s, false).energy;
    for i in 0..pos.len() {
        for x in 0..3 {
            let fd = -d5(|t| {
                let mut p = pos.to_vec();
                p[i][x] += t;
                e(&p, cell, shifts)
            });
            assert!(
                (r.forces[i][x] - fd).abs() < tol * (1.0 + fd.abs()),
                "force {i} {x}: {} vs {}",
                r.forces[i][x],
                fd
            );
        }
    }
    let ns = r.shift_gradient[0].len();
    let nsh = r.shift_gradient[0][0].len();
    for i in 0..pos.len() {
        for s in 0..ns {
            for k in 0..nsh {
                let q = (i * ns + s) * nsh + k;
                let fd = d5(|t| {
                    let mut sp = shifts.to_vec();
                    sp[q] += t;
                    e(pos, cell, &sp)
                });
                let g = r.shift_gradient[i][s][k];
                assert!(
                    (g - fd).abs() < tol * (1.0 + fd.abs()),
                    "shift {i} {s} {k}: {g} vs {fd}"
                );
            }
        }
    }
    if let Some(c) = cell {
        for x in 0..3 {
            for y in 0..3 {
                let strained = |eps: f64| {
                    let mut d = [[0.0; 3]; 3];
                    for a in 0..3 {
                        d[a][a] = 1.0;
                    }
                    d[x][y] += 0.5 * eps;
                    d[y][x] += 0.5 * eps;
                    let mul = |v: [f64; 3]| {
                        let mut o = [0.0; 3];
                        for b in 0..3 {
                            for a in 0..3 {
                                o[b] += v[a] * d[a][b];
                            }
                        }
                        o
                    };
                    let p: Vec<[f64; 3]> = pos.iter().map(|&v| mul(v)).collect();
                    let cc = [mul(c[0]), mul(c[1]), mul(c[2])];
                    e(&p, &Some(cc), shifts)
                };
                let fd = d5(strained);
                assert!(
                    (r.virial[x][y] - fd).abs() < tol * (1.0 + fd.abs()),
                    "virial {x} {y}: {} vs {}",
                    r.virial[x][y],
                    fd
                );
            }
        }
    }
}

#[test]
fn d_band_derivatives_match_finite_differences() {
    let mut m = canonical_d(9, 3.3, 0.5, 8.0);
    m.bandwidth = bopfox_rs::model::Bandwidth::Averaged;
    let shifts = [0.01, -0.02, 0.005, 0.003];
    check_fd(&m, &distorted(), &[0; 4], &cubic(A), &shifts, 1e-7);
}

#[test]
fn terminators_and_kernel_derivatives() {
    use bopfox_rs::model::{Bandwidth, Kernel, Terminator};
    for (t, k, b, nmom) in [
        (
            Terminator::Weighted,
            Kernel::Jackson,
            Bandwidth::Findeminemax,
            8,
        ),
        (
            Terminator::Oscillation,
            Kernel::None,
            Bandwidth::Gershgorin,
            9,
        ),
        (
            Terminator::Constantabn,
            Kernel::Jackson,
            Bandwidth::Highest,
            7,
        ),
    ] {
        let mut m = canonical_d(nmom, 3.3, 0.5, 4.0);
        m.terminator = t;
        m.kernel = k;
        m.bandwidth = b;
        m.nexp = Some(nmom + 4);
        check_fd(
            &m,
            &distorted(),
            &[0; 4],
            &cubic(A),
            &[0.0, 0.01, -0.01, 0.0],
            1e-7,
        );
    }
}

/// Two species, s, p and d shells, v0 bond integrals, repulsion, collinear
/// magnetism, finite cluster.
fn spd_model() -> Model {
    let mut bond = serde_json::Map::new();
    let names = [
        "sss", "sps", "sds", "pps", "ppp", "pds", "pdp", "dds", "ddp", "ddd",
    ];
    for (n, c) in names.iter().enumerate() {
        let pre = [-0.5, 1.0, -0.3, 1.5, -0.3, -0.4, 0.2, -1.1, 0.5, -0.05][n];
        let row = |a: f64| json!([pre * a, 3.0 + 0.1 * n as f64, 3.0, 2.0, 2.5]);
        bond.insert(
            c.to_string(),
            json!([[row(1.0), row(0.9)], [row(1.1), row(0.8)]]),
        );
    }
    serde_json::from_value(json!({
        "nmom": 7, "rcut": 3.4, "dcut": 0.4, "rcut2": 3.6, "dcut2": 0.5,
        "basis": ["s", "p", "d"], "radial": "v0", "spin": "collinear",
        "bandwidth": "averaged",
        "onsite": [[0.5, 6.0, 0.0], [1.0, 7.0, 0.3]],
        "n0": [[1.0, 0.0, 6.5], [1.0, 0.2, 7.0]],
        "hubbard": [8.0, 6.0],
        "stoner": [[0.0, 0.0, 1.1], [0.1, 0.0, 0.9]],
        "bond": bond,
        "repulsion": {"form": "v0",
            "pair": [[[0.6, 9.0, 2.5], [0.5, 8.0, 2.5]], [[0.5, 8.0, 2.5], [0.7, 9.5, 2.5]]],
            "density": [[[1.0, 2.8, 2.5], [0.9, 2.5, 2.5]], [[0.9, 2.5, 2.5], [1.1, 3.0, 2.5]]],
            "embed": [[-1.5, 0.5], [-1.3, 0.5]]}
    }))
    .unwrap()
}

#[test]
fn spd_magnetic_two_species_cluster() {
    let m = spd_model();
    m.validate().unwrap();
    let pos = vec![
        [0.0, 0.0, 0.0],
        [1.45, 1.4, 0.05],
        [1.4, -0.1, 1.5],
        [-0.05, 1.42, 1.38],
        [2.8, 1.5, 1.4],
    ];
    let species = [0, 1, 0, 1, 0];
    let n = pos.len();
    let mut shifts = vec![0.0; n * 2 * 3];
    for (q, s) in shifts.iter_mut().enumerate() {
        *s = 0.02 * ((q * 7 % 5) as f64 - 2.0) + if (q / 3) % 2 == 0 { -0.3 } else { 0.3 };
    }
    check_fd(&m, &pos, &species, &None, &shifts, 2e-7);
}

#[test]
fn spd_magnetic_periodic() {
    let m = spd_model();
    let pos = distorted()
        .iter()
        .map(|p| [p[0] * 0.8, p[1] * 0.8, p[2] * 0.8])
        .collect::<Vec<_>>();
    let mut shifts = vec![0.0; 4 * 2 * 3];
    for (q, s) in shifts.iter_mut().enumerate() {
        *s = if (q / 3) % 2 == 0 { -0.4 } else { 0.4 };
    }
    check_fd(&m, &pos, &[0, 1, 1, 0], &cubic(0.8 * A), &shifts, 2e-7);
}
