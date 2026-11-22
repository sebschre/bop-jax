#![allow(clippy::needless_range_loop)]

//! Timing against system size, as bop-jax/benchmarks/scaling.py: distorted fcc
//! supercells, canonical d band, 9 moments, energy + forces at fixed shifts.
//!     cargo run --release --example scaling [n_max]

use bopfox_rs::{Bop, Model, Topology};
use std::time::Instant;

const A: f64 = 3.5;

fn main() {
    let n_max: usize = std::env::args().nth(1).map_or(6, |s| s.parse().unwrap());
    let ch = |p: f64| serde_json::json!([[[p, 2.081, 5.0]]]);
    let model: Model = serde_json::from_value(serde_json::json!({
        "nmom": 9, "rcut": 3.3, "dcut": 0.5, "bandwidth": "averaged", "basis": ["d"],
        "onsite": [[0.0]], "n0": [[4.8]], "hubbard": [8.0],
        "bond": {"dds": ch(-1.458), "ddp": ch(0.972), "ddd": ch(-0.243)}
    }))
    .unwrap();
    let base = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    println!("threads {}", rayon::current_num_threads());
    println!("atoms  cluster  topology(s)  energy(s)  energy+force(s)  E/atom");
    let mut seed = 1u64;
    let mut noise = || {
        // xorshift, uniform in [-0.08, 0.08]
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64 / u64::MAX as f64 - 0.5) * 0.16
    };
    for n in 1..=n_max {
        let mut pos = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for b in &base {
                        pos.push([
                            A * (i as f64 + b[0]) + noise(),
                            A * (j as f64 + b[1]) + noise(),
                            A * (k as f64 + b[2]) + noise(),
                        ]);
                    }
                }
            }
        }
        let l = A * n as f64;
        let cell = Some([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]);
        let species = vec![0; pos.len()];
        let t0 = Instant::now();
        let topo = Topology::build(&pos, &species, cell.as_ref(), model.rcut, model.nmom, None);
        let t_topo = t0.elapsed().as_secs_f64();
        let bop = Bop::new(&model, &topo);
        let shifts = vec![0.0; pos.len()];
        let best = |grad: bool| {
            (0..3)
                .map(|_| {
                    let t = Instant::now();
                    let r = bop.compute(&pos, &cell, &shifts, grad);
                    (t.elapsed().as_secs_f64(), r.energy)
                })
                .fold((f64::INFINITY, 0.0), |a, b| if b.0 < a.0 { b } else { a })
        };
        let (t_e, _) = best(false);
        let (t_f, e) = best(true);
        println!(
            "{:5}  {:7}  {:11.4}  {:9.4}  {:15.4}  {:.10}",
            pos.len(),
            topo.clusters[0].origin.len(),
            t_topo,
            t_e,
            t_f,
            e / pos.len() as f64
        );
    }
}
