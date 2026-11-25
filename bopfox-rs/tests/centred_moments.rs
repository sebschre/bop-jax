//! Centred walks (`H - c`) give the moments of `H` by the binomial expansion
//! when the Hamiltonian is symmetric.
use bopfox_rs::moments::{moments, HamiltonianData};
use bopfox_rs::{Bop, Model, Topology};
#[test]
fn centred_moments_binomial_symmetric_h() {
    let names = [
        "sss", "sps", "sds", "pps", "ppp", "pds", "pdp", "dds", "ddp", "ddd",
    ];
    let pre = [-0.5, 1.0, -0.3, 1.5, -0.3, -0.4, 0.2, -1.1, 0.5, -0.05];
    let mut bond = serde_json::Map::new();
    for (n, c) in names.iter().enumerate() {
        let row = |a: f64| serde_json::json!([pre[n] * a, 3.0 + 0.1 * n as f64, 3.0, 2.0, 2.5]);
        bond.insert(
            c.to_string(),
            serde_json::json!([[row(1.0), row(0.9)], [row(0.9), row(0.8)]]),
        );
    }
    let m: Model = serde_json::from_value(serde_json::json!({
        "nmom": 4, "rcut": 3.4, "dcut": 0.4, "basis": ["s", "p", "d"], "radial": "v0",
        "bandwidth": "averaged", "onsite": [[0.5, 6.0, 0.0], [1.0, 7.0, 0.3]],
        "n0": [[1.0, 0.0, 6.5], [1.0, 0.2, 7.0]], "hubbard": [8.0, 6.0], "bond": bond}))
    .unwrap();
    let a = 2.85 * 0.8;
    let pos: Vec<[f64; 3]> = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
    .iter()
    .enumerate()
    .map(|(i, p)| {
        [
            p[0] * a + 0.01 * i as f64,
            p[1] * a,
            p[2] * a - 0.02 * i as f64,
        ]
    })
    .collect();
    let cell = Some([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]);
    let sp = [0, 1, 1, 0];
    let topo = Topology::build(&pos, &sp, cell.as_ref(), m.rcut, m.nmom, None);
    let bop = Bop::new(&m, &topo);
    let h = bop.hopping_blocks(&pos, &cell);
    let lev: Vec<f64> = (0..4)
        .flat_map(|i| {
            let e = &m.onsite[sp[i]];
            vec![e[0], e[1], e[1], e[1], e[2], e[2], e[2], e[2], e[2]]
        })
        .collect();
    let c = 0.6;
    let raw = moments(
        &topo.clusters[0],
        &HamiltonianData {
            norb: 9,
            hblk: &h,
            onsite: &lev,
            starts: &[],
            centre: &[],
        },
        4,
    )
    .1;
    let cen = vec![c; 9];
    let mc = moments(
        &topo.clusters[0],
        &HamiltonianData {
            norb: 9,
            hblk: &h,
            onsite: &lev,
            starts: &[],
            centre: &cen,
        },
        4,
    )
    .1;
    for d in 0..9 {
        let mp: Vec<f64> = (0..4).map(|k| mc[d * 5 + k]).collect();
        let back = [
            mp[0],
            mp[1] + c,
            mp[2] + 2.0 * c * mp[1] + c * c,
            mp[3] + 3.0 * c * mp[2] + 3.0 * c * c * mp[1] + c * c * c,
        ];
        for k in 0..4 {
            assert!(
                (back[k] - raw[d * 5 + k]).abs() < 1e-9 * (1.0 + raw[d * 5 + k].abs()),
                "orb {d} k {k}: {} vs {}",
                back[k],
                raw[d * 5 + k]
            );
        }
    }
}
