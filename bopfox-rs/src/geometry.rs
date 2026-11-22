//! Neighbour lists and the local cluster of every atom (`bopjax.geometry`).
//!
//! Moments are sums over closed hopping paths that start and end on one atom,
//! so each atom only needs the atoms it can reach in half the path length.
//! Cluster members are lattice translations of unit-cell atoms; the hopping
//! between two members is the block of the corresponding unit-cell bond.

use rayon::prelude::*;
use std::collections::HashMap;

/// Neighbour list in compressed rows: the bonds of atom `i` are
/// `start[i]..start[i + 1]`, bond `b` points to unit-cell atom `j[b]` shifted
/// by the lattice translation `t[b]`.
#[derive(Clone, Debug, Default)]
pub struct NeighbourList {
    pub start: Vec<usize>,
    pub j: Vec<usize>,
    pub t: Vec<[i32; 3]>,
}

impl NeighbourList {
    pub fn n_bonds(&self) -> usize {
        self.j.len()
    }
    pub fn bonds(&self, i: usize) -> std::ops::Range<usize> {
        self.start[i]..self.start[i + 1]
    }
    /// Owner atom of every bond.
    pub fn owners(&self) -> Vec<usize> {
        let mut o = vec![0; self.n_bonds()];
        for i in 0..self.start.len() - 1 {
            for b in self.bonds(i) {
                o[b] = i;
            }
        }
        o
    }
    /// Vector of bond `b` of atom `i`.
    pub fn vector(
        &self,
        b: usize,
        i: usize,
        pos: &[[f64; 3]],
        cell: Option<&[[f64; 3]; 3]>,
    ) -> [f64; 3] {
        let j = self.j[b];
        let mut r = [
            pos[j][0] - pos[i][0],
            pos[j][1] - pos[i][1],
            pos[j][2] - pos[i][2],
        ];
        if let Some(c) = cell {
            let t = self.t[b];
            for x in 0..3 {
                r[x] += t[0] as f64 * c[0][x] + t[1] as f64 * c[1][x] + t[2] as f64 * c[2][x];
            }
        }
        r
    }
}

/// Local cluster of one atom.  Slots are ordered by hop distance:
/// `size_le[h]` slots lie within `h` hops.  `links[link_start[c]..link_start[c+1]]`
/// are the bonds `(bond index of origin[c], slot of the neighbour)` that stay
/// inside the cluster.
#[derive(Clone, Debug, Default)]
pub struct Cluster {
    pub origin: Vec<usize>,
    pub size_le: Vec<usize>,
    pub link_start: Vec<usize>,
    pub links: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct Topology {
    pub n_atoms: usize,
    pub species: Vec<usize>,
    pub nbr: NeighbourList,
    pub clusters: Vec<Cluster>,
    pub nbr2: Option<NeighbourList>,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Distance between opposite faces of the cell along each lattice vector.
fn heights(cell: &[[f64; 3]; 3]) -> [f64; 3] {
    let vol = dot(cell[0], cross(cell[1], cell[2])).abs();
    let c = [
        cross(cell[1], cell[2]),
        cross(cell[2], cell[0]),
        cross(cell[0], cell[1]),
    ];
    [
        vol / dot(c[0], c[0]).sqrt(),
        vol / dot(c[1], c[1]).sqrt(),
        vol / dot(c[2], c[2]).sqrt(),
    ]
}

/// All pairs closer than `rcut`: binned (linear in the number of atoms)
/// when the cell holds at least three bins of width `rcut` along every
/// lattice direction (or always, for a finite cluster), otherwise by brute
/// force over atoms and lattice images.
pub fn neighbours(pos: &[[f64; 3]], cell: Option<&[[f64; 3]; 3]>, rcut: f64) -> NeighbourList {
    match cell {
        None => binned_open(pos, rcut),
        Some(c) => {
            let h = heights(c);
            let nb: Vec<usize> = h.iter().map(|hd| (hd / rcut).floor() as usize).collect();
            if nb.iter().all(|&b| b >= 3) {
                binned_periodic(pos, c, rcut, [nb[0], nb[1], nb[2]])
            } else {
                brute_force(pos, cell, rcut)
            }
        }
    }
}

fn from_rows(rows: Vec<Vec<(usize, [i32; 3])>>) -> NeighbourList {
    let mut nl = NeighbourList {
        start: vec![0],
        ..Default::default()
    };
    for mut row in rows {
        row.sort_unstable(); // a deterministic order, whatever the search
        for (j, t) in row {
            nl.j.push(j);
            nl.t.push(t);
        }
        nl.start.push(nl.j.len());
    }
    nl
}

/// Fractional coordinates are `s_d = sum_x pos_x inv[x][d]`.
fn inverse(c: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = dot(c[0], cross(c[1], c[2]));
    let r = [cross(c[1], c[2]), cross(c[2], c[0]), cross(c[0], c[1])];
    let mut inv = [[0.0; 3]; 3];
    for x in 0..3 {
        for d in 0..3 {
            inv[x][d] = r[d][x] / det;
        }
    }
    inv
}

fn binned_periodic(
    pos: &[[f64; 3]],
    c: &[[f64; 3]; 3],
    rcut: f64,
    nb: [usize; 3],
) -> NeighbourList {
    let inv = inverse(c);
    let n = pos.len();
    // wrapped fractional coordinates and the lattice shift that wraps them
    let mut frac = vec![[0.0; 3]; n];
    let mut wrap = vec![[0i32; 3]; n];
    let mut bin_of = vec![[0usize; 3]; n];
    for i in 0..n {
        for d in 0..3 {
            let s = pos[i][0] * inv[0][d] + pos[i][1] * inv[1][d] + pos[i][2] * inv[2][d];
            let f = s.floor();
            frac[i][d] = s - f;
            wrap[i][d] = f as i32;
            bin_of[i][d] = ((frac[i][d] * nb[d] as f64) as usize).min(nb[d] - 1);
        }
    }
    let idx = |b: [usize; 3]| (b[0] * nb[1] + b[1]) * nb[2] + b[2];
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); nb[0] * nb[1] * nb[2]];
    for i in 0..n {
        bins[idx(bin_of[i])].push(i);
    }
    let r2 = rcut * rcut;
    let rows = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = Vec::new();
            for dx in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dz in -1i32..=1 {
                        let mut b = [0usize; 3];
                        let mut w = [0i32; 3];
                        for (d, dd) in [dx, dy, dz].into_iter().enumerate() {
                            let v = bin_of[i][d] as i32 + dd;
                            w[d] = v.div_euclid(nb[d] as i32);
                            b[d] = v.rem_euclid(nb[d] as i32) as usize;
                        }
                        for &j in &bins[idx(b)] {
                            let ds = [
                                frac[j][0] + w[0] as f64 - frac[i][0],
                                frac[j][1] + w[1] as f64 - frac[i][1],
                                frac[j][2] + w[2] as f64 - frac[i][2],
                            ];
                            let mut dv = [0.0; 3];
                            for x in 0..3 {
                                dv[x] = ds[0] * c[0][x] + ds[1] * c[1][x] + ds[2] * c[2][x];
                            }
                            let dd = dot(dv, dv);
                            if dd < r2 && dd > 1e-16 {
                                // translation of j in the original (unwrapped) frame
                                let t = [
                                    w[0] + wrap[i][0] - wrap[j][0],
                                    w[1] + wrap[i][1] - wrap[j][1],
                                    w[2] + wrap[i][2] - wrap[j][2],
                                ];
                                row.push((j, t));
                            }
                        }
                    }
                }
            }
            row
        })
        .collect();
    from_rows(rows)
}

fn binned_open(pos: &[[f64; 3]], rcut: f64) -> NeighbourList {
    let n = pos.len();
    let mut lo = [f64::INFINITY; 3];
    for p in pos {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
        }
    }
    let key =
        |p: &[f64; 3]| -> [i64; 3] { [0, 1, 2].map(|d| ((p[d] - lo[d]) / rcut).floor() as i64) };
    let mut bins: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, p) in pos.iter().enumerate() {
        bins.entry(key(p)).or_default().push(i);
    }
    let r2 = rcut * rcut;
    let rows = (0..n)
        .into_par_iter()
        .map(|i| {
            let k = key(&pos[i]);
            let mut row = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(list) = bins.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                            for &j in list {
                                let d = [
                                    pos[j][0] - pos[i][0],
                                    pos[j][1] - pos[i][1],
                                    pos[j][2] - pos[i][2],
                                ];
                                let dd = dot(d, d);
                                if dd < r2 && dd > 1e-16 {
                                    row.push((j, [0, 0, 0]));
                                }
                            }
                        }
                    }
                }
            }
            row
        })
        .collect();
    from_rows(rows)
}

/// All pairs closer than `rcut`, brute force over atoms and lattice images.
pub fn brute_force(pos: &[[f64; 3]], cell: Option<&[[f64; 3]; 3]>, rcut: f64) -> NeighbourList {
    let shifts: Vec<([i32; 3], [f64; 3])> = match cell {
        None => vec![([0, 0, 0], [0.0; 3])],
        Some(c) => {
            let h = heights(c);
            let nc: Vec<i32> = h.iter().map(|hd| (rcut / hd).ceil() as i32).collect();
            let mut out = Vec::new();
            for a in -nc[0]..=nc[0] {
                for b in -nc[1]..=nc[1] {
                    for d in -nc[2]..=nc[2] {
                        let mut v = [0.0; 3];
                        for x in 0..3 {
                            v[x] = a as f64 * c[0][x] + b as f64 * c[1][x] + d as f64 * c[2][x];
                        }
                        out.push(([a, b, d], v));
                    }
                }
            }
            out
        }
    };
    let r2 = rcut * rcut;
    let rows: Vec<Vec<(usize, [i32; 3])>> = (0..pos.len())
        .into_par_iter()
        .map(|i| {
            let mut row = Vec::new();
            for (j, pj) in pos.iter().enumerate() {
                for (t, v) in &shifts {
                    let d = [
                        pj[0] + v[0] - pos[i][0],
                        pj[1] + v[1] - pos[i][1],
                        pj[2] + v[2] - pos[i][2],
                    ];
                    let dd = dot(d, d);
                    if dd < r2 && dd > 1e-16 {
                        row.push((j, *t));
                    }
                }
            }
            row
        })
        .collect();
    from_rows(rows)
}

type Key = (usize, i32, i32, i32);

/// Breadth-first cluster of atom `i` out to `n_hops` hops.
fn cluster(nbr: &NeighbourList, i: usize, n_hops: usize) -> Cluster {
    let mut slot: HashMap<Key, usize> = HashMap::new();
    let mut members: Vec<Key> = vec![(i, 0, 0, 0)];
    slot.insert((i, 0, 0, 0), 0);
    let mut size_le = vec![1];
    let mut frontier = 0..1;
    for _ in 0..n_hops {
        let end = members.len();
        for m in frontier.clone() {
            let (j, tx, ty, tz) = members[m];
            for b in nbr.bonds(j) {
                let t = nbr.t[b];
                let key = (nbr.j[b], tx + t[0], ty + t[1], tz + t[2]);
                if let std::collections::hash_map::Entry::Vacant(e) = slot.entry(key) {
                    e.insert(members.len());
                    members.push(key);
                }
            }
        }
        frontier = end..members.len();
        size_le.push(members.len());
    }
    let mut cl = Cluster {
        origin: members.iter().map(|k| k.0).collect(),
        size_le,
        link_start: vec![0],
        links: Vec::new(),
    };
    for &(j, tx, ty, tz) in &members {
        for b in nbr.bonds(j) {
            let t = nbr.t[b];
            if let Some(&c) = slot.get(&(nbr.j[b], tx + t[0], ty + t[1], tz + t[2])) {
                cl.links.push((b, c));
            }
        }
        cl.link_start.push(cl.links.len());
    }
    cl
}

impl Topology {
    /// Neighbour lists and clusters for the moments up to `nmom`.
    pub fn build(
        pos: &[[f64; 3]],
        species: &[usize],
        cell: Option<&[[f64; 3]; 3]>,
        rcut: f64,
        nmom: usize,
        rcut2: Option<f64>,
    ) -> Self {
        let nbr = neighbours(pos, cell, rcut);
        let n_hops = crate::moments::cluster_hops(nmom);
        let clusters = (0..pos.len())
            .into_par_iter()
            .map(|i| cluster(&nbr, i, n_hops))
            .collect();
        let nbr2 = rcut2.map(|r| neighbours(pos, cell, r));
        Topology {
            n_atoms: pos.len(),
            species: species.to_vec(),
            nbr,
            clusters,
            nbr2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lists_equal(a: &NeighbourList, b: &NeighbourList) {
        assert_eq!(a.start, b.start);
        assert_eq!(a.j, b.j);
        assert_eq!(a.t, b.t);
    }

    #[test]
    fn binned_search_finds_the_same_pairs() {
        // a skewed periodic cell with atoms outside it, and an open cluster
        let mut seed = 7u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as f64 / u64::MAX as f64
        };
        let cell = [[12.0, 0.0, 0.0], [2.5, 11.0, 0.0], [-1.5, 1.0, 13.0]];
        let pos: Vec<[f64; 3]> = (0..300)
            .map(|_| {
                let s = [rnd() * 1.4 - 0.2, rnd() * 1.4 - 0.2, rnd() * 1.4 - 0.2];
                [0, 1, 2].map(|x| s[0] * cell[0][x] + s[1] * cell[1][x] + s[2] * cell[2][x])
            })
            .collect();
        for rcut in [2.5, 3.9] {
            lists_equal(
                &neighbours(&pos, Some(&cell), rcut),
                &brute_force(&pos, Some(&cell), rcut),
            );
            lists_equal(
                &neighbours(&pos, None, rcut),
                &brute_force(&pos, None, rcut),
            );
        }
    }
}
