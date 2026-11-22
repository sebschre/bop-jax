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

/// All pairs closer than `rcut` (brute force over atoms and lattice images).
pub fn neighbours(pos: &[[f64; 3]], cell: Option<&[[f64; 3]; 3]>, rcut: f64) -> NeighbourList {
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
    let mut nl = NeighbourList {
        start: vec![0],
        ..Default::default()
    };
    for row in rows {
        for (j, t) in row {
            nl.j.push(j);
            nl.t.push(t);
        }
        nl.start.push(nl.j.len());
    }
    nl
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
