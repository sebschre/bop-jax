//! Moments of the local density of states and their exact derivatives.
//!
//! For atom `i` and every start orbital `d`, the walk builds the vectors
//! `w_a = H^a |i d>` for `a = 0..ceil(nmom / 2)` on the cluster of `i`
//! (`w_a` vanishes beyond `a` hops, so step `a` only touches slots within
//! `a + 1` hops), and the moments are `mu_k = <w_a | w_b>` with
//! `a = k / 2`, `b = k - a`.
//!
//! The backward pass is the analytic derivative of the moments with respect
//! to every bond block and onsite level: the adjoint vectors `wbar_a` are
//! carried back through the walk, and every application of a bond block
//! `H_b` contributes `wbar_{a+1} w_a^T` to `dE/dH_b`.  This is the sum over
//! the interference paths through bond `b` of BOPfox, organised so that no
//! path is enumerated and the cost is that of the forward walk.

use crate::geometry::Cluster;

/// Bond blocks and onsite levels of one spin channel.
pub struct HamiltonianData<'a> {
    pub norb: usize,
    /// `n_bonds * norb * norb`, block of bond b row-major
    pub hblk: &'a [f64],
    /// `n_atoms * norb` onsite levels of this spin channel
    pub onsite: &'a [f64],
    /// the start orbitals of the walk (an atom's own orbitals); empty: all
    pub starts: &'a [usize],
    /// per start column, a constant subtracted from every onsite level: the
    /// column walks `H - c`, so its moments are centred on c (the start
    /// orbital's own level keeps them small and the recursion well
    /// conditioned); empty: no shift
    pub centre: &'a [f64],
}

impl HamiltonianData<'_> {
    /// Number of start columns of the walk.
    pub fn n_starts(&self) -> usize {
        if self.starts.is_empty() {
            self.norb
        } else {
            self.starts.len()
        }
    }
    /// The shift of column c.
    #[inline(always)]
    fn shift(&self, c: usize) -> f64 {
        if self.centre.is_empty() {
            0.0
        } else {
            self.centre[c]
        }
    }
    /// Start orbital of column c.
    #[inline(always)]
    fn start(&self, c: usize) -> usize {
        if self.starts.is_empty() {
            c
        } else {
            self.starts[c]
        }
    }
}

/// Vectors of one walk: `w[a]` holds `slots * norb * n_starts` entries, layout
/// `[slot][orbital][start column]`.
pub struct Walk {
    pub w: Vec<Vec<f64>>,
}

impl Walk {
    pub fn len(&self) -> usize {
        self.w.iter().map(|x| x.len()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.w.is_empty()
    }
}

/// Hops a cluster needs for `nmom` moments: `mu_k = <w_a|w_b>` is evaluated
/// within `a = k / 2` hops, and `w_b` there only needs `w_{b-1}` within
/// `a` hops (odd k) or `a` hops minus one (even k).
pub fn cluster_hops(nmom: usize) -> usize {
    nmom / 2
}

/// Number of orbitals, known at compile time for the common bases so that
/// the block loops are unrolled.
trait Dim: Copy {
    fn n(self) -> usize;
}
#[derive(Clone, Copy)]
struct Fixed<const N: usize>;
impl<const N: usize> Dim for Fixed<N> {
    #[inline(always)]
    fn n(self) -> usize {
        N
    }
}
#[derive(Clone, Copy)]
struct Dynamic(usize);
impl Dim for Dynamic {
    #[inline(always)]
    fn n(self) -> usize {
        self.0
    }
}

macro_rules! dispatch {
    ($norb:expr, $ns:expr, $f:ident($($arg:expr),*)) => {
        match ($norb, $ns) {
            (1, 1) => $f(Fixed::<1>, Fixed::<1>, $($arg),*),
            (4, 4) => $f(Fixed::<4>, Fixed::<4>, $($arg),*),
            (5, 5) => $f(Fixed::<5>, Fixed::<5>, $($arg),*),
            (6, 6) => $f(Fixed::<6>, Fixed::<6>, $($arg),*),
            (9, 9) => $f(Fixed::<9>, Fixed::<9>, $($arg),*),
            (9, 5) => $f(Fixed::<9>, Fixed::<5>, $($arg),*),
            (9, 4) => $f(Fixed::<9>, Fixed::<4>, $($arg),*),
            (n, m) => $f(Dynamic(n), Dynamic(m), $($arg),*),
        }
    };
}

fn walk_impl<N: Dim, S: Dim>(
    dim: N,
    sdim: S,
    cl: &Cluster,
    h: &HamiltonianData,
    nmom: usize,
) -> Walk {
    let n = dim.n();
    let ns = sdim.n();
    let nn = n * ns;
    let n_steps = nmom.div_ceil(2);
    let top = cl.size_le.len() - 1;
    let mut w: Vec<Vec<f64>> = Vec::with_capacity(n_steps + 1);
    let mut w0 = vec![0.0; nn];
    for c in 0..ns {
        w0[h.start(c) * ns + c] = 1.0;
    }
    w.push(w0);
    for a in 0..n_steps {
        let (n_in, n_out) = (cl.size_le[a], cl.size_le[(a + 1).min(top)]);
        let prev = &w[a];
        let mut out = vec![0.0; n_out * nn];
        for c in 0..n_out {
            let oc = &mut out[c * nn..(c + 1) * nn];
            for &(b, c2) in &cl.links[cl.link_start[c]..cl.link_start[c + 1]] {
                if c2 >= n_in {
                    continue;
                }
                let blk = &h.hblk[b * n * n..(b + 1) * n * n];
                let wc = &prev[c2 * nn..(c2 + 1) * nn];
                for al in 0..n {
                    for be in 0..n {
                        let hv = blk[al * n + be];
                        if hv == 0.0 {
                            continue;
                        }
                        for d in 0..ns {
                            oc[al * ns + d] += hv * wc[be * ns + d];
                        }
                    }
                }
            }
            if c < n_in {
                let lev = &h.onsite[cl.origin[c] * n..(cl.origin[c] + 1) * n];
                let wc = &prev[c * nn..(c + 1) * nn];
                for al in 0..n {
                    for d in 0..ns {
                        oc[al * ns + d] += (lev[al] - h.shift(d)) * wc[al * ns + d];
                    }
                }
            }
        }
        w.push(out);
    }
    Walk { w }
}

/// The walk of one atom (all start orbitals at once).
pub fn walk(cl: &Cluster, h: &HamiltonianData, nmom: usize) -> Walk {
    dispatch!(h.norb, h.n_starts(), walk_impl(cl, h, nmom))
}

/// Moments `<i d|H^k|i d>` from a walk, layout `[orbital d][k]` (zero for
/// orbitals that are not start orbitals).
pub fn moments_of(cl: &Cluster, wk: &Walk, h: &HamiltonianData, nmom: usize) -> Vec<f64> {
    let (n, ns) = (h.norb, h.n_starts());
    let nn = n * ns;
    let mut mu = vec![0.0; n * (nmom + 1)];
    for k in 0..=nmom {
        let (a, b) = (k / 2, k - k / 2);
        let size = cl.size_le[a] * nn;
        let (wa, wb) = (&wk.w[a][..size], &wk.w[b][..size]);
        for (idx, (x, y)) in wa.iter().zip(wb).enumerate() {
            mu[h.start(idx % ns) * (nmom + 1) + k] += x * y;
        }
    }
    mu
}

/// Walk and moments of one atom.
pub fn moments(cl: &Cluster, h: &HamiltonianData, nmom: usize) -> (Walk, Vec<f64>) {
    let wk = walk(cl, h, nmom);
    let mu = moments_of(cl, &wk, h, nmom);
    (wk, mu)
}

#[allow(clippy::too_many_arguments)]
fn backward_impl<N: Dim, S: Dim, F: FnMut(usize, &[f64])>(
    dim: N,
    sdim: S,
    cl: &Cluster,
    h: &HamiltonianData,
    nmom: usize,
    wk: &Walk,
    gmu: &[f64],
    onsite_bar: &mut [f64],
    mut bond_grad: Option<&mut F>,
) {
    let n = dim.n();
    let ns = sdim.n();
    let nn = n * ns;
    let n_steps = nmom.div_ceil(2);
    let mut wbar: Vec<Vec<f64>> = wk.w.iter().map(|x| vec![0.0; x.len()]).collect();
    for k in 0..=nmom {
        let (a, b) = (k / 2, k - k / 2);
        let size = cl.size_le[a] * nn;
        for idx in 0..size {
            let g = gmu[h.start(idx % ns) * (nmom + 1) + k];
            let (xa, xb) = (wk.w[a][idx], wk.w[b][idx]);
            wbar[a][idx] += g * xb;
            wbar[b][idx] += g * xa;
        }
    }
    let mut hbar = vec![0.0; n * n];
    for a in (0..n_steps).rev() {
        let n_in = cl.size_le[a];
        let (lo, hi) = wbar.split_at_mut(a + 1);
        let (wbar_a, wbar_next) = (&mut lo[a], &hi[0]);
        let n_out = wbar_next.len() / nn;
        let wa = &wk.w[a];
        for c in 0..n_out {
            let gc = &wbar_next[c * nn..(c + 1) * nn];
            for &(b, c2) in &cl.links[cl.link_start[c]..cl.link_start[c + 1]] {
                if c2 >= n_in {
                    continue;
                }
                let blk = &h.hblk[b * n * n..(b + 1) * n * n];
                let src = &wa[c2 * nn..(c2 + 1) * nn];
                let dst = &mut wbar_a[c2 * nn..(c2 + 1) * nn];
                for al in 0..n {
                    for be in 0..n {
                        let hv = blk[al * n + be];
                        if hv == 0.0 {
                            continue;
                        }
                        for d in 0..ns {
                            dst[be * ns + d] += hv * gc[al * ns + d];
                        }
                    }
                }
                if let Some(f) = bond_grad.as_mut() {
                    for al in 0..n {
                        for be in 0..n {
                            let mut acc = 0.0;
                            for d in 0..ns {
                                acc += gc[al * ns + d] * src[be * ns + d];
                            }
                            hbar[al * n + be] = acc;
                        }
                    }
                    f(b, &hbar);
                }
            }
            if c < n_in {
                let o = cl.origin[c];
                let lev = &h.onsite[o * n..(o + 1) * n];
                let src = &wa[c * nn..(c + 1) * nn];
                let dst = &mut wbar_a[c * nn..(c + 1) * nn];
                for al in 0..n {
                    let mut acc = 0.0;
                    for d in 0..ns {
                        acc += gc[al * ns + d] * src[al * ns + d];
                        dst[al * ns + d] += (lev[al] - h.shift(d)) * gc[al * ns + d];
                    }
                    onsite_bar[o * n + al] += acc;
                }
            }
        }
    }
}

/// Backward pass.  `gmu[d * (nmom+1) + k]` is dE/dmu_k of start orbital d.
/// Calls `bond_grad(b, hbar)` with the contribution `dE/dH_b` (row-major,
/// `norb * norb`) of every application of bond block `b`, and adds dE/d(onsite)
/// to `onsite_bar` (`n_atoms * norb`).  Without `bond_grad` only the onsite
/// derivatives are formed, at about the cost of one forward walk.
pub fn backward<F: FnMut(usize, &[f64])>(
    cl: &Cluster,
    h: &HamiltonianData,
    nmom: usize,
    wk: &Walk,
    gmu: &[f64],
    onsite_bar: &mut [f64],
    bond_grad: Option<&mut F>,
) {
    dispatch!(
        h.norb,
        h.n_starts(),
        backward_impl(cl, h, nmom, wk, gmu, onsite_bar, bond_grad)
    )
}

#[allow(clippy::too_many_arguments)]
fn tangent_impl<N: Dim, S: Dim>(
    dim: N,
    sdim: S,
    cl: &Cluster,
    h: &HamiltonianData,
    nmom: usize,
    wk: &Walk,
    centre: usize,
    orbitals: (usize, usize),
) -> Vec<f64> {
    let n = dim.n();
    let ns = sdim.n();
    let nn = n * ns;
    let n_steps = nmom.div_ceil(2);
    let (o0, o1) = orbitals;
    let mut wd: Vec<Vec<f64>> = vec![vec![0.0; wk.w[0].len()]];
    for a in 0..n_steps {
        let n_in = cl.size_le[a];
        let n_out = wk.w[a + 1].len() / nn;
        let (prev, wa) = (&wd[a], &wk.w[a]);
        let mut out = vec![0.0; n_out * nn];
        for c in 0..n_out {
            let oc = &mut out[c * nn..(c + 1) * nn];
            if a > 0 {
                for &(b, c2) in &cl.links[cl.link_start[c]..cl.link_start[c + 1]] {
                    if c2 >= n_in {
                        continue;
                    }
                    let blk = &h.hblk[b * n * n..(b + 1) * n * n];
                    let wc = &prev[c2 * nn..(c2 + 1) * nn];
                    for al in 0..n {
                        for be in 0..n {
                            let hv = blk[al * n + be];
                            if hv == 0.0 {
                                continue;
                            }
                            for d in 0..ns {
                                oc[al * ns + d] += hv * wc[be * ns + d];
                            }
                        }
                    }
                }
            }
            if c < n_in {
                let lev = &h.onsite[cl.origin[c] * n..(cl.origin[c] + 1) * n];
                let wc = &prev[c * nn..(c + 1) * nn];
                for al in 0..n {
                    for d in 0..ns {
                        oc[al * ns + d] += (lev[al] - h.shift(d)) * wc[al * ns + d];
                    }
                }
                if cl.origin[c] == centre {
                    for al in o0..o1 {
                        for d in 0..ns {
                            oc[al * ns + d] += wa[c * nn + al * ns + d];
                        }
                    }
                }
            }
        }
        wd.push(out);
    }
    let mut mu = vec![0.0; n * (nmom + 1)];
    for k in 0..=nmom {
        let (a, b) = (k / 2, k - k / 2);
        let size = cl.size_le[a] * nn;
        for idx in 0..size {
            mu[h.start(idx % ns) * (nmom + 1) + k] +=
                wd[a][idx] * wk.w[b][idx] + wk.w[a][idx] * wd[b][idx];
        }
    }
    mu
}

/// Derivative of the moments of the cluster centre (layout as [`moments`])
/// with respect to a common shift of the onsite levels `orbitals` of unit-cell
/// atom `centre` (all its images in the cluster): one tangent walk.
pub fn moments_tangent(
    cl: &Cluster,
    h: &HamiltonianData,
    nmom: usize,
    wk: &Walk,
    centre: usize,
    orbitals: (usize, usize),
) -> Vec<f64> {
    dispatch!(
        h.norb,
        h.n_starts(),
        tangent_impl(cl, h, nmom, wk, centre, orbitals)
    )
}
