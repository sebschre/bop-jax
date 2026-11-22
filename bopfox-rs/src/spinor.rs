//! The spinor walk of non-collinear magnetism (`bopjax.bop.moments_noncollinear`).
//!
//! States are (orbital, spin) in the global spin frame, `M = 2 norb` per
//! atom.  The hopping is spin-diagonal (`H_b x 1`), the onsite block of atom
//! `i` is `O_i = sum_s level_{i s} (1 x P_s(e_i)) + xi L.S`, with `P_s` the
//! projectors on spin up/down along the local axis `e_i`.  The walk starts
//! from every (orbital d, global spin u) at once and gives the 2x2 moment
//! matrices `m_k[d][u][v] = <d u|H^k|d v>`; the moments of the local spin
//! channels are `mu_k[s][d] = Re tr(P_s m_k[d])`.
//!
//! The backward pass is that of `moments.rs` in complex arithmetic.  For a
//! real energy, the adjoint of a complex vector is `wbar = dE/dRe w + i dE/dIm w`;
//! a linear step `w' = A w` sends `wbar += A^H wbar'` back and contributes
//! `wbar' w^H` to the cotangent of `A`, and for a real parameter `t` of `A`,
//! `dE/dt = Re sum conj(Abar) dA/dt`.

use crate::complex::{C64, ZERO};
use crate::geometry::Cluster;

pub struct SpinorData<'a> {
    pub norb: usize,
    /// `n_bonds * norb * norb` real hopping blocks
    pub hblk: &'a [f64],
    /// `n_atoms * M * M` onsite blocks, row/column index `orbital * 2 + spin`
    pub onsite: &'a [C64],
    /// the onsite blocks couple no two orbitals (no spin-orbit coupling)
    pub orbital_diagonal: bool,
}

pub struct SpinorWalk {
    /// `w[a]`: `slots * M * M`, `[slot][row (orbital, spin)][column (start orbital, start spin)]`
    pub w: Vec<Vec<C64>>,
}

fn apply_onsite(o: &[C64], src: &[C64], dst: &mut [C64], m: usize, diag: bool, adjoint: bool) {
    // dst += O src (or O^H src)
    for r in 0..m {
        let (lo, hi) = if diag { (r & !1, (r & !1) + 2) } else { (0, m) };
        for q in lo..hi {
            let ov = if adjoint {
                o[q * m + r].conj()
            } else {
                o[r * m + q]
            };
            if ov == ZERO {
                continue;
            }
            let s = &src[q * m..(q + 1) * m];
            let d = &mut dst[r * m..(r + 1) * m];
            for c in 0..m {
                d[c] += ov * s[c];
            }
        }
    }
}

fn apply_hopping(blk: &[f64], src: &[C64], dst: &mut [C64], n: usize, transpose: bool) {
    let m = 2 * n;
    for al in 0..n {
        for be in 0..n {
            let hv = if transpose {
                blk[be * n + al]
            } else {
                blk[al * n + be]
            };
            if hv == 0.0 {
                continue;
            }
            for t in 0..2 {
                let s = &src[(be * 2 + t) * m..(be * 2 + t + 1) * m];
                let d = &mut dst[(al * 2 + t) * m..(al * 2 + t + 1) * m];
                for c in 0..m {
                    d[c] += s[c] * hv;
                }
            }
        }
    }
}

/// One step `w_{a+1} = H w_a` (plus `dO w_extra` on the slots of `centre`).
fn step(
    cl: &Cluster,
    h: &SpinorData,
    a: usize,
    n_out: usize,
    prev: &[C64],
    extra: Option<(&[C64], &[C64], usize)>,
) -> Vec<C64> {
    let n = h.norb;
    let m = 2 * n;
    let mm = m * m;
    let nn = n * n;
    let n_in = cl.size_le[a];
    let mut out = vec![ZERO; n_out * mm];
    for c in 0..n_out {
        let oc = &mut out[c * mm..(c + 1) * mm];
        for &(b, c2) in &cl.links[cl.link_start[c]..cl.link_start[c + 1]] {
            if c2 < n_in {
                apply_hopping(
                    &h.hblk[b * nn..(b + 1) * nn],
                    &prev[c2 * mm..(c2 + 1) * mm],
                    oc,
                    n,
                    false,
                );
            }
        }
        if c < n_in {
            let o = cl.origin[c];
            apply_onsite(
                &h.onsite[o * mm..(o + 1) * mm],
                &prev[c * mm..(c + 1) * mm],
                oc,
                m,
                h.orbital_diagonal,
                false,
            );
            if let Some((d_o, wa, centre)) = extra {
                if o == centre {
                    apply_onsite(d_o, &wa[c * mm..(c + 1) * mm], oc, m, false, false);
                }
            }
        }
    }
    out
}

pub fn walk(cl: &Cluster, h: &SpinorData, nmom: usize) -> SpinorWalk {
    let m = 2 * h.norb;
    let top = cl.size_le.len() - 1;
    let mut w0 = vec![ZERO; m * m];
    for r in 0..m {
        w0[r * m + r] = crate::complex::ONE;
    }
    let mut w = vec![w0];
    for a in 0..nmom.div_ceil(2) {
        let n_out = cl.size_le[(a + 1).min(top)];
        let next = step(cl, h, a, n_out, &w[a], None);
        w.push(next);
    }
    SpinorWalk { w }
}

/// `sum over the first slots of conj(x) y` for every pair of start columns:
/// `[d][k][u][v]` for the moments, from two families of vectors.
fn pair_moments(
    cl: &Cluster,
    x: &[Vec<C64>],
    y: &[Vec<C64>],
    norb: usize,
    nmom: usize,
    out: &mut [C64],
) {
    let m = 2 * norb;
    let mm = m * m;
    for k in 0..=nmom {
        let (a, b) = (k / 2, k - k / 2);
        for c in 0..cl.size_le[a] {
            let (xa, yb) = (&x[a][c * mm..(c + 1) * mm], &y[b][c * mm..(c + 1) * mm]);
            for r in 0..m {
                for d in 0..norb {
                    for u in 0..2 {
                        let xv = xa[r * m + d * 2 + u].conj();
                        for v in 0..2 {
                            out[((d * (nmom + 1) + k) * 2 + u) * 2 + v] +=
                                xv * yb[r * m + d * 2 + v];
                        }
                    }
                }
            }
        }
    }
}

/// Moment matrices `m_k[d][u][v]`, layout `((d * (nmom+1) + k) * 2 + u) * 2 + v`.
pub fn moment_matrices(cl: &Cluster, wk: &SpinorWalk, norb: usize, nmom: usize) -> Vec<C64> {
    let mut out = vec![ZERO; norb * (nmom + 1) * 4];
    pair_moments(cl, &wk.w, &wk.w, norb, nmom, &mut out);
    out
}

/// Derivative of the moment matrices when the onsite block of unit-cell
/// atom `centre` changes by `d_onsite` (`M x M`): one tangent walk.
pub fn moment_matrices_tangent(
    cl: &Cluster,
    h: &SpinorData,
    nmom: usize,
    wk: &SpinorWalk,
    centre: usize,
    d_onsite: &[C64],
) -> Vec<C64> {
    let m = 2 * h.norb;
    let mut wd: Vec<Vec<C64>> = vec![vec![ZERO; m * m]];
    for a in 0..nmom.div_ceil(2) {
        let n_out = wk.w[a + 1].len() / (m * m);
        let next = step(cl, h, a, n_out, &wd[a], Some((d_onsite, &wk.w[a], centre)));
        wd.push(next);
    }
    let mut out = vec![ZERO; h.norb * (nmom + 1) * 4];
    pair_moments(cl, &wd, &wk.w, h.norb, nmom, &mut out);
    pair_moments(cl, &wk.w, &wd, h.norb, nmom, &mut out);
    out
}

/// Backward pass for `E = sum_{d,k} Re tr(C_k[d] m_k[d])`, i.e.
/// `Re sum_{uv} C[v][u] m[u][v]`, with `cmat` laid out as the moment matrices
/// (`[d][k][v][u]` holding `C[v][u]`).  Adds the cotangent of the onsite blocks
/// to `onsite_bar` (`n_atoms * M * M`) and calls `bond_grad(b, dE/dH_b)`.
#[allow(clippy::too_many_arguments)]
pub fn backward<F: FnMut(usize, &[f64])>(
    cl: &Cluster,
    h: &SpinorData,
    nmom: usize,
    wk: &SpinorWalk,
    cmat: &[C64],
    onsite_bar: &mut [C64],
    mut bond_grad: Option<&mut F>,
) {
    let n = h.norb;
    let m = 2 * n;
    let mm = m * m;
    let nn = n * n;
    let n_steps = nmom.div_ceil(2);
    let mut wbar: Vec<Vec<C64>> = wk.w.iter().map(|x| vec![ZERO; x.len()]).collect();
    for k in 0..=nmom {
        let (a, b) = (k / 2, k - k / 2);
        for c in 0..cl.size_le[a] {
            for r in 0..m {
                for d in 0..n {
                    let cm = |v: usize, u: usize| cmat[((d * (nmom + 1) + k) * 2 + v) * 2 + u];
                    let base = c * mm + r * m + d * 2;
                    let (a0, a1) = (wk.w[a][base], wk.w[a][base + 1]);
                    let (b0, b1) = (wk.w[b][base], wk.w[b][base + 1]);
                    // Bbar_v += sum_u C[u][v] A_u ;  Abar_u += sum_v C[v][u] B_v
                    let bb0 = cm(0, 0) * a0 + cm(1, 0) * a1;
                    let bb1 = cm(0, 1) * a0 + cm(1, 1) * a1;
                    let ab0 = cm(0, 0) * b0 + cm(1, 0) * b1;
                    let ab1 = cm(0, 1) * b0 + cm(1, 1) * b1;
                    wbar[b][base] += bb0;
                    wbar[b][base + 1] += bb1;
                    wbar[a][base] += ab0;
                    wbar[a][base + 1] += ab1;
                }
            }
        }
    }
    let mut hbar = vec![0.0; nn];
    for a in (0..n_steps).rev() {
        let n_in = cl.size_le[a];
        let (lo, hi) = wbar.split_at_mut(a + 1);
        let (wbar_a, wbar_next) = (&mut lo[a], &hi[0]);
        let n_out = wbar_next.len() / mm;
        let wa = &wk.w[a];
        for c in 0..n_out {
            let gc = &wbar_next[c * mm..(c + 1) * mm];
            for &(b, c2) in &cl.links[cl.link_start[c]..cl.link_start[c + 1]] {
                if c2 >= n_in {
                    continue;
                }
                let blk = &h.hblk[b * nn..(b + 1) * nn];
                apply_hopping(blk, gc, &mut wbar_a[c2 * mm..(c2 + 1) * mm], n, true);
                if let Some(f) = bond_grad.as_mut() {
                    let src = &wa[c2 * mm..(c2 + 1) * mm];
                    for al in 0..n {
                        for be in 0..n {
                            let mut acc = 0.0;
                            for t in 0..2 {
                                let g = &gc[(al * 2 + t) * m..(al * 2 + t + 1) * m];
                                let s = &src[(be * 2 + t) * m..(be * 2 + t + 1) * m];
                                for q in 0..m {
                                    acc += g[q].dot_re(s[q]);
                                }
                            }
                            hbar[al * n + be] = acc;
                        }
                    }
                    f(b, &hbar);
                }
            }
            if c < n_in {
                let o = cl.origin[c];
                let ob = &h.onsite[o * mm..(o + 1) * mm];
                apply_onsite(
                    ob,
                    gc,
                    &mut wbar_a[c * mm..(c + 1) * mm],
                    m,
                    h.orbital_diagonal,
                    true,
                );
                let src = &wa[c * mm..(c + 1) * mm];
                let dst = &mut onsite_bar[o * mm..(o + 1) * mm];
                for r in 0..m {
                    let (lo, hi) = if h.orbital_diagonal {
                        (r & !1, (r & !1) + 2)
                    } else {
                        (0, m)
                    };
                    let g = &gc[r * m..(r + 1) * m];
                    for q in lo..hi {
                        let s = &src[q * m..(q + 1) * m];
                        let mut acc = ZERO;
                        for col in 0..m {
                            acc += g[col] * s[col].conj();
                        }
                        dst[r * m + q] += acc;
                    }
                }
            }
        }
    }
}
