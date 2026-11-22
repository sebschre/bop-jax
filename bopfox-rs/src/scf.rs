//! Self-consistency: the onsite shifts at which the energy is stationary,
//! dE/dshifts = 0 (the condition BOPfox solves), from gradients only.
//!
//! The gradient is `g = chi (target - shifts) + delta` (`docs/formulation.md`
//! of bopjax), with `chi = dN/dshifts` and `target = K N` the potentials the
//! occupations imply (onsite Hubbard `J` and Stoner `I`).  Its Jacobian is
//! close to `chi (K chi - 1)`.  The local approximation
//! `chi ~ -diag(D)`, `D` the density of states of each atom, spin and shell at
//! the Fermi level, gives a preconditioner that is block-diagonal in the atoms:
//!
//!   J0_i = D_i (K_i D_i + 1),   step = -J0_i^{-1} g_i
//!
//! It holds the right sign in the spin channel too, where a Stoner
//! instability (`I D > 1`) makes the Jacobian change sign.  Anderson (Pulay)
//! mixing of these preconditioned steps then converges like GMRES on the
//! linearised problem with one gradient evaluation per iteration, and
//! needs neither the Hessian nor Hessian-vector products.

use crate::energy::{Bop, Cell, Mode, Results};

#[derive(Clone, Debug)]
pub struct ScfOptions {
    /// convergence: max |dE/dshift| (eV)
    pub tol: f64,
    pub max_iter: usize,
    /// Anderson mixing parameter of the preconditioned step
    pub mixing: f64,
    /// previous steps kept by the Anderson mixing
    pub history: usize,
    /// largest change of one shift in one step (eV)
    pub max_step: f64,
    /// atoms whose shifts are solved for (default all); the others keep
    /// their starting shifts
    pub free: Option<Vec<bool>>,
    /// gradient evaluations between updates of the local response in the
    /// preconditioner (1: every evaluation; each update costs one tangent
    /// walk per shell)
    pub refresh: usize,
}

impl Default for ScfOptions {
    fn default() -> Self {
        ScfOptions {
            tol: 1e-10,
            max_iter: 200,
            mixing: 1.0,
            history: 24,
            max_step: 0.5,
            free: None,
            refresh: 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScfResult {
    /// converged shifts, `[atom][spin][shell]` flattened
    pub shifts: Vec<f64>,
    /// everything at the converged shifts, forces included
    pub results: Results,
    pub converged: bool,
    /// max |dE/dshift| after each gradient evaluation
    pub history: Vec<f64>,
}

fn flatten(g: &[Vec<Vec<f64>>]) -> Vec<f64> {
    g.iter().flatten().flatten().copied().collect()
}

fn max_abs(x: impl IntoIterator<Item = f64>) -> f64 {
    x.into_iter().fold(0.0, |m, v| {
        if v.is_finite() {
            m.max(v.abs())
        } else {
            f64::INFINITY
        }
    })
}

/// Solves `a x = b` in place (Gaussian elimination, partial pivoting).
/// Returns false if `a` is numerically singular.
fn solve_dense(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))
            .unwrap();
        if a[piv * n + col].abs() <= 1e-12 * scale || !a[piv * n + col].is_finite() {
            return false;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        for r in col + 1..n {
            let f = a[r * n + col] / a[col * n + col];
            for k in col..n {
                a[r * n + k] -= f * a[col * n + k];
            }
            b[r] -= f * b[col];
        }
    }
    for col in (0..n).rev() {
        let mut s = b[col];
        for k in col + 1..n {
            s -= a[col * n + k] * b[k];
        }
        b[col] = s / a[col * n + col];
    }
    true
}

/// Floor on the response of a channel in the preconditioner (states / eV):
/// an empty or full channel then gets about the potential-mixing step.
const RESPONSE_FLOOR: f64 = 0.05;

impl<'a> Bop<'a> {
    /// Preconditioned step `-J0^{-1} g`.
    ///
    /// `J0 = chi K chi - chi` with `chi = -(Q - q q^T / sigma)`: `Q` is the
    /// exact response of each atom's occupations to its own levels at fixed
    /// Fermi level (block-diagonal), `q = Q 1`, `sigma = 1^T Q 1`; the
    /// rank-one term keeps the electron count fixed (`chi 1 = 0`).  `K` is the
    /// onsite Hubbard and Stoner kernel.  Solved exactly in O(N) with the
    /// Sherman-Morrison formula; the uniform part of the step is arbitrary
    /// (the energy does not depend on it) and is removed by the caller.
    ///
    /// With frozen atoms (`free`) the step of their levels is zero and the
    /// remaining system `J0_ff step_f = -g_f` is solved by GMRES (applying
    /// `J0` costs O(N)).
    fn preconditioned_step(&self, res: &Results, g: &[f64], free: Option<&[bool]>) -> Vec<f64> {
        let m = self.model;
        let ns = m.nspin();
        let nsh = m.basis.len();
        let nc = ns * nsh;
        let n = self.topo.n_atoms;
        // per atom: Q, K, q
        let mut qmat = vec![0.0; n * nc * nc];
        let mut kmat = vec![0.0; n * nc * nc];
        let mut qvec = vec![0.0; n * nc];
        for i in 0..n {
            let sp = self.topo.species[i];
            let qi = &mut qmat[i * nc * nc..(i + 1) * nc * nc];
            for s in 0..ns {
                for k in 0..nsh {
                    for k2 in 0..nsh {
                        let (c, c2) = (s * nsh + k, s * nsh + k2);
                        let r = &res.local_response[i][s];
                        qi[c * nc + c2] = -0.5 * (r[k][k2] + r[k2][k]);
                    }
                }
            }
            for c in 0..nc {
                qi[c * nc + c] = qi[c * nc + c].max(RESPONSE_FLOOR);
            }
            for c in 0..nc {
                qvec[i * nc + c] = (0..nc).map(|c2| qi[c * nc + c2]).sum();
            }
            let ki = &mut kmat[i * nc * nc..(i + 1) * nc * nc];
            for c in 0..nc {
                for c2 in 0..nc {
                    let (s, k, s2, k2) = (c / nsh, c % nsh, c2 / nsh, c2 % nsh);
                    let mut kk = m.hubbard[sp];
                    if ns == 2 && k == k2 {
                        let ist = m.stoner[sp][k];
                        kk += if s == s2 { -0.5 * ist } else { 0.5 * ist };
                    }
                    ki[c * nc + c2] = kk;
                }
            }
        }
        let sigma: f64 = qvec.iter().sum();
        let diag_step = |g: &[f64]| -> Vec<f64> {
            (0..n * nc)
                .map(|q| -g[q] / qmat[(q / nc) * nc * nc + (q % nc) * (nc + 1)])
                .collect()
        };
        if let Some(free) = free {
            let block = |mat: &[f64], v: &[f64]| -> Vec<f64> {
                let mut out = vec![0.0; n * nc];
                for i in 0..n {
                    for r in 0..nc {
                        out[i * nc + r] = (0..nc)
                            .map(|c| mat[i * nc * nc + r * nc + c] * v[i * nc + c])
                            .sum();
                    }
                }
                out
            };
            let p_op = |v: &[f64]| -> Vec<f64> {
                let t = dot(&qvec, v) / sigma;
                let mut out = block(&qmat, v);
                out.iter_mut().zip(&qvec).for_each(|(o, q)| *o -= q * t);
                out
            };
            let mask = |v: &mut Vec<f64>| {
                for q in 0..n * nc {
                    if !free[q / nc] {
                        v[q] = 0.0;
                    }
                }
            };
            let op = |v: &[f64]| -> Vec<f64> {
                let pv = p_op(v);
                let mut w = block(&kmat, &pv);
                w.iter_mut().zip(v).for_each(|(a, b)| *a += b);
                let mut out = p_op(&w);
                mask(&mut out);
                out
            };
            let mut b: Vec<f64> = g.iter().map(|x| -x).collect();
            mask(&mut b);
            return gmres(op, &b, 1e-8, 40, 400).unwrap_or_else(|| {
                let mut d = diag_step(g);
                mask(&mut d);
                d
            });
        }
        let mut x1 = vec![0.0; n * nc];
        let mut x2 = vec![0.0; n * nc];
        let mut ok = sigma > 0.0;
        for i in 0..n {
            let qi = &qmat[i * nc * nc..(i + 1) * nc * nc];
            let ki = &kmat[i * nc * nc..(i + 1) * nc * nc];
            // u = -Q^{-1} g
            let mut u: Vec<f64> = g[i * nc..(i + 1) * nc].iter().map(|x| -x).collect();
            ok &= solve_dense(&mut qi.to_vec(), &mut u, nc);
            // M = 1 + K Q
            let mut mm = vec![0.0; nc * nc];
            for r in 0..nc {
                for c in 0..nc {
                    mm[r * nc + c] = if r == c { 1.0 } else { 0.0 }
                        + (0..nc)
                            .map(|t| ki[r * nc + t] * qi[t * nc + c])
                            .sum::<f64>();
                }
            }
            let mut kq: Vec<f64> = (0..nc)
                .map(|r| (0..nc).map(|t| ki[r * nc + t] * qvec[i * nc + t]).sum())
                .collect();
            ok &= solve_dense(&mut mm.clone(), &mut u, nc);
            ok &= solve_dense(&mut mm, &mut kq, nc);
            x1[i * nc..(i + 1) * nc].copy_from_slice(&u);
            x2[i * nc..(i + 1) * nc].copy_from_slice(&kq);
        }
        let denom = 1.0 - dot(&qvec, &x2) / sigma;
        if ok && denom.abs() > 1e-12 {
            let t = dot(&qvec, &x1) / denom;
            let step: Vec<f64> = (0..n * nc).map(|q| x1[q] + x2[q] * t / sigma).collect();
            if step.iter().all(|v| v.is_finite()) {
                return step;
            }
        }
        diag_step(g)
    }

    /// Self-consistent shifts from `shifts0` (default zero), and the energy,
    /// forces and everything else at them.
    pub fn scf(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        shifts0: Option<&[f64]>,
        opt: &ScfOptions,
    ) -> ScfResult {
        let n_shift = self.shift_len();
        let per_atom = n_shift / self.topo.n_atoms.max(1);
        let mut x = shifts0.map_or_else(|| vec![0.0; n_shift], |s| s.to_vec());
        assert_eq!(x.len(), n_shift, "shifts0 has the wrong length");
        let index: Vec<usize> = match &opt.free {
            None => (0..n_shift).collect(),
            Some(f) => (0..n_shift).filter(|&q| f[q / per_atom]).collect(),
        };
        // with every level free a uniform shift leaves the energy unchanged:
        // keep the mean of the shifts where it started
        let all_free = index.len() == n_shift;
        let k = index.len();

        let mut evals = 0usize;
        let mut response: Vec<Vec<Vec<Vec<f64>>>> = Vec::new();
        let mut eval = |y: &[f64], x: &mut Vec<f64>| -> (Results, Vec<f64>, f64) {
            for (q, &v) in index.iter().zip(y) {
                x[*q] = v;
            }
            let fresh = evals.is_multiple_of(opt.refresh.max(1));
            evals += 1;
            let mode = if fresh {
                Mode::ScfStep
            } else {
                Mode::ShiftGradient
            };
            let mut res = self.compute_mode(pos, cell, x, mode);
            if fresh {
                response.clone_from(&res.local_response);
            } else {
                res.local_response.clone_from(&response);
            }
            let g = flatten(&res.shift_gradient);
            let gmax = max_abs(index.iter().map(|&q| g[q]));
            let full = self.preconditioned_step(
                &res,
                &g,
                if all_free { None } else { opt.free.as_deref() },
            );
            let mut f: Vec<f64> = index.iter().map(|&q| full[q]).collect();
            if all_free && k > 0 {
                let mean = f.iter().sum::<f64>() / k as f64;
                f.iter_mut().for_each(|v| *v -= mean);
            }
            let fmax = max_abs(f.iter().copied());
            if fmax > opt.max_step {
                let s = opt.max_step / fmax;
                f.iter_mut().for_each(|v| *v *= s);
            }
            (res, f, gmax)
        };

        let mut y: Vec<f64> = index.iter().map(|&q| x[q]).collect();
        let (_, mut f, mut gmax) = eval(&y, &mut x);
        let mut history = vec![gmax];
        let (mut y_best, mut g_best) = (y.clone(), gmax);
        let mut dy: Vec<Vec<f64>> = Vec::new();
        let mut df: Vec<Vec<f64>> = Vec::new();
        let mut beta = opt.mixing;
        let mut it = 0;
        while gmax > opt.tol && it < opt.max_iter {
            it += 1;
            // Anderson: minimise |f - sum gamma_j df_j| over the history
            let h = df.len();
            let mut gamma = vec![0.0; h];
            if h > 0 {
                let mut a = vec![0.0; h * h];
                for r in 0..h {
                    for c in 0..h {
                        a[r * h + c] = dot(&df[r], &df[c]);
                    }
                }
                let diag = (0..h).map(|r| a[r * h + r]).fold(0.0, f64::max);
                for r in 0..h {
                    a[r * h + r] += 1e-10 * diag + 1e-300;
                }
                let mut b: Vec<f64> = (0..h).map(|r| dot(&df[r], &f)).collect();
                if solve_dense(&mut a, &mut b, h) {
                    gamma = b;
                } else {
                    dy.clear();
                    df.clear();
                }
            }
            let mut y_new: Vec<f64> = (0..k).map(|q| y[q] + beta * f[q]).collect();
            for (j, gm) in gamma.iter().enumerate().take(df.len()) {
                for q in 0..k {
                    y_new[q] -= gm * (dy[j][q] + beta * df[j][q]);
                }
            }
            if y_new.iter().any(|v| !v.is_finite()) {
                y_new = (0..k).map(|q| y[q] + beta * f[q]).collect();
            }
            let (_, f_new, g_new) = eval(&y_new, &mut x);
            history.push(g_new);
            if g_new < g_best {
                g_best = g_new;
                y_best.clone_from(&y_new);
            }
            if g_new.is_nan() || g_new > 10.0 * g_best {
                // diverging: restart from the best iterate with less mixing
                dy.clear();
                df.clear();
                beta *= 0.5;
                y.clone_from(&y_best);
                let (_, fb, gb) = eval(&y, &mut x);
                history.push(gb);
                f = fb;
                gmax = gb;
                continue;
            }
            dy.insert(0, (0..k).map(|q| y_new[q] - y[q]).collect());
            df.insert(0, (0..k).map(|q| f_new[q] - f[q]).collect());
            dy.truncate(opt.history);
            df.truncate(opt.history);
            y = y_new;
            f = f_new;
            gmax = g_new;
        }
        if g_best < gmax {
            y = y_best;
        }
        for (q, &v) in index.iter().zip(&y) {
            x[*q] = v;
        }
        let results = self.compute_mode(pos, cell, &x, Mode::Full);
        let g = flatten(&results.shift_gradient);
        let gmax = max_abs(index.iter().map(|&q| g[q]));
        ScfResult {
            shifts: x,
            results,
            converged: gmax <= opt.tol,
            history,
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Restarted GMRES for `op x = b` from `x = 0`; None if it does not reach
/// `|op x - b| <= tol |b|`.
fn gmres(
    op: impl Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    tol: f64,
    restart: usize,
    max_iter: usize,
) -> Option<Vec<f64>> {
    let n = b.len();
    let bnorm = dot(b, b).sqrt();
    let mut x = vec![0.0; n];
    if bnorm == 0.0 {
        return Some(x);
    }
    let mut iters = 0;
    while iters < max_iter {
        let ax = op(&x);
        let r: Vec<f64> = (0..n).map(|q| b[q] - ax[q]).collect();
        let beta = dot(&r, &r).sqrt();
        if beta <= tol * bnorm {
            return Some(x);
        }
        let mut v = vec![r.iter().map(|z| z / beta).collect::<Vec<f64>>()];
        let mut h = vec![vec![0.0; restart]; restart + 1];
        let (mut cs, mut sn) = (vec![0.0; restart], vec![0.0; restart]);
        let mut e = vec![0.0; restart + 1];
        e[0] = beta;
        let mut k_used = 0;
        for j in 0..restart {
            iters += 1;
            let mut w = op(&v[j]);
            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
                for q in 0..n {
                    w[q] -= h[i][j] * v[i][q];
                }
            }
            let wnorm = dot(&w, &w).sqrt();
            h[j + 1][j] = wnorm;
            for i in 0..j {
                let t = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = t;
            }
            let d = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            if d == 0.0 {
                break;
            }
            cs[j] = h[j][j] / d;
            sn[j] = h[j + 1][j] / d;
            h[j][j] = d;
            h[j + 1][j] = 0.0;
            e[j + 1] = -sn[j] * e[j];
            e[j] *= cs[j];
            k_used = j + 1;
            v.push(w.iter().map(|z| z / wnorm.max(1e-300)).collect());
            if e[j + 1].abs() <= tol * bnorm || iters >= max_iter {
                break;
            }
        }
        // back substitution
        let mut y = vec![0.0; k_used];
        for i in (0..k_used).rev() {
            let mut s = e[i];
            for l in i + 1..k_used {
                s -= h[i][l] * y[l];
            }
            y[i] = s / h[i][i];
        }
        for (i, yi) in y.iter().enumerate() {
            for q in 0..n {
                x[q] += yi * v[i][q];
            }
        }
        if !x.iter().all(|z| z.is_finite()) {
            return None;
        }
    }
    let ax = op(&x);
    let res = (0..n).map(|q| (b[q] - ax[q]).powi(2)).sum::<f64>().sqrt();
    (res <= tol * bnorm).then_some(x)
}
