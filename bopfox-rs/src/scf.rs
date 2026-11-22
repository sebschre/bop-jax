//! Self-consistency of the onsite shifts, from gradients only.
//!
//! Two conditions, as in `bopjax.make_potential`:
//!
//! * **stationary** (default, what BOPfox solves): dE/dshifts = 0.  The
//!   energy is then stationary in the shifts and the forces are partial
//!   derivatives.
//! * **potential**: the shifts equal the potentials their occupations imply,
//!   `shifts = K N(shifts)` (onsite Hubbard `J` and Stoner `I`), the usual
//!   charge and spin self-consistency of tight binding.  It stays well
//!   conditioned where moments saturate and the stationary point is far away,
//!   but the energy is not stationary there: the forces carry the response of
//!   the shifts, obtained by implicit differentiation (one adjoint linear solve,
//!   see [`Bop::scf`]).
//!
//! Both are solved by Anderson (Pulay) mixing of preconditioned steps, one
//! energy evaluation per iteration and no Hessian.  The gradient is
//! `g = chi (target - shifts) + delta` with `chi = dN/dshifts`, so
//! `dg/dshifts ~ chi K chi - chi` and `d(target - shifts)/dshifts ~ K chi - 1`.
//! The preconditioner models `chi = -(Q - q q^T / sigma)`: `Q` is the exact
//! response of each atom's occupations to its own levels at fixed Fermi level
//! (one tangent walk per shell, `local_response`), `q = Q 1`,
//! `sigma = 1^T Q 1`, and the rank-one term keeps the electron count fixed.
//! The resulting systems are solved exactly in O(N) (Sherman-Morrison), or by
//! GMRES on the free levels when some atoms are frozen.  Measured Jacobians
//! show that each ingredient matters: the DOS at the Fermi level underestimates
//! the local response by a third, with `J` = 8 eV the fixed-N coupling between
//! atoms is as large as the diagonal, and with Stoner magnetism the spin block
//! can change sign.

use crate::energy::{add3, symmetrised, Bop, Cell, Mode, Results};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Condition {
    /// dE/dshifts = 0 (BOPfox)
    Stationary,
    /// shifts = K N(shifts)
    Potential,
}

#[derive(Clone, Debug)]
pub struct ScfOptions {
    pub condition: Condition,
    /// convergence: max |dE/dshift| (stationary) or max |K N - shifts|
    /// (potential), eV
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
    /// evaluations between updates of the local response in the
    /// preconditioner (1: every evaluation; each update costs one tangent
    /// walk per shell)
    pub refresh: usize,
    /// relative tolerance of the adjoint solve (potential condition)
    pub adjoint_tol: f64,
    /// when the Anderson mixing stalls, continue with Newton-Krylov steps
    /// (Jacobian-vector products by finite differences of the residual)
    pub newton_fallback: bool,
}

impl ScfOptions {
    /// Defaults for the potential condition: smaller mixing and a shorter
    /// history than for the stationary one.  Far from self-consistency, near
    /// saturated moments (clusters, surfaces), longer histories carry stale
    /// information and full steps overshoot.  Measured: a 15-atom iron cluster
    /// converges in 80-120 evaluations with these settings and not at all
    /// with mixing 0.5; bulk iron needs 20-30.
    pub fn potential() -> Self {
        ScfOptions {
            condition: Condition::Potential,
            mixing: 0.3,
            history: 8,
            ..Default::default()
        }
    }
}

impl Default for ScfOptions {
    fn default() -> Self {
        ScfOptions {
            condition: Condition::Stationary,
            tol: 1e-10,
            max_iter: 200,
            mixing: 1.0,
            history: 24,
            max_step: 0.5,
            free: None,
            refresh: 4,
            adjoint_tol: 1e-10,
            newton_fallback: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScfResult {
    /// converged shifts, `[atom][spin][shell]` flattened
    pub shifts: Vec<f64>,
    /// everything at the converged shifts, forces included (for the potential
    /// condition with the response of the shifts)
    pub results: Results,
    pub converged: bool,
    /// the convergence measure after each evaluation
    pub history: Vec<f64>,
    /// final convergence measure
    pub residual: f64,
    /// operator applications of the adjoint solve (potential condition)
    pub adjoint_iterations: usize,
    /// evaluations of the residual (gradient or occupations), in total
    pub evaluations: usize,
    /// the Anderson history at the end, to continue from (`scf_continued`)
    pub mixing: MixingHistory,
}

/// Steps and step differences of the Anderson mixing, newest first.
#[derive(Clone, Debug, Default)]
pub struct MixingHistory {
    pub dy: Vec<Vec<f64>>,
    pub df: Vec<Vec<f64>>,
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

/// The model of the response and the onsite kernel, per atom.
struct Precond {
    n: usize,
    nc: usize,
    q: Vec<f64>,  // n * nc * nc, the local response Q (positive)
    k: Vec<f64>,  // n * nc * nc, the kernel K
    qv: Vec<f64>, // n * nc, Q 1
    sigma: f64,
    free: Option<Vec<bool>>,
}

impl Precond {
    fn new(bop: &Bop, res: &Results, free: Option<&[bool]>) -> Self {
        let m = bop.model;
        let ns = m.nspin();
        let nsh = m.basis.len();
        let nc = ns * nsh;
        let n = bop.topo.n_atoms;
        let mut q = vec![0.0; n * nc * nc];
        let mut k = vec![0.0; n * nc * nc];
        let mut qv = vec![0.0; n * nc];
        for i in 0..n {
            let sp = bop.topo.species[i];
            let qi = &mut q[i * nc * nc..(i + 1) * nc * nc];
            let r = &res.local_response[i];
            for c in 0..nc {
                for c2 in 0..nc {
                    let (rc, rc2) = (r[c / nsh][c % nsh][c2], r[c2 / nsh][c2 % nsh][c]);
                    qi[c * nc + c2] = -0.5 * (rc + rc2);
                }
            }
            for c in 0..nc {
                qi[c * nc + c] = qi[c * nc + c].max(RESPONSE_FLOOR);
            }
            for c in 0..nc {
                qv[i * nc + c] = (0..nc).map(|c2| qi[c * nc + c2]).sum();
            }
            let ki = &mut k[i * nc * nc..(i + 1) * nc * nc];
            for c in 0..nc {
                for c2 in 0..nc {
                    let (s, a, s2, b) = (c / nsh, c % nsh, c2 / nsh, c2 % nsh);
                    let mut kk = m.hubbard[sp];
                    if ns == 2 && a == b {
                        let ist = m.stoner[sp][a];
                        kk += if s == s2 { -0.5 * ist } else { 0.5 * ist };
                    }
                    ki[c * nc + c2] = kk;
                }
            }
        }
        let sigma = qv.iter().sum();
        Precond {
            n,
            nc,
            q,
            k,
            qv,
            sigma,
            free: free.map(|f| f.to_vec()),
        }
    }

    fn block(&self, mat: &[f64], v: &[f64]) -> Vec<f64> {
        let nc = self.nc;
        let mut out = vec![0.0; self.n * nc];
        for i in 0..self.n {
            for r in 0..nc {
                out[i * nc + r] = (0..nc)
                    .map(|c| mat[i * nc * nc + r * nc + c] * v[i * nc + c])
                    .sum();
            }
        }
        out
    }

    /// K v (the kernel is symmetric)
    fn kernel(&self, v: &[f64]) -> Vec<f64> {
        self.block(&self.k, v)
    }

    /// P v = Q v - q (q.v) / sigma  (= -chi v)
    fn p(&self, v: &[f64]) -> Vec<f64> {
        let t = dot(&self.qv, v) / self.sigma;
        let mut out = self.block(&self.q, v);
        out.iter_mut().zip(&self.qv).for_each(|(o, q)| *o -= q * t);
        out
    }

    fn mask(&self, v: &mut [f64]) {
        if let Some(f) = &self.free {
            for (q, x) in v.iter_mut().enumerate() {
                if !f[q / self.nc] {
                    *x = 0.0;
                }
            }
        }
    }

    /// `1 + A B` per atom block.
    fn one_plus(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let nc = self.nc;
        let mut out = vec![0.0; self.n * nc * nc];
        for i in 0..self.n {
            let o = i * nc * nc;
            for r in 0..nc {
                for c in 0..nc {
                    out[o + r * nc + c] = if r == c { 1.0 } else { 0.0 }
                        + (0..nc)
                            .map(|t| a[o + r * nc + t] * b[o + t * nc + c])
                            .sum::<f64>();
                }
            }
        }
        out
    }

    /// Solves `(B - a b^T / sigma) x = z`, B block-diagonal (Sherman-Morrison).
    fn rank_one_solve(&self, bmat: &[f64], a: &[f64], b: &[f64], z: &[f64]) -> Option<Vec<f64>> {
        let nc = self.nc;
        let mut x1 = z.to_vec();
        let mut x2 = a.to_vec();
        for i in 0..self.n {
            let blk = &bmat[i * nc * nc..(i + 1) * nc * nc];
            if !solve_dense(&mut blk.to_vec(), &mut x1[i * nc..(i + 1) * nc], nc)
                || !solve_dense(&mut blk.to_vec(), &mut x2[i * nc..(i + 1) * nc], nc)
            {
                return None;
            }
        }
        let denom = 1.0 - dot(b, &x2) / self.sigma;
        if self.sigma.is_nan() || self.sigma <= 0.0 || denom.abs() < 1e-12 {
            return None;
        }
        let t = dot(b, &x1) / denom;
        let x: Vec<f64> = x1
            .iter()
            .zip(&x2)
            .map(|(u, v)| u + v * t / self.sigma)
            .collect();
        x.iter().all(|v| v.is_finite()).then_some(x)
    }

    /// Solve `op x = z` on the free levels (GMRES), x zero elsewhere.
    fn masked_solve(&self, op: impl Fn(&[f64]) -> Vec<f64>, z: &[f64]) -> Option<Vec<f64>> {
        let mut b = z.to_vec();
        self.mask(&mut b);
        let op = |v: &[f64]| {
            let mut o = op(v);
            self.mask(&mut o);
            o
        };
        gmres(op, &b, 1e-8, 40, 400)
    }

    fn diagonal(&self, z: &[f64]) -> Vec<f64> {
        let nc = self.nc;
        let mut out: Vec<f64> = (0..self.n * nc)
            .map(|q| z[q] / self.q[(q / nc) * nc * nc + (q % nc) * (nc + 1)])
            .collect();
        self.mask(&mut out);
        out
    }

    /// Stationary condition: `step = -J0^{-1} g`, `J0 = P (K P + 1)`.
    fn stationary(&self, g: &[f64]) -> Vec<f64> {
        let neg: Vec<f64> = g.iter().map(|x| -x).collect();
        let solved = if self.free.is_none() {
            // P u = -g  =>  u = -Q^{-1} g (g sums to zero); then (1 + K P) step = u
            let mut u = neg.clone();
            let nc = self.nc;
            let mut ok = true;
            for i in 0..self.n {
                ok &= solve_dense(
                    &mut self.q[i * nc * nc..(i + 1) * nc * nc].to_vec(),
                    &mut u[i * nc..(i + 1) * nc],
                    nc,
                );
            }
            if ok {
                self.potential(&u)
            } else {
                None
            }
        } else {
            self.masked_solve(
                |v| {
                    let pv = self.p(v);
                    let mut w = self.kernel(&pv);
                    w.iter_mut().zip(v).for_each(|(a, b)| *a += b);
                    self.p(&w)
                },
                &neg,
            )
        };
        solved.unwrap_or_else(|| self.diagonal(&neg))
    }

    /// `(1 + K P)^{-1} r`: the Newton step of the potential condition
    /// `K N - shifts = r`, whose Jacobian is `K chi - 1 = -(1 + K P)`.
    fn potential(&self, r: &[f64]) -> Option<Vec<f64>> {
        if self.free.is_none() {
            let kq = self.kernel(&self.qv);
            self.rank_one_solve(&self.one_plus(&self.k, &self.q), &kq, &self.qv, r)
        } else {
            self.masked_solve(
                |v| {
                    let mut o = self.kernel(&self.p(v));
                    o.iter_mut().zip(v).for_each(|(a, b)| *a += b);
                    o
                },
                r,
            )
        }
    }

    /// `(1 + P K)^{-1} z`, the transpose of [`Precond::potential`].
    fn adjoint(&self, z: &[f64]) -> Vec<f64> {
        let solved = if self.free.is_none() {
            let kq = self.kernel(&self.qv);
            self.rank_one_solve(&self.one_plus(&self.q, &self.k), &self.qv, &kq, z)
        } else {
            self.masked_solve(
                |v| {
                    let mut o = self.p(&self.kernel(v));
                    o.iter_mut().zip(v).for_each(|(a, b)| *a += b);
                    o
                },
                z,
            )
        };
        solved.unwrap_or_else(|| {
            let mut o = z.to_vec();
            self.mask(&mut o);
            o
        })
    }
}

/// Anderson (type II, Pulay) mixing of the steps `f(y)` that `eval` returns
/// together with the convergence measure.  Keeps the best iterate; when the
/// measure grows tenfold over the best one it restarts from there with an
/// empty history and half the mixing.  Returns the best iterate and the
/// history of the measure.
fn anderson(
    mut eval: impl FnMut(&[f64]) -> (Vec<f64>, f64),
    y0: Vec<f64>,
    opt: &ScfOptions,
    previous: Option<&MixingHistory>,
) -> (Vec<f64>, Vec<f64>, MixingHistory) {
    let k = y0.len();
    let mut y = y0;
    let (mut f, mut err) = eval(&y);
    let mut history = vec![err];
    let (mut y_best, mut e_best) = (y.clone(), err);
    let (mut dy, mut df) = match previous {
        Some(p) if p.dy.first().is_some_and(|v| v.len() == k) => (p.dy.clone(), p.df.clone()),
        _ => (Vec::new(), Vec::new()),
    };
    let mut beta = opt.mixing;
    let mut it = 0;
    // evaluations since the best measure last fell by 10%
    let mut stalled = 0usize;
    let patience = opt.history.max(10);
    // restarts in a row that did not bring the best measure down tenfold
    let (mut futile, mut best_at_restart) = (0usize, f64::INFINITY);
    while err > opt.tol && it < opt.max_iter {
        it += 1;
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
        let (f_new, e_new) = eval(&y_new);
        history.push(e_new);
        stalled = if e_new < 0.9 * e_best { 0 } else { stalled + 1 };
        if e_new < e_best {
            e_best = e_new;
            y_best.clone_from(&y_new);
        }
        // diverging, or stuck on a plateau with a history that no longer
        // describes the problem: restart from the best iterate
        if e_new.is_nan() || e_new > 10.0 * e_best || stalled >= patience {
            stalled = 0;
            futile = if e_best > 0.1 * best_at_restart {
                futile + 1
            } else {
                0
            };
            best_at_restart = best_at_restart.min(e_best);
            if opt.newton_fallback && futile >= 2 {
                break; // leave it to Newton-Krylov
            }
            dy.clear();
            df.clear();
            beta *= 0.5;
            y.clone_from(&y_best);
            let (fb, eb) = eval(&y);
            history.push(eb);
            f = fb;
            err = eb;
            continue;
        }
        dy.insert(0, (0..k).map(|q| y_new[q] - y[q]).collect());
        df.insert(0, (0..k).map(|q| f_new[q] - f[q]).collect());
        dy.truncate(opt.history);
        df.truncate(opt.history);
        y = y_new;
        f = f_new;
        err = e_new;
    }
    let state = MixingHistory { dy, df };
    (if e_best < err { y_best } else { y }, history, state)
}

impl<'a> Bop<'a> {
    /// The potentials the occupations imply, `K N` (flat like the shifts).
    fn potential_target(&self, res: &Results) -> Vec<f64> {
        let m = self.model;
        let (ns, nsh) = (m.nspin(), m.basis.len());
        let mut t = vec![0.0; self.shift_len()];
        for i in 0..self.topo.n_atoms {
            let sp = self.topo.species[i];
            for s in 0..ns {
                for k in 0..nsh {
                    let mut v = m.hubbard[sp] * res.charge[i];
                    if !res.coulomb_potential.is_empty() {
                        v += res.coulomb_potential[i];
                    }
                    if ns == 2 {
                        let mk = res.occupations[i][0][k] - res.occupations[i][1][k];
                        v += if s == 0 { -0.5 } else { 0.5 } * m.stoner[sp][k] * mk;
                    }
                    t[(i * ns + s) * nsh + k] = v;
                }
            }
        }
        t
    }

    /// Self-consistent shifts from `shifts0` (default zero) under
    /// `opt.condition`, and the energy, forces and everything else at them.
    ///
    /// For the potential condition `r(x, R) = K N(x, R) - x = 0` the energy is
    /// not stationary in the shifts `x`, and
    /// `dE/dR = dE/dR|_x - u^T dr/dR` with `(dr/dx)^T u = dE/dx`.  The adjoint
    /// system is solved by right-preconditioned GMRES; each application of
    /// `(dr/dx)^T u = chi^T K u - u` is one backward pass seeded with the
    /// occupations (no bond derivatives), and `u^T dr/dR` one more with them.
    pub fn scf(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        shifts0: Option<&[f64]>,
        opt: &ScfOptions,
    ) -> ScfResult {
        self.scf_continued(pos, cell, shifts0, opt, None)
    }

    /// As [`Bop::scf`], starting the Anderson mixing with the history of a
    /// previous solve (`ScfResult::mixing`) of a nearby configuration, as in
    /// molecular dynamics: the steps of the previous solve already describe
    /// the response of the shifts, so the new solve converges in fewer
    /// evaluations.  If that history misleads, the usual safeguard (restart
    /// with an empty history) discards it.
    pub fn scf_continued(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        shifts0: Option<&[f64]>,
        opt: &ScfOptions,
        previous: Option<&MixingHistory>,
    ) -> ScfResult {
        let n_shift = self.shift_len();
        let per_atom = n_shift / self.topo.n_atoms.max(1);
        let mut x = shifts0.map_or_else(|| vec![0.0; n_shift], |s| s.to_vec());
        assert_eq!(x.len(), n_shift, "shifts0 has the wrong length");
        let index: Vec<usize> = match &opt.free {
            None => (0..n_shift).collect(),
            Some(f) => (0..n_shift).filter(|&q| f[q / per_atom]).collect(),
        };
        let all_free = index.len() == n_shift;
        let free = if all_free { None } else { opt.free.as_deref() };
        let k = index.len();
        let stationary = opt.condition == Condition::Stationary;

        let mut evals = 0usize;
        let mut pre: Option<Precond> = None;
        let eval = |y: &[f64]| -> (Vec<f64>, f64) {
            let mut xx = x.clone();
            for (q, &v) in index.iter().zip(y) {
                xx[*q] = v;
            }
            let fresh = pre.is_none() || evals.is_multiple_of(opt.refresh.max(1));
            evals += 1;
            let mode = match (stationary, fresh) {
                (true, true) => Mode::ScfStep,
                (true, false) => Mode::ShiftGradient,
                (false, true) => Mode::Response,
                (false, false) => Mode::Energy,
            };
            let res = self.compute_mode(pos, cell, &xx, mode);
            if fresh {
                pre = Some(Precond::new(self, &res, free));
            }
            let pc = pre.as_ref().unwrap();
            let (full, err) = if stationary {
                let g = flatten(&res.shift_gradient);
                (pc.stationary(&g), max_abs(index.iter().map(|&q| g[q])))
            } else {
                let t = self.potential_target(&res);
                let r: Vec<f64> = t.iter().zip(&xx).map(|(a, b)| a - b).collect();
                let err = max_abs(index.iter().map(|&q| r[q]));
                (pc.potential(&r).unwrap_or_else(|| r.clone()), err)
            };
            let mut f: Vec<f64> = index.iter().map(|&q| full[q]).collect();
            // with every level free a uniform shift leaves the energy
            // unchanged: keep the mean of the shifts where it started
            if stationary && all_free && k > 0 {
                let mean = f.iter().sum::<f64>() / k as f64;
                f.iter_mut().for_each(|v| *v -= mean);
            }
            let fmax = max_abs(f.iter().copied());
            if fmax > opt.max_step {
                let s = opt.max_step / fmax;
                f.iter_mut().for_each(|v| *v *= s);
            }
            (f, err)
        };
        let y0: Vec<f64> = index.iter().map(|&q| x[q]).collect();
        let (mut y, mut history, mixing) = anderson(eval, y0, opt, previous);
        let mut evaluations = evals;
        if opt.newton_fallback && history.iter().copied().fold(f64::INFINITY, f64::min) > opt.tol {
            let (y_nk, h_nk, used) = self.newton_krylov(pos, cell, &x, &index, free, opt, y);
            y = y_nk;
            history.extend(h_nk);
            evaluations += used;
        }
        for (q, &v) in index.iter().zip(&y) {
            x[*q] = v;
        }
        if stationary {
            let results = self.compute_mode(pos, cell, &x, Mode::Full);
            let g = flatten(&results.shift_gradient);
            let residual = max_abs(index.iter().map(|&q| g[q]));
            return ScfResult {
                shifts: x,
                results,
                converged: residual <= opt.tol,
                history,
                residual,
                adjoint_iterations: 0,
                evaluations: evaluations + 1,
                mixing,
            };
        }
        let mut r = self.implicit_forces(pos, cell, x, &index, free, opt, history);
        r.mixing = mixing;
        r.evaluations = evaluations + 1 + r.adjoint_iterations;
        r
    }

    /// The raw residual on the free levels (stationary: dE/dshifts;
    /// potential: K N - shifts), its max norm, and optionally the
    /// preconditioner there.
    #[allow(clippy::too_many_arguments)]
    fn residual_at(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        x: &[f64],
        index: &[usize],
        free: Option<&[bool]>,
        stationary: bool,
        precondition: bool,
    ) -> (Vec<f64>, f64, Option<Precond>) {
        let mode = match (stationary, precondition) {
            (true, true) => Mode::ScfStep,
            (true, false) => Mode::ShiftGradient,
            (false, true) => Mode::Response,
            (false, false) => Mode::Energy,
        };
        let res = self.compute_mode(pos, cell, x, mode);
        let full = if stationary {
            flatten(&res.shift_gradient)
        } else {
            self.potential_target(&res)
                .iter()
                .zip(x)
                .map(|(a, b)| a - b)
                .collect()
        };
        let f: Vec<f64> = index.iter().map(|&q| full[q]).collect();
        let err = max_abs(f.iter().copied());
        (f, err, precondition.then(|| Precond::new(self, &res, free)))
    }

    /// Newton-Krylov steps on the residual from `y0` (free levels): GMRES
    /// for `J step = -F`, right-preconditioned with the model Jacobian of
    /// [`Precond`], Jacobian-vector products by forward differences of the
    /// residual (one evaluation each), and a backtracking line search on
    /// max |F|.  Returns the best iterate, the measures of the Newton
    /// iterates and the number of evaluations.
    #[allow(clippy::too_many_arguments)]
    fn newton_krylov(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        x: &[f64],
        index: &[usize],
        free: Option<&[bool]>,
        opt: &ScfOptions,
        y0: Vec<f64>,
    ) -> (Vec<f64>, Vec<f64>, usize) {
        let stationary = opt.condition == Condition::Stationary;
        let all_free = free.is_none();
        let n_shift = x.len();
        let embed = |y: &[f64]| {
            let mut full = x.to_vec();
            for (q, &v) in index.iter().zip(y) {
                full[*q] = v;
            }
            full
        };
        let embed_zero = |v: &[f64]| {
            let mut full = vec![0.0; n_shift];
            for (q, &a) in index.iter().zip(v) {
                full[*q] = a;
            }
            full
        };
        let restrict = |full: &[f64]| -> Vec<f64> { index.iter().map(|&q| full[q]).collect() };
        let evals = std::cell::Cell::new(0usize);
        let residual = |y: &[f64], pre: bool| {
            evals.set(evals.get() + 1);
            self.residual_at(pos, cell, &embed(y), index, free, stationary, pre)
        };
        let mut y = y0;
        let (mut f, mut err, mut pc) = residual(&y, true);
        let mut history = vec![err];
        for _ in 0..30 {
            if err <= opt.tol {
                break;
            }
            let p = pc.take().unwrap();
            // M^{-1} ~ J^{-1}: stationary J0^{-1}; potential -(1 + K P)^{-1}
            let m_inv = |z: &[f64]| -> Vec<f64> {
                let full = embed_zero(z);
                let mut s = if stationary {
                    restrict(&p.stationary(&full))
                        .iter()
                        .map(|v| -v)
                        .collect::<Vec<f64>>()
                } else {
                    restrict(&p.potential(&full).unwrap_or(full.clone()))
                        .iter()
                        .map(|v| -v)
                        .collect()
                };
                if stationary && all_free && !s.is_empty() {
                    let mean = s.iter().sum::<f64>() / s.len() as f64;
                    s.iter_mut().for_each(|v| *v -= mean);
                }
                s
            };
            let scale = max_abs(y.iter().copied()).max(1.0);
            let jv = |v: &[f64]| -> Vec<f64> {
                let vn = max_abs(v.iter().copied());
                if vn == 0.0 {
                    return vec![0.0; v.len()];
                }
                let eps = 1e-7 * scale / vn;
                let yp: Vec<f64> = y.iter().zip(v).map(|(a, b)| a + eps * b).collect();
                let (fp, _, _) = residual(&yp, false);
                fp.iter().zip(&f).map(|(a, b)| (a - b) / eps).collect()
            };
            let rhs: Vec<f64> = f.iter().map(|v| -v).collect();
            let forcing = err.sqrt().clamp(1e-6, 0.1);
            let z = gmres(|z| jv(&m_inv(z)), &rhs, forcing, 20, 40).unwrap_or_else(|| rhs.clone());
            let step = m_inv(&z);
            let smax = max_abs(step.iter().copied());
            let cap = if smax > opt.max_step {
                opt.max_step / smax
            } else {
                1.0
            };
            let mut t = cap;
            let mut accepted = false;
            for _ in 0..8 {
                let y_new: Vec<f64> = y.iter().zip(&step).map(|(a, b)| a + t * b).collect();
                let (f_new, e_new, pc_new) = residual(&y_new, true);
                if e_new < err {
                    y = y_new;
                    f = f_new;
                    err = e_new;
                    pc = pc_new;
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            history.push(err);
            if !accepted {
                break;
            }
        }
        (y, history, evals.get())
    }

    /// Everything at potential-condition shifts `x`, with the forces and
    /// virial of the implicit function `x(R)`.
    #[allow(clippy::too_many_arguments)]
    fn implicit_forces(
        &self,
        pos: &[[f64; 3]],
        cell: &Cell,
        x: Vec<f64>,
        index: &[usize],
        free: Option<&[bool]>,
        opt: &ScfOptions,
        history: Vec<f64>,
    ) -> ScfResult {
        let n = self.topo.n_atoms;
        let n_shift = self.shift_len();
        let fw = self.forward(pos, cell, &x, true, true);
        let (mut res, p) = self.assemble(&fw, true);
        let mut grad = vec![[0.0; 3]; n];
        let mut virial = [[0.0; 3]; 3];
        self.add_pair_terms(
            pos,
            cell,
            true,
            &mut res.energy_atom,
            &mut grad,
            &mut virial,
        );
        let coulomb = Bop::has_coulomb(&fw);
        if coulomb {
            let (g, w) = self.coulomb_gradient(&fw, pos, cell, &res.charge, &res.charge);
            for i in 0..n {
                for c in 0..3 {
                    grad[i][c] += g[i][c];
                }
            }
            add3(&mut virial, &w);
        }
        res.energy = res.energy_atom.iter().sum();
        let per_atom = n_shift / n.max(1);
        // the full kernel dV/dN: onsite (Hubbard, Stoner) and intersite Coulomb
        let kernel = |pc: &Precond, u_full: &[f64]| -> Vec<f64> {
            let mut v = pc.kernel(u_full);
            if coulomb {
                let per: Vec<f64> = (0..n)
                    .map(|i| u_full[i * per_atom..(i + 1) * per_atom].iter().sum())
                    .collect();
                let inter = self.coulomb_apply(&fw, &per);
                for (q, x) in v.iter_mut().enumerate() {
                    *x += inter[q / per_atom];
                }
            }
            v
        };
        let mut ge = self.seeded_backward(&fw, pos, cell, 1.0, &p, true);
        self.add_explicit_zeeman(&res.moment, &mut ge);
        let (mut spin_field_grad, mut orbital_field_grad) =
            (ge.spin_field.clone(), ge.orbital_field.clone());
        let (sg, g_e, w_e) = (ge.shifts, ge.pos, ge.virial);
        let mut axes_grad = ge.axes;
        for i in 0..n {
            for c in 0..3 {
                grad[i][c] += g_e[i][c];
            }
        }
        add3(&mut virial, &w_e);
        let t = self.potential_target(&res);
        let residual = max_abs(index.iter().map(|&q| t[q] - x[q]));

        // adjoint: (dr/dx)^T u = dE/dx on the free levels
        let pc = Precond::new(self, &res, free);
        let embed = |u: &[f64]| {
            let mut full = vec![0.0; n_shift];
            for (q, &v) in index.iter().zip(u) {
                full[*q] = v;
            }
            full
        };
        let restrict = |full: &[f64]| -> Vec<f64> { index.iter().map(|&q| full[q]).collect() };
        let a_t = |u: &[f64]| -> Vec<f64> {
            let v = kernel(&pc, &embed(u));
            let chi_t = self
                .seeded_backward(&fw, pos, cell, 0.0, &self.unflatten(&v), false)
                .shifts;
            restrict(&chi_t).iter().zip(u).map(|(a, b)| a - b).collect()
        };
        // right preconditioner M^{-1} ~ (dr/dx)^{-T} = -(1 + P K)^{-1}
        let m_inv = |z: &[f64]| -> Vec<f64> {
            restrict(&pc.adjoint(&embed(z)))
                .iter()
                .map(|v| -v)
                .collect()
        };
        let rhs = restrict(&sg);
        let (z, calls) = gmres_count(|z| a_t(&m_inv(z)), &rhs, opt.adjoint_tol, 30, 300);
        let adjoint_ok = z.is_some();
        let u = m_inv(&z.unwrap_or_else(|| vec![0.0; rhs.len()]));
        // u^T dr/dR = d(v.N)/dR with v = K u
        let u_full = embed(&u);
        let v = kernel(&pc, &u_full);
        let gs = self.seeded_backward(&fw, pos, cell, 0.0, &self.unflatten(&v), true);
        let (mut g_s, mut w_s) = (gs.pos.clone(), gs.virial);
        if coulomb {
            // the target V_i = sum_j J_ij(R) q_j also depends on R directly
            let per: Vec<f64> = (0..n)
                .map(|i| u_full[i * per_atom..(i + 1) * per_atom].iter().sum())
                .collect();
            for (l, r) in [(&per, &res.charge), (&res.charge, &per)] {
                let (g, w) = self.coulomb_gradient(&fw, pos, cell, l, r);
                for i in 0..n {
                    for c in 0..3 {
                        g_s[i][c] += g[i][c];
                    }
                }
                add3(&mut w_s, &w);
            }
        }
        for i in 0..n {
            for c in 0..3 {
                grad[i][c] -= g_s[i][c];
            }
        }
        for (a, b) in axes_grad.iter_mut().zip(&gs.axes) {
            for c in 0..3 {
                a[c] -= b[c];
            }
        }
        res.axes_gradient = axes_grad;
        for (a, b) in spin_field_grad.iter_mut().zip(&gs.spin_field) {
            for c in 0..3 {
                a[c] -= b[c];
            }
        }
        for (a, b) in orbital_field_grad.iter_mut().zip(&gs.orbital_field) {
            for c in 0..3 {
                a[c] -= b[c];
            }
        }
        res.spin_field_gradient = spin_field_grad;
        res.orbital_field_gradient = orbital_field_grad;
        for a in 0..3 {
            for b in 0..3 {
                virial[a][b] -= w_s[a][b];
            }
        }
        res.forces = grad.iter().map(|g| [-g[0], -g[1], -g[2]]).collect();
        res.virial = symmetrised(virial);
        res.shift_gradient = self.unflatten(&sg);
        res.moments = fw.mu;
        ScfResult {
            shifts: x,
            results: res,
            converged: residual <= opt.tol && adjoint_ok,
            history,
            residual,
            adjoint_iterations: calls,
            evaluations: 0,
            mixing: MixingHistory::default(),
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
    gmres_count(op, b, tol, restart, max_iter).0
}

/// As [`gmres`], also returning the number of operator applications.
fn gmres_count(
    op: impl Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    tol: f64,
    restart: usize,
    max_iter: usize,
) -> (Option<Vec<f64>>, usize) {
    let calls = std::cell::Cell::new(0usize);
    let counted = |v: &[f64]| {
        calls.set(calls.get() + 1);
        op(v)
    };
    let x = gmres_inner(&counted, b, tol, restart, max_iter);
    (x, calls.get())
}

fn gmres_inner(
    op: &impl Fn(&[f64]) -> Vec<f64>,
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
