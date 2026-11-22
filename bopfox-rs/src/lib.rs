//! Analytic bond-order potentials in Rust.
//!
//! A port of `bopjax` (itself a re-implementation of the BOPfox methodology)
//! in which the derivatives are analytic instead of coming from automatic
//! differentiation: the moments of the Hamiltonian are differentiated by a
//! hand-written backward pass over the recursion walk (`moments.rs`); the
//! small per-bond and per-atom chains use forward-mode dual numbers.

#![allow(clippy::needless_range_loop)] // index loops read like the formulas

pub mod complex;
pub mod energy;
pub mod expansion;
pub mod geometry;
pub mod hamiltonian;
pub mod model;
pub mod moments;
pub mod pairwise;
#[cfg(feature = "python")]
mod python;
pub mod real;
pub mod scf;
pub mod screening;
pub mod spinor;

pub use energy::{Bop, Cell, Mode, Results, G_SPIN, MU_B};
pub use geometry::Topology;
pub use model::Model;
pub use scf::{Condition, MixingHistory, ScfOptions, ScfResult};
