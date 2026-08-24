//! Composite AD operations: first-order drivers, Jacobian, and Hessian
//! computations.
//!
//! Higher-level wrappers that orchestrate the modes in [`crate::forward`]
//! and [`crate::reverse`] to produce the canonical derivative objects
//! (`J ∈ R^{m × n}`, `H ∈ R^{n × n}`) without users wiring tapes or
//! seed matrices by hand. Both Hessian entry points are **exact** (no
//! finite-difference error): `compute_full_hessian` runs one dense
//! `Jet2Vec` forward pass; `compute_hessian` runs `n` forward-over-adjoint
//! sweeps (`AReal<Jet1<f64>>`), one per input direction — and
//! `compute_hessian_k` widens the storage scalar to `K` tangent lanes so
//! the same exact result takes `⌈n/K⌉` passes, with `compute_hessian_k_par`
//! distributing those independent blocks across a `rayon` worker pool.
//! All three produce bit-identical results.
//!
//! [`derivative`] adds the two first-order shapes those drivers do not
//! serve — a scalar derivative and a directional derivative in forward mode,
//! and a gradient from one reverse sweep — so callers manage no tape by hand.
//!
//! See also (theory): [`docs/theory/04-second-order-and-k-jets.md`](https://github.com/sercanatalik/xad-rs/blob/main/docs/theory/04-second-order-and-k-jets.md).

pub mod derivative;
pub mod hessian;
pub mod jacobian;

pub use derivative::{
    compute_derivative_fwd, compute_directional_derivative_fwd, compute_gradient_rev,
    compute_gradient_rev_with,
};
pub use hessian::{
    DenseHessian, compute_full_hessian, compute_hessian, compute_hessian_k, compute_hessian_k_par,
};
pub use jacobian::{compute_jacobian_rev, compute_jacobian_rev_with};
