//! Composite AD operations: Jacobian and Hessian computations.

pub mod jacobian;
pub mod hessian;

pub use jacobian::{compute_jacobian_rev, compute_jacobian_fwd, NamedJacobian, compute_named_jacobian};
pub use hessian::{compute_hessian, NamedHessian, compute_full_hessian};
