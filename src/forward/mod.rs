//! Forward-mode AD types.
//!
//! Forward mode augments every scalar with a *tangent* and propagates
//! both components together — the dual-number / k-jet construction. It
//! is the cheapest mode in this crate when the input dimension is small
//! and is what you want for a few directional derivatives or a full
//! Hessian on a moderate-sized input vector.
//!
//! - [`Jet1<T>`](jet1::Jet1) — single-variable forward mode, and the storage
//!   scalar of the single-direction forward-over-adjoint Hessian engine
//! - [`Jet2<T>`](jet2::Jet2) — second-order forward mode
//! - [`Jet2Vec`] — dense multi-variable second-order forward mode
//! - [`JetK<T, K>`](jetk::JetK) — K-wide forward mode, the tape storage
//!   scalar of the K-direction forward-over-adjoint Hessian engine
//!
//! See also (theory): [`docs/theory/02-forward-mode-and-dual-numbers.md`](https://github.com/sercanatalik/xad-rs/blob/main/docs/theory/02-forward-mode-and-dual-numbers.md)
//! and [`docs/theory/04-second-order-and-k-jets.md`](https://github.com/sercanatalik/xad-rs/blob/main/docs/theory/04-second-order-and-k-jets.md).

pub mod jet1;
pub mod jet1_passive;
pub mod jet2;
pub mod jet2vec;
pub mod jetk;

pub use jet1::Jet1;
pub use jet2::Jet2;
pub use jet2vec::Jet2Vec;
pub use jetk::JetK;
