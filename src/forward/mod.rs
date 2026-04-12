//! Forward-mode AD types.
//!
//! - [`FReal<T>`](freal::FReal) — single-variable forward mode
//! - [`Dual`](dual::Dual) — multi-variable forward mode
//! - [`Dual2<T>`](dual2::Dual2) — second-order forward mode
//! - [`Dual2Vec`](dual2vec::Dual2Vec) — dense multi-variable second-order forward mode
//!
//! Each module also contains its named wrapper counterpart
//! (`NamedFReal`, `NamedDual`, `NamedDual2`).

pub mod freal;
pub mod dual;
pub mod dual2;
pub mod dual2vec;

pub use freal::{FReal, NamedFReal};
pub use dual::{Dual, NamedDual};
pub use dual2::{Dual2, NamedDual2};
pub use dual2vec::Dual2Vec;
