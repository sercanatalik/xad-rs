//! Reverse-mode (adjoint) AD types.
//!
//! - [`AReal<T>`](areal::AReal) — reverse-mode active scalar, records on a [`Tape`](crate::tape::Tape)
//! - [`NamedAReal`](areal::NamedAReal) — named wrapper
//! - [`NamedTape`](areal::NamedTape) — named reverse-mode tape

pub mod areal;

pub use areal::{AReal, NamedAReal, NamedTape};
