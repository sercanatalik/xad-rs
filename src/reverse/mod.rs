//! Reverse-mode (adjoint) AD types.
//!
//! Reverse mode records a *computation graph* (the tape) on the forward
//! pass and walks the adjoint recurrence backwards through it to
//! compute the gradient of a scalar output. Time cost is `O(1)` in the
//! number of inputs and `O(m)` in the number of outputs — the right
//! choice for the typical quant setup of many risk factors and one PV.
//!
//! - [`AReal<T>`](areal::AReal) — reverse-mode active scalar, records on a [`Tape`](crate::tape::Tape)
//!
//! See also (theory): [`docs/theory/03-reverse-mode-and-taped-adjoints.md`](https://github.com/sercanatalik/xad-rs/blob/main/docs/theory/03-reverse-mode-and-taped-adjoints.md).

pub mod areal;

pub use areal::AReal;
