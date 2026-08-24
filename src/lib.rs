//! # `xad-rs` — Automatic Differentiation for Rust
//!
//! Exact, machine-precision derivatives of arbitrary numerical programs —
//! no finite-difference error, no symbolic manipulation.
//!
//! `xad-rs` ships four AD modes in a single crate, each suited to a
//! different problem shape.
//!
//! Conceptually, the crate is built around the [`Real`] trait — a unified
//! active-scalar abstraction that lets the same numerical body run
//! against `f64`, forward-mode, or reverse-mode types.
//!
//! See also (long-form theory): [`docs/README.md` on GitHub](https://github.com/sercanatalik/xad-rs/blob/main/docs/README.md).
//!
//! # Choosing what to program against
//!
//! Program your numerical logic once against the trait [`Real`]; pick
//! the concrete mode at the call site that matches your problem shape:
//!
//! ```
//! use xad_rs::prelude::*;
//! // Same body, four call sites — see below.
//! fn quadratic<R: Real>(x: &R) -> R {
//!     x.clone() * x.clone() + R::from(2.0_f64) * x.clone() + R::from(1.0_f64)
//! }
//! ```
//!
//! # Choosing a mode
//!
//! | Type | Mode | Order | Use when |
//! |---|---|---|---|
//! | [`f64`] | none (passive) | 0 | no derivatives needed |
//! | [`Jet1<T>`] | Forward | 1st | 1 input direction, many outputs |
//! | [`Jet2<T>`] | Forward, 2nd-order | 1st + 2nd | diagonal Hessian / gamma |
//! | [`AReal<T>`] + [`Tape`] | Reverse (adjoint) | 1st | many inputs, scalar output |
//! | [`Jet2Vec`] | Forward, dense 2nd-order | 1st + 2nd | full `n × n` Hessian, one pass |
//! | [`JetK<T, K>`] + [`Tape`] | Forward-over-adjoint | 1st + 2nd | full Hessian in `⌈n/K⌉` passes |
//!
//! Reverse mode breaks even with forward around `n ~ 4` inputs. For
//! `n >> 4` (e.g. 30-input swap pricer), reverse is dramatically faster.
//!
//! # Quick start — reverse mode
//!
//! ```
//! use xad_rs::{AReal, Tape, math};
//!
//! let mut tape = Tape::<f64>::new(true);
//! tape.activate();
//!
//! let x = AReal::input(3.0, &mut tape);
//! let y = AReal::input(4.0, &mut tape);
//!
//! // f(x, y) = x^2 * y + sin(x)
//! let mut f = &(&x * &x) * &y + math::ad::sin(&x);
//! f.register(&mut tape);
//! f.set_adjoint(&mut tape, 1.0);
//! tape.compute_adjoints();
//!
//! let dfdx = x.adjoint(&tape);  // 2xy + cos(x)
//! let dfdy = y.adjoint(&tape);  // x^2
//! assert!((dfdx - (2.0 * 3.0 * 4.0 + 3.0_f64.cos())).abs() < 1e-12);
//! assert!((dfdy - 9.0).abs() < 1e-12);
//! # xad_rs::Tape::<f64>::deactivate_all();
//! ```
//!
//! # Quick start — forward mode
//!
//! Seed one input direction and read its derivative, tape-free:
//!
//! ```
//! use xad_rs::Jet1;
//!
//! // f(x, y) = x^2 * y at (3, 4), seeded along x
//! let (x, y) = (Jet1::new(3.0, 1.0), Jet1::constant(4.0));
//! let f = &(&x * &x) * &y;
//! assert_eq!(f.value(), 36.0);
//! assert_eq!(f.derivative(), 24.0);  // df/dx = 2xy
//! ```
//!
//! # Second-order derivatives
//!
//! ```
//! use xad_rs::Jet2;
//!
//! let x: Jet2<f64> = Jet2::variable(2.0);
//! let y = x * x * x;  // x^3
//! assert_eq!(y.first_derivative(), 12.0);   // 3x^2
//! assert_eq!(y.second_derivative(), 12.0);  // 6x
//! ```
//!
//! # Module overview
//!
//! | Module | Contents |
//! |---|---|
//! | [`real`] | The unified active-scalar trait [`Real`] and its copyable refinement [`CopyableReal`] |
//! | [`passive`] | The passive-scalar bound [`Passive`] (`f64`) |
//! | [`prelude`] | `Real`, `CopyableReal`, `Passive`, `AReal`, `Jet1`, `Jet2`, `Tape`, `TapeStorage` |
//! | [`forward`] | `Jet1`, `Jet2`, `Jet2Vec`, `JetK` |
//! | [`reverse`] | `AReal` |
//! | [`math`] | AD-aware transcendentals (`sin`, `exp`, `erf`, `norm_cdf`, ...) |
//! | [`tape`] | Reverse-mode tape and thread-local active-tape slot |
//! | [`ops`] | `compute_derivative_fwd`, `compute_directional_derivative_fwd`, `compute_gradient_rev`, `compute_jacobian_rev`, `compute_hessian{,_k,_k_par}`, `compute_full_hessian` |

// Keep rustdoc links honest: a doc link to a renamed/removed item is a
// compile error, not a silently dead link.
#![deny(rustdoc::broken_intra_doc_links)]

pub(crate) mod elementaries;
pub mod passive;
pub mod real;
pub mod tape;
pub mod math;
pub mod forward;
pub mod reverse;
pub mod ops;
pub mod prelude;

// ---- re-exports: positional types ----
pub use forward::{Jet1, Jet2, Jet2Vec, JetK};
pub use reverse::AReal;
pub use tape::{Tape, TapeGuard, TapeStorage};
pub use passive::Passive;
pub use real::{CopyableReal, Real};


// ---- re-exports: composite operations ----
pub use ops::{
    compute_hessian, compute_hessian_k, compute_hessian_k_par, compute_jacobian_rev,
};
pub use ops::{compute_derivative_fwd, compute_directional_derivative_fwd, compute_gradient_rev};
pub use ops::{DenseHessian, compute_full_hessian};

