//! # `xad-rs` — Automatic Differentiation for Rust
//!
//! Exact, machine-precision derivatives of arbitrary numerical programs —
//! no finite-difference error, no symbolic manipulation.
//!
//! `xad-rs` ships four AD modes in a single crate, each suited to a
//! different problem shape. Every mode also has a **named** variant that
//! lets you read back gradients by variable name (`"spot"`, `"vol"`, ...)
//! instead of positional index.
//!
//! # Choosing a mode
//!
//! | Type | Mode | Order | Use when |
//! |---|---|---|---|
//! | [`FReal<T>`] | Forward | 1st | 1 input direction, many outputs |
//! | [`Dual`] | Forward, multi-var | 1st | full gradient in one pass |
//! | [`Dual2<T>`] | Forward, 2nd-order | 1st + 2nd | diagonal Hessian / gamma |
//! | [`AReal<T>`] + [`Tape`] | Reverse (adjoint) | 1st | many inputs, scalar output |
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
//! let mut x = AReal::new(3.0);
//! let mut y = AReal::new(4.0);
//! AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
//! AReal::register_input(std::slice::from_mut(&mut y), &mut tape);
//!
//! // f(x, y) = x^2 * y + sin(x)
//! let mut f = &(&x * &x) * &y + math::ad::sin(&x);
//! AReal::register_output(std::slice::from_mut(&mut f), &mut tape);
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
//! Seed all inputs in one pass and read the full gradient:
//!
//! ```
//! use xad_rs::Dual;
//!
//! let (x, y) = (Dual::variable(3.0, 0, 2), Dual::variable(4.0, 1, 2));
//! let f = &(&x * &x) * &y;  // x^2 * y
//! assert_eq!(f.partial(0), 24.0);  // df/dx = 2xy
//! assert_eq!(f.partial(1),  9.0);  // df/dy = x^2
//! ```
//!
//! # Named variables
//!
//! Access derivatives by name instead of index — useful in financial
//! models with many risk factors:
//!
//! ```
//! use xad_rs::{NamedForwardTape, NamedForwardScope};
//!
//! let mut ft = NamedForwardTape::new();
//! let spot_h   = ft.declare_dual("spot",   100.0);
//! let strike_h = ft.declare_dual("strike", 105.0);
//! let scope: NamedForwardScope = ft.freeze_dual();
//!
//! let spot   = scope.dual(spot_h);
//! let strike = scope.dual(strike_h);
//! let ratio  = spot / strike;
//!
//! assert!((ratio.partial("spot") - 1.0 / 105.0).abs() < 1e-14);
//! ```
//!
//! # Second-order derivatives
//!
//! ```
//! use xad_rs::Dual2;
//!
//! let x: Dual2<f64> = Dual2::variable(2.0);
//! let y = x * x * x;  // x^3
//! assert_eq!(y.first_derivative(), 12.0);   // 3x^2
//! assert_eq!(y.second_derivative(), 12.0);  // 6x
//! ```
//!
//! # Module overview
//!
//! | Module | Contents |
//! |---|---|
//! | [`forward`] | `FReal`, `Dual`, `Dual2`, `Dual2Vec` + named wrappers |
//! | [`reverse`] | `AReal`, `NamedAReal`, `NamedTape` |
//! | [`math`] | AD-aware transcendentals (`sin`, `exp`, `erf`, `norm_cdf`, ...) |
//! | [`tape`] | Reverse-mode tape and thread-local active-tape slot |
//! | [`ops`] | `compute_jacobian_*`, `compute_hessian`, `compute_full_hessian` |
//! | [`scalar`] | The [`Scalar`] trait bound (`f32`, `f64`) |
//! | [`registry`] | [`VarRegistry`] — ordered name-to-index map |
//! | [`forward_tape`] | [`NamedForwardTape`] / [`NamedForwardScope`] setup |

pub mod scalar;
pub mod tape;
pub mod math;
pub mod registry;
pub mod forward_tape;
pub mod forward;
pub mod reverse;
pub mod ops;

// ---- re-exports: positional types ----
pub use forward::{FReal, Dual, Dual2, Dual2Vec};
pub use reverse::AReal;
pub use tape::{Tape, TapeStorage};
pub use scalar::Scalar;

// ---- re-exports: named types ----
pub use forward::{NamedFReal, NamedDual, NamedDual2};
pub use reverse::{NamedAReal, NamedTape};
pub use registry::VarRegistry;
pub use forward_tape::{NamedForwardTape, NamedForwardScope, DualHandle, Dual2Handle};

// ---- re-exports: composite operations ----
pub use ops::{compute_jacobian_rev, compute_jacobian_fwd, compute_hessian};
pub use ops::{NamedJacobian, compute_named_jacobian};
pub use ops::{NamedHessian, compute_full_hessian};

/// C++ XAD-compatible type aliases for reverse mode (`f64`).
pub mod adj {
    use super::*;
    pub type TapeType = Tape<f64>;
    pub type ActiveType = AReal<f64>;
    pub type PassiveType = f64;
}

/// C++ XAD-compatible type aliases for reverse mode (`f32`).
pub mod adjf {
    use super::*;
    pub type TapeType = Tape<f32>;
    pub type ActiveType = AReal<f32>;
    pub type PassiveType = f32;
}

/// C++ XAD-compatible type aliases for forward mode (`f64`).
pub mod fwd {
    use super::*;
    pub type ActiveType = FReal<f64>;
    pub type PassiveType = f64;
}

/// C++ XAD-compatible type aliases for forward mode (`f32`).
pub mod fwdf {
    use super::*;
    pub type ActiveType = FReal<f32>;
    pub type PassiveType = f32;
}
