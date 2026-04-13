//! # `Dual2Vec` — dense multi-variable second-order forward-mode AD
//!
//! `Dual2Vec` stores `(value, grad ∈ ℝⁿ, hess ∈ ℝⁿˣⁿ)` and propagates all
//! three through a single forward pass. It is the dense-full-Hessian
//! companion to the single-direction seeded [`Dual2<T>`](crate::forward::dual2::Dual2):
//! where `Dual2<T>` gives you the first- and second-derivative along one
//! seeded direction, `Dual2Vec` gives you the full `n × n` Hessian (plus
//! gradient) for all `n` active inputs at once.
//!
//! Available only with the `dual2-vec` feature enabled.
//!
//! ## Quick example — `f(x, y) = x²y + y³` at `(1, 2)`
//!
//! ```
//! # #[cfg(feature = "dual2-vec")] {
//! use xad_rs::Dual2Vec;
//!
//! // Seed x on axis 0, y on axis 1 of a dim-2 input space
//! let x = Dual2Vec::variable(1.0, 0, 2);
//! let y = Dual2Vec::variable(2.0, 1, 2);
//!
//! // f = x²·y + y³
//! let f = &(&(&x * &x) * &y) + &(&(&y * &y) * &y);
//!
//! // value = 1²·2 + 8 = 10
//! assert_eq!(f.value(), 10.0);
//!
//! // grad = [∂f/∂x, ∂f/∂y] = [2xy, x² + 3y²] = [4, 13]
//! assert_eq!(f.gradient()[0], 4.0);
//! assert_eq!(f.gradient()[1], 13.0);
//!
//! // hess = [[2y, 2x], [2x, 6y]] = [[4, 2], [2, 12]]
//! assert_eq!(f.hessian()[[0, 0]], 4.0);
//! assert_eq!(f.hessian()[[0, 1]], 2.0);
//! assert_eq!(f.hessian()[[1, 0]], 2.0);
//! assert_eq!(f.hessian()[[1, 1]], 12.0);
//! # }
//! ```
//!
//! ## Cost model
//!
//! Per-op cost is `O(n²)` because the Hessian storage is a dense `n × n`
//! matrix and every binary op touches every Hessian entry (via an
//! upper-triangle loop plus the [`mirror_upper_triangle`] helper). Unary
//! elementaries are also `O(n²)` elementwise on the Hessian: the
//! multi-variable chain rule is
//!
//! ```text
//! grad'[i]   = f'(v) · grad[i]
//! hess'[i,j] = f''(v) · grad[i] · grad[j] + f'(v) · hess[i,j]
//! ```
//!
//! which is a rank-1 outer-product update plus a scaled `hess` copy.
//!
//! ## When to use `Dual2Vec` vs seeded `Dual2<T>`
//!
//! `Dual2Vec` and seeded `Dual2<T>` are complementary, not competitors. The
//! **crossover** between them is around `n ≈ 50–100`:
//!
//! | Situation                                       | Prefer              |
//! |--------------------------------------------------|---------------------|
//! | Full `n × n` Hessian, `n ≲ 50`                   | `Dual2Vec`          |
//! | Own-gamma (Hessian diagonal) only                | seeded `Dual2<T>`   |
//! | Single-direction second derivative               | seeded `Dual2<T>`   |
//! | Full Hessian, `n ≳ 100`                          | seeded `Dual2<T>` × n passes |
//! | Between `n ≈ 50` and `n ≈ 100`                   | benchmark both — depends on op mix |
//!
//! Below `n ≈ 50` `Dual2Vec` typically wins on one-shot convenience: one
//! forward pass hands you the full Hessian, and the `O(n²)` per-op cost
//! is amortized over the smaller `n`. Above `n ≈ 100` seeded `Dual2<T>`
//! with per-direction passes wins on total cost because each pass is
//! `O(n)` per op and the direction loop can share sub-expressions.
//!
//! ## Hessian symmetry
//!
//! Hessian symmetry is **structural**: every binary op computes the upper
//! triangle and mirrors it to the lower triangle via the `pub(crate)`
//! helper [`mirror_upper_triangle`]. Tests assert `assert_eq!(h, h.t())`
//! bit-exactly BEFORE ever calling [`Dual2Vec::symmetrize`]. The
//! `symmetrize` helper exists only as a user-facing safety valve for
//! downstream code that accumulates cross-module Hessians outside the
//! `Dual2Vec` hot path — internal code never needs it.
//!
//! ## `Div` is a direct closed form — NOT `Mul ∘ Recip`
//!
//! The `Div` impl below is written as a **direct closed form**. There is
//! a mandatory `// DO NOT derive Div as Mul∘Recip — precision loss at
//! b ≈ 1 with large numerator` comment guarding the `impl Div` block.
//!
//! Deriving `a / b` as `a * (1/b)` introduces a `1/b` intermediate whose
//! second-order chain rule loses precision as `b → 1` with large `|a|`
//! or large `|a''|`: the composed form stacks a `(1/b)` reciprocal onto
//! the numerator's grad and hess, and the resulting cancellation burns
//! roughly half the f64 mantissa bits on the primary cross-partial. The
//! direct form divides exactly once at the end (`inv_b = 1.0 / b.value`
//! applied to numerator difference terms), keeping the numerator's
//! precision intact.
//!
//! Similarly, `powf(x, k)` for a constant power `k` is a **direct**
//! chain rule (`k·uᵏ⁻¹` / `k·(k-1)·uᵏ⁻²`) rather than `exp(k·ln(x))`:
//! the composed form loses precision as `x → 0` because `ln(x) → -∞`
//! and the subsequent `exp` amplifies the error. `powd(x, y)` for an
//! *active* exponent `y` routes through `exp(y·ln(x))` because there is
//! no closed-form shortcut when both sides depend on active inputs.
//!
//! ## Elementary surface
//!
//! `Dual2Vec` supports **10 unary elementaries** stamped via the
//! `pub(crate) impl_unary_dual2vec!` macro from `(f, f', f'')` closure
//! triples:
//!
//! `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `tanh`, `atan`, `asinh`, `erf`
//!
//! The `erf` *value* goes through the Abramowitz-Stegun 7.1.26 polynomial
//! approximation (~1.5e-7 absolute error, same as `crate::math::erf`),
//! but its analytical derivatives `(2/√π)·exp(-u²)` and `-2u·g'(u)` use
//! pure `f64::exp` and are therefore f64-exact — Hessian cells flowing
//! through `erf` enjoy full f64 precision.
//!
//! ## Correctness proof
//!
//! Correctness evidence is two-tiered:
//!
//! - **Analytical cross-check suite** at tolerance `1e-13`
//!   (`tests/dual2vec_analytical.rs`): four hand-derived
//!   literal Hessians on `x²y + y³`, `sin(xy)`, `exp(x² + y²)`, and
//!   `(x - y)²·log(x + y)`. Every test asserts `hess == hess.t()`
//!   bit-exactly BEFORE any element comparison.
//! - **Deployment-scale Garman–Kohlhagen 6×6 Hessian two-tier check**
//!   at tolerance `1e-11` primary / `5e-5` secondary
//!   (`tests/dual2vec_finance.rs`): primary asserts three
//!   closed-form analytical Greeks (gamma = `H[0,0]`, volga = `H[4,4]`,
//!   vanna = `H[0,4]`) at 1e-11 against `exp(-rf·T)·φ(d1)/(S·σ·√T)` and
//!   friends; secondary asserts all 36 entries at 5e-5 against
//!   `xad_rs::hessian::compute_hessian` (finite-difference Hessian via
//!   reverse-mode gradient, `eps = 1e-7`). The 5e-5 bound is the honest
//!   FD truncation budget at the GK test point — see the
//!   `tests/dual2vec_finance.rs` module docs for the full derivation.
//!
//! ## No lifetime parameters
//!
//! `Dual2Vec` carries **zero lifetime parameters** — all fields are owned.
//! This preserves PyO3 binding compatibility (PyO3 cannot express Rust
//! lifetimes across the FFI boundary cleanly).

use ndarray::{Array1, Array2};
use std::ops::{Add, Div, Mul, Sub};

/// Dense multi-variable second-order forward-mode AD number.
///
/// See module docs for cost model and symmetry guarantees.
#[derive(Clone, Debug, PartialEq)]
pub struct Dual2Vec {
    /// Function value `f(x_1, ..., x_n)`
    pub value: f64,
    /// Gradient `[∂f/∂x_1, ..., ∂f/∂x_n]`
    pub grad: Array1<f64>,
    /// Hessian `H[i,j] = ∂²f / (∂x_i ∂x_j)`, symmetric `n × n`
    pub hess: Array2<f64>,
}

impl Dual2Vec {
    /// Create the `i`-th active variable in an `n`-dimensional input space.
    ///
    /// The resulting `Dual2Vec` has `value = value`, `grad = e_i` (unit
    /// vector in direction `i`), and `hess = 0`.
    #[inline]
    pub fn variable(value: f64, i: usize, n: usize) -> Self {
        let mut grad = Array1::<f64>::zeros(n);
        grad[i] = 1.0;
        let hess = Array2::<f64>::zeros((n, n));
        Self { value, grad, hess }
    }

    /// Create a derivative-free `Dual2Vec` (zero grad, zero hess) with
    /// the given `value` and input-space dimension `n`.
    #[inline]
    pub fn constant(value: f64, n: usize) -> Self {
        Self {
            value,
            grad: Array1::<f64>::zeros(n),
            hess: Array2::<f64>::zeros((n, n)),
        }
    }

    /// Value accessor.
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Gradient as a shared reference (no allocation).
    #[inline]
    pub fn gradient(&self) -> &Array1<f64> {
        &self.grad
    }

    /// Hessian as a shared reference (no allocation).
    #[inline]
    pub fn hessian(&self) -> &Array2<f64> {
        &self.hess
    }

    /// Input-space dimension `n` — the length of `grad` and the side of `hess`.
    #[inline]
    pub fn len(&self) -> usize {
        self.grad.len()
    }

    /// Returns `true` if this `Dual2Vec` has zero input-space dimension.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.grad.is_empty()
    }

    /// User-facing Hessian symmetrization helper — enforces `H ← 0.5·(H + Hᵀ)`.
    ///
    /// **Internal invariant:** `Dual2Vec` ops never produce asymmetric
    /// Hessians — every binary op computes the upper triangle and mirrors
    /// it via [`mirror_upper_triangle`], and every test asserts
    /// `hess == hess.t()` bit-exactly BEFORE calling this helper. The
    /// helper exists only as a user-facing safety valve for downstream
    /// code that accumulates cross-module Hessians outside the `Dual2Vec`
    /// hot path.
    pub fn symmetrize(&mut self) {
        let n = self.hess.nrows();
        debug_assert_eq!(
            n,
            self.hess.ncols(),
            "Dual2Vec::symmetrize on non-square hess"
        );
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (self.hess[[i, j]] + self.hess[[j, i]]);
                self.hess[[i, j]] = avg;
                self.hess[[j, i]] = avg;
            }
        }
    }
}

/// Mirror the upper triangle of a square matrix into the lower triangle.
///
/// Used by every binary op in `src/dual2vec.rs` to produce a structurally
/// symmetric Hessian from an upper-triangle computation. Off-diagonal
/// entries `H[j, i]` for `i < j` are overwritten with `H[i, j]`; the
/// diagonal is untouched.
///
/// `#[allow(dead_code)]`: Add/Sub produce structurally symmetric Hessians by
/// linearity alone (no mirror needed). Used by `Mul`/`Div` where the
/// upper-triangle computation is non-trivial.
#[allow(dead_code)]
#[inline]
pub(crate) fn mirror_upper_triangle(h: &mut Array2<f64>) {
    let n = h.nrows();
    debug_assert_eq!(n, h.ncols(), "mirror_upper_triangle on non-square matrix");
    for i in 0..n {
        for j in (i + 1)..n {
            h[[j, i]] = h[[i, j]];
        }
    }
}

// ============================================================================
// Add / Sub — linear propagation of value, grad, and hess
// ============================================================================
//
// Add/Sub do not produce any cross-term contribution — they are linear in all
// three components. Structural symmetry still holds trivially because
// `hess_a + hess_b` is symmetric whenever both inputs are. No call to
// `mirror_upper_triangle` is needed, but binary-op tests still assert
// `hess == hess.t()` bit-exactly to exercise the symmetry machinery.

#[inline]
fn shape_check(a: &Dual2Vec, b: &Dual2Vec) {
    debug_assert_eq!(
        a.grad.len(),
        b.grad.len(),
        "Dual2Vec gradient length mismatch: {} vs {}",
        a.grad.len(),
        b.grad.len()
    );
    debug_assert_eq!(
        a.hess.shape(),
        b.hess.shape(),
        "Dual2Vec hessian shape mismatch"
    );
}

impl Add<&Dual2Vec> for &Dual2Vec {
    type Output = Dual2Vec;
    #[inline]
    fn add(self, rhs: &Dual2Vec) -> Dual2Vec {
        shape_check(self, rhs);
        Dual2Vec {
            value: self.value + rhs.value,
            grad: &self.grad + &rhs.grad,
            hess: &self.hess + &rhs.hess,
        }
    }
}

impl Add for Dual2Vec {
    type Output = Dual2Vec;
    #[inline]
    fn add(self, rhs: Dual2Vec) -> Dual2Vec {
        (&self).add(&rhs)
    }
}

impl Sub<&Dual2Vec> for &Dual2Vec {
    type Output = Dual2Vec;
    #[inline]
    fn sub(self, rhs: &Dual2Vec) -> Dual2Vec {
        shape_check(self, rhs);
        Dual2Vec {
            value: self.value - rhs.value,
            grad: &self.grad - &rhs.grad,
            hess: &self.hess - &rhs.hess,
        }
    }
}

impl Sub for Dual2Vec {
    type Output = Dual2Vec;
    #[inline]
    fn sub(self, rhs: Dual2Vec) -> Dual2Vec {
        (&self).sub(&rhs)
    }
}

// ============================================================================
// Mul — second-order product rule with symmetric outer-product cross term
// ============================================================================
//
// For f = a·b with `a, b: Dual2Vec`:
//   f.value = a.value · b.value
//   f.grad  = a.value · b.grad + b.value · a.grad
//   f.hess  = a.value · b.hess + b.value · a.hess
//           + (a.grad ⊗ b.grad + b.grad ⊗ a.grad)   ← symmetric cross term
//
// The cross term `a.grad ⊗ b.grad + b.grad ⊗ a.grad` is the only part
// that isn't purely linear in the inputs — on f = x·y with
// `x = variable(_, 0, 2)` and `y = variable(_, 1, 2)`
// the outer products evaluate to
//   [[0, 1], [0, 0]] + [[0, 0], [1, 0]] = [[0, 1], [1, 0]]
// which gives the expected `H[0, 1] = H[1, 0] = 1`.
//
// Structural symmetry is preserved by computing the upper triangle only
// (i ≤ j) and mirroring via `mirror_upper_triangle`.

impl Mul<&Dual2Vec> for &Dual2Vec {
    type Output = Dual2Vec;
    #[inline]
    fn mul(self, rhs: &Dual2Vec) -> Dual2Vec {
        shape_check(self, rhs);
        let a_val = self.value;
        let b_val = rhs.value;
        let n = self.grad.len();

        // value
        let value = a_val * b_val;

        // grad: a.value · b.grad + b.value · a.grad
        let grad = a_val * &rhs.grad + b_val * &self.grad;

        // hess upper triangle only: H[i,j] for i ≤ j
        //   a_val · rhs.hess[i,j] + b_val · self.hess[i,j]
        //   + self.grad[i] * rhs.grad[j] + rhs.grad[i] * self.grad[j]
        //
        // Note: when i == j the cross becomes 2 · self.grad[i] · rhs.grad[i],
        // which is the correct diagonal of `grad_a ⊗ grad_b + grad_b ⊗ grad_a`.
        let mut hess = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let cross = self.grad[i] * rhs.grad[j] + rhs.grad[i] * self.grad[j];
                hess[[i, j]] = a_val * rhs.hess[[i, j]] + b_val * self.hess[[i, j]] + cross;
            }
        }
        mirror_upper_triangle(&mut hess);

        Dual2Vec { value, grad, hess }
    }
}

impl Mul for Dual2Vec {
    type Output = Dual2Vec;
    #[inline]
    fn mul(self, rhs: Dual2Vec) -> Dual2Vec {
        (&self).mul(&rhs)
    }
}

// ============================================================================
// Div — direct closed form (NOT Mul ∘ Recip)
// ============================================================================
//
// DO NOT derive Div as Mul∘Recip — precision loss at b ≈ 1 with large numerator
//
// Deriving `a / b` via `a * (1/b)` introduces a `1/b` intermediate that
// loses precision as `b → 1` with large `|a|` or large `|a''|`. The direct
// closed form keeps the numerator intact and divides exactly once at the
// end, preserving the full precision of `a`.
//
// Closed form for f = a / b:
//   v      = a.value / b.value
//   v'     = (a.grad - v · b.grad) / b.value           ← first-order
//   v''    = (a.hess - v · b.hess
//             - (v' ⊗ b.grad + b.grad ⊗ v')) / b.value  ← second-order
//
// The `(v' ⊗ b.grad + b.grad ⊗ v')` term is the symmetric cross between
// the already-computed gradient of the quotient and b's gradient. It
// plays the same structural role as Mul's `(a.grad ⊗ b.grad + …)` term
// and is mirrored from the upper triangle via `mirror_upper_triangle`
// for structural symmetry.

impl Div<&Dual2Vec> for &Dual2Vec {
    type Output = Dual2Vec;
    #[inline]
    fn div(self, rhs: &Dual2Vec) -> Dual2Vec {
        shape_check(self, rhs);
        let b_val = rhs.value;
        debug_assert!(b_val != 0.0, "Dual2Vec::div by zero value");
        let inv_b = 1.0 / b_val;
        let n = self.grad.len();

        // value
        let value = self.value * inv_b;

        // grad: (a.grad - value · b.grad) / b.value
        let grad = (&self.grad - value * &rhs.grad) * inv_b;

        // hess upper triangle only: H[i, j] for i ≤ j
        //   (a.hess[i,j] - value · b.hess[i,j]
        //    - grad[i] · b.grad[j] - b.grad[i] · grad[j]) / b.value
        let mut hess = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let cross = grad[i] * rhs.grad[j] + rhs.grad[i] * grad[j];
                hess[[i, j]] = (self.hess[[i, j]] - value * rhs.hess[[i, j]] - cross) * inv_b;
            }
        }
        mirror_upper_triangle(&mut hess);

        Dual2Vec { value, grad, hess }
    }
}

impl Div for Dual2Vec {
    type Output = Dual2Vec;
    #[inline]
    fn div(self, rhs: Dual2Vec) -> Dual2Vec {
        (&self).div(&rhs)
    }
}

// ============================================================================
// Unary elementaries — stamped via `impl_unary_dual2vec!` macro from
// `(f, f', f'')` triples.
// ============================================================================
//
// Multi-variable chain rule for a unary function `g(u)` applied to
// `u: Dual2Vec`:
//
//   out.value      = g(u.value)
//   out.grad[i]    = g'(u.value) · u.grad[i]
//   out.hess[i, j] = g''(u.value) · u.grad[i] · u.grad[j]
//                  + g'(u.value)  · u.hess[i, j]
//
// The `u.grad[i] · u.grad[j]` outer product is naturally symmetric (scalar
// multiplication commutes). Inherited `u.hess[i, j]` is symmetric by
// invariant. So the resulting Hessian is structurally symmetric with no
// explicit mirror needed — but we still call `mirror_upper_triangle` at the
// end for consistency with the Mul/Div pattern and to keep the structural-
// symmetry guarantee local to every op.

/// Stamp a unary elementary from an `(f, f', f'')` triple.
///
/// - `$name`: the method name on `Dual2Vec` (e.g. `sin`, `cos`).
/// - `$f`:   `|u: f64| -> f64` computing `f(u)`.
/// - `$fp`:  `|u: f64| -> f64` computing `f'(u)`.
/// - `$fpp`: `|u: f64| -> f64` computing `f''(u)`.
///
/// The macro is `pub(crate)` and NOT exported.
macro_rules! impl_unary_dual2vec {
    ($name:ident, $f:expr, $fp:expr, $fpp:expr) => {
        impl Dual2Vec {
            #[doc = concat!("Applies `", stringify!($name), "` elementwise via the multi-variable chain rule.")]
            #[doc = ""]
            #[doc = "Propagates value, gradient, and Hessian in one forward pass. See"]
            #[doc = "`src/dual2vec.rs` module docs for the chain-rule derivation."]
            #[inline]
            pub fn $name(self) -> Self {
                let u = self.value;
                let fv = ($f)(u);
                let fp = ($fp)(u);
                let fpp = ($fpp)(u);
                let n = self.grad.len();

                // grad'[i] = f'(u) · self.grad[i]
                let grad = fp * &self.grad;

                // hess'[i, j] = f''(u) · self.grad[i] · self.grad[j]
                //             + f'(u)  · self.hess[i, j]
                // Upper triangle only, then mirror.
                let mut hess = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    for j in i..n {
                        hess[[i, j]] =
                            fpp * self.grad[i] * self.grad[j] + fp * self.hess[[i, j]];
                    }
                }
                mirror_upper_triangle(&mut hess);

                Dual2Vec {
                    value: fv,
                    grad,
                    hess,
                }
            }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use impl_unary_dual2vec;

// --- sin: g(u) = sin u, g'(u) = cos u, g''(u) = -sin u ---
impl_unary_dual2vec!(sin, |u: f64| u.sin(), |u: f64| u.cos(), |u: f64| -u.sin());

// --- cos: g(u) = cos u, g'(u) = -sin u, g''(u) = -cos u ---
impl_unary_dual2vec!(cos, |u: f64| u.cos(), |u: f64| -u.sin(), |u: f64| -u.cos());

// --- tan: g(u) = tan u, g'(u) = sec² u = 1 + tan² u, g''(u) = 2·tan u · (1 + tan² u) ---
impl_unary_dual2vec!(
    tan,
    |u: f64| u.tan(),
    |u: f64| {
        let t = u.tan();
        1.0 + t * t
    },
    |u: f64| {
        let t = u.tan();
        2.0 * t * (1.0 + t * t)
    }
);

// --- exp: g(u) = g'(u) = g''(u) = exp u ---
impl_unary_dual2vec!(exp, |u: f64| u.exp(), |u: f64| u.exp(), |u: f64| u.exp());

// --- ln: g(u) = ln u, g'(u) = 1/u, g''(u) = -1/u² ---
impl_unary_dual2vec!(ln, |u: f64| u.ln(), |u: f64| 1.0 / u, |u: f64| -1.0
    / (u * u));

// --- sqrt: g(u) = √u, g'(u) = 1/(2√u), g''(u) = -1/(4·u^{3/2}) ---
impl_unary_dual2vec!(
    sqrt,
    |u: f64| u.sqrt(),
    |u: f64| 0.5 / u.sqrt(),
    |u: f64| -0.25 / (u * u.sqrt())
);

// --- tanh: g(u) = tanh u, g'(u) = sech² u = 1 - tanh² u, g''(u) = -2·tanh u · sech² u ---
impl_unary_dual2vec!(
    tanh,
    |u: f64| u.tanh(),
    |u: f64| {
        let t = u.tanh();
        1.0 - t * t
    },
    |u: f64| {
        let t = u.tanh();
        -2.0 * t * (1.0 - t * t)
    }
);

// --- atan: g(u) = atan u, g'(u) = 1/(1 + u²), g''(u) = -2u / (1 + u²)² ---
impl_unary_dual2vec!(
    atan,
    |u: f64| u.atan(),
    |u: f64| 1.0 / (1.0 + u * u),
    |u: f64| {
        let d = 1.0 + u * u;
        -2.0 * u / (d * d)
    }
);

// --- asinh: g(u) = asinh u, g'(u) = 1/√(1 + u²), g''(u) = -u / (1 + u²)^{3/2} ---
impl_unary_dual2vec!(
    asinh,
    |u: f64| u.asinh(),
    |u: f64| 1.0 / (1.0 + u * u).sqrt(),
    |u: f64| {
        let d = 1.0 + u * u;
        -u / (d * d.sqrt())
    }
);

// --- erf: g(u) = erf u, g'(u) = (2/√π)·exp(-u²), g''(u) = -2u·g'(u) ---
//
// The closed-form reference `N(x) = 0.5·(1 + erf(x/√2))` for the standard
// normal CDF needs `erf` as a Dual2Vec elementary so the chain-rule
// derivatives are exact rather than fighting an Abramowitz-Stegun rational-
// approximation error floor at 1e-11 tolerance.
//
// Note: the *value* computed by `crate::math::erf` is the A&S 7.1.26
// polynomial approximation (~1.5e-7 absolute error), but the *derivatives*
// `(2/√π)·exp(-u²)` and `-2u·(2/√π)·exp(-u²)` propagated by this macro use
// only `f64::exp` and exact arithmetic, so the Hessian elements flowing from
// `erf` enjoy full f64 precision.
impl_unary_dual2vec!(
    erf,
    |u: f64| crate::math::erf(u),
    |u: f64| std::f64::consts::FRAC_2_SQRT_PI * (-u * u).exp(),
    |u: f64| {
        let fp = std::f64::consts::FRAC_2_SQRT_PI * (-u * u).exp();
        -2.0 * u * fp
    }
);

// --- norm_cdf: Φ(x) = 0.5·(1 + erf(x/√2)) ---
// g'(x) = φ(x) = (1/√(2π))·exp(-x²/2)
// g''(x) = -x·φ(x)
impl_unary_dual2vec!(
    norm_cdf,
    |u: f64| crate::math::norm_cdf(u),
    |u: f64| crate::math::norm_pdf(u),
    |u: f64| -u * crate::math::norm_pdf(u)
);

// --- inv_norm_cdf: Φ⁻¹(p) ---
// g'(p) = 1 / φ(Φ⁻¹(p))
// g''(p) = Φ⁻¹(p) · g'(p)²
impl_unary_dual2vec!(
    inv_norm_cdf,
    |u: f64| crate::math::inv_norm_cdf(u),
    |u: f64| {
        let r = crate::math::inv_norm_cdf(u);
        1.0 / crate::math::norm_pdf(r)
    },
    |u: f64| {
        let r = crate::math::inv_norm_cdf(u);
        let gp = 1.0 / crate::math::norm_pdf(r);
        r * gp * gp
    }
);

// ============================================================================
// Power — direct-form `powf(k: f64)` and `powd(y: Dual2Vec)` via exp·ln
// ============================================================================
//
// `powf(k)` uses the direct chain-rule closed form `g(u) = u^k` with
// `g'(u) = k·u^{k-1}`, `g''(u) = k·(k-1)·u^{k-2}`. It is NOT
// `exp(k·ln(u))` — the exp·ln form loses precision as `u → 0` where
// `ln(u) → -∞`.
//
// `powd(y)` (both base and exponent are active) routes through
// `exp(y · ln(self))` — this is the correct formulation when
// the exponent itself depends on active inputs, and it composes the
// already-implemented ln / Mul / exp ops.

impl Dual2Vec {
    /// Raise `self` to a **constant** scalar power `k: f64` via the direct
    /// chain-rule closed form.
    ///
    /// The direct form avoids the precision loss of `exp(k·ln(u))` as
    /// `u → 0` (where `ln(u) → -∞`).
    ///
    /// Formula (single-variable chain rule `g(u) = u^k` lifted to multi-
    /// variable):
    ///
    /// ```text
    /// g'(u)  = k · u^{k-1}
    /// g''(u) = k · (k - 1) · u^{k-2}
    /// ```
    ///
    /// Multi-variable propagation is then identical in shape to the
    /// `impl_unary_dual2vec!`-stamped elementaries, but with `k` as a
    /// runtime parameter (so it can't be stamped from the macro).
    #[inline]
    pub fn powf(self, k: f64) -> Self {
        let u = self.value;
        let fv = u.powf(k);
        let fp = k * u.powf(k - 1.0);
        let fpp = k * (k - 1.0) * u.powf(k - 2.0);
        let n = self.grad.len();

        let grad = fp * &self.grad;

        let mut hess = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                hess[[i, j]] = fpp * self.grad[i] * self.grad[j] + fp * self.hess[[i, j]];
            }
        }
        mirror_upper_triangle(&mut hess);

        Dual2Vec {
            value: fv,
            grad,
            hess,
        }
    }

    /// Raise `self` to a `Dual2Vec` power — both base and exponent are
    /// active. Routes through `exp(y · ln(self))`.
    ///
    /// Slower than [`Dual2Vec::powf`] because it chains three nonlinear
    /// ops (`ln`, `*`, `exp`), but this is the correct formulation when
    /// the exponent itself depends on active inputs.
    #[inline]
    pub fn powd(self, y: Dual2Vec) -> Self {
        // f = x^y = exp(y · ln(x))
        let ln_x = self.ln();
        let prod = &y * &ln_x;
        prod.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_variable_constructor() {
        let x = Dual2Vec::variable(3.0, 0, 2);
        assert_eq!(x.value(), 3.0);
        assert_eq!(x.gradient(), &array![1.0, 0.0]);
        assert_eq!(x.hessian(), &Array2::<f64>::zeros((2, 2)));
    }

    #[test]
    fn test_variable_constructor_index_1() {
        let y = Dual2Vec::variable(5.0, 1, 3);
        assert_eq!(y.value(), 5.0);
        assert_eq!(y.gradient(), &array![0.0, 1.0, 0.0]);
        assert_eq!(y.hessian(), &Array2::<f64>::zeros((3, 3)));
    }

    #[test]
    fn test_constant_constructor() {
        let c = Dual2Vec::constant(7.5, 4);
        assert_eq!(c.value(), 7.5);
        assert_eq!(c.gradient(), &Array1::<f64>::zeros(4));
        assert_eq!(c.hessian(), &Array2::<f64>::zeros((4, 4)));
        assert_eq!(c.len(), 4);
        assert!(!c.is_empty());
    }

    #[test]
    fn test_mirror_upper_triangle_3x3() {
        let mut h = Array2::<f64>::zeros((3, 3));
        h[[0, 0]] = 1.0;
        h[[1, 1]] = 2.0;
        h[[2, 2]] = 3.0;
        h[[0, 1]] = 4.0;
        h[[0, 2]] = 5.0;
        h[[1, 2]] = 6.0;
        mirror_upper_triangle(&mut h);
        assert_eq!(h[[1, 0]], 4.0);
        assert_eq!(h[[2, 0]], 5.0);
        assert_eq!(h[[2, 1]], 6.0);
        // Diagonal untouched
        assert_eq!(h[[0, 0]], 1.0);
        assert_eq!(h[[1, 1]], 2.0);
        assert_eq!(h[[2, 2]], 3.0);
        // Bit-exact symmetry after mirror
        assert_eq!(h, h.t());
    }

    #[test]
    fn test_symmetrize_averages_halves() {
        let mut v = Dual2Vec::constant(0.0, 2);
        v.hess[[0, 0]] = 1.0;
        v.hess[[1, 1]] = 1.0;
        v.hess[[0, 1]] = 2.0;
        v.hess[[1, 0]] = 0.0;
        v.symmetrize();
        assert_eq!(v.hess[[0, 1]], 1.0);
        assert_eq!(v.hess[[1, 0]], 1.0);
        assert_eq!(v.hess, v.hess.t());
    }

    #[test]
    fn test_add_by_value() {
        let x = Dual2Vec::variable(3.0, 0, 2);
        let y = Dual2Vec::variable(4.0, 1, 2);
        let s = x + y;
        assert_eq!(s.value(), 7.0);
        assert_eq!(s.gradient(), &array![1.0, 1.0]);
        assert_eq!(s.hessian(), &Array2::<f64>::zeros((2, 2)));
        assert_eq!(s.hess, s.hess.t()); // bit-exact symmetry
    }

    #[test]
    fn test_sub_by_value() {
        let x = Dual2Vec::variable(5.0, 0, 2);
        let y = Dual2Vec::variable(2.0, 1, 2);
        let d = x - y;
        assert_eq!(d.value(), 3.0);
        assert_eq!(d.gradient(), &array![1.0, -1.0]);
        assert_eq!(d.hessian(), &Array2::<f64>::zeros((2, 2)));
        assert_eq!(d.hess, d.hess.t());
    }

    #[test]
    fn test_add_by_ref_matches_by_value() {
        let x = Dual2Vec::variable(3.0, 0, 2);
        let y = Dual2Vec::variable(4.0, 1, 2);
        let s_ref = &x + &y;
        let s_val = x.clone() + y.clone();
        assert_eq!(s_ref, s_val);
    }

    #[test]
    fn test_sub_by_ref_matches_by_value() {
        let x = Dual2Vec::variable(5.0, 0, 2);
        let y = Dual2Vec::variable(2.0, 1, 2);
        let d_ref = &x - &y;
        let d_val = x.clone() - y.clone();
        assert_eq!(d_ref, d_val);
    }

    // Debug-only test: `shape_check` uses `debug_assert_eq!` which is compiled
    // out in release mode, so ndarray's internal incompatible-shape panic fires
    // with a different (ndarray-owned) message. The test is gated to debug
    // builds so release-mode `cargo test --release` stays green.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "gradient length mismatch")]
    fn test_add_len_mismatch_panics() {
        let a = Dual2Vec::variable(1.0, 0, 2);
        let b = Dual2Vec::variable(1.0, 0, 3);
        let _ = &a + &b;
    }

    #[test]
    fn test_mul_cross_term_symmetric() {
        let x = Dual2Vec::variable(3.0, 0, 2);
        let y = Dual2Vec::variable(4.0, 1, 2);
        let f = &x * &y; // f = x·y at (3, 4) → value = 12
        assert_eq!(f.value(), 12.0);
        assert_eq!(f.gradient()[0], 4.0); // ∂(xy)/∂x = y
        assert_eq!(f.gradient()[1], 3.0); // ∂(xy)/∂y = x
        assert_eq!(f.hessian()[[0, 0]], 0.0);
        assert_eq!(f.hessian()[[1, 1]], 0.0);
        assert_eq!(f.hessian()[[0, 1]], 1.0); // ∂²(xy)/∂x∂y = 1
        assert_eq!(f.hessian()[[1, 0]], 1.0); // symmetric
    }

    #[test]
    fn test_mul_square_diagonal() {
        // f = x² at (5, _): value = 25, grad = [10, 0], hess[0,0] = 2 (d²/dx²(x²) = 2)
        let x = Dual2Vec::variable(5.0, 0, 2);
        let f = &x * &x;
        assert_eq!(f.value(), 25.0);
        assert_eq!(f.gradient(), &array![10.0, 0.0]);
        // Bit-exact structural symmetry BEFORE any Hessian element assertion
        assert_eq!(f.hess, f.hess.t());
        assert_eq!(f.hessian()[[0, 0]], 2.0);
        assert_eq!(f.hessian()[[1, 1]], 0.0);
        assert_eq!(f.hessian()[[0, 1]], 0.0);
    }

    #[test]
    fn test_mul_symmetry_bit_exact_on_3_vars() {
        // 3-variable mul chain: (x + y) * (y + z) at (1, 2, 3)
        let x = Dual2Vec::variable(1.0, 0, 3);
        let y = Dual2Vec::variable(2.0, 1, 3);
        let z = Dual2Vec::variable(3.0, 2, 3);
        let a = &x + &y;
        let b = &y + &z;
        let f = &a * &b;
        // f = (1+2)(2+3) = 15
        assert_eq!(f.value(), 15.0);
        // ∂f/∂x = b = 5; ∂f/∂y = a+b = 8; ∂f/∂z = a = 3
        assert_eq!(f.gradient(), &array![5.0, 8.0, 3.0]);
        // Bit-exact symmetry before ANY symmetrize call
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_div_direct_form_at_b_near_one() {
        // f(x, y) = x / y at (x, y) = (7.5, 1.0)
        // Chosen so b = 1 is the classical Mul∘Recip precision-loss point.
        //   value    = 7.5
        //   ∂f/∂x    = 1/y = 1
        //   ∂f/∂y    = -x/y² = -7.5
        //   ∂²f/∂x²  = 0
        //   ∂²f/∂y²  = 2x/y³ = 15
        //   ∂²f/∂x∂y = -1/y² = -1
        let x = Dual2Vec::variable(7.5, 0, 2);
        let y = Dual2Vec::variable(1.0, 1, 2);
        let f = &x / &y;
        assert_eq!(f.value(), 7.5);
        assert_eq!(f.gradient()[0], 1.0);
        assert_eq!(f.gradient()[1], -7.5);
        // Bit-exact structural symmetry BEFORE any element assertion
        assert_eq!(f.hess, f.hess.t());
        assert_eq!(f.hessian()[[0, 0]], 0.0);
        assert_eq!(f.hessian()[[1, 1]], 15.0);
        assert_eq!(f.hessian()[[0, 1]], -1.0);
        assert_eq!(f.hessian()[[1, 0]], -1.0);
    }

    #[test]
    fn test_div_by_constant() {
        // f(x) = x / 2 at x = 6 → 3, grad[0] = 0.5, hess = 0
        let x = Dual2Vec::variable(6.0, 0, 1);
        let two = Dual2Vec::constant(2.0, 1);
        let f = &x / &two;
        assert_eq!(f.value(), 3.0);
        assert_eq!(f.gradient()[0], 0.5);
        assert_eq!(f.hess, f.hess.t());
        assert_eq!(f.hessian()[[0, 0]], 0.0);
    }

    // ========================================================================
    // Unary elementary + pow unit tests
    //
    // Per-elementary smoke tests — the BULK analytical cross-check suite
    // lives in `tests/dual2vec_analytical.rs`. Each test here asserts
    // value / gradient / Hessian on a simple 2-variable point and checks
    // `hess == hess.t()` bit-exactly before any symmetrize call.
    // ========================================================================

    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_elementary_sin_at_xy() {
        // f(x, y) = sin(x + y) at (0.3, 0.4), sum = 0.7
        let x = Dual2Vec::variable(0.3, 0, 2);
        let y = Dual2Vec::variable(0.4, 1, 2);
        let f = (&x + &y).sin();
        let u: f64 = 0.7;
        assert_abs_diff_eq!(f.value(), u.sin(), epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], u.cos(), epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[1], u.cos(), epsilon = 1e-15);
        let expected_hess = -u.sin();
        assert_abs_diff_eq!(f.hessian()[[0, 0]], expected_hess, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian()[[1, 1]], expected_hess, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian()[[0, 1]], expected_hess, epsilon = 1e-15);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_elementary_cos_at_xy() {
        // f(x, y) = cos(x + y) at (0.3, 0.4), sum = 0.7
        let x = Dual2Vec::variable(0.3, 0, 2);
        let y = Dual2Vec::variable(0.4, 1, 2);
        let f = (&x + &y).cos();
        let u: f64 = 0.7;
        assert_abs_diff_eq!(f.value(), u.cos(), epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], -u.sin(), epsilon = 1e-15);
        let expected_hess = -u.cos();
        assert_abs_diff_eq!(f.hessian()[[0, 1]], expected_hess, epsilon = 1e-15);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_elementary_tan_at_xy() {
        let x = Dual2Vec::variable(0.3, 0, 2);
        let y = Dual2Vec::variable(0.4, 1, 2);
        let f = (&x + &y).tan();
        let u: f64 = 0.7;
        let t = u.tan();
        assert_abs_diff_eq!(f.value(), t, epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], 1.0 + t * t, epsilon = 1e-14);
        let expected_hess = 2.0 * t * (1.0 + t * t);
        assert_abs_diff_eq!(f.hessian()[[0, 1]], expected_hess, epsilon = 1e-14);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_elementary_exp_at_xy() {
        // f(x, y) = exp(x + y) at (0.3, 0.4), sum = 0.7, e = exp(0.7)
        let x = Dual2Vec::variable(0.3, 0, 2);
        let y = Dual2Vec::variable(0.4, 1, 2);
        let f = (&x + &y).exp();
        let e = 0.7_f64.exp();
        assert_abs_diff_eq!(f.value(), e, epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], e, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian()[[0, 0]], e, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian()[[1, 1]], e, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian()[[0, 1]], e, epsilon = 1e-15);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_elementary_ln_at_xy() {
        // f(x, y) = ln(x·y) at (2, 3), u = 6
        let x = Dual2Vec::variable(2.0, 0, 2);
        let y = Dual2Vec::variable(3.0, 1, 2);
        let f = (&x * &y).ln();
        assert_abs_diff_eq!(f.value(), 6.0_f64.ln(), epsilon = 1e-15);
        // ∂/∂x ln(xy) = 1/x = 0.5
        assert_abs_diff_eq!(f.gradient()[0], 0.5, epsilon = 1e-15);
        // ∂/∂y ln(xy) = 1/y ≈ 0.333...
        assert_abs_diff_eq!(f.gradient()[1], 1.0 / 3.0, epsilon = 1e-15);
        // ∂²/∂x² ln(xy) = -1/x² = -0.25
        assert_abs_diff_eq!(f.hessian()[[0, 0]], -0.25, epsilon = 1e-15);
        // ∂²/∂y² ln(xy) = -1/y² = -1/9
        assert_abs_diff_eq!(f.hessian()[[1, 1]], -1.0 / 9.0, epsilon = 1e-15);
        // ∂²/∂x∂y ln(xy) = 0 (symbolically)
        assert_abs_diff_eq!(f.hessian()[[0, 1]], 0.0, epsilon = 1e-14);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_elementary_sqrt_at_xy() {
        // f(x, y) = sqrt(x + y) at (1, 3), u = 4
        let x = Dual2Vec::variable(1.0, 0, 2);
        let y = Dual2Vec::variable(3.0, 1, 2);
        let f = (&x + &y).sqrt();
        assert_abs_diff_eq!(f.value(), 2.0, epsilon = 1e-15);
        // ∂/∂x √(x+y) = 1/(2√(x+y)) = 0.25
        assert_abs_diff_eq!(f.gradient()[0], 0.25, epsilon = 1e-15);
        // ∂²/∂x² √(x+y) = -1/(4·(x+y)^{3/2}) = -1/32
        assert_abs_diff_eq!(f.hessian()[[0, 0]], -1.0 / 32.0, epsilon = 1e-15);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_elementary_tanh_at_xy() {
        let x = Dual2Vec::variable(0.3, 0, 2);
        let y = Dual2Vec::variable(0.4, 1, 2);
        let f = (&x + &y).tanh();
        let u: f64 = 0.7;
        let t = u.tanh();
        assert_abs_diff_eq!(f.value(), t, epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], 1.0 - t * t, epsilon = 1e-14);
        let expected_hess = -2.0 * t * (1.0 - t * t);
        assert_abs_diff_eq!(f.hessian()[[0, 1]], expected_hess, epsilon = 1e-14);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_elementary_atan_at_xy() {
        let x = Dual2Vec::variable(0.3, 0, 2);
        let y = Dual2Vec::variable(0.4, 1, 2);
        let f = (&x + &y).atan();
        let u: f64 = 0.7;
        assert_abs_diff_eq!(f.value(), u.atan(), epsilon = 1e-15);
        let fp = 1.0 / (1.0 + u * u);
        assert_abs_diff_eq!(f.gradient()[0], fp, epsilon = 1e-15);
        let fpp = -2.0 * u / ((1.0 + u * u) * (1.0 + u * u));
        assert_abs_diff_eq!(f.hessian()[[0, 1]], fpp, epsilon = 1e-14);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_elementary_asinh_at_xy() {
        let x = Dual2Vec::variable(0.3, 0, 2);
        let y = Dual2Vec::variable(0.4, 1, 2);
        let f = (&x + &y).asinh();
        let u: f64 = 0.7;
        assert_abs_diff_eq!(f.value(), u.asinh(), epsilon = 1e-15);
        let fp = 1.0 / (1.0 + u * u).sqrt();
        assert_abs_diff_eq!(f.gradient()[0], fp, epsilon = 1e-15);
        let d = 1.0 + u * u;
        let fpp = -u / (d * d.sqrt());
        assert_abs_diff_eq!(f.hessian()[[0, 1]], fpp, epsilon = 1e-14);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_elementary_erf_at_xy() {
        // f(x, y) = erf(x + y) at (0.3, 0.4), u = 0.7
        //
        // Note: `crate::math::erf` uses the A&S 7.1.26 polynomial
        // approximation (~1.5e-7 value error), so the *value* assertion
        // relaxes to 1e-6. The *derivative* assertions stay at 1e-14
        // because g'/g'' use only f64::exp + arithmetic (no approximation).
        let x = Dual2Vec::variable(0.3, 0, 2);
        let y = Dual2Vec::variable(0.4, 1, 2);
        let f = (&x + &y).erf();
        let u: f64 = 0.7;
        let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
        let expected_fp = two_over_sqrt_pi * (-u * u).exp();
        let expected_fpp = -2.0 * u * expected_fp;
        // Value: A&S 7.1.26 approximation floor is ~1.5e-7
        assert_abs_diff_eq!(f.value(), crate::math::erf(u), epsilon = 1e-15);
        // Gradient: pure exp + arithmetic, full f64 precision
        assert_abs_diff_eq!(f.gradient()[0], expected_fp, epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[1], expected_fp, epsilon = 1e-15);
        // Hessian cross term: -2u·(2/√π)·exp(-u²) via macro's outer-product
        assert_abs_diff_eq!(f.hessian()[[0, 0]], expected_fpp, epsilon = 1e-14);
        assert_abs_diff_eq!(f.hessian()[[1, 1]], expected_fpp, epsilon = 1e-14);
        assert_abs_diff_eq!(f.hessian()[[0, 1]], expected_fpp, epsilon = 1e-14);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_powf_direct_form_at_xy() {
        // f(x, y) = (x + y)^3 at (1, 2), u = 3
        //   value  = 27
        //   grad   = 3·u² = 27 in both components (chain rule via (x+y))
        //   hess   = 6·u  = 18 in all 4 entries
        let x = Dual2Vec::variable(1.0, 0, 2);
        let y = Dual2Vec::variable(2.0, 1, 2);
        let f = (&x + &y).powf(3.0);
        assert_abs_diff_eq!(f.value(), 27.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.gradient()[0], 27.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.gradient()[1], 27.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.hessian()[[0, 0]], 18.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.hessian()[[1, 1]], 18.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.hessian()[[0, 1]], 18.0, epsilon = 1e-13);
        assert_eq!(f.hess, f.hess.t());
    }

    #[test]
    fn test_powd_via_exp_ln() {
        // f(x, y) = x^y at (2, 3) → 8
        //   ∂f/∂x = y·x^{y-1} = 3·4 = 12
        //   ∂f/∂y = x^y·ln x = 8·ln 2 ≈ 5.5451...
        let x = Dual2Vec::variable(2.0, 0, 2);
        let y = Dual2Vec::variable(3.0, 1, 2);
        let f = x.powd(y);
        assert_abs_diff_eq!(f.value(), 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(f.gradient()[0], 12.0, epsilon = 1e-12);
        assert_abs_diff_eq!(f.gradient()[1], 8.0 * 2.0_f64.ln(), epsilon = 1e-12);
        assert_eq!(f.hess, f.hess.t());
    }

    // Silence unused PI warning — it's available for future tests using π as a test point.
    #[allow(dead_code)]
    fn _use_pi() -> f64 {
        PI
    }
}
