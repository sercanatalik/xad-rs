//! # `Jet2Vec` — dense multi-variable second-order forward-mode AD
//!
//! `Jet2Vec<T>` stores `(value, grad ∈ ℝⁿ, hess ∈ ℝⁿˣⁿ)` and propagates all
//! three through a single forward pass. It is the dense-full-Hessian
//! companion to the single-direction seeded [`Jet2<T>`](crate::forward::jet2::Jet2):
//! where `Jet2<T>` gives you the first- and second-derivative along one
//! seeded direction, `Jet2Vec<T>` gives you the full `n × n` Hessian (plus
//! gradient) for all `n` active inputs at once.
//!
//! ## Quick example — `f(x, y) = x²y + y³` at `(1, 2)`
//!
//! ```
//! use xad_rs::Jet2Vec;
//!
//! // Seed x on axis 0, y on axis 1 of a dim-2 input space
//! let x = Jet2Vec::variable(1.0, 0, 2);
//! let y = Jet2Vec::variable(2.0, 1, 2);
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
//! let h = f.hessian();
//! assert_eq!(h[[0, 0]], 4.0);
//! assert_eq!(h[[0, 1]], 2.0);
//! assert_eq!(h[[1, 0]], 2.0);
//! assert_eq!(h[[1, 1]], 12.0);
//! ```
//!
//! ## Storage: packed upper triangle
//!
//! The Hessian is stored as its **packed upper triangle** — a `Vec<T>` of
//! `n(n+1)/2` entries in row-major order (`(i, j)` with `i ≤ j` at index
//! `i·n − i(i−1)/2 + (j − i)`), not a dense `n × n` matrix:
//!
//! - **Symmetry is structural by storage**: there is only one copy of each
//!   `H[i,j] = H[j,i]` cell, so asymmetry cannot exist, and the mirror
//!   writes of the previous dense layout are gone.
//! - **Half the memory traffic**: every binary op touches `n(n+1)/2`
//!   Hessian cells instead of `n²`, and the per-op loops walk the packed
//!   buffer as one contiguous stream with a running index — no strided
//!   `[j, i]` scatter.
//! - Measured (Apple M-series): −5% in-suite on `hessian/dense_jet2vec`
//!   (`n = 12`); standalone A/B of the op kernel alone: 1.30× / 1.39× /
//!   1.54× at `n = 12/32/48` over the dense-with-mirror layout — the win
//!   grows with `n` as Hessian traffic dominates. Results bit-identical
//!   (identical arithmetic expression order per cell).
//!
//! Read access: [`hessian()`](Jet2Vec::hessian) **materializes** a full
//! symmetric `Array2<T>` (one `n²` copy — negligible next to the per-op
//! savings); [`hessian_packed()`](Jet2Vec::hessian_packed) exposes the
//! packed buffer zero-copy; [`hessian_at(i, j)`](Jet2Vec::hessian_at)
//! reads one cell.
//!
//! ## Cost model
//!
//! Per-op cost is `O(n(n+1)/2)`: every binary op and unary elementary
//! updates each packed Hessian cell once via the multi-variable chain rule
//!
//! ```text
//! grad'[i]   = f'(v) · grad[i]
//! hess'[i,j] = f''(v) · grad[i] · grad[j] + f'(v) · hess[i,j]
//! ```
//!
//! which is a rank-1 outer-product update plus a scaled `hess` copy,
//! evaluated on the upper triangle only.
//!
//! ## When to use `Jet2Vec` vs seeded `Jet2<T>`
//!
//! `Jet2Vec` and seeded `Jet2<T>` are complementary, not competitors. The
//! **crossover** between them is around `n ≈ 50–100`:
//!
//! | Situation                                       | Prefer              |
//! |--------------------------------------------------|---------------------|
//! | Full `n × n` Hessian, `n ≲ 50`                   | `Jet2Vec`          |
//! | Own-gamma (Hessian diagonal) only                | seeded `Jet2<T>`   |
//! | Single-direction second derivative               | seeded `Jet2<T>`   |
//! | Full Hessian, `n ≳ 100`                          | seeded `Jet2<T>` × n passes |
//! | Between `n ≈ 50` and `n ≈ 100`                   | benchmark both — depends on op mix |
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
//! direct form divides exactly once at the end (`inv_b = T::one() / b.value`
//! applied to numerator difference terms), keeping the numerator's
//! precision intact.
//!
//! Similarly, `powf(x, k)` for a constant power `k` is a **direct**
//! chain rule (`k·uᵏ⁻¹` / `k·(k-1)·uᵏ⁻²`) rather than `exp(k·ln(x))`:
//! the composed form loses precision as `x → 0` because `ln(x) → -∞`
//! and the subsequent `exp` amplifies the error.
//!
//! ## Elementary surface
//!
//! `Jet2Vec` supports the unary elementaries stamped from the crate-wide
//! derivative table in `src/elementaries.rs` (`sin`, `cos`, `tan`, `exp`,
//! `ln`, `sqrt`, `tanh`, `atan`, `asinh`, `erf`, ...) from
//! `(f, f', f'')` closure triples.
//!
//! The `erf` *value* goes through the Abramowitz-Stegun 7.1.26 polynomial
//! approximation (~1.5e-7 absolute error, same as `crate::math::erf`),
//! but its analytical derivatives `(2/√π)·exp(-u²)` and `-2u·g'(u)` use
//! pure `T::exp` and are therefore precision-exact for the chosen `T` —
//! Hessian cells flowing through `erf` enjoy full `T` precision.
//!
//! ## Correctness proof
//!
//! Correctness evidence is two-tiered:
//!
//! - **Analytical cross-check suite** at tolerance `1e-13`
//!   (`tests/jet2vec_analytical.rs`): four hand-derived
//!   literal Hessians on `x²y + y³`, `sin(xy)`, `exp(x² + y²)`, and
//!   `(x - y)²·log(x + y)`.
//! - **Deployment-scale Garman–Kohlhagen 6×6 Hessian two-tier check**
//!   at tolerance `1e-11` primary / `5e-5` secondary
//!   (`tests/jet2vec_finance.rs`): primary asserts three
//!   closed-form analytical Greeks (gamma = `H[0,0]`, volga = `H[4,4]`,
//!   vanna = `H[0,4]`) at 1e-11 against `exp(-rf·T)·φ(d1)/(S·σ·√T)` and
//!   friends; secondary asserts all 36 entries at 5e-5 against
//!   `xad_rs::compute_hessian` (exact forward-over-adjoint,
//!   `AReal<Jet1<f64>>`). The 5e-5 bound absorbs the A&S `erf`
//!   approximation's derivative error in the nested value path of that
//!   cross-check — see the `tests/jet2vec_finance.rs` module docs.
//!
//! ## No lifetime parameters
//!
//! `Jet2Vec` carries **zero lifetime parameters** — all fields are owned.
//! This preserves PyO3 binding compatibility (PyO3 cannot express Rust
//! lifetimes across the FFI boundary cleanly).
// `suspicious_arithmetic_impl` fires on the correct dual-number / Hessian
// chain-rule shape (`a * gb + b * ga` etc.) which is not a bug.
#![allow(clippy::suspicious_arithmetic_impl)]

use crate::passive::Passive;
use ndarray::{Array1, Array2};
use std::ops::{Add, Div, Mul, Sub};

/// Number of packed upper-triangle entries (incl. diagonal) for dimension `n`.
#[inline]
fn tri_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Packed index of `(i, j)` with `i ≤ j` in the row-major upper triangle.
/// Row `i` starts at `i·n − i(i−1)/2`, written as `i(2n − i + 1)/2` so the
/// `i = 0` case doesn't underflow in debug builds.
#[inline]
fn tri_idx(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i <= j && j < n);
    i * (2 * n - i + 1) / 2 + (j - i)
}

/// Dense multi-variable second-order forward-mode AD number.
///
/// Generic over the passive scalar `T: Passive` (instantiable as `f32` or
/// `f64`); defaults to `f64` so existing call sites compile unchanged.
///
/// The Hessian is stored as its packed upper triangle — see the module
/// docs for the layout, symmetry, and cost model.
#[derive(Clone, Debug, PartialEq)]
pub struct Jet2Vec<T: Passive = f64> {
    /// Function value `f(x_1, ..., x_n)`
    value: T,
    /// Gradient `[∂f/∂x_1, ..., ∂f/∂x_n]`
    grad: Array1<T>,
    /// Packed upper triangle of the symmetric Hessian, row-major:
    /// `H[i,j]` (`i ≤ j`) at `tri_idx(n, i, j)`, length `n(n+1)/2`.
    hess: Vec<T>,
}

impl<T: Passive> Jet2Vec<T> {
    /// Create the `i`-th active variable in an `n`-dimensional input space.
    ///
    /// The resulting `Jet2Vec` has `value = value`, `grad = e_i` (unit
    /// vector in direction `i`), and `hess = 0`.
    #[inline]
    pub fn variable(value: T, i: usize, n: usize) -> Self {
        let mut grad = Array1::<T>::zeros(n);
        grad[i] = T::one();
        Self { value, grad, hess: vec![T::zero(); tri_len(n)] }
    }

    /// Create a derivative-free `Jet2Vec` (zero grad, zero hess) with
    /// the given `value` and input-space dimension `n`.
    #[inline]
    pub fn constant(value: T, n: usize) -> Self {
        Self {
            value,
            grad: Array1::<T>::zeros(n),
            hess: vec![T::zero(); tri_len(n)],
        }
    }

    /// Value accessor.
    #[inline]
    pub fn value(&self) -> T {
        self.value
    }

    /// Gradient as a shared reference (no allocation).
    #[inline]
    pub fn gradient(&self) -> &Array1<T> {
        &self.grad
    }

    /// Full symmetric `n × n` Hessian, **materialized** from the packed
    /// upper triangle (allocates one `Array2`). For zero-copy access use
    /// [`hessian_packed`](Jet2Vec::hessian_packed) or
    /// [`hessian_at`](Jet2Vec::hessian_at).
    pub fn hessian(&self) -> Array2<T> {
        let n = self.grad.len();
        let mut full = Array2::<T>::zeros((n, n));
        let mut k = 0;
        for i in 0..n {
            for j in i..n {
                let v = self.hess[k];
                full[[i, j]] = v;
                full[[j, i]] = v;
                k += 1;
            }
        }
        full
    }

    /// The packed upper triangle of the Hessian, zero-copy: `n(n+1)/2`
    /// entries in row-major order, `H[i,j]` (`i ≤ j`) at index
    /// `i·n − i(i−1)/2 + (j − i)`.
    #[inline]
    pub fn hessian_packed(&self) -> &[T] {
        &self.hess
    }

    /// One Hessian cell `H[i,j]` (either triangle — symmetry is
    /// structural).
    #[inline]
    pub fn hessian_at(&self, i: usize, j: usize) -> T {
        let n = self.grad.len();
        let (i, j) = if i <= j { (i, j) } else { (j, i) };
        self.hess[tri_idx(n, i, j)]
    }

    /// Input-space dimension `n` — the length of `grad` and the side of
    /// the (packed) Hessian.
    #[inline]
    pub fn len(&self) -> usize {
        self.grad.len()
    }

    /// Returns `true` if this `Jet2Vec` has zero input-space dimension.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.grad.is_empty()
    }
}

#[inline]
fn shape_check<T: Passive>(a: &Jet2Vec<T>, b: &Jet2Vec<T>) {
    debug_assert_eq!(
        a.grad.len(),
        b.grad.len(),
        "Jet2Vec gradient length mismatch: {} vs {}",
        a.grad.len(),
        b.grad.len()
    );
    debug_assert_eq!(
        a.hess.len(),
        b.hess.len(),
        "Jet2Vec hessian length mismatch"
    );
}

// ============================================================================
// Add / Sub — linear propagation of value, grad, and packed hess
// ============================================================================

impl<T: Passive> Add<&Jet2Vec<T>> for &Jet2Vec<T> {
    type Output = Jet2Vec<T>;
    #[inline]
    fn add(self, rhs: &Jet2Vec<T>) -> Jet2Vec<T> {
        shape_check(self, rhs);
        Jet2Vec {
            value: self.value + rhs.value,
            grad: &self.grad + &rhs.grad,
            hess: self.hess.iter().zip(&rhs.hess).map(|(&a, &b)| a + b).collect(),
        }
    }
}

impl<T: Passive> Add for Jet2Vec<T> {
    type Output = Jet2Vec<T>;
    #[inline]
    fn add(mut self, rhs: Jet2Vec<T>) -> Jet2Vec<T> {
        shape_check(&self, &rhs);
        self.value += rhs.value;
        self.grad += &rhs.grad;
        for (a, &b) in self.hess.iter_mut().zip(&rhs.hess) {
            *a += b;
        }
        self
    }
}

impl<T: Passive> Sub<&Jet2Vec<T>> for &Jet2Vec<T> {
    type Output = Jet2Vec<T>;
    #[inline]
    fn sub(self, rhs: &Jet2Vec<T>) -> Jet2Vec<T> {
        shape_check(self, rhs);
        Jet2Vec {
            value: self.value - rhs.value,
            grad: &self.grad - &rhs.grad,
            hess: self.hess.iter().zip(&rhs.hess).map(|(&a, &b)| a - b).collect(),
        }
    }
}

impl<T: Passive> Sub for Jet2Vec<T> {
    type Output = Jet2Vec<T>;
    #[inline]
    fn sub(mut self, rhs: Jet2Vec<T>) -> Jet2Vec<T> {
        shape_check(&self, &rhs);
        self.value -= rhs.value;
        self.grad -= &rhs.grad;
        for (a, &b) in self.hess.iter_mut().zip(&rhs.hess) {
            *a -= b;
        }
        self
    }
}

// ============================================================================
// Mul — second-order product rule with symmetric outer-product cross term
// ============================================================================
//
// For f = a·b with `a, b: Jet2Vec`:
//   f.value = a.value · b.value
//   f.grad  = a.value · b.grad + b.value · a.grad
//   f.hess  = a.value · b.hess + b.value · a.hess
//           + (a.grad ⊗ b.grad + b.grad ⊗ a.grad)   ← symmetric cross term
//
// Evaluated on the packed upper triangle only, walked as one contiguous
// stream with a running index `k` — the packed layout makes the mirror
// write of the previous dense layout structurally unnecessary.

impl<T: Passive> Mul<&Jet2Vec<T>> for &Jet2Vec<T> {
    type Output = Jet2Vec<T>;
    #[inline]
    fn mul(self, rhs: &Jet2Vec<T>) -> Jet2Vec<T> {
        shape_check(self, rhs);
        let a_val = self.value;
        let b_val = rhs.value;
        let n = self.grad.len();

        // grad: a.value · b.grad + b.value · a.grad. Built elementwise to
        // avoid the `ndarray::ScalarOperand` bound on `T`, which `Passive`
        // does not require.
        let grad: Array1<T> = self
            .grad
            .iter()
            .zip(rhs.grad.iter())
            .map(|(&ga, &gb)| a_val * gb + b_val * ga)
            .collect();

        let ga = self.grad.as_slice().expect("Jet2Vec grad must be contiguous");
        let gb = rhs.grad.as_slice().expect("Jet2Vec grad must be contiguous");
        let mut hess = Vec::with_capacity(self.hess.len());
        let mut k = 0;
        for i in 0..n {
            let ga_i = ga[i];
            let gb_i = gb[i];
            for j in i..n {
                let cross = ga_i * gb[j] + gb_i * ga[j];
                hess.push(a_val * rhs.hess[k] + b_val * self.hess[k] + cross);
                k += 1;
            }
        }

        Jet2Vec { value: a_val * b_val, grad, hess }
    }
}

impl<T: Passive> Mul for Jet2Vec<T> {
    type Output = Jet2Vec<T>;
    #[inline]
    fn mul(mut self, rhs: Jet2Vec<T>) -> Jet2Vec<T> {
        shape_check(&self, &rhs);
        let a_val = self.value;
        let b_val = rhs.value;
        let n = self.grad.len();
        self.value = a_val * b_val;

        // hess in place, reading the ORIGINAL gradients (the cross term
        // must not see the updated ones).
        {
            let ga = self.grad.as_slice().expect("Jet2Vec grad must be contiguous");
            let gb = rhs.grad.as_slice().expect("Jet2Vec grad must be contiguous");
            let mut k = 0;
            for i in 0..n {
                let ga_i = ga[i];
                let gb_i = gb[i];
                for j in i..n {
                    let cross = ga_i * gb[j] + gb_i * ga[j];
                    self.hess[k] = a_val * rhs.hess[k] + b_val * self.hess[k] + cross;
                    k += 1;
                }
            }
        }

        // grad in place: self.grad ← a_val · rhs.grad + b_val · self.grad.
        for (sg, &rg) in self.grad.iter_mut().zip(rhs.grad.iter()) {
            *sg = a_val * rg + b_val * *sg;
        }

        self
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
// the already-computed gradient of the quotient and b's gradient,
// evaluated on the packed upper triangle only.

impl<T: Passive> Div<&Jet2Vec<T>> for &Jet2Vec<T> {
    type Output = Jet2Vec<T>;
    #[inline]
    fn div(self, rhs: &Jet2Vec<T>) -> Jet2Vec<T> {
        shape_check(self, rhs);
        let b_val = rhs.value;
        debug_assert!(b_val != T::zero(), "Jet2Vec::div by zero value");
        let inv_b = T::one() / b_val;
        let n = self.grad.len();

        let value = self.value * inv_b;

        // grad: (a.grad - value · b.grad) / b.value, elementwise.
        let grad: Array1<T> = self
            .grad
            .iter()
            .zip(rhs.grad.iter())
            .map(|(&ga, &gb)| (ga - value * gb) * inv_b)
            .collect();

        // hess: cross term uses the NEW gradient (`grad` above), per the
        // closed-form derivation documented at the head of the Div block.
        let gq = grad.as_slice().expect("Jet2Vec grad must be contiguous");
        let gb = rhs.grad.as_slice().expect("Jet2Vec grad must be contiguous");
        let mut hess = Vec::with_capacity(self.hess.len());
        let mut k = 0;
        for i in 0..n {
            let gq_i = gq[i];
            let gb_i = gb[i];
            for j in i..n {
                let cross = gq_i * gb[j] + gb_i * gq[j];
                hess.push((self.hess[k] - value * rhs.hess[k] - cross) * inv_b);
                k += 1;
            }
        }

        Jet2Vec { value, grad, hess }
    }
}

impl<T: Passive> Div for Jet2Vec<T> {
    type Output = Jet2Vec<T>;
    #[inline]
    fn div(mut self, rhs: Jet2Vec<T>) -> Jet2Vec<T> {
        shape_check(&self, &rhs);
        let b_val = rhs.value;
        debug_assert!(b_val != T::zero(), "Jet2Vec::div by zero value");
        let inv_b = T::one() / b_val;
        let n = self.grad.len();
        let value = self.value * inv_b;
        self.value = value;

        // grad in place FIRST: the hess loop below needs the NEW gradient
        // (closed form) plus the original self.hess — untouched until then.
        for (sg, &rg) in self.grad.iter_mut().zip(rhs.grad.iter()) {
            *sg = (*sg - value * rg) * inv_b;
        }

        // hess in place using the just-updated self.grad.
        {
            let gq = self.grad.as_slice().expect("Jet2Vec grad must be contiguous");
            let gb = rhs.grad.as_slice().expect("Jet2Vec grad must be contiguous");
            let mut k = 0;
            for i in 0..n {
                let gq_i = gq[i];
                let gb_i = gb[i];
                for j in i..n {
                    let cross = gq_i * gb[j] + gb_i * gq[j];
                    self.hess[k] = (self.hess[k] - value * rhs.hess[k] - cross) * inv_b;
                    k += 1;
                }
            }
        }

        self
    }
}

// ============================================================================
// Unary elementaries — stamped via macro from the crate-wide derivative
// table (`src/elementaries.rs`).
// ============================================================================
//
// Multi-variable chain rule for a unary function `g(u)` applied to
// `u: Jet2Vec`:
//
//   out.value      = g(u.value)
//   out.grad[i]    = g'(u.value) · u.grad[i]
//   out.hess[i, j] = g''(u.value) · u.grad[i] · u.grad[j]
//                  + g'(u.value)  · u.hess[i, j]
//
// evaluated in place on the consumed `self`'s packed upper triangle
// (one contiguous stream, running index).

/// Update the packed Hessian in place for a unary elementary: reads the
/// ORIGINAL gradient (must run before the gradient is scaled).
#[inline]
fn accumulate_unary_hess_packed<T: Passive>(hess: &mut [T], grad: &Array1<T>, fp: T, fpp: T) {
    let g = grad.as_slice().expect("Jet2Vec grad must be contiguous");
    let n = g.len();
    let mut k = 0;
    for i in 0..n {
        let g_i = g[i];
        for &g_j in &g[i..n] {
            hess[k] = fp * hess[k] + fpp * g_i * g_j;
            k += 1;
        }
    }
}

/// Stamp a unary elementary from a crate-wide derivative-table entry
/// `(name, doc, |x| val, |x, r| d1, |x, r, d1| d2)` — see
/// `src/elementaries.rs`.
macro_rules! stamp_jet2vec_unary {
    ($name:ident, $doc:literal, $val:expr, $d1:expr, $d2:expr) => {
        impl<T: Passive> Jet2Vec<T> {
            #[doc = $doc]
            #[doc = ""]
            #[doc = "Propagates value, gradient, and Hessian in one forward pass via the"]
            #[doc = "multi-variable chain rule (see the module docs for the derivation)."]
            #[inline]
            pub fn $name(mut self) -> Self {
                let u = self.value;
                let fv: T = ($val)(u);
                let fp: T = ($d1)(u, fv);
                let fpp: T = ($d2)(u, fv, fp);
                self.value = fv;

                // hess in place using the ORIGINAL self.grad (must run
                // before grad is scaled below).
                accumulate_unary_hess_packed(&mut self.hess, &self.grad, fp, fpp);

                // grad in place: self.grad ← fp · self.grad.
                for d in self.grad.iter_mut() {
                    *d *= fp;
                }

                self
            }
        }
    };
}

crate::elementaries::for_each_unary_elementary!(stamp_jet2vec_unary);

// ============================================================================
// Power — direct-form `powf(k: T)`
// ============================================================================
//
// `powf(k)` uses the direct chain-rule closed form `g(u) = u^k` with
// `g'(u) = k·u^{k-1}`, `g''(u) = k·(k-1)·u^{k-2}`. It is NOT
// `exp(k·ln(u))` — the exp·ln form loses precision as `u → 0` where
// `ln(u) → -∞`.

impl<T: Passive> Jet2Vec<T> {
    /// Raise `self` to a **constant** scalar power `k: T` via the direct
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
    #[inline]
    pub fn powf(mut self, k: T) -> Self {
        let u = self.value;
        let two = T::from(2.0).unwrap();
        let fv = u.powf(k);
        let fp = k * u.powf(k - T::one());
        let fpp = k * (k - T::one()) * u.powf(k - two);
        self.value = fv;

        // Same shape as a unary elementary: in-place hess update + grad scale.
        accumulate_unary_hess_packed(&mut self.hess, &self.grad, fp, fpp);
        for d in self.grad.iter_mut() {
            *d *= fp;
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_variable_constructor() {
        let x = Jet2Vec::variable(3.0_f64, 0, 2);
        assert_eq!(x.value(), 3.0);
        assert_eq!(x.gradient(), &array![1.0, 0.0]);
        assert_eq!(x.hessian(), Array2::<f64>::zeros((2, 2)));
    }

    #[test]
    fn test_variable_constructor_index_1() {
        let y = Jet2Vec::variable(5.0_f64, 1, 3);
        assert_eq!(y.value(), 5.0);
        assert_eq!(y.gradient(), &array![0.0, 1.0, 0.0]);
        assert_eq!(y.hessian(), Array2::<f64>::zeros((3, 3)));
    }

    #[test]
    fn test_constant_constructor() {
        let c = Jet2Vec::constant(7.5_f64, 4);
        assert_eq!(c.value(), 7.5);
        assert_eq!(c.gradient(), &Array1::<f64>::zeros(4));
        assert_eq!(c.hessian(), Array2::<f64>::zeros((4, 4)));
        assert_eq!(c.len(), 4);
        assert!(!c.is_empty());
    }

    #[test]
    fn test_packed_layout_and_accessors() {
        // Fill H via ops and cross-check hessian_at / hessian_packed /
        // materialized hessian() against each other.
        let x = Jet2Vec::variable(1.5_f64, 0, 3);
        let y = Jet2Vec::variable(2.5_f64, 1, 3);
        let z = Jet2Vec::variable(0.5_f64, 2, 3);
        let f = (&(&x * &y) * &z).exp();
        let full = f.hessian();
        let n = 3;
        assert_eq!(f.hessian_packed().len(), n * (n + 1) / 2);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(full[[i, j]], f.hessian_at(i, j), "cell ({i},{j})");
                assert_eq!(full[[i, j]], full[[j, i]], "symmetry ({i},{j})");
            }
        }
        // Packed row-major layout: (i, j) at i*n - i(i-1)/2 + (j - i).
        assert_eq!(f.hessian_packed()[0], full[[0, 0]]);
        assert_eq!(f.hessian_packed()[1], full[[0, 1]]);
        assert_eq!(f.hessian_packed()[2], full[[0, 2]]);
        assert_eq!(f.hessian_packed()[3], full[[1, 1]]);
        assert_eq!(f.hessian_packed()[4], full[[1, 2]]);
        assert_eq!(f.hessian_packed()[5], full[[2, 2]]);
    }

    #[test]
    fn test_add_by_value() {
        let x = Jet2Vec::variable(3.0_f64, 0, 2);
        let y = Jet2Vec::variable(4.0_f64, 1, 2);
        let s = x + y;
        assert_eq!(s.value(), 7.0);
        assert_eq!(s.gradient(), &array![1.0, 1.0]);
        assert_eq!(s.hessian(), Array2::<f64>::zeros((2, 2)));
    }

    #[test]
    fn test_sub_by_value() {
        let x = Jet2Vec::variable(5.0_f64, 0, 2);
        let y = Jet2Vec::variable(2.0_f64, 1, 2);
        let d = x - y;
        assert_eq!(d.value(), 3.0);
        assert_eq!(d.gradient(), &array![1.0, -1.0]);
        assert_eq!(d.hessian(), Array2::<f64>::zeros((2, 2)));
    }

    #[test]
    fn test_add_by_ref_matches_by_value() {
        let x = Jet2Vec::variable(3.0_f64, 0, 2);
        let y = Jet2Vec::variable(4.0_f64, 1, 2);
        let s_ref = &x + &y;
        let s_val = x.clone() + y.clone();
        assert_eq!(s_ref, s_val);
    }

    #[test]
    fn test_sub_by_ref_matches_by_value() {
        let x = Jet2Vec::variable(5.0_f64, 0, 2);
        let y = Jet2Vec::variable(2.0_f64, 1, 2);
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
        let a = Jet2Vec::variable(1.0_f64, 0, 2);
        let b = Jet2Vec::variable(1.0_f64, 0, 3);
        let _ = &a + &b;
    }

    #[test]
    fn test_mul_cross_term_symmetric() {
        let x = Jet2Vec::variable(3.0_f64, 0, 2);
        let y = Jet2Vec::variable(4.0_f64, 1, 2);
        let f = &x * &y; // f = x·y at (3, 4) → value = 12
        assert_eq!(f.value(), 12.0);
        assert_eq!(f.gradient()[0], 4.0); // ∂(xy)/∂x = y
        assert_eq!(f.gradient()[1], 3.0); // ∂(xy)/∂y = x
        assert_eq!(f.hessian_at(0, 0), 0.0);
        assert_eq!(f.hessian_at(1, 1), 0.0);
        assert_eq!(f.hessian_at(0, 1), 1.0); // ∂²(xy)/∂x∂y = 1
        assert_eq!(f.hessian_at(1, 0), 1.0); // symmetric (structural)
    }

    #[test]
    fn test_mul_square_diagonal() {
        // f = x² at (5, _): value = 25, grad = [10, 0], hess[0,0] = 2
        let x = Jet2Vec::variable(5.0_f64, 0, 2);
        let f = &x * &x;
        assert_eq!(f.value(), 25.0);
        assert_eq!(f.gradient(), &array![10.0, 0.0]);
        assert_eq!(f.hessian_at(0, 0), 2.0);
        assert_eq!(f.hessian_at(1, 1), 0.0);
        assert_eq!(f.hessian_at(0, 1), 0.0);
    }

    #[test]
    fn test_mul_ref_matches_owned_on_3_vars() {
        // 3-variable mul chain: (x + y) * (y + z) at (1, 2, 3)
        let x = Jet2Vec::variable(1.0_f64, 0, 3);
        let y = Jet2Vec::variable(2.0_f64, 1, 3);
        let z = Jet2Vec::variable(3.0_f64, 2, 3);
        let a = &x + &y;
        let b = &y + &z;
        let f = &a * &b;
        // f = (1+2)(2+3) = 15
        assert_eq!(f.value(), 15.0);
        // ∂f/∂x = b = 5; ∂f/∂y = a+b = 8; ∂f/∂z = a = 3
        assert_eq!(f.gradient(), &array![5.0, 8.0, 3.0]);
        // Ref and owned forms must agree bit-exactly.
        assert_eq!(f, a.clone() * b.clone());
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
        let x = Jet2Vec::variable(7.5_f64, 0, 2);
        let y = Jet2Vec::variable(1.0_f64, 1, 2);
        let f = &x / &y;
        assert_eq!(f.value(), 7.5);
        assert_eq!(f.gradient()[0], 1.0);
        assert_eq!(f.gradient()[1], -7.5);
        assert_eq!(f.hessian_at(0, 0), 0.0);
        assert_eq!(f.hessian_at(1, 1), 15.0);
        assert_eq!(f.hessian_at(0, 1), -1.0);
        assert_eq!(f.hessian_at(1, 0), -1.0);
        // Ref and owned forms must agree bit-exactly.
        assert_eq!(f, x.clone() / y.clone());
    }

    #[test]
    fn test_div_by_constant() {
        // f(x) = x / 2 at x = 6 → 3, grad[0] = 0.5, hess = 0
        let x = Jet2Vec::variable(6.0_f64, 0, 1);
        let two = Jet2Vec::constant(2.0_f64, 1);
        let f = &x / &two;
        assert_eq!(f.value(), 3.0);
        assert_eq!(f.gradient()[0], 0.5);
        assert_eq!(f.hessian_at(0, 0), 0.0);
    }

    // ========================================================================
    // Unary elementary + pow unit tests
    //
    // Per-elementary smoke tests — the BULK analytical cross-check suite
    // lives in `tests/jet2vec_analytical.rs`.
    // ========================================================================

    use approx::assert_abs_diff_eq;

    #[test]
    fn test_elementary_sin_at_xy() {
        // f(x, y) = sin(x + y) at (0.3, 0.4), sum = 0.7
        let x = Jet2Vec::variable(0.3_f64, 0, 2);
        let y = Jet2Vec::variable(0.4_f64, 1, 2);
        let f = (&x + &y).sin();
        let u: f64 = 0.7;
        assert_abs_diff_eq!(f.value(), u.sin(), epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], u.cos(), epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[1], u.cos(), epsilon = 1e-15);
        let expected_hess = -u.sin();
        assert_abs_diff_eq!(f.hessian_at(0, 0), expected_hess, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian_at(1, 1), expected_hess, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian_at(0, 1), expected_hess, epsilon = 1e-15);
    }

    #[test]
    fn test_elementary_cos_at_xy() {
        // f(x, y) = cos(x + y) at (0.3, 0.4), sum = 0.7
        let x = Jet2Vec::variable(0.3_f64, 0, 2);
        let y = Jet2Vec::variable(0.4_f64, 1, 2);
        let f = (&x + &y).cos();
        let u: f64 = 0.7;
        assert_abs_diff_eq!(f.value(), u.cos(), epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], -u.sin(), epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian_at(0, 1), -u.cos(), epsilon = 1e-15);
    }

    #[test]
    fn test_elementary_tan_at_xy() {
        let x = Jet2Vec::variable(0.3_f64, 0, 2);
        let y = Jet2Vec::variable(0.4_f64, 1, 2);
        let f = (&x + &y).tan();
        let u: f64 = 0.7;
        let t = u.tan();
        assert_abs_diff_eq!(f.value(), t, epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], 1.0 + t * t, epsilon = 1e-14);
        let expected_hess = 2.0 * t * (1.0 + t * t);
        assert_abs_diff_eq!(f.hessian_at(0, 1), expected_hess, epsilon = 1e-14);
    }

    #[test]
    fn test_elementary_exp_at_xy() {
        // f(x, y) = exp(x + y) at (0.3, 0.4), sum = 0.7, e = exp(0.7)
        let x = Jet2Vec::variable(0.3_f64, 0, 2);
        let y = Jet2Vec::variable(0.4_f64, 1, 2);
        let f = (&x + &y).exp();
        let e = 0.7_f64.exp();
        assert_abs_diff_eq!(f.value(), e, epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], e, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian_at(0, 0), e, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian_at(1, 1), e, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian_at(0, 1), e, epsilon = 1e-15);
    }

    #[test]
    fn test_elementary_ln_at_xy() {
        // f(x, y) = ln(x·y) at (2, 3), u = 6
        let x = Jet2Vec::variable(2.0_f64, 0, 2);
        let y = Jet2Vec::variable(3.0_f64, 1, 2);
        let f = (&x * &y).ln();
        assert_abs_diff_eq!(f.value(), 6.0_f64.ln(), epsilon = 1e-15);
        // ∂/∂x ln(xy) = 1/x = 0.5
        assert_abs_diff_eq!(f.gradient()[0], 0.5, epsilon = 1e-15);
        // ∂/∂y ln(xy) = 1/y ≈ 0.333...
        assert_abs_diff_eq!(f.gradient()[1], 1.0 / 3.0, epsilon = 1e-15);
        // ∂²/∂x² ln(xy) = -1/x² = -0.25
        assert_abs_diff_eq!(f.hessian_at(0, 0), -0.25, epsilon = 1e-15);
        // ∂²/∂y² ln(xy) = -1/y² = -1/9
        assert_abs_diff_eq!(f.hessian_at(1, 1), -1.0 / 9.0, epsilon = 1e-15);
        // ∂²/∂x∂y ln(xy) = 0 (symbolically)
        assert_abs_diff_eq!(f.hessian_at(0, 1), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_elementary_sqrt_at_xy() {
        // f(x, y) = sqrt(x + y) at (1, 3), u = 4
        let x = Jet2Vec::variable(1.0_f64, 0, 2);
        let y = Jet2Vec::variable(3.0_f64, 1, 2);
        let f = (&x + &y).sqrt();
        assert_abs_diff_eq!(f.value(), 2.0, epsilon = 1e-15);
        // ∂/∂x √(x+y) = 1/(2√(x+y)) = 0.25
        assert_abs_diff_eq!(f.gradient()[0], 0.25, epsilon = 1e-15);
        // ∂²/∂x² √(x+y) = -1/(4·(x+y)^{3/2}) = -1/32
        assert_abs_diff_eq!(f.hessian_at(0, 0), -1.0 / 32.0, epsilon = 1e-15);
    }

    #[test]
    fn test_elementary_tanh_at_xy() {
        let x = Jet2Vec::variable(0.3_f64, 0, 2);
        let y = Jet2Vec::variable(0.4_f64, 1, 2);
        let f = (&x + &y).tanh();
        let u: f64 = 0.7;
        let t = u.tanh();
        assert_abs_diff_eq!(f.value(), t, epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], 1.0 - t * t, epsilon = 1e-14);
        let expected_hess = -2.0 * t * (1.0 - t * t);
        assert_abs_diff_eq!(f.hessian_at(0, 1), expected_hess, epsilon = 1e-14);
    }

    #[test]
    fn test_elementary_atan_at_xy() {
        let x = Jet2Vec::variable(0.3_f64, 0, 2);
        let y = Jet2Vec::variable(0.4_f64, 1, 2);
        let f = (&x + &y).atan();
        let u: f64 = 0.7;
        assert_abs_diff_eq!(f.value(), u.atan(), epsilon = 1e-15);
        let fp = 1.0 / (1.0 + u * u);
        assert_abs_diff_eq!(f.gradient()[0], fp, epsilon = 1e-15);
        let fpp = -2.0 * u / ((1.0 + u * u) * (1.0 + u * u));
        assert_abs_diff_eq!(f.hessian_at(0, 1), fpp, epsilon = 1e-14);
    }

    #[test]
    fn test_elementary_asinh_at_xy() {
        let x = Jet2Vec::variable(0.3_f64, 0, 2);
        let y = Jet2Vec::variable(0.4_f64, 1, 2);
        let f = (&x + &y).asinh();
        let u: f64 = 0.7;
        assert_abs_diff_eq!(f.value(), u.asinh(), epsilon = 1e-15);
        let fp = 1.0 / (1.0 + u * u).sqrt();
        assert_abs_diff_eq!(f.gradient()[0], fp, epsilon = 1e-15);
        let d = 1.0 + u * u;
        let fpp = -u / (d * d.sqrt());
        assert_abs_diff_eq!(f.hessian_at(0, 1), fpp, epsilon = 1e-14);
    }

    #[test]
    fn test_elementary_erf_at_xy() {
        // f(x, y) = erf(x + y) at (0.3, 0.4), u = 0.7
        //
        // Note: `crate::math::erf` uses the A&S 7.1.26 polynomial
        // approximation (~1.5e-7 value error). The *derivative* assertions
        // stay at 1e-14 because g'/g'' use only f64::exp + arithmetic.
        let x = Jet2Vec::variable(0.3_f64, 0, 2);
        let y = Jet2Vec::variable(0.4_f64, 1, 2);
        let f = (&x + &y).erf();
        let u: f64 = 0.7;
        let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
        let expected_fp = two_over_sqrt_pi * (-u * u).exp();
        let expected_fpp = -2.0 * u * expected_fp;
        assert_abs_diff_eq!(f.value(), crate::math::erf::<f64>(u), epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[0], expected_fp, epsilon = 1e-15);
        assert_abs_diff_eq!(f.gradient()[1], expected_fp, epsilon = 1e-15);
        assert_abs_diff_eq!(f.hessian_at(0, 0), expected_fpp, epsilon = 1e-14);
        assert_abs_diff_eq!(f.hessian_at(1, 1), expected_fpp, epsilon = 1e-14);
        assert_abs_diff_eq!(f.hessian_at(0, 1), expected_fpp, epsilon = 1e-14);
    }

    #[test]
    fn test_powf_direct_form_at_xy() {
        // f(x, y) = (x + y)^3 at (1, 2), u = 3
        //   value  = 27
        //   grad   = 3·u² = 27 in both components (chain rule via (x+y))
        //   hess   = 6·u  = 18 in all 4 entries
        let x = Jet2Vec::variable(1.0_f64, 0, 2);
        let y = Jet2Vec::variable(2.0_f64, 1, 2);
        let f = (&x + &y).powf(3.0);
        assert_abs_diff_eq!(f.value(), 27.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.gradient()[0], 27.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.gradient()[1], 27.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.hessian_at(0, 0), 18.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.hessian_at(1, 1), 18.0, epsilon = 1e-13);
        assert_abs_diff_eq!(f.hessian_at(0, 1), 18.0, epsilon = 1e-13);
    }
}
