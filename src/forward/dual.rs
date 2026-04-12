//! `Dual` - multi-variable first-order forward-mode AD type.
//!
//! A `Dual` carries a real value plus a vector of tangents (one per input
//! variable). A single forward pass through a function built from `Dual`
//! operations yields the value AND the full gradient — every partial
//! `∂f/∂x_i` — in one shot, without a tape.
//!
//! This complements the other AD types in the crate:
//! - [`FReal`](crate::freal::FReal) — forward mode, one tangent direction.
//! - [`Dual2`](crate::dual2::Dual2) — forward mode, one direction, second order.
//! - [`Dual`] (this type) — forward mode, *many* directions at once, first order.
//! - [`AReal`](crate::areal::AReal) — reverse mode (tape-based).
//!
//! The tangent vector is stored as a plain `Vec<f64>`. Every operator is
//! implemented as a single fused loop over the tangent slice (no temporary
//! intermediate buffers), and owned-value forms reuse the left operand's
//! allocation in place, so forward propagation is allocator-light and the
//! inner loops are autovectorizable.
//!
//! Typical use: seed each input variable `x_i` with
//! `Dual::variable(x_i, i, n)`, propagate through the function, then read
//! the gradient from `result.dual()`.
//!
//! ```
//! use xad_rs::Dual;
//! // f(x, y) = x^2 * y, at (x, y) = (3, 4)
//! // ∂f/∂x = 2xy = 24,  ∂f/∂y = x^2 = 9
//! let n = 2;
//! let x = Dual::variable(3.0, 0, n);
//! let y = Dual::variable(4.0, 1, n);
//! let f = &(&x * &x) * &y;
//! assert_eq!(f.real(), 36.0);
//! assert_eq!(f.partial(0), 24.0);
//! assert_eq!(f.partial(1), 9.0);
//! ```
//!
//! This module also contains [`NamedDual`], a named wrapper over `Dual`
//! that provides name-keyed gradient readback via a `NamedForwardTape` scope.

// The explicit `'a, 'b` lifetimes in the by-reference operator impls below are
// kept intentionally for readability — every `Add`/`Sub`/`Mul`/`Div` impl
// follows the same pattern and eliding some but not others would hurt
// symmetry. `suspicious_arithmetic_impl` fires on the correct dual-number
// chain rule `da*b + db*a`, which is not a bug.
#![allow(clippy::needless_lifetimes, clippy::suspicious_arithmetic_impl)]

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Multi-variable forward-mode dual number.
///
/// - `real`:  the function value `f(x_1, ..., x_n)`
/// - `dual`:  the gradient `[∂f/∂x_1, ..., ∂f/∂x_n]`
///
/// Note on storage: the tangent vector is a plain heap-backed `Vec<f64>`.
/// We evaluated a `SmallVec<[f64; N]>` implementation for several values
/// of `N` (see the CHANGELOG under stage 7 of the 2026-04 perf refactor);
/// it gave 40 – 58 % wins when the tangent fit inline but regressed
/// spilled workloads (e.g. the 30-input swap pricer) by 30 – 45 % because
/// SmallVec's spilled-mode discriminated-union layout bloats every heap
/// operand. A single inline cap that wins at both `n = 6` and `n = 30`
/// without blowing up register pressure does not exist, so we keep plain
/// `Vec` for predictable, uniform performance across tangent sizes.
#[derive(Clone, Debug)]
pub struct Dual {
    pub real: f64,
    pub dual: Vec<f64>,
}

#[inline]
fn len_check(a: &Dual, b: &Dual) {
    debug_assert_eq!(a.dual.len(), b.dual.len(), "Dual tangent length mismatch");
}

impl Dual {
    /// Create a `Dual` with the given value and gradient vector.
    #[inline]
    pub fn new(real: f64, dual: Vec<f64>) -> Self {
        Dual { real, dual }
    }

    /// Create a derivative-free `Dual` (all tangents zero).
    #[inline]
    pub fn constant(real: f64, n: usize) -> Self {
        Dual {
            real,
            dual: vec![0.0; n],
        }
    }

    /// Create the `i`-th active variable in an `n`-dimensional input space.
    ///
    /// The resulting `Dual` has `real = value` and `dual = e_i` (unit vector
    /// in direction `i`), so after propagation `result.dual[j] = ∂result/∂x_i`
    /// when `j = i` and zero from this seed alone.
    #[inline]
    pub fn variable(value: f64, i: usize, n: usize) -> Self {
        let mut dual = vec![0.0; n];
        dual[i] = 1.0;
        Dual { real: value, dual }
    }

    /// Value accessor.
    #[inline]
    pub fn real(&self) -> f64 {
        self.real
    }

    /// Full gradient as a slice view.
    #[inline]
    pub fn dual(&self) -> &[f64] {
        &self.dual
    }

    /// Partial derivative with respect to the `i`-th variable.
    #[inline]
    pub fn partial(&self, i: usize) -> f64 {
        self.dual[i]
    }

    /// Number of tracked variables (length of the gradient vector).
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.dual.len()
    }

    // ------------------------------------------------------------------
    // Math functions (first-order chain rule)
    // ------------------------------------------------------------------

    /// Internal: build a new tangent by scaling `self.dual` by `k` in a
    /// single fused pass.
    #[inline]
    fn scaled(&self, k: f64) -> Vec<f64> {
        self.dual.iter().map(|&d| d * k).collect()
    }

    /// `self^n` with scalar exponent.
    #[inline]
    pub fn powf(&self, n: f64) -> Dual {
        let vn = self.real.powf(n);
        let gp = n * self.real.powf(n - 1.0);
        Dual {
            real: vn,
            dual: self.scaled(gp),
        }
    }

    /// `self^n` with integer exponent.
    #[inline]
    pub fn powi(&self, n: i32) -> Dual {
        let vn = self.real.powi(n);
        let gp = (n as f64) * self.real.powi(n - 1);
        Dual {
            real: vn,
            dual: self.scaled(gp),
        }
    }

    /// Natural exponential.
    #[inline]
    pub fn exp(&self) -> Dual {
        let e = self.real.exp();
        Dual {
            real: e,
            dual: self.scaled(e),
        }
    }

    /// Natural logarithm.
    #[inline]
    pub fn ln(&self) -> Dual {
        let inv = 1.0 / self.real;
        Dual {
            real: self.real.ln(),
            dual: self.scaled(inv),
        }
    }

    /// Square root.
    #[inline]
    pub fn sqrt(&self) -> Dual {
        let s = self.real.sqrt();
        Dual {
            real: s,
            dual: self.scaled(0.5 / s),
        }
    }

    /// Sine.
    #[inline]
    pub fn sin(&self) -> Dual {
        Dual {
            real: self.real.sin(),
            dual: self.scaled(self.real.cos()),
        }
    }

    /// Cosine.
    #[inline]
    pub fn cos(&self) -> Dual {
        Dual {
            real: self.real.cos(),
            dual: self.scaled(-self.real.sin()),
        }
    }

    /// Tangent.
    #[inline]
    pub fn tan(&self) -> Dual {
        let t = self.real.tan();
        let sec2 = 1.0 + t * t;
        Dual {
            real: t,
            dual: self.scaled(sec2),
        }
    }

    /// Hyperbolic tangent.
    #[inline]
    pub fn tanh(&self) -> Dual {
        let t = self.real.tanh();
        Dual {
            real: t,
            dual: self.scaled(1.0 - t * t),
        }
    }

    /// Absolute value (sub-gradient at 0 is 0).
    #[inline]
    pub fn abs(&self) -> Dual {
        let sign = if self.real >= 0.0 { 1.0 } else { -1.0 };
        Dual {
            real: self.real.abs(),
            dual: self.scaled(sign),
        }
    }

    /// Error function `erf(self)`.
    ///
    /// Derivative: `erf'(x) = (2/√π) · exp(-x²)`.
    #[inline]
    pub fn erf(&self) -> Dual {
        let v = self.real;
        let r = crate::math::erf(v);
        // `FRAC_2_SQRT_PI` is a compile-time constant (2 / √π); the previous
        // expression `2.0 / PI.sqrt()` recomputed the square root on every
        // call because `f64::sqrt` is not const.
        let g = std::f64::consts::FRAC_2_SQRT_PI * (-v * v).exp();
        Dual {
            real: r,
            dual: self.scaled(g),
        }
    }

    /// Standard normal CDF: `Φ(x) = 0.5 · (1 + erf(x / √2))`.
    ///
    /// Derivative: `φ(x) = (1/√(2π)) · exp(-x²/2)` (the standard normal PDF).
    #[inline]
    pub fn norm_cdf(&self) -> Dual {
        let v = self.real;
        let r = crate::math::norm_cdf(v);
        let g = crate::math::norm_pdf(v);
        Dual {
            real: r,
            dual: self.scaled(g),
        }
    }

    /// Inverse standard normal CDF: `Φ⁻¹(p)`.
    ///
    /// Derivative: `1 / φ(Φ⁻¹(p)) = √(2π) · exp(Φ⁻¹(p)² / 2)`.
    #[inline]
    pub fn inv_norm_cdf(&self) -> Dual {
        let v = self.real;
        let r = crate::math::inv_norm_cdf(v);
        let g = 1.0 / crate::math::norm_pdf(r);
        Dual {
            real: r,
            dual: self.scaled(g),
        }
    }
}

// ============================================================================
// Operator implementations
// ============================================================================
//
// `&Dual op &Dual` builds a fresh tangent via a single fused loop.
// Owned forms reuse the left operand's allocation in place whenever possible,
// avoiding a heap allocation on the hot path.

// --- Add ---
impl<'a, 'b> Add<&'b Dual> for &'a Dual {
    type Output = Dual;
    #[inline]
    fn add(self, rhs: &'b Dual) -> Dual {
        len_check(self, rhs);
        Dual {
            real: self.real + rhs.real,
            dual: self
                .dual
                .iter()
                .zip(&rhs.dual)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

impl Add for Dual {
    type Output = Dual;
    #[inline]
    fn add(mut self, rhs: Dual) -> Dual {
        self += &rhs;
        self
    }
}

impl Add<&Dual> for Dual {
    type Output = Dual;
    #[inline]
    fn add(mut self, rhs: &Dual) -> Dual {
        self += rhs;
        self
    }
}

impl Add<Dual> for &Dual {
    type Output = Dual;
    #[inline]
    fn add(self, mut rhs: Dual) -> Dual {
        rhs += self;
        rhs
    }
}

impl Add<f64> for &Dual {
    type Output = Dual;
    #[inline]
    fn add(self, rhs: f64) -> Dual {
        Dual {
            real: self.real + rhs,
            dual: self.dual.clone(),
        }
    }
}

impl Add<f64> for Dual {
    type Output = Dual;
    #[inline]
    fn add(mut self, rhs: f64) -> Dual {
        self.real += rhs;
        self
    }
}

impl Add<&Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn add(self, rhs: &Dual) -> Dual {
        rhs + self
    }
}

impl Add<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn add(self, rhs: Dual) -> Dual {
        rhs + self
    }
}

// --- Sub ---
impl<'a, 'b> Sub<&'b Dual> for &'a Dual {
    type Output = Dual;
    #[inline]
    fn sub(self, rhs: &'b Dual) -> Dual {
        len_check(self, rhs);
        Dual {
            real: self.real - rhs.real,
            dual: self
                .dual
                .iter()
                .zip(&rhs.dual)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}

impl Sub for Dual {
    type Output = Dual;
    #[inline]
    fn sub(mut self, rhs: Dual) -> Dual {
        self -= &rhs;
        self
    }
}

impl Sub<&Dual> for Dual {
    type Output = Dual;
    #[inline]
    fn sub(mut self, rhs: &Dual) -> Dual {
        self -= rhs;
        self
    }
}

impl Sub<Dual> for &Dual {
    type Output = Dual;
    #[inline]
    fn sub(self, mut rhs: Dual) -> Dual {
        // Reuse rhs's allocation: compute (self - rhs) in-place.
        len_check(self, &rhs);
        rhs.real = self.real - rhs.real;
        for (b, &a) in rhs.dual.iter_mut().zip(&self.dual) {
            *b = a - *b;
        }
        rhs
    }
}

impl Sub<f64> for &Dual {
    type Output = Dual;
    #[inline]
    fn sub(self, rhs: f64) -> Dual {
        Dual {
            real: self.real - rhs,
            dual: self.dual.clone(),
        }
    }
}

impl Sub<f64> for Dual {
    type Output = Dual;
    #[inline]
    fn sub(mut self, rhs: f64) -> Dual {
        self.real -= rhs;
        self
    }
}

impl Sub<&Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn sub(self, rhs: &Dual) -> Dual {
        Dual {
            real: self - rhs.real,
            dual: rhs.dual.iter().map(|&d| -d).collect(),
        }
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn sub(self, mut rhs: Dual) -> Dual {
        rhs.real = self - rhs.real;
        for d in rhs.dual.iter_mut() {
            *d = -*d;
        }
        rhs
    }
}

// --- Mul ---
// d(a*b) = a.dual * b.real + b.dual * a.real
impl<'a, 'b> Mul<&'b Dual> for &'a Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, rhs: &'b Dual) -> Dual {
        len_check(self, rhs);
        let a = self.real;
        let b = rhs.real;
        Dual {
            real: a * b,
            dual: self
                .dual
                .iter()
                .zip(&rhs.dual)
                .map(|(&da, &db)| da * b + db * a)
                .collect(),
        }
    }
}

impl Mul for Dual {
    type Output = Dual;
    #[inline]
    fn mul(mut self, rhs: Dual) -> Dual {
        len_check(&self, &rhs);
        let a = self.real;
        let b = rhs.real;
        self.real = a * b;
        for (da, &db) in self.dual.iter_mut().zip(&rhs.dual) {
            *da = *da * b + db * a;
        }
        self
    }
}

impl Mul<&Dual> for Dual {
    type Output = Dual;
    #[inline]
    fn mul(mut self, rhs: &Dual) -> Dual {
        len_check(&self, rhs);
        let a = self.real;
        let b = rhs.real;
        self.real = a * b;
        for (da, &db) in self.dual.iter_mut().zip(&rhs.dual) {
            *da = *da * b + db * a;
        }
        self
    }
}

impl Mul<Dual> for &Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, mut rhs: Dual) -> Dual {
        len_check(self, &rhs);
        let a = self.real;
        let b = rhs.real;
        rhs.real = a * b;
        for (db, &da) in rhs.dual.iter_mut().zip(&self.dual) {
            *db = da * b + *db * a;
        }
        rhs
    }
}

impl Mul<f64> for &Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, rhs: f64) -> Dual {
        Dual {
            real: self.real * rhs,
            dual: self.dual.iter().map(|&d| d * rhs).collect(),
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;
    #[inline]
    fn mul(mut self, rhs: f64) -> Dual {
        self.real *= rhs;
        for d in self.dual.iter_mut() {
            *d *= rhs;
        }
        self
    }
}

impl Mul<&Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn mul(self, rhs: &Dual) -> Dual {
        rhs * self
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn mul(self, rhs: Dual) -> Dual {
        rhs * self
    }
}

// --- Div ---
// d(a/b) = (a.dual * b.real - b.dual * a.real) / b.real^2
impl<'a, 'b> Div<&'b Dual> for &'a Dual {
    type Output = Dual;
    #[inline]
    fn div(self, rhs: &'b Dual) -> Dual {
        len_check(self, rhs);
        let inv = 1.0 / rhs.real;
        let inv2 = inv * inv;
        let a_inv2 = self.real * inv2;
        Dual {
            real: self.real * inv,
            dual: self
                .dual
                .iter()
                .zip(&rhs.dual)
                .map(|(&da, &db)| da * inv - db * a_inv2)
                .collect(),
        }
    }
}

impl Div for Dual {
    type Output = Dual;
    #[inline]
    fn div(mut self, rhs: Dual) -> Dual {
        len_check(&self, &rhs);
        let inv = 1.0 / rhs.real;
        let inv2 = inv * inv;
        let a_inv2 = self.real * inv2;
        self.real *= inv;
        for (da, &db) in self.dual.iter_mut().zip(&rhs.dual) {
            *da = *da * inv - db * a_inv2;
        }
        self
    }
}

impl Div<&Dual> for Dual {
    type Output = Dual;
    #[inline]
    fn div(mut self, rhs: &Dual) -> Dual {
        len_check(&self, rhs);
        let inv = 1.0 / rhs.real;
        let inv2 = inv * inv;
        let a_inv2 = self.real * inv2;
        self.real *= inv;
        for (da, &db) in self.dual.iter_mut().zip(&rhs.dual) {
            *da = *da * inv - db * a_inv2;
        }
        self
    }
}

impl Div<Dual> for &Dual {
    type Output = Dual;
    #[inline]
    fn div(self, rhs: Dual) -> Dual {
        // rhs's storage is not easily reusable for this op (both tangents
        // combine linearly with non-unit coefficients), so fall back to
        // the ref/ref path.
        self / &rhs
    }
}

impl Div<f64> for &Dual {
    type Output = Dual;
    #[inline]
    fn div(self, rhs: f64) -> Dual {
        let inv = 1.0 / rhs;
        Dual {
            real: self.real * inv,
            dual: self.dual.iter().map(|&d| d * inv).collect(),
        }
    }
}

impl Div<f64> for Dual {
    type Output = Dual;
    #[inline]
    fn div(mut self, rhs: f64) -> Dual {
        let inv = 1.0 / rhs;
        self.real *= inv;
        for d in self.dual.iter_mut() {
            *d *= inv;
        }
        self
    }
}

impl Div<&Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn div(self, rhs: &Dual) -> Dual {
        let inv = 1.0 / rhs.real;
        let neg_scale = -self * inv * inv;
        Dual {
            real: self * inv,
            dual: rhs.dual.iter().map(|&d| d * neg_scale).collect(),
        }
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn div(self, mut rhs: Dual) -> Dual {
        let inv = 1.0 / rhs.real;
        let neg_scale = -self * inv * inv;
        rhs.real = self * inv;
        for d in rhs.dual.iter_mut() {
            *d *= neg_scale;
        }
        rhs
    }
}

// --- Neg ---
impl Neg for &Dual {
    type Output = Dual;
    #[inline]
    fn neg(self) -> Dual {
        Dual {
            real: -self.real,
            dual: self.dual.iter().map(|&d| -d).collect(),
        }
    }
}

impl Neg for Dual {
    type Output = Dual;
    #[inline]
    fn neg(mut self) -> Dual {
        self.real = -self.real;
        for d in self.dual.iter_mut() {
            *d = -*d;
        }
        self
    }
}

// --- Compound assignment ---
impl AddAssign<&Dual> for Dual {
    #[inline]
    fn add_assign(&mut self, rhs: &Dual) {
        len_check(self, rhs);
        self.real += rhs.real;
        for (a, &b) in self.dual.iter_mut().zip(&rhs.dual) {
            *a += b;
        }
    }
}

impl AddAssign for Dual {
    #[inline]
    fn add_assign(&mut self, rhs: Dual) {
        len_check(self, &rhs);
        self.real += rhs.real;
        for (a, &b) in self.dual.iter_mut().zip(&rhs.dual) {
            *a += b;
        }
    }
}

impl AddAssign<f64> for Dual {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        self.real += rhs;
    }
}

impl SubAssign<&Dual> for Dual {
    #[inline]
    fn sub_assign(&mut self, rhs: &Dual) {
        len_check(self, rhs);
        self.real -= rhs.real;
        for (a, &b) in self.dual.iter_mut().zip(&rhs.dual) {
            *a -= b;
        }
    }
}

impl SubAssign for Dual {
    #[inline]
    fn sub_assign(&mut self, rhs: Dual) {
        len_check(self, &rhs);
        self.real -= rhs.real;
        for (a, &b) in self.dual.iter_mut().zip(&rhs.dual) {
            *a -= b;
        }
    }
}

impl SubAssign<f64> for Dual {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        self.real -= rhs;
    }
}

impl MulAssign<&Dual> for Dual {
    #[inline]
    fn mul_assign(&mut self, rhs: &Dual) {
        len_check(self, rhs);
        let a = self.real;
        let b = rhs.real;
        self.real = a * b;
        for (da, &db) in self.dual.iter_mut().zip(&rhs.dual) {
            *da = *da * b + db * a;
        }
    }
}

impl MulAssign<f64> for Dual {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.real *= rhs;
        for d in self.dual.iter_mut() {
            *d *= rhs;
        }
    }
}

impl DivAssign<&Dual> for Dual {
    #[inline]
    fn div_assign(&mut self, rhs: &Dual) {
        len_check(self, rhs);
        let inv = 1.0 / rhs.real;
        let inv2 = inv * inv;
        let a_inv2 = self.real * inv2;
        self.real *= inv;
        for (da, &db) in self.dual.iter_mut().zip(&rhs.dual) {
            *da = *da * inv - db * a_inv2;
        }
    }
}

impl DivAssign<f64> for Dual {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        let inv = 1.0 / rhs;
        self.real *= inv;
        for d in self.dual.iter_mut() {
            *d *= inv;
        }
    }
}

// --- PartialEq / PartialOrd (value-only) ---
impl PartialEq for Dual {
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real
    }
}

impl PartialEq<f64> for Dual {
    fn eq(&self, other: &f64) -> bool {
        self.real == *other
    }
}

impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl PartialOrd<f64> for Dual {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(other)
    }
}

// --- Display ---
impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.real)
    }
}

// ============================================================================
// NamedDual — named wrapper over `Dual` (f64-only forward-mode)
// ============================================================================
//
// **Shape A (Phase 02.2):** does NOT carry an `Arc<VarRegistry>` field in
// release builds. The struct layout is a single `inner: Dual` field plus,
// under `#[cfg(debug_assertions)]` only, a `gen_id: u64` stamped by the
// owning `NamedForwardTape` scope for the cross-registry debug guard.
// Release builds are bit-for-bit equivalent to a pure `Dual` wrapper and
// carry zero atomic-refcount cost per operator.
//
// The only way to obtain a `NamedDual` is via the `NamedForwardTape`
// constructor API (see `NamedForwardTape::declare_dual` /
// `NamedForwardTape::freeze_dual`).

/// Labeled wrapper around the positional [`Dual`] type.
///
/// # Example
///
/// The constructor API for `NamedDual` is owned by `NamedForwardTape`
/// via the `declare_dual` / `freeze_dual` / `scope.dual(handle)` pattern.
/// See the module docs in `src/forward_tape.rs` for the full rationale.
///
/// ```
/// use xad_rs::{NamedForwardTape, NamedForwardScope};
///
/// let mut ft = NamedForwardTape::new();
/// let x_h = ft.declare_dual("x", 2.0);
/// let y_h = ft.declare_dual("y", 3.0);
/// let scope: NamedForwardScope = ft.freeze_dual();
///
/// let x = scope.dual(x_h);
/// let y = scope.dual(y_h);
/// let f = x * y + x - 2.0 * y;
///
/// assert_eq!(f.real(), 2.0);
/// assert_eq!(f.partial("x"), 4.0);
/// assert_eq!(f.partial("y"), 0.0);
/// ```
#[derive(Clone, Debug)]
pub struct NamedDual {
    pub(crate) inner: Dual,
    // NOTE: field name is `gen_id` — `gen` alone is a reserved keyword in
    // Rust 2024 edition. The D-01/D-02 CONTEXT blocks spell it as `gen`;
    // we carry the spelling adjustment forward to satisfy the compiler.
    #[cfg(debug_assertions)]
    pub(crate) gen_id: u64,
}

impl NamedDual {
    /// Internal constructor used by `NamedForwardTape` input / freeze
    /// paths. Reads the TLS active generation (debug builds only) to stamp
    /// the `gen_id` field. Not part of the public API.
    #[inline]
    pub(crate) fn __from_inner(inner: Dual) -> Self {
        Self {
            inner,
            #[cfg(debug_assertions)]
            gen_id: crate::forward_tape::current_gen(),
        }
    }

    /// Value part.
    #[inline]
    pub fn real(&self) -> f64 {
        self.inner.real
    }

    /// Partial derivative with respect to a named variable.
    ///
    /// Reads the active registry from the `NamedForwardTape` thread-local
    /// slot (stamped at `freeze_dual()` time). Returns `0.0` if `name` is
    /// in the registry but not touched by this value (same as positional
    /// `Dual::partial` on an unused slot). Panics if `name` is not in the
    /// registry, or if called outside a frozen `NamedForwardTape` scope.
    pub fn partial(&self, name: &str) -> f64 {
        let idx = crate::forward_tape::with_active_registry(|r| {
            let r =
                r.expect("NamedDual::partial called outside a frozen NamedForwardTape scope");
            r.index_of(name).unwrap_or_else(|| {
                panic!(
                    "NamedDual::partial: name {:?} not present in registry",
                    name
                )
            })
        });
        self.inner.partial(idx)
    }

    /// Full gradient as a `Vec<(name, partial)>`.
    ///
    /// Iteration order matches the active registry's insertion order —
    /// deterministic. Reads the active registry from the thread-local slot.
    /// Panics if called outside a frozen `NamedForwardTape` scope.
    pub fn gradient(&self) -> Vec<(String, f64)> {
        crate::forward_tape::with_active_registry(|r| {
            let r =
                r.expect("NamedDual::gradient called outside a frozen NamedForwardTape scope");
            let n = r.len();
            let mut out = Vec::with_capacity(n);
            for (i, name) in r.iter().enumerate() {
                out.push((name.to_string(), self.inner.partial(i)));
            }
            out
        })
    }

    /// Escape hatch: direct access to the inner positional `Dual`.
    #[inline]
    pub fn inner(&self) -> &Dual {
        &self.inner
    }

    // ============ Elementary math delegations ============
    // Each method forwards to the inherent `Dual` elementary and preserves
    // the parent's generation stamp explicitly (debug builds) — no TLS read
    // on the hot path.

    /// Natural exponential, preserving the parent scope's generation.
    #[inline]
    pub fn exp(&self) -> Self {
        Self {
            inner: self.inner.exp(),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Natural logarithm, preserving the parent scope's generation.
    #[inline]
    pub fn ln(&self) -> Self {
        Self {
            inner: self.inner.ln(),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Square root, preserving the parent scope's generation.
    #[inline]
    pub fn sqrt(&self) -> Self {
        Self {
            inner: self.inner.sqrt(),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Sine, preserving the parent scope's generation.
    #[inline]
    pub fn sin(&self) -> Self {
        Self {
            inner: self.inner.sin(),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Cosine, preserving the parent scope's generation.
    #[inline]
    pub fn cos(&self) -> Self {
        Self {
            inner: self.inner.cos(),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Tangent, preserving the parent scope's generation.
    #[inline]
    pub fn tan(&self) -> Self {
        Self {
            inner: self.inner.tan(),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Standard normal CDF, preserving the parent scope's generation.
    #[inline]
    pub fn norm_cdf(&self) -> Self {
        Self {
            inner: self.inner.norm_cdf(),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Inverse standard normal CDF, preserving the parent scope's generation.
    #[inline]
    pub fn inv_norm_cdf(&self) -> Self {
        Self {
            inner: self.inner.inv_norm_cdf(),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// `self^n` with scalar exponent, preserving the parent scope's generation.
    #[inline]
    pub fn powf(&self, n: f64) -> Self {
        Self {
            inner: self.inner.powf(n),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// `self^n` with integer exponent, preserving the parent scope's generation.
    #[inline]
    pub fn powi(&self, n: i32) -> Self {
        Self {
            inner: self.inner.powi(n),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }
}

impl fmt::Display for NamedDual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NamedDual({})", self.inner.real)
    }
}

// ============ Operator overloads — hand-written, Shape A ============
// No shared op-stamping macro is used: Shape A does not carry an
// `Arc` registry field on the per-value wrapper. The four reference
// variants (owned/owned, ref/ref, owned/ref, ref/owned) plus scalar-RHS
// variants are stamped explicitly via a local `__named_dual_binop!`
// macro, modelled on `__named_freal_binop!` in `src/named/freal.rs`
// but specialised for non-generic `NamedDual` with `f64` as the
// scalar RHS. Each wrapper-vs-wrapper impl performs a debug-only
// `check_gen` between the two operands' generations, then constructs
// the result preserving the LHS's generation stamp.

macro_rules! __named_dual_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl ::core::ops::$trait<NamedDual> for NamedDual {
            type Output = NamedDual;
            #[inline]
            fn $method(self, rhs: NamedDual) -> NamedDual {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedDual {
                    inner: self.inner $op rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl ::core::ops::$trait<&NamedDual> for &NamedDual {
            type Output = NamedDual;
            #[inline]
            fn $method(self, rhs: &NamedDual) -> NamedDual {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedDual {
                    inner: &self.inner $op &rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl ::core::ops::$trait<&NamedDual> for NamedDual {
            type Output = NamedDual;
            #[inline]
            fn $method(self, rhs: &NamedDual) -> NamedDual {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedDual {
                    inner: self.inner $op &rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl ::core::ops::$trait<NamedDual> for &NamedDual {
            type Output = NamedDual;
            #[inline]
            fn $method(self, rhs: NamedDual) -> NamedDual {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedDual {
                    inner: &self.inner $op rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl ::core::ops::$trait<f64> for NamedDual {
            type Output = NamedDual;
            #[inline]
            fn $method(self, rhs: f64) -> NamedDual {
                NamedDual {
                    inner: self.inner $op rhs,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl ::core::ops::$trait<f64> for &NamedDual {
            type Output = NamedDual;
            #[inline]
            fn $method(self, rhs: f64) -> NamedDual {
                NamedDual {
                    inner: &self.inner $op rhs,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
    };
}

__named_dual_binop!(Add, add, +);
__named_dual_binop!(Sub, sub, -);
__named_dual_binop!(Mul, mul, *);
__named_dual_binop!(Div, div, /);

impl ::core::ops::AddAssign<NamedDual> for NamedDual {
    #[inline]
    fn add_assign(&mut self, rhs: NamedDual) {
        #[cfg(debug_assertions)]
        crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
        self.inner += rhs.inner;
    }
}
impl ::core::ops::AddAssign<&NamedDual> for NamedDual {
    #[inline]
    fn add_assign(&mut self, rhs: &NamedDual) {
        #[cfg(debug_assertions)]
        crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
        self.inner += &rhs.inner;
    }
}

impl ::core::ops::Neg for NamedDual {
    type Output = NamedDual;
    #[inline]
    fn neg(self) -> NamedDual {
        NamedDual {
            inner: -self.inner,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }
}
impl ::core::ops::Neg for &NamedDual {
    type Output = NamedDual;
    #[inline]
    fn neg(self) -> NamedDual {
        NamedDual {
            inner: -&self.inner,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }
}

// ============ Scalar-on-LHS hand-written impls (orphan-rule escape) ============
// The inner `Dual` type supports the full `f64 op Dual` + `f64 op &Dual`
// surface (see above), so delegation is direct. Each result preserves
// the RHS's generation stamp (debug builds only).

// Add
impl ::core::ops::Add<NamedDual> for f64 {
    type Output = NamedDual;
    #[inline]
    fn add(self, rhs: NamedDual) -> NamedDual {
        NamedDual {
            inner: self + rhs.inner,
            #[cfg(debug_assertions)]
            gen_id: rhs.gen_id,
        }
    }
}
impl ::core::ops::Add<&NamedDual> for f64 {
    type Output = NamedDual;
    #[inline]
    fn add(self, rhs: &NamedDual) -> NamedDual {
        NamedDual {
            inner: self + &rhs.inner,
            #[cfg(debug_assertions)]
            gen_id: rhs.gen_id,
        }
    }
}

// Sub
impl ::core::ops::Sub<NamedDual> for f64 {
    type Output = NamedDual;
    #[inline]
    fn sub(self, rhs: NamedDual) -> NamedDual {
        NamedDual {
            inner: self - rhs.inner,
            #[cfg(debug_assertions)]
            gen_id: rhs.gen_id,
        }
    }
}
impl ::core::ops::Sub<&NamedDual> for f64 {
    type Output = NamedDual;
    #[inline]
    fn sub(self, rhs: &NamedDual) -> NamedDual {
        NamedDual {
            inner: self - &rhs.inner,
            #[cfg(debug_assertions)]
            gen_id: rhs.gen_id,
        }
    }
}

// Mul
impl ::core::ops::Mul<NamedDual> for f64 {
    type Output = NamedDual;
    #[inline]
    fn mul(self, rhs: NamedDual) -> NamedDual {
        NamedDual {
            inner: self * rhs.inner,
            #[cfg(debug_assertions)]
            gen_id: rhs.gen_id,
        }
    }
}
impl ::core::ops::Mul<&NamedDual> for f64 {
    type Output = NamedDual;
    #[inline]
    fn mul(self, rhs: &NamedDual) -> NamedDual {
        NamedDual {
            inner: self * &rhs.inner,
            #[cfg(debug_assertions)]
            gen_id: rhs.gen_id,
        }
    }
}

// Div
impl ::core::ops::Div<NamedDual> for f64 {
    type Output = NamedDual;
    #[inline]
    fn div(self, rhs: NamedDual) -> NamedDual {
        NamedDual {
            inner: self / rhs.inner,
            #[cfg(debug_assertions)]
            gen_id: rhs.gen_id,
        }
    }
}
impl ::core::ops::Div<&NamedDual> for f64 {
    type Output = NamedDual;
    #[inline]
    fn div(self, rhs: &NamedDual) -> NamedDual {
        NamedDual {
            inner: self / &rhs.inner,
            #[cfg(debug_assertions)]
            gen_id: rhs.gen_id,
        }
    }
}
