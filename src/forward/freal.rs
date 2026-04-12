//! `FReal` — active real for single-variable forward-mode AD.
//!
//! An [`FReal<T>`] carries both a value and a single scalar tangent
//! (the derivative with respect to one seeded input direction).
//! Propagation uses the chain rule applied pointwise at every operator;
//! there is no tape, no thread-local state, and the type is entirely
//! stack-allocated and `Copy`-able modulo the inner scalar's `Clone`.
//!
//! Forward single-variable mode is the cheapest AD mode in this crate —
//! operators compile to a handful of fmul/fadds — so it's the right
//! choice when you have **one input direction** (seed `derivative = 1`
//! on exactly one variable) and you don't need the multi-direction
//! batching that [`Dual`](crate::dual::Dual) provides.
//!
//! ```
//! use xad_rs::FReal;
//!
//! // f(x) = x² + 3x, at x = 2
//! // f'(x) = 2x + 3 = 7
//! let x = FReal::new(2.0_f64, 1.0);   // value 2, tangent 1
//! let f = &x * &x + &x * 3.0;
//! assert_eq!(f.value(), 10.0);
//! assert_eq!(f.derivative(), 7.0);
//! ```
//!
//! For **many** directions in a single forward pass, use
//! [`Dual`](crate::dual::Dual) instead. For reverse mode, use
//! [`AReal`](crate::areal::AReal).

use crate::scalar::Scalar;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Active real number for forward-mode AD.
///
/// Stores a value and its tangent (derivative with respect to a seed direction).
#[derive(Clone)]
pub struct FReal<T: Scalar> {
    value: T,
    derivative: T,
}

impl<T: Scalar> FReal<T> {
    /// Create a new FReal with the given value and derivative.
    pub fn new(value: T, derivative: T) -> Self {
        FReal { value, derivative }
    }

    /// Create an FReal with zero derivative (a constant).
    pub fn constant(value: T) -> Self {
        FReal {
            value,
            derivative: T::zero(),
        }
    }

    /// Get the underlying value.
    #[inline]
    pub fn value(&self) -> T {
        self.value
    }

    /// Set the underlying value.
    #[inline]
    pub fn set_value(&mut self, v: T) {
        self.value = v;
    }

    /// Get the derivative (tangent).
    #[inline]
    pub fn derivative(&self) -> T {
        self.derivative
    }

    /// Set the derivative (tangent).
    #[inline]
    pub fn set_derivative(&mut self, d: T) {
        self.derivative = d;
    }
}

impl<T: Scalar> From<T> for FReal<T> {
    fn from(value: T) -> Self {
        FReal::constant(value)
    }
}

impl From<i32> for FReal<f64> {
    fn from(value: i32) -> Self {
        FReal::constant(value as f64)
    }
}

impl From<i32> for FReal<f32> {
    fn from(value: i32) -> Self {
        FReal::constant(value as f32)
    }
}

// FReal + FReal
impl<T: Scalar> Add for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn add(self, rhs: FReal<T>) -> FReal<T> {
        FReal {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        }
    }
}

impl<T: Scalar> Add for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn add(self, rhs: &FReal<T>) -> FReal<T> {
        FReal {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        }
    }
}

impl<T: Scalar> Add<&FReal<T>> for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn add(self, rhs: &FReal<T>) -> FReal<T> {
        FReal {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        }
    }
}

impl<T: Scalar> Add<FReal<T>> for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn add(self, rhs: FReal<T>) -> FReal<T> {
        FReal {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        }
    }
}

// FReal + scalar
impl<T: Scalar> Add<T> for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn add(self, rhs: T) -> FReal<T> {
        FReal {
            value: self.value + rhs,
            derivative: self.derivative,
        }
    }
}

impl<T: Scalar> Add<T> for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn add(self, rhs: T) -> FReal<T> {
        FReal {
            value: self.value + rhs,
            derivative: self.derivative,
        }
    }
}

// scalar + FReal
impl Add<FReal<f64>> for f64 {
    type Output = FReal<f64>;
    #[inline]
    fn add(self, rhs: FReal<f64>) -> FReal<f64> {
        FReal {
            value: self + rhs.value,
            derivative: rhs.derivative,
        }
    }
}

impl Add<&FReal<f64>> for f64 {
    type Output = FReal<f64>;
    #[inline]
    fn add(self, rhs: &FReal<f64>) -> FReal<f64> {
        FReal {
            value: self + rhs.value,
            derivative: rhs.derivative,
        }
    }
}

impl Add<FReal<f32>> for f32 {
    type Output = FReal<f32>;
    #[inline]
    fn add(self, rhs: FReal<f32>) -> FReal<f32> {
        FReal {
            value: self + rhs.value,
            derivative: rhs.derivative,
        }
    }
}

impl Add<&FReal<f32>> for f32 {
    type Output = FReal<f32>;
    #[inline]
    fn add(self, rhs: &FReal<f32>) -> FReal<f32> {
        FReal {
            value: self + rhs.value,
            derivative: rhs.derivative,
        }
    }
}

// FReal - FReal
impl<T: Scalar> Sub for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn sub(self, rhs: FReal<T>) -> FReal<T> {
        FReal {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.derivative,
        }
    }
}

impl<T: Scalar> Sub for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn sub(self, rhs: &FReal<T>) -> FReal<T> {
        FReal {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.derivative,
        }
    }
}

impl<T: Scalar> Sub<&FReal<T>> for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn sub(self, rhs: &FReal<T>) -> FReal<T> {
        FReal {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.derivative,
        }
    }
}

impl<T: Scalar> Sub<FReal<T>> for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn sub(self, rhs: FReal<T>) -> FReal<T> {
        FReal {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.derivative,
        }
    }
}

// FReal - scalar
impl<T: Scalar> Sub<T> for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn sub(self, rhs: T) -> FReal<T> {
        FReal {
            value: self.value - rhs,
            derivative: self.derivative,
        }
    }
}

impl<T: Scalar> Sub<T> for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn sub(self, rhs: T) -> FReal<T> {
        FReal {
            value: self.value - rhs,
            derivative: self.derivative,
        }
    }
}

// scalar - FReal
impl Sub<FReal<f64>> for f64 {
    type Output = FReal<f64>;
    #[inline]
    fn sub(self, rhs: FReal<f64>) -> FReal<f64> {
        FReal {
            value: self - rhs.value,
            derivative: -rhs.derivative,
        }
    }
}

impl Sub<&FReal<f64>> for f64 {
    type Output = FReal<f64>;
    #[inline]
    fn sub(self, rhs: &FReal<f64>) -> FReal<f64> {
        FReal {
            value: self - rhs.value,
            derivative: -rhs.derivative,
        }
    }
}

impl Sub<FReal<f32>> for f32 {
    type Output = FReal<f32>;
    #[inline]
    fn sub(self, rhs: FReal<f32>) -> FReal<f32> {
        FReal {
            value: self - rhs.value,
            derivative: -rhs.derivative,
        }
    }
}

impl Sub<&FReal<f32>> for f32 {
    type Output = FReal<f32>;
    #[inline]
    fn sub(self, rhs: &FReal<f32>) -> FReal<f32> {
        FReal {
            value: self - rhs.value,
            derivative: -rhs.derivative,
        }
    }
}

// FReal * FReal
impl<T: Scalar> Mul for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn mul(self, rhs: FReal<T>) -> FReal<T> {
        // d(a*b) = a'*b + a*b'
        FReal {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
        }
    }
}

impl<T: Scalar> Mul for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn mul(self, rhs: &FReal<T>) -> FReal<T> {
        FReal {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
        }
    }
}

impl<T: Scalar> Mul<&FReal<T>> for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn mul(self, rhs: &FReal<T>) -> FReal<T> {
        FReal {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
        }
    }
}

impl<T: Scalar> Mul<FReal<T>> for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn mul(self, rhs: FReal<T>) -> FReal<T> {
        FReal {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
        }
    }
}

// FReal * scalar
impl<T: Scalar> Mul<T> for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn mul(self, rhs: T) -> FReal<T> {
        FReal {
            value: self.value * rhs,
            derivative: self.derivative * rhs,
        }
    }
}

impl<T: Scalar> Mul<T> for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn mul(self, rhs: T) -> FReal<T> {
        FReal {
            value: self.value * rhs,
            derivative: self.derivative * rhs,
        }
    }
}

// scalar * FReal
impl Mul<FReal<f64>> for f64 {
    type Output = FReal<f64>;
    #[inline]
    fn mul(self, rhs: FReal<f64>) -> FReal<f64> {
        FReal {
            value: self * rhs.value,
            derivative: self * rhs.derivative,
        }
    }
}

impl Mul<&FReal<f64>> for f64 {
    type Output = FReal<f64>;
    #[inline]
    fn mul(self, rhs: &FReal<f64>) -> FReal<f64> {
        FReal {
            value: self * rhs.value,
            derivative: self * rhs.derivative,
        }
    }
}

impl Mul<FReal<f32>> for f32 {
    type Output = FReal<f32>;
    #[inline]
    fn mul(self, rhs: FReal<f32>) -> FReal<f32> {
        FReal {
            value: self * rhs.value,
            derivative: self * rhs.derivative,
        }
    }
}

impl Mul<&FReal<f32>> for f32 {
    type Output = FReal<f32>;
    #[inline]
    fn mul(self, rhs: &FReal<f32>) -> FReal<f32> {
        FReal {
            value: self * rhs.value,
            derivative: self * rhs.derivative,
        }
    }
}

// FReal / FReal
impl<T: Scalar> Div for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn div(self, rhs: FReal<T>) -> FReal<T> {
        // d(a/b) = (a'*b - a*b') / b^2
        let inv_b = T::one() / rhs.value;
        FReal {
            value: self.value * inv_b,
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative) * inv_b * inv_b,
        }
    }
}

impl<T: Scalar> Div for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn div(self, rhs: &FReal<T>) -> FReal<T> {
        let inv_b = T::one() / rhs.value;
        FReal {
            value: self.value * inv_b,
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative) * inv_b * inv_b,
        }
    }
}

impl<T: Scalar> Div<&FReal<T>> for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn div(self, rhs: &FReal<T>) -> FReal<T> {
        let inv_b = T::one() / rhs.value;
        FReal {
            value: self.value * inv_b,
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative) * inv_b * inv_b,
        }
    }
}

impl<T: Scalar> Div<FReal<T>> for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn div(self, rhs: FReal<T>) -> FReal<T> {
        let inv_b = T::one() / rhs.value;
        FReal {
            value: self.value * inv_b,
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative) * inv_b * inv_b,
        }
    }
}

// FReal / scalar
impl<T: Scalar> Div<T> for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn div(self, rhs: T) -> FReal<T> {
        let inv = T::one() / rhs;
        FReal {
            value: self.value * inv,
            derivative: self.derivative * inv,
        }
    }
}

impl<T: Scalar> Div<T> for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn div(self, rhs: T) -> FReal<T> {
        let inv = T::one() / rhs;
        FReal {
            value: self.value * inv,
            derivative: self.derivative * inv,
        }
    }
}

// scalar / FReal
impl Div<FReal<f64>> for f64 {
    type Output = FReal<f64>;
    #[inline]
    fn div(self, rhs: FReal<f64>) -> FReal<f64> {
        let inv = 1.0 / rhs.value;
        FReal {
            value: self * inv,
            derivative: -self * rhs.derivative * inv * inv,
        }
    }
}

impl Div<&FReal<f64>> for f64 {
    type Output = FReal<f64>;
    #[inline]
    fn div(self, rhs: &FReal<f64>) -> FReal<f64> {
        let inv = 1.0 / rhs.value;
        FReal {
            value: self * inv,
            derivative: -self * rhs.derivative * inv * inv,
        }
    }
}

impl Div<FReal<f32>> for f32 {
    type Output = FReal<f32>;
    #[inline]
    fn div(self, rhs: FReal<f32>) -> FReal<f32> {
        let inv = 1.0 / rhs.value;
        FReal {
            value: self * inv,
            derivative: -self * rhs.derivative * inv * inv,
        }
    }
}

impl Div<&FReal<f32>> for f32 {
    type Output = FReal<f32>;
    #[inline]
    fn div(self, rhs: &FReal<f32>) -> FReal<f32> {
        let inv = 1.0 / rhs.value;
        FReal {
            value: self * inv,
            derivative: -self * rhs.derivative * inv * inv,
        }
    }
}

// Negation
impl<T: Scalar> Neg for FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn neg(self) -> FReal<T> {
        FReal {
            value: -self.value,
            derivative: -self.derivative,
        }
    }
}

impl<T: Scalar> Neg for &FReal<T> {
    type Output = FReal<T>;
    #[inline]
    fn neg(self) -> FReal<T> {
        FReal {
            value: -self.value,
            derivative: -self.derivative,
        }
    }
}

// Compound assignment
impl<T: Scalar> AddAssign for FReal<T> {
    fn add_assign(&mut self, rhs: FReal<T>) {
        self.value = self.value + rhs.value;
        self.derivative = self.derivative + rhs.derivative;
    }
}

impl<T: Scalar> AddAssign<&FReal<T>> for FReal<T> {
    fn add_assign(&mut self, rhs: &FReal<T>) {
        self.value = self.value + rhs.value;
        self.derivative = self.derivative + rhs.derivative;
    }
}

impl<T: Scalar> AddAssign<T> for FReal<T> {
    fn add_assign(&mut self, rhs: T) {
        self.value = self.value + rhs;
    }
}

impl<T: Scalar> SubAssign for FReal<T> {
    fn sub_assign(&mut self, rhs: FReal<T>) {
        self.value = self.value - rhs.value;
        self.derivative = self.derivative - rhs.derivative;
    }
}

impl<T: Scalar> SubAssign<&FReal<T>> for FReal<T> {
    fn sub_assign(&mut self, rhs: &FReal<T>) {
        self.value = self.value - rhs.value;
        self.derivative = self.derivative - rhs.derivative;
    }
}

impl<T: Scalar> SubAssign<T> for FReal<T> {
    fn sub_assign(&mut self, rhs: T) {
        self.value = self.value - rhs;
    }
}

impl<T: Scalar> MulAssign for FReal<T> {
    fn mul_assign(&mut self, rhs: FReal<T>) {
        let new_deriv = self.derivative * rhs.value + self.value * rhs.derivative;
        self.value = self.value * rhs.value;
        self.derivative = new_deriv;
    }
}

impl<T: Scalar> MulAssign<&FReal<T>> for FReal<T> {
    fn mul_assign(&mut self, rhs: &FReal<T>) {
        let new_deriv = self.derivative * rhs.value + self.value * rhs.derivative;
        self.value = self.value * rhs.value;
        self.derivative = new_deriv;
    }
}

impl<T: Scalar> MulAssign<T> for FReal<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.value = self.value * rhs;
        self.derivative = self.derivative * rhs;
    }
}

impl<T: Scalar> DivAssign for FReal<T> {
    fn div_assign(&mut self, rhs: FReal<T>) {
        let inv_b = T::one() / rhs.value;
        self.derivative =
            (self.derivative * rhs.value - self.value * rhs.derivative) * inv_b * inv_b;
        self.value *= inv_b;
    }
}

impl<T: Scalar> DivAssign<&FReal<T>> for FReal<T> {
    fn div_assign(&mut self, rhs: &FReal<T>) {
        let inv_b = T::one() / rhs.value;
        self.derivative =
            (self.derivative * rhs.value - self.value * rhs.derivative) * inv_b * inv_b;
        self.value *= inv_b;
    }
}

impl<T: Scalar> DivAssign<T> for FReal<T> {
    fn div_assign(&mut self, rhs: T) {
        let inv = T::one() / rhs;
        self.value *= inv;
        self.derivative *= inv;
    }
}

// Comparison
impl<T: Scalar> PartialEq for FReal<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Scalar> PartialEq<T> for FReal<T> {
    fn eq(&self, other: &T) -> bool {
        self.value == *other
    }
}

impl<T: Scalar> PartialOrd for FReal<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: Scalar> PartialOrd<T> for FReal<T> {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(other)
    }
}

impl<T: Scalar> fmt::Display for FReal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<T: Scalar> fmt::Debug for FReal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FReal({}, deriv={})", self.value, self.derivative)
    }
}

impl<T: Scalar> Default for FReal<T> {
    fn default() -> Self {
        FReal::constant(T::zero())
    }
}

// ============================================================================
// NamedFReal<T> — named wrapper over FReal<T> (generic forward-mode)
// ============================================================================

// `NamedFReal<T>` — named wrapper over `FReal<T>` (generic forward-mode).
//
// **Shape A (Phase 02.2):** does NOT carry an `Arc<VarRegistry>` field in
// release builds. The struct layout is a single `inner: FReal<T>` field
// plus, under `#[cfg(debug_assertions)]` only, a `gen_id: u64` stamped by the
// owning [`NamedForwardTape`] scope for the cross-registry debug guard.
// Release builds are bit-for-bit equivalent to a pure `FReal<T>` wrapper
// and carry zero atomic-refcount cost per operator.
//
// The only way to obtain a `NamedFReal<T>` is via
// [`NamedForwardTape::input_freal`] or
// [`NamedForwardTape::constant_freal`], which stamp the current TLS
// active-generation into the wrapper.
//
// `FReal<T>` carries ONE tangent direction — [`derivative`](Self::derivative)
// returns the same value regardless of which `name` is queried. The
// cross-generation check at binary-op time catches mixing values from
// different `NamedForwardTape` scopes in debug builds.

use crate::math;

/// Labeled wrapper around the positional [`FReal<T>`] type.
///
/// # Example
///
/// ```
/// use xad_rs::{NamedFReal, NamedForwardTape};
///
/// let mut ft = NamedForwardTape::new();
/// let x: NamedFReal<f64> = ft.input_freal("x", 2.0);
/// let _registry = ft.freeze();
/// let f = &x * &x + &x; // f(x) = x^2 + x, f'(x) = 2x + 1 = 5
/// assert_eq!(f.value(), 6.0);
/// assert_eq!(f.derivative("x"), 5.0);
/// ```
#[derive(Clone)]
pub struct NamedFReal<T: Scalar> {
    pub(crate) inner: FReal<T>,
    // NOTE: field name is `gen_id` — `gen` alone is a reserved keyword in
    // Rust 2024 edition. The D-01/D-02 CONTEXT blocks spell it as `gen`;
    // we carry the spelling adjustment forward to satisfy the compiler.
    #[cfg(debug_assertions)]
    pub(crate) gen_id: u64,
}

impl<T: Scalar> NamedFReal<T> {
    /// Internal constructor used by `NamedForwardTape::input_freal` and
    /// `NamedForwardTape::constant_freal`. Reads the TLS active
    /// generation (debug builds only) to stamp the `gen` field. Not part
    /// of the public API.
    #[inline]
    pub(crate) fn __from_inner(inner: FReal<T>) -> Self {
        Self {
            inner,
            #[cfg(debug_assertions)]
            gen_id: crate::forward_tape::current_gen(),
        }
    }

    /// Value part.
    #[inline]
    pub fn value(&self) -> T {
        self.inner.value()
    }

    /// Label-keyed single-direction derivative accessor.
    ///
    /// `FReal<T>` carries only one tangent direction, so this returns the
    /// current tangent value regardless of which `name` is queried. The
    /// cross-generation debug guard in binary-op impls catches mixing
    /// values from different `NamedForwardTape` scopes; users are
    /// expected to call `derivative(name)` only with names registered on
    /// the tape that constructed this value.
    #[inline]
    pub fn derivative(&self, _name: &str) -> T {
        self.inner.derivative()
    }

    /// Escape hatch: direct access to the inner positional `FReal<T>`.
    #[inline]
    pub fn inner(&self) -> &FReal<T> {
        &self.inner
    }

    // ============ Elementary math delegations ============
    // Forward to the free-function `math::fwd::*` surface, which operates
    // on `&FReal<T>` and returns `FReal<T>`. Each method preserves the
    // parent's generation stamp explicitly (debug builds) to avoid a TLS
    // read on the hot path.

    /// Natural exponential, preserving the parent scope's generation.
    #[inline]
    pub fn exp(&self) -> Self {
        Self {
            inner: math::fwd::exp(&self.inner),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Natural logarithm, preserving the parent scope's generation.
    #[inline]
    pub fn ln(&self) -> Self {
        Self {
            inner: math::fwd::ln(&self.inner),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Square root, preserving the parent scope's generation.
    #[inline]
    pub fn sqrt(&self) -> Self {
        Self {
            inner: math::fwd::sqrt(&self.inner),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Sine, preserving the parent scope's generation.
    #[inline]
    pub fn sin(&self) -> Self {
        Self {
            inner: math::fwd::sin(&self.inner),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Cosine, preserving the parent scope's generation.
    #[inline]
    pub fn cos(&self) -> Self {
        Self {
            inner: math::fwd::cos(&self.inner),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Tangent, preserving the parent scope's generation.
    #[inline]
    pub fn tan(&self) -> Self {
        Self {
            inner: math::fwd::tan(&self.inner),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Standard normal CDF, preserving the parent scope's generation.
    #[inline]
    pub fn norm_cdf(&self) -> Self {
        Self {
            inner: math::fwd::norm_cdf(&self.inner),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Inverse standard normal CDF, preserving the parent scope's generation.
    #[inline]
    pub fn inv_norm_cdf(&self) -> Self {
        Self {
            inner: math::fwd::inv_norm_cdf(&self.inner),
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }
}

impl<T: Scalar> fmt::Debug for NamedFReal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NamedFReal")
            .field("value", &self.inner.value())
            .field("derivative", &self.inner.derivative())
            .finish()
    }
}

impl<T: Scalar> fmt::Display for NamedFReal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NamedFReal({})", self.inner.value())
    }
}

// ============ Operator overloads — hand-written, Shape A ============
// No shared op-stamping macro is used: Shape A does not carry an
// `Arc` registry field on the per-value wrapper. The four reference
// variants (owned/owned, ref/ref, owned/ref, ref/owned) plus scalar-RHS
// variants are stamped explicitly via a local `__named_freal_binop!`
// macro, modelled on `__named_areal_binop!` in `src/named/areal.rs`
// but generalised over `<T: Scalar>`. Each impl performs a debug-only
// `check_gen` between the two operands' generations, then constructs
// the result preserving the LHS's generation stamp.

macro_rules! __named_freal_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Scalar> ::core::ops::$trait<NamedFReal<T>> for NamedFReal<T> {
            type Output = NamedFReal<T>;
            #[inline]
            fn $method(self, rhs: NamedFReal<T>) -> NamedFReal<T> {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedFReal {
                    inner: self.inner $op rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl<T: Scalar> ::core::ops::$trait<&NamedFReal<T>> for &NamedFReal<T> {
            type Output = NamedFReal<T>;
            #[inline]
            fn $method(self, rhs: &NamedFReal<T>) -> NamedFReal<T> {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedFReal {
                    inner: &self.inner $op &rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl<T: Scalar> ::core::ops::$trait<&NamedFReal<T>> for NamedFReal<T> {
            type Output = NamedFReal<T>;
            #[inline]
            fn $method(self, rhs: &NamedFReal<T>) -> NamedFReal<T> {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedFReal {
                    inner: self.inner $op &rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl<T: Scalar> ::core::ops::$trait<NamedFReal<T>> for &NamedFReal<T> {
            type Output = NamedFReal<T>;
            #[inline]
            fn $method(self, rhs: NamedFReal<T>) -> NamedFReal<T> {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedFReal {
                    inner: &self.inner $op rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl<T: Scalar> ::core::ops::$trait<T> for NamedFReal<T> {
            type Output = NamedFReal<T>;
            #[inline]
            fn $method(self, rhs: T) -> NamedFReal<T> {
                NamedFReal {
                    inner: self.inner $op rhs,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl<T: Scalar> ::core::ops::$trait<T> for &NamedFReal<T> {
            type Output = NamedFReal<T>;
            #[inline]
            fn $method(self, rhs: T) -> NamedFReal<T> {
                NamedFReal {
                    inner: &self.inner $op rhs,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
    };
}

__named_freal_binop!(Add, add, +);
__named_freal_binop!(Sub, sub, -);
__named_freal_binop!(Mul, mul, *);
__named_freal_binop!(Div, div, /);

impl<T: Scalar> ::core::ops::Neg for NamedFReal<T> {
    type Output = NamedFReal<T>;
    #[inline]
    fn neg(self) -> NamedFReal<T> {
        NamedFReal {
            inner: -self.inner,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }
}
impl<T: Scalar> ::core::ops::Neg for &NamedFReal<T> {
    type Output = NamedFReal<T>;
    #[inline]
    fn neg(self) -> NamedFReal<T> {
        NamedFReal {
            inner: -&self.inner,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }
}

// ============ Scalar-on-LHS hand-written impls for f64 and f32 ============
// Two concrete types (f64, f32), two variants (owned, ref), four ops =
// 16 impls. Inner FReal<T> already provides `f64 op FReal<f64>` and
// `f32 op FReal<f32>` for owned + ref, so delegation is direct. Each
// result preserves the RHS's generation stamp (debug builds only).

macro_rules! __named_freal_scalar_lhs {
    ($scalar:ty) => {
        impl ::core::ops::Add<NamedFReal<$scalar>> for $scalar {
            type Output = NamedFReal<$scalar>;
            #[inline]
            fn add(self, rhs: NamedFReal<$scalar>) -> NamedFReal<$scalar> {
                NamedFReal {
                    inner: self + rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Add<&NamedFReal<$scalar>> for $scalar {
            type Output = NamedFReal<$scalar>;
            #[inline]
            fn add(self, rhs: &NamedFReal<$scalar>) -> NamedFReal<$scalar> {
                NamedFReal {
                    inner: self + &rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Sub<NamedFReal<$scalar>> for $scalar {
            type Output = NamedFReal<$scalar>;
            #[inline]
            fn sub(self, rhs: NamedFReal<$scalar>) -> NamedFReal<$scalar> {
                NamedFReal {
                    inner: self - rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Sub<&NamedFReal<$scalar>> for $scalar {
            type Output = NamedFReal<$scalar>;
            #[inline]
            fn sub(self, rhs: &NamedFReal<$scalar>) -> NamedFReal<$scalar> {
                NamedFReal {
                    inner: self - &rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Mul<NamedFReal<$scalar>> for $scalar {
            type Output = NamedFReal<$scalar>;
            #[inline]
            fn mul(self, rhs: NamedFReal<$scalar>) -> NamedFReal<$scalar> {
                NamedFReal {
                    inner: self * rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Mul<&NamedFReal<$scalar>> for $scalar {
            type Output = NamedFReal<$scalar>;
            #[inline]
            fn mul(self, rhs: &NamedFReal<$scalar>) -> NamedFReal<$scalar> {
                NamedFReal {
                    inner: self * &rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Div<NamedFReal<$scalar>> for $scalar {
            type Output = NamedFReal<$scalar>;
            #[inline]
            fn div(self, rhs: NamedFReal<$scalar>) -> NamedFReal<$scalar> {
                NamedFReal {
                    inner: self / rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Div<&NamedFReal<$scalar>> for $scalar {
            type Output = NamedFReal<$scalar>;
            #[inline]
            fn div(self, rhs: &NamedFReal<$scalar>) -> NamedFReal<$scalar> {
                NamedFReal {
                    inner: self / &rhs.inner,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
    };
}

__named_freal_scalar_lhs!(f64);
__named_freal_scalar_lhs!(f32);
