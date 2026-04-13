//! `Dual2` - dedicated second-order forward-mode AD type.
//!
//! A `Dual2<T>` carries three values:
//!   - `value`: f(x)
//!   - `d1`:    f'(x)  (first derivative w.r.t. a seed direction)
//!   - `d2`:    f''(x) (second derivative w.r.t. the same seed direction)
//!
//! Unlike full Hessian machinery, `Dual2` tracks derivatives with respect to
//! a *single* scalar variable. This is exactly what you need to compute
//! diagonal elements of a Hessian (e.g. "own-gamma" in financial risk) in a
//! single forward pass, with no tape and no finite-difference error.
//!
//! Usage:
//! ```
//! use xad_rs::forward::dual2::Dual2;
//! // Compute f(x) = x^3 and its first/second derivatives at x = 2.
//! let x = Dual2::variable(2.0_f64);
//! let y = x * x * x;
//! assert_eq!(y.value(), 8.0);
//! assert_eq!(y.first_derivative(), 12.0); // 3x^2
//! assert_eq!(y.second_derivative(), 12.0); // 6x
//! ```
//!
//! This module also contains [`NamedDual2<T>`] — a named wrapper over
//! seeded `Dual2<T>`.
//!
//! Does NOT carry an `Arc<VarRegistry>` field in
//! release builds. The struct layout is `{ inner: Dual2<T>, seeded: Option<usize> }`
//! plus, under `#[cfg(debug_assertions)]` only, a `gen_id: u64` stamped by the
//! owning [`NamedForwardTape`] scope for the cross-registry debug guard.
//! Release builds carry zero atomic-refcount cost per operator.
//!
//! `Dual2<T>` tracks first AND second derivatives along ONE seed direction.
//! The named form gives that direction a name at construction time and
//! exposes [`first_derivative`](NamedDual2::first_derivative) /
//! [`second_derivative`](NamedDual2::second_derivative) accessors that
//! return the seeded values when the name matches and `T::zero()`
//! otherwise. The `seeded: Option<usize>` field is NOT the registry — it
//! is the positional index of the currently-seeded variable, a per-value
//! attribute that survives the minimal struct layout.
//!
//! Unlike `NamedDual` and `NamedFReal`, every binary op on
//! `NamedDual2` must propagate the `seeded: Option<usize>` field through
//! a private [`merge_seeded`] combinator. Operations between two
//! `NamedDual2<T>` values with different seeds panic in debug builds
//! (two-direction `Dual2` is a semantic violation).
//!
//! The only way to obtain a `NamedDual2<T>` is via the
//! `NamedForwardTape` constructor API (see module docs in
//! `src/forward_tape.rs`).

use crate::scalar::Scalar;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ============================================================================
// Dual2<T> — positional second-order forward-mode dual number
// ============================================================================

/// Second-order forward-mode dual number.
#[derive(Clone, Copy)]
pub struct Dual2<T: Scalar> {
    value: T,
    d1: T,
    d2: T,
}

impl<T: Scalar> Dual2<T> {
    /// Create a `Dual2` with explicit value, first and second derivative.
    #[inline]
    pub fn new(value: T, d1: T, d2: T) -> Self {
        Dual2 { value, d1, d2 }
    }

    /// Create a constant (derivative-free) `Dual2`.
    #[inline]
    pub fn constant(value: T) -> Self {
        Dual2 {
            value,
            d1: T::zero(),
            d2: T::zero(),
        }
    }

    /// Create the active variable: value with `d1 = 1`, `d2 = 0`.
    /// Use this to seed the single input direction to differentiate against.
    #[inline]
    pub fn variable(value: T) -> Self {
        Dual2 {
            value,
            d1: T::one(),
            d2: T::zero(),
        }
    }

    /// Underlying value.
    #[inline]
    pub fn value(&self) -> T {
        self.value
    }

    /// First derivative (tangent) along the seeded direction.
    #[inline]
    pub fn first_derivative(&self) -> T {
        self.d1
    }

    /// Second derivative along the seeded direction.
    #[inline]
    pub fn second_derivative(&self) -> T {
        self.d2
    }

    /// Raise `self` to a scalar power `n`: `self^n`.
    ///
    /// Applies the chain rule for 2nd order: for `g(u) = u^n`,
    /// `g'(u) = n u^{n-1}`, `g''(u) = n (n-1) u^{n-2}`, then
    /// `result.d1 = g'(v) * self.d1` and
    /// `result.d2 = g''(v) * self.d1^2 + g'(v) * self.d2`.
    pub fn powf(self, n: T) -> Dual2<T> {
        let v = self.value;
        let two = T::from(2.0).unwrap();
        let vn = v.powf(n);
        let gp = n * v.powf(n - T::one());
        let gpp = n * (n - T::one()) * v.powf(n - two);
        Dual2 {
            value: vn,
            d1: gp * self.d1,
            d2: gpp * self.d1 * self.d1 + gp * self.d2,
        }
    }

    /// Exponential `exp(self)`.
    pub fn exp(self) -> Dual2<T> {
        let e = self.value.exp();
        Dual2 {
            value: e,
            d1: e * self.d1,
            d2: e * self.d1 * self.d1 + e * self.d2,
        }
    }

    /// Natural logarithm `ln(self)`.
    pub fn ln(self) -> Dual2<T> {
        // g(u) = ln(u), g'(u) = 1/u, g''(u) = -1/u^2
        let v = self.value;
        let inv = T::one() / v;
        let inv2 = inv * inv;
        Dual2 {
            value: v.ln(),
            d1: inv * self.d1,
            d2: -inv2 * self.d1 * self.d1 + inv * self.d2,
        }
    }

    /// Square root.
    pub fn sqrt(self) -> Dual2<T> {
        // g(u) = u^{1/2}, g'(u) = 1/(2*sqrt(u)), g''(u) = -1/(4*u^{3/2})
        let v = self.value;
        let s = v.sqrt();
        let half = T::from(0.5).unwrap();
        let quarter = T::from(0.25).unwrap();
        let gp = half / s;
        let gpp = -quarter / (s * v);
        Dual2 {
            value: s,
            d1: gp * self.d1,
            d2: gpp * self.d1 * self.d1 + gp * self.d2,
        }
    }

    /// Sine.
    pub fn sin(self) -> Dual2<T> {
        // g(u) = sin(u), g'(u) = cos(u), g''(u) = -sin(u)
        let v = self.value;
        let (s, c) = (v.sin(), v.cos());
        Dual2 {
            value: s,
            d1: c * self.d1,
            d2: -s * self.d1 * self.d1 + c * self.d2,
        }
    }

    /// Cosine.
    pub fn cos(self) -> Dual2<T> {
        // g(u) = cos(u), g'(u) = -sin(u), g''(u) = -cos(u)
        let v = self.value;
        let (s, c) = (v.sin(), v.cos());
        Dual2 {
            value: c,
            d1: -s * self.d1,
            d2: -c * self.d1 * self.d1 - s * self.d2,
        }
    }

    /// Error function `erf(self)`.
    ///
    /// Derivatives:
    ///   `erf'(x)  = (2/√π) · exp(-x²)`
    ///   `erf''(x) = -2x · erf'(x)`
    pub fn erf(self) -> Dual2<T> {
        let v = self.value;
        let r = crate::math::erf(v);
        let two = T::from(2.0).unwrap();
        // Convert the compile-time `f64` constant `FRAC_2_SQRT_PI` into the
        // generic scalar `T` exactly once — cheaper than computing a sqrt.
        let two_over_sqrt_pi =
            T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap();
        let gp = two_over_sqrt_pi * (-v * v).exp();
        let gpp = -two * v * gp;
        Dual2 {
            value: r,
            d1: gp * self.d1,
            d2: gpp * self.d1 * self.d1 + gp * self.d2,
        }
    }

    /// Standard normal CDF: `Φ(x) = 0.5 · (1 + erf(x / √2))`.
    ///
    /// - `g'(x) = φ(x) = (1/√(2π)) · exp(-x²/2)`
    /// - `g''(x) = -x · φ(x)`
    pub fn norm_cdf(self) -> Dual2<T> {
        let v = self.value;
        let r = crate::math::norm_cdf(v);
        let gp = crate::math::norm_pdf(v);
        let gpp = -v * gp;
        Dual2 {
            value: r,
            d1: gp * self.d1,
            d2: gpp * self.d1 * self.d1 + gp * self.d2,
        }
    }

    /// Inverse standard normal CDF: `Φ⁻¹(p)`.
    ///
    /// - `g'(p) = 1 / φ(Φ⁻¹(p))`
    /// - `g''(p) = Φ⁻¹(p) · g'(p)²`
    pub fn inv_norm_cdf(self) -> Dual2<T> {
        let v = self.value;
        let r = crate::math::inv_norm_cdf(v);
        let gp = T::one() / crate::math::norm_pdf(r);
        let gpp = r * gp * gp;
        Dual2 {
            value: r,
            d1: gp * self.d1,
            d2: gpp * self.d1 * self.d1 + gp * self.d2,
        }
    }
}

// ============================================================================
// Operator implementations for Dual2<T>
// ============================================================================

// --- Add ---
impl<T: Scalar> Add for Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Dual2 {
            value: self.value + rhs.value,
            d1: self.d1 + rhs.d1,
            d2: self.d2 + rhs.d2,
        }
    }
}

impl<T: Scalar> Add<T> for Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn add(self, rhs: T) -> Self {
        Dual2 {
            value: self.value + rhs,
            d1: self.d1,
            d2: self.d2,
        }
    }
}

impl Add<Dual2<f64>> for f64 {
    type Output = Dual2<f64>;
    #[inline]
    fn add(self, rhs: Dual2<f64>) -> Dual2<f64> {
        Dual2 {
            value: self + rhs.value,
            d1: rhs.d1,
            d2: rhs.d2,
        }
    }
}

impl Add<Dual2<f32>> for f32 {
    type Output = Dual2<f32>;
    #[inline]
    fn add(self, rhs: Dual2<f32>) -> Dual2<f32> {
        Dual2 {
            value: self + rhs.value,
            d1: rhs.d1,
            d2: rhs.d2,
        }
    }
}

// --- Sub ---
impl<T: Scalar> Sub for Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Dual2 {
            value: self.value - rhs.value,
            d1: self.d1 - rhs.d1,
            d2: self.d2 - rhs.d2,
        }
    }
}

impl<T: Scalar> Sub<T> for Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        Dual2 {
            value: self.value - rhs,
            d1: self.d1,
            d2: self.d2,
        }
    }
}

impl Sub<Dual2<f64>> for f64 {
    type Output = Dual2<f64>;
    #[inline]
    fn sub(self, rhs: Dual2<f64>) -> Dual2<f64> {
        Dual2 {
            value: self - rhs.value,
            d1: -rhs.d1,
            d2: -rhs.d2,
        }
    }
}

impl Sub<Dual2<f32>> for f32 {
    type Output = Dual2<f32>;
    #[inline]
    fn sub(self, rhs: Dual2<f32>) -> Dual2<f32> {
        Dual2 {
            value: self - rhs.value,
            d1: -rhs.d1,
            d2: -rhs.d2,
        }
    }
}

// --- Mul ---
// (a*b)' = a'*b + a*b'
// (a*b)'' = a''*b + 2*a'*b' + a*b''
impl<T: Scalar> Mul for Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let two = T::from(2.0).unwrap();
        Dual2 {
            value: self.value * rhs.value,
            d1: self.d1 * rhs.value + self.value * rhs.d1,
            d2: self.d2 * rhs.value + two * self.d1 * rhs.d1 + self.value * rhs.d2,
        }
    }
}

impl<T: Scalar> Mul<T> for Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        Dual2 {
            value: self.value * rhs,
            d1: self.d1 * rhs,
            d2: self.d2 * rhs,
        }
    }
}

impl Mul<Dual2<f64>> for f64 {
    type Output = Dual2<f64>;
    #[inline]
    fn mul(self, rhs: Dual2<f64>) -> Dual2<f64> {
        Dual2 {
            value: self * rhs.value,
            d1: self * rhs.d1,
            d2: self * rhs.d2,
        }
    }
}

impl Mul<Dual2<f32>> for f32 {
    type Output = Dual2<f32>;
    #[inline]
    fn mul(self, rhs: Dual2<f32>) -> Dual2<f32> {
        Dual2 {
            value: self * rhs.value,
            d1: self * rhs.d1,
            d2: self * rhs.d2,
        }
    }
}

// --- Div ---
// a/b = a * (1/b)
// (1/b)'  = -b'/b^2
// (1/b)'' = 2 b'^2/b^3 - b''/b^2
impl<T: Scalar> Div for Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let two = T::from(2.0).unwrap();
        let inv_b = T::one() / rhs.value;
        let inv_b2 = inv_b * inv_b;
        let inv_b3 = inv_b2 * inv_b;
        let recip = Dual2 {
            value: inv_b,
            d1: -rhs.d1 * inv_b2,
            d2: two * rhs.d1 * rhs.d1 * inv_b3 - rhs.d2 * inv_b2,
        };
        self * recip
    }
}

impl<T: Scalar> Div<T> for Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn div(self, rhs: T) -> Self {
        let inv = T::one() / rhs;
        Dual2 {
            value: self.value * inv,
            d1: self.d1 * inv,
            d2: self.d2 * inv,
        }
    }
}

impl Div<Dual2<f64>> for f64 {
    type Output = Dual2<f64>;
    #[inline]
    fn div(self, rhs: Dual2<f64>) -> Dual2<f64> {
        Dual2::constant(self) / rhs
    }
}

impl Div<Dual2<f32>> for f32 {
    type Output = Dual2<f32>;
    #[inline]
    fn div(self, rhs: Dual2<f32>) -> Dual2<f32> {
        Dual2::constant(self) / rhs
    }
}

// --- Neg ---
impl<T: Scalar> Neg for Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn neg(self) -> Self {
        Dual2 {
            value: -self.value,
            d1: -self.d1,
            d2: -self.d2,
        }
    }
}

// --- Compound assignment ---
impl<T: Scalar> AddAssign for Dual2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Scalar> AddAssign<T> for Dual2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.value = self.value + rhs;
    }
}

impl<T: Scalar> SubAssign for Dual2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Scalar> SubAssign<T> for Dual2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.value = self.value - rhs;
    }
}

impl<T: Scalar> MulAssign for Dual2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Scalar> MulAssign<T> for Dual2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: Scalar> DivAssign for Dual2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: Scalar> DivAssign<T> for Dual2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

// --- Display / Debug / Default ---
impl<T: Scalar> fmt::Display for Dual2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<T: Scalar> fmt::Debug for Dual2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dual2(v={}, d1={}, d2={})",
            self.value, self.d1, self.d2
        )
    }
}

impl<T: Scalar> Default for Dual2<T> {
    fn default() -> Self {
        Dual2::constant(T::zero())
    }
}

impl<T: Scalar> From<T> for Dual2<T> {
    fn from(value: T) -> Self {
        Dual2::constant(value)
    }
}

// ============================================================================
// NamedDual2<T> — named wrapper over seeded Dual2<T>
// ============================================================================

/// Labeled wrapper around the positional [`Dual2<T>`] type.
///
/// # Example
///
/// The constructor API for `NamedDual2<f64>` is owned by
/// `NamedForwardTape` via the `declare_dual2_f64` / `freeze_dual` /
/// `scope.dual2(handle)` pattern. See the module docs in
/// `src/forward_tape.rs` for the full rationale.
///
/// ```
/// use xad_rs::{NamedForwardScope, NamedForwardTape};
///
/// let mut ft = NamedForwardTape::new();
/// let x_h = ft.declare_dual2_f64("x", 3.0);
/// let scope: NamedForwardScope = ft.freeze_dual();
///
/// let x = scope.dual2(x_h);
/// let f = x * x; // f(x) = x^2, f'(x) = 2x = 6, f''(x) = 2
///
/// assert_eq!(f.value(), 9.0);
/// assert_eq!(f.first_derivative("x"), 6.0);
/// assert_eq!(f.second_derivative("x"), 2.0);
/// ```
#[derive(Clone)]
pub struct NamedDual2<T: Scalar> {
    pub(crate) inner: Dual2<T>,
    pub(crate) seeded: Option<usize>,
    // NOTE: field name is `gen_id` — `gen` alone is a reserved keyword in
    // Rust 2024 edition.
    #[cfg(debug_assertions)]
    pub(crate) gen_id: u64,
}

impl<T: Scalar> NamedDual2<T> {
    /// Internal constructor used by `NamedForwardTape` input / freeze
    /// paths. Reads the TLS active generation (debug builds only) to stamp
    /// the `gen_id` field. Not part of the public API.
    #[inline]
    pub(crate) fn __from_parts(inner: Dual2<T>, seeded: Option<usize>) -> Self {
        Self {
            inner,
            seeded,
            #[cfg(debug_assertions)]
            gen_id: crate::forward_tape::current_gen(),
        }
    }

    /// Value part.
    #[inline]
    pub fn value(&self) -> T {
        self.inner.value()
    }

    /// First derivative with respect to `name`.
    ///
    /// Returns the seeded first derivative if `name` matches the seeded
    /// variable, else `T::zero()`. Reads the active registry from the
    /// `NamedForwardTape` thread-local slot. Panics if `name` is not in
    /// the registry, or if called outside a frozen `NamedForwardTape`
    /// scope.
    pub fn first_derivative(&self, name: &str) -> T {
        let idx = crate::forward_tape::with_active_registry(|r| {
            let r = r.expect(
                "NamedDual2::first_derivative called outside a frozen NamedForwardTape scope",
            );
            r.index_of(name).unwrap_or_else(|| {
                panic!(
                    "NamedDual2::first_derivative: name {:?} not present in registry",
                    name
                )
            })
        });
        if self.seeded == Some(idx) {
            self.inner.first_derivative()
        } else {
            T::zero()
        }
    }

    /// Second derivative (diagonal Hessian entry) with respect to `name`.
    ///
    /// Returns the seeded second derivative if `name` matches the seeded
    /// variable, else `T::zero()`. Panics if `name` is not in the registry,
    /// or if called outside a frozen `NamedForwardTape` scope.
    pub fn second_derivative(&self, name: &str) -> T {
        let idx = crate::forward_tape::with_active_registry(|r| {
            let r = r.expect(
                "NamedDual2::second_derivative called outside a frozen NamedForwardTape scope",
            );
            r.index_of(name).unwrap_or_else(|| {
                panic!(
                    "NamedDual2::second_derivative: name {:?} not present in registry",
                    name
                )
            })
        });
        if self.seeded == Some(idx) {
            self.inner.second_derivative()
        } else {
            T::zero()
        }
    }

    /// Escape hatch: direct access to the inner positional `Dual2<T>`.
    #[inline]
    pub fn inner(&self) -> &Dual2<T> {
        &self.inner
    }

    // ============ Elementary math delegations ============
    // Each method forwards to the inherent `Dual2<T>` elementary (which
    // takes `self` by value — `Dual2<T>` is `Copy`), preserves the
    // `seeded` field because unary elementaries cannot change which
    // direction is active, and stamps the parent's generation explicitly.

    /// Natural exponential, preserving the seed and the parent's generation.
    #[inline]
    pub fn exp(&self) -> Self {
        Self {
            inner: self.inner.exp(),
            seeded: self.seeded,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Natural logarithm, preserving the seed and the parent's generation.
    #[inline]
    pub fn ln(&self) -> Self {
        Self {
            inner: self.inner.ln(),
            seeded: self.seeded,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Square root, preserving the seed and the parent's generation.
    #[inline]
    pub fn sqrt(&self) -> Self {
        Self {
            inner: self.inner.sqrt(),
            seeded: self.seeded,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Sine, preserving the seed and the parent's generation.
    #[inline]
    pub fn sin(&self) -> Self {
        Self {
            inner: self.inner.sin(),
            seeded: self.seeded,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Cosine, preserving the seed and the parent's generation.
    #[inline]
    pub fn cos(&self) -> Self {
        Self {
            inner: self.inner.cos(),
            seeded: self.seeded,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Standard normal CDF, preserving the seed and the parent's generation.
    #[inline]
    pub fn norm_cdf(&self) -> Self {
        Self {
            inner: self.inner.norm_cdf(),
            seeded: self.seeded,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }

    /// Inverse standard normal CDF, preserving the seed and the parent's generation.
    #[inline]
    pub fn inv_norm_cdf(&self) -> Self {
        Self {
            inner: self.inner.inv_norm_cdf(),
            seeded: self.seeded,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }
}

impl<T: Scalar> fmt::Debug for NamedDual2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NamedDual2")
            .field("value", &self.inner.value())
            .field("first", &self.inner.first_derivative())
            .field("second", &self.inner.second_derivative())
            .field("seeded", &self.seeded)
            .finish()
    }
}

impl<T: Scalar> fmt::Display for NamedDual2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NamedDual2({})", self.inner.value())
    }
}

/// Merge two seeded directions. Same seed or one constant is fine;
/// two different seeds is a `Dual2` semantic violation — debug-panic,
/// release-pick-LHS.
#[inline]
pub(crate) fn merge_seeded(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some(x), Some(y)) if x == y => Some(x),
        (Some(_), Some(_)) => {
            #[cfg(debug_assertions)]
            panic!(
                "NamedDual2: operation between two differently-seeded variables; \
                 seeded Dual2 supports only one active direction"
            );
            #[cfg(not(debug_assertions))]
            a
        }
    }
}

// ============================ Operator impls for NamedDual2 ============================
// Hand-written local macro (no shared stamping macro) because of the
// `seeded` field merge logic.
// `Dual2<T>` is `Copy`, so inner-value ops are by-value throughout.
// Each wrapper-vs-wrapper impl performs a debug-only `check_gen` between
// the two operands' generations, then merges `seeded` via `merge_seeded`,
// then constructs the result preserving the LHS's generation stamp.
// Pattern: 6 variants per binary op × 4 ops = 24 impls, plus 2 Neg impls.

macro_rules! __named_d2_binop {
    ($trait_:ident, $method:ident, $op:tt) => {
        // owned + owned
        impl<T: Scalar> ::core::ops::$trait_<NamedDual2<T>> for NamedDual2<T> {
            type Output = NamedDual2<T>;
            #[inline]
            fn $method(self, rhs: NamedDual2<T>) -> NamedDual2<T> {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedDual2 {
                    inner: self.inner $op rhs.inner,
                    seeded: merge_seeded(self.seeded, rhs.seeded),
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        // ref + ref
        impl<T: Scalar> ::core::ops::$trait_<&NamedDual2<T>> for &NamedDual2<T> {
            type Output = NamedDual2<T>;
            #[inline]
            fn $method(self, rhs: &NamedDual2<T>) -> NamedDual2<T> {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedDual2 {
                    inner: self.inner $op rhs.inner,
                    seeded: merge_seeded(self.seeded, rhs.seeded),
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        // owned + ref
        impl<T: Scalar> ::core::ops::$trait_<&NamedDual2<T>> for NamedDual2<T> {
            type Output = NamedDual2<T>;
            #[inline]
            fn $method(self, rhs: &NamedDual2<T>) -> NamedDual2<T> {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedDual2 {
                    inner: self.inner $op rhs.inner,
                    seeded: merge_seeded(self.seeded, rhs.seeded),
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        // ref + owned
        impl<T: Scalar> ::core::ops::$trait_<NamedDual2<T>> for &NamedDual2<T> {
            type Output = NamedDual2<T>;
            #[inline]
            fn $method(self, rhs: NamedDual2<T>) -> NamedDual2<T> {
                #[cfg(debug_assertions)]
                crate::forward_tape::check_gen(self.gen_id, rhs.gen_id);
                NamedDual2 {
                    inner: self.inner $op rhs.inner,
                    seeded: merge_seeded(self.seeded, rhs.seeded),
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        // X op T (scalar, no cross-registry check)
        impl<T: Scalar> ::core::ops::$trait_<T> for NamedDual2<T> {
            type Output = NamedDual2<T>;
            #[inline]
            fn $method(self, rhs: T) -> NamedDual2<T> {
                NamedDual2 {
                    inner: self.inner $op rhs,
                    seeded: self.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
        impl<T: Scalar> ::core::ops::$trait_<T> for &NamedDual2<T> {
            type Output = NamedDual2<T>;
            #[inline]
            fn $method(self, rhs: T) -> NamedDual2<T> {
                NamedDual2 {
                    inner: self.inner $op rhs,
                    seeded: self.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: self.gen_id,
                }
            }
        }
    };
}

__named_d2_binop!(Add, add, +);
__named_d2_binop!(Sub, sub, -);
__named_d2_binop!(Mul, mul, *);
__named_d2_binop!(Div, div, /);

impl<T: Scalar> ::core::ops::Neg for NamedDual2<T> {
    type Output = NamedDual2<T>;
    #[inline]
    fn neg(self) -> NamedDual2<T> {
        NamedDual2 {
            inner: -self.inner,
            seeded: self.seeded,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }
}
impl<T: Scalar> ::core::ops::Neg for &NamedDual2<T> {
    type Output = NamedDual2<T>;
    #[inline]
    fn neg(self) -> NamedDual2<T> {
        NamedDual2 {
            inner: -self.inner,
            seeded: self.seeded,
            #[cfg(debug_assertions)]
            gen_id: self.gen_id,
        }
    }
}

// ============ Scalar-on-LHS impls (f64 and f32) ============
// The inner `Dual2<T>` only provides owned `f64 op Dual2<f64>` / `f32 op
// Dual2<f32>` (no ref variants). The named ref variants dereference the
// `Copy` inner to call the owned op.

macro_rules! __named_d2_scalar_lhs {
    ($scalar:ty) => {
        impl ::core::ops::Add<NamedDual2<$scalar>> for $scalar {
            type Output = NamedDual2<$scalar>;
            #[inline]
            fn add(self, rhs: NamedDual2<$scalar>) -> NamedDual2<$scalar> {
                NamedDual2 {
                    inner: self + rhs.inner,
                    seeded: rhs.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Add<&NamedDual2<$scalar>> for $scalar {
            type Output = NamedDual2<$scalar>;
            #[inline]
            fn add(self, rhs: &NamedDual2<$scalar>) -> NamedDual2<$scalar> {
                NamedDual2 {
                    inner: self + rhs.inner,
                    seeded: rhs.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Sub<NamedDual2<$scalar>> for $scalar {
            type Output = NamedDual2<$scalar>;
            #[inline]
            fn sub(self, rhs: NamedDual2<$scalar>) -> NamedDual2<$scalar> {
                NamedDual2 {
                    inner: self - rhs.inner,
                    seeded: rhs.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Sub<&NamedDual2<$scalar>> for $scalar {
            type Output = NamedDual2<$scalar>;
            #[inline]
            fn sub(self, rhs: &NamedDual2<$scalar>) -> NamedDual2<$scalar> {
                NamedDual2 {
                    inner: self - rhs.inner,
                    seeded: rhs.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Mul<NamedDual2<$scalar>> for $scalar {
            type Output = NamedDual2<$scalar>;
            #[inline]
            fn mul(self, rhs: NamedDual2<$scalar>) -> NamedDual2<$scalar> {
                NamedDual2 {
                    inner: self * rhs.inner,
                    seeded: rhs.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Mul<&NamedDual2<$scalar>> for $scalar {
            type Output = NamedDual2<$scalar>;
            #[inline]
            fn mul(self, rhs: &NamedDual2<$scalar>) -> NamedDual2<$scalar> {
                NamedDual2 {
                    inner: self * rhs.inner,
                    seeded: rhs.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Div<NamedDual2<$scalar>> for $scalar {
            type Output = NamedDual2<$scalar>;
            #[inline]
            fn div(self, rhs: NamedDual2<$scalar>) -> NamedDual2<$scalar> {
                NamedDual2 {
                    inner: self / rhs.inner,
                    seeded: rhs.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
        impl ::core::ops::Div<&NamedDual2<$scalar>> for $scalar {
            type Output = NamedDual2<$scalar>;
            #[inline]
            fn div(self, rhs: &NamedDual2<$scalar>) -> NamedDual2<$scalar> {
                NamedDual2 {
                    inner: self / rhs.inner,
                    seeded: rhs.seeded,
                    #[cfg(debug_assertions)]
                    gen_id: rhs.gen_id,
                }
            }
        }
    };
}

__named_d2_scalar_lhs!(f64);
__named_d2_scalar_lhs!(f32);
