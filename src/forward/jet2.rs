//! `Jet2` - dedicated second-order forward-mode AD type.
//!
//! A `Jet2<T>` carries three values:
//!   - `value`: f(x)
//!   - `d1`:    f'(x)  (first derivative w.r.t. a seed direction)
//!   - `d2`:    f''(x) (second derivative w.r.t. the same seed direction)
//!
//! Unlike full Hessian machinery, `Jet2` tracks derivatives with respect to
//! a *single* scalar variable. This is exactly what you need to compute
//! diagonal elements of a Hessian (e.g. "own-gamma" in financial risk) in a
//! single forward pass, with no tape and no finite-difference error.
//!
//! Usage:
//! ```
//! use xad_rs::forward::jet2::Jet2;
//! // Compute f(x) = x^3 and its first/second derivatives at x = 2.
//! let x = Jet2::variable(2.0_f64);
//! let y = x * x * x;
//! assert_eq!(y.value(), 8.0);
//! assert_eq!(y.first_derivative(), 12.0); // 3x^2
//! assert_eq!(y.second_derivative(), 12.0); // 6x
//! ```
//!
//! For the full `n × n` Hessian instead of one seeded direction, see
//! [`Jet2Vec`](crate::forward::jet2vec::Jet2Vec).

use crate::passive::Passive;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ============================================================================
// Jet2<T> — positional second-order forward-mode dual number
// ============================================================================

/// Second-order forward-mode dual number.
#[derive(Clone, Copy)]
pub struct Jet2<T: Passive> {
    value: T,
    d1: T,
    d2: T,
}

impl<T: Passive> Jet2<T> {
    /// Create a `Jet2` with explicit value, first and second derivative.
    #[inline]
    pub fn new(value: T, d1: T, d2: T) -> Self {
        Jet2 { value, d1, d2 }
    }

    /// Create a constant (derivative-free) `Jet2`.
    #[inline]
    pub fn constant(value: T) -> Self {
        Jet2 {
            value,
            d1: T::zero(),
            d2: T::zero(),
        }
    }

    /// Create the active variable: value with `d1 = 1`, `d2 = 0`.
    /// Use this to seed the single input direction to differentiate against.
    #[inline]
    pub fn variable(value: T) -> Self {
        Jet2 {
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
    pub fn powf(self, n: T) -> Jet2<T> {
        let v = self.value;
        let two = T::from(2.0).unwrap();
        let vn = v.powf(n);
        let gp = n * v.powf(n - T::one());
        let gpp = n * (n - T::one()) * v.powf(n - two);
        Jet2 {
            value: vn,
            d1: gp * self.d1,
            d2: gpp * self.d1 * self.d1 + gp * self.d2,
        }
    }

}


// ============================================================================
// Unary elementaries — stamped from the crate-wide derivative table
// ============================================================================
//
// Second-order chain rule for `g(u)` along the single seeded direction:
//
//   out.value = g(v)
//   out.d1    = g'(v)  · self.d1
//   out.d2    = g''(v) · self.d1² + g'(v) · self.d2
macro_rules! stamp_jet2_unary {
    ($name:ident, $doc:literal, $val:expr, $d1:expr, $d2:expr) => {
        impl<T: Passive> Jet2<T> {
            #[doc = $doc]
            #[inline]
            pub fn $name(self) -> Jet2<T> {
                let v = self.value;
                let r = ($val)(v);
                let gp = ($d1)(v, r);
                let gpp = ($d2)(v, r, gp);
                Jet2 {
                    value: r,
                    d1: gp * self.d1,
                    d2: gpp * self.d1 * self.d1 + gp * self.d2,
                }
            }
        }
    };
}
crate::elementaries::for_each_unary_elementary!(stamp_jet2_unary);

// ============================================================================
// Operator implementations for Jet2<T>
// ============================================================================

// `f64 op Jet2` (scalar LHS — the orphan rule forces a concrete float type).
macro_rules! impl_scalar_lhs_jet2_binop {
    ($trait:ident, $method:ident, ($l:ident, $r:ident) => $body:expr) => {
        impl_scalar_lhs_jet2_binop!(@one f64, $trait, $method, ($l, $r) => $body);
    };
    (@one $t:ty, $trait:ident, $method:ident, ($l:ident, $r:ident) => $body:expr) => {
        impl $trait<Jet2<$t>> for $t {
            type Output = Jet2<$t>;
            #[inline]
            fn $method(self, rhs: Jet2<$t>) -> Jet2<$t> {
                let ($l, $r) = (self, rhs);
                $body
            }
        }
    };
}

// --- Add ---
impl<T: Passive> Add for Jet2<T> {
    type Output = Jet2<T>;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Jet2 {
            value: self.value + rhs.value,
            d1: self.d1 + rhs.d1,
            d2: self.d2 + rhs.d2,
        }
    }
}

impl<T: Passive> Add<T> for Jet2<T> {
    type Output = Jet2<T>;
    #[inline]
    fn add(self, rhs: T) -> Self {
        Jet2 {
            value: self.value + rhs,
            d1: self.d1,
            d2: self.d2,
        }
    }
}

impl_scalar_lhs_jet2_binop!(Add, add, (l, r) => Jet2 {
    value: l + r.value,
    d1: r.d1,
    d2: r.d2,
});

// --- Sub ---
impl<T: Passive> Sub for Jet2<T> {
    type Output = Jet2<T>;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Jet2 {
            value: self.value - rhs.value,
            d1: self.d1 - rhs.d1,
            d2: self.d2 - rhs.d2,
        }
    }
}

impl<T: Passive> Sub<T> for Jet2<T> {
    type Output = Jet2<T>;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        Jet2 {
            value: self.value - rhs,
            d1: self.d1,
            d2: self.d2,
        }
    }
}

impl_scalar_lhs_jet2_binop!(Sub, sub, (l, r) => Jet2 {
    value: l - r.value,
    d1: -r.d1,
    d2: -r.d2,
});

// --- Mul ---
// (a*b)' = a'*b + a*b'
// (a*b)'' = a''*b + 2*a'*b' + a*b''
impl<T: Passive> Mul for Jet2<T> {
    type Output = Jet2<T>;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let two = T::from(2.0).unwrap();
        Jet2 {
            value: self.value * rhs.value,
            d1: self.d1 * rhs.value + self.value * rhs.d1,
            d2: self.d2 * rhs.value + two * self.d1 * rhs.d1 + self.value * rhs.d2,
        }
    }
}

impl<T: Passive> Mul<T> for Jet2<T> {
    type Output = Jet2<T>;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        Jet2 {
            value: self.value * rhs,
            d1: self.d1 * rhs,
            d2: self.d2 * rhs,
        }
    }
}

impl_scalar_lhs_jet2_binop!(Mul, mul, (l, r) => Jet2 {
    value: l * r.value,
    d1: l * r.d1,
    d2: l * r.d2,
});

// --- Div ---
// The *derivatives* go through the reciprocal jet, as they always have:
//   (1/b)'  = -b'/b^2
//   (1/b)'' = 2 b'^2/b^3 - b''/b^2
// then the product rule for `a * (1/b)`.
//
// The *value* does not. `a * (1/b)` rounds twice and lands up to 1 ulp away
// from `a / b`, which would make the number this mode returns differ from the
// number the passive scalar returns for the same operands. The quotient is
// written back over the product's value, leaving both derivatives bit-for-bit
// as before. See the passive-reference rule in `crate::real`.
impl<T: Passive> Div for Jet2<T> {
    type Output = Jet2<T>;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let two = T::from(2.0).unwrap();
        let inv_b = T::one() / rhs.value;
        let inv_b2 = inv_b * inv_b;
        let inv_b3 = inv_b2 * inv_b;
        let recip = Jet2 {
            value: inv_b,
            d1: -rhs.d1 * inv_b2,
            d2: two * rhs.d1 * rhs.d1 * inv_b3 - rhs.d2 * inv_b2,
        };
        let quotient = self.value / rhs.value;
        let mut out = self * recip;
        out.value = quotient;
        out
    }
}

impl<T: Passive> Div<T> for Jet2<T> {
    type Output = Jet2<T>;
    #[inline]
    fn div(self, rhs: T) -> Self {
        // Quotient for the value, reciprocal for the derivatives — see above.
        let inv = T::one() / rhs;
        Jet2 {
            value: self.value / rhs,
            d1: self.d1 * inv,
            d2: self.d2 * inv,
        }
    }
}

impl_scalar_lhs_jet2_binop!(Div, div, (l, r) => Jet2::constant(l) / r);

// --- Neg ---
impl<T: Passive> Neg for Jet2<T> {
    type Output = Jet2<T>;
    #[inline]
    fn neg(self) -> Self {
        Jet2 {
            value: -self.value,
            d1: -self.d1,
            d2: -self.d2,
        }
    }
}

// --- Compound assignment ---
// Delegates to the binary op (free for a `Copy` type). `AddAssign<T>` /
// `SubAssign<T>` therefore leave `d1`/`d2` untouched, matching `Add<T>`.
macro_rules! impl_jet2_assign {
    ($trait:ident, $method:ident, $bin:ident, $bin_method:ident) => {
        impl<T: Passive> $trait for Jet2<T> {
            #[inline]
            fn $method(&mut self, rhs: Jet2<T>) {
                *self = $bin::$bin_method(*self, rhs);
            }
        }
        impl<T: Passive> $trait<T> for Jet2<T> {
            #[inline]
            fn $method(&mut self, rhs: T) {
                *self = $bin::$bin_method(*self, rhs);
            }
        }
    };
}

impl_jet2_assign!(AddAssign, add_assign, Add, add);
impl_jet2_assign!(SubAssign, sub_assign, Sub, sub);
impl_jet2_assign!(MulAssign, mul_assign, Mul, mul);
impl_jet2_assign!(DivAssign, div_assign, Div, div);

// --- Display / Debug / Default ---
impl<T: Passive> fmt::Display for Jet2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<T: Passive> fmt::Debug for Jet2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Jet2(v={}, d1={}, d2={})",
            self.value, self.d1, self.d2
        )
    }
}

impl<T: Passive> Default for Jet2<T> {
    fn default() -> Self {
        Jet2::constant(T::zero())
    }
}

impl<T: Passive> From<T> for Jet2<T> {
    fn from(value: T) -> Self {
        Jet2::constant(value)
    }
}

impl From<i32> for Jet2<f64> {
    fn from(value: i32) -> Self {
        Jet2::constant(value as f64)
    }
}

impl<T: Passive> PartialEq for Jet2<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Passive> PartialOrd for Jet2<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

// ============================================================================
// impl Real for Jet2<f64>
// ============================================================================
//
// `Jet2<T>` is `Copy` (when `T: Copy`) and exposes its transcendentals as
// value-taking inherent methods. The `Real` trait's `&self`-taking methods
// dereference through `Copy` to reach them. `Real::powf(&self, exponent: Self)`
// requires a two-`Jet2` power, which is implemented as
// `exp(exponent * ln(self))` — both factors propagate first and second
// derivatives through the existing primitives. `Real::powi` delegates to the
// inherent `Jet2::powf(self, T)` with a lossless `i32` → `f64` cast.

/// Stamp one unary `Real` method body for `Jet2<f64>` from a crate-wide
/// derivative-table entry: delegate to the `Jet2` inherent method of the same
/// name, which the same table stamps.
macro_rules! impl_real_jet2_unary {
    ($name:ident, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn $name(&self) -> Self {
            Jet2::$name(*self)
        }
    };
}

impl crate::real::Real for Jet2<f64> {
    type Passive = f64;

    #[inline]
    fn value(&self) -> f64 {
        self.value
    }
    /// Zero value with zero first and second derivatives — a constant, not
    /// a seeded variable.
    #[inline]
    fn zero() -> Self {
        Jet2::constant(0.0)
    }
    /// Unit value with zero first and second derivatives — a constant, not
    /// a seeded variable.
    #[inline]
    fn one() -> Self {
        Jet2::constant(1.0)
    }
    #[inline]
    fn powf(&self, exponent: Self) -> Self {
        // Both derivatives come from `exp(v · ln u)`, which propagates first and
        // second order through the existing primitives. The *value* is written
        // back from `powf`, because `exp(v · ln u)` rounds three times where
        // `powf` rounds once and the two land on different `f64`s — and `powf`
        // is what the passive scalar and every other mode return for the same
        // operands. Same shape as `Div`: derivatives from the composed form,
        // value from the passive reference. See `crate::real`.
        let mut out = Jet2::exp(exponent * Jet2::ln(*self));
        out.value = self.value.powf(exponent.value);
        out
    }
    #[inline]
    fn powi(&self, exponent: i32) -> Self {
        // An integer power is taken by `powi`, not by routing through `powf`:
        // `f64::powi` multiplies and `f64::powf` goes through exp/ln, so they
        // are different functions in the last bit for a majority of operands.
        // Every other mode reaches `bv.powi(n)` here, and so must this one.
        let v = self.value;
        let n = f64::from(exponent);
        let gp = n * v.powi(exponent - 1);
        let gpp = n * (n - 1.0) * v.powi(exponent - 2);
        Jet2 {
            value: v.powi(exponent),
            d1: gp * self.d1,
            d2: gpp * self.d1 * self.d1 + gp * self.d2,
        }
    }
    // Component-wise loops. Forward mode has no tape, so there is no fused
    // encoding to preserve — the accumulation propagates value and tangent
    // together through the same arithmetic a binary chain would use, at the
    // same cost.
    #[inline]
    fn sum(xs: &[Self]) -> Self {
        let mut acc = Jet2::constant(0.0);
        for x in xs {
            acc += *x;
        }
        acc
    }
    #[inline]
    fn dot(xs: &[Self], ys: &[Self]) -> Self {
        assert_eq!(xs.len(), ys.len(), "dot: slice length mismatch");
        let mut acc = Jet2::constant(0.0);
        for (x, y) in xs.iter().zip(ys) {
            acc += *x * *y;
        }
        acc
    }
    #[inline]
    fn weighted_sum(ws: &[Self::Passive], xs: &[Self]) -> Self {
        assert_eq!(ws.len(), xs.len(), "weighted_sum: slice length mismatch");
        let mut acc = Jet2::constant(0.0);
        for (&w, x) in ws.iter().zip(xs) {
            acc += Jet2::constant(w) * *x;
        }
        acc
    }
    #[inline]
    fn weighted_dot(ws: &[Self::Passive], xs: &[Self], ys: &[Self]) -> Self {
        assert_eq!(ws.len(), xs.len(), "weighted_dot: slice length mismatch");
        assert_eq!(xs.len(), ys.len(), "weighted_dot: slice length mismatch");
        let mut acc = Jet2::constant(0.0);
        for ((&w, x), y) in ws.iter().zip(xs).zip(ys) {
            acc += Jet2::constant(w) * *x * *y;
        }
        acc
    }
    #[inline]
    fn max(&self, other: &Self) -> Self {
        // Derivative follows the winning branch (payoff-max convention).
        if self.value >= other.value { *self } else { *other }
    }
    #[inline]
    fn min(&self, other: &Self) -> Self {
        if self.value <= other.value { *self } else { *other }
    }

    crate::elementaries::for_each_unary_elementary!(impl_real_jet2_unary);
}
