//! `Jet1` — active real for single-variable forward-mode AD.
//!
//! An [`Jet1<T>`] carries both a value and a single scalar tangent
//! (the derivative with respect to one seeded input direction).
//! Propagation uses the chain rule applied pointwise at every operator;
//! there is no tape, no thread-local state, and the type is entirely
//! stack-allocated and `Copy`-able modulo the inner scalar's `Clone`.
//!
//! Forward single-variable mode is the cheapest AD mode in this crate —
//! operators compile to a handful of fmul/fadds — so it's the right
//! choice when you have **one input direction** (seed `derivative = 1`
//! on exactly one variable) and you don't need the multi-direction
//! batching that [`JetK<T, K>`](crate::forward::jetk::JetK) provides.
//!
//! `Jet1<f64>` is also the *storage scalar* of the single-direction
//! forward-over-adjoint Hessian engine: it implements
//! [`Passive`] and
//! [`TapeStorage`](crate::tape::TapeStorage), so `Tape<Jet1<f64>>` is a
//! reverse-mode tape whose every recorded multiplier carries its own
//! tangent — see [`compute_hessian`](crate::compute_hessian).
//!
//! ```
//! use xad_rs::Jet1;
//!
//! // f(x) = x² + 3x, at x = 2
//! // f'(x) = 2x + 3 = 7
//! let x = Jet1::new(2.0_f64, 1.0);   // value 2, tangent 1
//! let f = &x * &x + &x * 3.0;
//! assert_eq!(f.value(), 10.0);
//! assert_eq!(f.derivative(), 7.0);
//! ```
//!
//! For **many** directions in a single forward pass, use
//! [`JetK<T, K>`](crate::forward::jetk::JetK) instead. For reverse mode, use
//! [`AReal`](crate::reverse::areal::AReal).

use crate::passive::Passive;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Active real number for forward-mode AD.
///
/// Stores a value and its tangent (derivative with respect to a seed direction).
#[derive(Clone, Copy)]
pub struct Jet1<T: Passive> {
    value: T,
    derivative: T,
}

impl<T: Passive> Jet1<T> {
    /// Create a new Jet1 with the given value and derivative.
    pub fn new(value: T, derivative: T) -> Self {
        Jet1 { value, derivative }
    }

    /// Create an Jet1 with zero derivative (a constant).
    pub fn constant(value: T) -> Self {
        Jet1 {
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

impl<T: Passive> From<T> for Jet1<T> {
    fn from(value: T) -> Self {
        Jet1::constant(value)
    }
}

impl From<i32> for Jet1<f64> {
    fn from(value: i32) -> Self {
        Jet1::constant(value as f64)
    }
}

// ============================================================================
// Operators — permutation-stamped
// ============================================================================
//
// `Jet1` is `Copy`, so every ref/value permutation of a binary operator is
// the same computation; each macro takes ONE by-value body and stamps the
// remaining permutations as deref-and-delegate wrappers.

// `Jet1 op Jet1`: 4 ref/value permutations from one body.
macro_rules! impl_jet1_binop {
    ($trait:ident, $method:ident, ($a:ident, $b:ident) => $body:expr) => {
        impl<T: Passive> $trait<Jet1<T>> for Jet1<T> {
            type Output = Jet1<T>;
            #[inline]
            fn $method(self, rhs: Jet1<T>) -> Jet1<T> {
                let ($a, $b) = (self, rhs);
                $body
            }
        }
        impl<T: Passive> $trait<&Jet1<T>> for Jet1<T> {
            type Output = Jet1<T>;
            #[inline]
            fn $method(self, rhs: &Jet1<T>) -> Jet1<T> {
                $trait::$method(self, *rhs)
            }
        }
        impl<T: Passive> $trait<Jet1<T>> for &Jet1<T> {
            type Output = Jet1<T>;
            #[inline]
            fn $method(self, rhs: Jet1<T>) -> Jet1<T> {
                $trait::$method(*self, rhs)
            }
        }
        impl<T: Passive> $trait<&Jet1<T>> for &Jet1<T> {
            type Output = Jet1<T>;
            #[inline]
            fn $method(self, rhs: &Jet1<T>) -> Jet1<T> {
                $trait::$method(*self, *rhs)
            }
        }
    };
}

// `Jet1 op T` (scalar RHS): value and ref LHS forms from one body.
macro_rules! impl_jet1_binop_scalar_rhs {
    ($trait:ident, $method:ident, ($a:ident, $r:ident) => $body:expr) => {
        impl<T: Passive> $trait<T> for Jet1<T> {
            type Output = Jet1<T>;
            #[inline]
            fn $method(self, rhs: T) -> Jet1<T> {
                let ($a, $r) = (self, rhs);
                $body
            }
        }
        impl<T: Passive> $trait<T> for &Jet1<T> {
            type Output = Jet1<T>;
            #[inline]
            fn $method(self, rhs: T) -> Jet1<T> {
                $trait::$method(*self, rhs)
            }
        }
    };
}

// `f64 op Jet1` (scalar LHS — the orphan rule forces a concrete float
// type): stamps both RHS ref forms from one body.
macro_rules! impl_scalar_lhs_jet1_binop {
    ($trait:ident, $method:ident, ($l:ident, $r:ident) => $body:expr) => {
        impl_scalar_lhs_jet1_binop!(@one f64, $trait, $method, ($l, $r) => $body);
    };
    (@one $t:ty, $trait:ident, $method:ident, ($l:ident, $r:ident) => $body:expr) => {
        impl $trait<Jet1<$t>> for $t {
            type Output = Jet1<$t>;
            #[inline]
            fn $method(self, rhs: Jet1<$t>) -> Jet1<$t> {
                let ($l, $r) = (self, rhs);
                $body
            }
        }
        impl $trait<&Jet1<$t>> for $t {
            type Output = Jet1<$t>;
            #[inline]
            fn $method(self, rhs: &Jet1<$t>) -> Jet1<$t> {
                $trait::$method(self, *rhs)
            }
        }
    };
}

// Compound assigns delegate to the binary op (free for a `Copy` type).
macro_rules! impl_jet1_assign {
    ($trait:ident, $method:ident, $bin:ident, $bin_method:ident) => {
        impl<T: Passive> $trait for Jet1<T> {
            #[inline]
            fn $method(&mut self, rhs: Jet1<T>) {
                *self = $bin::$bin_method(*self, rhs);
            }
        }
        impl<T: Passive> $trait<&Jet1<T>> for Jet1<T> {
            #[inline]
            fn $method(&mut self, rhs: &Jet1<T>) {
                *self = $bin::$bin_method(*self, *rhs);
            }
        }
        impl<T: Passive> $trait<T> for Jet1<T> {
            #[inline]
            fn $method(&mut self, rhs: T) {
                *self = $bin::$bin_method(*self, rhs);
            }
        }
    };
}

impl_jet1_binop!(Add, add, (a, b) => Jet1 {
    value: a.value + b.value,
    derivative: a.derivative + b.derivative,
});
impl_jet1_binop!(Sub, sub, (a, b) => Jet1 {
    value: a.value - b.value,
    derivative: a.derivative - b.derivative,
});
impl_jet1_binop!(Mul, mul, (a, b) => Jet1 {
    // d(a*b) = a'·b + a·b'
    value: a.value * b.value,
    derivative: a.derivative * b.value + a.value * b.derivative,
});
impl_jet1_binop!(Div, div, (a, b) => {
    // d(a/b) = (a'·b - a·b') / b²
    let inv_b = T::one() / b.value;
    Jet1 {
        value: a.value * inv_b,
        derivative: (a.derivative * b.value - a.value * b.derivative) * inv_b * inv_b,
    }
});

impl_jet1_binop_scalar_rhs!(Add, add, (a, r) => Jet1 {
    value: a.value + r,
    derivative: a.derivative,
});
impl_jet1_binop_scalar_rhs!(Sub, sub, (a, r) => Jet1 {
    value: a.value - r,
    derivative: a.derivative,
});
impl_jet1_binop_scalar_rhs!(Mul, mul, (a, r) => Jet1 {
    value: a.value * r,
    derivative: a.derivative * r,
});
impl_jet1_binop_scalar_rhs!(Div, div, (a, r) => {
    let inv = T::one() / r;
    Jet1 {
        value: a.value * inv,
        derivative: a.derivative * inv,
    }
});

impl_scalar_lhs_jet1_binop!(Add, add, (l, r) => Jet1 {
    value: l + r.value,
    derivative: r.derivative,
});
impl_scalar_lhs_jet1_binop!(Sub, sub, (l, r) => Jet1 {
    value: l - r.value,
    derivative: -r.derivative,
});
impl_scalar_lhs_jet1_binop!(Mul, mul, (l, r) => Jet1 {
    value: l * r.value,
    derivative: l * r.derivative,
});
impl_scalar_lhs_jet1_binop!(Div, div, (l, r) => {
    // d(c/x) = -c·x' / x²
    let inv = 1.0 / r.value;
    Jet1 {
        value: l * inv,
        derivative: -l * r.derivative * inv * inv,
    }
});

impl_jet1_assign!(AddAssign, add_assign, Add, add);
impl_jet1_assign!(SubAssign, sub_assign, Sub, sub);
impl_jet1_assign!(MulAssign, mul_assign, Mul, mul);
impl_jet1_assign!(DivAssign, div_assign, Div, div);

// Negation
impl<T: Passive> Neg for Jet1<T> {
    type Output = Jet1<T>;
    #[inline]
    fn neg(self) -> Jet1<T> {
        Jet1 {
            value: -self.value,
            derivative: -self.derivative,
        }
    }
}

impl<T: Passive> Neg for &Jet1<T> {
    type Output = Jet1<T>;
    #[inline]
    fn neg(self) -> Jet1<T> {
        -*self
    }
}

// Comparison
impl<T: Passive> PartialEq for Jet1<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Passive> PartialEq<T> for Jet1<T> {
    fn eq(&self, other: &T) -> bool {
        self.value == *other
    }
}

impl<T: Passive> PartialOrd for Jet1<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: Passive> PartialOrd<T> for Jet1<T> {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(other)
    }
}

impl<T: Passive> fmt::Display for Jet1<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<T: Passive> fmt::Debug for Jet1<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Jet1({}, deriv={})", self.value, self.derivative)
    }
}

impl<T: Passive> Default for Jet1<T> {
    fn default() -> Self {
        Jet1::constant(T::zero())
    }
}

// ============================================================================
// impl Real for Jet1<f64>
// ============================================================================
//
// Delegates to `crate::math::fwd::*` free functions which propagate the
// tangent component via the chain rule. `powf` uses the two-Jet1 `pow` (not
// the Jet1-and-passive `powf`) because `Real::powf` requires `exponent: Self`.

/// Stamp one unary `Real` method body for `Jet1<f64>` from a crate-wide
/// derivative-table entry: delegate to the forward-mode free function of the
/// same name in `math::fwd`, which the same table stamps.
macro_rules! impl_real_jet1_unary {
    ($name:ident, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn $name(&self) -> Self {
            crate::math::fwd::$name(self)
        }
    };
}

impl crate::real::Real for Jet1<f64> {
    type Passive = f64;

    #[inline]
    fn value(&self) -> f64 {
        self.value
    }
    /// Zero value, zero tangent — a constant, not a seeded variable.
    #[inline]
    fn zero() -> Self {
        Jet1::constant(0.0)
    }
    /// Unit value, zero tangent — a constant, not a seeded variable.
    #[inline]
    fn one() -> Self {
        Jet1::constant(1.0)
    }
    #[inline]
    fn powf(&self, exponent: Self) -> Self {
        crate::math::fwd::pow(self, &exponent)
    }
    #[inline]
    fn powi(&self, exponent: i32) -> Self {
        crate::math::fwd::powi(self, exponent)
    }
    // Component-wise loops. Forward mode has no tape, so there is no fused
    // encoding to preserve — the accumulation propagates value and tangent
    // together through the same arithmetic a binary chain would use, at the
    // same cost.
    #[inline]
    fn sum(xs: &[Self]) -> Self {
        let mut acc = Jet1::constant(0.0);
        for x in xs {
            acc += *x;
        }
        acc
    }
    #[inline]
    fn dot(xs: &[Self], ys: &[Self]) -> Self {
        assert_eq!(xs.len(), ys.len(), "dot: slice length mismatch");
        let mut acc = Jet1::constant(0.0);
        for (x, y) in xs.iter().zip(ys) {
            acc += *x * *y;
        }
        acc
    }
    #[inline]
    fn weighted_sum(ws: &[Self::Passive], xs: &[Self]) -> Self {
        assert_eq!(ws.len(), xs.len(), "weighted_sum: slice length mismatch");
        let mut acc = Jet1::constant(0.0);
        for (&w, x) in ws.iter().zip(xs) {
            acc += Jet1::constant(w) * *x;
        }
        acc
    }
    #[inline]
    fn weighted_dot(ws: &[Self::Passive], xs: &[Self], ys: &[Self]) -> Self {
        assert_eq!(ws.len(), xs.len(), "weighted_dot: slice length mismatch");
        assert_eq!(xs.len(), ys.len(), "weighted_dot: slice length mismatch");
        let mut acc = Jet1::constant(0.0);
        for ((&w, x), y) in ws.iter().zip(xs).zip(ys) {
            acc += Jet1::constant(w) * *x * *y;
        }
        acc
    }
    #[inline]
    fn max(&self, other: &Self) -> Self {
        crate::math::fwd::max(self, other)
    }
    #[inline]
    fn min(&self, other: &Self) -> Self {
        crate::math::fwd::min(self, other)
    }

    crate::elementaries::for_each_unary_elementary!(impl_real_jet1_unary);
}
