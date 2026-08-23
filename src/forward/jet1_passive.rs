//! `Jet1<T>` as a [`Passive`] storage scalar — the foundation of the
//! forward-over-adjoint (nested) exact-Hessian engine.
//!
//! Implementing the full `num_traits` stack for `Jet1<T>` makes `Jet1<f64>`
//! satisfy the [`Passive`] bound, so the reverse-mode [`Tape`](crate::Tape) and
//! [`AReal`](crate::AReal) can be instantiated as `Tape<Jet1<f64>>` /
//! `AReal<Jet1<f64>>`: a tape whose *storage* scalar is itself a forward dual.
//! Recording with one input's tangent seeded to `1` and sweeping then yields,
//! in each input's adjoint, the gradient (value part) and one Hessian column
//! (tangent part) — exact forward-over-adjoint second order, reusing every
//! existing operator and all of `math.rs`.
//!
//! This is orthogonal to the [`Real`](crate::Real) trait `Jet1<f64>` also
//! implements: `Passive` is the *storage* bound (num-traits `Float`), `Real` is
//! the *active* surface. The method-name overlap (`sin`, `cos`, …) is resolved
//! by which trait the bound names — generic code over `T: Passive` uses the
//! `Float` methods; nothing calls a `Real` method through a `Passive` bound.

use num_traits::{Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use std::num::FpCategory;
use std::ops::{Rem, RemAssign};

use crate::forward::jet1::Jet1;
use crate::math;
use crate::passive::Passive;

// ---- Rem / RemAssign (needed for Num / NumAssign) ----

impl<T: Passive> Rem for Jet1<T> {
    type Output = Jet1<T>;
    #[inline]
    fn rem(self, rhs: Jet1<T>) -> Jet1<T> {
        let q = (self.value() / rhs.value()).trunc();
        Jet1::new(self.value() % rhs.value(), self.derivative() - q * rhs.derivative())
    }
}

impl<T: Passive> RemAssign for Jet1<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: Jet1<T>) {
        *self = *self % rhs;
    }
}

// ---- Zero / One ----

impl<T: Passive> Zero for Jet1<T> {
    #[inline]
    fn zero() -> Self {
        Jet1::constant(T::zero())
    }
    /// Both value **and** tangent must be zero — essential for the reverse
    /// sweep's zero-adjoint skip: a dual adjoint with value 0 but nonzero
    /// tangent carries a real second-order contribution and must not be skipped.
    #[inline]
    fn is_zero(&self) -> bool {
        self.value().is_zero() && self.derivative().is_zero()
    }
}

impl<T: Passive> One for Jet1<T> {
    #[inline]
    fn one() -> Self {
        Jet1::constant(T::one())
    }
}

// ---- Num ----

impl<T: Passive> Num for Jet1<T> {
    type FromStrRadixErr = T::FromStrRadixErr;
    #[inline]
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(s, radix).map(Jet1::constant)
    }
}

// ---- ToPrimitive / NumCast / FromPrimitive (operate on the value part) ----

impl<T: Passive> ToPrimitive for Jet1<T> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.value().to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.value().to_u64()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.value().to_f64()
    }
}

impl<T: Passive> NumCast for Jet1<T> {
    #[inline]
    fn from<N: ToPrimitive>(n: N) -> Option<Self> {
        <T as NumCast>::from(n).map(Jet1::constant)
    }
}

impl<T: Passive> FromPrimitive for Jet1<T> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        T::from_i64(n).map(Jet1::constant)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        T::from_u64(n).map(Jet1::constant)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        T::from_f64(n).map(Jet1::constant)
    }
}

// ---- Float ----

impl<T: Passive> Float for Jet1<T> {
    #[inline]
    fn nan() -> Self { Jet1::constant(T::nan()) }
    #[inline]
    fn infinity() -> Self { Jet1::constant(T::infinity()) }
    #[inline]
    fn neg_infinity() -> Self { Jet1::constant(T::neg_infinity()) }
    #[inline]
    fn neg_zero() -> Self { Jet1::constant(T::neg_zero()) }
    #[inline]
    fn min_value() -> Self { Jet1::constant(T::min_value()) }
    #[inline]
    fn min_positive_value() -> Self { Jet1::constant(T::min_positive_value()) }
    #[inline]
    fn max_value() -> Self { Jet1::constant(T::max_value()) }
    #[inline]
    fn epsilon() -> Self { Jet1::constant(T::epsilon()) }

    #[inline]
    fn is_nan(self) -> bool { self.value().is_nan() }
    #[inline]
    fn is_infinite(self) -> bool { self.value().is_infinite() }
    #[inline]
    fn is_finite(self) -> bool { self.value().is_finite() }
    #[inline]
    fn is_normal(self) -> bool { self.value().is_normal() }
    #[inline]
    fn is_sign_positive(self) -> bool { self.value().is_sign_positive() }
    #[inline]
    fn is_sign_negative(self) -> bool { self.value().is_sign_negative() }
    #[inline]
    fn classify(self) -> FpCategory { self.value().classify() }
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) { self.value().integer_decode() }

    #[inline]
    fn floor(self) -> Self { Jet1::new(self.value().floor(), T::zero()) }
    #[inline]
    fn ceil(self) -> Self { Jet1::new(self.value().ceil(), T::zero()) }
    #[inline]
    fn round(self) -> Self { Jet1::new(self.value().round(), T::zero()) }
    #[inline]
    fn trunc(self) -> Self { Jet1::new(self.value().trunc(), T::zero()) }
    #[inline]
    fn signum(self) -> Self { Jet1::new(self.value().signum(), T::zero()) }
    #[inline]
    fn fract(self) -> Self { Jet1::new(self.value().fract(), self.derivative()) }
    #[inline]
    fn abs(self) -> Self {
        Jet1::new(self.value().abs(), self.derivative() * self.value().signum())
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        if self.value().is_sign_negative() != sign.value().is_sign_negative() {
            Jet1::new(-self.value(), -self.derivative())
        } else {
            self
        }
    }
    #[inline]
    fn to_degrees(self) -> Self {
        Jet1::new(self.value().to_degrees(), self.derivative().to_degrees())
    }
    #[inline]
    fn to_radians(self) -> Self {
        Jet1::new(self.value().to_radians(), self.derivative().to_radians())
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self.value() >= other.value() { self } else { other }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self.value() <= other.value() { self } else { other }
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    #[inline]
    fn recip(self) -> Self {
        let inv = T::one() / self.value();
        Jet1::new(inv, -self.derivative() * inv * inv)
    }

    // transcendentals delegate to the tested forward-mode functions
    #[inline]
    fn powi(self, n: i32) -> Self { math::fwd::powi(&self, n) }
    #[inline]
    fn powf(self, n: Self) -> Self { math::fwd::pow(&self, &n) }
    #[inline]
    fn sqrt(self) -> Self { math::fwd::sqrt(&self) }
    #[inline]
    fn cbrt(self) -> Self { math::fwd::cbrt(&self) }
    #[inline]
    fn exp(self) -> Self { math::fwd::exp(&self) }
    #[inline]
    fn exp2(self) -> Self { math::fwd::exp2(&self) }
    #[inline]
    fn exp_m1(self) -> Self { math::fwd::exp_m1(&self) }
    #[inline]
    fn ln(self) -> Self { math::fwd::ln(&self) }
    #[inline]
    fn ln_1p(self) -> Self { math::fwd::ln_1p(&self) }
    #[inline]
    fn log(self, base: Self) -> Self { math::fwd::ln(&self) / math::fwd::ln(&base) }
    #[inline]
    fn log2(self) -> Self { math::fwd::log2(&self) }
    #[inline]
    fn log10(self) -> Self { math::fwd::log10(&self) }
    #[inline]
    fn hypot(self, other: Self) -> Self { math::fwd::hypot(&self, &other) }
    #[inline]
    fn sin(self) -> Self { math::fwd::sin(&self) }
    #[inline]
    fn cos(self) -> Self { math::fwd::cos(&self) }
    #[inline]
    fn tan(self) -> Self { math::fwd::tan(&self) }
    #[inline]
    fn asin(self) -> Self { math::fwd::asin(&self) }
    #[inline]
    fn acos(self) -> Self { math::fwd::acos(&self) }
    #[inline]
    fn atan(self) -> Self { math::fwd::atan(&self) }
    #[inline]
    fn atan2(self, other: Self) -> Self { math::fwd::atan2(&self, &other) }
    #[inline]
    fn sin_cos(self) -> (Self, Self) { (math::fwd::sin(&self), math::fwd::cos(&self)) }
    #[inline]
    fn sinh(self) -> Self { math::fwd::sinh(&self) }
    #[inline]
    fn cosh(self) -> Self { math::fwd::cosh(&self) }
    #[inline]
    fn tanh(self) -> Self { math::fwd::tanh(&self) }
    #[inline]
    fn asinh(self) -> Self { math::fwd::asinh(&self) }
    #[inline]
    fn acosh(self) -> Self { math::fwd::acosh(&self) }
    #[inline]
    fn atanh(self) -> Self { math::fwd::atanh(&self) }
    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        if self.value() <= other.value() { Jet1::constant(T::zero()) } else { self - other }
    }
}

// ---- Passive ----
// With the full num_traits stack, `Jet1<T>` satisfies the storage bound.
//
// `erf_value` / `inv_norm_cdf_value` are overridden (instead of taking the
// polynomial defaults) so the tangent is the EXACT analytic derivative.
// Evaluating the A&S / Acklam polynomials in `Jet1` arithmetic would give a
// tangent equal to the polynomial's own derivative (error ~1e-6), which the
// forward-over-adjoint engine (`Tape<Jet1<f64>>`) then leaks into second-order
// results whenever a downstream op records the erf value as a multiplier.
impl<T: Passive> Passive for Jet1<T> {
    #[inline]
    fn erf_value(self) -> Self {
        let v = self.value();
        let d = T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap() * (-v * v).exp();
        Jet1::new(v.erf_value(), d * self.derivative())
    }

    #[inline]
    fn inv_norm_cdf_value(self) -> Self {
        let r = self.value().inv_norm_cdf_value();
        let d = T::one() / crate::math::norm_pdf(r);
        Jet1::new(r, d * self.derivative())
    }
}
