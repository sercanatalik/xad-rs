//! AD-aware transcendental functions.
//!
//! Provides `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`,
//! `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `exp2`, `ln`, `log2`, `log10`,
//! `ln_1p`, `exp_m1`, `sqrt`, `cbrt`, `abs`, `atan2`, `pow`, `powf`, `powi`,
//! `hypot`, `max`, `min`, `smooth_abs`, `smooth_max`, `smooth_min`, `erf`,
//! `erfc`, `norm_cdf`, and `inv_norm_cdf` — each in two AD-aware flavours with the correct chain-rule
//! derivative propagation already plumbed through:
//!
//! - [`ad`] — reverse-mode variants that operate on [`AReal`] and record
//!   onto the currently active tape.
//! - [`fwd`] — forward-mode variants that operate on [`FReal`].
//!
//! Transcendental methods directly on [`Dual`](crate::dual::Dual) and
//! [`Dual2`](crate::dual2::Dual2) live on those types (inherent methods),
//! not in this module.
//!
//! There is also a scalar [`erf`] that operates on a plain `f64` / `f32`
//! and is exposed at the module root for callers that need `erf` without
//! any AD layer (it uses the Abramowitz & Stegun 7.1.26 polynomial
//! approximation, accurate to ~1.5e-7).
//!
//! # Example
//!
//! ```
//! use xad_rs::AReal;
//! use xad_rs::Tape;
//! use xad_rs::math;
//!
//! let mut tape = Tape::<f64>::new(true);
//! tape.activate();
//!
//! let mut x = AReal::new(1.0_f64);
//! AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
//!
//! // f(x) = exp(sin(x)), f'(x) = cos(x) · exp(sin(x))
//! let f = math::ad::exp(&math::ad::sin(&x));
//! f.set_adjoint(&mut tape, 1.0);
//! tape.compute_adjoints();
//!
//! let expected = 1.0_f64.cos() * 1.0_f64.sin().exp();
//! assert!((x.adjoint(&tape) - expected).abs() < 1e-12);
//! # xad_rs::Tape::<f64>::deactivate_all();
//! ```

use crate::reverse::areal::{record_binary_op, record_unary_op, AReal};
use crate::forward::freal::FReal;
use crate::scalar::Scalar;
use crate::tape::TapeStorage;

// ============================================================================
// Macros for generating unary math functions
// ============================================================================

macro_rules! impl_unary_math {
    ($name:ident, $val_fn:expr, $deriv_fn:expr) => {
        #[inline]
        pub fn $name<T: TapeStorage>(x: &AReal<T>) -> AReal<T> {
            let v = x.value();
            let result = $val_fn(v);
            let deriv = $deriv_fn(v, result);
            record_unary_op(result, x.slot(), deriv)
        }
    };
}

macro_rules! impl_unary_math_fwd {
    ($name:ident, $val_fn:expr, $deriv_fn:expr) => {
        #[inline]
        pub fn $name<T: Scalar>(x: &FReal<T>) -> FReal<T> {
            let v = x.value();
            let result = $val_fn(v);
            let deriv = $deriv_fn(v, result);
            FReal::new(result, deriv * x.derivative())
        }
    };
}

/// AD-aware math functions for `AReal` (reverse mode).
pub mod ad {
    use super::*;

    impl_unary_math!(sin, |x: T| x.sin(), |x: T, _r: T| x.cos());
    impl_unary_math!(cos, |x: T| x.cos(), |x: T, _r: T| -x.sin());
    impl_unary_math!(tan, |x: T| x.tan(), |_x: T, r: T| T::one() + r * r);
    impl_unary_math!(asin, |x: T| x.asin(), |x: T, _r: T| T::one() / (T::one() - x * x).sqrt());
    impl_unary_math!(acos, |x: T| x.acos(), |x: T, _r: T| -T::one() / (T::one() - x * x).sqrt());
    impl_unary_math!(atan, |x: T| x.atan(), |x: T, _r: T| T::one() / (T::one() + x * x));

    impl_unary_math!(sinh, |x: T| x.sinh(), |x: T, _r: T| x.cosh());
    impl_unary_math!(cosh, |x: T| x.cosh(), |x: T, _r: T| x.sinh());
    impl_unary_math!(tanh, |x: T| x.tanh(), |_x: T, r: T| T::one() - r * r);
    impl_unary_math!(asinh, |x: T| x.asinh(), |x: T, _r: T| T::one() / (x * x + T::one()).sqrt());
    impl_unary_math!(acosh, |x: T| x.acosh(), |x: T, _r: T| T::one() / (x * x - T::one()).sqrt());
    impl_unary_math!(atanh, |x: T| x.atanh(), |x: T, _r: T| T::one() / (T::one() - x * x));

    impl_unary_math!(exp, |x: T| x.exp(), |_x: T, r: T| r);
    impl_unary_math!(exp2, |x: T| x.exp2(), |_x: T, r: T| r * T::from(2.0).unwrap().ln());
    impl_unary_math!(ln, |x: T| x.ln(), |x: T, _r: T| T::one() / x);
    impl_unary_math!(log2, |x: T| x.log2(), |x: T, _r: T| T::one() / (x * T::from(2.0).unwrap().ln()));
    impl_unary_math!(log10, |x: T| x.log10(), |x: T, _r: T| T::one() / (x * T::from(10.0).unwrap().ln()));
    impl_unary_math!(ln_1p, |x: T| x.ln_1p(), |x: T, _r: T| T::one() / (T::one() + x));
    impl_unary_math!(exp_m1, |x: T| x.exp() - T::one(), |_x: T, r: T| r + T::one());

    impl_unary_math!(sqrt, |x: T| x.sqrt(), |_x: T, r: T| T::from(0.5).unwrap() / r);
    impl_unary_math!(cbrt, |x: T| x.cbrt(), |_x: T, r: T| T::one() / (T::from(3.0).unwrap() * r * r));
    impl_unary_math!(abs, |x: T| x.abs(), |x: T, _r: T| if x >= T::zero() { T::one() } else { -T::one() });

    #[inline]
    pub fn atan2<T: TapeStorage>(y: &AReal<T>, x: &AReal<T>) -> AReal<T> {
        let yv = y.value();
        let xv = x.value();
        let result = yv.atan2(xv);
        let denom = xv * xv + yv * yv;
        record_binary_op(result, y.slot(), xv / denom, x.slot(), -yv / denom)
    }

    #[inline]
    pub fn pow<T: TapeStorage>(base: &AReal<T>, exponent: &AReal<T>) -> AReal<T> {
        let bv = base.value();
        let ev = exponent.value();
        let result = bv.powf(ev);
        let d_base = ev * bv.powf(ev - T::one());
        let d_exp = result * bv.ln();
        record_binary_op(result, base.slot(), d_base, exponent.slot(), d_exp)
    }

    #[inline]
    pub fn powf<T: TapeStorage>(base: &AReal<T>, exponent: T) -> AReal<T> {
        let bv = base.value();
        let result = bv.powf(exponent);
        let deriv = exponent * bv.powf(exponent - T::one());
        record_unary_op(result, base.slot(), deriv)
    }

    #[inline]
    pub fn powi<T: TapeStorage>(base: &AReal<T>, exponent: i32) -> AReal<T> {
        let bv = base.value();
        let result = bv.powi(exponent);
        let deriv = T::from(exponent).unwrap() * bv.powi(exponent - 1);
        record_unary_op(result, base.slot(), deriv)
    }

    #[inline]
    pub fn hypot<T: TapeStorage>(x: &AReal<T>, y: &AReal<T>) -> AReal<T> {
        let xv = x.value();
        let yv = y.value();
        let result = xv.hypot(yv);
        let inv_r = T::one() / result;
        record_binary_op(result, x.slot(), xv * inv_r, y.slot(), yv * inv_r)
    }

    /// `max(a, b)` with correct adjoint propagation.
    ///
    /// Records a **unary** op on the live branch only — recording a binary
    /// op with a zero multiplier on the inactive branch would waste one tape
    /// slot and one multiply on every reverse sweep.
    #[inline]
    pub fn max<T: TapeStorage>(a: &AReal<T>, b: &AReal<T>) -> AReal<T> {
        if a.value() >= b.value() {
            record_unary_op(a.value(), a.slot(), T::one())
        } else {
            record_unary_op(b.value(), b.slot(), T::one())
        }
    }

    /// `min(a, b)` with correct adjoint propagation. See [`max`] for the
    /// rationale behind the unary (rather than binary-with-zero) encoding.
    #[inline]
    pub fn min<T: TapeStorage>(a: &AReal<T>, b: &AReal<T>) -> AReal<T> {
        if a.value() <= b.value() {
            record_unary_op(a.value(), a.slot(), T::one())
        } else {
            record_unary_op(b.value(), b.slot(), T::one())
        }
    }

    pub fn smooth_abs<T: TapeStorage>(x: &AReal<T>, c: T) -> AReal<T> {
        let xv = x.value();
        let result = (xv * xv + c).sqrt();
        let deriv = xv / result;
        record_unary_op(result, x.slot(), deriv)
    }

    pub fn smooth_max<T: TapeStorage>(a: &AReal<T>, b: &AReal<T>, c: T) -> AReal<T> {
        let half = T::from(0.5).unwrap();
        let diff = a - b;
        let sa = smooth_abs(&diff, c);
        let sum = a + b;
        (sum + sa) * half
    }

    pub fn smooth_min<T: TapeStorage>(a: &AReal<T>, b: &AReal<T>, c: T) -> AReal<T> {
        let half = T::from(0.5).unwrap();
        let diff = a - b;
        let sa = smooth_abs(&diff, c);
        let sum = a + b;
        (sum - sa) * half
    }

    #[inline]
    pub fn erf<T: TapeStorage>(x: &AReal<T>) -> AReal<T> {
        let xv = x.value();
        let result = super::erf(xv);
        let two_over_sqrt_pi = T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap();
        let deriv = two_over_sqrt_pi * (-xv * xv).exp();
        record_unary_op(result, x.slot(), deriv)
    }

    #[inline]
    pub fn erfc<T: TapeStorage>(x: &AReal<T>) -> AReal<T> {
        let xv = x.value();
        let result = T::one() - super::erf(xv);
        let two_over_sqrt_pi = T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap();
        let deriv = -two_over_sqrt_pi * (-xv * xv).exp();
        record_unary_op(result, x.slot(), deriv)
    }

    /// Standard normal CDF: `Φ(x) = 0.5 · (1 + erf(x / √2))`.
    ///
    /// Derivative: `φ(x) = (1/√(2π)) · exp(-x²/2)` (the standard normal PDF).
    #[inline]
    pub fn norm_cdf<T: TapeStorage>(x: &AReal<T>) -> AReal<T> {
        let xv = x.value();
        let result = super::norm_cdf(xv);
        let deriv = super::norm_pdf(xv);
        record_unary_op(result, x.slot(), deriv)
    }

    /// Inverse standard normal CDF: `Φ⁻¹(p)`.
    ///
    /// Derivative: `1 / φ(Φ⁻¹(p)) = √(2π) · exp(Φ⁻¹(p)² / 2)`.
    #[inline]
    pub fn inv_norm_cdf<T: TapeStorage>(x: &AReal<T>) -> AReal<T> {
        let xv = x.value();
        let result = super::inv_norm_cdf(xv);
        let deriv = T::one() / super::norm_pdf(result);
        record_unary_op(result, x.slot(), deriv)
    }

    pub fn is_nan<T: TapeStorage>(x: &AReal<T>) -> bool { x.value().is_nan() }
    pub fn is_infinite<T: TapeStorage>(x: &AReal<T>) -> bool { x.value().is_infinite() }
    pub fn is_finite<T: TapeStorage>(x: &AReal<T>) -> bool { x.value().is_finite() }
    pub fn is_normal<T: TapeStorage>(x: &AReal<T>) -> bool { x.value().is_normal() }
    pub fn signum<T: TapeStorage>(x: &AReal<T>) -> T { x.value().signum() }
    pub fn floor<T: TapeStorage>(x: &AReal<T>) -> T { x.value().floor() }
    pub fn ceil<T: TapeStorage>(x: &AReal<T>) -> T { x.value().ceil() }
    pub fn round<T: TapeStorage>(x: &AReal<T>) -> T { x.value().round() }
    pub fn trunc<T: TapeStorage>(x: &AReal<T>) -> T { x.value().trunc() }
    pub fn fract<T: TapeStorage>(x: &AReal<T>) -> T { x.value().fract() }
}

/// AD-aware math functions for `FReal` (forward mode).
pub mod fwd {
    use super::*;

    impl_unary_math_fwd!(sin, |x: T| x.sin(), |x: T, _r: T| x.cos());
    impl_unary_math_fwd!(cos, |x: T| x.cos(), |x: T, _r: T| -x.sin());
    impl_unary_math_fwd!(tan, |x: T| x.tan(), |_x: T, r: T| T::one() + r * r);
    impl_unary_math_fwd!(asin, |x: T| x.asin(), |x: T, _r: T| T::one() / (T::one() - x * x).sqrt());
    impl_unary_math_fwd!(acos, |x: T| x.acos(), |x: T, _r: T| -T::one() / (T::one() - x * x).sqrt());
    impl_unary_math_fwd!(atan, |x: T| x.atan(), |x: T, _r: T| T::one() / (T::one() + x * x));

    impl_unary_math_fwd!(sinh, |x: T| x.sinh(), |x: T, _r: T| x.cosh());
    impl_unary_math_fwd!(cosh, |x: T| x.cosh(), |x: T, _r: T| x.sinh());
    impl_unary_math_fwd!(tanh, |x: T| x.tanh(), |_x: T, r: T| T::one() - r * r);
    impl_unary_math_fwd!(asinh, |x: T| x.asinh(), |x: T, _r: T| T::one() / (x * x + T::one()).sqrt());
    impl_unary_math_fwd!(acosh, |x: T| x.acosh(), |x: T, _r: T| T::one() / (x * x - T::one()).sqrt());
    impl_unary_math_fwd!(atanh, |x: T| x.atanh(), |x: T, _r: T| T::one() / (T::one() - x * x));

    impl_unary_math_fwd!(exp, |x: T| x.exp(), |_x: T, r: T| r);
    impl_unary_math_fwd!(exp2, |x: T| x.exp2(), |_x: T, r: T| r * T::from(2.0).unwrap().ln());
    impl_unary_math_fwd!(ln, |x: T| x.ln(), |x: T, _r: T| T::one() / x);
    impl_unary_math_fwd!(log2, |x: T| x.log2(), |x: T, _r: T| T::one() / (x * T::from(2.0).unwrap().ln()));
    impl_unary_math_fwd!(log10, |x: T| x.log10(), |x: T, _r: T| T::one() / (x * T::from(10.0).unwrap().ln()));
    impl_unary_math_fwd!(ln_1p, |x: T| x.ln_1p(), |x: T, _r: T| T::one() / (T::one() + x));
    impl_unary_math_fwd!(exp_m1, |x: T| x.exp() - T::one(), |_x: T, r: T| r + T::one());

    impl_unary_math_fwd!(sqrt, |x: T| x.sqrt(), |_x: T, r: T| T::from(0.5).unwrap() / r);
    impl_unary_math_fwd!(cbrt, |x: T| x.cbrt(), |_x: T, r: T| T::one() / (T::from(3.0).unwrap() * r * r));
    impl_unary_math_fwd!(abs, |x: T| x.abs(), |x: T, _r: T| if x >= T::zero() { T::one() } else { -T::one() });

    #[inline]
    pub fn atan2<T: Scalar>(y: &FReal<T>, x: &FReal<T>) -> FReal<T> {
        let yv = y.value();
        let xv = x.value();
        let result = yv.atan2(xv);
        let denom = xv * xv + yv * yv;
        let deriv = (xv * y.derivative() - yv * x.derivative()) / denom;
        FReal::new(result, deriv)
    }

    #[inline]
    pub fn pow<T: Scalar>(base: &FReal<T>, exponent: &FReal<T>) -> FReal<T> {
        let bv = base.value();
        let ev = exponent.value();
        let result = bv.powf(ev);
        let d_base = ev * bv.powf(ev - T::one());
        let d_exp = result * bv.ln();
        FReal::new(result, d_base * base.derivative() + d_exp * exponent.derivative())
    }

    #[inline]
    pub fn powf<T: Scalar>(base: &FReal<T>, exponent: T) -> FReal<T> {
        let bv = base.value();
        let result = bv.powf(exponent);
        let deriv = exponent * bv.powf(exponent - T::one());
        FReal::new(result, deriv * base.derivative())
    }

    #[inline]
    pub fn powi<T: Scalar>(base: &FReal<T>, exponent: i32) -> FReal<T> {
        let bv = base.value();
        let result = bv.powi(exponent);
        let deriv = T::from(exponent).unwrap() * bv.powi(exponent - 1);
        FReal::new(result, deriv * base.derivative())
    }

    #[inline]
    pub fn hypot<T: Scalar>(x: &FReal<T>, y: &FReal<T>) -> FReal<T> {
        let xv = x.value();
        let yv = y.value();
        let result = xv.hypot(yv);
        let inv_r = T::one() / result;
        let deriv = xv * inv_r * x.derivative() + yv * inv_r * y.derivative();
        FReal::new(result, deriv)
    }

    pub fn max<T: Scalar>(a: &FReal<T>, b: &FReal<T>) -> FReal<T> {
        if a.value() >= b.value() { a.clone() } else { b.clone() }
    }

    pub fn min<T: Scalar>(a: &FReal<T>, b: &FReal<T>) -> FReal<T> {
        if a.value() <= b.value() { a.clone() } else { b.clone() }
    }

    pub fn smooth_abs<T: Scalar>(x: &FReal<T>, c: T) -> FReal<T> {
        let xv = x.value();
        let result = (xv * xv + c).sqrt();
        let deriv = xv / result;
        FReal::new(result, deriv * x.derivative())
    }

    #[inline]
    pub fn erf<T: Scalar>(x: &FReal<T>) -> FReal<T> {
        let xv = x.value();
        let result = super::erf(xv);
        let two_over_sqrt_pi = T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap();
        let deriv = two_over_sqrt_pi * (-xv * xv).exp();
        FReal::new(result, deriv * x.derivative())
    }

    #[inline]
    pub fn erfc<T: Scalar>(x: &FReal<T>) -> FReal<T> {
        let xv = x.value();
        let result = T::one() - super::erf(xv);
        let two_over_sqrt_pi = T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap();
        let deriv = -two_over_sqrt_pi * (-xv * xv).exp();
        FReal::new(result, deriv * x.derivative())
    }

    /// Standard normal CDF for `FReal` (forward mode).
    #[inline]
    pub fn norm_cdf<T: Scalar>(x: &FReal<T>) -> FReal<T> {
        let xv = x.value();
        let result = super::norm_cdf(xv);
        let deriv = super::norm_pdf(xv);
        FReal::new(result, deriv * x.derivative())
    }

    /// Inverse standard normal CDF for `FReal` (forward mode).
    #[inline]
    pub fn inv_norm_cdf<T: Scalar>(x: &FReal<T>) -> FReal<T> {
        let xv = x.value();
        let result = super::inv_norm_cdf(xv);
        let deriv = T::one() / super::norm_pdf(result);
        FReal::new(result, deriv * x.derivative())
    }
}

/// Error function `erf(x)`, computed with the Abramowitz & Stegun 7.1.26
/// polynomial approximation (accurate to ~1.5e-7).
///
/// Exposed at `xad_rs::math::erf` so examples can call it on a plain scalar.
/// The AD-aware variants live in [`ad::erf`] and [`fwd::erf`].
#[inline]
pub fn erf<T: Scalar>(x: T) -> T {
    let a1 = T::from(0.254829592).unwrap();
    let a2 = T::from(-0.284496736).unwrap();
    let a3 = T::from(1.421413741).unwrap();
    let a4 = T::from(-1.453152027).unwrap();
    let a5 = T::from(1.061405429).unwrap();
    let p = T::from(0.3275911).unwrap();

    let sign = if x < T::zero() { -T::one() } else { T::one() };
    let x = x.abs();
    let t = T::one() / (T::one() + p * x);
    let y = T::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Standard normal PDF: `φ(x) = (1/√(2π)) · exp(-x²/2)`.
///
/// Used internally by `norm_cdf` and `inv_norm_cdf` AD variants for the
/// derivative. Exposed publicly for callers that need the density on a
/// plain scalar.
#[inline]
pub fn norm_pdf<T: Scalar>(x: T) -> T {
    let inv_sqrt_2pi = T::from(std::f64::consts::FRAC_2_SQRT_PI * 0.5).unwrap();
    inv_sqrt_2pi * (T::from(-0.5).unwrap() * x * x).exp()
}

/// Standard normal CDF: `Φ(x) = 0.5 · (1 + erf(x / √2))`.
///
/// Uses the same A&S 7.1.26 `erf` approximation as [`erf`], so the
/// value accuracy is ~1.5e-7. AD-aware variants live in [`ad::norm_cdf`]
/// and [`fwd::norm_cdf`].
#[inline]
pub fn norm_cdf<T: Scalar>(x: T) -> T {
    let half = T::from(0.5).unwrap();
    let frac_1_sqrt_2 = T::from(std::f64::consts::FRAC_1_SQRT_2).unwrap();
    half * (T::one() + erf(x * frac_1_sqrt_2))
}

/// Inverse standard normal CDF: `Φ⁻¹(p)`.
///
/// Uses the rational approximation from Peter Acklam (accurate to ~1.15e-9
/// over the full `(0, 1)` domain). No external dependencies.
///
/// # Panics
///
/// Panics if `p` is outside `(0, 1)` (exclusive).
#[inline]
pub fn inv_norm_cdf<T: Scalar>(p: T) -> T {
    let zero = T::zero();
    let one = T::one();
    let half = T::from(0.5).unwrap();

    assert!(p > zero && p < one, "inv_norm_cdf: p must be in (0, 1)");

    // Acklam's rational approximation coefficients.
    let a1 = T::from(-3.969683028665376e+01).unwrap();
    let a2 = T::from( 2.209460984245205e+02).unwrap();
    let a3 = T::from(-2.759285104469687e+02).unwrap();
    let a4 = T::from( 1.383577518672690e+02).unwrap();
    let a5 = T::from(-3.066479806614716e+01).unwrap();
    let a6 = T::from( 2.506628277459239e+00).unwrap();

    let b1 = T::from(-5.447609879822406e+01).unwrap();
    let b2 = T::from( 1.615858368580409e+02).unwrap();
    let b3 = T::from(-1.556989798598866e+02).unwrap();
    let b4 = T::from( 6.680131188771972e+01).unwrap();
    let b5 = T::from(-1.328068155288572e+01).unwrap();

    let c1 = T::from(-7.784894002430293e-03).unwrap();
    let c2 = T::from(-3.223964580411365e-01).unwrap();
    let c3 = T::from(-2.400758277161838e+00).unwrap();
    let c4 = T::from(-2.549732539343734e+00).unwrap();
    let c5 = T::from( 4.374664141464968e+00).unwrap();
    let c6 = T::from( 2.938163982698783e+00).unwrap();

    let d1 = T::from( 7.784695709041462e-03).unwrap();
    let d2 = T::from( 3.224671290700398e-01).unwrap();
    let d3 = T::from( 2.445134137142996e+00).unwrap();
    let d4 = T::from( 3.754408661907416e+00).unwrap();

    let p_low  = T::from(0.02425).unwrap();
    let p_high = one - p_low;

    if p < p_low {
        // Left tail.
        let q = (-T::from(2.0).unwrap() * p.ln()).sqrt();
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + one)
    } else if p <= p_high {
        // Central region.
        let q = p - half;
        let r = q * q;
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + one)
    } else {
        // Right tail.
        let q = (-T::from(2.0).unwrap() * (one - p).ln()).sqrt();
        -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + one)
    }
}
