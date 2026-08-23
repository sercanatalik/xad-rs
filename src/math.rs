//! AD-aware transcendental functions.
//!
//! Provides `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`,
//! `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `exp2`, `ln`, `log2`, `log10`,
//! `ln_1p`, `exp_m1`, `sqrt`, `cbrt`, `abs`, `atan2`, `pow`, `powf`, `powi`,
//! `hypot`, `max`, `min`, `erf`, `erfc`, `norm_cdf`, and `inv_norm_cdf` —
//! each in two AD-aware flavours with the correct chain-rule derivative
//! propagation already plumbed through:
//!
//! - [`ad`] — reverse-mode variants that operate on [`AReal`] and record
//!   onto the currently active tape. This module additionally provides the
//!   **fused n-ary recorders** [`ad::sum`], [`ad::dot`],
//!   [`ad::weighted_sum`], and [`ad::weighted_dot`], which record a whole
//!   accumulation as one tape statement.
//! - [`fwd`] — forward-mode variants that operate on [`Jet1`].
//!
//! Transcendental methods directly on [`Jet2`](crate::forward::jet2::Jet2)
//! and [`Jet2Vec`](crate::forward::jet2vec::Jet2Vec) live on those types
//! (inherent methods), not in this module.
//!
//! The Gaussian family also has passive forms at the module root —
//! [`erf`], [`erfc`], [`norm_pdf`], [`norm_cdf`], [`inv_norm_cdf`] — for
//! callers that need them without any AD layer. These are exactly what
//! `impl Real for f64` delegates to, so a generic body instantiated at
//! `f64` agrees with the same formula written non-generically.
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

use crate::reverse::areal::{record_binary_op, record_nary_op_bounded, record_unary_op, AReal};
use crate::forward::jet1::Jet1;
use crate::passive::Passive;
use crate::tape::TapeStorage;

// ============================================================================
// Macros for generating unary math functions
// ============================================================================

// Both stamps consume entries from the crate-wide derivative table in
// `src/elementaries.rs` — `(name, doc, |x| val, |x, r| d1, |x, r, d1| d2)`.
// First-order modes never expand the `d2` closure.

macro_rules! stamp_ad_unary {
    ($name:ident, $doc:literal, $val:expr, $d1:expr, $d2:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $name<T: TapeStorage>(x: &AReal<T>) -> AReal<T> {
            let v = x.value();
            let result = ($val)(v);
            let deriv = ($d1)(v, result);
            record_unary_op(result, x.slot(), deriv)
        }
    };
}

macro_rules! stamp_fwd_unary {
    ($name:ident, $doc:literal, $val:expr, $d1:expr, $d2:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $name<T: Passive>(x: &Jet1<T>) -> Jet1<T> {
            let v = x.value();
            let result = ($val)(v);
            let deriv = ($d1)(v, result);
            Jet1::new(result, deriv * x.derivative())
        }
    };
}

/// `∂(b^e)/∂b = e·b^{e-1}`, computed from the already-evaluated `result
/// = b^e` as `e·result/b` — one `powf` per `pow`/`powf` call instead of
/// two (NOTE(perf): ~36% off the scalar core of `pow` on Apple M-series;
/// differs from the `powf(e - 1)` form by ≤ 1 ulp). `b == 0` must take
/// the direct form: `result/b` would be `0/0 = NaN` where the true
/// partial is `0` (e > 1), `1` (e == 1), or `±∞` (e < 1) — all of which
/// `e·b^{e-1}` produces correctly.
#[inline]
fn pow_d_base<T: Passive>(b: T, e: T, result: T) -> T {
    if b == T::zero() {
        e * b.powf(e - T::one())
    } else {
        e * result / b
    }
}

/// AD-aware math functions for `AReal` (reverse mode).
pub mod ad {
    use super::*;

    crate::elementaries::for_each_unary_elementary!(stamp_ad_unary);

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
        let d_base = pow_d_base(bv, ev, result);
        let d_exp = result * bv.ln();
        record_binary_op(result, base.slot(), d_base, exponent.slot(), d_exp)
    }

    #[inline]
    pub fn powf<T: TapeStorage>(base: &AReal<T>, exponent: T) -> AReal<T> {
        let bv = base.value();
        let result = bv.powf(exponent);
        let deriv = pow_d_base(bv, exponent, result);
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

    // ------------------------------------------------------------------
    // Fused n-ary recorders
    //
    // Accumulation loops dominate real pricers (swap legs, Monte Carlo
    // payoff averages). Written with binary operators, a length-n sum
    // records n-1 statements / 2(n-1) operands and a dot product records
    // 2n-1 statements / 4n-2 operands. The helpers below record ONE
    // statement each (n and 2n operands respectively), shrinking the tape
    // the reverse sweep is memory-bound on.
    // ------------------------------------------------------------------

    /// Fused sum `Σᵢ xs[i]`, recorded as a **single** tape statement with
    /// one operand per active input (∂/∂xᵢ = 1) instead of a chain of
    /// `n - 1` binary adds.
    ///
    /// An empty slice returns an unrecorded zero constant.
    pub fn sum<T: TapeStorage>(xs: &[AReal<T>]) -> AReal<T> {
        let mut value = T::zero();
        for x in xs {
            value += x.value();
        }
        if xs.is_empty() {
            return AReal::new(value);
        }
        record_nary_op_bounded(value, xs.len(), xs.iter().map(|x| (T::one(), x.slot())))
    }

    /// Fused dot product `Σᵢ xs[i]·ys[i]`, recorded as a **single** tape
    /// statement with two operands per pair (∂/∂xᵢ = yᵢ, ∂/∂yᵢ = xᵢ)
    /// instead of `2n - 1` binary statements.
    ///
    /// Empty slices return an unrecorded zero constant.
    ///
    /// # Panics
    /// Panics if `xs.len() != ys.len()`.
    pub fn dot<T: TapeStorage>(xs: &[AReal<T>], ys: &[AReal<T>]) -> AReal<T> {
        assert_eq!(xs.len(), ys.len(), "dot: slice length mismatch");
        let mut value = T::zero();
        for (x, y) in xs.iter().zip(ys) {
            value += x.value() * y.value();
        }
        if xs.is_empty() {
            return AReal::new(value);
        }
        record_nary_op_bounded(
            value,
            2 * xs.len(),
            xs.iter()
                .zip(ys)
                .flat_map(|(x, y)| [(y.value(), x.slot()), (x.value(), y.slot())]),
        )
    }

    /// Fused weighted sum `Σᵢ ws[i]·xs[i]` with **passive** weights,
    /// recorded as a **single** statement with one operand per active input
    /// (∂/∂xᵢ = wᵢ). The discounted-cashflow shape: for passive weights this
    /// halves the operand count of [`dot`] (which must also record
    /// ∂/∂wᵢ = xᵢ) and replaces a `2n - 1`-statement binary chain.
    ///
    /// Empty slices return an unrecorded zero constant.
    ///
    /// # Panics
    /// Panics if `ws.len() != xs.len()`.
    pub fn weighted_sum<T: TapeStorage>(ws: &[T], xs: &[AReal<T>]) -> AReal<T> {
        assert_eq!(ws.len(), xs.len(), "weighted_sum: slice length mismatch");
        if xs.is_empty() {
            return AReal::new(T::zero());
        }
        let mut value = T::zero();
        for (&w, x) in ws.iter().zip(xs) {
            value += w * x.value();
        }
        record_nary_op_bounded(value, xs.len(), ws.iter().zip(xs).map(|(&w, x)| (w, x.slot())))
    }

    /// Fused weighted dot product `Σᵢ ws[i]·xs[i]·ys[i]` with **passive**
    /// weights, recorded as a **single** statement with two operands per
    /// term (∂/∂xᵢ = wᵢ·yᵢ, ∂/∂yᵢ = wᵢ·xᵢ). The premium-leg shape
    /// (accrual · discount · survival with passive year fractions):
    /// composed from binary ops it costs `3n - 1` statements, and even
    /// [`dot`] can't absorb the weights without first recording `n`
    /// scaling statements.
    ///
    /// Empty slices return an unrecorded zero constant.
    ///
    /// # Panics
    /// Panics if `ws`, `xs`, and `ys` don't all have the same length.
    pub fn weighted_dot<T: TapeStorage>(ws: &[T], xs: &[AReal<T>], ys: &[AReal<T>]) -> AReal<T> {
        assert_eq!(ws.len(), xs.len(), "weighted_dot: slice length mismatch");
        assert_eq!(xs.len(), ys.len(), "weighted_dot: slice length mismatch");
        if xs.is_empty() {
            return AReal::new(T::zero());
        }
        let mut value = T::zero();
        for ((&w, x), y) in ws.iter().zip(xs).zip(ys) {
            value += w * x.value() * y.value();
        }
        record_nary_op_bounded(
            value,
            2 * xs.len(),
            ws.iter().zip(xs).zip(ys).flat_map(|((&w, x), y)| {
                [(w * y.value(), x.slot()), (w * x.value(), y.slot())]
            }),
        )
    }
}

/// AD-aware math functions for `Jet1` (forward mode).
pub mod fwd {
    use super::*;

    crate::elementaries::for_each_unary_elementary!(stamp_fwd_unary);

    #[inline]
    pub fn atan2<T: Passive>(y: &Jet1<T>, x: &Jet1<T>) -> Jet1<T> {
        let yv = y.value();
        let xv = x.value();
        let result = yv.atan2(xv);
        let denom = xv * xv + yv * yv;
        let deriv = (xv * y.derivative() - yv * x.derivative()) / denom;
        Jet1::new(result, deriv)
    }

    #[inline]
    pub fn pow<T: Passive>(base: &Jet1<T>, exponent: &Jet1<T>) -> Jet1<T> {
        let bv = base.value();
        let ev = exponent.value();
        let result = bv.powf(ev);
        let d_base = pow_d_base(bv, ev, result);
        let d_exp = result * bv.ln();
        Jet1::new(result, d_base * base.derivative() + d_exp * exponent.derivative())
    }

    #[inline]
    pub fn powf<T: Passive>(base: &Jet1<T>, exponent: T) -> Jet1<T> {
        let bv = base.value();
        let result = bv.powf(exponent);
        let deriv = pow_d_base(bv, exponent, result);
        Jet1::new(result, deriv * base.derivative())
    }

    #[inline]
    pub fn powi<T: Passive>(base: &Jet1<T>, exponent: i32) -> Jet1<T> {
        let bv = base.value();
        let result = bv.powi(exponent);
        let deriv = T::from(exponent).unwrap() * bv.powi(exponent - 1);
        Jet1::new(result, deriv * base.derivative())
    }

    #[inline]
    pub fn hypot<T: Passive>(x: &Jet1<T>, y: &Jet1<T>) -> Jet1<T> {
        let xv = x.value();
        let yv = y.value();
        let result = xv.hypot(yv);
        let inv_r = T::one() / result;
        let deriv = xv * inv_r * x.derivative() + yv * inv_r * y.derivative();
        Jet1::new(result, deriv)
    }

    pub fn max<T: Passive>(a: &Jet1<T>, b: &Jet1<T>) -> Jet1<T> {
        if a.value() >= b.value() { *a } else { *b }
    }

    pub fn min<T: Passive>(a: &Jet1<T>, b: &Jet1<T>) -> Jet1<T> {
        if a.value() <= b.value() { *a } else { *b }
    }

}

/// Error function `erf(x)` on a passive scalar.
///
/// Routed through [`Passive::erf_value`], so plain `f64` gets the
/// full-precision split evaluation in [`erf_impl`] (measured worst relative
/// error `1.3e-15` against a correctly-rounded reference over a dense
/// `[-6.5, 6.5]` sweep) while `Jet1<T>` gets that value paired with the
/// **exact** analytic tangent. The AD-aware variants live in [`ad::erf`]
/// and [`fwd::erf`].
#[inline]
pub fn erf<T: Passive>(x: T) -> T {
    x.erf_value()
}

/// Complementary error function `erfc(x) = 1 - erf(x)` on a passive scalar.
///
/// Evaluated as `1 - erf_value(x)` — the same expression the crate-wide
/// derivative table uses for its `erfc` entry, so the passive
/// [`Real::erfc`](crate::Real::erfc) method and this function are
/// bit-identical. For `x > 3` the subtraction loses relative precision in
/// the tail (`erfc` there is the small quantity); `erf_impl` computes the
/// tail from the Gauss continued fraction internally, but the `1 - erf`
/// identity is what every AD surface differentiates, so it is what this
/// function returns.
#[inline]
pub fn erfc<T: Passive>(x: T) -> T {
    T::one() - x.erf_value()
}

/// Full-precision `erf` — the default body of [`Passive::erf_value`].
/// Never call this directly on a `Jet1`; use [`erf`] (or the trait method)
/// so the exact-tangent override applies.
///
/// Two regimes, both cancellation-free by construction:
///
/// - `|x| ≤ 3`: the confluent-hypergeometric series
///   `erf(x) = (2/√π)·e^{−x²}·Σ_{n≥0} x·(2x²)ⁿ/(2n+1)!!` — every term is
///   positive, so the sum carries no alternating-series cancellation; ≤ 43
///   terms to machine epsilon at the switch point.
/// - `|x| > 3`: `erf = 1 − erfc` with `erfc` from the Gauss continued
///   fraction `erfc(x) = (e^{−x²}/√π) / (x + (1/2)/(x + (2/2)/(x + …)))`,
///   evaluated backward at fixed depth 24 (depth 20 already reaches machine
///   precision at the switch point, measured). `erfc(3) ≈ 2.2e-5`, so the
///   `1 − erfc` subtraction costs no relative precision in `erf`.
/// - `|x| ≥ 6`: `erfc < 2⁻⁵⁴`, so `erf` saturates at `±1` exactly in `f64`.
///
/// Replaces the Abramowitz & Stegun 7.1.26 polynomial (~1.5e-7 absolute),
/// whose error was measurable through `norm_cdf` in downstream option
/// pricers — and whose *derivative* disagreed with the exact-tangent
/// override by ~1e-5 locally, making finite differences of the value the
/// approximate side of any AD-vs-FD comparison.
pub(crate) fn erf_impl<T: Passive>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    let sign = if x < T::zero() { -T::one() } else { T::one() };
    let ax = x.abs();
    if ax >= T::from(6.0).unwrap() {
        return sign;
    }
    let x2 = ax * ax;
    if ax <= T::from(3.0).unwrap() {
        let two_x2 = x2 + x2;
        let mut term = ax;
        let mut sum = ax;
        let mut n = 1u32;
        // ≤ 43 iterations at |x| = 3; the cap is an overflow backstop, not a
        // convergence budget.
        while n <= 200 {
            term = term * two_x2 / T::from(2 * n + 1).unwrap();
            sum += term;
            if term <= sum * T::epsilon() {
                break;
            }
            n += 1;
        }
        let two_over_sqrt_pi = T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap();
        sign * two_over_sqrt_pi * (-x2).exp() * sum
    } else {
        let mut f = T::zero();
        let mut k = 24u32;
        while k >= 1 {
            f = T::from(k).unwrap() * T::from(0.5).unwrap() / (ax + f);
            k -= 1;
        }
        let sqrt_pi = T::from(2.0 / std::f64::consts::FRAC_2_SQRT_PI).unwrap();
        let erfc = (-x2).exp() / (sqrt_pi * (ax + f));
        sign * (T::one() - erfc)
    }
}

/// Standard normal PDF: `φ(x) = (1/√(2π)) · exp(-x²/2)`.
///
/// Used internally by `norm_cdf` and `inv_norm_cdf` AD variants for the
/// derivative. Exposed publicly for callers that need the density on a
/// plain scalar.
#[inline]
pub fn norm_pdf<T: Passive>(x: T) -> T {
    let inv_sqrt_2pi = T::from(1.0 / (2.0 * std::f64::consts::PI).sqrt()).unwrap();
    inv_sqrt_2pi * (T::from(-0.5).unwrap() * x * x).exp()
}

/// Standard normal CDF: `Φ(x) = 0.5 · (1 + erf(x / √2))`.
///
/// Uses the same full-precision `erf` as [`erf`], so the value is accurate
/// to a few ulp in absolute terms (the far negative tail's *relative*
/// accuracy is still bounded by the `1 + erf` subtraction, as for any
/// erf-based CDF). AD-aware variants live in [`ad::norm_cdf`] and
/// [`fwd::norm_cdf`].
#[inline]
pub fn norm_cdf<T: Passive>(x: T) -> T {
    let half = T::from(0.5).unwrap();
    let frac_1_sqrt_2 = T::from(std::f64::consts::FRAC_1_SQRT_2).unwrap();
    half * (T::one() + erf(x * frac_1_sqrt_2))
}

/// Inverse standard normal CDF: `Φ⁻¹(p)` on a passive scalar.
///
/// Routed through [`Passive::inv_norm_cdf_value`]: `f32`/`f64` use
/// Acklam's rational approximation (~1.15e-9); `Jet1<T>` pairs that value
/// with the exact tangent `1/φ(Φ⁻¹(p))`.
///
/// # Panics
///
/// Panics if `p` is outside `(0, 1)` (exclusive).
#[inline]
pub fn inv_norm_cdf<T: Passive>(p: T) -> T {
    p.inv_norm_cdf_value()
}

/// Acklam's rational approximation — the default body of
/// [`Passive::inv_norm_cdf_value`].
#[inline]
pub(crate) fn inv_norm_cdf_poly<T: Passive>(p: T) -> T {
    let zero = T::zero();
    let one = T::one();
    let half = T::from(0.5).unwrap();

    assert!(p > zero && p < one, "inv_norm_cdf: p must be in (0, 1)");

    // Acklam's rational approximation coefficients.
    let a1 = T::from(-3.969683028665376e+01).unwrap();
    let a2 = T::from( 2.209460984245205e+02).unwrap();
    let a3 = T::from(-2.759285104469687e+02).unwrap();
    let a4 = T::from( 1.38357751867269e+02).unwrap();
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
