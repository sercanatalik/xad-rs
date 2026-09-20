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
//! - [`fwdk`] — the same forward-mode variants on the K-lane
//!   [`JetK`]: every chain rule scales all
//!   `K` tangent lanes at once.
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

use crate::reverse::areal::{
    record_binary_op, record_nary_op_bounded, record_nary_unit_op_bounded, record_unary_op,
    record_unary_unit_op, AReal,
};
use crate::forward::jet1::Jet1;
use crate::forward::jetk::JetK;
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

// The K-lane stamp is the `Jet1` stamp with the single tangent replaced by
// the lane array: `JetK::chain` scales every lane by the same `deriv` the
// `Jet1` stamp multiplies its one tangent by, so lane `i` of the result is
// bit-identical to the `Jet1` tangent seeded along direction `i`.
macro_rules! stamp_fwdk_unary {
    ($name:ident, $doc:literal, $val:expr, $d1:expr, $d2:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $name<T: Passive, const K: usize>(x: &JetK<T, K>) -> JetK<T, K> {
            let v = x.value;
            let result = ($val)(v);
            let deriv = ($d1)(v, result);
            x.chain(result, deriv)
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
    /// Records a **unary** unit op on the live branch only — recording a
    /// binary op with a zero multiplier on the inactive branch would waste
    /// one tape slot and one multiply on every reverse sweep.
    #[inline]
    pub fn max<T: TapeStorage>(a: &AReal<T>, b: &AReal<T>) -> AReal<T> {
        if a.value() >= b.value() {
            record_unary_unit_op(a.value(), a.slot(), false)
        } else {
            record_unary_unit_op(b.value(), b.slot(), false)
        }
    }

    /// `min(a, b)` with correct adjoint propagation. See [`max`] for the
    /// rationale behind the unary (rather than binary-with-zero) encoding.
    #[inline]
    pub fn min<T: TapeStorage>(a: &AReal<T>, b: &AReal<T>) -> AReal<T> {
        if a.value() <= b.value() {
            record_unary_unit_op(a.value(), a.slot(), false)
        } else {
            record_unary_unit_op(b.value(), b.slot(), false)
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
        // Every operand is `+1`: recorded as unit operands (see `crate::tape`).
        record_nary_unit_op_bounded(value, xs.len(), xs.iter().map(|x| (x.slot(), false)))
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

/// AD-aware math functions for `JetK` (K-lane forward mode).
///
/// Stamped from the same derivative table as [`fwd`], so every unary here
/// computes the value and the per-lane tangent factor with exactly the
/// closures `fwd` uses; the binaries below mirror `fwd`'s hand-written set
/// through [`JetK::chain2`](crate::forward::jetk::JetK), sharing
/// `pow_d_base`.
pub mod fwdk {
    use super::*;

    crate::elementaries::for_each_unary_elementary!(stamp_fwdk_unary);

    #[inline]
    pub fn atan2<T: Passive, const K: usize>(y: &JetK<T, K>, x: &JetK<T, K>) -> JetK<T, K> {
        let yv = y.value;
        let xv = x.value;
        let result = yv.atan2(xv);
        let denom = xv * xv + yv * yv;
        y.chain2(*x, result, xv / denom, -yv / denom)
    }

    #[inline]
    pub fn pow<T: Passive, const K: usize>(base: &JetK<T, K>, exponent: &JetK<T, K>) -> JetK<T, K> {
        let bv = base.value;
        let ev = exponent.value;
        let result = bv.powf(ev);
        let d_base = pow_d_base(bv, ev, result);
        let d_exp = result * bv.ln();
        base.chain2(*exponent, result, d_base, d_exp)
    }

    #[inline]
    pub fn powf<T: Passive, const K: usize>(base: &JetK<T, K>, exponent: T) -> JetK<T, K> {
        let bv = base.value;
        let result = bv.powf(exponent);
        base.chain(result, pow_d_base(bv, exponent, result))
    }

    #[inline]
    pub fn powi<T: Passive, const K: usize>(base: &JetK<T, K>, exponent: i32) -> JetK<T, K> {
        let bv = base.value;
        let result = bv.powi(exponent);
        let deriv = T::from(exponent).unwrap() * bv.powi(exponent - 1);
        base.chain(result, deriv)
    }

    #[inline]
    pub fn hypot<T: Passive, const K: usize>(x: &JetK<T, K>, y: &JetK<T, K>) -> JetK<T, K> {
        let xv = x.value;
        let yv = y.value;
        let result = xv.hypot(yv);
        let inv_r = T::one() / result;
        x.chain2(*y, result, xv * inv_r, yv * inv_r)
    }

    pub fn max<T: Passive, const K: usize>(a: &JetK<T, K>, b: &JetK<T, K>) -> JetK<T, K> {
        if a.value >= b.value { *a } else { *b }
    }

    pub fn min<T: Passive, const K: usize>(a: &JetK<T, K>, b: &JetK<T, K>) -> JetK<T, K> {
        if a.value <= b.value { *a } else { *b }
    }
}

/// Error function `erf(x)` on a passive scalar.
///
/// Routed through [`Passive::erf_value`], so plain `f64` gets the
/// full-precision piecewise rational in [`erf_impl`] (about 1 ulp against a
/// correctly rounded reference; `tests/erf_precision.rs` pins 5 ulp) while
/// `Jet1<T>` gets that value paired with the **exact** analytic tangent. The AD-aware variants live in [`ad::erf`]
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
/// The piecewise minimax rational of Sun's `s_erf.c` (fdlibm), transcribed
/// generically over `T: Passive`. Four regimes on `|x|`, each a fixed-cost
/// rational in the regime's natural variable, plus saturation:
///
/// - `|x| < 0.84375`: `x + x·P(x²)/Q(x²)`, with a tiny-argument short cut
///   below `2⁻²⁸` so `erf(x) ≈ (2/√π)·x` rounds correctly.
/// - `0.84375 ≤ |x| < 1.25`: `erf(1) + P(s)/Q(s)`, `s = |x| − 1`.
/// - `1.25 ≤ |x| < 6`: `1 − e^{−x² − 9/16 + R(1/x²)/S(1/x²)} / |x|`, with two
///   coefficient sets split at `|x| = 1/0.35`. The `x²` is formed from a copy
///   of `x` with its low 32 mantissa bits cleared, so `e^{−z² − 9/16}` is
///   exact enough for the correction factor `e^{(z−x)(z+x) + R/S}` to carry
///   no cancellation.
/// - `|x| ≥ 6`: `erfc < 2⁻⁵⁴`, so `erf` saturates at `±1` exactly in `f64`.
///
/// Worst error is about 1 ulp against a correctly rounded reference
/// (`tests/erf_precision.rs` pins 5 ulp over a grid concentrated at the regime
/// boundaries). This replaced the confluent-hypergeometric series /
/// Gauss continued fraction of 6.x, whose per-term dependent divisions cost
/// 68–78 ns per call at the abscissae option pricers hit; the rational costs
/// a single `exp` plus one rational at ~7–11 ns, at the same accuracy. It
/// is a value-only change: every mode reaches `erf` through
/// [`Passive::erf_value`] on the passive scalar, and the derivative table
/// keeps the analytic `(2/√π)·e^{−x²}` (see the passive-reference rule in
/// [`crate::real`]).
pub(crate) fn erf_impl<T: Passive>(x: T) -> T {
    #[inline(always)]
    fn c<T: Passive>(v: f64) -> T {
        T::from(v).unwrap()
    }
    if x.is_nan() {
        return x;
    }
    let sign = if x < T::zero() { -T::one() } else { T::one() };
    let ax = x.abs();

    if ax < c(0.84375) {
        if ax < c(3.725_290_298_461_914e-9) {
            // |x| < 2^-28: erf(x) = x·(1 + 2/√π) to full precision, spelled
            // as in fdlibm to avoid spurious underflow.
            return c::<T>(0.125) * (c::<T>(8.0) * x + c::<T>(1.027_033_336_764_100_690_53) * x);
        }
        let z = x * x;
        let r = c::<T>(1.283_791_670_955_125_585_61e-1)
            + z * (c::<T>(-3.250_421_072_470_014_993_70e-1)
                + z * (c::<T>(-2.848_174_957_559_851_047_66e-2)
                    + z * (c::<T>(-5.770_270_296_489_441_591_57e-3)
                        + z * c::<T>(-2.376_301_665_665_016_260_84e-5))));
        let s = T::one()
            + z * (c::<T>(3.979_172_239_591_553_528_19e-1)
                + z * (c::<T>(6.502_224_998_876_729_444_85e-2)
                    + z * (c::<T>(5.081_306_281_875_765_627_76e-3)
                        + z * (c::<T>(1.324_947_380_043_216_445_26e-4)
                            + z * c::<T>(-3.960_228_278_775_368_123_20e-6)))));
        return x + x * (r / s);
    }

    if ax < c(1.25) {
        let s = ax - T::one();
        let p = c::<T>(-2.362_118_560_752_659_440_77e-3)
            + s * (c::<T>(4.148_561_186_837_483_316_66e-1)
                + s * (c::<T>(-3.722_078_760_357_013_238_47e-1)
                    + s * (c::<T>(3.183_466_199_011_617_536_74e-1)
                        + s * (c::<T>(-1.108_946_942_823_966_774_76e-1)
                            + s * (c::<T>(3.547_830_432_561_823_593_71e-2)
                                + s * c::<T>(-2.166_375_594_868_790_843_00e-3))))));
        let q = T::one()
            + s * (c::<T>(1.064_208_804_008_442_282_86e-1)
                + s * (c::<T>(5.403_979_177_021_710_489_37e-1)
                    + s * (c::<T>(7.182_865_441_419_626_628_68e-2)
                        + s * (c::<T>(1.261_712_198_087_616_421_12e-1)
                            + s * (c::<T>(1.363_708_391_202_905_073_62e-2)
                                + s * c::<T>(1.198_449_984_679_910_741_70e-2))))));
        let erx: T = c(8.450_629_115_104_675_292_97e-1);
        return sign * (erx + p / q);
    }

    if ax >= c(6.0) {
        return sign;
    }

    let s = T::one() / (ax * ax);
    let (r, q) = if ax < c(1.0 / 0.35) {
        let r = c::<T>(-9.864_944_034_847_148_227_05e-3)
            + s * (c::<T>(-6.938_585_727_071_817_643_72e-1)
                + s * (c::<T>(-1.055_862_622_532_329_098_14e1)
                    + s * (c::<T>(-6.237_533_245_032_600_603_96e1)
                        + s * (c::<T>(-1.623_966_694_625_734_703_55e2)
                            + s * (c::<T>(-1.846_050_929_067_110_359_94e2)
                                + s * (c::<T>(-8.128_743_550_630_659_342_46e1)
                                    + s * c::<T>(-9.814_329_344_169_145_485_92)))))));
        let q = T::one()
            + s * (c::<T>(1.965_127_166_743_925_712_92e1)
                + s * (c::<T>(1.376_577_541_435_190_426_00e2)
                    + s * (c::<T>(4.345_658_774_752_292_288_21e2)
                        + s * (c::<T>(6.453_872_717_332_678_803_36e2)
                            + s * (c::<T>(4.290_081_400_275_678_333_86e2)
                                + s * (c::<T>(1.086_350_055_417_794_351_34e2)
                                    + s * (c::<T>(6.570_249_770_319_281_701_35)
                                        + s * c::<T>(-6.042_441_521_485_809_874_38e-2))))))));
        (r, q)
    } else {
        let r = c::<T>(-9.864_942_924_700_099_285_97e-3)
            + s * (c::<T>(-7.992_832_376_805_230_065_74e-1)
                + s * (c::<T>(-1.775_795_491_775_475_198_89e1)
                    + s * (c::<T>(-1.606_363_848_558_219_160_62e2)
                        + s * (c::<T>(-6.375_664_433_683_896_277_22e2)
                            + s * (c::<T>(-1.025_095_131_611_077_249_54e3)
                                + s * c::<T>(-4.835_191_916_086_513_970_19e2))))));
        let q = T::one()
            + s * (c::<T>(3.033_806_074_348_245_829_24e1)
                + s * (c::<T>(3.257_925_129_965_739_188_26e2)
                    + s * (c::<T>(1.536_729_586_084_436_959_94e3)
                        + s * (c::<T>(3.199_858_219_508_595_539_08e3)
                            + s * (c::<T>(2.553_050_406_433_164_425_83e3)
                                + s * (c::<T>(4.745_285_412_069_553_672_15e2)
                                    + s * c::<T>(-2.244_095_244_658_581_833_62e1)))))));
        (r, q)
    };
    // `z`: `ax` with its low 32 mantissa bits cleared, so `z·z` is exact in
    // `f64` and the split `e^{−z²−9/16}·e^{(z−ax)(z+ax)+R/S}` is
    // cancellation-free. Exact for `f64`; for `f32` the cleared value is
    // still representable.
    let z: T = c(f64::from_bits(ax.to_f64().unwrap().to_bits() & 0xffff_ffff_0000_0000));
    let tail = (-z * z - c::<T>(0.5625)).exp() * ((z - ax) * (z + ax) + r / q).exp();
    sign * (T::one() - tail / ax)
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
