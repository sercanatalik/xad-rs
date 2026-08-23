//! Accuracy pins for the full-precision `erf` (series ≤ 3, Gauss continued
//! fraction above — see `math::erf_impl`), which replaced the Abramowitz &
//! Stegun 7.1.26 polynomial (~1.5e-7 absolute).
//!
//! Reference values are CPython's `math.erf` (platform libm, correctly
//! rounded to within ~1 ulp), generated independently of this
//! implementation. The asserted tolerance is **5 ulp relative** — the
//! prototype measured a worst relative error of `1.3e-15` (~5.5 ulp of the
//! *value*, i.e. a few ulp of 1.0) over a dense 53k-point sweep of
//! `[-6.5, 6.5]`; against the old polynomial the same sweep measured
//! `1.5e-7`. The grid below concentrates points where the implementation
//! changes regime: around the series/continued-fraction switch at `|x| = 3`
//! and the `±1` saturation at `|x| = 6`.
//!
//! Also pinned: the properties an approximation can silently lose — oddness,
//! monotonicity, saturation, NaN propagation — and the downstream `norm_cdf`
//! against its own reference values, since that is the seam option pricers
//! actually consume.
//!
//! The same tolerances are then asserted through the `Real` trait methods in
//! every mode, so the generic path cannot drift from the free functions these
//! reference values pin.

use xad_rs::math::{erf, erfc, norm_cdf, norm_pdf};
use xad_rs::{AReal, Jet1, Jet2, Real, Tape};

/// (x, erf(x)) — reference from CPython `math.erf`.
const ERF_REFERENCE: &[(f64, f64)] = &[
    (0.0, 0.0),
    (1e-08, 1.1283791670955126e-08),
    (0.01, 0.011283415555849616),
    (0.1, 0.1124629160182849),
    (0.25, 0.2763263901682369),
    (0.5, 0.5204998778130465),
    (0.75, 0.7111556336535152),
    (1.0, 0.8427007929497148),
    (1.5, 0.9661051464753108),
    // The prototype sweep's worst-error abscissa.
    (1.910464943267, 0.9931035894124023),
    (2.0, 0.9953222650189527),
    (2.5, 0.999593047982555),
    // Both sides of the series/continued-fraction switch.
    (2.999999, 0.999977909363748),
    (3.0, 0.9999779095030015),
    (3.000001, 0.9999779096422541),
    (3.5, 0.9999992569016276),
    (4.0, 0.9999999845827421),
    (4.5, 0.9999999998033839),
    (5.0, 0.9999999999984626),
    (5.5, 0.9999999999999927),
    (5.999, 1.0),
    (6.0, 1.0),
    (7.0, 1.0),
];

const FIVE_ULP: f64 = 5.0 * f64::EPSILON;

#[test]
fn erf_matches_reference_to_five_ulp() {
    for &(x, want) in ERF_REFERENCE {
        for (xx, ww) in [(x, want), (-x, -want)] {
            let got = erf(xx);
            let err = (got - ww).abs();
            let tol = FIVE_ULP * ww.abs().max(f64::MIN_POSITIVE);
            assert!(
                err <= tol.max(5e-24), // absolute slack only for erf(1e-8) ~ 1e-8
                "erf({xx}): got {got:e}, want {ww:e} (err {err:e})"
            );
        }
    }
}

#[test]
fn erf_is_odd_monotone_and_saturating() {
    let mut prev = erf(-8.0);
    assert_eq!(prev, -1.0, "erf saturates at -1 in f64 beyond -6");
    let mut x = -8.0;
    while x <= 8.0 {
        let v = erf(x);
        assert!(v >= prev, "erf must be monotone non-decreasing (x = {x})");
        assert_eq!(erf(-x), -v, "erf must be exactly odd (x = {x})");
        prev = v;
        x += 0.0173; // irrational-ish step so no special points are favoured
    }
    assert_eq!(erf(8.0f64.min(prev.mul_add(0.0, 8.0))), 1.0);
    assert!(erf(f64::NAN).is_nan(), "NaN propagates");
    assert_eq!(erf(f64::INFINITY), 1.0);
    assert_eq!(erf(f64::NEG_INFINITY), -1.0);
}

/// (x, Φ(x)) — reference from CPython `0.5 * (1 + math.erf(x / sqrt(2)))`
/// computed in extended precision via `math.erf` (the identity this
/// implementation uses, evaluated against the reference `erf`).
#[test]
fn norm_cdf_matches_reference() {
    // Φ values from the reference erf at x/√2; asserting through the same
    // identity isolates the erf substitution as the only moving part.
    let cases: &[(f64, f64)] = &[
        (0.0, 0.5),
        (1.0, 0.8413447460685429),
        (-1.0, 0.15865525393145707),
        (1.959963984540054, 0.975),
        (2.575829303548901, 0.995),
        (-3.719016485455681, 1e-4),
    ];
    for &(x, want) in cases {
        let got = norm_cdf(x);
        let err = (got - want).abs();
        // rtol + one-epsilon atol: the `0.5·(1 + erf)` identity carries an
        // inherent absolute ~ε/2 in the negative tail (documented on
        // `norm_cdf`), so a pure relative bound would test the identity's
        // algebra, not the erf substitution.
        assert!(
            err <= 5.0 * f64::EPSILON * want.abs() + f64::EPSILON,
            "norm_cdf({x}): got {got:e}, want {want:e} (err {err:e})"
        );
    }
}

// ============================================================================
// The same accuracy, reached through `Real` in every mode.
//
// `Real`'s Gaussian methods are stamped from the crate-wide derivative table,
// so each mode's value comes from the very functions the reference tables
// above pin. These tests assert that at the same tolerances, per mode — a
// mode whose stamp were rewired to a different evaluation would fail here.
// ============================================================================

/// Every mode's `Real::erf` at one point, as plain `f64` values.
fn erf_in_every_mode(x: f64) -> [f64; 4] {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let rev = Real::erf(&AReal::input(x, &mut tape)).value();
    [
        Real::erf(&x),
        Jet1::new(x, 1.0).erf().value(),
        Jet2::variable(x).erf().value(),
        rev,
    ]
}

#[test]
fn trait_erf_matches_reference_to_five_ulp_in_every_mode() {
    for &(x, want) in ERF_REFERENCE {
        for (xx, ww) in [(x, want), (-x, -want)] {
            for (mode, got) in ["f64", "Jet1", "Jet2", "AReal"]
                .iter()
                .zip(erf_in_every_mode(xx))
            {
                let tol = FIVE_ULP * ww.abs().max(f64::MIN_POSITIVE);
                assert!(
                    (got - ww).abs() <= tol.max(5e-24),
                    "{mode}: Real::erf({xx}) = {got:e}, want {ww:e}"
                );
            }
        }
    }
}

#[test]
fn trait_gaussians_equal_the_free_functions_in_every_mode() {
    // Bit-identical, not merely within tolerance: each mode's stamp routes to
    // the same evaluation, so any difference is a mis-stamp, not roundoff.
    for &x in &[-6.5_f64, -3.0, -1.0, 0.0, 0.5, 2.999, 3.0, 3.001, 6.5] {
        let mut tape = Tape::<f64>::new(true);
        let _rec = tape.record();
        let a = AReal::input(x, &mut tape);
        let j1 = Jet1::new(x, 1.0);
        let j2 = Jet2::variable(x);

        for (name, want, got) in [
            ("erf", erf(x), [Real::erf(&x), j1.erf().value(), j2.erf().value(), Real::erf(&a).value()]),
            ("erfc", erfc(x), [Real::erfc(&x), j1.erfc().value(), j2.erfc().value(), Real::erfc(&a).value()]),
            ("norm_pdf", norm_pdf(x), [Real::norm_pdf(&x), j1.norm_pdf().value(), j2.norm_pdf().value(), Real::norm_pdf(&a).value()]),
            ("norm_cdf", norm_cdf(x), [Real::norm_cdf(&x), j1.norm_cdf().value(), j2.norm_cdf().value(), Real::norm_cdf(&a).value()]),
        ] {
            for (mode, g) in ["f64", "Jet1", "Jet2", "AReal"].iter().zip(got) {
                assert_eq!(g, want, "{mode}: Real::{name}({x})");
            }
        }
    }

    for &p in &[1e-6_f64, 0.02425, 0.2, 0.5, 0.8, 0.97575, 1.0 - 1e-6] {
        let mut tape = Tape::<f64>::new(true);
        let _rec = tape.record();
        let a = AReal::input(p, &mut tape);
        let want = xad_rs::math::inv_norm_cdf(p);
        for (mode, g) in ["f64", "Jet1", "Jet2", "AReal"].iter().zip([
            Real::inv_norm_cdf(&p),
            Jet1::new(p, 1.0).inv_norm_cdf().value(),
            Jet2::variable(p).inv_norm_cdf().value(),
            Real::inv_norm_cdf(&a).value(),
        ]) {
            assert_eq!(g, want, "{mode}: Real::inv_norm_cdf({p})");
        }
    }
}

#[test]
fn trait_norm_cdf_matches_reference_in_every_mode() {
    let cases: &[(f64, f64)] = &[
        (0.0, 0.5),
        (1.0, 0.8413447460685429),
        (-1.0, 0.15865525393145707),
        (1.959963984540054, 0.975),
        (2.575829303548901, 0.995),
        (-3.719016485455681, 1e-4),
    ];
    for &(x, want) in cases {
        let mut tape = Tape::<f64>::new(true);
        let _rec = tape.record();
        let a = AReal::input(x, &mut tape);
        for (mode, got) in ["f64", "Jet1", "Jet2", "AReal"].iter().zip([
            Real::norm_cdf(&x),
            Jet1::new(x, 1.0).norm_cdf().value(),
            Jet2::variable(x).norm_cdf().value(),
            Real::norm_cdf(&a).value(),
        ]) {
            assert!(
                (got - want).abs() <= 5.0 * f64::EPSILON * want.abs() + f64::EPSILON,
                "{mode}: Real::norm_cdf({x}) = {got:e}, want {want:e}"
            );
        }
    }
}

/// `erfc = 1 - erf` is the identity every AD surface differentiates, so the
/// passive free function must be exactly that expression — not an
/// independently evaluated tail.
#[test]
fn erfc_is_exactly_one_minus_erf() {
    for &x in &[-6.5_f64, -3.0, -0.5, 0.0, 0.5, 3.0, 6.5] {
        assert_eq!(erfc(x), 1.0 - erf(x), "erfc({x})");
    }
    assert_eq!(erfc(0.0), 1.0);
    assert_eq!(erfc(f64::INFINITY), 0.0);
    assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
    assert!(erfc(f64::NAN).is_nan());
}
