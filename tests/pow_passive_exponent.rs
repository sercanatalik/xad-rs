//! `pow` with a derivative-free exponent: no `ln(base)`, and a finite base
//! partial wherever the value is finite — in every mode.
//!
//! Before this change the forward modes computed `d_base·b' + (b^e·ln b)·0`,
//! and for a negative or zero base the `ln` is NaN, so `powf(-3, 2)` had value
//! 9 and derivative NaN. Reverse mode escaped only because the inactive
//! operand was dropped at push time.

use xad_rs::math::{ad, fwd, fwdk};
use xad_rs::{AReal, Jet1, Jet2, JetK, Real, Tape, compute_gradient_rev};

fn rev_d(b: f64, e: f64) -> (f64, f64) {
    let (v, g) = compute_gradient_rev(&[b], |x| x[0].powf(AReal::from(e)));
    (v, g[0])
}
fn jet1_d(b: f64, e: f64) -> (f64, f64) {
    let y = Jet1::new(b, 1.0).powf(Jet1::constant(e));
    (y.value(), y.derivative())
}
fn jetk_d(b: f64, e: f64) -> (f64, f64) {
    let y = JetK::<f64, 4>::new(b, [0.0, 1.0, 0.0, 0.0]).powf(JetK::constant(e));
    (y.value, y.tangents[1])
}
fn jet2_d(b: f64, e: f64) -> (f64, f64, f64) {
    let y = <Jet2<f64> as Real>::powf(&Jet2::variable(b), Jet2::constant(e));
    (y.value(), y.first_derivative(), y.second_derivative())
}

#[test]
fn positive_base_matches_the_passive_exponent_spelling_bit_for_bit() {
    for &(b, e) in &[(1.7_f64, 2.5_f64), (0.3, -1.25), (4.0, 0.5), (2.0, 30.0)] {
        // Reverse: the two-AReal `pow` with an unrecorded exponent equals `powf(base, e)`.
        let (v, d) = rev_d(b, e);
        let (v2, g2) = compute_gradient_rev(&[b], |x| ad::powf(&x[0], e));
        assert_eq!((v, d), (v2, g2[0]), "reverse ({b}, {e})");
        // Jet1 / JetK against their `powf(base, e)` free functions.
        let j = fwd::powf(&Jet1::new(b, 1.0), e);
        assert_eq!(jet1_d(b, e), (j.value(), j.derivative()), "Jet1 ({b}, {e})");
        let k = fwdk::powf(&JetK::<f64, 4>::new(b, [0.0, 1.0, 0.0, 0.0]), e);
        assert_eq!(jetk_d(b, e), (k.value, k.tangents[1]), "JetK ({b}, {e})");
        // Jet2 against its inherent direct form.
        let d2 = Jet2::variable(b).powf(e);
        assert_eq!(jet2_d(b, e), (d2.value(), d2.first_derivative(), d2.second_derivative()), "Jet2 ({b}, {e})");
        // And the value is the passive scalar's.
        assert_eq!(v, b.powf(e));
    }
}

#[test]
fn negative_base_with_an_integer_exponent_has_a_finite_derivative() {
    let (b, e) = (-3.0_f64, 2.0_f64);
    let want_d = e * b.powf(e - 1.0); // -6
    let want_d2 = e * (e - 1.0) * b.powf(e - 2.0); // 2
    assert_eq!(rev_d(b, e), (9.0, want_d), "reverse");
    assert_eq!(jet1_d(b, e), (9.0, want_d), "Jet1");
    assert_eq!(jetk_d(b, e), (9.0, want_d), "JetK");
    assert_eq!(jet2_d(b, e), (9.0, want_d, want_d2), "Jet2");
    // Odd power keeps the sign.
    let (b, e) = (-2.0_f64, 3.0_f64);
    assert_eq!(jet1_d(b, e), (-8.0, 12.0));
    assert_eq!(jetk_d(b, e), (-8.0, 12.0));
    assert_eq!(jet2_d(b, e).1, 12.0);
}

#[test]
fn zero_base_has_the_partial_pow_d_base_defines() {
    // e > 1: partial 0; e == 1: partial 1. Neither may be NaN.
    for (e, want) in [(2.0_f64, 0.0_f64), (1.0, 1.0), (3.5, 0.0)] {
        assert_eq!(rev_d(0.0, e).1, want, "reverse e={e}");
        assert_eq!(jet1_d(0.0, e).1, want, "Jet1 e={e}");
        assert_eq!(jetk_d(0.0, e).1, want, "JetK e={e}");
        assert!(!jet2_d(0.0, e).1.is_nan(), "Jet2 e={e}");
    }
}

#[test]
fn an_active_exponent_still_carries_the_log_partial() {
    let (b, e) = (1.7_f64, 2.5_f64);
    let want = b.powf(e) * b.ln();
    let j = fwd::pow(&Jet1::constant(b), &Jet1::new(e, 1.0));
    assert_eq!(j.derivative(), want);
    let k = fwdk::pow(&JetK::<f64, 2>::constant(b), &JetK::new(e, [0.0, 1.0]));
    assert_eq!(k.tangents[1], want);
    let (_, g) = compute_gradient_rev(&[b, e], |x| x[0].powf(x[1]));
    assert_eq!(g[1], want);
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let x = AReal::input(b, &mut tape);
    let y = AReal::input(e, &mut tape);
    let _ = ad::pow(&x, &y);
    assert_eq!(tape.num_operations(), 2, "active exponent records two operands");
    let _ = ad::pow(&x, &AReal::from(e));
    assert_eq!(tape.num_operations(), 3, "passive exponent records one operand");
}
