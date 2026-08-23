use xad_rs::{AReal, Jet1, Jet2, Real, Tape};

/// The point the body is evaluated at. Chosen so the quotient below is one
/// where a correctly rounded `a / b` and the two-rounding `a * (1/b)` land on
/// different `f64`s — see `poly_point_distinguishes_the_two_quotient_forms`.
const POLY_X0: f64 = 2.6;

/// The shift in the denominator, chosen with `POLY_X0` for the same reason.
const POLY_SHIFT: f64 = 1.1;

/// `(x - 1)^2 / (x + 1.1)` — addition, subtraction, multiplication *and*
/// division in one body.
///
/// The division is not decoration. A mode is supposed to change which
/// derivatives are available, not which number comes out, and division is the
/// operation where that property is easiest to lose: a quotient formed as
/// `a * (1/b)` rounds twice where `a / b` rounds once. A body without a
/// division cannot see that, which is how it went unnoticed here.
fn poly<R: Real>(x: &R) -> R {
    let num = x.clone() * x.clone() - R::from(2.0_f64) * x.clone() + R::from(1.0_f64);
    num / (x.clone() + R::from(POLY_SHIFT))
}

/// The body above is only a gate on division at a point where the two
/// spellings of its quotient actually differ — they agree for most inputs, and
/// for *every* input whose numerator is a power of two, since rescaling by one
/// is exact either way. Pin that, so moving `POLY_X0` cannot silently turn the
/// test below back into one that passes against a reciprocal-built quotient.
#[test]
fn poly_point_distinguishes_the_two_quotient_forms() {
    let num = POLY_X0 * POLY_X0 - 2.0 * POLY_X0 + 1.0;
    let den = POLY_X0 + POLY_SHIFT;
    assert_ne!(
        num * (1.0 / den),
        num / den,
        "poly's evaluation point no longer distinguishes a/b from a*(1/b)"
    );
}

#[test]
fn poly_value_agrees_across_modes() {
    // The passive scalar is the reference, and the comparison is bit-exact:
    // an active mode may carry extra derivative information, but it does not
    // get to return a different number.
    let v_ref = poly(&POLY_X0);

    let j1 = Jet1::new(POLY_X0, 1.0);
    assert_eq!(poly(&j1).value(), v_ref, "Jet1 value");

    let j2 = Jet2::variable(POLY_X0);
    assert_eq!(poly(&j2).value(), v_ref, "Jet2 value");

    let mut tape = Tape::<f64>::new(true);
    tape.activate();
    let mut x = AReal::new(POLY_X0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    let got = poly(&x).value();
    Tape::<f64>::deactivate_all();
    assert_eq!(got, v_ref, "AReal value");
}

// ============================================================================
// Extended Real surface: one generic body exercising the full elementary
// set (including the methods added when Real grew from 7 to 24 methods),
// run under all four Real implementors.
// ============================================================================

/// A contrived kernel that touches every extended-`Real` method at least
/// once, on inputs kept inside every function's domain.
///
/// Every argument is mapped into its function's domain explicitly rather
/// than passed through as-is: `acosh` needs `> 1`, and `inv_norm_cdf`
/// **panics** outside the open unit interval, so a naive `x.inv_norm_cdf()`
/// here would either abort the test or hand the cross-mode comparison a NaN
/// to report as a false disagreement.
fn kitchen_sink<R: Real>(x: &R) -> R {
    // x is around 0.4 — in-domain for asin/acos/atanh (|x| < 1).
    let a = x.sin().cos().tan().atan();
    let b = x.asin() + x.acos() + x.atanh();
    let c = x.sinh() + x.cosh() + x.tanh() + x.asinh();
    let d = (x.clone() + R::from(1.5_f64)).acosh(); // arg ~1.9 > 1
    let e = x.exp() + x.exp2() + x.exp_m1();
    let f = (x.clone() + R::from(1.0_f64)).ln()
        + (x.clone() + R::from(1.0_f64)).log2()
        + (x.clone() + R::from(1.0_f64)).log10()
        + x.ln_1p();
    let g = x.sqrt() + x.cbrt() + (-x.clone()).abs();
    let h = x.max(&R::from(0.2_f64)) + x.min(&R::from(0.9_f64));
    a + b + c + d + e + f + g + h + gaussians(x)
}

/// The Gaussian family, on arguments mapped into each function's domain.
///
/// `inv_norm_cdf` is fed `Φ(x)`, which is in `(0, 1)` by construction for
/// any finite `x` — so the composition is the identity in exact arithmetic
/// and stays a valid probe of both methods' derivatives.
fn gaussians<R: Real>(x: &R) -> R {
    let p = x.norm_cdf(); // in (0, 1)
    x.erf() + x.erfc() + x.norm_pdf() + p.clone() + p.inv_norm_cdf()
}

/// Reverse and forward mode accumulate the *same* per-operation partials in
/// different orders — forward multiplies each tangent into the chain as it
/// goes, reverse sums adjoint contributions back-to-front — so a body whose
/// derivative is a sum of several paths can differ in the last bit. Values
/// are still compared bit-exactly; only forward-vs-reverse derivatives get
/// this few-ulp bound.
fn assert_agrees_to_a_few_ulp(a: f64, b: f64, what: &str) {
    let tol = 8.0 * f64::EPSILON * (1.0 + a.abs().max(b.abs()));
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol:e})");
}

#[test]
fn extended_real_surface_agrees_across_modes() {
    let x0 = 0.4_f64;

    // Reference: plain f64 through the same generic body.
    let v_ref = kitchen_sink(&x0);
    // Reference derivative: central finite difference of the f64 path.
    let h = 1e-7;
    let d_ref = (kitchen_sink(&(x0 + h)) - kitchen_sink(&(x0 - h))) / (2.0 * h);

    // Forward Jet1: value bit-equal, derivative vs FD.
    let j1 = kitchen_sink(&Jet1::new(x0, 1.0));
    assert_eq!(j1.value(), v_ref);
    assert!((j1.derivative() - d_ref).abs() < 1e-6, "Jet1 d1 {} vs FD {}", j1.derivative(), d_ref);

    // Second-order Jet2: value and d1 agree with Jet1.
    let j2 = kitchen_sink(&Jet2::variable(x0));
    assert_eq!(j2.value(), v_ref);
    assert_eq!(j2.first_derivative(), j1.derivative());

    // Reverse AReal: value bit-equal, adjoint bit-equal to the Jet1 tangent.
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let x = AReal::input(x0, &mut tape);
    let mut y = kitchen_sink(&x);
    y.register(&mut tape);
    y.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert_eq!(y.value(), v_ref);
    assert_agrees_to_a_few_ulp(x.adjoint(&tape), j1.derivative(), "AReal vs Jet1 d1");
}

// ============================================================================
// Numeric identities: `Real::zero()` / `Real::one()` must behave as identities
// on the *value* and leave the derivative untouched in every active mode.
// ============================================================================

/// `((x + 0) * 1)` — must be indistinguishable from `x` in value and in every
/// derivative the mode carries.
fn through_identities<R: Real>(x: &R) -> R {
    (x.clone() + R::zero()) * R::one()
}

#[test]
fn identities_are_neutral_in_every_mode() {
    let x0 = 1.7_f64;

    // Passive.
    assert_eq!(through_identities(&x0), x0);
    assert_eq!(<f64 as Real>::zero(), 0.0);
    assert_eq!(<f64 as Real>::one(), 1.0);

    // Forward first order: value and tangent both survive.
    let j1 = Jet1::new(x0, 1.0);
    let r1 = through_identities(&j1);
    assert_eq!(r1.value(), x0);
    assert_eq!(r1.derivative(), 1.0);
    // The identities themselves carry no tangent.
    assert_eq!(<Jet1<f64> as Real>::zero().derivative(), 0.0);
    assert_eq!(<Jet1<f64> as Real>::one().derivative(), 0.0);

    // Forward second order: value, d1 and d2 all survive.
    let j2 = through_identities(&Jet2::variable(x0));
    assert_eq!(j2.value(), x0);
    assert_eq!(j2.first_derivative(), 1.0);
    assert_eq!(j2.second_derivative(), 0.0);
    assert_eq!(<Jet2<f64> as Real>::one().first_derivative(), 0.0);
    assert_eq!(<Jet2<f64> as Real>::one().second_derivative(), 0.0);

    // Reverse: value and adjoint both survive.
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let x = AReal::input(x0, &mut tape);
    let mut y = through_identities(&x);
    y.register(&mut tape);
    y.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert_eq!(y.value(), x0);
    assert_eq!(x.adjoint(&tape), 1.0);
}

#[test]
fn reverse_identities_are_unrecorded_constants() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let zero = <AReal<f64> as Real>::zero();
    let one = <AReal<f64> as Real>::one();
    assert_eq!(zero.value(), 0.0);
    assert_eq!(one.value(), 1.0);
    // A literal has no antecedent: producing one must not grow the tape.
    assert_eq!(tape.num_statements(), 0);
    assert_eq!(tape.num_operations(), 0);
}

// ============================================================================
// The Gaussian family through the trait: values agree across all four
// implementors, and the active modes' derivatives agree with each other and
// with a finite difference of the passive path.
// ============================================================================

#[test]
fn gaussian_family_agrees_across_modes() {
    for &x0 in &[-1.8_f64, -0.35, 0.0, 0.62, 2.4] {
        let v_ref = gaussians(&x0);

        // Forward first order.
        let j1 = gaussians(&Jet1::new(x0, 1.0));
        assert_eq!(j1.value(), v_ref, "Jet1 value at x={x0}");

        // Forward second order: value and d1 must match Jet1 bit-for-bit.
        let j2 = gaussians(&Jet2::variable(x0));
        assert_eq!(j2.value(), v_ref, "Jet2 value at x={x0}");
        assert_eq!(j2.first_derivative(), j1.derivative(), "Jet2 d1 at x={x0}");

        // Reverse.
        let mut tape = Tape::<f64>::new(true);
        let _rec = tape.record();
        let x = AReal::input(x0, &mut tape);
        let mut y = gaussians(&x);
        y.register(&mut tape);
        y.set_adjoint(&mut tape, 1.0);
        tape.compute_adjoints();
        assert_eq!(y.value(), v_ref, "AReal value at x={x0}");
        assert_agrees_to_a_few_ulp(
            x.adjoint(&tape),
            j1.derivative(),
            &format!("AReal vs Jet1 d1 at x={x0}"),
        );

        // Finite difference of the passive path — the only approximate side.
        let h = 1e-6 * (1.0 + x0.abs());
        let fd = (gaussians(&(x0 + h)) - gaussians(&(x0 - h))) / (2.0 * h);
        assert!(
            (j1.derivative() - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
            "AD {} vs FD {fd} at x={x0}",
            j1.derivative()
        );
    }
}

/// The tangents of the Gaussian methods must be the *exact* analytic ones,
/// not the derivative of the approximation each value is computed with.
#[test]
fn gaussian_tangents_are_exact() {
    use std::f64::consts::FRAC_2_SQRT_PI;

    for &x0 in &[-2.7_f64, -0.9, 0.0, 0.45, 1.6, 3.4] {
        let seed = || Jet1::new(x0, 1.0);

        // Φ'(x) = φ(x), bit-exactly the passive density.
        assert_eq!(
            seed().norm_cdf().derivative(),
            xad_rs::math::norm_pdf(x0),
            "Φ' at x={x0}"
        );
        // φ'(x) = -x·φ(x).
        assert_eq!(
            seed().norm_pdf().derivative(),
            -x0 * xad_rs::math::norm_pdf(x0),
            "φ' at x={x0}"
        );
        // erf'(x) = (2/√π)·e^{-x²}; erfc' = -erf'.
        let want = FRAC_2_SQRT_PI * (-x0 * x0).exp();
        assert_eq!(seed().erf().derivative(), want, "erf' at x={x0}");
        assert_eq!(seed().erfc().derivative(), -want, "erfc' at x={x0}");
    }

    // (Φ⁻¹)'(p) = 1/φ(Φ⁻¹(p)) — the analytic reciprocal density, not the
    // derivative of Acklam's rational approximation.
    for &p in &[0.01_f64, 0.15, 0.5, 0.83, 0.995] {
        let out = Jet1::new(p, 1.0).inv_norm_cdf();
        assert_eq!(
            out.derivative(),
            1.0 / xad_rs::math::norm_pdf(out.value()),
            "(Φ⁻¹)' at p={p}"
        );
    }
}

/// `φ''(x) = (x² − 1)·φ(x)` through the trait in the second-order mode,
/// checked against a finite difference of the analytic first derivative.
#[test]
fn density_second_derivative_through_the_trait() {
    for &x0 in &[-2.1_f64, -0.5, 0.0, 0.8, 1.9] {
        let d2 = Jet2::variable(x0).norm_pdf().second_derivative();
        assert_eq!(d2, (x0 * x0 - 1.0) * xad_rs::math::norm_pdf(x0), "closed form");

        let h = 1e-5;
        let d1_at = |x: f64| Jet1::new(x, 1.0).norm_pdf().derivative();
        let fd = (d1_at(x0 + h) - d1_at(x0 - h)) / (2.0 * h);
        assert!(
            (d2 - fd).abs() <= 1e-9 * (1.0 + d2.abs()),
            "φ'' at x={x0}: {d2} vs fd(φ') {fd}"
        );
    }
}
