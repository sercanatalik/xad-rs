use xad_rs::{AReal, Jet1, Jet2, JetK, Real, Tape};

// f(x) = sin(x^2); f'(x) = 2x * cos(x^2)
fn f<R: Real>(x: &R) -> R {
    let x2 = x.clone() * x.clone();
    x2.sin()
}

#[test]
fn derivative_at_three_agrees_across_modes() {
    let x0 = 3.0_f64;

    let analytic = 2.0 * x0 * (x0 * x0).cos();

    // (1) finite difference on f64
    let h = 1e-5;
    let fd = (f(&(x0 + h)) - f(&(x0 - h))) / (2.0 * h);

    // (2) Jet1 forward
    let j1 = Jet1::new(x0, 1.0);
    let j1_deriv = f(&j1).derivative();

    // (3) Jet2 forward (first derivative)
    let j2 = Jet2::variable(x0);
    let j2_deriv = f(&j2).first_derivative();

    // (4) JetK forward, lane 0 seeded
    let jk_deriv = f(&JetK::<f64, 2>::new(x0, [1.0, 0.0])).tangents[0];

    // (5) AReal reverse
    let mut tape = Tape::<f64>::new(true);
    tape.activate();
    let mut x = AReal::new(x0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    let mut y = f(&x);
    AReal::register_output(std::slice::from_mut(&mut y), &mut tape);
    y.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    let areal_deriv = x.adjoint(&tape);
    Tape::<f64>::deactivate_all();

    // All five results within FD-limited tolerance.
    let tol = 1e-5;
    assert!((fd - analytic).abs() < tol, "FD vs analytic: {} vs {}", fd, analytic);
    assert!((j1_deriv - analytic).abs() < tol, "Jet1 vs analytic: {} vs {}", j1_deriv, analytic);
    assert!((j2_deriv - analytic).abs() < tol, "Jet2 vs analytic: {} vs {}", j2_deriv, analytic);
    assert!((jk_deriv - analytic).abs() < tol, "JetK vs analytic: {} vs {}", jk_deriv, analytic);
    // No division in `f`: the K-lane tangent is Jet1's, bit for bit.
    assert_eq!(jk_deriv, j1_deriv, "JetK lane 0 vs Jet1");
    assert!((areal_deriv - analytic).abs() < tol, "AReal vs analytic: {} vs {}", areal_deriv, analytic);
}

// f(x) = Φ(x)·erf(x) + φ(x) — a body routed entirely through the Gaussian
// family, which only became writable generically once `Real`'s method set was
// stamped from the elementary table.
//
// f'(x) = φ(x)·erf(x) + Φ(x)·(2/√π)·e^{-x²} − x·φ(x)
fn gaussian_body<R: Real>(x: &R) -> R {
    x.norm_cdf() * x.erf() + x.norm_pdf()
}

#[test]
fn gaussian_derivative_agrees_across_modes() {
    use std::f64::consts::FRAC_2_SQRT_PI;

    for &x0 in &[-1.4_f64, -0.25, 0.0, 0.6, 2.2] {
        let phi = xad_rs::math::norm_pdf(x0);
        let analytic = phi * xad_rs::math::erf(x0)
            + xad_rs::math::norm_cdf(x0) * FRAC_2_SQRT_PI * (-x0 * x0).exp()
            - x0 * phi;

        // (1) finite difference on f64
        let h = 1e-6;
        let fd = (gaussian_body(&(x0 + h)) - gaussian_body(&(x0 - h))) / (2.0 * h);

        // (2) Jet1 forward
        let j1_deriv = gaussian_body(&Jet1::new(x0, 1.0)).derivative();

        // (3) Jet2 forward (first derivative)
        let j2_deriv = gaussian_body(&Jet2::variable(x0)).first_derivative();

        // (4) JetK forward, lane 0 seeded
        let jk_deriv = gaussian_body(&JetK::<f64, 2>::new(x0, [1.0, 0.0])).tangents[0];

        // (5) AReal reverse
        let mut tape = Tape::<f64>::new(true);
        let _rec = tape.record();
        let x = AReal::input(x0, &mut tape);
        let mut y = gaussian_body(&x);
        y.register(&mut tape);
        y.set_adjoint(&mut tape, 1.0);
        tape.compute_adjoints();
        let areal_deriv = x.adjoint(&tape);
        drop(_rec);

        // The three active modes are exact: they carry the analytic tangent,
        // not the derivative of the value approximation. Only FD is loose.
        let exact = 1e-13 * (1.0 + analytic.abs());
        assert!((j1_deriv - analytic).abs() < exact, "Jet1 at {x0}: {j1_deriv} vs {analytic}");
        assert!((j2_deriv - analytic).abs() < exact, "Jet2 at {x0}: {j2_deriv} vs {analytic}");
        assert!((jk_deriv - analytic).abs() < exact, "JetK at {x0}: {jk_deriv} vs {analytic}");
        assert!(
            (areal_deriv - analytic).abs() < exact,
            "AReal at {x0}: {areal_deriv} vs {analytic}"
        );
        assert!((fd - analytic).abs() < 1e-8, "FD at {x0}: {fd} vs {analytic}");
    }
}
