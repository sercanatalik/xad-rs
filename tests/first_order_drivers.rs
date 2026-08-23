//! First-order drivers: `compute_derivative_fwd`,
//! `compute_directional_derivative_fwd`, `compute_gradient_rev`.
//!
//! Each is checked against a closed form, and the two `Rⁿ → R` drivers are
//! cross-checked against each other and against the existing Jacobian
//! driver — the gradient must be that driver's single row.

use xad_rs::{
    compute_derivative_fwd, compute_directional_derivative_fwd, compute_gradient_rev,
    compute_jacobian_rev, AReal, Real, Tape,
};

/// `f(x) = x³·ln(x)`; `f'(x) = 3x²·ln(x) + x²`.
fn scalar<R: Real>(x: &R) -> R {
    x.powi(3) * x.ln()
}

/// `f(x, y, z) = x²·y + sin(z)·y + exp(x·z)`, written once against the trait
/// so both the forward and the reverse driver evaluate the same body.
fn multi<R: Real>(v: &[R]) -> R {
    let (x, y, z) = (v[0].clone(), v[1].clone(), v[2].clone());
    x.clone() * x.clone() * y.clone() + z.sin() * y + (x * z).exp()
}

const POINT: [f64; 3] = [0.7, 1.3, -0.4];

/// Closed-form gradient of `multi` at `POINT`.
fn multi_grad(p: &[f64]) -> [f64; 3] {
    let (x, y, z) = (p[0], p[1], p[2]);
    [
        2.0 * x * y + z * (x * z).exp(),
        x * x + z.sin(),
        y * z.cos() + x * (x * z).exp(),
    ]
}

#[test]
fn scalar_derivative_matches_the_closed_form() {
    for &x0 in &[0.3_f64, 1.0, 2.5, 7.25] {
        let (v, d) = compute_derivative_fwd(x0, scalar);
        let want_v = x0.powi(3) * x0.ln();
        let want_d = 3.0 * x0 * x0 * x0.ln() + x0 * x0;
        assert_eq!(v, want_v, "value at x={x0}");
        assert!(
            (d - want_d).abs() <= 1e-14 * (1.0 + want_d.abs()),
            "derivative at x={x0}: got {d}, want {want_d}"
        );
    }
}

#[test]
fn scalar_derivative_creates_no_tape() {
    // A tape live on the thread must stay untouched: forward mode records
    // nothing.
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let (_, d) = compute_derivative_fwd(2.0_f64, scalar);
    assert!(d.is_finite());
    assert_eq!(tape.num_statements(), 0);
    assert_eq!(tape.num_operations(), 0);
}

#[test]
fn unit_seeds_recover_the_partial_derivatives() {
    let want = multi_grad(&POINT);
    for i in 0..3 {
        let mut seed = [0.0; 3];
        seed[i] = 1.0;
        let (v, d) = compute_directional_derivative_fwd(&POINT, &seed, multi);
        assert!((v - multi(&POINT)).abs() < 1e-15, "value on direction {i}");
        assert!(
            (d - want[i]).abs() <= 1e-13 * (1.0 + want[i].abs()),
            "∂f/∂x{i}: got {d}, want {}",
            want[i]
        );
    }
}

#[test]
fn a_general_seed_is_the_inner_product_of_gradient_and_seed() {
    let g = multi_grad(&POINT);
    for seed in [[0.5_f64, 2.0, -1.5], [1.0, 1.0, 1.0], [-0.25, 0.0, 3.0]] {
        let (_, d) = compute_directional_derivative_fwd(&POINT, &seed, multi);
        let want: f64 = g.iter().zip(&seed).map(|(a, b)| a * b).sum();
        assert!(
            (d - want).abs() <= 1e-13 * (1.0 + want.abs()),
            "∇f·v: got {d}, want {want}"
        );
    }
}

#[test]
#[should_panic(expected = "seed length must match inputs")]
fn a_mismatched_seed_length_panics() {
    compute_directional_derivative_fwd(&POINT, &[1.0, 0.0], multi);
}

#[test]
fn gradient_matches_the_closed_form_and_the_forward_partials() {
    let (v, g) = compute_gradient_rev(&POINT, multi);
    assert!((v - multi(&POINT)).abs() < 1e-15, "value");

    let want = multi_grad(&POINT);
    for i in 0..3 {
        assert!(
            (g[i] - want[i]).abs() <= 1e-13 * (1.0 + want[i].abs()),
            "closed form ∂f/∂x{i}: got {}, want {}",
            g[i],
            want[i]
        );

        // ... and against the forward driver, one unit direction at a time.
        let mut seed = [0.0; 3];
        seed[i] = 1.0;
        let (_, fwd) = compute_directional_derivative_fwd(&POINT, &seed, multi);
        assert!(
            (g[i] - fwd).abs() <= 1e-13 * (1.0 + fwd.abs()),
            "forward vs reverse ∂f/∂x{i}: {} vs {fwd}",
            g[i]
        );
    }
}

#[test]
fn gradient_equals_the_jacobians_single_row() {
    let (_, g) = compute_gradient_rev(&POINT, multi);
    let jac = compute_jacobian_rev(&POINT, |v: &[AReal<f64>]| vec![multi(v)]);
    assert_eq!(jac.shape(), &[1, 3]);
    assert_eq!(g, jac.row(0).to_vec(), "gradient must be the Jacobian's row");
}

#[test]
fn gradient_leaves_no_tape_active() {
    // The driver's tape must be gone by the time it returns, so a second
    // call — or any other recording — does not hit the no-nesting panic.
    let (_, g1) = compute_gradient_rev(&POINT, multi);
    let (_, g2) = compute_gradient_rev(&POINT, multi);
    assert_eq!(g1, g2);

    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let x = AReal::input(1.0_f64, &mut tape);
    assert_eq!(x.value(), 1.0);
}

#[test]
fn drivers_agree_on_a_body_written_once_against_the_trait() {
    // The same generic body reaches all three drivers plus the passive
    // evaluation, and every value must be identical.
    let v_passive = multi(&POINT);
    let (v_fwd, _) = compute_directional_derivative_fwd(&POINT, &[1.0, 0.0, 0.0], multi);
    let (v_rev, _) = compute_gradient_rev(&POINT, multi);
    assert_eq!(v_passive, v_fwd);
    assert_eq!(v_passive, v_rev);

    // And the scalar driver on a one-input slice of the same shape.
    let (v_scalar, _) = compute_derivative_fwd(POINT[0], scalar);
    assert_eq!(v_scalar, scalar(&POINT[0]));
}
