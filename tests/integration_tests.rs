use approx::assert_relative_eq;
use xad_rs::AReal;
use xad_rs::Jet2;
use xad_rs::Jet1;
use xad_rs::math;
use xad_rs::Tape;

const EPS: f64 = 1e-10;
const EPS_LOOSE: f64 = 1e-6;

// ============================================================================
// Basic reverse-mode tests
// ============================================================================

#[test]
fn test_areal_basic_add() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(3.0);
    let mut y = AReal::new(5.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut y), &mut tape);

    let mut z = &x + &y; // z = x + y = 8
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 8.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 1.0, epsilon = EPS); // dz/dx = 1
    assert_relative_eq!(y.adjoint(&tape), 1.0, epsilon = EPS); // dz/dy = 1

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_basic_mul() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(3.0);
    let mut y = AReal::new(5.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut y), &mut tape);

    let mut z = &x * &y; // z = x * y = 15
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 15.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 5.0, epsilon = EPS); // dz/dx = y
    assert_relative_eq!(y.adjoint(&tape), 3.0, epsilon = EPS); // dz/dy = x

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_basic_div() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(6.0);
    let mut y = AReal::new(3.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut y), &mut tape);

    let mut z = &x / &y; // z = x/y = 2
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 2.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 1.0 / 3.0, epsilon = EPS); // dz/dx = 1/y
    assert_relative_eq!(y.adjoint(&tape), -6.0 / 9.0, epsilon = EPS); // dz/dy = -x/y^2

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_basic_sub() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(7.0);
    let mut y = AReal::new(3.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut y), &mut tape);

    let mut z = &x - &y;
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 4.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 1.0, epsilon = EPS);
    assert_relative_eq!(y.adjoint(&tape), -1.0, epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_neg() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(4.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = -&x;
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), -4.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), -1.0, epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_scalar_ops() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(3.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    // z = 2*x + 1 = 7
    let mut z = &x * 2.0 + 1.0;
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 7.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 2.0, epsilon = EPS); // dz/dx = 2

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_compound_expression() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(2.0);
    let mut y = AReal::new(3.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut y), &mut tape);

    // z = x^2 * y + x * y^2
    // dz/dx = 2*x*y + y^2 = 2*2*3 + 9 = 21
    // dz/dy = x^2 + 2*x*y = 4 + 12 = 16
    let x_sq = &x * &x;
    let y_sq = &y * &y;
    let term1 = &x_sq * &y;
    let term2 = &x * &y_sq;
    let mut z = term1 + term2;
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 4.0 * 3.0 + 2.0 * 9.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 21.0, epsilon = EPS);
    assert_relative_eq!(y.adjoint(&tape), 16.0, epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

// ============================================================================
// Math function tests (reverse mode)
// ============================================================================

#[test]
fn test_areal_sin() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(1.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = math::ad::sin(&x);
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 1.0_f64.sin(), epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 1.0_f64.cos(), epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_cos() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(1.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = math::ad::cos(&x);
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 1.0_f64.cos(), epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), -1.0_f64.sin(), epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_exp() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(2.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = math::ad::exp(&x);
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 2.0_f64.exp(), epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 2.0_f64.exp(), epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_ln() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(3.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = math::ad::ln(&x);
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 3.0_f64.ln(), epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 1.0 / 3.0, epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_sqrt() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(4.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = math::ad::sqrt(&x);
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 2.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 0.25, epsilon = EPS); // 1/(2*sqrt(4)) = 0.25

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_powf() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(2.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = math::ad::powf(&x, 3.0); // x^3 = 8
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 8.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 12.0, epsilon = EPS); // 3*x^2 = 12

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_tanh() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(0.5);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = math::ad::tanh(&x);
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    let tv = 0.5_f64.tanh();
    assert_relative_eq!(z.value(), tv, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 1.0 - tv * tv, epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_areal_erf() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(0.5);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = math::ad::erf(&x);
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    // erf(0.5) ≈ 0.5204998778
    assert_relative_eq!(z.value(), 0.5204998778, epsilon = 1e-5);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    // d(erf)/dx = 2/sqrt(pi) * exp(-x^2)
    let expected = 2.0 / std::f64::consts::PI.sqrt() * (-0.25_f64).exp();
    assert_relative_eq!(x.adjoint(&tape), expected, epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

// ---- norm_cdf / inv_norm_cdf across all types ----

#[test]
fn test_norm_cdf_scalar_known_values() {
    // Φ(0) = 0.5 exactly
    assert_relative_eq!(math::norm_cdf(0.0_f64), 0.5, epsilon = 1e-7);
    // Φ(-∞) → 0, Φ(+∞) → 1  (test large values)
    assert!(math::norm_cdf(-8.0_f64) < 1e-15);
    assert!(math::norm_cdf(8.0_f64) > 1.0 - 1e-15);
    // Φ(1) ≈ 0.8413447
    assert_relative_eq!(math::norm_cdf(1.0_f64), 0.8413447, epsilon = 2e-7);
}

#[test]
fn test_inv_norm_cdf_scalar_round_trip() {
    for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
        let x = math::inv_norm_cdf(p);
        let p_back = math::norm_cdf(x);
        // Round-trip accuracy limited by erf's A&S approximation (~1.5e-7).
        assert_relative_eq!(p_back, p, epsilon = 5e-7);
    }
}

#[test]
fn test_norm_cdf_dual_derivative() {
    // d/dx Φ(x) = φ(x) = (1/√(2π)) · exp(-x²/2)
    let x = Jet1::new(0.5, 1.0);
    let f = math::fwd::norm_cdf(&x);
    let expected_pdf = math::norm_pdf(0.5);
    assert_relative_eq!(f.value(), math::norm_cdf(0.5), epsilon = 1e-7);
    assert_relative_eq!(f.derivative(), expected_pdf, epsilon = 1e-12);
}

#[test]
fn test_inv_norm_cdf_dual_derivative() {
    // d/dp Φ⁻¹(p) = 1 / φ(Φ⁻¹(p))
    let p = Jet1::new(0.3, 1.0);
    let f = math::fwd::inv_norm_cdf(&p);
    let r = math::inv_norm_cdf(0.3);
    let expected_deriv = 1.0 / math::norm_pdf(r);
    assert_relative_eq!(f.value(), r, epsilon = 1e-9);
    assert_relative_eq!(f.derivative(), expected_deriv, epsilon = 1e-10);
}

#[test]
fn test_norm_cdf_dual2_second_derivative() {
    // Φ''(x) = -x · φ(x)
    let x = Jet2::variable(1.0_f64);
    let f = x.norm_cdf();
    let pdf = math::norm_pdf(1.0);
    assert_relative_eq!(f.first_derivative(), pdf, epsilon = 1e-12);
    assert_relative_eq!(f.second_derivative(), -pdf, epsilon = 1e-12);
}

#[test]
fn test_inv_norm_cdf_dual2_second_derivative() {
    // g'(p) = 1/φ(Φ⁻¹(p)),  g''(p) = Φ⁻¹(p) · g'(p)²
    let p = Jet2::variable(0.7_f64);
    let f = p.inv_norm_cdf();
    let r = math::inv_norm_cdf(0.7);
    let gp = 1.0 / math::norm_pdf(r);
    let gpp = r * gp * gp;
    assert_relative_eq!(f.value(), r, epsilon = 1e-9);
    assert_relative_eq!(f.first_derivative(), gp, epsilon = 1e-10);
    assert_relative_eq!(f.second_derivative(), gpp, epsilon = 1e-8);
}

#[test]
fn test_norm_cdf_areal_adjoint() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(0.5);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = math::ad::norm_cdf(&x);
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), math::norm_cdf(0.5), epsilon = 1e-7);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), math::norm_pdf(0.5), epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_inv_norm_cdf_areal_adjoint() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut p = AReal::new(0.3);
    AReal::register_input(std::slice::from_mut(&mut p), &mut tape);

    let mut z = math::ad::inv_norm_cdf(&p);
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    let r = math::inv_norm_cdf(0.3);
    assert_relative_eq!(z.value(), r, epsilon = 1e-9);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    let expected_deriv = 1.0 / math::norm_pdf(r);
    assert_relative_eq!(p.adjoint(&tape), expected_deriv, epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_norm_cdf_freal() {
    let x = Jet1::new(0.5, 1.0);
    let f = math::fwd::norm_cdf(&x);
    assert_relative_eq!(f.value(), math::norm_cdf(0.5), epsilon = 1e-7);
    assert_relative_eq!(f.derivative(), math::norm_pdf(0.5), epsilon = 1e-12);
}

#[test]
fn test_norm_cdf_inv_round_trip_fwd() {
    // Φ⁻¹(Φ(x)) = x  — verifies both functions compose correctly through AD.
    let x = Jet1::new(0.8, 1.0);
    let f = math::fwd::inv_norm_cdf(&math::fwd::norm_cdf(&x));
    assert_relative_eq!(f.value(), 0.8, epsilon = 1e-6);
    assert_relative_eq!(f.derivative(), 1.0, epsilon = 1e-5);
}

// ============================================================================
// Forward-mode tests
// ============================================================================

#[test]
fn test_freal_basic_add() {
    let x = Jet1::new(3.0, 1.0); // dx/dx = 1
    let y = Jet1::new(5.0, 0.0); // dy/dx = 0

    let z = &x + &y;
    assert_relative_eq!(z.value(), 8.0, epsilon = EPS);
    assert_relative_eq!(z.derivative(), 1.0, epsilon = EPS);
}

#[test]
fn test_freal_basic_mul() {
    let x = Jet1::new(3.0, 1.0);
    let y = Jet1::new(5.0, 0.0);

    let z = &x * &y;
    assert_relative_eq!(z.value(), 15.0, epsilon = EPS);
    assert_relative_eq!(z.derivative(), 5.0, epsilon = EPS); // d(xy)/dx = y
}

#[test]
fn test_freal_basic_div() {
    let x = Jet1::new(6.0, 1.0);
    let y = Jet1::new(3.0, 0.0);

    let z = &x / &y;
    assert_relative_eq!(z.value(), 2.0, epsilon = EPS);
    assert_relative_eq!(z.derivative(), 1.0 / 3.0, epsilon = EPS);
}

#[test]
fn test_freal_compound() {
    // f(x) = x^2 + 2x + 1 at x=3
    // f'(x) = 2x + 2 = 8
    let x = Jet1::new(3.0, 1.0);
    let z = &x * &x + &x * 2.0 + 1.0;
    assert_relative_eq!(z.value(), 16.0, epsilon = EPS);
    assert_relative_eq!(z.derivative(), 8.0, epsilon = EPS);
}

#[test]
fn test_freal_sin() {
    let x = Jet1::new(1.0, 1.0);
    let z = math::fwd::sin(&x);
    assert_relative_eq!(z.value(), 1.0_f64.sin(), epsilon = EPS);
    assert_relative_eq!(z.derivative(), 1.0_f64.cos(), epsilon = EPS);
}

#[test]
fn test_freal_exp() {
    let x = Jet1::new(2.0, 1.0);
    let z = math::fwd::exp(&x);
    assert_relative_eq!(z.value(), 2.0_f64.exp(), epsilon = EPS);
    assert_relative_eq!(z.derivative(), 2.0_f64.exp(), epsilon = EPS);
}

#[test]
fn test_freal_ln() {
    let x = Jet1::new(3.0, 1.0);
    let z = math::fwd::ln(&x);
    assert_relative_eq!(z.value(), 3.0_f64.ln(), epsilon = EPS);
    assert_relative_eq!(z.derivative(), 1.0 / 3.0, epsilon = EPS);
}

#[test]
fn test_freal_chain_rule() {
    // f(x) = exp(sin(x)) at x=1
    // f'(x) = cos(x) * exp(sin(x))
    let x = Jet1::new(1.0, 1.0);
    let z = math::fwd::exp(&math::fwd::sin(&x));
    let expected_deriv = 1.0_f64.cos() * 1.0_f64.sin().exp();
    assert_relative_eq!(z.value(), 1.0_f64.sin().exp(), epsilon = EPS);
    assert_relative_eq!(z.derivative(), expected_deriv, epsilon = EPS);
}

// ============================================================================
// Jacobian tests
// ============================================================================

#[test]
fn test_jacobian_rev() {
    // f(x, y) = [x*y, x+y]
    // J = [[y, x], [1, 1]]
    let jac = xad_rs::compute_jacobian_rev(&[3.0, 5.0], |inputs| {
        vec![&inputs[0] * &inputs[1], &inputs[0] + &inputs[1]]
    });

    assert_eq!(jac.dim(), (2, 2));
    assert_relative_eq!(jac[[0, 0]], 5.0, epsilon = EPS); // dy/dx1 = x2
    assert_relative_eq!(jac[[0, 1]], 3.0, epsilon = EPS); // dy/dx2 = x1
    assert_relative_eq!(jac[[1, 0]], 1.0, epsilon = EPS);
    assert_relative_eq!(jac[[1, 1]], 1.0, epsilon = EPS);
}

// ============================================================================
// Hessian test
// ============================================================================

#[test]
fn test_hessian() {
    // f(x, y) = x^2*y + y^3
    // df/dx = 2xy, df/dy = x^2 + 3y^2
    // H = [[2y, 2x], [2x, 6y]]
    // At (2, 3): H = [[6, 4], [4, 18]]
    let hess = xad_rs::compute_hessian(&[2.0, 3.0], |inputs| {
        let x_sq = &inputs[0] * &inputs[0];
        let y_cu = &inputs[1] * &inputs[1] * &inputs[1];
        x_sq * &inputs[1] + y_cu
    });

    assert_relative_eq!(hess[[0, 0]], 6.0, epsilon = EPS_LOOSE);
    assert_relative_eq!(hess[[0, 1]], 4.0, epsilon = EPS_LOOSE);
    assert_relative_eq!(hess[[1, 0]], 4.0, epsilon = EPS_LOOSE);
    assert_relative_eq!(hess[[1, 1]], 18.0, epsilon = EPS_LOOSE);
}

// ============================================================================
// Tape management tests
// ============================================================================

#[test]
fn test_tape_new_recording() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    // First computation
    let mut x = AReal::new(2.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    let mut y = &x * &x;
    AReal::register_output(std::slice::from_mut(&mut y), &mut tape);

    y.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert_relative_eq!(x.adjoint(&tape), 4.0, epsilon = EPS);

    // New recording
    tape.new_recording();

    let mut x2 = AReal::new(3.0);
    AReal::register_input(std::slice::from_mut(&mut x2), &mut tape);
    let mut y2 = &x2 * &x2;
    AReal::register_output(std::slice::from_mut(&mut y2), &mut tape);

    y2.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert_relative_eq!(x2.adjoint(&tape), 6.0, epsilon = EPS);

    Tape::<f64>::deactivate_all();
}

// ============================================================================
// Comparison & conversion tests
// ============================================================================

#[test]
fn test_areal_comparison() {
    let a = AReal::new(3.0);
    let b = AReal::new(5.0);
    assert!(a < b);
    assert!(b > a);
    assert!(a != b);
    assert!(a == AReal::new(3.0));
    assert!(a < 4.0);
}

#[test]
fn test_areal_from_scalar() {
    let a: AReal<f64> = AReal::from(2.75);
    assert_relative_eq!(a.value(), 2.75, epsilon = EPS);
    assert!(!a.should_record());
}

#[test]
fn test_freal_comparison() {
    let a = Jet1::new(3.0, 1.0);
    let b = Jet1::new(5.0, 2.0);
    assert!(a < b);
    assert!(a != b);
}

// ============================================================================
// Compound assignment tests
// ============================================================================

#[test]
fn test_areal_add_assign() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(2.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    let mut z = AReal::new(1.0);
    z += &x; // z = 1 + x = 3
    z += &x; // z = 1 + x + x = 5
    AReal::register_output(std::slice::from_mut(&mut z), &mut tape);

    assert_relative_eq!(z.value(), 5.0, epsilon = EPS);

    z.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_relative_eq!(x.adjoint(&tape), 2.0, epsilon = EPS); // dz/dx = 2

    Tape::<f64>::deactivate_all();
}

#[test]
fn test_freal_mul_assign() {
    let mut x = Jet1::new(3.0, 1.0);
    let y = Jet1::new(4.0, 0.0);
    x *= &y;
    // x = 3*4 = 12, dx = 1*4 + 3*0 = 4
    assert_relative_eq!(x.value(), 12.0, epsilon = EPS);
    assert_relative_eq!(x.derivative(), 4.0, epsilon = EPS);
}

// ============================================================================
// Black-Scholes example (practical use case)
// ============================================================================

#[test]
fn test_black_scholes_greeks() {
    // Simple Black-Scholes call price: C = S*N(d1) - K*exp(-rT)*N(d2)
    // We compute dC/dS (delta) and dC/dr (rho) using reverse mode

    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut spot = AReal::new(100.0);
    let mut rate = AReal::new(0.05);
    let mut vol = AReal::new(0.2);
    AReal::register_input(std::slice::from_mut(&mut spot), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut rate), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut vol), &mut tape);

    let strike = 100.0;
    let t = 1.0;

    // d1 = (ln(S/K) + (r + vol^2/2)*T) / (vol*sqrt(T))
    let ln_sk = math::ad::ln(&(&spot / strike));
    let vol_sq = &vol * &vol;
    let half_vol_sq = &vol_sq * 0.5;
    let r_plus = &rate + &half_vol_sq;
    let numerator = &ln_sk + &r_plus * t;
    let denom = &vol * t.sqrt();
    let d1 = &numerator / &denom;
    let d2 = &d1 - &denom;

    // N(x) ≈ 0.5 * (1 + erf(x/sqrt(2)))
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    let n_d1_arg = &d1 * sqrt2_inv;
    let n_d2_arg = &d2 * sqrt2_inv;
    let erf_d1 = math::ad::erf(&n_d1_arg);
    let erf_d2 = math::ad::erf(&n_d2_arg);
    let n_d1 = (&erf_d1 + 1.0) * 0.5;
    let n_d2 = (&erf_d2 + 1.0) * 0.5;

    let discount = math::ad::exp(&(&rate * (-t)));
    let mut call = &spot * &n_d1 - strike * &discount * &n_d2;
    AReal::register_output(std::slice::from_mut(&mut call), &mut tape);

    // Call price should be positive and reasonable
    assert!(call.value() > 0.0);
    assert!(call.value() < 100.0);

    call.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    let delta = spot.adjoint(&tape);
    let rho = rate.adjoint(&tape);
    let vega = vol.adjoint(&tape);

    // Delta should be between 0 and 1 for a call
    assert!(delta > 0.0 && delta < 1.0, "delta = {}", delta);
    // Rho should be positive for a call
    assert!(rho > 0.0, "rho = {}", rho);
    // Vega should be positive
    assert!(vega > 0.0, "vega = {}", vega);

    Tape::<f64>::deactivate_all();
}

// ============================================================================
// Jet2 tests (dedicated second-order forward mode)
// ============================================================================

#[test]
fn test_jet2_polynomial() {
    // f(x) = x^3 at x = 2: f=8, f'=12, f''=12
    let x = Jet2::variable(2.0_f64);
    let y = x * x * x;
    assert_relative_eq!(y.value(), 8.0, epsilon = EPS);
    assert_relative_eq!(y.first_derivative(), 12.0, epsilon = EPS);
    assert_relative_eq!(y.second_derivative(), 12.0, epsilon = EPS);
}

#[test]
fn test_jet2_powf() {
    // f(x) = x^4 at x = 3: f=81, f'=108, f''=108
    let x = Jet2::variable(3.0_f64);
    let y = x.powf(4.0);
    assert_relative_eq!(y.value(), 81.0, epsilon = EPS);
    assert_relative_eq!(y.first_derivative(), 108.0, epsilon = EPS);
    assert_relative_eq!(y.second_derivative(), 108.0, epsilon = EPS);
}

#[test]
fn test_jet2_exp_ln() {
    // f(x) = ln(exp(x)) = x. So f'=1, f''=0 exactly (chain-rule cancellation).
    let x = Jet2::variable(1.7_f64);
    let y = x.exp().ln();
    assert_relative_eq!(y.value(), 1.7, epsilon = EPS);
    assert_relative_eq!(y.first_derivative(), 1.0, epsilon = 1e-12);
    assert_relative_eq!(y.second_derivative(), 0.0, epsilon = 1e-12);
}

#[test]
fn test_jet2_sin() {
    // f(x) = sin(x) at x = 0.5
    let x = Jet2::variable(0.5_f64);
    let y = x.sin();
    assert_relative_eq!(y.value(), 0.5_f64.sin(), epsilon = EPS);
    assert_relative_eq!(y.first_derivative(), 0.5_f64.cos(), epsilon = EPS);
    assert_relative_eq!(y.second_derivative(), -0.5_f64.sin(), epsilon = EPS);
}

#[test]
fn test_jet2_div() {
    // f(x) = 1/x^2 at x = 2: f=0.25, f'=-2/x^3=-0.25, f''=6/x^4=0.375
    let x = Jet2::variable(2.0_f64);
    let y = Jet2::constant(1.0) / (x * x);
    assert_relative_eq!(y.value(), 0.25, epsilon = EPS);
    assert_relative_eq!(y.first_derivative(), -0.25, epsilon = EPS);
    assert_relative_eq!(y.second_derivative(), 0.375, epsilon = EPS);
}

