//! Labeled-hessian integration cross-check suite.
//!
//! Re-runs the 4 analytical expressions from `tests/dual2vec_analytical.rs`
//! through `compute_full_hessian` and asserts the returned
//! `NamedHessian` matches the same hand-derived literal Hessians at
//! tolerance `1e-13`. Also asserts:
//!
//! - `NamedHessian.vars.index_of("x") == Some(0)` and
//!   `NamedHessian.vars.index_of("y") == Some(1)` — confirms
//!   `VarRegistry` insertion order is preserved end-to-end from the
//!   `&[(String, f64)]` input slice through the output registry.
//! - `lh.hessian == lh.hessian.t()` bit-exactly BEFORE any literal
//!   check (symmetry-first assertion discipline).
//!
//! Literals are NOT re-derived — they are copied from
//! `tests/dual2vec_analytical.rs`, which has the full hand-derivation
//! comments. The expression closures mirror the `dual2vec_analytical`
//! test expressions byte-for-byte, adjusted only for the
//! `compute_full_hessian` closure argument type `&[Dual2Vec]`.
//!
//! Feature gate: `named-hessian` (transitively `named` + `dual2-vec`).


use approx::assert_abs_diff_eq;
use std::f64::consts::PI;
use xad_rs::Dual2Vec;
use xad_rs::compute_full_hessian;

/// Tolerance matching `tests/dual2vec_analytical.rs`. Do NOT loosen.
const TOL: f64 = 1e-13;

/// Test 1: `f(x, y) = x²·y + y³` at `(1, 2)`.
/// Expected Hessian `[[4, 2], [2, 12]]`. Literals copied from
/// `tests/dual2vec_analytical.rs::test_hessian_x2y_plus_y3`.
#[test]
fn test_compute_full_hessian_x2y_plus_y3() {
    let inputs = vec![("x".to_string(), 1.0), ("y".to_string(), 2.0)];
    let lh = compute_full_hessian(&inputs, |v: &[Dual2Vec]| {
        let x = &v[0];
        let y = &v[1];
        // f = x²·y + y³ = (x * x) * y + (y * y) * y
        &(&(x * x) * y) + &(&(y * y) * y)
    });

    // Registry insertion order (end-to-end contract).
    assert_eq!(lh.vars.index_of("x"), Some(0));
    assert_eq!(lh.vars.index_of("y"), Some(1));
    assert_eq!(lh.vars.len(), 2);

    // Shape + bit-exact symmetry BEFORE literal checks.
    assert_eq!(lh.hessian.dim(), (2, 2));
    assert_eq!(lh.hessian, lh.hessian.t());
    assert_eq!(lh.gradient.len(), 2);

    // Literal value + gradient + Hessian.
    assert_abs_diff_eq!(lh.value, 10.0, epsilon = TOL);
    assert_abs_diff_eq!(lh.gradient[0], 4.0, epsilon = TOL);
    assert_abs_diff_eq!(lh.gradient[1], 13.0, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[0, 0]], 4.0, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[0, 1]], 2.0, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[1, 0]], 2.0, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[1, 1]], 12.0, epsilon = TOL);
}

/// Test 2: `f(x, y) = sin(x·y)` at `(1, π/2)`.
/// Expected Hessian `[[-π²/4, -π/2], [-π/2, -1]]`. Literals copied from
/// `tests/dual2vec_analytical.rs::test_hessian_sin_xy`.
#[test]
fn test_compute_full_hessian_sin_xy() {
    let inputs = vec![("x".to_string(), 1.0), ("y".to_string(), PI / 2.0)];
    let lh = compute_full_hessian(&inputs, |v: &[Dual2Vec]| {
        // (&x * &y).sin()
        // The unary .sin() consumes self, so produce an owned Dual2Vec
        // from the &·& product first.
        (&v[0] * &v[1]).sin()
    });

    assert_eq!(lh.vars.index_of("x"), Some(0));
    assert_eq!(lh.vars.index_of("y"), Some(1));
    assert_eq!(lh.vars.len(), 2);

    assert_eq!(lh.hessian.dim(), (2, 2));
    assert_eq!(lh.hessian, lh.hessian.t());
    assert_eq!(lh.gradient.len(), 2);

    // At u = π/2, sin u = 1 up to f64 precision of π/2 itself.
    assert_abs_diff_eq!(lh.value, 1.0, epsilon = TOL);
    assert_abs_diff_eq!(lh.gradient[0], 0.0, epsilon = TOL);
    assert_abs_diff_eq!(lh.gradient[1], 0.0, epsilon = TOL);

    let expected_xx = -(PI * PI) / 4.0;
    let expected_yy = -1.0;
    let expected_xy = -PI / 2.0;
    assert_abs_diff_eq!(lh.hessian[[0, 0]], expected_xx, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[1, 1]], expected_yy, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[0, 1]], expected_xy, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[1, 0]], expected_xy, epsilon = TOL);
}

/// Test 3: `f(x, y) = exp(x² + y²)` at `(0.5, 0.5)`.
/// Expected Hessian `e · [[3, 1], [1, 3]]`. Literals copied from
/// `tests/dual2vec_analytical.rs::test_hessian_exp_x2_plus_y2`.
#[test]
fn test_compute_full_hessian_exp_x2_plus_y2() {
    let inputs = vec![("x".to_string(), 0.5), ("y".to_string(), 0.5)];
    let lh = compute_full_hessian(&inputs, |v: &[Dual2Vec]| {
        let x = &v[0];
        let y = &v[1];
        (&(x * x) + &(y * y)).exp()
    });

    assert_eq!(lh.vars.index_of("x"), Some(0));
    assert_eq!(lh.vars.index_of("y"), Some(1));
    assert_eq!(lh.vars.len(), 2);

    assert_eq!(lh.hessian.dim(), (2, 2));
    assert_eq!(lh.hessian, lh.hessian.t());
    assert_eq!(lh.gradient.len(), 2);

    let e = 0.5_f64.exp();
    assert_abs_diff_eq!(lh.value, e, epsilon = TOL);
    assert_abs_diff_eq!(lh.gradient[0], e, epsilon = TOL);
    assert_abs_diff_eq!(lh.gradient[1], e, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[0, 0]], 3.0 * e, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[1, 1]], 3.0 * e, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[0, 1]], e, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[1, 0]], e, epsilon = TOL);
}

/// Test 4: `f(x, y) = (x - y)² · log(x + y)` at `(2, 1)`.
///
/// Expected (see `tests/dual2vec_analytical.rs::test_hessian_x_minus_y_squared_log_x_plus_y`
/// for the 40-line hand derivation):
/// - `H[0, 0] = 2·ln(3) + 11/9`
/// - `H[1, 1] = 2·ln(3) − 13/9`
/// - `H[0, 1] = H[1, 0] = -2·ln(3) − 1/9`
#[test]
fn test_compute_full_hessian_x_minus_y_squared_log() {
    let inputs = vec![("x".to_string(), 2.0), ("y".to_string(), 1.0)];
    let lh = compute_full_hessian(&inputs, |v: &[Dual2Vec]| {
        let x = &v[0];
        let y = &v[1];
        // f = (x - y)² · ln(x + y)
        let diff = x - y;
        let sum = x + y;
        &(&diff * &diff) * &sum.ln()
    });

    assert_eq!(lh.vars.index_of("x"), Some(0));
    assert_eq!(lh.vars.index_of("y"), Some(1));
    assert_eq!(lh.vars.len(), 2);

    assert_eq!(lh.hessian.dim(), (2, 2));
    assert_eq!(lh.hessian, lh.hessian.t());
    assert_eq!(lh.gradient.len(), 2);

    let ln3 = 3.0_f64.ln();

    // value: (x-y)² · ln(x+y) = 1 · ln 3 = ln 3
    assert_abs_diff_eq!(lh.value, ln3, epsilon = TOL);

    // gradient
    assert_abs_diff_eq!(lh.gradient[0], 2.0 * ln3 + 1.0 / 3.0, epsilon = TOL);
    assert_abs_diff_eq!(lh.gradient[1], -2.0 * ln3 + 1.0 / 3.0, epsilon = TOL);

    // Hessian
    let h00 = 2.0 * ln3 + 11.0 / 9.0;
    let h11 = 2.0 * ln3 - 13.0 / 9.0;
    let h01 = -2.0 * ln3 - 1.0 / 9.0;
    assert_abs_diff_eq!(lh.hessian[[0, 0]], h00, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[1, 1]], h11, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[0, 1]], h01, epsilon = TOL);
    assert_abs_diff_eq!(lh.hessian[[1, 0]], h01, epsilon = TOL);
}
