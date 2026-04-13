//! Analytical Hessian cross-check suite for `Dual2Vec`.
//!
//! Hand-derived literal Hessians at tolerance <= 1e-13. No xad-rs code
//! path computes the reference values — every expected entry is either a
//! compile-time numeric literal or a `std::f64::consts` expression. Every
//! test asserts `hess == hess.t()` bit-exactly as its first check, and no
//! test ever averages the two triangles before asserting (the Hessian is
//! structurally symmetric straight out of the op pipeline).
//!
//! The four expressions:
//!   1. `f(x, y) = x²·y + y³`            at `(1, 2)`
//!   2. `f(x, y) = sin(x·y)`             at `(1, π/2)`
//!   3. `f(x, y) = exp(x² + y²)`         at `(0.5, 0.5)`
//!   4. `f(x, y) = (x - y)² · log(x + y)` at `(2, 1)`


use approx::assert_abs_diff_eq;
use std::f64::consts::PI;
use xad_rs::Dual2Vec;

/// Tolerance for every analytical cross-check in this suite (<= 1e-13).
/// Do NOT loosen.
const TOL: f64 = 1e-13;

/// `f(x, y) = x²·y + y³` at `(x, y) = (1, 2)`.
///
/// Hand derivation:
/// - `f = 1·2 + 8 = 10`
/// - `∂f/∂x = 2xy = 4`
/// - `∂f/∂y = x² + 3y² = 13`
/// - `∂²f/∂x² = 2y = 4`
/// - `∂²f/∂y² = 6y = 12`
/// - `∂²f/∂x∂y = 2x = 2`
///
/// Expected Hessian: `[[4, 2], [2, 12]]`.
#[test]
fn test_hessian_x2y_plus_y3() {
    let x = Dual2Vec::variable(1.0, 0, 2);
    let y = Dual2Vec::variable(2.0, 1, 2);
    // f = x²·y + y³ = (x * x) * y + (y * y) * y
    let f = &(&(&x * &x) * &y) + &(&(&y * &y) * &y);

    // Bit-exact symmetry BEFORE any other assertion.
    assert_eq!(f.hessian(), &f.hessian().t());

    assert_abs_diff_eq!(f.value(), 10.0, epsilon = TOL);
    assert_abs_diff_eq!(f.gradient()[0], 4.0, epsilon = TOL);
    assert_abs_diff_eq!(f.gradient()[1], 13.0, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[0, 0]], 4.0, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[0, 1]], 2.0, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[1, 0]], 2.0, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[1, 1]], 12.0, epsilon = TOL);
}

/// `f(x, y) = sin(x·y)` at `(x, y) = (1, π/2)`.
///
/// Hand derivation:
/// - Let `u = xy`, `c = cos u`, `s = sin u`. At `u = π/2`: `c = 0`, `s = 1`.
/// - `f = sin(π/2) = 1`
/// - `∂f/∂x = y·c = (π/2)·0 = 0`
/// - `∂f/∂y = x·c = 1·0 = 0`
/// - `∂²f/∂x² = -y²·s = -π²/4`
/// - `∂²f/∂y² = -x²·s = -1`
/// - `∂²f/∂x∂y = c - xy·s = 0 - (π/2)·1 = -π/2`
///
/// Expected Hessian: `[[-π²/4, -π/2], [-π/2, -1]]`.
#[test]
fn test_hessian_sin_xy() {
    let x = Dual2Vec::variable(1.0, 0, 2);
    let y = Dual2Vec::variable(PI / 2.0, 1, 2);
    let f = (&x * &y).sin();

    // Bit-exact symmetry BEFORE any other assertion.
    assert_eq!(f.hessian(), &f.hessian().t());

    // At u = π/2, sin u = 1 up to f64 precision of π/2 itself.
    assert_abs_diff_eq!(f.value(), 1.0, epsilon = TOL);
    assert_abs_diff_eq!(f.gradient()[0], 0.0, epsilon = TOL);
    assert_abs_diff_eq!(f.gradient()[1], 0.0, epsilon = TOL);

    let expected_xx = -(PI * PI) / 4.0;
    let expected_yy = -1.0;
    let expected_xy = -PI / 2.0;
    assert_abs_diff_eq!(f.hessian()[[0, 0]], expected_xx, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[1, 1]], expected_yy, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[0, 1]], expected_xy, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[1, 0]], expected_xy, epsilon = TOL);
}

/// `f(x, y) = exp(x² + y²)` at `(x, y) = (0.5, 0.5)`.
///
/// Hand derivation:
/// - `u = 0.25 + 0.25 = 0.5`
/// - `e = exp(0.5) ≈ 1.6487212707001282`
/// - `∂f/∂x = 2x·e = e`
/// - `∂f/∂y = 2y·e = e`
/// - `∂²f/∂x² = (2 + 4x²)·e = 3e`
/// - `∂²f/∂y² = (2 + 4y²)·e = 3e`
/// - `∂²f/∂x∂y = 4xy·e = e`
///
/// Expected Hessian: `e · [[3, 1], [1, 3]]`.
#[test]
fn test_hessian_exp_x2_plus_y2() {
    let x = Dual2Vec::variable(0.5, 0, 2);
    let y = Dual2Vec::variable(0.5, 1, 2);
    let f = (&(&x * &x) + &(&y * &y)).exp();

    // Bit-exact symmetry BEFORE any other assertion.
    assert_eq!(f.hessian(), &f.hessian().t());

    let e = 0.5_f64.exp();
    assert_abs_diff_eq!(f.value(), e, epsilon = TOL);
    assert_abs_diff_eq!(f.gradient()[0], e, epsilon = TOL);
    assert_abs_diff_eq!(f.gradient()[1], e, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[0, 0]], 3.0 * e, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[1, 1]], 3.0 * e, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[0, 1]], e, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[1, 0]], e, epsilon = TOL);
}

/// `f(x, y) = (x - y)² · log(x + y)` at `(x, y) = (2, 1)`.
///
/// Hand derivation (reproduced here in full so the test is self-contained).
///
/// Let `d = x - y`, `s = x + y`, `ℓ = ln(s)`. At `(2, 1)`:
/// `d = 1`, `s = 3`, `ℓ = ln 3 ≈ 1.0986122886681098`.
///
/// First derivatives (product rule on `d² · ℓ`):
/// - `∂f/∂x = 2d·ℓ + d²/s = 2·ℓ + 1/3`
/// - `∂f/∂y = -2d·ℓ + d²/s = -2·ℓ + 1/3`
///
/// Second derivatives:
///
/// `∂²f/∂x²` from `∂f/∂x = 2d·ℓ + d²/s`:
/// - `∂/∂x [2d·ℓ] = 2·ℓ + 2d · (1/s) = 2ℓ + 2d/s`
/// - `∂/∂x [d²/s]` (quotient rule): `(2d·s − d²)/s²`
/// - Sum at (2,1): `2ℓ + 2/3 + 5/9 = 2ℓ + 11/9`
///
/// `∂²f/∂y²` from `∂f/∂y = -2d·ℓ + d²/s`:
/// - `∂/∂y [-2d·ℓ] = 2·ℓ + (-2d) · (1/s) = 2ℓ − 2d/s`
/// - `∂/∂y [d²/s]`: `(-2d·s − d²)/s²`
/// - Sum at (2,1): `2ℓ − 2/3 − 7/9 = 2ℓ − 13/9`
///
/// `∂²f/∂x∂y` from `∂f/∂x = 2d·ℓ + d²/s` differentiated w.r.t. y:
/// - `∂/∂y [2d·ℓ] = -2·ℓ + 2d · (1/s) = -2ℓ + 2d/s`
/// - `∂/∂y [d²/s]`: `(-2d·s − d²)/s²`
/// - Sum at (2,1): `-2ℓ + 2/3 − 7/9 = -2ℓ − 1/9`
///
/// Symmetric cross-check from `∂f/∂y` differentiated w.r.t. x:
/// - `∂/∂x [-2d·ℓ] = -2·ℓ + (-2d)·(1/s) = -2ℓ − 2d/s`
/// - `∂/∂x [d²/s]`: `(2d·s − d²)/s²`
/// - Sum at (2,1): `-2ℓ − 2/3 + 5/9 = -2ℓ − 1/9` ✓ matches `∂²f/∂x∂y`.
///
/// Numeric values with `ln(3) = 1.0986122886681098`:
/// - `H[0, 0] = 2·ln(3) + 11/9 ≈ 3.419446799558441`
/// - `H[1, 1] = 2·ln(3) − 13/9 ≈ 0.752780132891775`
/// - `H[0, 1] = H[1, 0] = -2·ln(3) − 1/9 ≈ -2.308335688447330`
///
/// Function value and gradient at (2, 1):
/// - `f = d²·ℓ = 1 · ln(3) = ln 3`
/// - `∂f/∂x = 2·ln(3) + 1/3 ≈ 2.530557910669552`
/// - `∂f/∂y = -2·ln(3) + 1/3 ≈ -1.863891244002885`
#[test]
fn test_hessian_x_minus_y_squared_log_x_plus_y() {
    let x = Dual2Vec::variable(2.0, 0, 2);
    let y = Dual2Vec::variable(1.0, 1, 2);
    // f = (x - y)² · ln(x + y)
    let diff = &x - &y;
    let sum = &x + &y;
    let f = &(&diff * &diff) * &sum.ln();

    // Bit-exact symmetry BEFORE any other assertion.
    assert_eq!(f.hessian(), &f.hessian().t());

    let ln3 = 3.0_f64.ln();

    // value: (x-y)² · ln(x+y) = 1 · ln 3 = ln 3
    assert_abs_diff_eq!(f.value(), ln3, epsilon = TOL);

    // gradient
    assert_abs_diff_eq!(f.gradient()[0], 2.0 * ln3 + 1.0 / 3.0, epsilon = TOL);
    assert_abs_diff_eq!(f.gradient()[1], -2.0 * ln3 + 1.0 / 3.0, epsilon = TOL);

    // Hessian
    let h00 = 2.0 * ln3 + 11.0 / 9.0;
    let h11 = 2.0 * ln3 - 13.0 / 9.0;
    let h01 = -2.0 * ln3 - 1.0 / 9.0;
    assert_abs_diff_eq!(f.hessian()[[0, 0]], h00, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[1, 1]], h11, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[0, 1]], h01, epsilon = TOL);
    assert_abs_diff_eq!(f.hessian()[[1, 0]], h01, epsilon = TOL);
}
