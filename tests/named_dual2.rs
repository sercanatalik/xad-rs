//! LBLF-06 cross-check tests for `NamedDual2<f64>`.
//!
//! Unlike `NamedDual` and `NamedFReal`, `NamedDual2` tracks ONE
//! seeded direction and its second derivative. Tests verify that both
//! first and second derivative accessors match the positional `Dual2`
//! for four expressions, and that operations between two differently-
//! seeded variables panic in debug builds.
//!
//! Migrated to the Shape A `NamedForwardTape` + `declare_dual2_f64` +
//! `freeze_dual` + `scope.dual2(handle)` API in Plan 02.2-02 Task 5.


use xad_rs::Dual2;
use xad_rs::NamedForwardTape;

#[test]
fn test_expr1_polynomial() {
    // f(x) = x^3, f'(2) = 12, f''(2) = 12
    let xp: Dual2<f64> = Dual2::variable(2.0);
    let fp = xp * xp * xp;

    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 2.0);
    let scope = ft.freeze_dual();
    let xl = scope.dual2(x_h);
    let fl = xl.clone() * xl.clone() * xl.clone();

    assert_eq!(fl.value(), fp.value());
    assert_eq!(fl.first_derivative("x"), fp.first_derivative());
    assert_eq!(fl.second_derivative("x"), fp.second_derivative());
}

#[test]
fn test_expr2_division() {
    // f(x) = (x + 1) / (x - 2)
    let xp: Dual2<f64> = Dual2::variable(5.0);
    let fp = (xp + 1.0_f64) / (xp - 2.0_f64);

    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 5.0);
    let scope = ft.freeze_dual();
    let xl = scope.dual2(x_h);
    let fl = (xl.clone() + 1.0_f64) / (xl.clone() - 2.0_f64);

    assert_eq!(fl.value(), fp.value());
    assert_eq!(fl.first_derivative("x"), fp.first_derivative());
    assert_eq!(fl.second_derivative("x"), fp.second_derivative());
}

#[test]
fn test_expr3_trig() {
    // f(x) = sin(x)
    //
    // NamedDual2 forwards elementaries (sin/cos/exp/ln/sqrt) directly;
    // the named accessor is asserted end-to-end.
    let xp: Dual2<f64> = Dual2::variable(0.5);
    let fp = xp.sin();

    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 0.5);
    let scope = ft.freeze_dual();
    let xl = scope.dual2(x_h);
    let fl = xl.sin();

    assert_eq!(fl.value(), fp.value());
    assert_eq!(fl.first_derivative("x"), fp.first_derivative());
    assert_eq!(fl.second_derivative("x"), fp.second_derivative());
}

#[test]
fn test_expr4_exp_times_x() {
    // f(x) = exp(x) * x
    //
    // Labeled accessor is asserted end-to-end — `NamedDual2::exp`
    // forwards through the wrapper, then multiplication uses the named
    // operator surface.
    let xp: Dual2<f64> = Dual2::variable(1.0);
    let fp = xp.exp() * xp;

    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 1.0);
    let scope = ft.freeze_dual();
    let xl = scope.dual2(x_h);
    let fl = xl.exp() * xl.clone();

    assert_eq!(fl.value(), fp.value());
    assert_eq!(fl.first_derivative("x"), fp.first_derivative());
    assert_eq!(fl.second_derivative("x"), fp.second_derivative());
}

#[test]
fn test_constant_has_no_seed() {
    let mut ft = NamedForwardTape::new();
    let _x_h = ft.declare_dual2_f64("x", 0.0);
    let scope = ft.freeze_dual();
    let c = scope.constant_dual2_f64(42.0);
    assert_eq!(c.value(), 42.0);
    assert_eq!(c.first_derivative("x"), 0.0);
    assert_eq!(c.second_derivative("x"), 0.0);
}

#[test]
fn test_constant_plus_variable_preserves_seed() {
    // Adding a constant (no seed) to a variable (seeded) must preserve
    // the variable's seeded direction.
    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 3.0);
    let scope = ft.freeze_dual();
    let x = scope.dual2(x_h);
    let c = scope.constant_dual2_f64(10.0);
    let f = x.clone() + c;
    assert_eq!(f.value(), 13.0);
    assert_eq!(f.first_derivative("x"), 1.0);
    assert_eq!(f.second_derivative("x"), 0.0);
}

#[test]
fn test_seeded_direction_returns_inner_derivs() {
    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 3.0);
    let scope = ft.freeze_dual();
    let x = scope.dual2(x_h);
    let f = x.clone() * x.clone(); // f(x) = x^2
    assert_eq!(f.value(), 9.0);
    assert_eq!(f.first_derivative("x"), 6.0); // 2x
    assert_eq!(f.second_derivative("x"), 2.0);
}

#[test]
fn test_non_seeded_name_returns_zero() {
    // Registry has two names but the variable is seeded on only one.
    // Use declare_dual to register "y" (we only care that it ends up in
    // the final registry — its seeded direction is never read).
    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 3.0);
    let _y_h = ft.declare_dual2_f64("y", 0.0);
    let scope = ft.freeze_dual();
    let x = scope.dual2(x_h);
    // f(x) = x^2 — only x is active in this expression, so y's seed
    // is irrelevant.
    let f = x.clone() * x.clone();
    assert_eq!(f.first_derivative("x"), 6.0);
    assert_eq!(f.second_derivative("x"), 2.0);
    // `y` is in the registry but not the active direction for `f`.
    assert_eq!(f.first_derivative("y"), 0.0);
    assert_eq!(f.second_derivative("y"), 0.0);
}

#[test]
#[should_panic(expected = "not present in registry")]
fn test_first_derivative_unknown_name_panics() {
    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 1.0);
    let scope = ft.freeze_dual();
    let x = scope.dual2(x_h);
    let _ = x.first_derivative("missing");
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "cross-registry forward-mode op detected")]
fn test_cross_registry_panics_in_debug() {
    // Two tapes alive simultaneously — nested scopes. Each declared
    // value is stamped with its tape's TLS generation; the binary op
    // panics via `check_gen`.
    let mut ft_a = NamedForwardTape::new();
    let xa_h = ft_a.declare_dual2_f64("x", 1.0);
    let scope_a = ft_a.freeze_dual();
    let mut ft_b = NamedForwardTape::new();
    let xb_h = ft_b.declare_dual2_f64("x", 2.0);
    let scope_b = ft_b.freeze_dual();
    let xa = scope_a.dual2(xa_h);
    let xb = scope_b.dual2(xb_h);
    let _ = xa + xb;
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "differently-seeded variables")]
fn test_two_different_seeds_panics_in_debug() {
    // Two inputs on the SAME tape — same generation, but different
    // seeded positional indices. The merge_seeded combinator panics
    // in debug because `Dual2` supports only one active direction.
    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 1.0);
    let y_h = ft.declare_dual2_f64("y", 2.0);
    let scope = ft.freeze_dual();
    let x = scope.dual2(x_h);
    let y = scope.dual2(y_h);
    let _ = x + y;
}

#[test]
fn test_scalar_on_lhs() {
    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual2_f64("x", 2.0);
    let scope = ft.freeze_dual();
    let x = scope.dual2(x_h);

    // f64 + NamedDual2<f64> (owned)
    let a = 1.0_f64 + x.clone();
    assert_eq!(a.value(), 3.0);
    assert_eq!(a.first_derivative("x"), 1.0);

    // f64 - &NamedDual2<f64>
    let b = 5.0_f64 - x;
    assert_eq!(b.value(), 3.0);
    assert_eq!(b.first_derivative("x"), -1.0);

    // f64 * NamedDual2<f64>
    let c = 3.0_f64 * x.clone();
    assert_eq!(c.value(), 6.0);
    assert_eq!(c.first_derivative("x"), 3.0);

    // f64 / &NamedDual2<f64>
    let d = 8.0_f64 / x;
    assert_eq!(d.value(), 4.0);
    // d(8/x)/dx = -8/x^2 = -2
    assert_eq!(d.first_derivative("x"), -2.0);
}
