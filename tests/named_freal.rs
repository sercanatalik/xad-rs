//! LBLF-06 cross-check tests for `NamedFReal<f64>`.
//!
//! Each test asserts bit-exact (`assert_eq!`) equality between the named
//! wrapper and a positional `FReal<f64>` running the same expression.
//!
//! Phase 02.2 Plan 02.2-01: rewritten to the `NamedForwardTape` API.
//! The wrapper no longer carries an `Arc<VarRegistry>` field; values are
//! obtained exclusively via `ft.input_freal` / `ft.constant_freal`. The
//! LBLF-06 bit-exactness guarantee holds byte-identically after the
//! rewrite — that is the whole point of this file.


use xad_rs::FReal;
use xad_rs::{NamedFReal, NamedForwardTape};
use xad_rs::math;

#[test]
fn test_expr1_arithmetic() {
    // f(x) = x*x + x - 2.0
    let xp: FReal<f64> = FReal::new(3.0, 1.0);
    let fp = &(&xp * &xp) + &xp - 2.0;

    let mut ft = NamedForwardTape::new();
    let xl: NamedFReal<f64> = ft.input_freal("x", 3.0);
    let _registry = ft.freeze();
    let fl = &(&xl * &xl) + &xl - 2.0_f64;

    assert_eq!(fl.value(), fp.value());
    assert_eq!(fl.derivative("x"), fp.derivative());
}

#[test]
fn test_expr2_division() {
    // f(x) = (x + 1.0) / (x - 2.0)
    let xp: FReal<f64> = FReal::new(5.0, 1.0);
    let fp = &(&xp + 1.0) / &(&xp - 2.0);

    let mut ft = NamedForwardTape::new();
    let xl: NamedFReal<f64> = ft.input_freal("x", 5.0);
    let _registry = ft.freeze();
    let fl = &(&xl + 1.0_f64) / &(&xl - 2.0_f64);

    assert_eq!(fl.value(), fp.value());
    assert_eq!(fl.derivative("x"), fp.derivative());
}

#[test]
fn test_expr3_trig() {
    // f(x) = sin(x) + cos(x)
    //
    // NamedFReal forwards elementaries via `math::fwd::*` inside the
    // wrapper; the named accessor is asserted end-to-end.
    let xp: FReal<f64> = FReal::new(0.5, 1.0);
    let fp = math::fwd::sin(&xp) + math::fwd::cos(&xp);

    let mut ft = NamedForwardTape::new();
    let xl: NamedFReal<f64> = ft.input_freal("x", 0.5);
    let _registry = ft.freeze();
    let fl = &xl.sin() + &xl.cos();

    assert_eq!(fl.value(), fp.value());
    assert_eq!(fl.derivative("x"), fp.derivative());
}

#[test]
fn test_expr4_exp_sqrt() {
    // f(x) = exp(sqrt(x))
    //
    // Labeled accessor is asserted end-to-end — the composed
    // `exp(sqrt(x))` flows through `NamedFReal::sqrt` and
    // `NamedFReal::exp` without touching `.inner()`.
    let xp: FReal<f64> = FReal::new(4.0, 1.0);
    let fp = math::fwd::exp(&math::fwd::sqrt(&xp));

    let mut ft = NamedForwardTape::new();
    let xl: NamedFReal<f64> = ft.input_freal("x", 4.0);
    let _registry = ft.freeze();
    let fl = xl.sqrt().exp();

    assert_eq!(fl.value(), fp.value());
    assert_eq!(fl.derivative("x"), fp.derivative());
}

#[test]
fn test_constant_has_zero_derivative() {
    let mut ft = NamedForwardTape::new();
    // Register an input so the frozen registry is non-empty (not strictly
    // required for constant_freal, but mirrors typical usage).
    let _x: NamedFReal<f64> = ft.input_freal("x", 0.0);
    let c: NamedFReal<f64> = ft.constant_freal(42.0_f64);
    let _registry = ft.freeze();
    assert_eq!(c.value(), 42.0);
    assert_eq!(c.derivative("x"), 0.0);
}

#[test]
fn test_scalar_on_lhs() {
    // Exercise every f64 op NamedFReal<f64> path.
    let mut ft = NamedForwardTape::new();
    let x: NamedFReal<f64> = ft.input_freal("x", 2.0);
    let _registry = ft.freeze();

    let a = 1.0 + x.clone();
    assert_eq!(a.value(), 3.0);
    assert_eq!(a.derivative("x"), 1.0);

    let b = 5.0 - &x;
    assert_eq!(b.value(), 3.0);
    assert_eq!(b.derivative("x"), -1.0);

    let c = 3.0 * x.clone();
    assert_eq!(c.value(), 6.0);
    assert_eq!(c.derivative("x"), 3.0);

    let d = 8.0 / &x;
    assert_eq!(d.value(), 4.0);
    // d(8/x)/dx = -8/x^2 = -2.0
    assert_eq!(d.derivative("x"), -2.0);
}
