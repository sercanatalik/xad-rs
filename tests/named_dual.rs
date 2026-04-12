//! LBLF-06 cross-check tests for `NamedDual`.
//!
//! Every test constructs the same expression in both positional and named
//! form and asserts bit-exact (NOT approximate) equality of value and
//! gradient. Any drift would be a wrapper bug, not a rounding artifact.
//!
//! Migrated to the Shape A `NamedForwardTape` + `declare_dual` +
//! `freeze_dual` + `scope.dual(handle)` API in Plan 02.2-02 Task 5.


use xad_rs::Dual;
use xad_rs::NamedForwardTape;

#[test]
fn test_expr1_arithmetic_only() {
    // f(x, y) = x*y + x - 2*y
    let xp = Dual::variable(2.0, 0, 2);
    let yp = Dual::variable(3.0, 1, 2);
    let fp = &(&xp * &yp) + &xp - 2.0 * &yp;

    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual("x", 2.0);
    let y_h = ft.declare_dual("y", 3.0);
    let scope = ft.freeze_dual();
    let xl = scope.dual(x_h);
    let yl = scope.dual(y_h);
    let fl = &(xl * yl) + xl - 2.0 * yl;

    assert_eq!(fl.real(), fp.real);
    assert_eq!(fl.partial("x"), fp.partial(0));
    assert_eq!(fl.partial("y"), fp.partial(1));
}

#[test]
fn test_expr2_division() {
    // f(x, y) = (x + 1) / (y - 3)
    let xp = Dual::variable(5.0, 0, 2);
    let yp = Dual::variable(7.0, 1, 2);
    let fp = &(&xp + 1.0) / &(&yp - 3.0);

    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual("x", 5.0);
    let y_h = ft.declare_dual("y", 7.0);
    let scope = ft.freeze_dual();
    let xl = scope.dual(x_h);
    let yl = scope.dual(y_h);
    let fl = &(xl + 1.0) / &(yl - 3.0);

    assert_eq!(fl.real(), fp.real);
    assert_eq!(fl.partial("x"), fp.partial(0));
    assert_eq!(fl.partial("y"), fp.partial(1));
}

#[test]
fn test_expr3_trig_exp() {
    // f(x, y) = sin(x) * exp(y)
    //
    // NamedDual forwards elementary math methods (sin/cos/exp/ln/sqrt/tan)
    // directly; the named accessor is asserted end-to-end.
    let xp = Dual::variable(0.5, 0, 2);
    let yp = Dual::variable(0.3, 1, 2);
    let fp = &xp.sin() * &yp.exp();

    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual("x", 0.5);
    let y_h = ft.declare_dual("y", 0.3);
    let scope = ft.freeze_dual();
    let xl = scope.dual(x_h);
    let yl = scope.dual(y_h);
    let fl = &xl.sin() * &yl.exp();

    assert_eq!(fl.real(), fp.real);
    assert_eq!(fl.partial("x"), fp.partial(0));
    assert_eq!(fl.partial("y"), fp.partial(1));
}

#[test]
fn test_expr4_sqrt_compound() {
    // f(x, y) = sqrt(x*x + y*y)
    //
    // Labeled accessor is asserted end-to-end — the elementary `sqrt`
    // flows through `NamedDual::sqrt` without touching `.inner()`.
    let xp = Dual::variable(3.0, 0, 2);
    let yp = Dual::variable(4.0, 1, 2);
    let sum_p = &(&xp * &xp) + &(&yp * &yp);
    let fp = sum_p.sqrt();

    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual("x", 3.0);
    let y_h = ft.declare_dual("y", 4.0);
    let scope = ft.freeze_dual();
    let xl = scope.dual(x_h);
    let yl = scope.dual(y_h);
    let sum_l = &(xl * xl) + &(yl * yl);
    let fl = sum_l.sqrt();

    assert_eq!(fl.real(), fp.real);
    assert_eq!(fl.partial("x"), fp.partial(0));
    assert_eq!(fl.partial("y"), fp.partial(1));
}

#[test]
fn test_gradient_returns_insertion_order() {
    // If registry has ["z", "a", "m"], gradient() MUST yield them in that
    // order, not alphabetical ["a", "m", "z"].
    let mut ft = NamedForwardTape::new();
    let z_h = ft.declare_dual("z", 1.0);
    let a_h = ft.declare_dual("a", 2.0);
    let m_h = ft.declare_dual("m", 3.0);
    let scope = ft.freeze_dual();
    let z = scope.dual(z_h);
    let a = scope.dual(a_h);
    let m = scope.dual(m_h);
    let f = z + a + m;
    let grad = f.gradient();
    assert_eq!(grad.len(), 3);
    assert_eq!(grad[0].0, "z");
    assert_eq!(grad[1].0, "a");
    assert_eq!(grad[2].0, "m");
    assert_eq!(grad[0].1, 1.0);
    assert_eq!(grad[1].1, 1.0);
    assert_eq!(grad[2].1, 1.0);
}

#[test]
fn test_real_and_constant() {
    let mut ft = NamedForwardTape::new();
    let _x_h = ft.declare_dual("x", 0.0);
    let _y_h = ft.declare_dual("y", 0.0);
    let scope = ft.freeze_dual();
    let c = scope.constant_dual(42.0);
    assert_eq!(c.real(), 42.0);
    assert_eq!(c.partial("x"), 0.0);
    assert_eq!(c.partial("y"), 0.0);
}

#[test]
#[should_panic(expected = "not present in registry")]
fn test_partial_unknown_name_panics() {
    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual("x", 1.0);
    let scope = ft.freeze_dual();
    let x = scope.dual(x_h);
    let _ = x.partial("missing");
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "cross-registry forward-mode op detected")]
fn test_cross_registry_add_panics_in_debug() {
    // Two tapes alive simultaneously — nested scopes. Each scope
    // stamped its values with its own TLS generation. The binary op
    // panics via the `check_gen` guard.
    let mut ft_a = NamedForwardTape::new();
    let xa_h = ft_a.declare_dual("x", 2.0);
    let scope_a = ft_a.freeze_dual();
    let mut ft_b = NamedForwardTape::new();
    let xb_h = ft_b.declare_dual("x", 3.0);
    let scope_b = ft_b.freeze_dual();
    let xa = scope_a.dual(xa_h);
    let xb = scope_b.dual(xb_h);
    let _ = xa + xb;
}

#[test]
fn test_scalar_on_lhs_ops() {
    // Exercise every f64 op NamedDual (owned and ref variants).
    let mut ft = NamedForwardTape::new();
    let x_h = ft.declare_dual("x", 2.0);
    let y_h = ft.declare_dual("y", 3.0);
    let scope = ft.freeze_dual();
    let x = scope.dual(x_h);
    let y = scope.dual(y_h);

    // f64 + NamedDual, f64 - NamedDual, f64 * NamedDual, f64 / NamedDual
    let a = 1.0 + x.clone();
    assert_eq!(a.real(), 3.0);
    assert_eq!(a.partial("x"), 1.0);

    let b = 10.0 - y;
    assert_eq!(b.real(), 7.0);
    assert_eq!(b.partial("y"), -1.0);

    let c = 3.0 * x.clone();
    assert_eq!(c.real(), 6.0);
    assert_eq!(c.partial("x"), 3.0);

    let d = 12.0 / y;
    assert_eq!(d.real(), 4.0);
    // d(12/y)/dy = -12/y^2 = -12/9
    assert_eq!(d.partial("y"), -12.0 / 9.0);
}
