//! Tests for `NamedForwardTape` — the Shape A forward-mode scope.
//!
//! Phase 02.2 Plan 02.2-01. Covers:
//!
//! - Basic input / freeze / compute / read-back smoke cycle.
//! - TLS save-restore discipline: a nested `NamedForwardTape` must not
//!   corrupt the outer tape's generation when it drops.
//! - Cross-generation debug guard: values from two live tapes panic when
//!   mixed in a binary op in debug builds.


use xad_rs::{NamedFReal, NamedForwardTape};

/// Smoke test: basic input / freeze / compute / read-back cycle.
#[test]
fn forward_tape_basic_smoke() {
    let mut ft = NamedForwardTape::new();
    let x: NamedFReal<f64> = ft.input_freal("x", 3.0);
    let _registry = ft.freeze();
    let f = &x * &x + &x; // 9 + 3 = 12, f'(x) = 2*3 + 1 = 7
    assert_eq!(f.value(), 12.0);
    assert_eq!(f.derivative("x"), 7.0);
}

/// TLS save-restore: a nested `NamedForwardTape` must not corrupt the
/// outer tape's generation when it drops.
#[test]
fn forward_tape_nested_scopes_save_restore() {
    let mut outer = NamedForwardTape::new();
    let x_outer: NamedFReal<f64> = outer.input_freal("x", 1.0);
    let _r_outer = outer.freeze();
    {
        let mut inner = NamedForwardTape::new();
        let _y: NamedFReal<f64> = inner.input_freal("y", 2.0);
        let _r_inner = inner.freeze();
        // inner drops here — prev_gen restored to outer's generation
    }
    // Outer must still compute correctly after inner's Drop.
    let f = &x_outer * 2.0_f64 + &x_outer; // 2*1 + 1 = 3
    assert_eq!(f.value(), 3.0);
    assert_eq!(f.derivative("x"), 3.0); // d/dx (3x) = 3
}

/// Cross-generation panic in debug builds: taking a value from tape A
/// and operating on it with a value from tape B (after tape A dropped)
/// must panic under `#[cfg(debug_assertions)]`.
///
/// This test runs only in debug builds.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cross-registry forward-mode op detected")]
fn forward_tape_cross_generation_panics_in_debug() {
    let mut tape_a = NamedForwardTape::new();
    let a: NamedFReal<f64> = tape_a.input_freal("a", 1.0);
    let _r_a = tape_a.freeze();
    let mut tape_b = NamedForwardTape::new();
    let b: NamedFReal<f64> = tape_b.input_freal("b", 2.0);
    let _r_b = tape_b.freeze();
    // Both tapes alive simultaneously — nested. `a` was stamped with
    // tape_a's generation, `b` with tape_b's. The binary op panics.
    let _ = &a + &b;
}

/// Repeated input registrations for the same name are idempotent — first
/// insertion wins and the resulting registry contains one slot.
#[test]
fn forward_tape_repeated_input_is_idempotent() {
    let mut ft = NamedForwardTape::new();
    let _x1: NamedFReal<f64> = ft.input_freal("x", 1.0);
    let _x2: NamedFReal<f64> = ft.input_freal("x", 2.0);
    let registry = ft.freeze();
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.index_of("x"), Some(0));
}
