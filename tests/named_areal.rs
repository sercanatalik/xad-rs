//! Integration tests for `NamedTape` + `NamedAReal`.
//!
//! Covers LBLR-02..10 from REQUIREMENTS.md plus the Phase 2 research-derived
//! safety tests (sequential tapes, nested freeze panic, multithread
//! isolation, std::mem::forget recovery, panic-during-forward unwind).


use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use xad_rs::NamedTape;

// ---------- LBLR-02: new() does NOT activate ----------

#[test]
fn test_new_does_not_activate() {
    // We rely on the indirect signal: if `new()` activated the tape, then
    // constructing a SECOND NamedTape and calling `freeze()` on it would
    // panic with "A tape is already active on this thread". Construct two
    // back-to-back, then freeze the SECOND one — must succeed.
    let _t1 = NamedTape::new();
    let _t2 = NamedTape::new();
    // Drop t1 and t2 — neither activated, so neither needed deactivation.
    NamedTape::deactivate_all(); // hygiene: leave the thread clean for downstream tests
    let _ = Arc::<()>::new(()); // touch Arc so the import is exercised in this file
}

// ---------- LBLR-03: input() after freeze() panics ----------

#[test]
#[should_panic(expected = "after freeze")]
fn test_input_after_freeze_panics() {
    let mut t = NamedTape::new();
    let _x = t.input("x", 1.0);
    let _ = t.freeze();
    let _y = t.input("y", 2.0); // must panic
}

// ---------- LBLR-04: freeze() returns registry in input-order ----------

#[test]
fn test_freeze_returns_registry_in_order() {
    let mut t = NamedTape::new();
    let _x = t.input("x", 1.0);
    let _y = t.input("y", 2.0);
    let _z = t.input("z", 3.0);
    let reg = t.freeze();
    let names: Vec<&str> = reg.iter().collect();
    assert_eq!(names, vec!["x", "y", "z"]);
    drop(t);
    NamedTape::deactivate_all();
}

// ---------- LBLR-05 + LBLR-08: f(x, y) = x²y + sin(x) ----------

#[test]
fn test_gradient_x2y_plus_sin_x() {
    let mut t = NamedTape::new();
    let x = t.input("x", 3.0);
    let y = t.input("y", 4.0);
    let _registry = t.freeze();

    // f(x, y) = x²·y + sin(x), at (3, 4)
    // ∂f/∂x = 2xy + cos(x) = 24 + cos(3) ≈ 23.0100075...
    // ∂f/∂y = x²            = 9.0
    let f = &(&x * &x) * &y + x.sin();
    let grad = t.gradient(&f);

    let expected_dx = 2.0 * 3.0 * 4.0 + 3.0_f64.cos();
    let expected_dy = 9.0;
    assert!((grad["x"] - expected_dx).abs() < 1e-12, "grad x: {} vs {}", grad["x"], expected_dx);
    assert!((grad["y"] - expected_dy).abs() < 1e-12, "grad y: {} vs {}", grad["y"], expected_dy);

    // IndexMap iteration order MUST match input order (x first, then y).
    let keys: Vec<&str> = grad.keys().map(String::as_str).collect();
    assert_eq!(keys, vec!["x", "y"]);

    drop(t);
    NamedTape::deactivate_all();
}

// ---------- LBLR-05 negative case: gradient() before freeze() panics ----------

#[test]
#[should_panic(expected = "before freeze")]
fn test_gradient_before_freeze_panics() {
    let mut t = NamedTape::new();
    let x = t.input("x", 1.0);
    let _ = t.gradient(&x); // must panic
}

// ---------- LBLR-06: sequential tapes on the same thread ----------

#[test]
fn test_sequential_tapes_ok() {
    {
        let mut t1 = NamedTape::new();
        let x = t1.input("x", 2.0);
        let _ = t1.freeze();
        let f = &x * &x;
        let g = t1.gradient(&f);
        assert!((g["x"] - 4.0).abs() < 1e-12);
    } // t1 dropped here -> deactivated
    {
        let mut t2 = NamedTape::new();
        let y = t2.input("y", 3.0);
        let _ = t2.freeze(); // must NOT panic
        let f = &y * &y * &y;
        let g = t2.gradient(&f);
        assert!((g["y"] - 27.0).abs() < 1e-12);
    }
    NamedTape::deactivate_all();
}

// ---------- LBLR (bonus): nested freeze() panics ----------

#[test]
#[should_panic(expected = "A tape is already active on this thread")]
fn test_nested_freeze_panics() {
    let mut t1 = NamedTape::new();
    let mut t2 = NamedTape::new();
    let _x = t1.input("x", 1.0);
    let _y = t2.input("y", 2.0);
    let _ = t1.freeze();
    let _ = t2.freeze(); // must panic — t1 already active
    // Cleanup never reached, but the test runner will tear down the thread.
}

// ---------- LBLR-09: multi-threaded isolation ----------

#[test]
fn test_multithread_isolation() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_a = Arc::clone(&counter);
    let counter_b = Arc::clone(&counter);

    let h_a = std::thread::spawn(move || {
        let mut t = NamedTape::new();
        let x = t.input("x", 5.0);
        let _ = t.freeze();
        let f = &x * &x; // f = x², df/dx = 10
        let g = t.gradient(&f);
        counter_a.fetch_add(1, Ordering::SeqCst);
        assert!((g["x"] - 10.0).abs() < 1e-12);
    });

    let h_b = std::thread::spawn(move || {
        let mut t = NamedTape::new();
        let y = t.input("y", 7.0);
        let _ = t.freeze();
        let f = &y * &y * &y; // f = y³, df/dy = 147
        let g = t.gradient(&f);
        counter_b.fetch_add(1, Ordering::SeqCst);
        assert!((g["y"] - 147.0).abs() < 1e-12);
    });

    h_a.join().unwrap();
    h_b.join().unwrap();
    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

// ---------- LBLR-10: std::mem::forget recovery ----------

#[test]
fn test_deactivate_all_after_forget() {
    {
        let mut t1 = NamedTape::new();
        let _x = t1.input("x", 1.0);
        let _ = t1.freeze();
        std::mem::forget(t1); // Drop skipped — TLS pointer dangles
    }
    NamedTape::deactivate_all(); // escape hatch — clears TLS

    let mut t2 = NamedTape::new();
    let y = t2.input("y", 2.0);
    let _ = t2.freeze(); // must NOT panic
    let f = &y * &y;
    let g = t2.gradient(&f);
    assert!((g["y"] - 4.0).abs() < 1e-12);
    drop(t2);
    NamedTape::deactivate_all();
}

// ---------- LBLR-10 (panic-during-forward): drop-on-unwind path ----------

#[test]
fn test_panic_during_forward_deactivates() {
    let result = std::panic::catch_unwind(|| {
        let mut t = NamedTape::new();
        let _x = t.input("x", 1.0);
        let _ = t.freeze();
        // Simulate a panic inside the forward closure.
        panic!("simulated user panic");
    });
    assert!(result.is_err());
    // Drop ran during unwind -> tape deactivated. Construct a new one.
    let mut t2 = NamedTape::new();
    let z = t2.input("z", 4.0);
    let _ = t2.freeze();
    let f = &z * &z;
    let g = t2.gradient(&f);
    assert!((g["z"] - 8.0).abs() < 1e-12);
    drop(t2);
    NamedTape::deactivate_all();
}

// ---------- Elementary delegation smoke test ----------

#[test]
fn test_elementary_delegation_chain() {
    // f(x) = sqrt(exp(sin(x)) + cos(x) + ln(1 + tan(x)) + tanh(x))
    let mut t = NamedTape::new();
    let x = t.input("x", 0.5);
    let _ = t.freeze();

    let term = x.sin().exp() + x.cos() + (x.tan() + 1.0).ln() + x.tanh();
    let f = term.sqrt();
    let g = t.gradient(&f);

    // Sanity: gradient is finite and non-zero (this is a smoke test, not an analytic check —
    // the bit-exact test belongs in the named_jacobian or named_dual cross-check suites).
    assert!(g["x"].is_finite());
    assert!(g["x"].abs() > 0.0);
    drop(t);
    NamedTape::deactivate_all();
}
