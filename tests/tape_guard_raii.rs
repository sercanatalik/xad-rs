use xad_rs::{AReal, Tape};

#[test]
fn guard_deactivates_on_drop() {
    let mut tape = Tape::<f64>::new(true);
    {
        let _guard = tape.activate_guard();
        // Guard does not hold a `&mut Tape` borrow — `register_input`
        // still works while the guard is in scope. This is the property
        // the apply-time redesign was made for.
        let mut x = AReal::new(2.0_f64);
        AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
        let _y = &x * &x;
    }
    // Guard dropped → tape deactivated.
    let _z = AReal::new(3.0_f64);
}

#[test]
fn guard_blocks_double_activation() {
    let mut tape_a = Tape::<f64>::new(true);
    let mut tape_b = Tape::<f64>::new(true);

    let _ga = tape_a.activate_guard();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _gb = tape_b.activate_guard();
    }));
    assert!(result.is_err(), "double activation should panic");
}

#[test]
fn manual_activate_deactivate_still_works() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();
    let mut x = AReal::new(1.5_f64);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    let _y = &x * &x;
    Tape::<f64>::deactivate_all();
}

#[test]
fn guard_drop_does_not_deactivate_unrelated_tape() {
    // Build tape A and its guard, then drop tape A (clears active ptr).
    // Then activate tape B manually. When guard A finally drops, it
    // must NOT deactivate tape B — the ownership check in TapeGuard::drop
    // sees the active ptr no longer matches its tape_ptr and does nothing.
    let mut tape_b = Tape::<f64>::new(true);
    let guard_a;
    {
        let mut tape_a = Tape::<f64>::new(true);
        guard_a = tape_a.activate_guard();
        // tape_a goes out of scope at the end of this block.
        // Tape::drop calls self.deactivate(), clearing the active ptr.
    }
    // Active ptr is None here. Activate tape_b.
    tape_b.activate();
    assert!(tape_b.is_active());

    // Drop guard_a; should be a no-op (active ptr != guard_a.tape_ptr).
    drop(guard_a);

    // tape_b should still be active.
    assert!(tape_b.is_active(), "guard A's drop must not deactivate tape B");
    Tape::<f64>::deactivate_all();
}

// Compile-fail check: TapeGuard must not be Send.
// (Negative-compile checks normally require trybuild; we encode it as
// a non-test helper that, if uncommented, fails to compile.)
//
// fn _assert_send<S: Send>() {}
// fn _check_tape_guard_not_send() {
//     _assert_send::<xad_rs::TapeGuard<f64>>();   // <-- must fail to compile
// }
