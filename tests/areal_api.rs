//! Ergonomic `AReal` API: `Copy` semantics, `input`/`register`, and
//! `From<i32>` symmetry.

use xad_rs::{AReal, Tape};

#[test]
fn input_and_register_round_trip() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();

    let x = AReal::input(3.0, &mut tape);
    let y = AReal::input(4.0, &mut tape);

    let mut f = &(&x * &x) * &y; // x^2 * y
    f.register(&mut tape);
    f.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    assert_eq!(x.adjoint(&tape), 2.0 * 3.0 * 4.0); // 2xy
    assert_eq!(y.adjoint(&tape), 9.0); // x^2
}

#[test]
fn copies_share_slot_and_adjoint() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();

    let x = AReal::input(3.0, &mut tape);
    let y = x; // Copy: same slot, same recorded variable

    assert_eq!(x.slot(), y.slot());

    let mut f = &x * &y; // x * x = x^2
    f.register(&mut tape);
    f.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    // Both operands feed the same slot, so the adjoint accumulates to 2x.
    assert_eq!(x.adjoint(&tape), 6.0);
    assert_eq!(y.adjoint(&tape), 6.0);
}

#[test]
fn register_is_idempotent() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();

    let mut x = AReal::new(1.5);
    x.register(&mut tape);
    let slot = x.slot();
    x.register(&mut tape); // second call must not re-slot
    assert_eq!(x.slot(), slot);
    assert_eq!(tape.num_variables(), 1);
}

#[test]
fn register_gives_constant_output_a_slot() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();

    // A constant that never touched the tape still gets an adjoint slot.
    let mut c = AReal::new(42.0);
    assert!(!c.should_record());
    c.register(&mut tape);
    assert!(c.should_record());
    c.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert_eq!(c.adjoint(&tape), 1.0);
}

#[test]
fn from_i32_for_f64() {
    let a: AReal<f64> = 7.into();
    assert_eq!(a.value(), 7.0_f64);
}

#[test]
fn passive_only_ops_record_nothing() {
    // Ops whose operands are all unregistered constants must not touch the
    // tape: no slot, no dead nullary statement.
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();

    let a = AReal::new(2.0);
    let b = AReal::new(3.0);

    let p = &a * &b; // passive * passive
    let s = &a + 1.5; // passive + scalar
    let n = -a; // neg(passive)
    assert_eq!(p.value(), 6.0);
    assert_eq!(s.value(), 3.5);
    assert_eq!(n.value(), -2.0);
    assert!(!p.should_record());
    assert!(!s.should_record());
    assert!(!n.should_record());
    assert_eq!(tape.num_variables(), 0);
    assert_eq!(tape.num_statements(), 0);
    assert_eq!(tape.num_operations(), 0);

    // The passive result still behaves as a correct constant downstream.
    let x = AReal::input(4.0, &mut tape);
    let mut f = &x * &p; // f = 6x
    f.register(&mut tape);
    f.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert_eq!(x.adjoint(&tape), 6.0);
}
