//! The two tape misuse cases that used to reach undefined behaviour from
//! safe code: moving an active tape, and reusing an `AReal` from an earlier
//! recording. Both are now either sound or a clear panic.

use xad_rs::{AReal, Tape, compute_gradient_rev_with};

struct Pricer {
    tape: Tape<f64>,
}

impl Pricer {
    /// Activates on the stack, then moves the handle into the struct — the
    /// pattern that used to leave the thread-local pointing at a dead frame.
    fn new() -> Self {
        let mut tape = Tape::<f64>::new(true);
        tape.activate();
        Pricer { tape }
    }
}

#[test]
fn an_active_tape_survives_being_moved() {
    let mut p = Pricer::new();
    assert!(p.tape.is_active(), "activation must follow the handle");

    let mut boxed = vec![std::mem::replace(&mut p.tape, Tape::with_capacity(0, 0))];
    let tape = &mut boxed[0];
    assert!(tape.is_active());

    let x = AReal::input(3.0, tape);
    let mut f = &x * &x;
    f.register(tape);
    f.set_adjoint(tape, 1.0);
    tape.compute_adjoints();
    assert_eq!(x.adjoint(tape), 6.0);
    tape.deactivate();
}

#[test]
#[should_panic(expected = "not on this recording")]
fn a_stale_areal_from_an_earlier_recording_panics_instead_of_writing_out_of_bounds() {
    let mut tape = Tape::<f64>::new(true);
    let stale = {
        let _rec = tape.record();
        // Many inputs so the stale slot lies well past the next recording.
        let _pad: Vec<AReal<f64>> = (0..1000)
            .map(|i| AReal::input(i as f64, &mut tape))
            .collect();
        AReal::input(2.0, &mut tape)
    };
    let _ = compute_gradient_rev_with(&mut tape, &[3.0], |a| a[0] * stale);
}
