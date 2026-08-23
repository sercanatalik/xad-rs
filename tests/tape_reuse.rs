//! Tests for tape reuse: `Tape::with_capacity` and the `Tape::record`
//! scoped-reuse guard.

use xad_rs::{AReal, Real, Tape};

/// f(x, y) = x²·y + sin(x); returns (df/dx, df/dy).
fn grad(tape: &mut Tape<f64>, x0: f64, y0: f64) -> (f64, f64) {
    let mut x = AReal::new(x0);
    let mut y = AReal::new(y0);
    AReal::register_input(std::slice::from_mut(&mut x), tape);
    AReal::register_input(std::slice::from_mut(&mut y), tape);

    let xx = &x * &x;
    let xxy = &xx * &y;
    let sinx = x.sin();
    let mut f = &xxy + &sinx;

    AReal::register_output(std::slice::from_mut(&mut f), tape);
    f.set_adjoint(tape, 1.0);
    tape.compute_adjoints();
    (x.adjoint(tape), y.adjoint(tape))
}

#[test]
fn reused_tape_matches_fresh_tape() {
    let points = [(1.0_f64, 2.0), (0.5, -1.3), (3.0, 0.7), (2.2, 2.2), (-0.4, 1.1)];

    // Fresh tape per valuation.
    let fresh: Vec<(f64, f64)> = points
        .iter()
        .map(|&(x0, y0)| {
            let mut tape = Tape::<f64>::new(true);
            tape.activate();
            let g = grad(&mut tape, x0, y0);
            Tape::<f64>::deactivate_all();
            g
        })
        .collect();

    // One reused tape via `record()`.
    let mut tape = Tape::<f64>::with_capacity(16, 64);
    let reused: Vec<(f64, f64)> = points
        .iter()
        .map(|&(x0, y0)| {
            let _rec = tape.record();
            grad(&mut tape, x0, y0)
        })
        .collect();

    // Bit-identical: reuse must not perturb the computation.
    assert_eq!(fresh, reused);
    // Spot-check correctness: df/dy = x² at the first point (1,2) → 1.0.
    assert!((reused[0].1 - 1.0).abs() < 1e-12);
}

#[test]
fn record_guard_deactivates_on_drop() {
    let mut tape = Tape::<f64>::new(true);
    {
        let _rec = tape.record();
        assert!(tape.is_active());
    } // guard drops here
    assert!(!tape.is_active());

    // A second tape can now activate — proving the thread-local slot is free.
    let mut other = Tape::<f64>::new(true);
    other.activate(); // would panic if the first were still active
    Tape::<f64>::deactivate_all();
}

#[test]
fn record_guard_deactivates_on_panic() {
    let mut tape = Tape::<f64>::new(true);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _rec = tape.record();
        panic!("boom inside recording");
    }));
    assert!(r.is_err());
    // The guard's Drop ran during unwinding → this tape is no longer active.
    assert!(!tape.is_active());
}

#[test]
fn with_capacity_stays_within_reserve() {
    let mut tape = Tape::<f64>::with_capacity(32, 128);
    let _rec = tape.record();
    let (dx, _dy) = grad(&mut tape, 1.5, 2.0);
    // Small recording stays well within the reserved 128 operations.
    assert!(tape.num_operations() <= 128);
    // df/dx = 2xy + cos(x) at (1.5, 2.0).
    let expected = 2.0 * 1.5 * 2.0 + 1.5_f64.cos();
    assert!((dx - expected).abs() < 1e-12);
}
