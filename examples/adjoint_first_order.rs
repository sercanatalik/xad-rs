//! First-order adjoint mode - the full gradient in one reverse sweep.
//!
//! Computes
//!     y = f(x0, x1, x2, x3)
//! and its first-order partial derivatives
//!     dy/dx0, dy/dx1, dy/dx2, dy/dx3
//! in a single reverse-mode (adjoint) sweep.
//!
//! With many inputs and a single scalar output, reverse mode delivers the
//! full gradient at the cost of one forward evaluation plus one backward
//! pass — independent of the number of inputs.
//!
//! Function under test:
//!     a = sin(x0) * cos(x1)
//!     b = x2 * x3 - tan(x1 - x2)
//!     c = a + 2 * b
//!     y = c * c
//!
//! Analytic gradient (chain rule, sec^2 = 1 + tan^2):
//!     dc/dx0 =  cos(x0) * cos(x1)
//!     dc/dx1 = -sin(x0) * sin(x1) - 2 * (1 + tan^2(x1 - x2))
//!     dc/dx2 =  2 * (x3 + (1 + tan^2(x1 - x2)))
//!     dc/dx3 =  2 * x2
//!     dy/dxi =  2 * c * dc/dxi

use xad_rs::{AReal, Tape, math};

fn main() {
    // Inputs (matching the upstream sample).
    let x0v = 1.0_f64;
    let x1v = 1.5_f64;
    let x2v = 1.3_f64;
    let x3v = 1.2_f64;

    // Tape for first-order adjoint computation.
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    // Independent variables.
    let mut x0 = AReal::new(x0v);
    let mut x1 = AReal::new(x1v);
    let mut x2 = AReal::new(x2v);
    let mut x3 = AReal::new(x3v);

    // Register inputs. (In `xad-rs`, `Tape::new_recording` resets the
    // tape and would invalidate the slots we are about to hand out, so the
    // recording boundary is implicitly the moment the tape is created — there
    // is no separate `newRecording()` step here as in the C++ original.)
    AReal::register_input(std::slice::from_mut(&mut x0), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut x1), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut x2), &mut tape);
    AReal::register_input(std::slice::from_mut(&mut x3), &mut tape);

    // y = f(x0, x1, x2, x3)
    //   a = sin(x0) * cos(x1)
    //   b = x2 * x3 - tan(x1 - x2)
    //   c = a + 2 * b
    //   y = c * c
    let a = math::ad::sin(&x0) * math::ad::cos(&x1);
    let b = &x2 * &x3 - math::ad::tan(&(&x1 - &x2));
    let c = &a + 2.0 * &b;
    let mut y = &c * &c;

    // Register the output, seed its adjoint, and roll back the tape.
    AReal::register_output(std::slice::from_mut(&mut y), &mut tape);
    y.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    // -------- Output --------
    println!("First-order adjoint mode - full gradient in one reverse sweep");
    println!("==================================================");
    println!("Inputs: x0={x0v}, x1={x1v}, x2={x2v}, x3={x3v}");
    println!();
    println!("y = {:.10}", y.value());
    println!();
    println!("First-order derivatives (adjoint mode):");
    println!("  dy/dx0 = {:>14.10}", x0.adjoint(&tape));
    println!("  dy/dx1 = {:>14.10}", x1.adjoint(&tape));
    println!("  dy/dx2 = {:>14.10}", x2.adjoint(&tape));
    println!("  dy/dx3 = {:>14.10}", x3.adjoint(&tape));

    // -------- Analytic cross-check --------
    let av = x0v.sin() * x1v.cos();
    let t = (x1v - x2v).tan();
    let sec2 = 1.0 + t * t;
    let bv = x2v * x3v - t;
    let cv = av + 2.0 * bv;

    let dc_dx0 = x0v.cos() * x1v.cos();
    let dc_dx1 = -x0v.sin() * x1v.sin() - 2.0 * sec2;
    let dc_dx2 = 2.0 * (x3v + sec2);
    let dc_dx3 = 2.0 * x2v;

    let expected = [
        2.0 * cv * dc_dx0,
        2.0 * cv * dc_dx1,
        2.0 * cv * dc_dx2,
        2.0 * cv * dc_dx3,
    ];
    let computed = [
        x0.adjoint(&tape),
        x1.adjoint(&tape),
        x2.adjoint(&tape),
        x3.adjoint(&tape),
    ];

    println!();
    println!("Analytic gradient (reference):");
    println!("  dy/dx0 = {:>14.10}", expected[0]);
    println!("  dy/dx1 = {:>14.10}", expected[1]);
    println!("  dy/dx2 = {:>14.10}", expected[2]);
    println!("  dy/dx3 = {:>14.10}", expected[3]);

    let max_err = computed
        .iter()
        .zip(expected.iter())
        .map(|(c, e)| (c - e).abs())
        .fold(0.0_f64, f64::max);

    println!();
    println!("Max |AD - analytic| = {max_err:.2e}");
    assert!(max_err < 1e-12, "gradient mismatch vs. analytic: {max_err}");
}
