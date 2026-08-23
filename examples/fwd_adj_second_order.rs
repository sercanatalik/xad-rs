//! Second-order forward-over-adjoint - gradient and one Hessian row in one sweep.
//!
//! Computes
//!     y          = f(x0, x1, x2, x3)
//!     dy/dx_i    = first-order partials (gradient)
//!     d²y/dx0dx_j = second-order partials w.r.t. x0 (one Hessian row)
//!
//! The classic C++ formulation achieves this in a single tape sweep with a
//! `fwd_adj<double>` active type: it is a forward-mode tangent layered
//! on top of an adjoint, so seeding the forward direction `derivative(value(x0)) = 1`
//! and the adjoint `value(derivative(y)) = 1` then calling `computeAdjoints`
//! produces the gradient and one Hessian row in one go.
//!
//! `xad-rs` expresses the same construction by instantiating the reverse-mode
//! tape over a forward-dual *storage* scalar: `Tape<Jet1<f64>>` recording
//! `AReal<Jet1<f64>>`. Seed input x0's tangent to 1, record, and sweep once —
//! then each input's adjoint is itself a dual whose **value** is `dy/dx_j` and
//! whose **tangent** is `d²y/dx0dx_j`. That is exactly `fwd_adj`, and it is
//! what [`xad_rs::compute_hessian`] runs once per input direction to build a
//! full Hessian. This sample does both: the raw seed-and-sweep (matching the
//! upstream output), then the driver, cross-checked against each other and
//! against the analytic answer.
//!
//! Function under test (same as `adjoint_first_order.rs`):
//!     a = sin(x0) * cos(x1)
//!     b = x2 * x3 - tan(x1 - x2)
//!     c = a + 2 * b
//!     y = c * c
//!
//! Analytic second-order partials w.r.t. x0 (sec² = 1 + tan²):
//!     dc/dx0      =  cos(x0) * cos(x1)
//!     d²c/dx0dx0  = -sin(x0) * cos(x1)
//!     d²c/dx0dx1  = -cos(x0) * sin(x1)
//!     d²c/dx0dx2  =  0
//!     d²c/dx0dx3  =  0
//!     d²y/dx0dxj  =  2 * (dc/dx0) * (dc/dxj) + 2c * d²c/dx0dxj

use xad_rs::{compute_hessian, math, AReal, Jet1, Tape, TapeStorage};

/// y = (sin(x0)·cos(x1) + 2·(x2·x3 − tan(x1 − x2)))²
///
/// Generic over the tape storage scalar, so the identical body records on a
/// plain `Tape<f64>` (first order) and on `Tape<Jet1<f64>>` (second order).
/// `2·b` is written `b + b` to keep the body free of scalar conversions.
fn f<T: TapeStorage>(v: &[AReal<T>]) -> AReal<T> {
    let a = &math::ad::sin(&v[0]) * &math::ad::cos(&v[1]);
    let b = &(&v[2] * &v[3]) - &math::ad::tan(&(&v[1] - &v[2]));
    let c = &a + &(&b + &b);
    &c * &c
}

fn main() {
    let xs = [1.0_f64, 1.5, 1.3, 1.2];
    let [x0v, x1v, x2v, x3v] = xs;

    // -------- Forward-over-adjoint: one recording, one sweep --------
    // The tape's storage scalar is a forward dual. Seeding x0's tangent to 1
    // makes every recorded multiplier carry its own derivative w.r.t. x0, so
    // the single reverse sweep propagates first AND second order together.
    let (value, grad, hess_row0) = {
        let mut tape = Tape::<Jet1<f64>>::new(true);
        let _rec = tape.record();

        let mut inputs: Vec<AReal<Jet1<f64>>> = xs
            .iter()
            .enumerate()
            .map(|(i, &v)| AReal::new(Jet1::new(v, if i == 0 { 1.0 } else { 0.0 })))
            .collect();
        AReal::register_input(&mut inputs, &mut tape);

        let mut y = f(&inputs);
        AReal::register_output(std::slice::from_mut(&mut y), &mut tape);
        y.set_adjoint(&mut tape, Jet1::constant(1.0));
        tape.compute_adjoints();

        // Input j's adjoint is a dual: value = dy/dx_j, tangent = d²y/dx0dx_j.
        let grad: Vec<f64> = inputs.iter().map(|x| x.adjoint(&tape).value()).collect();
        let row0: Vec<f64> = inputs
            .iter()
            .map(|x| x.adjoint(&tape).derivative())
            .collect();
        (y.value().value(), grad, row0)
    }; // guard drops here — the driver below activates its own tape

    // -------- The same engine, driven for the full n × n Hessian --------
    let hess = compute_hessian(&xs, f);

    // -------- Output --------
    println!("Second-order forward-over-adjoint - gradient and one Hessian row");
    println!("===========================================================");
    println!("Inputs: x0={x0v}, x1={x1v}, x2={x2v}, x3={x3v}");
    println!();
    println!("y = {value:.10}");
    println!();
    println!("First-order derivatives (value part of each input's adjoint):");
    for (i, g) in grad.iter().enumerate() {
        println!("  dy/dx{i} = {g:>14.10}");
    }
    println!();
    println!("Second-order derivatives w.r.t. x0 (tangent part of the same adjoints):");
    for (j, h) in hess_row0.iter().enumerate() {
        println!("  d2y/dx0dx{j} = {h:>14.10}");
    }

    // -------- Analytic cross-check --------
    let av = x0v.sin() * x1v.cos();
    let t = (x1v - x2v).tan();
    let sec2 = 1.0 + t * t;
    let bv = x2v * x3v - t;
    let cv = av + 2.0 * bv;

    // First derivatives of c
    let dc_dx0 = x0v.cos() * x1v.cos();
    let dc_dx1 = -x0v.sin() * x1v.sin() - 2.0 * sec2;
    let dc_dx2 = 2.0 * (x3v + sec2);
    let dc_dx3 = 2.0 * x2v;

    // Second derivatives of c w.r.t. x0
    let d2c_dx0dx0 = -x0v.sin() * x1v.cos();
    let d2c_dx0dx1 = -x0v.cos() * x1v.sin();
    let d2c_dx0dx2 = 0.0;
    let d2c_dx0dx3 = 0.0;

    // Apply chain rule: y = c², so d²y/dxidxj = 2·dc/dxi·dc/dxj + 2c·d²c/dxidxj
    let expected_grad = [
        2.0 * cv * dc_dx0,
        2.0 * cv * dc_dx1,
        2.0 * cv * dc_dx2,
        2.0 * cv * dc_dx3,
    ];
    let expected_hess_row0 = [
        2.0 * dc_dx0 * dc_dx0 + 2.0 * cv * d2c_dx0dx0,
        2.0 * dc_dx0 * dc_dx1 + 2.0 * cv * d2c_dx0dx1,
        2.0 * dc_dx0 * dc_dx2 + 2.0 * cv * d2c_dx0dx2,
        2.0 * dc_dx0 * dc_dx3 + 2.0 * cv * d2c_dx0dx3,
    ];

    println!();
    println!("Analytic gradient (reference):");
    for (i, e) in expected_grad.iter().enumerate() {
        println!("  dy/dx{i} = {e:>14.10}");
    }
    println!();
    println!("Analytic Hessian row 0 (reference):");
    for (j, e) in expected_hess_row0.iter().enumerate() {
        println!("  d2y/dx0dx{j} = {e:>14.10}");
    }

    let max_grad_err = (0..4)
        .map(|i| (grad[i] - expected_grad[i]).abs())
        .fold(0.0_f64, f64::max);
    let max_hess_err = (0..4)
        .map(|j| (hess_row0[j] - expected_hess_row0[j]).abs())
        .fold(0.0_f64, f64::max);

    // The driver runs the identical per-column computation, so row 0 of its
    // Hessian must match the hand-rolled sweep bit-for-bit, not just closely.
    let driver_row0_matches = (0..4).all(|j| hess[[j, 0]] == hess_row0[j]);

    println!();
    println!("Max |gradient - analytic|         = {max_grad_err:.2e}");
    println!("Max |Hessian row 0 - analytic|    = {max_hess_err:.2e}");
    println!("compute_hessian row 0 bit-identical to the raw sweep: {driver_row0_matches}");
    println!();
    println!("Full 4x4 Hessian (compute_hessian, {} passes):", xs.len());
    for i in 0..4 {
        let row: Vec<String> = (0..4).map(|j| format!("{:>12.6}", hess[[i, j]])).collect();
        println!("  {}", row.join(" "));
    }

    assert!(max_grad_err < 1e-12, "gradient mismatch: {max_grad_err}");
    assert!(max_hess_err < 1e-12, "Hessian-row mismatch: {max_hess_err}");
    assert!(driver_row0_matches, "driver disagrees with the raw sweep");
}
