//! First-order drivers: `compute_derivative_fwd`,
//! `compute_directional_derivative_fwd`, `compute_gradient_fwd_k`,
//! `compute_gradient_rev`.
//!
//! Each is checked against a closed form, and the three `Rⁿ → R` drivers are
//! cross-checked against each other and against the existing Jacobian
//! driver — the gradient must be that driver's single row.

use xad_rs::{
    AReal, Real, Tape, compute_derivative_fwd, compute_directional_derivative_fwd,
    compute_gradient_fwd_k, compute_gradient_rev, compute_gradient_rev_with,
    compute_jacobian_rev, compute_jacobian_rev_with,
};

/// `f(x) = x³·ln(x)`; `f'(x) = 3x²·ln(x) + x²`.
fn scalar<R: Real>(x: &R) -> R {
    x.powi(3) * x.ln()
}

/// `f(x, y, z) = x²·y + sin(z)·y + exp(x·z)`, written once against the trait
/// so both the forward and the reverse driver evaluate the same body.
fn multi<R: Real>(v: &[R]) -> R {
    let (x, y, z) = (v[0].clone(), v[1].clone(), v[2].clone());
    x.clone() * x.clone() * y.clone() + z.sin() * y + (x * z).exp()
}

const POINT: [f64; 3] = [0.7, 1.3, -0.4];

/// Closed-form gradient of `multi` at `POINT`.
fn multi_grad(p: &[f64]) -> [f64; 3] {
    let (x, y, z) = (p[0], p[1], p[2]);
    [
        2.0 * x * y + z * (x * z).exp(),
        x * x + z.sin(),
        y * z.cos() + x * (x * z).exp(),
    ]
}

#[test]
fn scalar_derivative_matches_the_closed_form() {
    for &x0 in &[0.3_f64, 1.0, 2.5, 7.25] {
        let (v, d) = compute_derivative_fwd(x0, scalar);
        let want_v = x0.powi(3) * x0.ln();
        let want_d = 3.0 * x0 * x0 * x0.ln() + x0 * x0;
        assert_eq!(v, want_v, "value at x={x0}");
        assert!(
            (d - want_d).abs() <= 1e-14 * (1.0 + want_d.abs()),
            "derivative at x={x0}: got {d}, want {want_d}"
        );
    }
}

#[test]
fn scalar_derivative_creates_no_tape() {
    // A tape live on the thread must stay untouched: forward mode records
    // nothing.
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let (_, d) = compute_derivative_fwd(2.0_f64, scalar);
    assert!(d.is_finite());
    assert_eq!(tape.num_statements(), 0);
    assert_eq!(tape.num_operations(), 0);
}

#[test]
fn unit_seeds_recover_the_partial_derivatives() {
    let want = multi_grad(&POINT);
    for i in 0..3 {
        let mut seed = [0.0; 3];
        seed[i] = 1.0;
        let (v, d) = compute_directional_derivative_fwd(&POINT, &seed, multi);
        assert!((v - multi(&POINT)).abs() < 1e-15, "value on direction {i}");
        assert!(
            (d - want[i]).abs() <= 1e-13 * (1.0 + want[i].abs()),
            "∂f/∂x{i}: got {d}, want {}",
            want[i]
        );
    }
}

#[test]
fn a_general_seed_is_the_inner_product_of_gradient_and_seed() {
    let g = multi_grad(&POINT);
    for seed in [[0.5_f64, 2.0, -1.5], [1.0, 1.0, 1.0], [-0.25, 0.0, 3.0]] {
        let (_, d) = compute_directional_derivative_fwd(&POINT, &seed, multi);
        let want: f64 = g.iter().zip(&seed).map(|(a, b)| a * b).sum();
        assert!(
            (d - want).abs() <= 1e-13 * (1.0 + want.abs()),
            "∇f·v: got {d}, want {want}"
        );
    }
}

#[test]
#[should_panic(expected = "seed length must match inputs")]
fn a_mismatched_seed_length_panics() {
    compute_directional_derivative_fwd(&POINT, &[1.0, 0.0], multi);
}

#[test]
fn gradient_matches_the_closed_form_and_the_forward_partials() {
    let (v, g) = compute_gradient_rev(&POINT, multi);
    assert!((v - multi(&POINT)).abs() < 1e-15, "value");

    let want = multi_grad(&POINT);
    for i in 0..3 {
        assert!(
            (g[i] - want[i]).abs() <= 1e-13 * (1.0 + want[i].abs()),
            "closed form ∂f/∂x{i}: got {}, want {}",
            g[i],
            want[i]
        );

        // ... and against the forward driver, one unit direction at a time.
        let mut seed = [0.0; 3];
        seed[i] = 1.0;
        let (_, fwd) = compute_directional_derivative_fwd(&POINT, &seed, multi);
        assert!(
            (g[i] - fwd).abs() <= 1e-13 * (1.0 + fwd.abs()),
            "forward vs reverse ∂f/∂x{i}: {} vs {fwd}",
            g[i]
        );
    }
}

/// `multi` has no division, so each K-lane partial is the `Jet1` partial bit
/// for bit, whatever block it landed in: K < n (two blocks), K = n, K > n.
#[test]
fn k_lane_gradient_is_the_per_direction_forward_gradient_bit_for_bit() {
    let want_value = multi(&POINT);
    let jet1: Vec<f64> = (0..3)
        .map(|i| {
            let mut seed = [0.0; 3];
            seed[i] = 1.0;
            compute_directional_derivative_fwd(&POINT, &seed, multi).1
        })
        .collect();
    let closed = multi_grad(&POINT);

    let (v2, g2) = compute_gradient_fwd_k::<2, _>(&POINT, multi);
    let (v3, g3) = compute_gradient_fwd_k::<3, _>(&POINT, multi);
    let (v8, g8) = compute_gradient_fwd_k::<8, _>(&POINT, multi);
    for (k, (v, g)) in [(2, (v2, &g2)), (3, (v3, &g3)), (8, (v8, &g8))] {
        assert_eq!(v, want_value, "K={k}: value vs f64");
        assert_eq!(g.len(), 3, "K={k}: gradient length");
        assert_eq!(**g, jet1[..], "K={k}: gradient vs Jet1 per direction");
        for i in 0..3 {
            assert!((g[i] - closed[i]).abs() < 1e-13, "K={k}: grad[{i}] {} vs {}", g[i], closed[i]);
        }
    }
}

/// With a quotient in the body the K-lane and `Jet1` tangents accumulate in
/// a different order, so they agree to a few ulp rather than to the bit;
/// against reverse mode the same few-ulp bound applies to every body.
#[test]
fn k_lane_gradient_agrees_with_jet1_and_reverse_on_a_dividing_body() {
    fn dividing<R: Real>(v: &[R]) -> R {
        (v[0].clone() * v[1].clone() + v[2].sin()) / (v[1].clone() + v[2].clone() * v[2].clone())
    }
    let (vk, gk) = compute_gradient_fwd_k::<2, _>(&POINT, dividing);
    let (vr, gr) = compute_gradient_rev(&POINT, dividing);
    assert_eq!(vk, dividing(&POINT), "K-lane value vs f64");
    assert_eq!(vk, vr, "K-lane value vs reverse value");
    for i in 0..3 {
        let mut seed = [0.0; 3];
        seed[i] = 1.0;
        let j1 = compute_directional_derivative_fwd(&POINT, &seed, dividing).1;
        let tol = 8.0 * f64::EPSILON * (1.0 + j1.abs());
        assert!((gk[i] - j1).abs() <= tol, "grad[{i}] K-lane {} vs Jet1 {j1}", gk[i]);
        assert!((gk[i] - gr[i]).abs() <= tol, "grad[{i}] K-lane {} vs reverse {}", gk[i], gr[i]);
    }
}

#[test]
fn k_lane_gradient_records_nothing_on_an_active_tape() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let x = AReal::input(1.0, &mut tape);
    let _touch = x * 2.0; // one statement, so the count below is not trivially zero
    let before = (tape.num_statements(), tape.num_operations());
    let (_, g) = compute_gradient_fwd_k::<4, _>(&POINT, multi);
    assert_eq!(g.len(), 3);
    assert_eq!((tape.num_statements(), tape.num_operations()), before);
}

#[test]
fn k_lane_gradient_of_no_inputs_is_the_value_and_an_empty_gradient() {
    let (v, g) = compute_gradient_fwd_k::<4, _>(&[], |_| xad_rs::JetK::constant(7.5));
    assert_eq!(v, 7.5);
    assert!(g.is_empty());
}

#[test]
fn gradient_equals_the_jacobians_single_row() {
    let (_, g) = compute_gradient_rev(&POINT, multi);
    let jac = compute_jacobian_rev(&POINT, |v: &[AReal<f64>]| vec![multi(v)]);
    assert_eq!(jac.shape(), &[1, 3]);
    assert_eq!(
        g,
        jac.row(0).to_vec(),
        "gradient must be the Jacobian's row"
    );
}

#[test]
fn gradient_leaves_no_tape_active() {
    // The driver's tape must be gone by the time it returns, so a second
    // call — or any other recording — does not hit the no-nesting panic.
    let (_, g1) = compute_gradient_rev(&POINT, multi);
    let (_, g2) = compute_gradient_rev(&POINT, multi);
    assert_eq!(g1, g2);

    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let x = AReal::input(1.0_f64, &mut tape);
    assert_eq!(x.value(), 1.0);
}

#[test]
fn drivers_agree_on_a_body_written_once_against_the_trait() {
    // The same generic body reaches all three drivers plus the passive
    // evaluation, and every value must be identical.
    let v_passive = multi(&POINT);
    let (v_fwd, _) = compute_directional_derivative_fwd(&POINT, &[1.0, 0.0, 0.0], multi);
    let (v_rev, _) = compute_gradient_rev(&POINT, multi);
    assert_eq!(v_passive, v_fwd);
    assert_eq!(v_passive, v_rev);

    // And the scalar driver on a one-input slice of the same shape.
    let (v_scalar, _) = compute_derivative_fwd(POINT[0], scalar);
    assert_eq!(v_scalar, scalar(&POINT[0]));
}

// --- the with-forms: a borrowed tape, reused ------------------------------------

/// A reverse-mode body, as the reuse tests pass them around.
type Body = fn(&[AReal<f64>]) -> AReal<f64>;

/// Three functions of different recording sizes, to drive growth and shrink
/// of a reused tape.
fn small<R: Real>(v: &[R]) -> R {
    v[0].clone() * v[1].clone()
}
fn large<R: Real>(v: &[R]) -> R {
    let mut acc = v[0].clone();
    for _ in 0..2_000 {
        acc = (acc.clone() * v[1].clone()).sin() + v[2].clone();
    }
    acc
}

/// Bare vs `_with` on a fresh tape: the same value and gradient, bit for bit.
#[test]
fn the_with_form_on_a_fresh_tape_is_the_bare_form_exactly() {
    let (v0, g0) = compute_gradient_rev(&POINT, multi);
    let mut tape = Tape::<f64>::new(true);
    let (v1, g1) = compute_gradient_rev_with(&mut tape, &POINT, multi);
    assert_eq!(v0, v1);
    assert_eq!(g0, g1);
    assert!(
        !tape.is_active(),
        "the with-form must leave the tape inactive"
    );
}

/// One tape reused across three different functions returns, for each,
/// exactly what the bare form returns — reuse leaks nothing between
/// recordings, growing and shrinking as the recordings do.
#[test]
fn reuse_across_functions_leaks_nothing() {
    let mut tape = Tape::<f64>::new(true);
    let fns: [Body; 3] = [multi, large, small];
    for f in fns {
        let (vb, gb) = compute_gradient_rev(&POINT, f);
        let (vw, gw) = compute_gradient_rev_with(&mut tape, &POINT, f);
        assert_eq!(vb, vw);
        assert_eq!(gb, gw);
    }
    // And once more with the small one after the large: the retained
    // capacity is a floor, not a cap, and the small recording is unaffected.
    let (vb, gb) = compute_gradient_rev(&POINT, small);
    let (vw, gw) = compute_gradient_rev_with(&mut tape, &POINT, small);
    assert_eq!((vb, gb), (vw, gw));
}

/// The Jacobian with-form agrees entry by entry, on a fresh and a reused tape.
#[test]
fn the_jacobian_with_form_agrees_entry_by_entry() {
    let f = |v: &[AReal<f64>]| vec![multi(v), small(v), v[2].clone().exp()];
    let bare = compute_jacobian_rev(&POINT, f);
    let mut tape = Tape::<f64>::new(true);
    let fresh = compute_jacobian_rev_with(&mut tape, &POINT, f);
    let _ = compute_jacobian_rev_with(&mut tape, &POINT, |v| vec![large(v)]);
    let reused = compute_jacobian_rev_with(&mut tape, &POINT, f);
    assert_eq!(bare, fresh);
    assert_eq!(bare, reused);
    assert!(!tape.is_active());
}

/// A panic inside the differentiated function leaves no tape active, for
/// either driver, either form — the next call succeeds instead of hitting
/// "a tape is already active".
#[test]
fn a_panicking_function_leaves_no_tape_active() {
    let boom = |_: &[AReal<f64>]| -> AReal<f64> { panic!("inside the function") };
    let boom_vec = |_: &[AReal<f64>]| -> Vec<AReal<f64>> { panic!("inside the function") };
    assert!(std::panic::catch_unwind(|| compute_gradient_rev(&POINT, boom)).is_err());
    assert!(std::panic::catch_unwind(|| compute_jacobian_rev(&POINT, boom_vec)).is_err());
    let mut tape = Tape::<f64>::new(true);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compute_jacobian_rev_with(&mut tape, &POINT, boom_vec)
        }))
        .is_err()
    );
    // The proof: a fresh recording opens without complaint on this thread.
    let (v, _) = compute_gradient_rev(&POINT, multi);
    assert!(v.is_finite());
    let (v, _) = compute_gradient_rev_with(&mut tape, &POINT, multi);
    assert!(v.is_finite());
}

/// The reuse figure, measured: bare (a fresh tape per call) against the
/// with-form on one retained tape, on a recording of ~6k statements.
/// Printed for the record; timing is not asserted.
#[test]
fn driver_reuse_measured() {
    use std::time::Instant;
    let n = 200;
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = compute_gradient_rev(&POINT, large);
    }
    let bare = t0.elapsed();
    let mut tape = Tape::<f64>::new(true);
    let t1 = Instant::now();
    for _ in 0..n {
        let _ = compute_gradient_rev_with(&mut tape, &POINT, large);
    }
    let reused = t1.elapsed();
    println!(
        "driver reuse: bare {:?} vs with-reused {:?} per call, ratio {:.2}x",
        bare / n,
        reused / n,
        bare.as_secs_f64() / reused.as_secs_f64().max(1e-12)
    );
}
