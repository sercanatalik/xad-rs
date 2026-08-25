//! Fused n-ary recorders (`math::ad::sum` / `dot` / `weighted_sum` /
//! `weighted_dot`): gradients must match the equivalent binary-operator
//! composition exactly, while recording fewer tape statements and operands.

use xad_rs::{AReal, Jet1, Jet2, JetK, Real, Tape, math};

fn grad_of<F>(inputs: &[f64], f: F) -> (f64, Vec<f64>, usize, usize)
where
    F: Fn(&[AReal<f64>]) -> AReal<f64>,
{
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let mut ad: Vec<AReal<f64>> = inputs.iter().map(|&v| AReal::new(v)).collect();
    AReal::register_input(&mut ad, &mut tape);
    let mut y = f(&ad);
    AReal::register_output(std::slice::from_mut(&mut y), &mut tape);
    y.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    let grads = ad.iter().map(|x| x.adjoint(&tape)).collect();
    // Registered inputs each record a zero-operand statement (slots are 1:1
    // with statements); subtract them so we compare *op* statement counts.
    let op_statements = tape.num_statements() - inputs.len();
    (y.value(), grads, op_statements, tape.num_operations())
}

#[test]
fn fused_sum_matches_binary_chain() {
    let xs: Vec<f64> = (1..=10).map(|i| i as f64 * 0.7).collect();

    let (v_fused, g_fused, s_fused, o_fused) = grad_of(&xs, math::ad::sum);
    let (v_chain, g_chain, s_chain, o_chain) = grad_of(&xs, |ad| {
        let mut acc = ad[0];
        for x in &ad[1..] {
            acc = &acc + x;
        }
        acc
    });

    assert_eq!(v_fused, v_chain);
    assert_eq!(g_fused, g_chain);
    assert!(g_fused.iter().all(|&g| g == 1.0));

    // 1 statement / n operands vs n-1 statements / 2(n-1) operands.
    assert_eq!((s_fused, o_fused), (1, xs.len()));
    assert_eq!((s_chain, o_chain), (xs.len() - 1, 2 * (xs.len() - 1)));
}

#[test]
fn fused_dot_matches_binary_composition() {
    let n = 6;
    let all: Vec<f64> = (0..2 * n).map(|i| 0.3 + i as f64 * 0.11).collect();

    let (v_fused, g_fused, s_fused, o_fused) =
        grad_of(&all, |ad| math::ad::dot(&ad[..n], &ad[n..]));
    let (v_comp, g_comp, s_comp, o_comp) = grad_of(&all, |ad| {
        let mut acc = &ad[0] * &ad[n];
        for i in 1..n {
            acc = &acc + &(&ad[i] * &ad[n + i]);
        }
        acc
    });

    assert!((v_fused - v_comp).abs() < 1e-14);
    for (gf, gc) in g_fused.iter().zip(&g_comp) {
        assert!((gf - gc).abs() < 1e-14);
    }
    // Analytical: d/dx_i = y_i, d/dy_i = x_i.
    for i in 0..n {
        assert_eq!(g_fused[i], all[n + i]);
        assert_eq!(g_fused[n + i], all[i]);
    }

    assert_eq!((s_fused, o_fused), (1, 2 * n));
    assert_eq!((s_comp, o_comp), (2 * n - 1, 4 * n - 2));
}

#[test]
fn fused_ops_skip_inactive_operands() {
    // Mix registered inputs with unrecorded constants: constants contribute
    // to the value but push no operands (slot u32::MAX is filtered).
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let mut x = AReal::new(2.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    let c = AReal::new(5.0); // never registered

    let mut y = math::ad::sum(&[x, c, x]);
    assert_eq!(y.value(), 9.0);
    assert_eq!(tape.num_operations(), 2); // only the two active x operands

    AReal::register_output(std::slice::from_mut(&mut y), &mut tape);
    y.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert_eq!(x.adjoint(&tape), 2.0);
}

#[test]
fn fused_ops_without_active_tape_and_empty_input() {
    // No active tape: plain value math, nothing recorded.
    let a = AReal::new(3.0);
    let b = AReal::new(4.0);
    assert_eq!(math::ad::sum(&[a, b]).value(), 7.0);
    assert_eq!(
        math::ad::dot(std::slice::from_ref(&a), std::slice::from_ref(&b)).value(),
        12.0
    );
    // Empty slices: unrecorded zero, even with an active tape.
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let z = math::ad::sum::<f64>(&[]);
    assert_eq!(z.value(), 0.0);
    assert!(!z.should_record());
    assert_eq!(tape.num_statements(), 0);
}

#[test]
fn nary_statement_through_vector_sweep() {
    // Fused ops must also propagate correctly through the vector reverse
    // sweep used by compute_jacobian_rev.
    let inputs = [1.5_f64, 2.5, -0.5];
    let jac = xad_rs::compute_jacobian_rev(&inputs, |ad| {
        vec![
            math::ad::sum(ad),
            math::ad::dot(&ad[..1], &ad[1..2]),
            math::ad::weighted_sum(&[2.0, 0.0, -1.0], ad),
        ]
    });
    assert_eq!(jac.row(0).to_vec(), vec![1.0, 1.0, 1.0]);
    assert_eq!(jac.row(1).to_vec(), vec![2.5, 1.5, 0.0]);
    assert_eq!(jac.row(2).to_vec(), vec![2.0, 0.0, -1.0]);
}

#[test]
fn fused_weighted_sum_matches_binary_composition() {
    let xs: Vec<f64> = (0..7).map(|i| 0.4 + i as f64 * 0.9).collect();
    let ws: Vec<f64> = (0..7).map(|i| 0.95_f64.powi(i)).collect();

    let (v_fused, g_fused, s_fused, o_fused) =
        grad_of(&xs, |ad| math::ad::weighted_sum(&ws, ad));
    let (v_comp, g_comp, ..) = grad_of(&xs, |ad| {
        let mut acc = &ad[0] * ws[0];
        for (x, &w) in ad[1..].iter().zip(&ws[1..]) {
            acc = &acc + &(x * w);
        }
        acc
    });

    assert!((v_fused - v_comp).abs() < 1e-14);
    for (gf, gc) in g_fused.iter().zip(&g_comp) {
        assert!((gf - gc).abs() < 1e-14);
    }
    // Analytical: d/dx_i = w_i, exactly.
    assert_eq!(g_fused, ws);
    assert_eq!((s_fused, o_fused), (1, xs.len()));
}

#[test]
fn fused_weighted_dot_matches_binary_composition() {
    // The CDS premium-leg shape: passive accruals, active discount factors
    // and survival probabilities.
    let n = 6;
    let all: Vec<f64> = (0..2 * n).map(|i| 0.2 + i as f64 * 0.13).collect();
    let ws: Vec<f64> = (0..n).map(|i| 0.25 + 0.01 * i as f64).collect();

    let (v_fused, g_fused, s_fused, o_fused) =
        grad_of(&all, |ad| math::ad::weighted_dot(&ws, &ad[..n], &ad[n..]));
    let (v_comp, g_comp, s_comp, o_comp) = grad_of(&all, |ad| {
        let mut acc = &(&ad[0] * &ad[n]) * ws[0];
        for i in 1..n {
            acc = &acc + &(&(&ad[i] * &ad[n + i]) * ws[i]);
        }
        acc
    });

    assert!((v_fused - v_comp).abs() < 1e-14);
    for (gf, gc) in g_fused.iter().zip(&g_comp) {
        assert!((gf - gc).abs() < 1e-14);
    }
    // Analytical: d/dx_i = w_i·y_i, d/dy_i = w_i·x_i.
    for i in 0..n {
        assert!((g_fused[i] - ws[i] * all[n + i]).abs() < 1e-15);
        assert!((g_fused[n + i] - ws[i] * all[i]).abs() < 1e-15);
    }

    // 1 statement / 2n operands vs 3n-1 statements / 5n-2 operands (the
    // n active·active multiplies push 2 operands, the n passive scalings 1,
    // the n-1 adds 2).
    assert_eq!((s_fused, o_fused), (1, 2 * n));
    assert_eq!((s_comp, o_comp), (3 * n - 1, 5 * n - 2));
}

#[test]
fn new_fused_ops_edge_cases() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();

    // Empty inputs: unrecorded constants, nothing on the tape.
    assert_eq!(math::ad::weighted_sum::<f64>(&[], &[]).value(), 0.0);
    assert_eq!(math::ad::weighted_dot::<f64>(&[], &[], &[]).value(), 0.0);
    assert_eq!(tape.num_operations(), 0);
}

// ============================================================================
// The same fused recorders reached through `Real`'s aggregate methods.
//
// A `Real` implementation that delegated an aggregate to a binary-operator
// chain would still produce the correct gradient — only the tape size would
// change. These tests therefore assert the *statement and operand counts*
// through the trait method, so that regression fails rather than passing
// silently.
// ============================================================================

#[test]
fn trait_aggregates_record_one_nary_statement() {
    let n = 6;
    let all: Vec<f64> = (0..2 * n).map(|i| 0.3 + i as f64 * 0.11).collect();
    let ws: Vec<f64> = (0..n).map(|i| 0.95_f64.powi(i as i32)).collect();

    // sum: 1 statement / n operands (a binary chain would be n-1 / 2(n-1)).
    let (_, g, s_cnt, o_cnt) = grad_of(&all, |ad| <AReal<f64> as Real>::sum(ad));
    assert_eq!((s_cnt, o_cnt), (1, all.len()));
    assert!(g.iter().all(|&d| d == 1.0));

    // dot: 1 statement / 2n operands (a binary composition: 2n-1 / 4n-2).
    let (_, _, s_cnt, o_cnt) =
        grad_of(&all, |ad| <AReal<f64> as Real>::dot(&ad[..n], &ad[n..]));
    assert_eq!((s_cnt, o_cnt), (1, 2 * n));

    // weighted_sum: 1 statement / n operands.
    let (_, g, s_cnt, o_cnt) =
        grad_of(&all[..n], |ad| <AReal<f64> as Real>::weighted_sum(&ws, ad));
    assert_eq!((s_cnt, o_cnt), (1, n));
    assert_eq!(g, ws);

    // weighted_dot: 1 statement / 2n operands (binary: 3n-1 / 5n-2).
    let (_, _, s_cnt, o_cnt) =
        grad_of(&all, |ad| <AReal<f64> as Real>::weighted_dot(&ws, &ad[..n], &ad[n..]));
    assert_eq!((s_cnt, o_cnt), (1, 2 * n));
}

#[test]
fn trait_aggregates_are_the_crate_fused_recorders() {
    // Bit-identical to `math::ad::*`, not merely numerically close: the
    // trait method must *be* the fused recorder, not a re-derivation.
    let n = 5;
    let all: Vec<f64> = (0..2 * n).map(|i| 0.4 + i as f64 * 0.17).collect();
    let ws: Vec<f64> = (0..n).map(|i| 0.9_f64.powi(i as i32)).collect();

    for (via_trait, via_math) in [
        (
            grad_of(&all, |ad| <AReal<f64> as Real>::sum(ad)),
            grad_of(&all, math::ad::sum),
        ),
        (
            grad_of(&all, |ad| <AReal<f64> as Real>::dot(&ad[..n], &ad[n..])),
            grad_of(&all, |ad| math::ad::dot(&ad[..n], &ad[n..])),
        ),
        (
            grad_of(&all[..n], |ad| <AReal<f64> as Real>::weighted_sum(&ws, ad)),
            grad_of(&all[..n], |ad| math::ad::weighted_sum(&ws, ad)),
        ),
        (
            grad_of(&all, |ad| <AReal<f64> as Real>::weighted_dot(&ws, &ad[..n], &ad[n..])),
            grad_of(&all, |ad| math::ad::weighted_dot(&ws, &ad[..n], &ad[n..])),
        ),
    ] {
        assert_eq!(via_trait, via_math);
    }
}

// ---------------------------------------------------------------------------
// Cross-mode agreement with the equivalent binary-operator chain.
// ---------------------------------------------------------------------------

/// The four aggregates, written once against the trait, folded into one
/// scalar so a single derivative comparison covers all of them.
fn aggregates<R: Real>(xs: &[R], ys: &[R], ws: &[R::Passive]) -> R {
    R::sum(xs) + R::dot(xs, ys) + R::weighted_sum(ws, xs) + R::weighted_dot(ws, xs, ys)
}

/// The same expression built from binary operators only.
/// Bound to `Passive = f64` only so the weights can be lifted with the
/// trait's `From<f64>`; `Real` carries no `From<Self::Passive>` conversion.
fn aggregates_by_chain<R: Real<Passive = f64>>(xs: &[R], ys: &[R], ws: &[f64]) -> R {
    let mut acc = R::zero();
    for x in xs {
        acc = acc + x.clone();
    }
    for (x, y) in xs.iter().zip(ys) {
        acc = acc + x.clone() * y.clone();
    }
    for (w, x) in ws.iter().zip(xs) {
        acc = acc + R::from(*w) * x.clone();
    }
    for ((w, x), y) in ws.iter().zip(xs).zip(ys) {
        acc = acc + R::from(*w) * x.clone() * y.clone();
    }
    acc
}

const XS: [f64; 4] = [0.7, 1.3, -0.4, 2.1];
const YS: [f64; 4] = [1.1, -0.6, 0.9, 0.25];
const WS: [f64; 4] = [0.25, 0.5, 0.75, 1.0];

#[test]
fn trait_aggregates_match_binary_chain_in_every_mode() {
    // --- passive ---
    let v_agg = aggregates(&XS, &YS, &WS);
    let v_chain = aggregates_by_chain(&XS, &YS, &WS);
    assert!((v_agg - v_chain).abs() < 1e-14, "f64: {v_agg} vs {v_chain}");

    // --- forward first order, seeded on xs[0] ---
    let seed = |i: usize| Jet1::new(XS[i], if i == 0 { 1.0 } else { 0.0 });
    let xs1: Vec<Jet1<f64>> = (0..4).map(seed).collect();
    let ys1: Vec<Jet1<f64>> = YS.iter().map(|&y| Jet1::constant(y)).collect();
    let a1 = aggregates(&xs1, &ys1, &WS);
    let c1 = aggregates_by_chain(&xs1, &ys1, &WS);
    assert!((a1.value() - c1.value()).abs() < 1e-14);
    assert!(
        (a1.derivative() - c1.derivative()).abs() < 1e-14,
        "Jet1 d1: {} vs {}",
        a1.derivative(),
        c1.derivative()
    );
    // Analytic: d/dx0 of (Σx + Σxy + Σwx + Σwxy) = 1 + y0 + w0 + w0·y0.
    let want = 1.0 + YS[0] + WS[0] + WS[0] * YS[0];
    assert!((a1.derivative() - want).abs() < 1e-14);

    // --- K-lane forward: lane 0 on xs[0], lane 1 on xs[1] — two gradient
    // entries from one pass, each checked against the chain and the analytic
    // partial. Lanes 2 and 3 are never seeded and must stay zero.
    let xsk: Vec<JetK<f64, 4>> = (0..4)
        .map(|i| {
            let mut t = [0.0; 4];
            if i < 2 {
                t[i] = 1.0;
            }
            JetK::new(XS[i], t)
        })
        .collect();
    let ysk: Vec<JetK<f64, 4>> = YS.iter().map(|&y| JetK::constant(y)).collect();
    let ak = aggregates(&xsk, &ysk, &WS);
    let ck = aggregates_by_chain(&xsk, &ysk, &WS);
    assert_eq!(ak.value, v_agg, "JetK value vs f64");
    assert_eq!(ak.value, ck.value, "JetK value vs chain");
    assert_eq!(ak.tangents, ck.tangents, "JetK tangents vs chain");
    for i in 0..2 {
        let want = 1.0 + YS[i] + WS[i] + WS[i] * YS[i];
        assert!((ak.tangents[i] - want).abs() < 1e-14, "JetK lane {i}: {} vs {want}", ak.tangents[i]);
    }
    assert_eq!(ak.tangents[2..], [0.0, 0.0]);
    assert_eq!(ak.tangents[0], a1.derivative(), "JetK lane 0 vs Jet1 tangent");

    // --- forward second order ---
    let xs2: Vec<Jet2<f64>> = (0..4)
        .map(|i| if i == 0 { Jet2::variable(XS[i]) } else { Jet2::constant(XS[i]) })
        .collect();
    let ys2: Vec<Jet2<f64>> = YS.iter().map(|&y| Jet2::constant(y)).collect();
    let a2 = aggregates(&xs2, &ys2, &WS);
    let c2 = aggregates_by_chain(&xs2, &ys2, &WS);
    assert!((a2.value() - c2.value()).abs() < 1e-14);
    assert!((a2.first_derivative() - c2.first_derivative()).abs() < 1e-14);
    assert!((a2.second_derivative() - c2.second_derivative()).abs() < 1e-14);

    // --- reverse ---
    let rev = |use_aggregates: bool| {
        let mut tape = Tape::<f64>::new(true);
        let _rec = tape.record();
        let mut xs: Vec<AReal<f64>> = XS.iter().map(|&v| AReal::new(v)).collect();
        AReal::register_input(&mut xs, &mut tape);
        let ys: Vec<AReal<f64>> = YS.iter().map(|&v| AReal::new(v)).collect();
        let mut y = if use_aggregates {
            aggregates(&xs, &ys, &WS)
        } else {
            aggregates_by_chain(&xs, &ys, &WS)
        };
        AReal::register_output(std::slice::from_mut(&mut y), &mut tape);
        y.set_adjoint(&mut tape, 1.0);
        tape.compute_adjoints();
        let g: Vec<f64> = xs.iter().map(|x| x.adjoint(&tape)).collect();
        (y.value(), g)
    };
    let (v_r, g_r) = rev(true);
    let (v_c, g_c) = rev(false);
    assert!((v_r - v_c).abs() < 1e-14);
    for (i, (a, b)) in g_r.iter().zip(&g_c).enumerate() {
        assert!((a - b).abs() < 1e-14, "reverse grad[{i}]: {a} vs {b}");
        // Analytic: ∂/∂xᵢ = 1 + yᵢ + wᵢ + wᵢ·yᵢ.
        let want = 1.0 + YS[i] + WS[i] + WS[i] * YS[i];
        assert!((a - want).abs() < 1e-14, "reverse grad[{i}]: {a} vs {want}");
    }
    // The forward tangent on xs[0] must equal the reverse adjoint on xs[0].
    assert!((a1.derivative() - g_r[0]).abs() < 1e-14);
    // And the passive value must equal the active ones.
    assert!((v_agg - v_r).abs() < 1e-14);
}

#[test]
fn passive_aggregates_record_nothing_with_a_tape_active() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    // A tape is live on this thread, but `f64`'s aggregates run no AD
    // machinery at all.
    let v = aggregates(&XS, &YS, &WS);
    assert!(v.is_finite());
    assert_eq!(tape.num_statements(), 0);
    assert_eq!(tape.num_operations(), 0);
}

#[test]
fn trait_aggregates_handle_empty_slices() {
    let empty: [f64; 0] = [];
    assert_eq!(<f64 as Real>::sum(&empty), 0.0);
    assert_eq!(<f64 as Real>::dot(&empty, &empty), 0.0);
    assert_eq!(<f64 as Real>::weighted_sum(&empty, &empty), 0.0);
    assert_eq!(<f64 as Real>::weighted_dot(&empty, &empty, &empty), 0.0);

    let e1: [Jet1<f64>; 0] = [];
    assert_eq!(<Jet1<f64> as Real>::sum(&e1).value(), 0.0);
    let e2: [Jet2<f64>; 0] = [];
    assert_eq!(<Jet2<f64> as Real>::sum(&e2).value(), 0.0);
    let ek: [JetK<f64, 3>; 0] = [];
    assert_eq!(<JetK<f64, 3> as Real>::sum(&ek).value, 0.0);
    assert_eq!(<JetK<f64, 3> as Real>::dot(&ek, &ek).tangents, [0.0; 3]);
    assert_eq!(<JetK<f64, 3> as Real>::weighted_sum(&empty, &ek).value, 0.0);
    assert_eq!(<JetK<f64, 3> as Real>::weighted_dot(&empty, &ek, &ek).value, 0.0);

    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let ea: [AReal<f64>; 0] = [];
    assert_eq!(<AReal<f64> as Real>::sum(&ea).value(), 0.0);
    assert_eq!(tape.num_statements(), 0);
}
