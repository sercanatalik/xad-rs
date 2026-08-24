//! The acceptance gate for the operand-ergonomics bounds: **one body written
//! two ways must be the same number.**
//!
//! `CopyableReal` and the passive-operand bounds on `Real` are a type-level
//! change — they add no arithmetic. That is exactly what makes them dangerous
//! to ship on a green test suite: the shorter spelling resolves to a
//! *different operator impl* than the longer one (`tau * x` records a unary
//! tape statement where `AReal::from(tau) * x` records a binary one; `x /
//! tau` takes the scalar-RHS `Div` where `x / R::from(tau)` takes the
//! two-`Real` one). Every existing test uses the old spelling, so every
//! existing test would stay green if the new spelling landed a different
//! number.
//!
//! So this file writes one body twice — once with clones and `R::from` lifts,
//! once with neither — and asserts the two agree bit-for-bit in all four
//! modes, values and derivatives.
//!
//! # Why the body returns its terms instead of their sum
//!
//! Because a sum hides exactly the defect this file exists to catch. The
//! first version of this test returned `a + b + c + d + e + f` and compared
//! the totals; mutating one passive-position `Div` back to the two-rounding
//! `l * (1/r)` spelling **did not fail it**. The mutated term was around
//! `0.4` and the total around `10`, so the 1-ulp error landed below the
//! total's own ulp and was rounded away.
//!
//! The comparison is therefore term by term, one term per operator impl
//! under test. Each of the eight passive-operand impls the bounds oblige a
//! mode to supply gets its own assertion at its own magnitude, where a 1-ulp
//! divergence has nothing to hide behind.
//!
//! See `openspec/changes/operand-ergonomics/design.md`, decision D3.

use xad_rs::{AReal, CopyableReal, Jet1, Jet2, Real, Tape};

/// Evaluation point, passive weight, passive rate. Pinned by
/// [`the_point_distinguishes_the_two_quotient_forms`] to a place where a
/// correctly rounded `a / b` and the two-rounding `a * (1/b)` disagree for
/// every division in the body, so this file also re-gates the division rule
/// from 6.1.0 across both spellings.
const X0: f64 = 3.0;
const W: f64 = 1.2;
const TAU: f64 = 1.25;

/// One term per operator impl under test, in the order the bodies build them.
const TERMS: [&str; 10] = [
    "x - w",      // Sub<Passive> — passive right
    "x + w",      // Add<Passive> — passive right
    "(x-w)*(x+w)", // Mul<Self>
    "tau + x",    // Passive: Add<Self> — passive left
    "num / den",  // Div<Self>
    "tau - x",    // Passive: Sub<Self> — passive left
    "tau * x",    // Passive: Mul<Self> — passive left
    "tau / x",    // Passive: Div<Self> — passive left
    "x * tau",    // Mul<Passive> — passive right
    "x / tau",    // Div<Passive> — passive right
];

/// The body as it must be written today: every reuse of `x` spelled as a
/// clone, every passive operand lifted with `R::from`.
///
/// Bound to `Passive = f64` because `R::from` takes an `f64` — which is
/// exactly the tax the bare twin below removes.
fn lifted<R: Real<Passive = f64>>(x: &R, w: f64, tau: f64) -> [R; 10] {
    let t0 = x.clone() - R::from(w);
    let t1 = x.clone() + R::from(w);
    let num = t0.clone() * t1.clone();
    let den = R::from(tau) + x.clone();
    [
        t0,
        t1,
        num.clone(),
        den.clone(),
        num / den,
        R::from(tau) - x.clone(),
        R::from(tau) * x.clone(),
        R::from(tau) / x.clone(),
        x.clone() * R::from(tau),
        x.clone() / R::from(tau),
    ]
}

/// The same body as [`lifted`], written the way the mathematics reads: `x` is
/// used ten times without a clone, and the passive operands meet the active
/// scalar in both positions without a lift.
///
/// **The passive type is left as `R::Passive` on purpose.** Writing the bound
/// as `CopyableReal<Passive = f64>` — as [`lifted`] must — normalizes the
/// projection away and with it the left-hand bounds, and `tau * x` stops
/// compiling while `x * tau` keeps working. The right-hand bounds survive the
/// pin because they sit on `Self`; the left-hand ones do not because they sit
/// on the projection. So a body that wants a passive operand on the left must
/// name `R::Passive` rather than pin it, which is the spelling a caller would
/// reach for anyway. [`pinning_the_passive_type_is_not_required`] holds that
/// shape.
fn bare<R: CopyableReal>(x: R, w: R::Passive, tau: R::Passive) -> [R; 10] {
    let t0 = x - w;
    let t1 = x + w;
    let num = t0 * t1;
    let den = tau + x;
    [
        t0,
        t1,
        num,
        den,
        num / den,
        tau - x,
        tau * x,
        tau / x,
        x * tau,
        x / tau,
    ]
}

/// The pair above only gates the division rule at a point where the two
/// spellings of a quotient actually differ — they agree for most inputs, and
/// for every input whose numerator is a power of two. Pin that for **all
/// three** divisions in the body (the active/active one and both passive
/// positions), so moving `X0`/`W`/`TAU` cannot silently turn the tests below
/// back into ones that would pass against a reciprocal-built quotient.
///
/// This assertion is not decoration: the first constants tried here looked
/// entirely reasonable and distinguished none of the three.
#[test]
fn the_point_distinguishes_the_two_quotient_forms() {
    let num = (X0 - W) * (X0 + W);
    let den = TAU + X0;
    for (a, b, what) in [(num, den, "num / den"), (TAU, X0, "tau / x"), (X0, TAU, "x / tau")] {
        assert_ne!(
            a * (1.0 / b),
            a / b,
            "{what}: the evaluation point no longer distinguishes a/b from a*(1/b)"
        );
    }
}

/// A body over an unpinned `R::Passive` reaches every concrete mode — the
/// spelling `bare` uses is not a special case that only works for `f64`.
#[test]
fn pinning_the_passive_type_is_not_required() {
    fn takes_passive_on_both_sides<R: CopyableReal>(x: R, t: R::Passive) -> R {
        (t + x) + (x + t) + (t - x) + (x - t) + (t * x) + (x * t) + (t / x) + (x / t)
    }
    // Two modes is enough to show the bound is not accidentally concrete.
    let _ = takes_passive_on_both_sides(2.0_f64, 0.5);
    let _ = takes_passive_on_both_sides(Jet1::new(2.0, 1.0), 0.5);
}

/// All four in-crate modes meet the copyable bound. This is a compile-time
/// assertion: it is the check that the sub-trait is actually inhabited by
/// everything that implements `Real` today, which is what makes the blanket
/// impl in `src/real.rs` load-bearing rather than decorative.
#[test]
fn every_in_crate_mode_is_copyable() {
    fn assert_copyable<R: CopyableReal>() {}
    assert_copyable::<f64>();
    assert_copyable::<AReal<f64>>();
    assert_copyable::<Jet1<f64>>();
    assert_copyable::<Jet2<f64>>();
}

/// Compare two spellings term by term on one projection of the mode.
fn assert_terms_agree<R>(bare: &[R; 10], lifted: &[R; 10], project: impl Fn(&R) -> f64, what: &str) {
    for i in 0..10 {
        assert_eq!(
            project(&bare[i]),
            project(&lifted[i]),
            "{what}: term `{}` differs between spellings",
            TERMS[i]
        );
    }
}

#[test]
fn spellings_agree_under_f64() {
    assert_terms_agree(&bare(X0, W, TAU), &lifted(&X0, W, TAU), |v| *v, "f64");
}

#[test]
fn spellings_agree_under_jet1() {
    let x = Jet1::new(X0, 1.0);
    let (b, l) = (bare(x, W, TAU), lifted(&x, W, TAU));
    assert_terms_agree(&b, &l, Jet1::value, "Jet1 value");
    assert_terms_agree(&b, &l, Jet1::derivative, "Jet1 derivative");
    // ...and the values are still the passive ones, not mode-dependent numbers.
    assert_terms_agree(&b, &bare(X0, W, TAU).map(Jet1::constant), Jet1::value, "Jet1 vs f64");
}

#[test]
fn spellings_agree_under_jet2() {
    let x = Jet2::variable(X0);
    let (b, l) = (bare(x, W, TAU), lifted(&x, W, TAU));
    assert_terms_agree(&b, &l, Jet2::value, "Jet2 value");
    assert_terms_agree(&b, &l, Jet2::first_derivative, "Jet2 d1");
    assert_terms_agree(&b, &l, Jet2::second_derivative, "Jet2 d2");
    assert_terms_agree(&b, &bare(X0, W, TAU).map(Jet2::constant), Jet2::value, "Jet2 vs f64");
}

#[test]
fn spellings_agree_under_areal() {
    // Each spelling gets its own tape: the two record *different statement
    // shapes* for the same mathematics (unary where the lifted form records
    // binary), and the claim is that value and adjoint come out the same
    // anyway.
    let b = areal_run(|x| bare(*x, W, TAU));
    let l = areal_run(|x| lifted(x, W, TAU));

    for i in 0..10 {
        assert_eq!(b[i].0, l[i].0, "AReal value: term `{}` differs", TERMS[i]);
        assert_eq!(b[i].1, l[i].1, "AReal adjoint: term `{}` differs", TERMS[i]);
        assert_eq!(b[i].0, bare(X0, W, TAU)[i], "AReal value vs f64: term `{}`", TERMS[i]);
    }
}

/// Tape the given body at `X0` and return `(value, d/dx)` for each of its ten
/// terms. Each term gets its own tape so the ten adjoint sweeps do not see
/// each other's seeds.
fn areal_run(body: impl Fn(&AReal<f64>) -> [AReal<f64>; 10]) -> [(f64, f64); 10] {
    std::array::from_fn(|i| {
        let mut tape = Tape::<f64>::new(true);
        tape.activate();
        let mut x = AReal::new(X0);
        AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
        let mut out = body(&x)[i];
        out.register(&mut tape);
        out.set_adjoint(&mut tape, 1.0);
        tape.compute_adjoints();
        let result = (out.value(), x.adjoint(&tape));
        Tape::<f64>::deactivate_all();
        result
    })
}

/// The shared derivatives are the analytic ones, not just each other's.
/// Without this the pairs above would still pass if *both* spellings were
/// wrong in the same way.
#[test]
fn the_shared_derivatives_are_the_analytic_ones() {
    let den = TAU + X0;
    let expected = [
        1.0,                                                  // x - w
        1.0,                                                  // x + w
        2.0 * X0,                                             // x² - w²
        1.0,                                                  // tau + x
        (2.0 * X0 * den - (X0 * X0 - W * W)) / (den * den),   // num / den
        -1.0,                                                 // tau - x
        TAU,                                                  // tau * x
        -TAU / (X0 * X0),                                     // tau / x
        TAU,                                                  // x * tau
        1.0 / TAU,                                            // x / tau
    ];

    let fwd = bare(Jet1::new(X0, 1.0), W, TAU).map(|t| t.derivative());
    let rev = areal_run(|x| bare(*x, W, TAU)).map(|(_, d)| d);

    for i in 0..10 {
        for (got, mode) in [(fwd[i], "Jet1"), (rev[i], "AReal")] {
            let tol = 32.0 * f64::EPSILON * (1.0 + expected[i].abs().max(got.abs()));
            assert!(
                (got - expected[i]).abs() <= tol,
                "{mode} d/dx `{}`: {got} vs analytic {} (tol {tol:e})",
                TERMS[i],
                expected[i]
            );
        }
    }
}
