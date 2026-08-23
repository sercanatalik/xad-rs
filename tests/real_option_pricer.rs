//! The headline claim, end to end: a closed-form option value written
//! **once** against `Real` — with no mode-specific mathematical import —
//! evaluated under the passive, forward first-order, forward second-order,
//! and reverse modes.
//!
//! Before the Gaussian family reached `Real`, this body could not be
//! written generically: `norm_cdf` was only in `math::ad` / `math::fwd`, so
//! a pricer had to name one of them and was welded to that mode.
//!
//! Note the imports below: the crate's `math` module appears only in the
//! *reference* expressions the test checks against, never in the pricer.

use xad_rs::{
    compute_derivative_fwd, compute_directional_derivative_fwd, compute_gradient_rev, Jet2, Real,
};

/// Black–Scholes value of a European call on a continuous-dividend asset.
///
/// `d₁ = (ln(S/K) + (r − q + σ²/2)·T) / (σ√T)`, `d₂ = d₁ − σ√T`,
/// `C = S·e^{−qT}·Φ(d₁) − K·e^{−rT}·Φ(d₂)`.
fn black_scholes_call<R: Real>(s: &R, k: &R, r: &R, q: &R, vol: &R, t: &R) -> R {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = vol.clone() * sqrt_t;
    let drift = r.clone() - q.clone() + vol.clone() * vol.clone() * R::from(0.5_f64);
    let d1 = ((s.clone() / k.clone()).ln() + drift * t.clone()) / vol_sqrt_t.clone();
    let d2 = d1.clone() - vol_sqrt_t;
    s.clone() * (-(q.clone() * t.clone())).exp() * d1.norm_cdf()
        - k.clone() * (-(r.clone() * t.clone())).exp() * d2.norm_cdf()
}

/// `[S, K, r, q, σ, T]` — the pricer's inputs as one slice, so the same body
/// reaches the gradient and directional-derivative drivers.
fn call_of<R: Real>(v: &[R]) -> R {
    black_scholes_call(&v[0], &v[1], &v[2], &v[3], &v[4], &v[5])
}

const P: [f64; 6] = [100.0, 95.0, 0.03, 0.01, 0.2, 1.5];
const S: usize = 0;
const VOL: usize = 4;

/// `d₁` and `d₂` at `P`, from the crate's passive Gaussians — the reference
/// side of every Greek assertion below.
fn d1_d2() -> (f64, f64) {
    let (s, k, r, q, vol, t) = (P[0], P[1], P[2], P[3], P[4], P[5]);
    let vol_sqrt_t = vol * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * vol * vol) * t) / vol_sqrt_t;
    (d1, d1 - vol_sqrt_t)
}

#[test]
fn one_body_prices_identically_in_every_mode() {
    let v_passive = call_of(&P);

    // Closed-form reference, written non-generically against `math`.
    let (d1, d2) = d1_d2();
    let want = P[0] * (-P[3] * P[5]).exp() * xad_rs::math::norm_cdf(d1)
        - P[1] * (-P[2] * P[5]).exp() * xad_rs::math::norm_cdf(d2);
    assert_eq!(v_passive, want, "generic and non-generic values must agree");

    // Forward first order (seeded on S) and reverse must return the same
    // value bit-for-bit — same operations, same order, only the derivative
    // machinery differs.
    let mut seed = [0.0; 6];
    seed[S] = 1.0;
    let (v_fwd, _) = compute_directional_derivative_fwd(&P, &seed, call_of);
    let (v_rev, _) = compute_gradient_rev(&P, call_of);
    assert_eq!(v_passive, v_fwd, "passive vs forward value");
    assert_eq!(v_passive, v_rev, "passive vs reverse value");

    // Forward second order.
    let jets: Vec<Jet2<f64>> = P
        .iter()
        .enumerate()
        .map(|(i, &v)| if i == S { Jet2::variable(v) } else { Jet2::constant(v) })
        .collect();
    assert_eq!(call_of(&jets).value(), v_passive, "passive vs Jet2 value");

    // Sanity: the price sits between its intrinsic value and the spot.
    assert!(v_passive > (P[0] - P[1]).max(0.0) && v_passive < P[0]);
}

#[test]
fn delta_and_vega_agree_across_modes_and_with_the_closed_form() {
    let (d1, _) = d1_d2();
    let want_delta = (-P[3] * P[5]).exp() * xad_rs::math::norm_cdf(d1);
    // Vega = S·e^{−qT}·φ(d₁)·√T.
    let want_vega = P[0] * (-P[3] * P[5]).exp() * xad_rs::math::norm_pdf(d1) * P[5].sqrt();

    // Reverse: every Greek from one sweep.
    let (_, grad) = compute_gradient_rev(&P, call_of);

    for (i, want) in [(S, want_delta), (VOL, want_vega)] {
        let mut seed = [0.0; 6];
        seed[i] = 1.0;
        let (_, fwd) = compute_directional_derivative_fwd(&P, &seed, call_of);

        let tol = 1e-11 * (1.0 + want.abs());
        assert!((fwd - want).abs() <= tol, "forward Greek {i}: {fwd} vs {want}");
        assert!((grad[i] - want).abs() <= tol, "reverse Greek {i}: {} vs {want}", grad[i]);
    }
}

#[test]
fn gamma_comes_from_the_second_order_mode() {
    // Γ = e^{−qT}·φ(d₁) / (S·σ·√T).
    let (d1, _) = d1_d2();
    let want =
        (-P[3] * P[5]).exp() * xad_rs::math::norm_pdf(d1) / (P[0] * P[4] * P[5].sqrt());

    let jets: Vec<Jet2<f64>> = P
        .iter()
        .enumerate()
        .map(|(i, &v)| if i == S { Jet2::variable(v) } else { Jet2::constant(v) })
        .collect();
    let out = call_of(&jets);

    assert!(
        (out.second_derivative() - want).abs() <= 1e-11 * (1.0 + want.abs()),
        "gamma: {} vs {want}",
        out.second_derivative()
    );
    // And its first derivative is the same delta the other modes report.
    let (_, grad) = compute_gradient_rev(&P, call_of);
    assert!((out.first_derivative() - grad[S]).abs() < 1e-12);
}

#[test]
fn a_single_input_pricer_reaches_the_scalar_driver() {
    // The sequential-root-solve shape: everything but vol held fixed, so the
    // pricer becomes `R → R` and the scalar driver applies directly.
    let price_of_vol = |vol: &xad_rs::Jet1<f64>| {
        let c = |i: usize| xad_rs::Jet1::constant(P[i]);
        black_scholes_call(&c(0), &c(1), &c(2), &c(3), vol, &c(5))
    };
    let (v, vega) = compute_derivative_fwd(P[VOL], price_of_vol);

    assert_eq!(v, call_of(&P), "scalar driver value");
    let (d1, _) = d1_d2();
    let want = P[0] * (-P[3] * P[5]).exp() * xad_rs::math::norm_pdf(d1) * P[5].sqrt();
    assert!((vega - want).abs() <= 1e-11 * (1.0 + want.abs()), "vega: {vega} vs {want}");
    // Vega is strictly positive for a live option — the property a vol
    // root-solve depends on.
    assert!(vega > 0.0);
}
