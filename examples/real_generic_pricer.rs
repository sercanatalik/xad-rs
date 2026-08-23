//! `Real`-generic pricing — one body, every mode, plus the first-order drivers.
//!
//! Every other example picks a mode first and writes the pricer against it.
//! This one goes the other way: the payoff is written **once** against the
//! mode-agnostic [`Real`] trait,
//!
//!     fn call_price<R: Real>(s: &R, k: &R, r: &R, vol: &R, t: &R) -> R
//!
//! and then evaluated unchanged under `f64` (no derivatives), `AReal<f64>`
//! (reverse), `Jet1<f64>` (forward), and `Jet2<f64>` (forward, second order).
//! Nothing about the body knows which mode it is running in — the Gaussian
//! family (`norm_cdf`) and the transcendentals live on `Real` itself, so the
//! closed form is ordinary generic Rust.
//!
//! It also exercises the three first-order drivers, which own their tape
//! management so the caller does no registration, seeding, or sweeping:
//!
//!   - [`compute_derivative_fwd`] — `R → R`, one evaluation, no tape.
//!   - [`compute_directional_derivative_fwd`] — `Rⁿ → R` along a seed.
//!   - [`compute_gradient_rev`] — `Rⁿ → R`, whole gradient from one sweep.
//!
//! Finally it shows why [`Real::weighted_sum`] exists: on a portfolio leg it
//! records **one** n-ary tape statement where the equivalent `+` chain records
//! one per term. Same gradient, smaller tape, shorter sweep.
//!
//! Model (Black–Scholes European call):
//!     C  = S·N(d1) − K·e^(−r·T)·N(d2)
//!     d1 = [ln(S/K) + (r + ½σ²)·T] / (σ·√T)
//!     d2 = d1 − σ·√T
//!
//! Every number printed is cross-checked against the closed-form Greeks.

use xad_rs::{
    AReal, Jet1, Jet2, Real, Tape, compute_derivative_fwd, compute_directional_derivative_fwd,
    compute_gradient_rev,
};

// ---------------------------------------------------------------------------
// The one and only pricing body. No mode appears anywhere in it.
// ---------------------------------------------------------------------------

/// Black–Scholes European call value, generic over the active scalar.
fn call_price<R: Real>(s: &R, k: &R, r: &R, vol: &R, t: &R) -> R {
    let sqrt_t = t.sqrt();
    let half = R::from(0.5_f64);
    let d1 = ((s.clone() / k.clone()).ln()
        + (r.clone() + half * vol.clone() * vol.clone()) * t.clone())
        / (vol.clone() * sqrt_t.clone());
    let d2 = d1.clone() - vol.clone() * sqrt_t;
    s.clone() * d1.norm_cdf() - k.clone() * (-r.clone() * t.clone()).exp() * d2.norm_cdf()
}

// ---------------------------------------------------------------------------
// Closed-form reference, written non-generically on purpose.
// ---------------------------------------------------------------------------

struct Analytic {
    price: f64,
    delta: f64,
    vega: f64,
    rho: f64,
    gamma: f64,
}

fn analytic(s: f64, k: f64, r: f64, vol: f64, t: f64) -> Analytic {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    // Reuse the crate's own normal CDF/PDF so the reference and the AD path
    // share one evaluation of the Gaussian — any difference printed below is
    // then attributable to the differentiation, not to two different `N(x)`.
    let n = |x: f64| xad_rs::math::norm_cdf(x);
    let phi = |x: f64| xad_rs::math::norm_pdf(x);
    Analytic {
        price: s * n(d1) - k * (-r * t).exp() * n(d2),
        delta: n(d1),
        vega: s * phi(d1) * sqrt_t,
        rho: k * t * (-r * t).exp() * n(d2),
        gamma: phi(d1) / (s * vol * sqrt_t),
    }
}

fn check(label: &str, got: f64, want: f64) {
    let err = (got - want).abs();
    let tol = 1e-10 * (1.0 + want.abs());
    let mark = if err <= tol { "ok" } else { "MISMATCH" };
    println!("  {label:<26} {got:>16.10}   (analytic {want:>16.10})  {mark}");
    assert!(err <= tol, "{label}: got {got}, want {want}");
}

fn main() {
    let (s, k, r, vol, t) = (100.0_f64, 100.0, 0.05, 0.20, 1.0);
    let a = analytic(s, k, r, vol, t);

    println!("Real-generic Black–Scholes call");
    println!("===============================");
    println!("S = {s}   K = {k}   r = {r}   sigma = {vol}   T = {t}");
    println!();

    // -- 1. Passive -------------------------------------------------------
    // R = f64. Monomorphization erases the trait: this is the plain formula,
    // and no tape exists even if one were active on the thread.
    println!("1. Passive (R = f64) — value only");
    let px = call_price(&s, &k, &r, &vol, &t);
    check("price", px, a.price);
    println!();

    // -- 2. Reverse, all Greeks in one sweep -------------------------------
    // compute_gradient_rev creates the tape, registers the inputs, seeds the
    // output adjoint, sweeps, and deactivates — all internally.
    println!("2. Reverse (R = AReal<f64>) — full gradient from ONE sweep");
    let (v_rev, grad) = compute_gradient_rev(&[s, k, r, vol, t], |x| {
        call_price(&x[0], &x[1], &x[2], &x[3], &x[4])
    });
    check("price", v_rev, a.price);
    check("delta  dC/dS", grad[0], a.delta);
    check("rho    dC/dr", grad[2], a.rho);
    check("vega   dC/dsigma", grad[3], a.vega);
    println!("  (dC/dK and dC/dT come from the same sweep at no extra cost)");
    println!();

    // -- 3. Forward, scalar driver ----------------------------------------
    // R → R: spot is the only active input, everything else is a constant
    // jet. No tape is created or activated.
    println!("3. Forward (R = Jet1<f64>) — compute_derivative_fwd, no tape");
    let (v_fwd, delta_fwd) = compute_derivative_fwd(s, |spot| {
        call_price(
            spot,
            &Jet1::constant(k),
            &Jet1::constant(r),
            &Jet1::constant(vol),
            &Jet1::constant(t),
        )
    });
    check("price", v_fwd, a.price);
    check("delta", delta_fwd, a.delta);
    println!();

    // -- 4. Forward, directional driver ------------------------------------
    // A unit seed recovers a single partial; a general seed gives the whole
    // directional derivative in ONE evaluation.
    println!("4. Forward — compute_directional_derivative_fwd");
    let price_of = |x: &[Jet1<f64>]| call_price(&x[0], &x[1], &x[2], &x[3], &x[4]);
    let point = [s, k, r, vol, t];

    let (_, vega_dir) =
        compute_directional_derivative_fwd(&point, &[0.0, 0.0, 0.0, 1.0, 0.0], price_of);
    check("vega (unit seed)", vega_dir, a.vega);

    // Seed vol and r together: one pass yields the blended sensitivity.
    let seed = [0.0, 0.0, 1.0, 1.0, 0.0];
    let (_, blended) = compute_directional_derivative_fwd(&point, &seed, price_of);
    check("d/d(sigma) + d/dr", blended, a.vega + a.rho);
    println!();

    // -- 5. Forward, second order ------------------------------------------
    // The same body again, seeded in spot, now carrying curvature.
    println!("5. Forward 2nd order (R = Jet2<f64>) — gamma from the same body");
    let out = call_price(
        &Jet2::variable(s),
        &Jet2::constant(k),
        &Jet2::constant(r),
        &Jet2::constant(vol),
        &Jet2::constant(t),
    );
    check("price", out.value(), a.price);
    check("delta", out.first_derivative(), a.delta);
    check("gamma", out.second_derivative(), a.gamma);
    println!();

    // -- 6. Fused aggregate ------------------------------------------------
    // A portfolio leg: notionals are contract data (passive weights), the
    // per-name prices are active. `Real::weighted_sum` records the whole leg
    // as one n-ary statement.
    println!("6. Real::weighted_sum — one tape statement instead of n");
    let notionals = [
        1.0e6_f64, -2.5e6, 0.75e6, 3.0e6, -1.2e6, 0.4e6, 2.2e6, -0.9e6,
    ];
    let spots = [95.0_f64, 100.0, 105.0, 98.5, 102.5, 110.0, 90.0, 100.5];

    let (fused_stmts, fused_pv) = leg_pv(&notionals, &spots, k, r, vol, t, true);
    let (chain_stmts, chain_pv) = leg_pv(&notionals, &spots, k, r, vol, t, false);

    println!("  weighted_sum : {fused_stmts:>4} tape statements   PV = {fused_pv:.6}");
    println!("  `+` chain    : {chain_stmts:>4} tape statements   PV = {chain_pv:.6}");
    assert!(
        (fused_pv - chain_pv).abs() <= 1e-6 * (1.0 + chain_pv.abs()),
        "fused and chained PV disagree"
    );
    assert!(
        fused_stmts < chain_stmts,
        "weighted_sum must record fewer statements than the chain"
    );
    println!(
        "  identical PV, {} fewer statements for the reverse sweep to walk",
        chain_stmts - fused_stmts
    );
    println!();

    println!("All modes agree with the closed form.");
}

/// Price a portfolio leg on its own tape and report `(statements, pv)`.
///
/// `fused = true` accumulates with [`Real::weighted_sum`]; `false` uses the
/// binary `+` chain the trait method replaces. Everything else is identical,
/// so the statement counts are directly comparable.
fn leg_pv(
    notionals: &[f64],
    spots: &[f64],
    k: f64,
    r: f64,
    vol: f64,
    t: f64,
    fused: bool,
) -> (usize, f64) {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();

    let mut inputs: Vec<AReal<f64>> = spots.iter().map(|&v| AReal::new(v)).collect();
    AReal::register_input(&mut inputs, &mut tape);

    let ka = AReal::new(k);
    let ra = AReal::new(r);
    let va = AReal::new(vol);
    let ta = AReal::new(t);

    let prices: Vec<AReal<f64>> = inputs
        .iter()
        .map(|spot| call_price(spot, &ka, &ra, &va, &ta))
        .collect();

    let before = tape.num_statements();
    let pv = if fused {
        <AReal<f64> as Real>::weighted_sum(notionals, &prices)
    } else {
        let mut acc = <AReal<f64> as Real>::zero();
        for (&w, p) in notionals.iter().zip(&prices) {
            acc = acc + AReal::new(w) * p.clone();
        }
        acc
    };
    let statements = tape.num_statements() - before;

    (statements, pv.value())
}
