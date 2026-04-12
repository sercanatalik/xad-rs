//! FX option pricer — Garman–Kohlhagen with Greeks via three AD modes.
//!
//! Prices a European FX call (or put) under the Garman–Kohlhagen model
//! (Black–Scholes extended to FX with separate domestic and foreign
//! interest rates), and reports the full Greek vector computed three
//! complementary ways:
//!
//!   1. **Forward-mode delta (multi-variable `Dual`)** — all 6 inputs
//!      (S, K, T, σ, r_d, r_f) are seeded as active variables in a single
//!      6-D forward pass, and **one** pass yields the whole gradient.
//!
//!   2. **Reverse-mode (AReal + tape)** — the pricing expression is
//!      recorded on a tape and a single reverse sweep yields the same
//!      6-D gradient.
//!
//!   3. **Forward-mode 2nd order (`Dual2`)** — spot alone is seeded as
//!      the active variable; one forward pass yields value, delta, and
//!      spot-gamma (∂²C/∂S²) exactly, with no finite differences.
//!
//! All three modes are cross-checked against the closed-form analytical
//! Garman–Kohlhagen Greeks.
//!
//! Model:
//!     C = S · e^(−r_f·T) · N(d1)  −  K · e^(−r_d·T) · N(d2)
//!     P = K · e^(−r_d·T) · N(−d2) − S · e^(−r_f·T) · N(−d1)
//!     d1 = [ln(S/K) + (r_d − r_f + ½σ²)·T] / (σ·√T)
//!     d2 = d1 − σ·√T
//! with the standard normal CDF expressed via erf:
//!     N(x) = ½·(1 + erf(x / √2))

use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::hint::black_box;
use std::time::Instant;

use xad_rs::AReal;
use xad_rs::Dual;
use xad_rs::Dual2;
use xad_rs::math::{ad, erf};
use xad_rs::Tape;

/// Number of timed repetitions per mode.
const N_TRIALS: usize = 10_000;

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // `Put` is part of the demonstration API
enum Side {
    Call,
    Put,
}

// ============================================================================
// Plain-f64 reference implementation
// ============================================================================

/// Standard normal CDF: `N(x) = ½·(1 + erf(x/√2))`.
#[inline]
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

/// Standard normal PDF: `φ(x) = exp(−x²/2) / √(2π)`.
#[inline]
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Closed-form Garman–Kohlhagen price for a European FX option.
fn price_gk_f64(
    side: Side,
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r_d: f64,
    r_f: f64,
) -> f64 {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * sqrt_t;
    let d1 = ((s / k).ln() + (r_d - r_f + 0.5 * sigma * sigma) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    let df_d = (-r_d * t).exp();
    let df_f = (-r_f * t).exp();
    match side {
        Side::Call => s * df_f * norm_cdf(d1) - k * df_d * norm_cdf(d2),
        Side::Put => k * df_d * norm_cdf(-d2) - s * df_f * norm_cdf(-d1),
    }
}

/// Closed-form Greeks for a Garman–Kohlhagen **call**.
///
/// Returns `(delta=∂C/∂S, dK=∂C/∂K, dT=∂C/∂T, vega=∂C/∂σ, rho_d=∂C/∂r_d,
/// rho_f=∂C/∂r_f, gamma=∂²C/∂S²)`.
///
/// Note: `dT` is the raw derivative with respect to time-to-expiry. The
/// conventional "theta" quoted by traders is `−dT`.
fn analytic_greeks_call(
    s: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r_d: f64,
    r_f: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * sqrt_t;
    let d1 = ((s / k).ln() + (r_d - r_f + 0.5 * sigma * sigma) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    let df_d = (-r_d * t).exp();
    let df_f = (-r_f * t).exp();
    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let phi_d1 = norm_pdf(d1);

    let delta = df_f * nd1;
    let d_k = -df_d * nd2;
    // ∂C/∂T  =  −r_f·S·D_f·N(d1) + r_d·K·D_d·N(d2) + S·D_f·φ(d1)·σ/(2√T)
    let d_t = -r_f * s * df_f * nd1
        + r_d * k * df_d * nd2
        + s * df_f * phi_d1 * sigma / (2.0 * sqrt_t);
    let vega = s * df_f * phi_d1 * sqrt_t;
    let rho_d = k * t * df_d * nd2;
    let rho_f = -s * t * df_f * nd1;
    let gamma = df_f * phi_d1 / (s * sigma * sqrt_t);

    (delta, d_k, d_t, vega, rho_d, rho_f, gamma)
}

// ============================================================================
// Forward-mode full gradient (multi-variable `Dual`)
// ============================================================================

/// Prices the option using `Dual`; every operator in the chain propagates
/// tangents for all 6 inputs simultaneously.
fn price_gk_dual(
    side: Side,
    s: &Dual,
    k: &Dual,
    t: &Dual,
    sigma: &Dual,
    r_d: &Dual,
    r_f: &Dual,
) -> Dual {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * &sqrt_t;

    let sigma_sq = sigma * sigma;
    let rate_diff = r_d - r_f;
    let drift = rate_diff + sigma_sq * 0.5;

    let ln_s_k = (s / k).ln();
    let d1 = (ln_s_k + &drift * t) / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;

    let df_d = (-(r_d * t)).exp();
    let df_f = (-(r_f * t)).exp();

    // N(x) = ½·(1 + erf(x/√2))
    let e1 = (&d1 * FRAC_1_SQRT_2).erf();
    let e2 = (&d2 * FRAC_1_SQRT_2).erf();

    match side {
        Side::Call => {
            let nd1 = (e1 + 1.0) * 0.5;
            let nd2 = (e2 + 1.0) * 0.5;
            s * &df_f * nd1 - k * &df_d * nd2
        }
        Side::Put => {
            let n_neg_d1 = (1.0 - e1) * 0.5;
            let n_neg_d2 = (1.0 - e2) * 0.5;
            k * &df_d * n_neg_d2 - s * &df_f * n_neg_d1
        }
    }
}

// ============================================================================
// Reverse-mode full gradient (`AReal` on a tape)
// ============================================================================

/// Prices the option using `AReal`; every operator records an entry on the
/// active tape so a single reverse sweep yields the full 6-D gradient.
fn price_gk_areal(
    side: Side,
    s: &AReal<f64>,
    k: &AReal<f64>,
    t: &AReal<f64>,
    sigma: &AReal<f64>,
    r_d: &AReal<f64>,
    r_f: &AReal<f64>,
) -> AReal<f64> {
    let sqrt_t = ad::sqrt(t);
    let vol_sqrt_t = sigma * &sqrt_t;

    let sigma_sq = sigma * sigma;
    let rate_diff = r_d - r_f;
    let drift = rate_diff + sigma_sq * 0.5;

    let s_over_k = s / k;
    let ln_s_k = ad::ln(&s_over_k);
    let d1 = (ln_s_k + drift * t) / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;

    let rd_t = r_d * t;
    let rf_t = r_f * t;
    let df_d = ad::exp(&(-rd_t));
    let df_f = ad::exp(&(-rf_t));

    let d1_scaled = &d1 * FRAC_1_SQRT_2;
    let d2_scaled = &d2 * FRAC_1_SQRT_2;
    let e1 = ad::erf(&d1_scaled);
    let e2 = ad::erf(&d2_scaled);

    match side {
        Side::Call => {
            let nd1 = (e1 + 1.0) * 0.5;
            let nd2 = (e2 + 1.0) * 0.5;
            s * &df_f * nd1 - k * &df_d * nd2
        }
        Side::Put => {
            let n_neg_d1 = (1.0 - e1) * 0.5;
            let n_neg_d2 = (1.0 - e2) * 0.5;
            k * &df_d * n_neg_d2 - s * &df_f * n_neg_d1
        }
    }
}

// ============================================================================
// Forward-mode second order (`Dual2`)
// ============================================================================

/// Prices the option using `Dual2`; by seeding *only spot* as the active
/// variable, one forward pass yields (value, delta, ∂²/∂S²) exactly.
fn price_gk_dual2(
    side: Side,
    s: Dual2<f64>,
    k: Dual2<f64>,
    t: Dual2<f64>,
    sigma: Dual2<f64>,
    r_d: Dual2<f64>,
    r_f: Dual2<f64>,
) -> Dual2<f64> {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * sqrt_t;

    let sigma_sq = sigma * sigma;
    let rate_diff = r_d - r_f;
    let drift = rate_diff + sigma_sq * 0.5;

    let ln_s_k = (s / k).ln();
    let d1 = (ln_s_k + drift * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;

    let df_d = (-(r_d * t)).exp();
    let df_f = (-(r_f * t)).exp();

    let e1 = (d1 * FRAC_1_SQRT_2).erf();
    let e2 = (d2 * FRAC_1_SQRT_2).erf();

    match side {
        Side::Call => {
            let nd1 = (e1 + 1.0) * 0.5;
            let nd2 = (e2 + 1.0) * 0.5;
            s * df_f * nd1 - k * df_d * nd2
        }
        Side::Put => {
            let n_neg_d1 = (1.0 - e1) * 0.5;
            let n_neg_d2 = (1.0 - e2) * 0.5;
            k * df_d * n_neg_d2 - s * df_f * n_neg_d1
        }
    }
}

// ============================================================================
// Main: price, differentiate, cross-check, time
// ============================================================================

fn main() {
    // -------- Sample FX option (EUR/USD 6-month, ATM-ish) --------
    let side = Side::Call;
    let s_val: f64 = 1.0850; // spot
    let k_val: f64 = 1.1000; // strike
    let t_val: f64 = 0.50; // years to expiry
    let sigma_val: f64 = 0.085; // vol
    let r_d_val: f64 = 0.0450; // domestic (USD) rate
    let r_f_val: f64 = 0.0350; // foreign  (EUR) rate

    // -------- Closed-form reference --------
    let price_analytic = price_gk_f64(side, s_val, k_val, t_val, sigma_val, r_d_val, r_f_val);
    let (delta_a, dk_a, dt_a, vega_a, rho_d_a, rho_f_a, gamma_a) =
        analytic_greeks_call(s_val, k_val, t_val, sigma_val, r_d_val, r_f_val);

    // ================================================================
    // Pass 1 — Forward-mode full gradient via `Dual` (single 6-D pass)
    // ================================================================
    //
    // Input seeds are reused across trials — for a Dual forward pass the
    // only work per trial is the pricing expression itself.

    let n_vars = 6;
    let s_d = Dual::variable(s_val, 0, n_vars);
    let k_d = Dual::variable(k_val, 1, n_vars);
    let t_d = Dual::variable(t_val, 2, n_vars);
    let sigma_d = Dual::variable(sigma_val, 3, n_vars);
    let r_d_d = Dual::variable(r_d_val, 4, n_vars);
    let r_f_d = Dual::variable(r_f_val, 5, n_vars);

    let mut price_fwd = 0.0_f64;
    let mut grad_fwd = [0.0_f64; 6];

    let t_fwd = Instant::now();
    for _ in 0..N_TRIALS {
        // `black_box` the inputs so the optimiser can't hoist the whole
        // (constant) pricing expression out of the loop.
        let v = price_gk_dual(
            side,
            black_box(&s_d),
            black_box(&k_d),
            black_box(&t_d),
            black_box(&sigma_d),
            black_box(&r_d_d),
            black_box(&r_f_d),
        );
        price_fwd = v.real();
        grad_fwd.copy_from_slice(v.dual());
        black_box(&grad_fwd);
    }
    let fwd_elapsed = t_fwd.elapsed();
    let fwd_avg = fwd_elapsed / N_TRIALS as u32;

    // ================================================================
    // Pass 2 — Reverse-mode full gradient via `AReal` + fresh tape
    // ================================================================

    let mut price_rev = 0.0_f64;
    let mut grad_rev = [0.0_f64; 6];
    let mut tape_num_ops = 0usize;
    let mut tape_mem = 0usize;

    let t_rev = Instant::now();
    for _ in 0..N_TRIALS {
        let mut tape = Tape::<f64>::new(true);
        tape.activate();

        let mut inputs = [
            AReal::new(s_val),
            AReal::new(k_val),
            AReal::new(t_val),
            AReal::new(sigma_val),
            AReal::new(r_d_val),
            AReal::new(r_f_val),
        ];
        AReal::register_input(&mut inputs, &mut tape);

        let mut v = price_gk_areal(
            side, &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5],
        );

        AReal::register_output(std::slice::from_mut(&mut v), &mut tape);
        v.set_adjoint(&mut tape, 1.0);
        tape.compute_adjoints();

        price_rev = v.value();
        for (i, x) in inputs.iter().enumerate() {
            grad_rev[i] = x.adjoint(&tape);
        }

        tape_num_ops = tape.num_operations();
        tape_mem = tape.memory();

        Tape::<f64>::deactivate_all();
        drop(tape);
        black_box(&grad_rev);
    }
    let rev_elapsed = t_rev.elapsed();
    let rev_avg = rev_elapsed / N_TRIALS as u32;

    // ================================================================
    // Pass 3 — Spot-gamma via `Dual2` (one forward pass)
    // ================================================================

    let mut price_d2 = 0.0_f64;
    let mut delta_d2 = 0.0_f64;
    let mut gamma_d2 = 0.0_f64;

    let t_d2 = Instant::now();
    for _ in 0..N_TRIALS {
        // `black_box` the inputs so the optimiser can't hoist the whole
        // (constant) pricing expression out of the loop.
        let sv = black_box(s_val);
        let kv = black_box(k_val);
        let tv = black_box(t_val);
        let sigv = black_box(sigma_val);
        let rdv = black_box(r_d_val);
        let rfv = black_box(r_f_val);
        let v = price_gk_dual2(
            side,
            Dual2::variable(sv),
            Dual2::constant(kv),
            Dual2::constant(tv),
            Dual2::constant(sigv),
            Dual2::constant(rdv),
            Dual2::constant(rfv),
        );
        price_d2 = v.value();
        delta_d2 = v.first_derivative();
        gamma_d2 = v.second_derivative();
        black_box(&gamma_d2);
    }
    let d2_elapsed = t_d2.elapsed();
    let d2_avg = d2_elapsed / N_TRIALS as u32;

    // ================================================================
    // Output
    // ================================================================

    println!("FX Option Pricer — Garman–Kohlhagen");
    println!("====================================");
    println!("Side         = {:?}", side);
    println!("S (spot)     = {}", s_val);
    println!("K (strike)   = {}", k_val);
    println!("T (years)    = {}", t_val);
    println!("σ (vol)      = {}", sigma_val);
    println!("r_d          = {}", r_d_val);
    println!("r_f          = {}", r_f_val);
    println!();
    println!("Price (analytic)  = {:.10}", price_analytic);
    println!("Price (Dual fwd)  = {:.10}", price_fwd);
    println!("Price (AReal rev) = {:.10}", price_rev);
    println!("Price (Dual2)     = {:.10}", price_d2);
    println!();

    let names = [
        "delta  (dC/dS)",
        "       dC/dK",
        "       dC/dT",
        "vega   (dC/dσ)",
        "rho_d  (dC/dr_d)",
        "rho_f  (dC/dr_f)",
    ];
    let ana = [delta_a, dk_a, dt_a, vega_a, rho_d_a, rho_f_a];

    println!(
        "  {:<18} {:>16} {:>16} {:>16}",
        "Greek", "Analytic", "Dual (fwd)", "AReal (rev)"
    );
    println!(
        "  {:-<18} {:->16} {:->16} {:->16}",
        "", "", "", ""
    );
    for i in 0..6 {
        println!(
            "  {:<18} {:>16.10} {:>16.10} {:>16.10}",
            names[i], ana[i], grad_fwd[i], grad_rev[i]
        );
    }
    println!();
    println!("Spot gamma (d²C/dS²):");
    println!("  analytic = {:.10}", gamma_a);
    println!("  Dual2    = {:.10}", gamma_d2);
    println!("  (Dual2 delta = {:.10}, matches delta column above)", delta_d2);

    // -------- Cross-check against the analytic reference --------
    let max_err_fwd = (0..6)
        .map(|i| (grad_fwd[i] - ana[i]).abs())
        .fold(0.0_f64, f64::max);
    let max_err_rev = (0..6)
        .map(|i| (grad_rev[i] - ana[i]).abs())
        .fold(0.0_f64, f64::max);
    let gamma_err = (gamma_d2 - gamma_a).abs();
    let price_err = (price_fwd - price_analytic)
        .abs()
        .max((price_rev - price_analytic).abs())
        .max((price_d2 - price_analytic).abs());

    println!();
    println!("Consistency:");
    println!("  max |price_AD − analytic|        = {:.2e}", price_err);
    println!("  max |grad_Dual  − analytic|      = {:.2e}", max_err_fwd);
    println!("  max |grad_AReal − analytic|      = {:.2e}", max_err_rev);
    println!("  |gamma_Dual2 − analytic|         = {:.2e}", gamma_err);

    // erf() uses the A&S 7.1.26 approximation (~1.5e-7 abs error), so the
    // AD–vs–analytic discrepancy lives at the same order of magnitude.
    let tol = 5e-6;
    assert!(
        price_err < tol,
        "AD price disagrees with analytic by {:.2e}",
        price_err
    );
    assert!(
        max_err_fwd < tol,
        "Forward-mode Greeks disagree with analytic by {:.2e}",
        max_err_fwd
    );
    assert!(
        max_err_rev < tol,
        "Reverse-mode Greeks disagree with analytic by {:.2e}",
        max_err_rev
    );
    assert!(
        gamma_err < tol,
        "Dual2 gamma disagrees with analytic by {:.2e}",
        gamma_err
    );

    // -------- Tape stats from the reverse-mode pass --------
    println!();
    println!("Reverse tape (AReal):");
    println!("  num_operations = {}", tape_num_ops);
    println!("  memory (bytes) = {}", tape_mem);

    // -------- Timings --------
    println!();
    println!("Calculation time  (averaged over {} trials):", N_TRIALS);
    println!(
        "  {:<46} {:>12}  {:>14}",
        "mode", "avg/iter", "total"
    );
    println!(
        "  {:-<46} {:->12}  {:->14}",
        "", "", ""
    );
    println!(
        "  {:<46} {:>12.3?}  {:>14.3?}",
        "Forward-mode full gradient (Dual, 6-D)", fwd_avg, fwd_elapsed
    );
    println!(
        "  {:<46} {:>12.3?}  {:>14.3?}",
        "Reverse-mode full gradient (AReal + tape)", rev_avg, rev_elapsed
    );
    println!(
        "  {:<46} {:>12.3?}  {:>14.3?}",
        "Forward-mode spot gamma (Dual2)", d2_avg, d2_elapsed
    );
}
