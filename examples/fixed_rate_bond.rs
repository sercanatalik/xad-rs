//! Fixed-rate bond pricer — classic yield / duration / convexity with AD.
//!
//! Prices a vanilla fixed-coupon bond under a flat yield-to-maturity (YTM)
//! assumption and derives its sensitivity metrics using three AD modes,
//! cross-checked against closed-form analytic expressions:
//!
//!   1. **Reverse-mode (`AReal` + tape)** — one tape pass gives `dP/dy`.
//!      Because the input is a single scalar, the reverse-mode cost is
//!      dominated by the tape build/sweep overhead.
//!
//!   2. **Forward single-variable (`FReal`)** — one tape-free forward
//!      pass gives `dP/dy`. This is the idiomatic choice for a
//!      single-input gradient in xad.
//!
//!   3. **Forward second-order (`Dual2`)** — one forward pass gives
//!      value, `dP/dy`, **and** `d²P/dy²` simultaneously. That's
//!      modified duration *and* convexity from a single evaluation,
//!      with no finite differences.
//!
//! Derived risk metrics:
//!     modified duration   D*   =  −(1/P)·(dP/dy)
//!     convexity            C   =   (1/P)·(d²P/dy²)
//!
//! Model (discrete compounding, constant periodic rate):
//!     P = Σ_{t=1..n} C·(1+y/f)^(−t)  +  F·(1+y/f)^(−n)
//!     C = F · c / f   (per-period coupon)
//!     r = y / f       (per-period yield)

use std::hint::black_box;
use std::time::Instant;

use xad_rs::AReal;
use xad_rs::Dual2;
use xad_rs::FReal;
use xad_rs::math::{ad, fwd};
use xad_rs::Tape;

/// Number of timed repetitions per mode.
const N_TRIALS: usize = 100_000;

/// Bond contract data.
#[derive(Clone, Copy)]
struct Bond {
    /// Par / face value returned at maturity.
    face: f64,
    /// Annual coupon rate (e.g. 0.05 for 5%).
    coupon_rate: f64,
    /// Total number of coupon periods to maturity.
    n_periods: usize,
    /// Coupon payments per year (e.g. 2 for semi-annual).
    freq: f64,
}

impl Bond {
    /// Per-period coupon cashflow.
    #[inline]
    fn coupon(&self) -> f64 {
        self.coupon_rate * self.face / self.freq
    }
}

// ============================================================================
// Plain-f64 reference pricer and closed-form sensitivities
// ============================================================================

/// Dirty price from the discounted cashflow sum.
fn price_bond_f64(b: Bond, y: f64) -> f64 {
    let r = y / b.freq;
    let c = b.coupon();
    let one_plus = 1.0 + r;
    let mut p = 0.0;
    for t in 1..=b.n_periods {
        p += c * one_plus.powi(-(t as i32));
    }
    p += b.face * one_plus.powi(-(b.n_periods as i32));
    p
}

/// Closed-form first and second yield derivatives:
///   dP/dy  = (1/f) · Σ_t [−t·CF_t·(1+r)^(−t−1)]
///   d²P/dy² = (1/f²) · Σ_t [t·(t+1)·CF_t·(1+r)^(−t−2)]
///
/// Returns `(dPdy, d2Pdy2)`.
fn analytic_derivs(b: Bond, y: f64) -> (f64, f64) {
    let r = y / b.freq;
    let c = b.coupon();
    let f = b.freq;
    let one_plus = 1.0 + r;

    let mut dpdr = 0.0_f64;
    let mut d2pdr2 = 0.0_f64;

    // Coupon flows t = 1..n
    for t in 1..=b.n_periods {
        let tf = t as f64;
        let df_tp1 = one_plus.powi(-(t as i32 + 1));
        let df_tp2 = one_plus.powi(-(t as i32 + 2));
        dpdr -= tf * c * df_tp1;
        d2pdr2 += tf * (tf + 1.0) * c * df_tp2;
    }
    // Principal at t = n
    let n = b.n_periods as f64;
    let df_np1 = one_plus.powi(-(b.n_periods as i32 + 1));
    let df_np2 = one_plus.powi(-(b.n_periods as i32 + 2));
    dpdr -= n * b.face * df_np1;
    d2pdr2 += n * (n + 1.0) * b.face * df_np2;

    // dP/dy = (1/f) dP/dr;  d²P/dy² = (1/f²) d²P/dr²
    (dpdr / f, d2pdr2 / (f * f))
}

// ============================================================================
// Reverse-mode pricer (AReal)
// ============================================================================

fn price_bond_areal(b: Bond, y: &AReal<f64>) -> AReal<f64> {
    let c = b.coupon();
    let r = y / b.freq; // &AReal / f64
    let one_plus = r + 1.0; // AReal + f64
    let mut p = AReal::new(0.0);
    for t in 1..=b.n_periods {
        let df = ad::powi(&one_plus, -(t as i32));
        p += c * &df;
    }
    let df_n = ad::powi(&one_plus, -(b.n_periods as i32));
    p += b.face * &df_n;
    p
}

// ============================================================================
// Forward single-variable pricer (FReal)
// ============================================================================

fn price_bond_freal(b: Bond, y: &FReal<f64>) -> FReal<f64> {
    let c = b.coupon();
    let r = y / b.freq;
    let one_plus = r + 1.0;
    let mut p = FReal::constant(0.0);
    for t in 1..=b.n_periods {
        let df = fwd::powi(&one_plus, -(t as i32));
        p += c * &df;
    }
    let df_n = fwd::powi(&one_plus, -(b.n_periods as i32));
    p += b.face * &df_n;
    p
}

// ============================================================================
// Forward second-order pricer (Dual2)
// ============================================================================
//
// Seed y as a `Dual2::variable`; one forward pass yields value, dP/dy,
// and d²P/dy² in the returned `Dual2`.

fn price_bond_dual2(b: Bond, y: Dual2<f64>) -> Dual2<f64> {
    let c = b.coupon();
    let r = y / b.freq; // Dual2 / f64
    let one_plus = r + 1.0;
    let mut p = Dual2::constant(0.0);
    for t in 1..=b.n_periods {
        let df = one_plus.powf(-(t as f64));
        p += c * df;
    }
    let df_n = one_plus.powf(-(b.n_periods as f64));
    p += b.face * df_n;
    p
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    // 10-year 5% semi-annual coupon bond, face 1000, yielded at 4% YTM.
    let bond = Bond {
        face: 1_000.0,
        coupon_rate: 0.05,
        n_periods: 20, // 10 years × 2 coupons/yr
        freq: 2.0,
    };
    let y_val: f64 = 0.04;

    // -------- Closed-form reference --------
    let price_ref = price_bond_f64(bond, y_val);
    let (dpdy_ref, d2pdy2_ref) = analytic_derivs(bond, y_val);
    let mod_dur_ref = -dpdy_ref / price_ref;
    let conv_ref = d2pdy2_ref / price_ref;

    // ==================================================================
    // Pass 1 — Reverse-mode dP/dy via AReal + tape
    // ==================================================================

    let mut price_rev = 0.0_f64;
    let mut dpdy_rev = 0.0_f64;
    let mut tape_ops = 0usize;
    let mut tape_mem = 0usize;

    let t_rev = Instant::now();
    for _ in 0..N_TRIALS {
        let mut tape = Tape::<f64>::new(true);
        tape.activate();

        let mut y_ad = AReal::new(black_box(y_val));
        AReal::register_input(std::slice::from_mut(&mut y_ad), &mut tape);

        let mut p = price_bond_areal(bond, &y_ad);
        AReal::register_output(std::slice::from_mut(&mut p), &mut tape);
        p.set_adjoint(&mut tape, 1.0);
        tape.compute_adjoints();

        price_rev = p.value();
        dpdy_rev = y_ad.adjoint(&tape);

        tape_ops = tape.num_operations();
        tape_mem = tape.memory();

        Tape::<f64>::deactivate_all();
        drop(tape);
        black_box(&dpdy_rev);
    }
    let rev_elapsed = t_rev.elapsed();
    let rev_avg = rev_elapsed / N_TRIALS as u32;

    // ==================================================================
    // Pass 2 — Forward single-variable dP/dy via FReal (tape-free)
    // ==================================================================

    let mut price_fwd = 0.0_f64;
    let mut dpdy_fwd = 0.0_f64;

    let t_fwd = Instant::now();
    for _ in 0..N_TRIALS {
        // `FReal::new(value, 1.0)` seeds y as the active tangent direction.
        let y_fr = FReal::new(black_box(y_val), 1.0);
        let p = price_bond_freal(bond, &y_fr);
        price_fwd = p.value();
        dpdy_fwd = p.derivative();
        black_box(&dpdy_fwd);
    }
    let fwd_elapsed = t_fwd.elapsed();
    let fwd_avg = fwd_elapsed / N_TRIALS as u32;

    // ==================================================================
    // Pass 3 — Forward 2nd-order: value + dP/dy + d²P/dy² via Dual2
    // ==================================================================

    let mut price_d2 = 0.0_f64;
    let mut dpdy_d2 = 0.0_f64;
    let mut d2pdy2_d2 = 0.0_f64;

    let t_d2 = Instant::now();
    for _ in 0..N_TRIALS {
        let y_d2 = Dual2::variable(black_box(y_val));
        let p = price_bond_dual2(bond, y_d2);
        price_d2 = p.value();
        dpdy_d2 = p.first_derivative();
        d2pdy2_d2 = p.second_derivative();
        black_box(&d2pdy2_d2);
    }
    let d2_elapsed = t_d2.elapsed();
    let d2_avg = d2_elapsed / N_TRIALS as u32;

    // Derived metrics from each mode
    let mod_dur_rev = -dpdy_rev / price_rev;
    let mod_dur_fwd = -dpdy_fwd / price_fwd;
    let mod_dur_d2 = -dpdy_d2 / price_d2;
    let conv_d2 = d2pdy2_d2 / price_d2;

    // ==================================================================
    // Output
    // ==================================================================

    println!("Fixed-Rate Bond Pricer — YTM / Duration / Convexity");
    println!("====================================================");
    println!(
        "face         = {}    coupon rate  = {}",
        bond.face, bond.coupon_rate
    );
    println!(
        "n_periods    = {}     freq         = {} (semi-annual: 2)",
        bond.n_periods, bond.freq
    );
    println!("YTM (y)      = {}", y_val);
    println!();
    println!("Price (analytic)  = {:.8}", price_ref);
    println!("Price (AReal rev) = {:.8}", price_rev);
    println!("Price (FReal fwd) = {:.8}", price_fwd);
    println!("Price (Dual2)     = {:.8}", price_d2);
    println!();

    println!(
        "  {:<22} {:>18} {:>18} {:>18} {:>18}",
        "metric", "analytic", "AReal (rev)", "FReal (fwd)", "Dual2 (fwd²)"
    );
    println!(
        "  {:-<22} {:->18} {:->18} {:->18} {:->18}",
        "", "", "", "", ""
    );
    println!(
        "  {:<22} {:>18.10} {:>18.10} {:>18.10} {:>18.10}",
        "dP/dy", dpdy_ref, dpdy_rev, dpdy_fwd, dpdy_d2
    );
    println!(
        "  {:<22} {:>18.10} {:>18} {:>18} {:>18.10}",
        "d²P/dy²", d2pdy2_ref, "— (1st only)", "— (1st only)", d2pdy2_d2
    );
    println!(
        "  {:<22} {:>18.10} {:>18.10} {:>18.10} {:>18.10}",
        "modified duration", mod_dur_ref, mod_dur_rev, mod_dur_fwd, mod_dur_d2
    );
    println!(
        "  {:<22} {:>18.10} {:>18} {:>18} {:>18.10}",
        "convexity", conv_ref, "— (1st only)", "— (1st only)", conv_d2
    );

    // -------- Consistency checks --------
    let price_err = (price_rev - price_ref)
        .abs()
        .max((price_fwd - price_ref).abs())
        .max((price_d2 - price_ref).abs());
    let dpdy_err = (dpdy_rev - dpdy_ref)
        .abs()
        .max((dpdy_fwd - dpdy_ref).abs())
        .max((dpdy_d2 - dpdy_ref).abs());
    let d2_err = (d2pdy2_d2 - d2pdy2_ref).abs();

    println!();
    println!("Consistency:");
    println!("  max |price_AD − analytic|    = {:.2e}", price_err);
    println!("  max |dP/dy_AD − analytic|    = {:.2e}", dpdy_err);
    println!("  |d²P/dy²_Dual2 − analytic|   = {:.2e}", d2_err);

    // All three AD modes follow exactly the same discrete-sum pricing
    // recipe as the analytic code, so agreement should be at machine
    // precision (a few ulps).
    let tol = 1e-9;
    assert!(price_err < tol, "price disagreement: {:.2e}", price_err);
    assert!(dpdy_err < tol, "dP/dy disagreement: {:.2e}", dpdy_err);
    assert!(d2_err < tol, "d²P/dy² disagreement: {:.2e}", d2_err);

    // -------- Tape stats --------
    println!();
    println!("Reverse tape (AReal):");
    println!("  num_operations = {}", tape_ops);
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
        "Reverse-mode dP/dy (AReal + tape)", rev_avg, rev_elapsed
    );
    println!(
        "  {:<46} {:>12.3?}  {:>14.3?}",
        "Forward-mode dP/dy (FReal, tape-free)", fwd_avg, fwd_elapsed
    );
    println!(
        "  {:<46} {:>12.3?}  {:>14.3?}",
        "Forward 2nd-order dP/dy + d²P/dy² (Dual2)", d2_avg, d2_elapsed
    );
}
