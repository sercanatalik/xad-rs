//! Swap Pricer - Rust port of XAD's SwapPricer sample, extended with gamma.
//!
//! Computes the discount rate sensitivities of a simple interest rate swap
//! using three modes, and times each of them:
//!
//!   1. **Delta** (first-order) via reverse-mode (adjoint) AD — one tape pass
//!      produces the full gradient in the C++ sample's style.
//!
//!   2. **Delta** (first-order) via *forward-mode only* AD using the
//!      multi-variable `Dual` type — a **single** forward pass produces the
//!      entire gradient (all `dv/dr_i`) without any tape. This is the mode
//!      added to the sample to illustrate pure forward-mode delta.
//!
//!   3. **Gamma** (second-order, diagonal) via the dedicated `Dual2` forward
//!      type — one forward pass per rate produces both the first and second
//!      derivative w.r.t. that rate exactly (no finite differences).
//!
//! Original C++ source:
//!   <https://github.com/auto-differentiation/xad/blob/main/samples/SwapPricer>
//!
//! The swap is priced as the difference between the floating-leg bond value
//! and the fixed-leg bond value (from the payer's perspective when
//! `is_fixed_pay = true`). Each cashflow is discounted by
//!     1 / (1 + r_t)^mat_t
//! using a per-tenor discount rate `r_t`.
//!
//! ## Performance notes
//!
//! The pricer is structured so that the per-tenor discount factor
//! `(1 + r_t)^mat_t` is computed **once** and reused by both the fixed and
//! floating legs. Splitting the two legs into independent loops (as in the
//! original C++ sample) doubles the expensive `powf` + division count and
//! bloats the AD tape / forward tangent traffic by ~2x. Sharing the
//! discount factor is a pure algorithmic win that benefits all three modes
//! proportionally.

use std::hint::black_box;
use std::time::Instant;

use xad_rs::AReal;
use xad_rs::Dual;
use xad_rs::Dual2;
use xad_rs::math;
use xad_rs::{Tape, TapeStorage};

/// Number of timed repetitions of each pricing pass.
const N_TRIALS: usize = 1000;

/// Prices a simple interest-rate swap (reverse-mode AD version).
///
/// Discount factors `(1 + r_t)^mat_t` are computed once per tenor and shared
/// between the fixed and floating legs — this halves the number of expensive
/// AD `powf` + division ops vs. the naive "one loop per leg" formulation and
/// directly shrinks the tape.
fn price_swap<T: TapeStorage>(
    disc_rates: &[AReal<T>],
    is_fixed_pay: bool,
    maturities: &[T],
    float_rates: &[T],
    fixed_rate: T,
    face_value: T,
) -> AReal<T> {
    let n = disc_rates.len();
    assert_eq!(n, maturities.len());
    assert_eq!(n, float_rates.len());

    let one = T::one();
    let fixed_cashflow = face_value * fixed_rate;

    // Shared discount factors: discounts[t] = (1 + r_t)^mat_t
    let discounts: Vec<AReal<T>> = (0..n)
        .map(|t| {
            let base = &disc_rates[t] + one;
            math::ad::powf(&base, maturities[t])
        })
        .collect();

    // Coupon legs (one division per tenor, per leg, using shared discounts).
    let mut b_fix = AReal::<T>::new(T::zero());
    let mut b_flt = AReal::<T>::new(T::zero());
    for t in 0..n {
        b_fix += AReal::new(fixed_cashflow) / &discounts[t];
        b_flt += AReal::new(face_value * float_rates[t]) / &discounts[t];
    }

    // Principal repayment at maturity (same discount factor on both legs).
    let last_discount = &discounts[n - 1];
    b_fix += AReal::new(face_value) / last_discount;
    b_flt += AReal::new(face_value) / last_discount;

    if is_fixed_pay {
        b_flt - b_fix
    } else {
        b_fix - b_flt
    }
}

/// Prices a simple interest-rate swap using `Dual2<f64>` for second-order
/// forward-mode differentiation.
///
/// Uses the same discount-factor sharing trick as the reverse-mode pricer.
///
/// The returned `Dual2` carries:
///   - `value()`              = swap value `v`
///   - `first_derivative()`   = `dv/dr_i` for the `Dual2::variable`-seeded rate
///   - `second_derivative()`  = `d²v/dr_i²` for the same rate (diagonal gamma)
fn price_swap_dual2(
    disc_rates: &[Dual2<f64>],
    is_fixed_pay: bool,
    maturities: &[f64],
    float_rates: &[f64],
    fixed_rate: f64,
    face_value: f64,
) -> Dual2<f64> {
    let n = disc_rates.len();
    assert_eq!(n, maturities.len());
    assert_eq!(n, float_rates.len());

    let fixed_cashflow = face_value * fixed_rate;

    // Shared discount factors: discounts[t] = (1 + r_t)^mat_t
    let discounts: Vec<Dual2<f64>> = (0..n)
        .map(|t| {
            let base = disc_rates[t] + 1.0;
            base.powf(maturities[t])
        })
        .collect();

    let mut b_fix = Dual2::constant(0.0);
    let mut b_flt = Dual2::constant(0.0);
    for t in 0..n {
        b_fix += Dual2::constant(fixed_cashflow) / discounts[t];
        b_flt += Dual2::constant(face_value * float_rates[t]) / discounts[t];
    }

    let last_discount = discounts[n - 1];
    b_fix += Dual2::constant(face_value) / last_discount;
    b_flt += Dual2::constant(face_value) / last_discount;

    if is_fixed_pay {
        b_flt - b_fix
    } else {
        b_fix - b_flt
    }
}

/// Prices a simple interest-rate swap using the multi-variable `Dual` type
/// for first-order **forward-mode** differentiation.
///
/// All `n` discount rates are seeded simultaneously as active variables
/// (`disc_rates[i] = Dual::variable(r_i, i, n)`), so a single forward pass
/// through this function produces the full gradient `dv/dr_i` for every `i`,
/// readable via `result.dual()` — no tape required.
///
/// Uses the same discount-factor sharing trick as the reverse-mode pricer.
fn price_swap_dual(
    disc_rates: &[Dual],
    is_fixed_pay: bool,
    maturities: &[f64],
    float_rates: &[f64],
    fixed_rate: f64,
    face_value: f64,
) -> Dual {
    let n = disc_rates.len();
    assert_eq!(n, maturities.len());
    assert_eq!(n, float_rates.len());

    let n_vars = disc_rates[0].num_vars();
    let fixed_cashflow = face_value * fixed_rate;

    // Shared discount factors: discounts[t] = (1 + r_t)^mat_t
    let discounts: Vec<Dual> = (0..n)
        .map(|t| {
            let base = &disc_rates[t] + 1.0;
            base.powf(maturities[t])
        })
        .collect();

    let mut b_fix = Dual::constant(0.0, n_vars);
    let mut b_flt = Dual::constant(0.0, n_vars);
    for t in 0..n {
        b_fix += fixed_cashflow / &discounts[t];
        b_flt += (face_value * float_rates[t]) / &discounts[t];
    }

    let last_discount = &discounts[n - 1];
    b_fix += face_value / last_discount;
    b_flt += face_value / last_discount;

    if is_fixed_pay {
        b_flt - b_fix
    } else {
        b_fix - b_flt
    }
}

/// Deterministic pseudo-random generator so runs are reproducible.
fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() {
    // -------- Initialize dummy input data --------
    let n_rates: usize = 30;
    let face_value: f64 = 10_000_000.0;
    let fixed_rate: f64 = 0.03;
    let is_fixed_pay = true;

    let mut seed: u64 = 0x5EED_1234_ABCD_0001;
    let mut float_rates: Vec<f64> = Vec::with_capacity(n_rates);
    let mut disc_rates: Vec<f64> = Vec::with_capacity(n_rates);
    let mut maturities: Vec<f64> = Vec::with_capacity(n_rates);
    for i in 0..n_rates {
        float_rates.push(0.01 + lcg(&mut seed) * 0.1);
        disc_rates.push(0.01 + lcg(&mut seed) * 0.06);
        maturities.push((i + 1) as f64);
    }

    // ========================================================================
    // Pass 1 — Delta via reverse-mode AD (AReal)
    // ========================================================================
    //
    // Runs the full "create tape → record → reverse-sweep → drop" cycle
    // `N_TRIALS` times. Results from the last iteration are kept for the
    // correctness checks and printed output.

    let mut value_adjoint = 0.0_f64;
    let mut deltas: Vec<f64> = vec![0.0; n_rates];
    let mut tape_num_vars = 0;
    let mut tape_num_stmts = 0;
    let mut tape_num_ops = 0;
    let mut tape_mem = 0;

    let t_rev = Instant::now();
    for _ in 0..N_TRIALS {
        let mut tape = Tape::<f64>::new(true);
        tape.activate();

        let mut disc_rates_ad: Vec<AReal<f64>> =
            disc_rates.iter().map(|&r| AReal::new(r)).collect();
        AReal::register_input(&mut disc_rates_ad, &mut tape);

        let mut v = price_swap(
            &disc_rates_ad,
            is_fixed_pay,
            &maturities,
            &float_rates,
            fixed_rate,
            face_value,
        );

        AReal::register_output(std::slice::from_mut(&mut v), &mut tape);
        v.set_adjoint(&mut tape, 1.0);
        tape.compute_adjoints();

        value_adjoint = v.value();
        for (i, r) in disc_rates_ad.iter().enumerate() {
            deltas[i] = r.adjoint(&tape);
        }

        tape_num_vars = tape.num_variables();
        tape_num_stmts = tape.num_statements();
        tape_num_ops = tape.num_operations();
        tape_mem = tape.memory();

        Tape::<f64>::deactivate_all();
        drop(tape);

        // Prevent the optimiser from dead-code-eliminating the trial.
        black_box(&deltas);
    }
    let rev_elapsed = t_rev.elapsed();
    let rev_avg = rev_elapsed / N_TRIALS as u32;

    // ========================================================================
    // Pass 2 — Delta via forward-mode-only AD (multi-variable `Dual`)
    // ========================================================================
    //
    // All `n_rates` discount rates are seeded as active variables of a single
    // `n_rates`-dimensional input space, so a **single** forward pass produces
    // the full gradient `dv/dr_i` for every `i`, with no tape.

    let mut value_fwd_delta = 0.0_f64;
    let mut deltas_fwd_only: Vec<f64> = vec![0.0; n_rates];

    let t_fwd_delta = Instant::now();
    for _ in 0..N_TRIALS {
        let disc_rates_d: Vec<Dual> = disc_rates
            .iter()
            .enumerate()
            .map(|(i, &r)| Dual::variable(r, i, n_rates))
            .collect();

        let v_d = price_swap_dual(
            &disc_rates_d,
            is_fixed_pay,
            &maturities,
            &float_rates,
            fixed_rate,
            face_value,
        );

        value_fwd_delta = v_d.real();
        deltas_fwd_only.copy_from_slice(v_d.dual());

        black_box(&deltas_fwd_only);
    }
    let fwd_delta_elapsed = t_fwd_delta.elapsed();
    let fwd_delta_avg = fwd_delta_elapsed / N_TRIALS as u32;

    // ========================================================================
    // Pass 3 — Gamma (diagonal) via Dual2 second-order forward mode
    // ========================================================================
    //
    // For each rate r_i, we seed `disc_rates_d2[i] = Dual2::variable(r_i)`
    // and leave all other rates as `Dual2::constant`. One forward pass then
    // yields (v, dv/dr_i, d²v/dr_i²). We also cross-check dv/dr_i against the
    // reverse-mode delta.

    let mut gammas: Vec<f64> = vec![0.0; n_rates];
    let mut deltas_fwd: Vec<f64> = vec![0.0; n_rates];
    let mut value_fwd = 0.0_f64;

    let t_fwd_gamma = Instant::now();
    for _ in 0..N_TRIALS {
        for i in 0..n_rates {
            let disc_rates_d2: Vec<Dual2<f64>> = disc_rates
                .iter()
                .enumerate()
                .map(|(k, &r)| {
                    if k == i {
                        Dual2::variable(r)
                    } else {
                        Dual2::constant(r)
                    }
                })
                .collect();

            let v_d2 = price_swap_dual2(
                &disc_rates_d2,
                is_fixed_pay,
                &maturities,
                &float_rates,
                fixed_rate,
                face_value,
            );

            if i == 0 {
                value_fwd = v_d2.value();
            }
            deltas_fwd[i] = v_d2.first_derivative();
            gammas[i] = v_d2.second_derivative();
        }
        black_box(&gammas);
    }
    let fwd_gamma_elapsed = t_fwd_gamma.elapsed();
    let fwd_gamma_avg = fwd_gamma_elapsed / N_TRIALS as u32;

    // ========================================================================
    // Output
    // ========================================================================

    println!("Swap Pricer - Rust port of XAD sample (with Gamma)");
    println!("==================================================");
    println!("n_rates      = {}", n_rates);
    println!("face_value   = {}", face_value);
    println!("fixed_rate   = {}", fixed_rate);
    println!("is_fixed_pay = {}", is_fixed_pay);
    println!();
    println!("Swap value   v (adjoint)    = {:.6}", value_adjoint);
    println!("Swap value   v (Dual fwd)   = {:.6}", value_fwd_delta);
    println!("Swap value   v (Dual2)      = {:.6}", value_fwd);
    println!();

    println!("Discount rate risk (1bp shift):");
    println!(
        "  {:<4} {:>16} {:>16} {:>16} {:>18}",
        "i", "DV01 (adjoint)", "DV01 (Dual)", "DV01 (Dual2)", "Gamma (d²v/dr²)"
    );
    println!(
        "  {:-<4} {:->16} {:->16} {:->16} {:->18}",
        "", "", "", "", ""
    );
    for i in 0..n_rates {
        let dv01_rev = deltas[i] * 0.0001;
        let dv01_fwd_only = deltas_fwd_only[i] * 0.0001;
        let dv01_fwd = deltas_fwd[i] * 0.0001;
        println!(
            "  {:<4} {:>16.6} {:>16.6} {:>16.6} {:>18.6}",
            i, dv01_rev, dv01_fwd_only, dv01_fwd, gammas[i]
        );
    }

    // -------- Aggregated risk --------
    let total_dv01: f64 = deltas.iter().map(|d| d * 0.0001).sum();
    let max_abs_gamma = gammas.iter().fold(0.0_f64, |m, &g| m.max(g.abs()));
    let trace_gamma: f64 = gammas.iter().sum(); // trace of the Hessian

    println!();
    println!("Total DV01         = {:.6}", total_dv01);
    println!("Trace(Hessian)     = {:.6}", trace_gamma);
    println!("Max |gamma_i|      = {:.6}", max_abs_gamma);

    // -------- Consistency checks --------
    let max_delta_err_dual2 = deltas
        .iter()
        .zip(deltas_fwd.iter())
        .fold(0.0_f64, |m, (a, f)| m.max((a - f).abs()));
    let max_delta_err_dual = deltas
        .iter()
        .zip(deltas_fwd_only.iter())
        .fold(0.0_f64, |m, (a, f)| m.max((a - f).abs()));
    println!(
        "Max |delta_adj - delta_dual |  = {:.2e}",
        max_delta_err_dual
    );
    println!(
        "Max |delta_adj - delta_dual2|  = {:.2e}",
        max_delta_err_dual2
    );
    assert!(
        max_delta_err_dual < 1e-8,
        "Forward Dual delta disagrees with adjoint delta: {}",
        max_delta_err_dual
    );
    assert!(
        max_delta_err_dual2 < 1e-8,
        "Dual2 delta disagrees with adjoint delta: {}",
        max_delta_err_dual2
    );

    // -------- Tape stats --------
    println!();
    println!("Reverse tape stats:");
    println!("  num_variables  = {}", tape_num_vars);
    println!("  num_statements = {}", tape_num_stmts);
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
        "Reverse-mode delta (AReal, 1 tape pass)",
        rev_avg,
        rev_elapsed
    );
    println!(
        "  {:<46} {:>12.3?}  {:>14.3?}",
        "Forward-mode delta (Dual, 1 forward pass)",
        fwd_delta_avg,
        fwd_delta_elapsed
    );
    println!(
        "  {:<46} {:>12.3?}  {:>14.3?}",
        format!("Forward-mode gamma (Dual2, {} fwd passes)", n_rates),
        fwd_gamma_avg,
        fwd_gamma_elapsed
    );
}
