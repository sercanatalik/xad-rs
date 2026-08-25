//! K-lane forward gradient vs. reverse mode — the crossover, measured.
//!
//! One body written once against `Real` is differentiated five ways and the
//! results are cross-checked against a closed-form gradient:
//!
//!   1. **Reverse, fresh tape** — `compute_gradient_rev`: one recording plus
//!      one sweep, allocating a tape per call.
//!   2. **Reverse, warm tape** — `compute_gradient_rev_with`: the same on a
//!      tape the caller keeps, so allocation is out of the loop.
//!   3. **`Jet1`, one pass per input** — `compute_directional_derivative_fwd`
//!      with a unit seed, `n` times.
//!   4. **`JetK<K>`, `⌈n/K⌉` passes** — `compute_gradient_fwd_k::<K, _>` at
//!      K = 4, 8 and 16: `K` tangent lanes evolve in one pass, no tape.
//!
//! Two bodies, so both sides of the crossover are printed:
//!
//!   - **Garman–Kohlhagen**, `n = 6` — inside one `JetK<8>` pass.
//!   - **A 30-tenor swap**, `n = 30` — four `JetK<8>` passes against one
//!     reverse sweep, the regime the README's `swap_pricer` figure lives in.
//!
//! Every arm's gradient is asserted against the closed form to `1e-10`, the
//! values are asserted bit-identical across arms, and the K-lane gradient is
//! asserted against `Jet1`'s and reverse mode's to a few ulp (both bodies
//! divide, so the lane-0 tangent is not bit-identical to `Jet1`'s — the two
//! modes accumulate a quotient's tangent in a different order).
//!
//! Timing follows `fx_option.rs`: `Instant` around `N_TRIALS` iterations,
//! averaged; that average is taken `REPEATS` times and the minimum reported,
//! which is the figure the README's "Pick a mode" paragraph cites.

use std::f64::consts::PI;
use std::hint::black_box;
use std::time::{Duration, Instant};

use xad_rs::{
    CopyableReal, Real, Tape, compute_directional_derivative_fwd, compute_gradient_fwd_k,
    compute_gradient_rev, compute_gradient_rev_with,
};

/// Timed iterations per measurement.
const N_TRIALS: usize = 10_000;
/// Measurements per arm; the minimum average is reported.
const REPEATS: usize = 5;

// ============================================================================
// Body 1 — Garman–Kohlhagen call, six inputs (S, K, T, σ, r_d, r_f)
// ============================================================================

/// The price, written once against the trait. `norm_cdf` is a `Real` method,
/// so the Gaussian goes through the crate's full-precision `erf`.
fn gk_call<R: CopyableReal<Passive = f64>>(s: R, k: R, t: R, sigma: R, r_d: R, r_f: R) -> R {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * sqrt_t;
    let drift = r_d - r_f + sigma * sigma * 0.5;
    let d1 = ((s / k).ln() + drift * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    let df_d = (-(r_d * t)).exp();
    let df_f = (-(r_f * t)).exp();
    s * df_f * d1.norm_cdf() - k * df_d * d2.norm_cdf()
}

/// Closed-form gradient `(∂/∂S, ∂/∂K, ∂/∂T, ∂/∂σ, ∂/∂r_d, ∂/∂r_f)`.
fn gk_gradient(p: &[f64]) -> Vec<f64> {
    let (s, k, t, sigma, r_d, r_f) = (p[0], p[1], p[2], p[3], p[4], p[5]);
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * sqrt_t;
    let d1 = ((s / k).ln() + (r_d - r_f + 0.5 * sigma * sigma) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    let df_d = (-r_d * t).exp();
    let df_f = (-r_f * t).exp();
    let nd1 = xad_rs::math::norm_cdf(d1);
    let nd2 = xad_rs::math::norm_cdf(d2);
    let phi_d1 = (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt();
    vec![
        df_f * nd1,
        -df_d * nd2,
        -r_f * s * df_f * nd1 + r_d * k * df_d * nd2 + s * df_f * phi_d1 * sigma / (2.0 * sqrt_t),
        s * df_f * phi_d1 * sqrt_t,
        k * t * df_d * nd2,
        -s * t * df_f * nd1,
    ]
}

// ============================================================================
// Body 2 — a 30-tenor swap, one discount rate per tenor
// ============================================================================

/// Fixed-pay swap: `Σ (float_i − fixed)·face / (1 + r_i)^{m_i}` plus the
/// principal exchange netting to zero — the shape of `swap_pricer.rs`'s body,
/// spelled against the trait. `powf` takes an active exponent, so the passive
/// maturity is lifted once per tenor.
fn swap_pv<R: CopyableReal<Passive = f64>>(rates: &[R], maturities: &[f64], coupons: &[f64]) -> R {
    let mut pv = R::zero();
    for ((&r, &m), &c) in rates.iter().zip(maturities).zip(coupons) {
        let discount = (r + 1.0).powf(R::from(m));
        pv = pv + discount.powi(-1) * c;
    }
    pv
}

/// `∂/∂r_i = −m_i·c_i·(1 + r_i)^{−m_i−1}`.
fn swap_gradient(rates: &[f64], maturities: &[f64], coupons: &[f64]) -> Vec<f64> {
    rates
        .iter()
        .zip(maturities)
        .zip(coupons)
        .map(|((&r, &m), &c)| -m * c * (1.0 + r).powf(-m - 1.0))
        .collect()
}

/// A body the harness can evaluate in every mode: the one generic method is
/// what lets one `arms` serve both bodies without a dispatch inside the
/// timed loop.
trait Body {
    fn eval<R: CopyableReal<Passive = f64>>(&self, v: &[R]) -> R;
}

struct Gk;

impl Body for Gk {
    fn eval<R: CopyableReal<Passive = f64>>(&self, v: &[R]) -> R {
        gk_call(v[0], v[1], v[2], v[3], v[4], v[5])
    }
}

struct Swap {
    maturities: Vec<f64>,
    coupons: Vec<f64>,
}

impl Body for Swap {
    fn eval<R: CopyableReal<Passive = f64>>(&self, v: &[R]) -> R {
        swap_pv(v, &self.maturities, &self.coupons)
    }
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
}

// ============================================================================
// Harness
// ============================================================================

struct Arm {
    name: &'static str,
    value: f64,
    grad: Vec<f64>,
    per_call: Duration,
}

/// Runs `f` `N_TRIALS` times, `REPEATS` times over, and keeps the fastest
/// average. `f` returns the `(value, gradient)` it computed, which is kept
/// from the last call for the cross-checks.
fn measure(name: &'static str, mut f: impl FnMut() -> (f64, Vec<f64>)) -> Arm {
    let mut best = Duration::MAX;
    let mut last = (0.0, Vec::new());
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..N_TRIALS {
            last = black_box(f());
        }
        best = best.min(start.elapsed() / N_TRIALS as u32);
    }
    Arm { name, value: last.0, grad: last.1, per_call: best }
}

/// Every arm of one body: reverse fresh/warm, `Jet1 × n`, `JetK` at 4/8/16.
fn arms<B: Body>(point: &[f64], body: &B) -> Vec<Arm> {
    let n = point.len();
    let mut warm = Tape::<f64>::new(true);
    vec![
        measure("Reverse, fresh tape (compute_gradient_rev)", || {
            compute_gradient_rev(black_box(point), |v| body.eval(v))
        }),
        measure("Reverse, warm tape (compute_gradient_rev_with)", || {
            compute_gradient_rev_with(&mut warm, black_box(point), |v| body.eval(v))
        }),
        measure("Jet1, one pass per input", || {
            let p = black_box(point);
            let mut value = 0.0;
            let mut seed = vec![0.0; n];
            let grad = (0..n)
                .map(|i| {
                    seed[i] = 1.0;
                    let (v, d) = compute_directional_derivative_fwd(p, &seed, |v| body.eval(v));
                    seed[i] = 0.0;
                    value = v;
                    d
                })
                .collect();
            (value, grad)
        }),
        measure("JetK<4>, ⌈n/4⌉ passes", || {
            compute_gradient_fwd_k::<4, _>(black_box(point), |v| body.eval(v))
        }),
        measure("JetK<8>, ⌈n/8⌉ passes", || {
            compute_gradient_fwd_k::<8, _>(black_box(point), |v| body.eval(v))
        }),
        measure("JetK<16>, ⌈n/16⌉ passes", || {
            compute_gradient_fwd_k::<16, _>(black_box(point), |v| body.eval(v))
        }),
    ]
}

const GK_POINT: [f64; 6] = [1.0850, 1.1000, 0.50, 0.085, 0.0450, 0.0350];

fn check(body: &str, arms: &[Arm], closed: &[f64], f64_value: f64) {
    let ulp = |a: f64, b: f64| 16.0 * f64::EPSILON * (1.0 + a.abs().max(b.abs()));
    for arm in arms {
        assert_eq!(arm.value, f64_value, "{body}: {} value is not the f64 value", arm.name);
        assert_eq!(arm.grad.len(), closed.len(), "{body}: {} gradient length", arm.name);
        for (i, (g, c)) in arm.grad.iter().zip(closed).enumerate() {
            let tol = 1e-10 * (1.0 + c.abs());
            assert!((g - c).abs() <= tol, "{body}: {} grad[{i}] = {g} vs closed form {c}", arm.name);
        }
    }
    // The K-lane arms agree with Jet1 and with reverse to a few ulp, not to
    // the bit: both bodies divide.
    let jet1 = &arms[2];
    let rev = &arms[1];
    for arm in &arms[3..] {
        for i in 0..closed.len() {
            assert!(
                (arm.grad[i] - jet1.grad[i]).abs() <= ulp(arm.grad[i], jet1.grad[i]),
                "{body}: {} grad[{i}] vs Jet1: {} vs {}", arm.name, arm.grad[i], jet1.grad[i]
            );
            assert!(
                (arm.grad[i] - rev.grad[i]).abs() <= ulp(arm.grad[i], rev.grad[i]),
                "{body}: {} grad[{i}] vs reverse: {} vs {}", arm.name, arm.grad[i], rev.grad[i]
            );
        }
    }
}

fn report(title: &str, n: usize, arms: &[Arm]) {
    println!();
    println!("{title}  (n = {n}, min of {REPEATS} averages over {N_TRIALS} calls)");
    println!("  {:<48} {:>12} {:>10}", "arm", "per call", "vs warm");
    println!("  {:-<48} {:->12} {:->10}", "", "", "");
    let warm = arms[1].per_call.as_secs_f64();
    for arm in arms {
        println!(
            "  {:<48} {:>12.3?} {:>9.2}×",
            arm.name,
            arm.per_call,
            warm / arm.per_call.as_secs_f64()
        );
    }
}

fn main() {
    // -------- Body 1: Garman–Kohlhagen, n = 6 --------
    let gk_arms = arms(&GK_POINT, &Gk);
    check("Garman–Kohlhagen", &gk_arms, &gk_gradient(&GK_POINT), Gk.eval(&GK_POINT));

    // -------- Body 2: 30-tenor swap, n = 30 (data as in `swap_pricer.rs`) --------
    let n = 30;
    let mut seed: u64 = 0x5EED_1234_ABCD_0001;
    let (face, fixed) = (10_000_000.0, 0.03);
    let mut rates = Vec::with_capacity(n);
    let mut coupons = Vec::with_capacity(n);
    for _ in 0..n {
        let float = 0.01 + lcg(&mut seed) * 0.1;
        rates.push(0.01 + lcg(&mut seed) * 0.06);
        coupons.push((float - fixed) * face);
    }
    let swap = Swap { maturities: (1..=n).map(|i| i as f64).collect(), coupons };
    let swap_arms = arms(&rates, &swap);
    check(
        "30-tenor swap",
        &swap_arms,
        &swap_gradient(&rates, &swap.maturities, &swap.coupons),
        swap.eval(&rates),
    );

    println!("K-lane forward gradient vs. reverse mode");
    println!("========================================");
    println!("Garman–Kohlhagen value = {:.10}", gk_arms[0].value);
    println!("30-tenor swap value    = {:.4}", swap_arms[0].value);
    println!();
    println!("Every arm's gradient matches the closed form to 1e-10; values are");
    println!("bit-identical across arms; K-lane vs Jet1 and vs reverse agree to");
    println!("a few ulp (both bodies divide).");
    report("Garman–Kohlhagen", 6, &gk_arms);
    report("30-tenor swap", 30, &swap_arms);
    println!();
    println!("Reading the table: `vs warm` is the warm-tape reverse time divided by");
    println!("the arm's time, so > 1× means the arm beats reverse mode.");
}
