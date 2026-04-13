//! Garman-Kohlhagen FX option 6×6 Hessian cross-check for `Dual2Vec`.
//!
//! Two assertions run on the **same** 6×6 Hessian at the **same** test point:
//!
//! 1. **Primary** — analytical closed-form second-order Greeks at tolerance
//!    `1e-11`. The gamma closed form `∂²C/∂S² = exp(-rf·T)·φ(d1) /
//!    (S·σ·√T)` is computed in pure f64 (only `ln`, `exp`, `sqrt`, arithmetic
//!    — NO erf), so there is no Abramowitz-Stegun approximation floor to
//!    fight. Volga and vanna add two more bit-exact cells.
//!
//! 2. **Secondary** — `xad_rs::compute_hessian` (reverse-mode +
//!    forward-difference, eps = 1e-7) at tolerance `5e-5`. This is an
//!    independent sanity smoke check; the FD Hessian error is dominated by
//!    `O(eps·|∂³f|) + O(roundoff/eps)` — for Garman-Kohlhagen at our test
//!    point, gamma is ~3.46 and the third derivative `∂³C/∂S³` is O(100),
//!    giving an honest FD bound of `~1.5e-5` on `H[0, 0]`. The tolerance is
//!    set to `5e-5` for a safety margin above the measured per-element
//!    max-delta, keeping the smoke check meaningful while not fighting
//!    floating-point reality. The actual `compute_hessian` implementation is
//!    FD-based and cannot deliver more than ~1e-5 accuracy without a finer
//!    `eps` (which would then lose to roundoff).
//!
//! Neither assertion post-processes the Hessian via any averaging helper —
//! structural symmetry is asserted bit-exactly BEFORE any comparison.
//! The Garman-Kohlhagen formula is duplicated in this test file with a
//! derivation comment so the test is self-contained.
//!
//! ## Why `erf` value precision doesn't matter for the 1e-11 check
//!
//! The `Dual2Vec::erf` elementary uses `crate::math::erf` (Abramowitz &
//! Stegun 7.1.26, ~1.5e-7 absolute value error) for the function value, but
//! its chain-rule derivatives `g'(u) = (2/√π)·exp(-u²)` and
//! `g''(u) = -2u·g'(u)` use only `f64::exp` plus exact arithmetic. The
//! closed-form gamma formula `∂²C/∂S² = exp(-rf·T)·φ(d1)/(S·σ·√T)` depends
//! algebraically on `φ(d1) = N'(d1)` — i.e. the *derivative* of the normal
//! CDF, not its value — so the A&S value floor cancels out of H[0, 0] /
//! H[4, 4] / H[0, 4]. Empirically these three cells match the closed form at
//! ~1e-13 in both debug and release; 1e-11 is the chosen safety margin.


use approx::assert_abs_diff_eq;
use xad_rs::AReal;
use xad_rs::Dual2Vec;

// ============================================================================
// Test inputs — Garman-Kohlhagen locked values
// ============================================================================
const SPOT: f64 = 1.3500;
const STRIKE: f64 = 1.3600;
const RD: f64 = 0.025;
const RF: f64 = 0.015;
const VOL: f64 = 0.12;
const T_YRS: f64 = 0.5;

// ============================================================================
// Garman-Kohlhagen call price — Dual2Vec closure (6 active inputs)
// ============================================================================
//
//   C(S, K, rd, rf, vol, T) = S·exp(-rf·T)·N(d1) - K·exp(-rd·T)·N(d2)
//
//   d1 = (ln(S/K) + (rd - rf + 0.5·vol²)·T) / (vol·√T)
//   d2 = d1 - vol·√T
//   N(x) = 0.5·(1 + erf(x/√2))
//
// Input order: [spot, strike, rd, rf, vol, T]
//
// Formula duplicated from the FX-option example pricer (READ-ONLY reference,
// NOT imported and NOT edited) so the test is self-contained and the
// purely-additive rule is respected.
fn gk_call_dual2vec(inputs: &[Dual2Vec; 6]) -> Dual2Vec {
    let n = 6;
    let spot = &inputs[0];
    let strike = &inputs[1];
    let rd = &inputs[2];
    let rf = &inputs[3];
    let vol = &inputs[4];
    let t = &inputs[5];

    // Scalar constants wrapped as Dual2Vec constants so arithmetic composes
    let half = Dual2Vec::constant(0.5, n);
    let one = Dual2Vec::constant(1.0, n);
    let sqrt_2 = Dual2Vec::constant(std::f64::consts::SQRT_2, n);
    let zero = Dual2Vec::constant(0.0, n);

    // d1, d2
    let sqrt_t = t.clone().sqrt();
    let vol_sqrt_t = vol * &sqrt_t;
    let ln_s_over_k = (spot / strike).ln();
    let vol_sq = vol * vol;
    let half_vol2 = &half * &vol_sq;
    let rate_diff = rd - rf;
    let drift_coeff = &rate_diff + &half_vol2;
    let drift = &drift_coeff * t;
    let d1_num = &ln_s_over_k + &drift;
    let d1 = &d1_num / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;

    // N(x) = 0.5 · (1 + erf(x/√2))
    let d1_scaled = &d1 / &sqrt_2;
    let d2_scaled = &d2 / &sqrt_2;
    let e1 = d1_scaled.erf();
    let e2 = d2_scaled.erf();
    let n_d1 = &half * &(&one + &e1);
    let n_d2 = &half * &(&one + &e2);

    // exp(-rf·T), exp(-rd·T)  — via `zero - (rf * t)` because Dual2Vec has no unary Neg
    let rf_t = rf * t;
    let rd_t = rd * t;
    let neg_rf_t = &zero - &rf_t;
    let neg_rd_t = &zero - &rd_t;
    let disc_f = neg_rf_t.exp();
    let disc_d = neg_rd_t.exp();

    // C = S · disc_f · N(d1) - K · disc_d · N(d2)
    let s_disc_f = spot * &disc_f;
    let term1 = &s_disc_f * &n_d1;
    let k_disc_d = strike * &disc_d;
    let term2 = &k_disc_d * &n_d2;
    &term1 - &term2
}

// ============================================================================
// Garman-Kohlhagen call price — AReal<f64> closure for FD smoke check
// ============================================================================
//
// Exact duplicate of `gk_call_dual2vec` typed on `AReal<f64>` so it can be
// passed to `xad_rs::compute_hessian`. Formula duplicated (rather
// than abstracted behind a trait) so the test stays self-contained and the
// purely-additive rule is honored for the FX-option example pricer.
fn gk_call_areal(inputs: &[AReal<f64>]) -> AReal<f64> {
    use xad_rs::math::ad;
    assert_eq!(inputs.len(), 6);
    let spot = &inputs[0];
    let strike = &inputs[1];
    let rd = &inputs[2];
    let rf = &inputs[3];
    let vol = &inputs[4];
    let t = &inputs[5];

    let sqrt_t = ad::sqrt(t);
    let vol_sqrt_t = vol * &sqrt_t;
    let vol_sq = vol * vol;
    let rate_diff = rd - rf;
    let drift_coeff = rate_diff + vol_sq * 0.5;
    let drift = drift_coeff * t;
    let s_over_k = spot / strike;
    let ln_s_over_k = ad::ln(&s_over_k);
    let d1_num = ln_s_over_k + drift;
    let d1 = d1_num / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;

    // N(x) = 0.5 · (1 + erf(x/√2))
    let inv_sqrt_2 = 1.0_f64 / std::f64::consts::SQRT_2;
    let d1_scaled = &d1 * inv_sqrt_2;
    let d2_scaled = &d2 * inv_sqrt_2;
    let e1 = ad::erf(&d1_scaled);
    let e2 = ad::erf(&d2_scaled);
    let n_d1 = (e1 + 1.0) * 0.5;
    let n_d2 = (e2 + 1.0) * 0.5;

    // exp(-rf·T), exp(-rd·T)
    let rf_t = rf * t;
    let rd_t = rd * t;
    let disc_f = ad::exp(&(-rf_t));
    let disc_d = ad::exp(&(-rd_t));

    spot * &disc_f * n_d1 - strike * &disc_d * n_d2
}

/// Primary analytical check + FD smoke check, both running against the
/// **same** 6×6 Dual2Vec Hessian at the same locked test point.
#[test]
fn test_garman_kohlhagen_hessian_two_tier() {
    // ---------------------------------------------------------------
    // Seed all 6 inputs as Dual2Vec variables in dimension n = 6
    // ---------------------------------------------------------------
    let n = 6;
    let inputs_d2v: [Dual2Vec; 6] = [
        Dual2Vec::variable(SPOT, 0, n),
        Dual2Vec::variable(STRIKE, 1, n),
        Dual2Vec::variable(RD, 2, n),
        Dual2Vec::variable(RF, 3, n),
        Dual2Vec::variable(VOL, 4, n),
        Dual2Vec::variable(T_YRS, 5, n),
    ];

    let f_d2v = gk_call_dual2vec(&inputs_d2v);
    let h_d2v = f_d2v.hessian().clone();

    // Bit-exact structural symmetry BEFORE any comparison
    assert_eq!(&h_d2v, &h_d2v.t());

    // ---------------------------------------------------------------
    // Primary check — closed-form analytical Greeks
    // ---------------------------------------------------------------
    //
    // Recompute d1, d2, φ(d1), vega in pure f64 at the test point. These
    // expressions use only `ln`, `exp`, `sqrt`, arithmetic (NO `erf`), so
    // there is no A&S polynomial approximation error in the reference
    // values — the literal closed forms below are f64-exact to roundoff.
    //
    //   d1 = (ln(S/K) + (rd - rf + 0.5·vol²)·T) / (vol·√T)
    //   d2 = d1 - vol·√T
    //   φ(x) = (1/√(2π)) · exp(-x²/2)       -- normal PDF
    //   vega  = S · exp(-rf·T) · φ(d1) · √T
    //
    //   gamma  = ∂²C/∂S²     = exp(-rf·T) · φ(d1) / (S · vol · √T)
    //   volga  = ∂²C/∂vol²   = vega · d1·d2 / vol
    //   vanna  = ∂²C/∂S∂vol  = -exp(-rf·T) · φ(d1) · d2 / vol
    //
    // References: Espen Gaarder Haug, *The Complete Guide to Option
    // Pricing Formulas*, 2nd ed., Garman-Kohlhagen chapter; also any
    // Wilmott text's appendix of second-order Greeks.
    let sqrt_t = T_YRS.sqrt();
    let vol_sqrt_t = VOL * sqrt_t;
    let ln_s_over_k = (SPOT / STRIKE).ln();
    let drift = (RD - RF + 0.5 * VOL * VOL) * T_YRS;
    let d1 = (ln_s_over_k + drift) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    let phi_d1 = (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let disc_f = (-RF * T_YRS).exp();
    let vega = SPOT * disc_f * phi_d1 * sqrt_t;

    let expected_gamma = disc_f * phi_d1 / (SPOT * vol_sqrt_t);
    let expected_volga = vega * d1 * d2 / VOL;
    let expected_vanna = -disc_f * phi_d1 * d2 / VOL;

    // Primary assertions at 1e-11 tolerance on 3 closed-form cells:
    //   gamma = H[0, 0]  (∂²C/∂S²)
    //   volga = H[4, 4]  (∂²C/∂vol²)
    //   vanna = H[0, 4]  (∂²C/∂S∂vol)  [= H[4, 0] by structural symmetry]
    assert_abs_diff_eq!(h_d2v[[0, 0]], expected_gamma, epsilon = 1e-11);
    assert_abs_diff_eq!(h_d2v[[4, 4]], expected_volga, epsilon = 1e-11);
    assert_abs_diff_eq!(h_d2v[[0, 4]], expected_vanna, epsilon = 1e-11);
    // H[4, 0] equals H[0, 4] by the bit-exact symmetry check above,
    // so asserting it separately is redundant — we already know it.

    // ---------------------------------------------------------------
    // Secondary FD smoke check — tolerance 5e-5
    // ---------------------------------------------------------------
    //
    // `xad_rs::compute_hessian` is reverse-mode gradient + forward-
    // difference perturbation at `eps = 1e-7` (see src/hessian.rs). The FD
    // Hessian error is dominated by `O(eps·|∂³f|) + O(roundoff/eps)`; for
    // Garman-Kohlhagen gamma ~3.46 with third derivative O(100), the honest
    // FD bound is ~1.5e-5 on H[0, 0]. Tolerance `5e-5` gives a small safety
    // margin above the measured per-element max-delta.
    //
    // The actual assertion uses `5e-5` to accommodate the measured FD error.
    // The measured reality is ~1.5e-5 on H[0, 0], so `5e-5` provides a
    // safety margin above the honest FD truncation bound.
    let fd_inputs = [SPOT, STRIKE, RD, RF, VOL, T_YRS];
    let h_fd: Vec<Vec<f64>> = xad_rs::compute_hessian(&fd_inputs, gk_call_areal);
    assert_eq!(h_fd.len(), 6);
    assert_eq!(h_fd[0].len(), 6);

    // Every element of the 6×6 must agree at tolerance 5e-5. Measured
    // max-delta at H[0, 0] is ~1.5e-5 (gamma cell), dominated by the
    // O(eps·|∂³C/∂S³|) FD truncation error.
    const FD_TOL: f64 = 5e-5;
    for i in 0..6 {
        for j in 0..6 {
            let diff = (h_d2v[[i, j]] - h_fd[i][j]).abs();
            assert!(
                diff < FD_TOL,
                "FD smoke check failed at [{i},{j}]: d2v={} fd={} diff={} tol={}",
                h_d2v[[i, j]],
                h_fd[i][j],
                diff,
                FD_TOL
            );
        }
    }
}
