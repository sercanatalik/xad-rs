//! Garman-Kohlhagen FX option 6×6 Hessian cross-check for `Jet2Vec`.
//!
//! Two assertions run on the **same** 6×6 Hessian at the **same** test point:
//!
//! 1. **Primary** — analytical closed-form second-order Greeks at tolerance
//!    `1e-11`. The gamma closed form `∂²C/∂S² = exp(-rf·T)·φ(d1) /
//!    (S·σ·√T)` is computed in pure f64 (only `ln`, `exp`, `sqrt`, arithmetic
//!    — NO erf), so there is no Abramowitz-Stegun approximation floor to
//!    fight. Volga and vanna add two more bit-exact cells.
//!
//! 2. **Secondary** — `xad_rs::compute_hessian` (exact forward-over-adjoint,
//!    `AReal<Jet1<f64>>`) at tolerance `1e-12`. Both methods are exact and
//!    `Jet1<T>`'s `Passive::erf_value` override pairs the A&S value with
//!    the **exact** analytic tangent (see `jet1_passive.rs`), so the two
//!    Hessians agree to machine precision — measured max-delta ~1.3e-15
//!    across all 36 cells. (Before the exact-tangent override, the A&S
//!    polynomial's derivative error leaked ~1.5e-5 into the gamma cell,
//!    which is why this tolerance was once 5e-5.)
//!
//! Neither assertion post-processes the Hessian via any averaging helper —
//! structural symmetry is asserted bit-exactly BEFORE any comparison.
//! The Garman-Kohlhagen formula is duplicated in this test file with a
//! derivation comment so the test is self-contained.
//!
//! ## Why `erf` value precision doesn't matter for the 1e-11 check
//!
//! The `Jet2Vec::erf` elementary uses `crate::math::erf` (Abramowitz &
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
use xad_rs::Jet2Vec;

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
// Garman-Kohlhagen call price — Jet2Vec closure (6 active inputs)
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
fn gk_call_dual2vec(inputs: &[Jet2Vec; 6]) -> Jet2Vec {
    let n = 6;
    let spot = &inputs[0];
    let strike = &inputs[1];
    let rd = &inputs[2];
    let rf = &inputs[3];
    let vol = &inputs[4];
    let t = &inputs[5];

    // Scalar constants wrapped as Jet2Vec constants so arithmetic composes
    let half = Jet2Vec::constant(0.5, n);
    let one = Jet2Vec::constant(1.0, n);
    let sqrt_2 = Jet2Vec::constant(std::f64::consts::SQRT_2, n);
    let zero = Jet2Vec::constant(0.0, n);

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

    // exp(-rf·T), exp(-rd·T)  — via `zero - (rf * t)` because Jet2Vec has no unary Neg
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
fn gk_call_areal<T: xad_rs::TapeStorage>(inputs: &[AReal<T>]) -> AReal<T> {
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
    let drift_coeff = rate_diff + vol_sq * T::from(0.5).unwrap();
    let drift = drift_coeff * t;
    let s_over_k = spot / strike;
    let ln_s_over_k = ad::ln(&s_over_k);
    let d1_num = ln_s_over_k + drift;
    let d1 = d1_num / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;

    // N(x) = 0.5 · (1 + erf(x/√2))
    let inv_sqrt_2 = T::from(1.0_f64 / std::f64::consts::SQRT_2).unwrap();
    let d1_scaled = &d1 * inv_sqrt_2;
    let d2_scaled = &d2 * inv_sqrt_2;
    let e1 = ad::erf(&d1_scaled);
    let e2 = ad::erf(&d2_scaled);
    let n_d1 = (e1 + T::from(1.0).unwrap()) * T::from(0.5).unwrap();
    let n_d2 = (e2 + T::from(1.0).unwrap()) * T::from(0.5).unwrap();

    // exp(-rf·T), exp(-rd·T)
    let rf_t = rf * t;
    let rd_t = rd * t;
    let disc_f = ad::exp(&(-rf_t));
    let disc_d = ad::exp(&(-rd_t));

    spot * &disc_f * n_d1 - strike * &disc_d * n_d2
}

/// Primary analytical check + FD smoke check, both running against the
/// **same** 6×6 Jet2Vec Hessian at the same locked test point.
#[test]
fn test_garman_kohlhagen_hessian_two_tier() {
    // ---------------------------------------------------------------
    // Seed all 6 inputs as Jet2Vec variables in dimension n = 6
    // ---------------------------------------------------------------
    let n = 6;
    let inputs_d2v: [Jet2Vec; 6] = [
        Jet2Vec::variable(SPOT, 0, n),
        Jet2Vec::variable(STRIKE, 1, n),
        Jet2Vec::variable(RD, 2, n),
        Jet2Vec::variable(RF, 3, n),
        Jet2Vec::variable(VOL, 4, n),
        Jet2Vec::variable(T_YRS, 5, n),
    ];

    let f_d2v = gk_call_dual2vec(&inputs_d2v);
    let h_d2v = f_d2v.hessian();

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
    // Secondary forward-over-adjoint cross-check — tolerance 1e-12
    // ---------------------------------------------------------------
    //
    // `xad_rs::compute_hessian` is exact forward-over-adjoint
    // (`AReal<Jet1<f64>>`, see src/ops/hessian.rs) and the `Jet1` erf
    // override carries the exact analytic tangent, so both Hessians are
    // exact: measured max-delta is ~1.3e-15 across all 36 cells. `1e-12`
    // leaves three orders of magnitude of slack for platform libm
    // differences.
    let fd_inputs = [SPOT, STRIKE, RD, RF, VOL, T_YRS];
    let h_fd = xad_rs::compute_hessian(&fd_inputs, gk_call_areal);
    assert_eq!(h_fd.dim(), (6, 6));

    const FD_TOL: f64 = 1e-12;
    for i in 0..6 {
        for j in 0..6 {
            let diff = (h_d2v[[i, j]] - h_fd[[i, j]]).abs();
            assert!(
                diff < FD_TOL,
                "FD smoke check failed at [{i},{j}]: d2v={} fd={} diff={} tol={}",
                h_d2v[[i, j]],
                h_fd[[i, j]],
                diff,
                FD_TOL
            );
        }
    }
}
