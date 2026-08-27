//! A power must not depend on the AD mode it is computed in.
//!
//! `tests/division_value_identity.rs` states the rule and the reason: a mode
//! decides which *derivatives* are available, never which number comes out, and
//! the passive `f64` result is the referent. Division is where the property is
//! easiest to lose. Powers are where it is easiest to lose *quietly*, because
//! there are two ways to lose it and neither looks like an approximation:
//!
//! 1. **`powi` routed through `powf`.** `f64::powi` multiplies; `f64::powf`
//!    goes through `exp`/`ln`. They are different functions, and they land on
//!    different `f64`s for roughly *half* of ordinary operands — not a rare
//!    corner. A mode that spells its integer power `v.powf(n as f64)` because
//!    the derivative table is written in terms of a real exponent returns a
//!    different number from the passive scalar on most calls.
//!
//! 2. **`powf` composed as `exp(v · ln u)`.** The composition is the natural
//!    way to get both derivative orders out of primitives that already have
//!    them, and it rounds three times where `powf` rounds once.
//!
//! Both were present in `Jet2` alone, which is why the sweep runs every mode
//! rather than the one that was wrong: the point of the file is that a mode
//! added later is swept without anybody remembering to add it.
//!
//! Discovered from a product-level bit-identity gate one library up, where two
//! lattice-priced families disagreed between the passive and second-order
//! forward modes — `powi(j)` down a discount ladder — while the first-order
//! forward and reverse modes agreed with passive exactly.

use xad_rs::{AReal, Jet1, Jet2, JetK, Real, Tape};

// ============================================================================
// A seeded generator, so a failure names a reproducible input.
// ============================================================================

/// SplitMix64, as `division_value_identity.rs` uses, and for the same reason:
/// reproducible without a dev-dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1)`, from the top 53 bits.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    fn in_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// How many `(base, exponent)` pairs the sweep draws.
const SWEEP: usize = 4_000;

/// Fixed, so a failure names a reproducible input rather than a lucky draw.
const SEED: u64 = 0x50E7_0D1E_50E7_0D1E;

/// Bases are kept strictly positive: this sweep is about rounding, and a
/// negative base makes `powf` a NaN, which would mask a 1-ulp disagreement
/// rather than expose one. `powi` is swept over both signs separately below.
fn bases() -> Vec<f64> {
    let mut rng = SplitMix64::new(SEED);
    (0..SWEEP)
        .map(|i| {
            if i % 2 == 0 {
                // Near unity — the regime a discount ladder lives in, and where
                // a wrong spelling is least likely to look wrong.
                rng.in_range(0.3, 1.7)
            } else {
                let e = (rng.next_u64() % 21) as i32 - 10;
                rng.in_range(0.5, 1.0) * (2.0_f64).powi(e)
            }
        })
        .collect()
}

/// Integer exponents, both signs, spanning the range a repeated-squaring
/// implementation takes different paths through.
const POWERS: [i32; 12] = [-41, -12, -7, -3, -1, 0, 1, 2, 3, 5, 11, 37];

/// Real exponents drawn alongside the bases.
fn real_exponents() -> Vec<f64> {
    let mut rng = SplitMix64::new(SEED ^ 0xFFFF_FFFF_FFFF_FFFF);
    (0..SWEEP).map(|_| rng.in_range(-6.0, 6.0)).collect()
}

// ============================================================================
// Failure accounting — a broken site must name itself, not just fail.
// ============================================================================

/// Counts disagreements per `(mode, form)` and keeps the first, so a run
/// against a partially-corrected implementation reports every mode still wrong
/// rather than stopping at the first.
#[derive(Default)]
struct Report {
    rows: Vec<(&'static str, usize, Option<(f64, f64, f64, f64)>)>,
}

impl Report {
    fn check(&mut self, what: &'static str, base: f64, exp: f64, got: f64, want: f64) {
        // Bit-identity, not approximate equality. NaN never arises: bases are
        // positive and `want` is finite for every drawn pair.
        if got == want {
            return;
        }
        match self.rows.iter_mut().find(|(name, _, _)| *name == what) {
            Some(row) => row.1 += 1,
            None => self.rows.push((what, 1, Some((base, exp, got, want)))),
        }
    }

    fn assert_clean(&self, total: usize) {
        if self.rows.is_empty() {
            return;
        }
        let mut msg = format!(
            "power value differs from the passive result in {} mode/form \
             combinations (seed {SEED:#x}, {total} pairs):\n",
            self.rows.len()
        );
        for (what, count, first) in &self.rows {
            let (base, exp, got, want) = first.expect("a counted row keeps its first case");
            msg.push_str(&format!(
                "  {what}: {count}/{total} disagree; first at base={base:e} \
                 exponent={exp} got {got:e} ({:#x}) want {want:e} ({:#x})\n",
                got.to_bits(),
                want.to_bits()
            ));
        }
        panic!("{msg}");
    }
}

// ============================================================================
// The premise: the two spellings really do disagree, and not rarely.
// ============================================================================
//
// Without this, a later change that made `powi` and `powf` agree everywhere —
// or a base range drawn where they happen to — would turn every test below
// into one that passes while checking nothing.

#[test]
fn the_two_power_spellings_disagree_on_a_large_fraction_of_operands() {
    let mut powi_vs_powf = 0usize;
    let mut composed_vs_powf = 0usize;
    let mut total = 0usize;
    for base in bases() {
        for n in POWERS {
            total += 1;
            if base.powi(n) != base.powf(f64::from(n)) {
                powi_vs_powf += 1;
            }
        }
    }
    for (base, exp) in bases().into_iter().zip(real_exponents()) {
        if (exp * base.ln()).exp() != base.powf(exp) {
            composed_vs_powf += 1;
        }
    }
    // Measured at ~58% and ~9% respectively for this seed; asserted well under
    // both, since the claim is "not rare", not a pinned rate.
    assert!(
        powi_vs_powf * 5 > total,
        "powi and powf agreed on all but {powi_vs_powf} of {total} operands — \
         the sweeps below would no longer be checking anything"
    );
    assert!(
        composed_vs_powf * 100 > SWEEP,
        "exp(v·ln u) and powf agreed on all but {composed_vs_powf} of {SWEEP} \
         operands — same"
    );
}

// ============================================================================
// The sweep, one leg per mode.
// ============================================================================

/// Every mode's `Real::powi`, against `f64::powi` of the same operands.
///
/// Both signs of the base: an integer power is defined for a negative base and
/// a mode is not entitled to lose that by routing through `powf`, which is NaN
/// there. That is the same defect as the rounding one, in its loudest form.
#[test]
fn every_mode_takes_an_integer_power_the_way_the_passive_scalar_does() {
    let mut r = Report::default();
    let mut total = 0usize;
    let mut tape = Tape::<f64>::new(true);
    for base in bases() {
        for signed in [base, -base] {
            for n in POWERS {
                total += 1;
                let want = Real::powi(&signed, n);
                let e = f64::from(n);

                r.check("Jet1", signed, e, Real::powi(&Jet1::new(signed, 1.0), n).value(), want);
                r.check("Jet2", signed, e, Real::powi(&Jet2::variable(signed), n).value(), want);
                r.check(
                    "JetK",
                    signed,
                    e,
                    Real::powi(&JetK::<f64, 4>::new(signed, [1.0, 0.0, 0.0, 0.0]), n).value(),
                    want,
                );
                let got = {
                    let _rec = tape.record();
                    Real::powi(&AReal::<f64>::from(signed), n).value()
                };
                r.check("AReal", signed, e, got, want);
            }
        }
    }
    r.assert_clean(total);
}

/// Every mode's `Real::powf`, against `f64::powf` of the same operands.
#[test]
fn every_mode_takes_a_real_power_the_way_the_passive_scalar_does() {
    let mut r = Report::default();
    let mut total = 0usize;
    let mut tape = Tape::<f64>::new(true);
    for (base, exp) in bases().into_iter().zip(real_exponents()) {
        total += 1;
        let want = Real::powf(&base, exp);

        r.check(
            "Jet1",
            base,
            exp,
            Real::powf(&Jet1::new(base, 1.0), Jet1::constant(exp)).value(),
            want,
        );
        r.check(
            "Jet2",
            base,
            exp,
            Real::powf(&Jet2::variable(base), Jet2::constant(exp)).value(),
            want,
        );
        r.check(
            "JetK",
            base,
            exp,
            Real::powf(
                &JetK::<f64, 4>::new(base, [1.0, 0.0, 0.0, 0.0]),
                JetK::<f64, 4>::constant(exp),
            )
            .value(),
            want,
        );
        let got = {
            let _rec = tape.record();
            Real::powf(&AReal::<f64>::from(base), AReal::<f64>::from(exp)).value()
        };
        r.check("AReal", base, exp, got, want);
    }
    r.assert_clean(total);
}

/// Correcting a value must not have cost the derivatives.
///
/// `Div` established the shape this fix follows — derivatives from the composed
/// form, value from the passive reference — and the risk it carries is that the
/// write-back is applied to the wrong field, or that a rewritten derivative
/// table drifts from the one it replaced. Both orders are checked against a
/// central difference of the corrected value itself, so the check cannot be
/// satisfied by the same mistake twice.
#[test]
fn the_corrected_powers_still_carry_both_derivative_orders() {
    let h = 1e-4;
    for base in [0.4_f64, 0.97, 1.0, 1.6, 3.3] {
        for n in [-7i32, -1, 2, 5] {
            let at = |x: f64| Real::powi(&x, n);
            let j = Real::powi(&Jet2::variable(base), n);
            let d1 = (at(base + h) - at(base - h)) / (2.0 * h);
            let d2 = (at(base + h) - 2.0 * at(base) + at(base - h)) / (h * h);
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
            assert!(
                rel(j.first_derivative(), d1) < 1e-5,
                "powi({base}, {n}) first derivative {} vs {d1}",
                j.first_derivative()
            );
            assert!(
                rel(j.second_derivative(), d2) < 1e-3,
                "powi({base}, {n}) second derivative {} vs {d2}",
                j.second_derivative()
            );
        }
        for e in [-2.5_f64, 0.5, 1.7, 4.2] {
            let at = |x: f64| Real::powf(&x, e);
            let j = Real::powf(&Jet2::variable(base), Jet2::constant(e));
            let d1 = (at(base + h) - at(base - h)) / (2.0 * h);
            let d2 = (at(base + h) - 2.0 * at(base) + at(base - h)) / (h * h);
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
            assert!(
                rel(j.first_derivative(), d1) < 1e-5,
                "powf({base}, {e}) first derivative {} vs {d1}",
                j.first_derivative()
            );
            assert!(
                rel(j.second_derivative(), d2) < 1e-3,
                "powf({base}, {e}) second derivative {} vs {d2}",
                j.second_derivative()
            );
        }
    }
}
