//! A value must not depend on the AD mode it is computed in.
//!
//! A mode decides which *derivatives* are available; it must not decide which
//! number comes out. Division is the operation where that property is easiest
//! to lose, because `a / b` and `a * (1/b)` are not the same function in
//! binary floating point: the first rounds once, the second rounds twice, and
//! they land on different `f64`s for a large fraction of inputs. Any mode that
//! forms a quotient from a reciprocal it holds for the partials therefore
//! returns a value up to 1 ulp away from what the passive scalar returns for
//! the same operands.
//!
//! The passive `f64` result is the reference here. That is not a claim that it
//! is the more accurate of two candidates in general — it is that a library
//! offering one generic body under several modes needs one of them to be the
//! referent, and the mode without AD machinery is what a caller comparing
//! against a hand-written implementation will have.
//!
//! The sweep is randomised from a fixed seed rather than a handful of
//! fixtures. The two spellings agree on most inputs, so any single hand-picked
//! pair is as likely as not to miss the divergence — which is exactly how a
//! division-free uniformity body came to pass for as long as it did.

use xad_rs::{AReal, Jet1, Jet2, Jet2Vec, JetK, Tape};

// ============================================================================
// A seeded generator, so a failure is reproducible without a dev-dependency.
// ============================================================================

/// SplitMix64 — the standard seeding generator: one `u64` of state, no
/// warm-up, and equidistributed low bits, which is all this sweep needs.
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

/// How many pairs the sweep draws.
///
/// The divergence this guards against affects roughly a quarter of random
/// operand pairs, so the probability of a broken site surviving this many
/// draws is nil; the count is set by how long the reverse-mode leg takes to
/// record, not by how many samples the property needs.
const SWEEP: usize = 20_000;

/// Fixed, so a failure names a reproducible input rather than a lucky draw.
const SEED: u64 = 0x5EED_0D1E_5EED_0D1E;

/// Operand pairs: half drawn from `[0.3, 1.0]²`, the regime where the two
/// spellings of a quotient were measured to disagree most often, and half
/// spread over both signs and ~40 binades, so a site that only diverges away
/// from unity is caught as well.
///
/// `b` is kept away from zero in both halves: this sweep is about rounding,
/// and an infinity would mask a 1-ulp disagreement rather than expose one.
fn operand_pairs() -> Vec<(f64, f64)> {
    let mut rng = SplitMix64::new(SEED);
    (0..SWEEP)
        .map(|i| {
            if i % 2 == 0 {
                (rng.in_range(0.3, 1.0), rng.in_range(0.3, 1.0))
            } else {
                let scale = |rng: &mut SplitMix64| {
                    let e = (rng.next_u64() % 41) as i32 - 20;
                    let sign = if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
                    sign * rng.in_range(0.5, 1.0) * (2.0_f64).powi(e)
                };
                (scale(&mut rng), scale(&mut rng))
            }
        })
        .collect()
}

// ============================================================================
// Failure accounting — a broken site must name itself, not just fail.
// ============================================================================

/// Counts disagreements per `(mode, operand form)` and keeps the first one, so
/// a run against a partially-corrected implementation reports exactly which
/// modes and which forms are still wrong rather than stopping at the first.
#[derive(Default)]
struct Report {
    rows: Vec<(&'static str, usize, Option<(f64, f64, f64, f64)>)>,
}

impl Report {
    fn check(&mut self, what: &'static str, a: f64, b: f64, got: f64, want: f64) {
        // Bit-identity, not approximate equality: `want` is the correctly
        // rounded quotient and the mode has no licence to return anything
        // else. NaN never arises — `b` is bounded away from zero.
        if got == want {
            return;
        }
        match self.rows.iter_mut().find(|(name, _, _)| *name == what) {
            Some(row) => row.1 += 1,
            None => self.rows.push((what, 1, Some((a, b, got, want)))),
        }
    }

    fn assert_clean(&self) {
        if self.rows.is_empty() {
            return;
        }
        let mut msg = format!(
            "division value differs from the passive quotient in {} of {} \
             mode/operand-form combinations (seed {SEED:#x}, {SWEEP} pairs):\n",
            self.rows.len(),
            SWEEP
        );
        for (what, count, first) in &self.rows {
            let (a, b, got, want) = first.expect("a counted row always keeps its first case");
            msg.push_str(&format!(
                "  {what}: {count}/{SWEEP} disagree; first at a={a:e} b={b:e} \
                 got {got:e} ({:#x}) want {want:e} ({:#x})\n",
                got.to_bits(),
                want.to_bits()
            ));
        }
        panic!("{msg}");
    }
}

// ============================================================================
// The sweep, one leg per mode.
// ============================================================================
//
// Three operand forms are exercised per mode: active ÷ active, active ÷
// passive, and passive ÷ active. They are separate `impl`s at every site, so a
// correction applied to the first form and missed at the other two would still
// leave two thirds of the divisions a caller writes returning the wrong value.
//
// `JetK` and `Jet2Vec` have no scalar-operand `impl`s by design, so their
// passive operand is lifted with `constant` — a distinct path through the same
// `Div`, and the closest thing those types have to a mixed form.

#[test]
fn forward_first_order_division_matches_the_passive_quotient() {
    let mut r = Report::default();
    for (a, b) in operand_pairs() {
        let want = a / b;
        let ja = Jet1::new(a, 1.0);
        let jb = Jet1::new(b, 1.0);
        r.check("Jet1 active/active", a, b, (ja / jb).value(), want);
        r.check("Jet1 active/passive", a, b, (ja / b).value(), want);
        r.check("Jet1 passive/active", a, b, (a / jb).value(), want);
    }
    r.assert_clean();
}

#[test]
fn forward_second_order_division_matches_the_passive_quotient() {
    let mut r = Report::default();
    for (a, b) in operand_pairs() {
        let want = a / b;
        let ja = Jet2::variable(a);
        let jb = Jet2::variable(b);
        r.check("Jet2 active/active", a, b, (ja / jb).value(), want);
        r.check("Jet2 active/passive", a, b, (ja / b).value(), want);
        r.check("Jet2 passive/active", a, b, (a / jb).value(), want);
    }
    r.assert_clean();
}

#[test]
fn k_lane_forward_division_matches_the_passive_quotient() {
    let mut r = Report::default();
    for (a, b) in operand_pairs() {
        let want = a / b;
        let ja = JetK::<f64, 2>::new(a, [1.0, 0.0]);
        let jb = JetK::<f64, 2>::new(b, [0.0, 1.0]);
        r.check("JetK active/active", a, b, (ja / jb).value, want);
        r.check(
            "JetK active/passive",
            a,
            b,
            (ja / JetK::constant(b)).value,
            want,
        );
        r.check(
            "JetK passive/active",
            a,
            b,
            (JetK::constant(a) / jb).value,
            want,
        );
    }
    r.assert_clean();
}

#[test]
fn dense_second_order_division_matches_the_passive_quotient() {
    let mut r = Report::default();
    for (a, b) in operand_pairs() {
        let want = a / b;
        let ja = Jet2Vec::variable(a, 0, 2);
        let jb = Jet2Vec::variable(b, 1, 2);
        // Two separate `impl`s — by reference and by value — so two sites.
        r.check("Jet2Vec active/active (&)", a, b, (&ja / &jb).value(), want);
        r.check(
            "Jet2Vec active/active (owned)",
            a,
            b,
            (ja.clone() / jb.clone()).value(),
            want,
        );
        r.check(
            "Jet2Vec active/passive",
            a,
            b,
            (&ja / &Jet2Vec::constant(b, 2)).value(),
            want,
        );
        r.check(
            "Jet2Vec passive/active",
            a,
            b,
            (&Jet2Vec::constant(a, 2) / &jb).value(),
            want,
        );
    }
    r.assert_clean();
}

#[test]
fn reverse_division_matches_the_passive_quotient() {
    let mut r = Report::default();
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    for (a, b) in operand_pairs() {
        let want = a / b;
        let xa = AReal::input(a, &mut tape);
        let xb = AReal::input(b, &mut tape);
        r.check("AReal active/active", a, b, (xa / xb).value(), want);
        r.check("AReal active/passive", a, b, (xa / b).value(), want);
        r.check("AReal passive/active", a, b, (a / xb).value(), want);
    }
    drop(_rec);
    r.assert_clean();
}

// ============================================================================
// The sweep only proves anything if the inputs it draws can tell the two
// spellings apart. This test pins that, so a future narrowing of the operand
// range cannot quietly turn the sweep above into a tautology.
// ============================================================================

#[test]
fn the_sweep_draws_inputs_that_distinguish_the_two_spellings() {
    let diverging = operand_pairs()
        .into_iter()
        .filter(|&(a, b)| a * (1.0 / b) != a / b)
        .count();
    // Measured at roughly a quarter of random pairs. Asserting a floor rather
    // than the exact count keeps this from being a change-detector while still
    // failing loudly if the generator stops producing discriminating inputs.
    assert!(
        diverging > SWEEP / 10,
        "only {diverging} of {SWEEP} drawn pairs distinguish a/b from a*(1/b); \
         the sweep would pass against a reciprocal-built quotient"
    );
}
