# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [6.1.0] - 2026-08-23

### Fixed — division now returns the same number in every mode

A quotient's value no longer depends on the AD mode it was computed in.

Every `Div` impl in the crate recorded its value as `a * (1/b)`, reusing the
reciprocal it needs for the `∂/∂a = 1/b` partial. 

The correction records the correctly rounded quotient at all eleven `Div`
sites — `AReal` (3 operand forms), `Jet1` (3), `Jet2` (2), `Jet2Vec` (2),
`JetK` (1) — and keeps the reciprocal for the partials.

- **Derivatives are bit-identical.** Verified across all five modes over 300
  operand pairs, comparing raw bit patterns before and after. Two sites —
  `JetK` and `Jet2Vec` — form a partial from the quotient itself, and there
  the reciprocal spelling is deliberately retained for the derivative, so this
  release moves values only. Substituting the corrected quotient there would
  be a strict accuracy improvement for those partials, and is left as a
  separate change with its own justification.
- **The second-order chain rule is untouched.** `Jet2Vec`'s direct closed
  form — which exists so that a quotient's Hessian does not lose roughly half
  the mantissa to cancellation as `b → 1` — is unchanged in form and in
  output.

**Values shift by up to 1 ulp wherever an active mode divides.** That is the
correction, not a regression: the new value is the correctly rounded one, and
it now matches what the same expression gives in `f64`. Code pinning golden
numbers produced by an active mode will see them move. The passive (`f64`)
path never changed, so anything already compared against it now agrees where
it previously did not. This is the reason for a minor rather than a patch:
the API is source-compatible, but the numbers are worth announcing.

**Throughput.** A division op now costs two divisions instead of one division
and one multiply. Measured on Apple M-series with fat LTO, against a
purpose-built harness (the crate ships no benchmark suite as of 6.0.0), taking
the minimum of 40 trials and normalising each mode against the passive `f64`
kernel in the same process: a back-to-back chain of divisions is **≈10–12%
slower in the forward jet modes** (`Jet1`, `Jet2`). Reverse-mode recording,
`JetK`, and a mixed kernel with one division per six operations all landed
inside the ±6% run-to-run noise on that machine. Nowhere near the doubling the
op-cost arithmetic suggests: the two divisions are independent and the divider
pipelines. No hot path regressed enough to justify exposing the reciprocal
form as a separate fast-division entry point, so none was added.

### Added

- `tests/division_value_identity.rs` — a seeded randomised sweep asserting
  bit-exact agreement between the passive quotient and every active mode's
  value, across active÷active, active÷passive and passive÷active for each of
  the five modes. Randomised rather than fixture-based on purpose: the two
  spellings agree on most inputs, so any single hand-picked pair is as likely
  as not to miss the divergence. A companion test pins that the sweep's own
  inputs can still tell the two spellings apart, so narrowing the operand
  range cannot quietly turn it into a tautology.

### Changed

- `tests/real_uniformity.rs` — the cross-mode body is now
  `(x - 1)² / (x + 1.1)`, so it contains every arithmetic operation. It was
  `x² - 2x + 1`, and division being the one operation it omitted is how this
  survived. Its comparison against the passive result is now bit-exact rather
  than `< 1e-12`, and its evaluation point is pinned by a test asserting that
  the point actually distinguishes `a / b` from `a * (1/b)` — most points do
  not, and every point whose numerator is a power of two never can.
- `src/real.rs` — the module documentation now states the **passive-reference
  rule**: a mode determines which derivatives are available, not which number
  comes out, and where they would differ the active mode is brought to the
  passive result. An implementor of a future mode inherits that obligation.

