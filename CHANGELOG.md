# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [7.1.0] - 2026-08-24

### Added — the reverse drivers on a tape the caller owns

`compute_gradient_rev_with(tape, inputs, f)` and
`compute_jacobian_rev_with(tape, inputs, f)`: each begins a fresh recording on
the caller's tape through `Tape::record()`, retaining its allocation, and
returns with the tape inactive and ready for the next call. The bare forms are
unchanged in signature and now delegate — one recording body per driver, the
bare spelling owning its tape and the `_with` spelling borrowing the caller's.
Both forms return the same value and derivatives bit for bit, on a fresh tape
and across reuse.

The tape itself learned reuse in 6.x (`record()`, ~3× on many-small-tapes
workloads); the drivers had not, and paid a fresh `Tape::new` per call. What
made it measurable: a downstream lattice valuation records ~2.3 million
statements per reverse pass, once per position per risk run.

### Fixed

`compute_jacobian_rev` scoped its activation with `activate()` /
`deactivate_all()`, which a panic inside the function skipped, leaving the
thread's tape active. It now uses the RAII guard the gradient driver already
used; a panicking function leaves no tape active for either driver.

### Dropped from the roadmap — second order through a generic body

The 6.0 roadmap carried two items toward second-order sensitivities from a
`Real`-generic body: `Real`-generic second-order drivers, and a `Real`
implementation for `AReal<Jet1<_>>` storage. Neither is needed. `Jet2`
implements `Real`, and the downstream library reached σ-gamma through a
short-rate lattice, bond convexity, and the directional curvature of a P&L
explain — each by instantiating one generic body at `Jet2`, with no backend
change. Both items are dropped rather than deferred; a deferred item with no
consumer is a promise nobody can plan against. What remains genuinely
unreachable — a dense Hessian in one pass — is `Jet2Vec` as a `Real` mode,
and it stays gated on a caller needing one.

## [7.0.0] - 2026-08-24

### Changed — a generic body reads the way the mathematics does

`Real` lets a numerical body be written once and evaluated in any mode. It did
not let that body be *written* the way the formula reads. Two taxes fell on
every generic expression, neither a correctness problem, both paid at every
arithmetic site forever:

- **A borrowed operand needed an explicit clone.** `Real` required no
  `Copy`-like bound, so generic code could not use a value twice without
  cloning it — even where the concrete mode was `f64` and the clone compiled
  away.
- **A passive operand had to be lifted at the call site.** A schedule weight,
  a year fraction, a notional — `f64` by construction and never
  differentiated — could not meet an active scalar without `R::from(..)`.

A linear interpolation, the smallest non-trivial kernel there is, had to be
spelled `y0.clone() + (y1 - y0) * R::from(w)`. It is now:

```rust
fn lerp<R: CopyableReal>(y0: R, y1: R, w: R::Passive) -> R {
    y0 + (y1 - y0) * w
}
```

**`Real` gains passive-operand bounds in both positions.** `x * tau` and
`tau * x` both compile for a passive `tau`. The right-hand position is a
supertrait bound on `Self`; the left-hand position is a bound on the `Passive`
associated type, because `impl<T> Mul<Self> for T` is not writable — which is
why the crate's scalar-left macros each name a concrete `f64`. A *bound* is
not an impl and is not subject to the orphan rule: it obliges each mode to
supply the concrete impls it already had. All four in-crate modes satisfied
the new bounds with **no edit to any operator implementation**.

**This is breaking for anyone implementing `Real` out of crate.** In-crate it
is four modes and cost nothing. An out-of-crate mode must now provide
`{Add, Sub, Mul, Div}` against its passive scalar in both operand positions.
See the `Stability` note under 6.0.0: `Real` is public and unsealed, so this
requires a major version even though every in-crate implementor updates itself.

**It is not breaking for callers.** Existing code with explicit clones and
lifts compiles unchanged — the release permits shorter spellings, it does not
forbid the current ones. Adopting them is an unforced readability sweep.

### Added

- **`CopyableReal`** — `trait CopyableReal: Real + Copy`, blanket-implemented
  for every `Real` that is `Copy`, so no mode implements it explicitly.
  Re-exported from the crate root and the prelude. A body wanting to use an
  operand twice without a clone asks for `CopyableReal`.

  The bound is deliberately **not** on `Real` itself, and the reason is
  forward-looking rather than present: every mode the crate ships today is
  `Copy`, including `JetK`, whose storage is a fixed `[T; K]`. A `Copy` bound
  on `Real` would exclude nothing that implements it and cost no migration.
  What it would exclude is any mode carrying heap storage — and a dense
  second-order mode is necessarily shaped that way, since its tangent count is
  the input-space dimension. `Jet2Vec` is already held out of `Real` for a
  separate reason (`From<f64>` cannot know that dimension); a `Copy` bound
  would add a second obstacle, and unlike the first one it would be permanent,
  unwindable only by another major release. So the bound sits on a sub-trait
  and `Real` stays open.

- `tests/operand_spelling_identity.rs` — the acceptance gate for this release.

### Known constraint

**Do not pin the passive type when a passive operand appears on the left.**
Writing `R: Real<Passive = f64>` normalizes the projection away and with it
the left-hand bounds: `x * tau` keeps compiling, `tau * x` stops. A body that
wants a passive operand on the left must name `R::Passive` rather than pin it
— which is the spelling a caller would reach for anyway. Held by a test.

### No values changed

Nothing here touches arithmetic. No operator body, no entry in the elementary
table, and no tape path was modified; the diff is a trait header, an empty
sub-trait, documentation, and two re-export lines. The cross-mode
bit-identity guarantee that caught the 6.1.0 division defect passes unchanged.

That guarantee is necessary but not sufficient here, and the distinction is
the point of the new test. A type-level change can compile, pass every
existing test, and still move numbers — because the shorter spelling resolves
to a *different operator implementation* than the longer one (`tau * x`
records a unary tape statement where `AReal::from(tau) * x` records a binary
one; `x / tau` takes the scalar-RHS `Div` where `x / R::from(tau)` takes the
two-`Real` one). Every pre-existing test uses only the longer spelling, so
every one of them would have stayed green if the shorter spelling had landed a
different number.

So `tests/operand_spelling_identity.rs` writes one body twice — once with
clones and lifts, once with neither — and compares them in all four modes on
values and on every derivative each mode provides.

**It compares term by term, not totals, and that is not a stylistic choice.**
The first version of the test summed its terms and compared the sums;
reverting a passive-position `Div` to the two-rounding `l * (1/r)` spelling did
**not** fail it — the mutated term was around `0.4` against a total around
`10`, so the 1-ulp error landed below the total's own ulp and was rounded
away. The body now returns its ten terms and each is asserted at its own
magnitude, one term per operator implementation under test. Verified against
four separate reciprocal mutations (`f64/AReal`, `Jet1/f64`, `Jet2/f64`,
`AReal/f64`): each fails, naming the term that moved. The evaluation point is
itself pinned by a test to one where all three divisions in the body actually
separate `a / b` from `a * (1/b)` — the first constants tried separated none
of them.

### Performance

**Not measured, and therefore not claimed.** The crate has shipped no
benchmark suite since 6.0.0. What can be said without measuring is structural:
the release adds no arithmetic, and the blanket `impl<R: Real + Copy>
CopyableReal for R {}` has no methods to dispatch.

## [6.1.0] - 2026-08-23

### Fixed — division now returns the same number in every mode

A quotient's value no longer depends on the AD mode it was computed in.

Every `Div` impl in the crate recorded its value as `a * (1/b)`, reusing the
reciprocal it needs for the `∂/∂a = 1/b` partial. That spelling rounds twice
where IEEE division rounds once, so an active mode returned a value up to
1 ulp away from what `f64` returns for the same operands. Over the 20,000
randomised pairs the new property test sweeps, **27.5% disagreed**. Addition,
subtraction and multiplication were unaffected — they always recorded their
direct expressions — which is what made division the one arithmetic operation
whose result changed with the mode.

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

## [6.0.0] - 2026-08-23

**Relicensed to MIT.** Versions 0.4.1 through 5.1.0 were published under
AGPL-3.0-or-later and remain available under that license; 6.0.0 onward is MIT.

`xad-rs` computes exact, machine-precision derivatives of arbitrary numerical
programs — no finite-difference error, no symbolic manipulation, just the
chain rule applied to the program as it runs. It is an independent Rust
implementation of the AD architecture popularized by the C++
[XAD](https://github.com/auto-differentiation/xad) library, and it is
shaped for the workload that motivates AAD in quant finance: many small
valuations, many risk factors, one scalar PV.

### The `Real` trait

The crate is built around one mode-agnostic active-scalar trait. A numerical
body is written once as `fn f<R: Real>(..) -> R` and evaluated against `f64`
(no derivatives), `Jet1<f64>` (forward, first order), `Jet2<f64>` (forward,
second order), or `AReal<f64>` (reverse) — including `AReal<JetK<f64, K>>`,
which is how a `Real`-generic body yields an exact Hessian.

`Real` provides:

- **27 unary elementaries** — the transcendentals (`ln`, `exp`, `sqrt`, the
  trigonometric, inverse-trigonometric, and hyperbolic families, `exp2`,
  `log2`, `log10`, `ln_1p`, `exp_m1`, `cbrt`, `abs`) and the **Gaussian
  family** (`erf`, `erfc`, `norm_pdf`, `norm_cdf`, `inv_norm_cdf`). The
  Gaussian entries are what let a closed-form option value be written once as
  `fn price<R: Real>(..) -> R` and priced in every mode.
- **Powers** — `powf` (arbitrary-real exponent) and `powi`.
- **Numeric identities** — `zero()` and `one()`, so generic code need not
  convert from a float literal at every site. These are associated
  *functions*, not associated `const` items, so a mode whose representation is
  not a compile-time constant can still implement the trait. In reverse mode
  both are unrecorded constants: producing one does not grow the tape.
- **Fused aggregates** — `sum`, `dot`, `weighted_sum`, `weighted_dot`, taking
  slices, with per-mode bodies. These are not conveniences. The reverse-mode
  bodies record **one** n-ary tape statement where a binary-operator chain
  records `n - 1`, so a hot accumulation loop ported to `R: Real` keeps its
  tape size and sweep cost. They are trait methods rather than free generic
  functions precisely because a free function cannot dispatch to a per-mode
  body without specialisation, which would cost reverse mode exactly that
  fused recording. `tests/fused_ops.rs` asserts the statement and operand
  counts *through the trait method*, so a delegation to a binary chain fails a
  test rather than silently regressing. The weighted forms take passive
  weights: accrual factors and notionals are contract data, not
  differentiable market inputs.
- **Branch selection** — `max` and `min`, with the derivative following the
  winning branch (the standard sub-gradient convention for payoffs). `abs`
  follows the same convention, taking `f'(0) = +1`.

**One derivative table, five surfaces.** `src/elementaries.rs` is the single
source of truth for unary elementary derivatives: each entry carries a value
closure, a first-derivative closure, and a second-derivative closure. The
`Real` trait declarations, all four `Real` implementations, `math::ad`,
`math::fwd`, and the inherent methods on `Jet2` and `Jet2Vec` are all *stamped*
from that table. No parallel list of method names exists anywhere in the crate,
so adding a table entry adds the trait method and every implementation of it
with no other edit — drift is prevented by construction rather than detected by
a test. Each active mode carries the exact analytic tangent through the
`Passive::erf_value` / `Passive::inv_norm_cdf_value` hooks, so a trait method
never propagates the derivative of a value approximation.

### Modes

| Type | Mode | Order | Use when |
|---|---|---|---|
| `f64` | none (passive) | 0 | no derivatives needed |
| `Jet1<T>` | forward | 1st | one input direction, many outputs |
| `Jet2<T>` | forward | 1st + 2nd | diagonal Hessian / gamma along one direction |
| `AReal<T>` + `Tape` | reverse (adjoint) | 1st | many inputs, scalar output |
| `Jet2Vec` | forward, dense | 1st + 2nd | full `n × n` Hessian in one pass |
| `JetK<T, K>` + `Tape` | forward-over-adjoint | 1st + 2nd | full Hessian in `⌈n/K⌉` passes |

Reverse mode breaks even with forward around `n ≈ 4` inputs. The forward jets
nest inside the tape as storage scalars, which is what makes the
forward-over-adjoint engine possible.

`Jet2Vec` deliberately has no `Real` impl: the trait's `From<f64>` would
require knowing the input-space dimension. Use it directly for a full Hessian
in one pass.

### Derivative drivers

`ops` provides entry points that own their tape management, so callers do no
registration, seeding, or sweeping by hand:

| Driver | Mode | Shape |
|---|---|---|
| `compute_derivative_fwd` | forward | `R → R`; no tape created or activated |
| `compute_directional_derivative_fwd` | forward | `Rⁿ → R` along a caller-supplied seed |
| `compute_gradient_rev` | reverse | `Rⁿ → R`; value and gradient from one sweep |
| `compute_jacobian_rev` | reverse | `Rⁿ → Rᵐ`; full Jacobian in one vector sweep |
| `compute_hessian` | forward-over-adjoint | full `n × n` Hessian, `n` passes |
| `compute_hessian_k::<K, _>` | forward-over-adjoint | full Hessian in `⌈n/K⌉` passes |
| `compute_hessian_k_par::<K, _>` | forward-over-adjoint | as above, across a `rayon` pool |
| `compute_full_hessian` | forward, dense | value, gradient, and Hessian from one `Jet2Vec` pass |

`compute_hessian`, `compute_hessian_k`, and `compute_hessian_k_par` produce
bit-identical results. `compute_full_hessian` returns a `DenseHessian` whose
`value`, `gradient`, and `hessian` fields are read positionally, following the
order of the inputs passed in.

### The tape

A packed three-buffer tape (statements, operands, multipliers) with a
thread-local active-tape slot, following the layout XAD popularized.

- `Tape::record` is an RAII guard: it activates the tape, resets the recording
  in place, and deactivates on drop — including on unwind from a panic.
- `Tape::new_recording` reuses the allocation across valuations. On a
  many-small-tapes workload this is worth **~2.4×**, because allocation churn
  was about 60% of the runtime.
- `Tape::with_capacity` sizes the buffers up front when the recording size is
  known.
- `Tape::compute_adjoints_vector` drives a multi-direction reverse sweep;
  `compute_jacobian_rev` is built on it and is ~1.75× faster than
  row-at-a-time on a wide Jacobian.
- The tape is thread-local, so `rayon` workers each hold their own recording
  and need no coordination.

### Numerical accuracy

**`erf` is full precision.** A two-regime, cancellation-free evaluation:

- `|x| ≤ 3`: the confluent-hypergeometric series
  `(2/√π)·e^{−x²}·Σ x·(2x²)ⁿ/(2n+1)!!` — all-positive terms, at most 43 of
  them at the switch point;
- `|x| > 3`: `1 − erfc` via the Gauss continued fraction, evaluated backward
  at fixed depth 24, saturating at `±1` from `|x| = 6` (where `erfc < 2⁻⁵⁴`).

Worst measured relative error: **1.3e-15**, over a dense 53k-point sweep of
`[-6.5, 6.5]` against a correctly-rounded reference. Pinned in
`tests/erf_precision.rs` (reference table at 5 ulp, plus oddness, monotonicity,
saturation, NaN, and `norm_cdf` at its seam).

Because the AD tangent for `erf` is exact analytically, finite differences of
the value — not the AD result — are the approximate side of any AD-vs-FD
comparison.

**Known limitation — `erfc` in the far tail.** `math::erfc` is evaluated as
`1 - erf(x)`, the same expression the derivative table uses, so the trait
method and the free function are bit-identical. It therefore inherits that
subtraction's cancellation wherever `erfc` is the small quantity. Measured
relative error against a high-precision reference: ~1e-11 at `x = 3`, ~1e-9 at
`x = 4`, **~1.5e-5 at `x = 5`**, and **exactly `0.0` for `x ≥ 6`**. Use it for
`|x| ≤ 4`. The identity is what every AD surface differentiates, so the
*derivative* is exact throughout — only the value in the tail is affected.

### Performance

All figures measured on Apple M-series with `lto = "fat"`, single-threaded
unless stated otherwise.

- Widening the Hessian pass with `compute_hessian_k::<K, _>` seeds `K` tangent
  lanes per recording: **8.2×** at `K = 8` on a 48-input, 2000-op kernel,
  single-threaded, and **14.7×** via `compute_hessian_k_par`.
- On a 30-input swap pricer, the entire 30 × 30 Hessian takes 4 passes
  (**14.8 µs**), against 25.3 µs for the diagonal alone via 30 seeded `Jet2`
  passes.
- A `fn f<R: Real>(…)` body at `R = f64` runs within **~1%** of hand-written
  `f64` — monomorphization erases the trait.
- `[profile.release]` uses `lto = "fat"` and `codegen-units = 1`.

Measured and rejected: a struct-of-arrays tape layout (regresses small
cache-resident tapes 12–16%) and expression-template fusion (no isolated
bottleneck).

Checkpointing is not implemented. For a forward pass large enough to exhaust
memory, decompose it manually and accumulate chunk gradients, or drop to
forward mode where `n` is small.

### Stability

`Real` is a public, **unsealed** trait. Adding a required method to it is a
breaking change for any out-of-crate implementor, so such additions require a
major version — even though its unary method set is generated from the
elementary table and all in-crate implementors update themselves. Code that
only *calls* `Real` is unaffected by that class of change.

The supported surface is the `Real` trait, the reverse-mode `Tape` / `AReal`,
the forward jets (`Jet1`, `Jet2`, `Jet2Vec`, `JetK`), and the `ops` drivers
built on them. The rule that keeps it honest: *a mode without a runnable
example is not a supported mode.* The runnable programs in `examples/` define
what is offered.

### Requirements

Rust 1.85 or newer (edition 2024). Licensed MIT.
