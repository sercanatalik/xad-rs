# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [6.0.0] - 2026-08-23

Initial release.

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
