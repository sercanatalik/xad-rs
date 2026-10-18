# xad-rs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
[![MSRV: 1.98](https://img.shields.io/badge/MSRV-1.98-blue.svg)](#installation)

**Exact reverse-mode automatic differentiation for Rust.** Record a program on
a tape as it runs, then sweep the tape backwards once to get the derivative of
the output with respect to every input. No finite-difference error, no symbolic
manipulation — just the chain rule, applied in reverse.

> Independent Rust implementation of the AD architecture popularized by the C++
> [XAD](https://github.com/auto-differentiation/xad) library. Not affiliated with
> the upstream project.

## Installation

```toml
[dependencies]
xad-rs = "1.0.0"
```

Requires Rust 1.98 or newer (edition 2024).

## Documentation

Everything ships inside the archive; no link below leaves it.

- **API** — `cargo doc --open`: the crate-level page explains which mode fits
  which problem shape and every public item is documented.
- **Theory** — six chapters under `docs/theory/`, indexed by `docs/README.md`,
  from "what is AD?" to the numerical analysis of adjoints. Start with
  `docs/theory/03-reverse-mode-and-taped-adjoints.md` for the tape.
- **Examples** — runnable, cross-checked against analytic answers, under
  `examples/`; `cargo run --release --example <name>`.

## Why reverse mode

Reverse mode computes the full gradient of a scalar-valued function in **one
sweep**, whatever the number of inputs. The cost of the gradient is a small
constant multiple of the cost of the function itself. That is the shape of most
quant workloads: a price with dozens of market inputs, and every first-order
sensitivity wanted at once.

| Your problem | Use | Read more |
|---|---|---|
| Full gradient, any number of inputs, scalar output | `compute_gradient_rev`, or `Tape` + `AReal<T>` by hand | 03 — Reverse mode & taped adjoints |
| Full `m × n` Jacobian, `m` outputs | `compute_jacobian_rev` — one vector sweep | 03 — Reverse mode & taped adjoints |
| Many valuations in a loop | `compute_gradient_rev_with` or `Tape::record` on one retained tape | Performance, below |

## The pieces

- **`AReal<T>`** — the active scalar. Behaves like `T` in arithmetic and in the
  `math::ad` elementary functions; every operation it takes part in is recorded.
- **`Tape<T>`** — the recording. Thread-local, activated explicitly, replayed
  backwards by `compute_adjoints`. Owns the adjoint storage.
- **`Real`** — a trait implemented for both `f64` and `AReal<f64>`. A kernel
  written once as `fn f<R: Real>(..) -> R` runs passively for the plain value
  and on the tape for the gradient, with a bit-identical primal in both cases.
- **Drivers** — `compute_gradient_rev`, `compute_jacobian_rev`, and their
  `_with` variants. They own the tape lifecycle so most callers never touch
  `Tape` directly.

The `Real` unary method set is *generated* from the crate's single derivative
table in `src/elementaries.rs`, the same table that stamps `math::ad`. There is
no parallel list to drift: adding a table entry adds the trait method and the
`AReal` implementation at once.

## Quick start

### Gradient via the driver

```rust
use xad_rs::{Real, compute_gradient_rev};

// f: R^n -> R, full gradient from one reverse sweep; the driver owns the tape.
let (v, grad) = compute_gradient_rev(&[3.0_f64, 4.0], |x| {
    x[0].clone() * x[0].clone() * x[1].clone() + x[0].sin()
});

assert!((v - (36.0 + 3.0_f64.sin())).abs() < 1e-12);
assert!((grad[0] - (24.0 + 3.0_f64.cos())).abs() < 1e-12);   // 2xy + cos(x)
assert_eq!(grad[1], 9.0);                                     // x^2
```

### Gradient via the tape, by hand

```rust
use xad_rs::{AReal, Tape, math};

let mut tape = Tape::<f64>::new(true);
tape.activate();

let mut x = AReal::new(3.0);
let mut y = AReal::new(4.0);
AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
AReal::register_input(std::slice::from_mut(&mut y), &mut tape);

// f(x, y) = x^2 * y + sin(x)
let mut f = &(&x * &x) * &y + math::ad::sin(&x);
AReal::register_output(std::slice::from_mut(&mut f), &mut tape);
f.set_adjoint(&mut tape, 1.0);
tape.compute_adjoints();

assert!((x.adjoint(&tape) - (2.0 * 3.0 * 4.0 + 3.0_f64.cos())).abs() < 1e-12);
assert_eq!(y.adjoint(&tape), 9.0);
```

### One body, passive or taped

The Gaussian family is on `Real`, so a closed-form pricer is written once and
evaluated either for the price alone or for the price plus every greek. The
primal value is bit-identical in both cases.

```rust
use xad_rs::{Real, compute_gradient_rev};

fn call_price<R: Real>(s: &R, k: &R, r: &R, vol: &R, t: &R) -> R {
    let sqrt_t = t.sqrt();
    let d1 = ((s.clone() / k.clone()).ln()
        + (r.clone() + vol.clone() * vol.clone() / R::from(2.0_f64)) * t.clone())
        / (vol.clone() * sqrt_t.clone());
    let d2 = d1.clone() - vol.clone() * sqrt_t;
    s.clone() * d1.norm_cdf() - k.clone() * (-r.clone() * t.clone()).exp() * d2.norm_cdf()
}

// Passive — just the price, no AD machinery.
let px = call_price(&100.0_f64, &100.0, &0.05, &0.2, &1.0);

// Reverse — price and all five first-order greeks from ONE sweep.
let (v, g) = compute_gradient_rev(&[100.0_f64, 100.0, 0.05, 0.2, 1.0], |a| {
    call_price(&a[0], &a[1], &a[2], &a[3], &a[4])
});
let (delta, rho, vega) = (g[0], g[2], g[3]);
```

Both agree with the analytic Black–Scholes formulas to machine precision
(`px == v == 10.4505835722`, `delta == 0.6368306512`).

### Jacobian

```rust
use xad_rs::{AReal, compute_jacobian_rev, math};

// f: R^4 -> R^4, all sixteen partials from one vector sweep.
let j = compute_jacobian_rev(&[1.0_f64, 1.5, 1.3, 1.2], |x: &[AReal<f64>]| vec![
    math::ad::sin(&(&x[0] + &x[1])),
    math::ad::sin(&(&x[1] + &x[2])),
    math::ad::cos(&(&x[2] + &x[3])),
    math::ad::cos(&(&x[3] + &x[0])),
]);
assert!((j[[0, 0]] - (1.0_f64 + 1.5).cos()).abs() < 1e-12);
```

## Performance

Quant workloads are dominated by *many small valuations* (a curve bootstrap is a
Newton loop; risk is positions × scenarios). The levers, all measured on
Apple M-series with `lto = "fat"`:

- **Reuse the tape** across valuations with `compute_gradient_rev_with`, or
  `Tape::record` (RAII guard around `new_recording`) when driving the tape by
  hand, instead of a fresh `Tape::new` each time. A fresh tape reserves room
  for a small valuation (256 statements / 512 operands), so on bodies that fit,
  reuse is worth ~1.3× on a six-input body and ~5% on a 30-input one; it
  matters most once a recording outgrows the reserve.
- **Vector reverse mode** — `compute_jacobian_rev` recovers a full `m × n`
  Jacobian in one sweep (`Tape::compute_adjoints_vector`), ~1.75× on a wide
  Jacobian. A caller with its own recording can drive
  `Tape::compute_adjoints_vector` directly.
- **Fuse reductions** — `Real::weighted_sum` records one tape statement where
  an equivalent `+` chain records one per term.
- **Parallelize** independent valuations with `rayon` plus one `Tape::record`
  per worker — the tape is thread-local, so workers need no coordination.
- **Tune the build** — `[profile.release]` uses `lto = "fat"`, `codegen-units = 1`.
- **Primal is free** — a `fn f<R: Real>(…)` body at `R = f64` runs within ~1% of
  hand-written `f64` (monomorphization erases the trait).

Measured-and-rejected: a struct-of-arrays tape layout (regresses small cache-resident
tapes 12–16%) and expression-template fusion (no isolated bottleneck).

## Examples

Run with `cargo run --release --example <name>`.

| Example (under `examples/`) | What it demonstrates |
|---|---|
| `adjoint_first_order.rs` | Full 4-input gradient in one reverse sweep |
| `jacobian.rs` | 4×4 Jacobian via `compute_jacobian_rev`, cross-checked against analytic |
| `real_generic_pricer.rs` | One `fn call_price<R: Real>` body priced passively and on the tape, and `Real::weighted_sum` recording 1 tape statement where a `+` chain records 16 |
| `fx_option.rs` | Garman–Kohlhagen FX option greeks via reverse mode, cross-checked against analytic |
| `swap_pricer.rs` | 30-input IRS: DV01 and the full bucketed delta vector via reverse mode |
| `fixed_rate_bond.rs` | YTM and duration via reverse mode |

## License

MIT. See `LICENSE.md`, which ships in the archive.

## Acknowledgements

- The C++ [XAD](https://github.com/auto-differentiation/xad) library —
  architectural inspiration, and the model for this crate's financial examples.
- [QuantLibAAD](https://github.com/auto-differentiation/QuantLibAAD) — the
  XAD-instrumented QuantLib build; reference for the AAD-on-quant-finance
  patterns the financial examples in this crate are modelled after.
- [`num-traits`](https://crates.io/crates/num-traits) — the numeric trait
  vocabulary `Real` sits beside; [`ndarray`](https://crates.io/crates/ndarray) —
  the dense storage behind the Jacobian driver;
  [`rayon`](https://crates.io/crates/rayon) — the work-stealing pool used for
  parallel valuations. Three at run time, deliberately few.
