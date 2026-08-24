# xad-rs

[![Crates.io](https://img.shields.io/crates/v/xad-rs.svg)](https://crates.io/crates/xad-rs)
[![Docs.rs](https://docs.rs/xad-rs/badge.svg)](https://docs.rs/xad-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
[![MSRV: 1.85](https://img.shields.io/badge/MSRV-1.85-blue.svg)](#installation)

**Exact automatic differentiation for Rust.** Forward and reverse mode, first and
second order, built around a single mode-agnostic `Real` trait. No
finite-difference error, no symbolic manipulation — just the chain rule, applied
to your program as it runs.

> Independent Rust implementation of the AD architecture popularized by the C++
> [XAD](https://github.com/auto-differentiation/xad) library. Not affiliated with
> the upstream project.

## Installation

```toml
[dependencies]
xad-rs = "7.1"
```

Requires Rust 1.85 or newer (edition 2024).

## Documentation

- **Theory** — derivations and decision-tree guidance: [`docs/README.md`](docs/README.md).
- **API** — generated rustdoc: [docs.rs/xad-rs](https://docs.rs/xad-rs).
- **Examples** — runnable, cross-checked against analytic answers: [`examples/`](examples/).


## Pick a mode

| Your problem | Mode | Read more |
|---|---|---|
| Just the value, no derivatives | `f64` | [01 — AD as a discipline](docs/theory/01-automatic-differentiation.md) |
| One input direction, any number of outputs | `Jet1<T>`, or `compute_derivative_fwd` | [02 — Forward mode & dual numbers](docs/theory/02-forward-mode-and-dual-numbers.md) |
| Full gradient, any number of inputs, scalar output | `compute_gradient_rev`, or `Tape` + `AReal<T>` by hand | [03 — Reverse mode & taped adjoints](docs/theory/03-reverse-mode-and-taped-adjoints.md) |
| Gamma / diagonal Hessian along one direction | `Jet2<T>` | [04 — Second-order & k-jets](docs/theory/04-second-order-and-k-jets.md) |
| Full n × n Hessian, n ≲ 50 | `Jet2Vec` via `compute_full_hessian` | [04 — Second-order & k-jets](docs/theory/04-second-order-and-k-jets.md) |
| Full n × n Hessian, larger n | `compute_hessian_k::<K, _>` — nested `Tape<JetK<f64, K>>` | [04 — Second-order & k-jets](docs/theory/04-second-order-and-k-jets.md) |

Reverse mode breaks even with forward around n ≈ 4 inputs, and the crate's own
examples show it: at n = 1 `Jet1` is 17× faster than a tape pass
(`fixed_rate_bond`), while at n = 30 reverse wins outright (`swap_pricer`). For a
full Hessian, the K-wide engine needs `⌈n/K⌉` passes instead of `n` — on the
30-input swap pricer that is the **entire 30 × 30 matrix in 4 passes (14.8 µs)**,
against 25.3 µs for the diagonal alone via 30 seeded `Jet2` passes.

`Real` is implemented for `f64`, `AReal<f64>`, `Jet1<f64>`, and `Jet2<f64>`, so a
kernel written once as `fn f<R: Real>(..) -> R` runs under all of them — including
`AReal<JetK<f64, K>>`, which is how a `Real`-generic body gets an exact Hessian.
`Jet2Vec` lacks the impl because the trait's `From<f64>` requires knowing the
input-space dimension; use it directly for a full Hessian in one pass.

The trait's unary method set is *generated* from the crate's single derivative
table in `src/elementaries.rs` — the same table that stamps `math::ad`,
`math::fwd`, and the `Jet2` / `Jet2Vec` inherent methods. There is no parallel
list to drift: adding a table entry adds the trait method and all four
implementations at once.

## By mode — quick recipes

### Mode-agnostic (one body, every mode)

The Gaussian family is on `Real`, so a closed-form pricer is written once and
evaluated in whichever mode the caller needs. The mode decides which
derivatives come back with the price, not what the price is: every operation
produces a bit-identical value in every mode, with the passive `f64` result as
the reference.

```rust
use xad_rs::{Jet2, Real, compute_gradient_rev};

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

// Forward, second order — gamma from the same body, seeded in spot.
let gamma = call_price(
    &Jet2::variable(100.0), &Jet2::constant(100.0), &Jet2::constant(0.05),
    &Jet2::constant(0.2),   &Jet2::constant(1.0),
).second_derivative();
```

All four agree with the analytic Black–Scholes formulas to machine precision
(`px == v == 10.4505835722`, `delta == 0.6368306512`, `gamma == 0.0187620173`).

### First-order drivers

```rust
use xad_rs::{Real, compute_derivative_fwd, compute_directional_derivative_fwd, compute_gradient_rev};

// f: R -> R, forward mode, no tape created.
let (v, dv) = compute_derivative_fwd(2.0_f64, |x| x.clone() * x.ln());

// f: R^n -> R along one seed direction, forward mode, no tape.
let (v, dir) = compute_directional_derivative_fwd(
    &[3.0_f64, 4.0], &[1.0, 0.0], |x| x[0].clone() * x[0].clone() * x[1].clone());

// f: R^n -> R, full gradient from one reverse sweep; the driver owns the tape.
let (v, grad) = compute_gradient_rev(&[3.0_f64, 4.0], |x| {
    x[0].clone() * x[0].clone() * x[1].clone() + x[0].sin()
});
```

Reverse breaks even against repeated forward passes around `n ~ 4`.

### Reverse (gradient via tape)

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


## Performance

Quant workloads are dominated by *many small valuations* (a curve bootstrap is a
Newton loop; risk is positions × scenarios). The levers, all measured on
Apple M-series with `lto = "fat"`:

- **Reuse the tape** across valuations with `Tape::record` (RAII guard around
  `new_recording`) instead of a fresh `Tape::new` each time — **~2.4×** on a
  many-small-tapes workload (allocation churn was ~60% of the time).
- **Vector reverse mode** — `compute_jacobian_rev` recovers a full `m × n` Jacobian
  in one sweep (`Tape::compute_adjoints_vector`), ~1.75× on a wide Jacobian. A
  caller with its own recording can drive `Tape::compute_adjoints_vector` directly.
- **Widen the Hessian pass** — `compute_hessian_k::<K, _>` seeds `K` tangent lanes
  per recording, so an `n × n` Hessian costs `⌈n/K⌉` passes instead of `n`:
  measured 8.2× at K = 8 on a 48-input, 2000-op kernel, single-threaded, and
  14.7× via `compute_hessian_k_par`. Results are bit-identical to `compute_hessian`.
- **Parallelize** independent valuations with `rayon` plus one `Tape::record`
  per worker — the tape is thread-local, so workers need no coordination.
- **Tune the build** — `[profile.release]` uses `lto = "fat"`, `codegen-units = 1`.
- **Primal is free** — a `fn f<R: Real>(…)` body at `R = f64` runs within ~1% of
  hand-written `f64` (monomorphization erases the trait).

Measured-and-rejected: a struct-of-arrays tape layout (regresses small cache-resident
tapes 12–16%) and expression-template fusion (no isolated bottleneck).

## Examples

Run with `cargo run --release --example <name>`.

| Example | What it demonstrates |
|---|---|
| [`real_generic_pricer.rs`](examples/real_generic_pricer.rs) | One `fn call_price<R: Real>` body priced under `f64` / `AReal` / `Jet1` / `Jet2`, the three first-order drivers, and `Real::weighted_sum` recording 1 tape statement where a `+` chain records 16 |
| [`swap_pricer.rs`](examples/swap_pricer.rs) | 30-input IRS: DV01 via reverse, diagonal gamma via `Jet2`, and the full 30×30 Hessian via `compute_hessian_k` |
| [`fx_option.rs`](examples/fx_option.rs) | Garman–Kohlhagen FX option greeks via reverse mode and `Jet2` spot-gamma, cross-checked against analytic |
| [`fixed_rate_bond.rs`](examples/fixed_rate_bond.rs) | YTM, duration, convexity |
| [`jacobian.rs`](examples/jacobian.rs) | 4×4 Jacobian via reverse mode |
| [`hessian.rs`](examples/hessian.rs) | 4×4 Hessian via `compute_full_hessian` with analytic cross-check |
| [`adjoint_first_order.rs`](examples/adjoint_first_order.rs) | Full 4-input gradient in one reverse sweep |
| [`fwd_adj_second_order.rs`](examples/fwd_adj_second_order.rs) | Forward-over-adjoint: one seeded sweep over `Tape<Jet1<f64>>` gives the gradient *and* a Hessian row; cross-checked against `compute_hessian` |



## License

MIT. See [`LICENSE.md`](LICENSE.md).

## Acknowledgements

- The C++ [XAD](https://github.com/auto-differentiation/xad) library —
  architectural inspiration, and the model for this crate's financial examples.
- [QuantLibAAD](https://github.com/auto-differentiation/QuantLibAAD) — the
  XAD-instrumented QuantLib build; reference for the AAD-on-quant-finance
  patterns the financial examples in this crate are modelled after.
- [`num-traits`](https://crates.io/crates/num-traits) and
  [`ndarray`](https://crates.io/crates/ndarray) for the underlying primitives.
