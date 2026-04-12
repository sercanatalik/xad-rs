# xad-rs

[![Crates.io](https://img.shields.io/crates/v/xad-rs.svg)](https://crates.io/crates/xad-rs)
[![Docs.rs](https://docs.rs/xad-rs/badge.svg)](https://docs.rs/xad-rs)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE.md)
[![MSRV: 1.85](https://img.shields.io/badge/MSRV-1.85-blue.svg)](#installation)

**Exact automatic differentiation for Rust** — forward-mode, reverse-mode,
first- and second-order, with named variable support for ergonomic gradient
readback.

> Unofficial Rust port of the C++ [XAD](https://github.com/auto-differentiation/xad)
> library. Not affiliated with the upstream project.

---

## Choosing a mode

| Type | Mode | Order | Best for |
|---|---|---|---|
| `FReal<T>` | Forward | 1st | 1 input direction, many outputs |
| `Dual` | Forward, multi-var | 1st | full gradient in one pass |
| `Dual2<T>` | Forward, 2nd-order | 1st + 2nd | diagonal Hessian / gamma |
| `AReal<T>` + `Tape` | Reverse (adjoint) | 1st | many inputs, scalar output |
| `Dual2Vec` | Forward, dense 2nd | 1st + 2nd (full) | full n x n Hessian, n < ~50 |

Reverse mode breaks even with forward around n ~ 4 inputs. For n >> 4
(e.g. 30-input swap pricer), reverse is dramatically faster.

Every mode also has a **named** variant (`NamedDual`, `NamedTape`, etc.)
that lets you read gradients by variable name instead of positional index.

---

## Installation

```toml
[dependencies]
xad-rs = "0.3"
```

**MSRV:** 1.85 (Rust edition 2024).

---

## Quick start

### Reverse mode

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

println!("df/dx = {}", x.adjoint(&tape));  // 2xy + cos(x)
println!("df/dy = {}", y.adjoint(&tape));  // x^2
```

### Forward mode (full gradient)

```rust
use xad_rs::Dual;

let (x, y) = (Dual::variable(3.0, 0, 2), Dual::variable(4.0, 1, 2));
let f = &(&x * &x) * &y;  // x^2 * y

assert_eq!(f.partial(0), 24.0);  // df/dx = 2xy
assert_eq!(f.partial(1),  9.0);  // df/dy = x^2
```

### Second-order derivatives

```rust
use xad_rs::Dual2;

let x = Dual2::variable(2.0_f64);
let y = x * x * x;  // x^3
assert_eq!(y.first_derivative(),  12.0);  // 3x^2
assert_eq!(y.second_derivative(), 12.0);  // 6x
```

### Named variables

Access derivatives by name — useful in financial models with many risk factors:

```rust
use xad_rs::{NamedForwardTape, NamedForwardScope};

let mut ft = NamedForwardTape::new();
let spot_h   = ft.declare_dual("spot",   100.0);
let strike_h = ft.declare_dual("strike", 105.0);
let scope: NamedForwardScope = ft.freeze_dual();

let spot   = scope.dual(spot_h);
let strike = scope.dual(strike_h);
let ratio  = spot / strike;

assert!((ratio.partial("spot") - 1.0 / 105.0).abs() < 1e-14);
```

Named reverse mode returns gradients as `IndexMap<String, f64>`:

```rust
use xad_rs::NamedTape;

let mut tape = NamedTape::new();
let x = tape.input("x", 3.0);
let y = tape.input("y", 4.0);
let _registry = tape.freeze();

let f = &(&x * &x) * &y + x.sin();
let grad = tape.gradient(&f);

assert!((grad["x"] - (2.0 * 3.0 * 4.0 + 3.0_f64.cos())).abs() < 1e-12);
assert!((grad["y"] - 9.0).abs() < 1e-12);
```

### Jacobian and Hessian

```rust
use xad_rs::{compute_jacobian_rev, compute_hessian};

// f: R^2 -> R^2, f(x, y) = [x*y, x + y]
let jac = compute_jacobian_rev(&[3.0, 5.0], |v| {
    vec![&v[0] * &v[1], &v[0] + &v[1]]
});

// g: R^2 -> R, g(x, y) = x^2 * y + y^3
let hess = compute_hessian(&[2.0, 3.0], |v| {
    let x2 = &v[0] * &v[0];
    let y3 = &v[1] * &v[1] * &v[1];
    x2 * &v[1] + y3
});
```

### Dense full Hessian (Dual2Vec)

```rust,ignore
use xad_rs::Dual2Vec;

let x = Dual2Vec::variable(1.0, 0, 2);
let y = Dual2Vec::variable(2.0, 1, 2);
let f = &(&(&x * &x) * &y) + &(&(&y * &y) * &y);

assert_eq!(f.hessian()[[0, 0]], 4.0);   // d2f/dx2 = 2y
assert_eq!(f.hessian()[[0, 1]], 2.0);   // d2f/dxdy = 2x
assert_eq!(f.hessian()[[1, 1]], 12.0);  // d2f/dy2 = 6y
```

Per-op cost is O(n^2). For n > ~50, prefer seeded `Dual2<T>` with n passes.

---

## Crate structure

```
src/
  forward/          FReal, Dual, Dual2, Dual2Vec + Named wrappers
  reverse/          AReal, NamedAReal, NamedTape
  ops/              compute_jacobian_*, compute_hessian, compute_full_hessian
  math.rs           AD-aware transcendentals (sin, exp, erf, norm_cdf, ...)
  tape.rs           Reverse-mode tape and thread-local active-tape slot
  scalar.rs         Scalar trait bound (f32, f64)
  registry.rs       VarRegistry — ordered name-to-index map
  forward_tape.rs   NamedForwardTape / NamedForwardScope setup
```

---

## Examples

| Example | What it demonstrates |
|---|---|
| [`swap_pricer.rs`](examples/swap_pricer.rs) | 30-input IRS DV01 and gamma via reverse, Dual, and Dual2 |
| [`fx_option.rs`](examples/fx_option.rs) | Garman-Kohlhagen FX option Greeks |
| [`fixed_rate_bond.rs`](examples/fixed_rate_bond.rs) | YTM / duration / convexity |
| [`jacobian.rs`](examples/jacobian.rs) | 4x4 Jacobian (reverse mode) |
| [`hessian.rs`](examples/hessian.rs) | 4x4 Hessian with analytic cross-check |

```sh
cargo run --release --example swap_pricer
```

---

## Design notes

- **Tape storage is thread-local.** One `Tape<T>` per thread; `NamedTape` is `!Send`.
- **Forward mode is allocation-light.** `Dual` keeps tangents in a single `Vec<f64>` with fused, autovectorizable loops.
- **Zero-alloc operator fast paths.** Every `AReal` binary op uses fixed-arity `Tape::push_binary` / `push_unary` — no intermediate `Vec` per op.

---

## Tests

```sh
cargo test
```

165 tests covering operator correctness, transcendentals, second-order derivatives, Jacobian/Hessian helpers, named variable readback, and cross-mode consistency.

---

## License

AGPL-3.0-or-later, matching the upstream XAD project. See [`LICENSE.md`](LICENSE.md).

---

## Acknowledgements

- The C++ [XAD](https://github.com/auto-differentiation/xad) library — architectural inspiration and source of the financial examples.
- [`num-traits`](https://crates.io/crates/num-traits) for generic scalar plumbing.
