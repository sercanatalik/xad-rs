# 04 — Second-order and k-jets

> k-jets as truncated Taylor series and as the rings
> `R[ε]/(ε^{k+1})`, Faà di Bruno's higher-order chain rule, single-
> direction second order (`Jet2`), the dense full-Hessian type
> `Jet2Vec`, Hessian symmetry, the forward-over-reverse construction
> for Hessian-vector products, and edge pushing for sparse Hessians.
> Anchors: `Jet2`, `Jet2Vec`, `ops::compute_hessian`,
> `ops::compute_full_hessian`.

## Overview

Some computations need second-order information. The two big use cases:

- **Curvature for optimisation.** Newton-style methods need the
  Hessian `H = ∂²f / ∂x_i ∂x_j` to step.
- **Convexity-style risk in finance.** Gamma (the second derivative of
  PV with respect to spot) and cross-gammas matter for hedge
  rebalancing and capital calculations.

`xad-rs` ships two forward-mode types for this:

- [`Jet2<T>`](https://docs.rs/xad-rs/latest/xad_rs/forward/jet2/struct.Jet2.html) — a *single-direction* second-order forward
  type. Carries `(value, d1, d2)`: value, first derivative along the
  seeded direction, and second derivative along the same direction.
  Useful for own-gamma / one-direction curvature.
- [`Jet2Vec<T>`](https://docs.rs/xad-rs/latest/xad_rs/forward/jet2vec/struct.Jet2Vec.html) — a *multi-direction* second-order forward
  type. Carries `(value, grad ∈ R^n, hess ∈ R^{n×n})`: value, full
  gradient, and full Hessian. One forward pass for everything.

Both are exact (no finite differences). The composite helpers
[`compute_hessian`](https://docs.rs/xad-rs/latest/xad_rs/ops/hessian/fn.compute_hessian.html)
(reverse-mode with finite-difference perturbation of the gradient,
approximate) and
[`compute_full_hessian`](https://docs.rs/xad-rs/latest/xad_rs/ops/hessian/fn.compute_full_hessian.html)
(`Jet2Vec` based, exact) wrap these.

## Theory

### k-jets as truncated Taylor series

The systematic generalisation of dual numbers to higher orders is to
take a longer quotient of the polynomial ring:

```math
J^k(\mathbb{R}) \;=\; \mathbb{R}[\varepsilon]\,/\,(\varepsilon^{k+1}).
```

In words: the **k-jet ring** `J^k(R)` consists of polynomials in `ε`
of degree at most `k`. The `k = 1` case is the dual numbers of chapter
02; `k = 2` adds second-order information; `k → ∞` (formal power
series) recovers the full Taylor algebra.

A 2-jet is a triple `(v_0, v_1, v_2)` interpreted as

```math
v \;=\; v_0 + v_1\,\varepsilon + \tfrac{1}{2}\,v_2\,\varepsilon^2,
\qquad \varepsilon^3 \equiv 0.
```

The factor `1/2` follows the Taylor convention so that
`v(t) = v_0 + v_1 t + (v_2 / 2) t² + O(t³)` reads as the truncated
Taylor expansion you'd write by hand: `v_0` is the value, `v_1` the
first derivative along the seeded direction, `v_2` the second
derivative along that direction.

Multiplication of 2-jets follows from `ε³ = 0`:

```math
(a_0 + a_1\varepsilon + \tfrac{1}{2}a_2\varepsilon^2)(b_0 + b_1\varepsilon + \tfrac{1}{2}b_2\varepsilon^2)
\;=\;
a_0 b_0 + (a_0 b_1 + a_1 b_0)\,\varepsilon + \tfrac{1}{2}(a_0 b_2 + 2 a_1 b_1 + a_2 b_0)\,\varepsilon^2.
```

In words: the value of the product is the product of the values; the
first-derivative part is `a_0 b_1 + a_1 b_0` (product rule), and the
second-derivative part is `a_0 b_2 + 2 a_1 b_1 + a_2 b_0`. The middle
term — the `2 a_1 b_1` — is what makes second-order forward more
expensive than first-order: it is a genuinely new contribution beyond
re-applying the product rule.

For smooth `f`:

```math
f(a_0 + a_1\varepsilon + \tfrac{1}{2}a_2\varepsilon^2)
\;=\; f(a_0) + f'(a_0) a_1 \varepsilon + \tfrac{1}{2}\bigl(f''(a_0) a_1^2 + f'(a_0) a_2\bigr)\varepsilon^2.
```

In words: applying a smooth `f` to a 2-jet keeps the value `f(a_0)`,
scales the first derivative by `f'(a_0)`, and the new second-derivative
part picks up two contributions: the convexity `f''(a_0) a_1²` and the
prior curvature `f'(a_0) a_2`. This is the second-order chain rule.

`Jet2<T>` is exactly this triple. See [`src/forward/jet2.rs`](https://github.com/sercanatalik/xad-rs/blob/main/src/forward/jet2.rs)
for the literal product- and chain-rule impls.

### Faà di Bruno's formula and why we stop at k=2

The higher-order chain rule generalises Leibniz/product-rule
combinatorics. The closed form is **Faà di Bruno's formula**:

```math
\frac{d^k}{dt^k} f(g(t)) \;=\; \sum_{\pi \in \mathcal{P}(k)} f^{(|\pi|)}(g(t))\,\prod_{B \in \pi} g^{(|B|)}(t),
```

where the sum ranges over all set partitions `π` of `{1, …, k}`,
`|π|` is the number of blocks, and `|B|` is the size of each block.
Equivalently, the `n`-th Taylor coefficient of `f ∘ g` is

```math
\frac{1}{k!}(f \circ g)^{(k)}(t) \;=\; \sum_{\substack{m_1, \ldots, m_k \ge 0 \\ \sum j m_j = k}} \frac{f^{(\sum m_j)}(g(t))}{m_1! \cdots m_k!}\,\prod_{j=1}^{k}\biggl(\frac{g^{(j)}(t)}{j!}\biggr)^{m_j}.
```

The number of terms grows as the number of partitions of `k` — the
**Bell number** `B_k`: `B_1 = 1, B_2 = 2, B_3 = 5, B_4 = 15, B_5 = 52`,
…. This means a generic order-`k` AD has `O(B_k)` work per elementary
op, which grows superexponentially in `k`. For `k = 2` (Bell number
2), the chain rule has the two summands shown above (`f''(a) a_1² +
f'(a) a_2`). For `k = 3` you would have 5; for `k = 4`, 15.

This combinatorial growth is the practical reason higher-order AD
beyond second-order is rarely used: the per-op cost balloons, the
storage for higher tensors balloons faster (a 4-tensor on `n` inputs
is `n⁴` numbers), and there are few applications outside Taylor-series
integrators that need it.

`xad-rs` exposes `Jet1` and `Jet2`. Higher orders are not implemented;
if you need them, the formal-power-series view of `R[[ε]]` lets you
build a general k-jet type via templated arrays of length `k+1` — but
in practice you would reach for an integrator (e.g. *TIDES* in C,
*TaylorSeries.jl* in Julia) instead.

### From single-direction to full Hessian

`Jet2<T>` tracks first and second derivatives along *one* seeded
direction. If you want the full Hessian, you have two options:

1. **Per-column scan.** Run `Jet2` `n` times, once per input
   direction; each pass yields one diagonal Hessian entry plus enough
   information to back out cross-terms in restricted settings (via
   polarisation: `∂²f/∂x_i ∂x_j = ½(∂²f/∂(x_i+x_j)² − ∂²f/∂x_i² −
   ∂²f/∂x_j²)`). This is `O(n)` forward passes, each carrying constant
   extra state, with the polarisation trick costing one additional
   `(i, j)` pass per off-diagonal entry — about `n²/2` passes total.
2. **Propagate everything at once.** Carry the full gradient and full
   Hessian as live state on every intermediate. This is what `Jet2Vec`
   does: a single forward pass yields value, gradient, and the dense
   `n × n` Hessian.

The multi-direction second-order chain rule, written out for
`Jet2Vec` (see the module docs in [`src/forward/jet2vec.rs`](https://github.com/sercanatalik/xad-rs/blob/main/src/forward/jet2vec.rs#L40-L53)),
is:

```math
g'[i] \;=\; f'(v)\cdot g[i],
\qquad
H'[i,j] \;=\; f''(v)\cdot g[i]\,g[j] + f'(v)\cdot H[i,j].
```

In words: under a smooth `f`, each gradient entry scales by `f'(v)`,
and each Hessian entry picks up a *rank-one* outer-product term
`f''(v) g[i] g[j]` plus a scaled copy of the previous Hessian
`f'(v) H[i,j]`. The outer product is the multi-direction generalisation
of the `2 a_1 b_1` cross-term from the scalar case.

For a binary `φ(u, v)` the rule generalises further: gradient of the
output is `φ_u g_u + φ_v g_v`, and Hessian of the output picks up
*every* cross-term involving the two input gradients and Hessians,
weighted by the appropriate second-order partial of `φ`. The relevant
identities are the Hessian product rule and chain rule:

```math
H_{\varphi(u,v)} \;=\; \varphi_u H_u + \varphi_v H_v
    + \varphi_{uu}\, g_u g_u^{\!\top}
    + \varphi_{uv}\, (g_u g_v^{\!\top} + g_v g_u^{\!\top})
    + \varphi_{vv}\, g_v g_v^{\!\top},
```

which `Jet2Vec`'s binary-op impls compute directly.

### Hessian symmetry — Clairaut–Schwarz

For any twice-continuously-differentiable `f`, Clairaut–Schwarz gives

```math
\frac{\partial^2 f}{\partial x_i \,\partial x_j}
\;=\;
\frac{\partial^2 f}{\partial x_j \,\partial x_i}.
```

In words: mixed partials are equal — the Hessian is symmetric. This is
not just a mathematical curiosity; it has an operational consequence
for AD. Storing the full `n × n` Hessian as a dense matrix doubles the
storage relative to storing only the upper triangle.

`xad-rs`'s `Jet2Vec` exploits this structurally: every binary op
computes the upper triangle and mirrors to the lower via a
`pub(crate)` helper. The library's tests assert `assert_eq!(h, h.t())`
bit-exactly, not just within a tolerance — the storage *is*
structurally symmetric. This means the cost-per-op is `O(n(n+1)/2)`,
not `O(n²)`, modulo the mirror.

Round-off can still break symmetry numerically when the Hessian is
ill-conditioned; chapter 07 quantifies this. The `Jet2Vec` impl
sidesteps it by computing only the upper triangle and mirroring, so
symmetry holds at the *storage* level even when it would not hold to
last-bit precision under naive evaluation.

### Forward-over-reverse: Hessian-vector products at `O(P)`

For very large `n`, even `Jet2Vec`'s `O(n²)` per-op cost is too much.
The alternative is **forward-over-reverse** (often abbreviated FoR or
HVP, for Hessian-vector product): combine forward mode (over
directions) with reverse mode (over outputs) to compute Hessian-vector
products `H v` in `O(P)` time per direction, without ever materialising
the full Hessian.

The construction. Start from a scalar function `f : R^n → R`. The
gradient `g(x) = ∇f(x)` is itself a function `R^n → R^n`. Its Jacobian
is the Hessian: `J_g(x) = H_f(x)`. A Jacobian-vector product on `g`
gives `H_f v`. So:

```math
H_f(x)\,v \;=\; J_g(x)\,v \;=\; (\text{forward-mode JVP applied to } g).
```

Operationally, run reverse mode on `f` to obtain `g`, but *carry
tangents through every step*: every intermediate of the reverse sweep
becomes a `Jet1` whose primal is the usual reverse-mode adjoint and
whose tangent is the directional derivative of that adjoint along `v`.
The output's tangent is `H v`. The total work is `~2P` for the forward
record + `~2P` for the reverse sweep + a constant factor for the
tangent propagation through the reverse sweep, so `O(P)` per HVP.

This is the standard reverse-over-forward construction that mainstream
reverse-mode AD frameworks use for their `vhp` / `hvp` operators.
`xad-rs` does not currently expose a built-in HVP helper, but the
ingredients exist:
the reverse-mode pipeline reads tangent values out of `AReal`-on-`Jet1`
when the right type plumbing is wired up. For the common case
`n ≲ 50`, `Jet2Vec` via `compute_full_hessian` is fast enough; for
larger `n` consider rolling HVP by hand or switching to a library with
native HVP support.

### Edge pushing for sparse Hessians

When the Hessian is **sparse** (only `O(n)` non-zero entries, common
for separable functions and certain physical simulations), there is a
specialised algorithm called **edge pushing** that computes the entire
sparse Hessian in time proportional to the sparsity, not to `n²`.

The idea: instead of propagating a dense Hessian on every intermediate,
maintain only the non-zero edges of the *Hessian graph* (the bipartite
graph whose entries `(i, j)` correspond to non-zero `H_{ij}`). Each
elementary `φ_k` of arity `r` adds at most `O(r²)` new edges to the
Hessian graph; for typical `r = 2` this is constant per-step.

Edge pushing was introduced by Gower & Mello (2012); the algorithm is
implemented in specialised graph-coloring AD tooling. `xad-rs` does not currently expose
edge pushing; the `Jet2Vec` dense impl is the default. If your problem
has structurally sparse Hessian and `n > 100`, the right move is to
either roll a hand-written symbolic Hessian or compute Hessian-vector
products on demand via forward-over-reverse.

### Two `compute_hessian` helpers, two different things

The `ops` module exposes two functions that look superficially similar:

- [`compute_hessian`](https://docs.rs/xad-rs/latest/xad_rs/ops/hessian/fn.compute_hessian.html) — repeated reverse-mode passes with
  **finite-difference perturbation** of the gradient. Approximate
  (`O(1e-7)` accuracy, see chapter 07), `O(n)` reverse-mode passes,
  no `Jet2Vec` required. Useful when you need a Hessian and don't have
  or don't want the extra dependency on second-order forward.
- [`compute_full_hessian`](https://docs.rs/xad-rs/latest/xad_rs/ops/hessian/fn.compute_full_hessian.html) — a single `Jet2Vec` forward pass.
  Exact (machine precision), one pass, but `O(n²)` per-op cost. The
  right choice for `n ≲ 50` when you want exactness.

Pick `compute_full_hessian` by default; fall back to `compute_hessian`
only if you cannot use `Jet2Vec` for some reason.

## Cost model

Let `P` be the flop count of the primal, `n` the input dimension.

| Operation | Time | Memory (per live value) | Output |
|---|---|---|---|
| `Jet2<T>` single direction | `~3P` | 3 `T`s | one diagonal Hessian entry + one gradient entry along the seeded direction |
| `Jet2Vec<T>` full Hessian | `~(1 + n + n²) P` | one `T` + length-`n` gradient + `n × n` Hessian | value + full gradient + full Hessian |
| `compute_hessian` (FD over reverse) | `~5P · n` (one reverse sweep per direction, plus base) | `O(P)` tape | full Hessian, approximate |
| `compute_full_hessian` (one `Jet2Vec` pass) | `~(1 + n + n²) P` | as above | value + gradient + exact Hessian |
| Forward-over-reverse HVP | `~7P` per direction | `O(P)` tape | one column of `H v` per call, exact |

For `n ≲ 50` the `O(n²)` per-op cost of `Jet2Vec` is dominated by other
constants and the convenience of one-pass exactness wins. Above
`n ≈ 100` the picture flips and you either want per-column seeded
`Jet2` passes, hand-rolled HVP, or (if the Hessian is sparse) edge
pushing.

## Anchored API

- [`xad_rs::Jet2<T>`](https://docs.rs/xad-rs/latest/xad_rs/forward/jet2/struct.Jet2.html).
  - `Jet2::variable(value)`, `Jet2::constant(value)`,
    `Jet2::value()`, `Jet2::first_derivative()`,
    `Jet2::second_derivative()`.
  - Inherent transcendentals: `exp`, `ln`, `sin`, `cos`, `powf`,
    `sqrt`, ... — see the type's rustdoc.
- [`xad_rs::Jet2Vec`](https://docs.rs/xad-rs/latest/xad_rs/forward/jet2vec/struct.Jet2Vec.html).
  - `Jet2Vec::variable(value, i, n)`, `Jet2Vec::constant(value, n)`,
    `Jet2Vec::value()`, `Jet2Vec::gradient()`, `Jet2Vec::hessian()`.
- [`xad_rs::compute_hessian`](https://docs.rs/xad-rs/latest/xad_rs/ops/hessian/fn.compute_hessian.html) — finite-difference over reverse-mode (approximate).
- [`xad_rs::compute_full_hessian`](https://docs.rs/xad-rs/latest/xad_rs/ops/hessian/fn.compute_full_hessian.html) — `Jet2Vec` based (exact). Returns a [`DenseHessian`](https://docs.rs/xad-rs/latest/xad_rs/ops/hessian/struct.DenseHessian.html).

## Worked example

A 3×3 Hessian via `compute_full_hessian`, compared against the analytic
answer in comments.

```rust
use xad_rs::{Jet2Vec, compute_full_hessian};

fn main() {
    // f(x, y, z) = x^2 * y + y^3 + z^2 * x
    //
    // gradient: [2*x*y + z^2,  x^2 + 3*y^2,  2*z*x]
    // Hessian (symmetric):
    //   [[2*y, 2*x,  2*z],
    //    [2*x, 6*y,    0],
    //    [2*z,   0,  2*x]]
    let inputs = vec![
        ("x".to_string(), 1.0_f64),
        ("y".to_string(), 2.0_f64),
        ("z".to_string(), 3.0_f64),
    ];

    let result = compute_full_hessian(&inputs, |v: &[Jet2Vec]| -> Jet2Vec {
        let t1 = &(&v[0] * &v[0]) * &v[1];
        let t2 = &(&v[1] * &v[1]) * &v[1];
        let t3 = &(&v[2] * &v[2]) * &v[0];
        &(&t1 + &t2) + &t3
    });

    let h = &result.hessian;
    let (x, y, z) = (1.0_f64, 2.0, 3.0);
    assert!((h[[0, 0]] - 2.0 * y).abs() < 1e-12); //  4
    assert!((h[[1, 1]] - 6.0 * y).abs() < 1e-12); // 12
    assert!((h[[2, 2]] - 2.0 * x).abs() < 1e-12); //  2
    assert!((h[[0, 1]] - 2.0 * x).abs() < 1e-12); //  2
    assert!((h[[0, 2]] - 2.0 * z).abs() < 1e-12); //  6
    assert!((h[[1, 2]] - 0.0).abs()         < 1e-12); //  0
    // Symmetry:
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert_eq!(h[[i, j]], h[[j, i]]);
        }
    }
}
```

One forward pass yields value, gradient, AND the exact `3 × 3` Hessian
— no finite differences, no per-direction loop in user code. The
`compute_full_hessian` wrapper hands you back a `DenseHessian`, whose
`value`, `gradient`, and `hessian` fields are read **positionally**:
`gradient[i]` and `hessian[[i, j]]` follow the order of the `&[f64]`
inputs you passed in. Apply human-readable names at the call site if you
want them.

## Common pitfalls

- **Mixing seeds across two `Jet2` values.** `Jet2<T>` tracks one seed
  direction. Operating between two `Jet2` values whose seeds came from
  different input slots violates the single-direction invariant
  (`NamedJet2` panics in debug builds on this). Use `Jet2Vec` instead.
- **Reading `Jet2::second_derivative()` after multiple ops.** The
  second-derivative chain rule includes the `2 a_1 b_1` cross-term —
  you cannot just multiply value derivatives and expect to recover
  second derivatives. `Jet2`'s impls do this correctly; users who
  hand-roll second-order propagation often skip the cross-term and
  produce wrong gammas.
- **Pulling values from `Jet2Vec` mid-computation.** The `value()` /
  `gradient()` / `hessian()` accessors are for final readback. Calling
  them in the middle of a tree of expressions and re-wrapping into a
  new `Jet2Vec::constant` discards all the propagated curvature.
- **Building Hessians by finite-differencing first-order AD.** It
  works (`compute_hessian` does this) but you pay for `n` reverse
  sweeps and you eat the `O(1e-7)` floor of the finite-difference step.
  Prefer `compute_full_hessian` when `n` is small enough.
- **Forgetting that `Jet2Vec` storage is `O(n²)`.** For `n = 100`,
  each live `Jet2Vec` carries `~10^4` floats just for the Hessian —
  `~80 KB`. A program with a thousand intermediate values needs `~80
  MB`. Past `n ≈ 50–100`, switch to per-direction passes or HVP.
- **Assuming higher-order = better.** Going past `k = 2` is rare in
  practice. Faà di Bruno's Bell-number cost growth makes
  `k = 3, 4, …` expensive per op, and storage of order-`k` tensors
  scales as `n^k`. Use second order for curvature; use a Taylor
  integrator if you genuinely need higher.

## References

- **Griewank, A. and Walther, A.** *Evaluating Derivatives*, 2nd ed.,
  chapters 5 and 13, for the second-order chain rule and the
  forward-over-reverse construction.
- **Pearlmutter, B. A.** *Fast exact multiplication by the Hessian*,
  Neural Computation, 1994. The classical reference for Hessian-vector
  products at `O(P)` cost.
- **Faà di Bruno, F.** *Note sur une nouvelle formule de calcul
  différentiel*. Quarterly J. Pure Appl. Math. 1 (1857), 359–360. The
  higher-order chain rule.
- **Gower, R. M. and Mello, M. P.** *A new framework for the
  computation of Hessians*. Optimization Methods and Software 27
  (2012), 251–273. Edge pushing for sparse Hessians.
- **Auto-differentiation team**, *XAD: Comprehensive C++ Automatic
  Differentiation*, <https://auto-differentiation.github.io/>, in
  particular its Hessian sample, which this chapter's
  `examples/hessian.rs` parallels.
