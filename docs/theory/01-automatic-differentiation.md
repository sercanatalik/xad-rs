# 01 — Automatic differentiation as a discipline

> What AD is, why it computes *exact* derivatives, how it differs from
> numerical bumping and from symbolic differentiation, the
> Wengert-list formalism that underlies every mode, and why the
> `Real` trait can present every AD mode as a drop-in replacement for
> `f64`.

## Overview

Automatic differentiation (AD) is a way to compute the derivative of a
numerical program — *the actual derivative the math defines, to machine
precision* — by piggy-backing on the program's evaluation. AD is
neither a numerical approximation (no step size, no truncation error)
nor a symbolic manipulation (no formula explosion, no manipulation of
expression trees by hand). It is closer to a structural reinterpretation
of arithmetic: every elementary operation in the program is augmented
with a small extra computation that propagates derivative information.

`xad-rs` ships two flavours of AD that the rest of this documentation
will develop in detail:

- **Forward mode** (chapter 02) — augment each scalar with a tangent
  vector, propagate forward through the program, read derivatives from
  the output's tangent. Types: `Jet1`, `JetK`, `Jet2`, `Jet2Vec`.
- **Reverse mode** (chapter 03) — record a *tape* of every elementary
  operation, then replay it backwards once to obtain the gradient. Types:
  `Tape`, `AReal`.

Both modes compute the same mathematical object — derivatives — but
their cost profiles are dual to each other, so the right choice depends
on the shape of your problem. The job of this chapter is to set up the
language to compare them in the next two.

## Theory

### What "derivative of a program" means

Let `F: R^n -> R^m` be a function implemented by a (finite) Rust program
of elementary operations: arithmetic, transcendentals (`exp`, `sin`,
...), and control flow. At a given input `x`, `F` is composed of a
finite sequence of elementary calls; for almost every `x` (in the
piecewise-smooth sense formalised below) the program coincides with a
smooth function on a neighbourhood of `x`, and the chain rule applies.
The Jacobian `J(x) = ∂F/∂x` is the m × n matrix of partial derivatives,
and the gradient (in the m = 1 case) is its single row, transposed.

```math
J_{ij}(x) \;=\; \frac{\partial F_i}{\partial x_j}(x).
```

In words: entry `(i, j)` of the Jacobian is the partial derivative of
the `i`-th output with respect to the `j`-th input, evaluated at `x`.

AD computes `J(x)` (or a Jacobian-vector product, or a vector-Jacobian
product) by walking the program's elementary operations exactly once
more — never differentiating across branches it didn't take, never
inflating expressions symbolically, never picking a step size.

### The Wengert list (evaluation trace, formalised)

The mental model shared by every AD mode is the **Wengert list**
(Wengert 1964), also called the *evaluation trace* or *single-assignment
form* of the program: a finite, ordered sequence of single-assignment
statements

```math
v_k \;=\; \varphi_k(v_{j_1},\,v_{j_2},\,\ldots,\,v_{j_{r_k}}),
\qquad j_1, \ldots, j_{r_k} \prec k,
```

where each `v_k` (for `k > n`) is the result of one *elementary*
operation `φ_k` applied to earlier values, and each operand index
`j_l` strictly precedes `k`. The `n` input slots `v_1, …, v_n` are the
program's inputs; the last `m` slots are designated outputs.

In words: every numerical program — no matter how complex — can be
flattened, at a given input, into a finite linear sequence of
`v_k = φ_k(…)` assignments where each step is one of a small library of
elementary operations and depends only on already-computed slots.

Three properties of the Wengert list matter for AD:

1. **Locality.** Each `φ_k` has a *known* local Jacobian
   `(∂φ_k / ∂v_{j_l})_l` that can be evaluated in constant time alongside
   the primal step. This is the only piece of derivative information AD
   needs at each elementary call.
2. **Acyclicity.** The dependence relation `j_l ≺ k` is a strict partial
   order; the Wengert list is a topological sort of a directed acyclic
   graph (DAG), the **computation graph**.
3. **Composability.** The full Jacobian of the program is the product
   along DAG paths of the local Jacobians; the chain rule gives the
   composition law explicitly.

For a program

```text
v1 = x1                # input
v2 = x2                # input
v3 = v1 * v2           # elementary
v4 = sin(v1)           # elementary
v5 = v3 + v4           # elementary
y  = v5                # output
```

the Wengert list is `v1, v2, v3, v4, v5` and the computation DAG has
edges `v1 → v3`, `v2 → v3`, `v1 → v4`, `v3 → v5`, `v4 → v5`. Forward
mode walks this DAG from `v1` upward, carrying tangent components;
reverse mode walks from `y` downward, carrying adjoints. The two modes
correspond to two different *orderings* for accumulating the same product
of edge-local Jacobians along the DAG, an observation we develop in
chapters 02–03.

### Three ways to differentiate a program

| Method | Idea | Error | Cost in n, m | Notes |
|---|---|---|---|---|
| **Numerical (bumping)** | Approximate `∂F/∂x_j ≈ (F(x + h e_j) − F(x)) / h` | Yes — truncation + cancellation | `n + 1` forward passes (forward diff) or `2n` (central diff) | Cheap to implement, fragile near singularities, step-size tuning is delicate (chapter 07 quantifies the trade-off) |
| **Symbolic** | Construct a closed-form expression for `∂F/∂x_j`, then evaluate | None *in principle*; in practice subject to numerical evaluation error | Can explode (expression swell) | Useful for hand-derived greeks; impractical for long programs |
| **Automatic** | Augment each elementary op with its local derivative; chain rule does the rest | None (machine precision) | One pass + bookkeeping; either `O(n)` (forward) or `O(m)` (reverse) | The subject of this library |

The "no truncation error" claim deserves emphasis: AD does not pick a
step size `h`. It applies the chain rule directly to the elementary
operations as the program runs, so the only floating-point error in the
output of AD is the same kind of round-off you'd see in the primal
computation — never the `O(h)` truncation error of finite differences,
and never the catastrophic cancellation of small `h`. A formal error
analysis appears in chapter 07.

### Exactness, stated precisely

**Claim (Wengert-list correctness).** Let `F` be a program that, at the
point `x`, evaluates a Wengert list `(v_k, φ_k, j_·)` whose elementary
operations `φ_k` are each `C¹` on a neighbourhood of their operand
values. Then forward and reverse mode, applied to the same list, both
return the exact Jacobian of the program *as a piecewise-smooth function*
at `x`, modulo only floating-point round-off in the local-Jacobian
arithmetic.

The proof is the chain rule, applied edge-by-edge along the DAG. The
formal statement — including the precise definition of "the function
the program represents" and bounds on round-off propagation — is
Griewank & Walther, Theorems 3.1 and 3.4. The non-trivial qualifier is
**piecewise-smooth**: branches break local smoothness, and we address
that next.

### Branches, kinks, and the piecewise-smooth setting

Most numerical programs contain branches: `if x > 0.0 { f1(x) } else
{ f2(x) }`, `x.abs()`, `x.max(0.0)`, `min(a, b)`. AD does not
differentiate the *meta-program* (the function that selects between
branches). It differentiates **whichever branch the program actually
took at `x`**.

Formally: the program induces a partition of input space `R^n` into
finitely many open regions on each of which the program coincides with a
single smooth function. On the interior of each region, AD returns the
correct gradient of *that* smooth function. At a region boundary (a
"kink"), the gradient has a jump discontinuity, and AD silently picks
one side — typically the side the branch test fell on. This is correct
in the sense of subgradients and Clarke generalised derivatives
(Khan & Barton 2013), but it is not the gradient of the program's
*continuous* interpretation, because no such gradient exists at the
kink.

Two ways to handle this:

- **Accept the subgradient.** For optimisation, this is fine: gradient
  descent on a piecewise-smooth function with kinks (ReLU networks,
  hinge losses, payoff caps) is well-defined and convergent under mild
  assumptions.
- **Smooth the program.** For sensitivities that need to vary
  continuously with inputs (greeks across a strike, risk near a
  barrier), replace `max`, `min`, `abs` with smooth surrogates. `xad-rs`
  ships `math::ad::smooth_max`, `smooth_min`, `smooth_abs` exactly for
  this case.

### The `Real` trait — one abstraction over every mode

`xad-rs` exposes a single trait, [`Real`](https://docs.rs/xad-rs/latest/xad_rs/real/trait.Real.html),
that every mode implements: `f64`, `AReal<f64>` (reverse), `Jet1<f64>`
(forward, single direction), and `Jet2<f64>` (forward, single direction,
second order). Mode-agnostic numerical code is written once against
`R: Real` and instantiated at the call site against whichever concrete
type matches the problem shape.

```math
\text{generic body } \in \;\{R: \text{Real}\} \;\;\xrightarrow{\text{instantiate}}\;\;
\{\,\text{f64},\; \text{AReal<f64>},\; \text{Jet1<f64>},\; \text{Jet2<f64>}\,\}
```

In words: a single generic function body — say a quadratic, a swap
present-value, or a Black–Scholes pricer — can be evaluated against the
no-AD type (`f64`), the reverse-mode type (`AReal<f64>`), the
first-order forward-mode type (`Jet1<f64>`), or the second-order
forward-mode type (`Jet2<f64>`), without code duplication.

There is a category-theoretic way to phrase this seam: each `Real`
instance is the carrier of a ring (with the usual `+, *, neg, inv`) and
the elementary functions are *ring homomorphisms* augmented with
derivative metadata. The `Real::value()` projection is a ring
homomorphism `R → R_passive` — it forgets the derivative data and
returns the underlying primal. We make this view explicit in chapter 02
when we describe forward mode as evaluating the original program in the
dual-number ring `R[ε]/(ε²)`.

The trait requires the usual scalar operations (`+`, `-`, `*`, `/`, and
the elementary transcendentals: `ln`, `exp`, `sqrt`, `sin`, `cos`,
`powf`, `powi`), the conversions `From<f64>` and `From<i32>`, the numeric
identities `zero` / `one`, and the fused aggregates `sum`, `dot`,
`weighted_sum`, `weighted_dot`.

The Gaussian family — `erf`, `erfc`, `norm_pdf`, `norm_cdf`,
`inv_norm_cdf` — is on `Real` itself rather than on a separate extension
trait, so an option-style pricer is ordinary `R: Real` code. The unary
method set is *generated* from the crate's single derivative table
(`src/elementaries.rs`), the same table that stamps the mode-specific
surfaces, so the trait cannot drift out of step with them.

## Cost model

We will quote costs in units of "primal flop count" — the number of
floating-point operations the original program would execute. Call this
`P` (sometimes called the *time complexity* of the primal in Griewank &
Walther's notation, denoted `TIME(F)`).

| Mode | Time to get one Jacobian-vector product (one column of `J ⋅ v`) | Time to get one vector-Jacobian product (one row of `u^T ⋅ J`) | Memory |
|---|---|---|---|
| Numerical (forward differences) | `2P` | not directly applicable | O(1) |
| Forward AD | `cP` for a small constant `c` (≈ 2–4) | not directly applicable | O(1) extra |
| Reverse AD | not directly applicable | `cP` (constant `c` ≈ 3–5) | **O(P) for the tape** |

The last column matters. Forward mode is essentially free in memory;
reverse mode pays an `O(P)` memory price for the recorded tape. The
asymptotic comparisons in chapters 02 and 03 build on this base rate.

The full complexity bounds (with explicit constants and operation
counts) are proved in Griewank & Walther, chapter 3. The headline
results are:

- **Cheap gradient principle.** For any `F : R^n → R`, reverse-mode AD
  computes the full gradient `∇F(x) ∈ R^n` in time bounded by a small
  constant (typically 3–5) times the primal time — *regardless of `n`*.
  This is sometimes called the *Baur–Strassen theorem* (chapter 03)
  after its original formulation.
- **Cheap JVP principle.** For any `F : R^n → R^m`, forward-mode AD
  computes a Jacobian-vector product `J(x) v` in time bounded by a
  small constant (typically 2–4) times the primal time — *regardless of
  `m`*.

The crossover is therefore at `n ~ m`. For `n ≫ m` (the typical quant
setup: many risk factors, one PV), reverse wins. For `n ≪ m` (one
control variable, many outputs), forward wins. In `xad-rs` the
empirical crossover sits near `n ≈ 4`; chapter 03 explains why.

## Anchored API

This chapter does not anchor any single type — its job is to set up
shared vocabulary. The trait is the anchor:

- [`xad_rs::Real`](https://docs.rs/xad-rs/latest/xad_rs/real/trait.Real.html) — the unified active-scalar trait, including the Gaussian family (`erf` / `norm_cdf` / ...).
- [`xad_rs::Passive`](https://docs.rs/xad-rs/latest/xad_rs/passive/trait.Passive.html) — the bound on the underlying (storage) scalar.

## Worked example — same body, three instantiations

The same generic function evaluated against three concrete types
(`f64`, `Jet1<f64>`, `AReal<f64>`) returns the same primal value, and
the active types additionally yield the derivative. This is the seam
the rest of the documentation builds on.

```rust
use xad_rs::prelude::*;
use xad_rs::Tape;

fn f<R: Real>(x: &R) -> R {
    // f(x) = x^2 + 3x + 1
    x.clone() * x.clone() + R::from(3.0_f64) * x.clone() + R::from(1.0_f64)
}

fn main() {
    let x_pass = 2.0_f64;
    let v_pass = f(&x_pass);            // primal only
    assert_eq!(v_pass, 11.0);           // 4 + 6 + 1

    // Forward mode, one direction. Seed tangent = 1.
    let x_fwd = Jet1::<f64>::new(2.0, 1.0);
    let v_fwd = f(&x_fwd);
    assert_eq!(v_fwd.value(), 11.0);
    assert_eq!(v_fwd.derivative(), 7.0); // f'(2) = 2*2 + 3

    // Reverse mode. Activate a tape, register x, run, sweep, read.
    let mut tape = Tape::<f64>::new(true);
    tape.activate();
    let mut x_rev = AReal::<f64>::new(2.0);
    AReal::register_input(std::slice::from_mut(&mut x_rev), &mut tape);
    let mut y = f(&x_rev);
    AReal::register_output(std::slice::from_mut(&mut y), &mut tape);
    y.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert_eq!(y.value(), 11.0);
    assert_eq!(x_rev.adjoint(&tape), 7.0);
    Tape::<f64>::deactivate_all();
}
```

The function body `f` is identical in all three call sites — that is the
payoff of programming against `R: Real`. Forward mode tells you the
derivative by reading `Jet1::derivative()`; reverse mode tells you by
sweeping the tape and reading `AReal::adjoint(&tape)`. Both return `7.0`
to machine precision, because both implement *the same chain rule*
mechanically — they just walk the evaluation trace in opposite
directions.

## Where the next chapters go

- Chapter 02 develops **forward mode** algebraically as evaluation over
  the dual-number ring `R[ε]/(ε²)`, proves the chain-rule consistency
  there, and generalises to seed matrices via `JetK`.
- Chapter 03 develops **reverse mode** as a reverse traversal of the
  computation DAG, derives the adjoint recurrence, and states the
  Baur–Strassen complexity bound.
- Chapter 04 lifts both to **second order** via k-jets `R[ε]/(ε^{k+1})`
  and explains the multi-direction Hessian.
- Chapter 05 covers the **named-variable** ergonomics that make these
  modes usable for real quant pricers.
- Chapter 06 looks under the hood at **representation choices** — tape
  layout, operator overloading vs source transformation.
- Chapter 07 quantifies **numerical accuracy** of AD versus finite
  differences and explains why AD has no step-size dilemma.

## Common pitfalls

- **Confusing AD with finite differences.** AD has no step size. If you
  see a tutorial that picks `h = 1e-8` and talks about "AD", it is
  finite differences, not AD. Chapter 07 makes the accuracy gap
  quantitative.
- **Branches that depend on the input.** AD differentiates the branch
  you actually took. If your function uses `if x > 0.0 { ... }`, the
  derivative is correct on each side of `0` but discontinuous *at* `0`.
  Use the smoothing helpers in `math::ad` if you need continuity, or
  accept the subgradient if you do not.
- **Calling `.value()` on intermediate results.** The whole point of
  the active types is that they carry derivative metadata. If you
  flatten an intermediate value back to `f64`, you cut the chain rule
  there and the downstream derivative is wrong (zero, usually). The
  `value()` accessor is for *final* readback only.
- **Activating a tape twice on the same thread.** Reverse mode uses a
  thread-local active-tape pointer; two `Tape::activate()` calls on the
  same thread without a `deactivate()` between them panic. Use
  `Tape::activate_guard()` for RAII-scoped activation.
- **Picking the wrong mode for the problem shape.** This is the most
  common cost mistake. If `n = 1` and `m = 1000`, do not reach for the
  tape — use `Jet1`. If `n = 1000` and `m = 1`, do not reach for
  forward mode at all — use the tape. Chapters 02 and 03 develop the
  cost model in detail.

## References

- **Wengert, R. E.** *A simple automatic derivative evaluation program*.
  Communications of the ACM 7 (1964), 463–464. The original
  evaluation-trace formulation now called the *Wengert list*.
- **Griewank, A. and Walther, A.** *Evaluating Derivatives: Principles
  and Techniques of Algorithmic Differentiation*, 2nd ed. SIAM, 2008.
  Chapters 1–3 for the evaluation-trace formalism, exactness proof, and
  full complexity bounds.
- **Baur, W. and Strassen, V.** *The complexity of partial derivatives*.
  Theoretical Computer Science 22 (1983), 317–330. The cheap-gradient
  principle in its original form.
- **Naumann, U.** *The Art of Differentiating Computer Programs*, SIAM,
  2012. Chapters 2–4 for the program-transformation view of AD and a
  practical taxonomy of modes.
- **Khan, K. A. and Barton, P. I.** *Evaluating an element of the
  Clarke generalized Jacobian of a composite piecewise differentiable
  function*. ACM TOMS 39 (2013), 23. The formal framework for AD on
  non-smooth programs.
- **Auto-differentiation team**, *XAD: Comprehensive C++ Automatic
  Differentiation*. <https://auto-differentiation.github.io/>. Upstream
  C++ library; semantically related but Rust idioms differ.
