# 03 — Reverse mode and taped adjoints

> The computation graph, the adjoint recurrence derived from the
> chain rule, the Baur–Strassen theorem and the cheap-gradient
> principle, a Lagrangian view of reverse mode, the tape memory model,
> vertex elimination orderings, and the tape-reuse model exposed by
> `Tape::record` / `Tape::new_recording`. Anchors: `Tape`, `AReal`,
> `ops::compute_gradient_rev`.

## Overview

Reverse-mode AD computes the gradient of a scalar function
`f: R^n → R` in a single backward sweep, with time cost that is
*independent of `n`* to leading order. The price is memory: the forward
pass must record a **tape** of every elementary operation, and the
reverse sweep replays that tape.

In `xad-rs`, the tape is [`Tape<T>`](https://docs.rs/xad-rs/latest/xad_rs/tape/struct.Tape.html);
the active scalar that records onto it is [`AReal<T>`](https://docs.rs/xad-rs/latest/xad_rs/reverse/areal/struct.AReal.html).
The pair is what you want for any quant-style problem with many inputs
and one output: a 30-factor swap, a yield-curve calibration, a maximum-
likelihood loss.

## Theory

### The computation graph

Take the same evaluation trace from chapter 01:

```text
v1 = x1                # input
v2 = x2                # input
v3 = v1 * v2           # elementary
v4 = sin(v1)           # elementary
v5 = v3 + v4           # elementary
y  = v5                # output
```

Read as a directed acyclic graph (DAG): the inputs are sources, `y` is
the sink, and each interior node `v_k` has edges in from the operands
of the elementary that produced it. Each edge `v_j → v_k` carries the
local partial derivative `c_{kj} = ∂v_k/∂v_j` evaluated during the
forward pass. That partial derivative is the *only* piece of
information about `v_j` the reverse sweep will need at that edge.

We call this the **linearised computation graph**: the topology is the
same as the original Wengert DAG, but the edge weights are the
*numerical* local partials evaluated at `x`. AD is, at root, the problem
of efficiently computing path sums over this weighted graph.

### Adjoints

For a scalar output `y`, the **adjoint** of an intermediate `v_k` is

```math
\bar v_k \;=\; \frac{\partial y}{\partial v_k}.
```

In words: `\bar v_k` is the sensitivity of the final output `y` to the
intermediate value `v_k`. The adjoint of the output is `\bar y = 1` by
definition (`∂y/∂y = 1`). The adjoint of an input `x_j` is the gradient
entry we want:

```math
\bar x_j \;=\; \frac{\partial y}{\partial x_j} \;=\; (\nabla f)_j.
```

In words: the adjoint of input `x_j` is the gradient of `y` with respect
to `x_j`. Filling in the adjoint at every input recovers the full
gradient.

### The adjoint recurrence

Consider any interior node `v_k`. It has edges *out* to children `c`,
each child being a node `v_c` with `v_k` as one of its operands. The
chain rule gives:

```math
\bar v_k \;=\; \sum_{c \,:\, v_k \to v_c}\; \frac{\partial v_c}{\partial v_k}\; \bar v_c
       \;=\; \sum_{c \,:\, v_k \to v_c}\; c_{c\,k}\; \bar v_c.
```

In words: the adjoint of `v_k` is the sum, over every child `v_c` that
consumed `v_k`, of (local partial of the child with respect to `v_k`)
times (adjoint of that child). This is the chain rule, traversed
*backward* through the graph.

This recurrence is well-posed iff we walk the graph in reverse
topological order — process every child of `v_k` *before* `v_k` itself.
The forward pass produced exactly such an order (insertion order on the
tape), so the reverse sweep is a single backward pass through the
recorded tape.

### Why the tape is *correct* (not approximate)

Each edge in the computation graph carries an exact local partial.
Multiplying along edges and summing over all paths from `y` back to
`x_j` gives `∂y/∂x_j` *exactly* (in the same sense as chapter 01 —
modulo floating-point round-off, with zero truncation error). The
reverse sweep implements this multiply-and-sum-over-paths efficiently:
each edge is visited exactly once, and each node's adjoint contribution
to its parents is "scattered" rather than re-derived per path.

> **Theorem (Baur–Strassen 1983; Griewank–Walther 3.4).**
> For any program `F : R^n → R` whose Wengert list has length
> `P = |Φ|`, reverse-mode AD computes the full gradient
> `∇F(x) ∈ R^n` in `O(P)` arithmetic operations — independent of `n`
> — and `O(P)` memory.

The constant hidden in `O(P)` is small (typically 3–5×) and well
characterised: each elementary `φ_k` contributes its primal cost plus a
constant number of multiply-add operations on the reverse pass (one per
operand). This is the **cheap-gradient principle**: gradients cost the
same as function values, up to a constant factor, *no matter how many
inputs there are*.

This is striking. Naively, you might expect the cost of computing `n`
partial derivatives to scale with `n`. Reverse mode shows that for a
scalar output the cost is `O(1)` in `n`. The reason is structural:
forward mode propagates `n` real numbers along each edge (the tangent
vector), while reverse mode propagates a *single* real number along
each edge (the adjoint of the sink). The cost depends on what you
propagate along each edge, not on the number of inputs.

### A Lagrangian view

There is a clean re-derivation of the adjoint recurrence as the KKT
conditions of a constrained optimisation. Consider the problem of
*maximising* `y = v_P` subject to the program constraints
`v_k = φ_k(v_{j_·})` for `k > n`. Introduce Lagrange multipliers
`λ_k` for each constraint:

```math
\mathcal{L}(v_1, \ldots, v_P, \lambda) \;=\; v_P + \sum_{k>n} \lambda_k\,(\varphi_k - v_k).
```

Stationarity in `v_k` for `k < P` gives

```math
\frac{\partial \mathcal{L}}{\partial v_k}
\;=\; -\lambda_k + \sum_{c : v_k \to v_c} \lambda_c \frac{\partial \varphi_c}{\partial v_k}
\;=\; 0.
```

Solving for `λ_k`:

```math
\lambda_k \;=\; \sum_{c : v_k \to v_c} \lambda_c\,\frac{\partial \varphi_c}{\partial v_k},
```

with the boundary condition `λ_P = 1` (from stationarity in `v_P`).
This is *exactly* the adjoint recurrence. Identifying `λ_k = \bar v_k`,
the reverse sweep is a backward solve of the KKT system. This view is
useful when reverse-mode AD shows up in constrained optimisation,
optimal-control problems, and PDE-constrained inverse problems, where
the adjoints quite literally *are* dual variables.

### Vertex elimination ordering

There is a more general framework that subsumes both forward and
reverse mode: **vertex elimination on the linearised graph**.

Given the linearised computation DAG with edge weights `c_{ji} = ∂v_j/
∂v_i`, the Jacobian `J_{ij} = ∂y_i/∂x_j` can be computed by repeatedly
*eliminating* intermediate vertices. To eliminate a vertex `v`:

1. For every pair (parent `p`, child `c`) of `v`, add a new direct edge
   `p → c` with weight `c_{cv} · c_{vp}` (chain rule applied through
   `v`).
2. If `p → c` already exists, *accumulate* the new weight onto the
   existing one (i.e. add — paths superpose).
3. Delete `v` from the graph.

Eliminating all intermediates leaves a bipartite graph from inputs to
outputs whose edge weights are exactly the Jacobian entries.

The order of elimination is a free choice and dramatically affects
cost:

- **Forward mode** corresponds to eliminating vertices in *topological
  order* (sources first).
- **Reverse mode** corresponds to eliminating vertices in *reverse
  topological order* (sinks first).
- Other orderings (mixed mode) can be better than either pure mode for
  particular Jacobian shapes — this is what *cross-country mode* or
  *mixed-mode AD* exploits.

> **Theorem (Naumann 2004).** Finding the optimal vertex elimination
> ordering is NP-hard in general.

So in practice we settle for forward mode (one pass, `O(n)` outputs)
or reverse mode (one pass, `O(m)` outputs), and accept that for
particular Jacobian sparsity patterns there exist better orderings we
will not find automatically. `xad-rs` implements forward and reverse;
mixed-mode is not provided.

### Tape data layout in `xad-rs`

`xad-rs`'s `Tape<T>` uses a compact three-buffer layout (the classic
XAD-style layout):

- `statements: Vec<Statement>` — one entry per recorded *variable* (an
  LHS slot). Each `Statement` stores the LHS slot id and an upper-bound
  pointer into the operations buffer.
- `operations: Vec<Operation<T>>` — a packed stream of `(multiplier,
  operand_slot)` pairs. A statement's operand range is
  `[prev_statement.op_end, self.op_end)`, so operand lookup is O(1) and
  the tape is a single linear scan in both directions.
- `derivatives: Vec<T>` — indexed directly by slot number. After
  `compute_adjoints` this is the adjoint (gradient) vector.

Slots are handed out monotonically by `Tape::register_variable`, so
`derivatives.len() == num_variables` is an invariant — no per-slot
bounds-checks are needed on the hot reverse-sweep loop. Chapter 06
discusses why this layout was chosen over the alternatives.

The forward pass adds three things to the tape per binary op:

- One `Statement` (one push to `statements`),
- Two `Operation`s (two pushes to `operations`, one per operand,
  carrying the local partial as the `multiplier`),
- One slot in `derivatives`.

So the tape memory cost is `O(P)` where `P` is the number of
elementary operations in the forward pass. For a 30-input swap pricer
with `P ≈ 10⁴` operations, the tape is on the order of `10⁴ × 16`
bytes ≈ 160 KB.

The reverse sweep, implemented by `Tape::compute_adjoints`, does a
single linear scan from the end of the tape to the beginning. For each
`Statement` it reads the LHS adjoint, iterates the statement's operands,
and scatters `multiplier * adjoint` to each operand's slot in
`derivatives`. That is the adjoint recurrence above, in code.

### Sparsity of the linearised graph

In real programs the per-statement fan-in (`r_k` in chapter 01) is
small — almost always 2 (binary op) or 1 (unary). The linearised
DAG is therefore *very sparse*: typically `Θ(P)` edges, not `Θ(P²)`.
The tape's flat `(multiplier, operand)` packed stream is exactly the
right representation for this sparsity: each operation pair is one edge,
and the reverse sweep visits each edge once.

For a general statement with `r` operands, the tape stores `r`
operations per statement and the reverse sweep does `r` scatter
updates per statement. Across the whole tape that is `Σ_k r_k = E`
total edge visits, where `E` is the edge count of the linearised DAG.
For typical numerical programs `E = O(P)` with a small constant (≈ 2),
which is where the `~5P` figure in the cost model below comes from.

### `AReal` records onto whichever tape is active

`AReal<T>` is the active scalar. It wraps a primal `value` and a tape
`slot`. Every binary operator (`+`, `-`, `*`, `/`, ...) on `AReal`:

1. Reads the thread-local active-tape pointer.
2. Allocates a fresh slot via `tape.register_variable()`.
3. Pushes the operands' contributions onto the operations buffer with
   the correct local-partial multipliers.

The thread-local pointer is set by `Tape::activate()` (or the RAII
guard `Tape::activate_guard()`); a null pointer encodes "no active
tape", in which case `AReal` arithmetic produces the right primal
value but records nothing — useful for warm-up passes or for code paths
that do not need derivatives.

The standard workflow is:

```text
let mut tape = Tape::<f64>::new(true);
tape.activate();

let mut x = AReal::new(...);
let mut y = AReal::new(...);
AReal::register_input(&mut [x, y], &mut tape);   // hand out slots, no statement emitted

let mut z = ... f(x, y) ...;                     // builds the tape statement-by-statement
AReal::register_output(&mut [z], &mut tape);

z.set_adjoint(&mut tape, 1.0);                   // seed the reverse sweep
tape.compute_adjoints();                          // sweep

let dz_dx = x.adjoint(&tape);
let dz_dy = y.adjoint(&tape);

Tape::<f64>::deactivate_all();
```

`register_input` and `register_output` do *not* emit statements — inputs
never appear on the LHS of an operation, and the final output is
typically already on the tape. Their job is just to hand out tape
slots so the adjoint readback finds the right entry in `derivatives`.

### Tape reuse: `record` / `new_recording`

The tape is append-only during a forward pass. Rather than a
partial-rollback marker, the supported lifecycle hook is whole-recording
reuse: `Tape::new_recording` resets the tape's statement and operation
buffers **in place**, keeping the already-allocated capacity, and
`Tape::record` wraps activate-and-reset in an RAII `TapeGuard` that
deactivates on drop — including on unwind from a panic.

This is the lever that matters for the workload reverse mode is usually
run on: many small valuations rather than one huge one. Reusing a tape
across valuations instead of constructing a fresh `Tape::new` each time
is worth ~2.4× on that shape, because allocation churn was about 60% of
the runtime. `Tape::with_capacity` sizes the buffers up front when the
recording size is known.

Adjoints are a separate buffer from the statement and operation buffers,
so call `clear_derivatives` if you also want a clean adjoint vector.

### Checkpointing — *not implemented*, mentioned for completeness

For very long forward passes (think: `P ≈ 10^9` operations, GB-scale
tapes), the `O(P)` memory cost of a full tape becomes untenable.
**Checkpointing** is the standard remedy: store only a sparse set of
intermediate states, and re-run forward segments on demand during the
reverse sweep. This trades extra forward work for less memory.

The optimal checkpoint placement for a uniform serial computation is
the *binomial checkpointing* schedule of Griewank's *revolve* algorithm
(Griewank & Walther 2000): for a forward pass of `L` steps and `c`
allowed snapshots, revolve achieves memory `O(c)` and an additional
forward-work multiplier of `O(log_c L)` — i.e. a logarithmic-in-`L`
slowdown for a fixed checkpoint budget.

`xad-rs` does *not* currently implement automatic checkpointing. If
you hit a memory wall, the practical options are:

- Manual decomposition: split your forward pass into chunks, compute
  the gradient chunk-by-chunk, and accumulate by hand.
  `Tape::new_recording` resets the tape allocation in place.
- Switch to forward mode if `n` is small enough — forward mode has
  `O(1)` memory per direction.
- Use the upstream C++ XAD's checkpointing support (see
  <https://auto-differentiation.github.io/>); the upstream library has
  this feature today.

This is a candidate follow-up change for `xad-rs` if there is user
demand.

## Cost model

Let `P` be the flop count of the primal pass and `n` the input
dimension. For a scalar-output function (`m = 1`):

| Operation | Time | Memory |
|---|---|---|
| Forward pass (primal only) | `P` | O(1) extra |
| Forward pass on `AReal` (records the tape) | `~3P` | `O(P)` tape |
| Reverse sweep (`compute_adjoints`) | `~2P` | (reuses the tape) |
| **Total gradient via reverse mode** | `~5P` | `O(P)` |

The asymptotic win: this is `~5P` regardless of `n`. Against *one
direction per forward pass* (`Jet1`, `n` passes) the crossover is at
`n ≈ 4` in practice; for `n = 30`, reverse is ~8× faster than `Jet1 × 30`
(measured, `examples/jetk_gradient.rs`), and for `n = 100` closer to ~20×.

The K-lane forward mode changes the constant, not the asymptote. `JetK<f64,
K>` evolves `K` lanes per pass, so a gradient costs `⌈n/K⌉` passes of a
value `K + 1` scalars wide — nominally `~(1 + n) P` in flops, but with
`O(K)` memory per live value and no tape, and the lane loops vectorise.
Measured on the same machine (`compute_gradient_fwd_k`, Apple M4 Pro): on a
six-input body one `JetK<8>` pass is 2.4× faster than a warm-tape reverse
sweep, and at `n = 30` two `JetK<16>` passes still match the sweep. The
practical rule is `K ≈ n` rounded up to 4/8/16 for `n ≲ 16`; beyond that,
reverse — the tape's `~5P` is flat in `n` and the lane count is not.

For a vector-output function (`m > 1`), reverse mode needs *one sweep
per output you want adjoints of*. If you want `m` rows of the Jacobian,
the cost is `m × (5P)` — and once `m > n`, you should switch to
forward.

The full bound (with proved constants) is in Griewank & Walther
Theorem 3.4.

## Anchored API

- [`xad_rs::Tape<T>`](https://docs.rs/xad-rs/latest/xad_rs/tape/struct.Tape.html) — the recording tape.
  - `Tape::new(_activate)`, `Tape::activate()`, `Tape::activate_guard()` (RAII),
    `Tape::deactivate_all()`, `Tape::new_recording()`.
  - `Tape::compute_adjoints()`, `Tape::derivative(slot)`,
    `Tape::clear_derivatives()`.
  - `Tape::record()` (RAII), `Tape::with_capacity(..)` — tape reuse.
  - `Tape::num_variables()`, `Tape::num_operations()`,
    `Tape::num_statements()`, `Tape::memory()` — introspection.
- [`xad_rs::AReal<T>`](https://docs.rs/xad-rs/latest/xad_rs/reverse/areal/struct.AReal.html) — active reverse-mode scalar.
  - `AReal::new(value)`, `AReal::value()`,
    `AReal::register_input(&mut [..], &mut tape)`,
    `AReal::register_output(&mut [..], &mut tape)`.
  - `AReal::set_adjoint(&mut tape, v)`, `AReal::adjoint(&tape)`.
- [`xad_rs::compute_gradient_rev`](https://docs.rs/xad-rs/latest/xad_rs/ops/derivative/fn.compute_gradient_rev.html) — value and full gradient from one sweep, with the tape managed for you.
- [`xad_rs::math::ad`](https://docs.rs/xad-rs/latest/xad_rs/math/ad/index.html) — AD-aware transcendentals that record on the active tape.

## Worked example

A four-input pricer with a `sin` and an `exp` in it: build the tape,
sweep, read the gradient.

```rust
use xad_rs::{AReal, Tape, math};

fn main() {
    // f(a, b, c, d) = a * b + sin(c) + exp(d)
    // grad = (b, a, cos(c), exp(d))
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut a = AReal::new(2.0);
    let mut b = AReal::new(3.0);
    let mut c = AReal::new(0.5);
    let mut d = AReal::new(1.0);
    let mut inputs = [a.clone(), b.clone(), c.clone(), d.clone()];
    AReal::register_input(&mut inputs, &mut tape);
    let [ra, rb, rc, rd] = inputs;
    a = ra; b = rb; c = rc; d = rd;

    let mut y = &(&a * &b) + math::ad::sin(&c) + math::ad::exp(&d);
    AReal::register_output(std::slice::from_mut(&mut y), &mut tape);
    y.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    let g = (a.adjoint(&tape), b.adjoint(&tape), c.adjoint(&tape), d.adjoint(&tape));
    assert!((g.0 - 3.0).abs() < 1e-12);                 // ∂/∂a = b
    assert!((g.1 - 2.0).abs() < 1e-12);                 // ∂/∂b = a
    assert!((g.2 - 0.5_f64.cos()).abs() < 1e-12);       // ∂/∂c = cos(c)
    assert!((g.3 - 1.0_f64.exp()).abs() < 1e-12);       // ∂/∂d = exp(d)

    Tape::<f64>::deactivate_all();
}
```

One forward pass builds the tape; one backward pass `compute_adjoints`
fills the gradient. Adding a fifth, sixth, or thirtieth input would not
change the wall-clock cost of the reverse sweep meaningfully — the
sweep is bounded by the tape length, not the input count. That is the
cheap-gradient principle in code.

## Common pitfalls

- **Forgetting `Tape::activate()`.** A freshly constructed `Tape` is
  *not* active. `AReal` arithmetic on an inactive tape produces correct
  primal values but records nothing; the adjoint readback then returns
  zero. The `_activate` argument to `Tape::new` mirrors the upstream
  C++ XAD API shape and is ignored.
- **Two tapes active on one thread.** `Tape::activate()` panics if a
  tape is already active. Use `Tape::activate_guard()` for RAII-scoped
  activation in long-running code paths.
- **Calling `register_input` after the forward pass.** Inputs must be
  registered *before* they are used in a recorded operation; otherwise
  the operation's operand slot is `INVALID_SLOT` and adjoint
  contributions are silently dropped on that edge.
- **Moving an active tape.** The active-tape pointer is a raw pointer
  to a stack/heap address. If you `std::mem::forget` a tape or shadow
  it with another tape at the same address by accident, the TLS
  pointer becomes dangling. `Tape::deactivate_all()` is the recovery
  hook; or use `activate_guard()` to bind activation lifetime to the
  scope.
- **Re-using a tape across iterations without clearing derivatives.**
  Adjoints accumulate. If you run the forward pass twice and call
  `compute_adjoints` twice without `clear_derivatives` in between, the
  second gradient is the sum of the two, not the second alone. Use
  `Tape::new_recording()` to fully reset, or `clear_derivatives()` for
  just the adjoint buffer.
- **Using `Tape::compute_adjoints` and then continuing the forward
  pass.** Don't. Once you've swept, the adjoints are written; further
  forward ops will record onto the tape but the bookkeeping invariant
  between statements and derivative slots is now mid-sweep. Sweep last.

## References

- **Baur, W. and Strassen, V.** *The complexity of partial
  derivatives*, Theoretical Computer Science 22 (1983), 317–330. The
  original cheap-gradient bound behind reverse mode's `O(1)`-in-`n`
  time.
- **Griewank, A. and Walther, A.** *Evaluating Derivatives*, 2nd ed.,
  chapter 3 ("The reverse, or adjoint, mode") for the full adjoint-
  recurrence derivation and the proof of Theorem 3.4. Chapter 12
  covers checkpointing and the revolve algorithm.
- **Naumann, U.** *Optimal accumulation of Jacobian matrices by
  elimination methods on the dual computational graph*, Mathematical
  Programming 99 (2004), 399–421. The NP-hardness of optimal vertex
  elimination ordering.
- **Naumann, U.** *The Art of Differentiating Computer Programs*,
  chapter 3, for the source-transformation view of reverse mode (which
  `xad-rs` does not use — it operator-overloads — but which is
  informative background).
- **Auto-differentiation team**, *XAD: Comprehensive C++ Automatic
  Differentiation*, <https://auto-differentiation.github.io/>, in
  particular its tape layout and checkpointing pages.
