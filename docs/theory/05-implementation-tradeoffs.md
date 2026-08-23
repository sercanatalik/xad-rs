# 05 — Implementation tradeoffs

> Operator overloading versus source-to-source transformation, the
> three-buffer packed tape layout `xad-rs` inherits from upstream
> XAD, the rationale for thread-local active-tape pointers, the
> arithmetic of slot allocation, and where Rust's affine type system
> helps and where it constrains us. Anchors: `Tape`, `AReal`,
> `Statement`, `Operation`.

## Overview

The mathematics of AD is mode-agnostic, but its *implementation* is
not. Two implementations of the same forward-mode chain rule can differ
by 10× in wall-clock cost. This chapter explains the implementation
decisions behind `xad-rs`:

- Why `xad-rs` uses **operator overloading** (overloading `+`, `*`,
  …) rather than **source-to-source transformation** (rewriting the
  source code at compile time).
- Why the tape uses a **three-buffer packed layout** (statements,
  operations, derivatives) instead of a tree of nodes.
- Why active-tape state is **thread-local**, what that buys us, and
  what it costs in terms of API ergonomics.
- How Rust's type system **helps** (the `Real` trait, `Drop` for
  RAII deactivation) and where it **constrains** (no operator
  overloading on borrows, no implicit `Copy` for our active types).

The goal is to give you enough of the implementation's reasoning that
the API choices in chapters 02–05 stop feeling arbitrary.

## Theory

### Operator overloading vs source transformation

There are two mainstream ways to implement AD over a host language:

1. **Operator overloading.** Replace the scalar type with an *active*
   type whose arithmetic operators chain-rule their inputs. The source
   code is unchanged; only the type changes. C++ examples: XAD and
   other operator-overloading AD libraries. Julia: dual-number AD
   libraries. Rust: `xad-rs`.
2. **Source-to-source transformation.** Parse the source code, run an
   AD compiler pass that generates a *new* program implementing the
   derivative, and compile that. This is the approach taken by C and
   Fortran source-transformation AD tools. The user writes plain `f64`
   and gets back a generated gradient routine in plain `f64`.

The tradeoffs are well established:

| Concern | Operator overloading | Source transformation |
|---|---|---|
| Implementation cost | low — one library | high — a compiler pass |
| Runtime overhead | moderate — virtual-function-like dispatch, allocations | zero in principle — emitted code is just `f64` |
| Composability with the host language | excellent — works inside any container, closure, generic | poor — the AD compiler must understand the host's full surface area |
| Higher-order modes | natural via type composition (e.g. `Jet1<AReal<f64>>`) | requires re-running the compiler pass |
| Cross-language and cross-vendor portability | high — needs only an operator-overloading-friendly language | low — one compiler pass per language |
| Debuggability | normal — set breakpoints in the user code | hard — debugging generated code |

`xad-rs` takes the operator-overloading path. The decisive factor is
**composability**: a Rust pricer body lives inside generic functions,
trait objects, closures, iterator chains, and async tasks. A
source-transformation pass would have to track types and lifetimes
through all of that, which is doable in principle but a multi-year
compiler-engineering investment. Operator overloading buys us
correctness on day one across the entire language surface.

The cost is runtime overhead: every elementary operation on `AReal` or
`Jet1` does a small constant amount of work beyond `f64` — a heap-
pointer dereference for the active tape, a couple of pushes onto vecs,
sometimes a bounds check. For typical pricer code this overhead is a
factor of 3–5× over plain `f64`, which the cheap-gradient principle
(chapter 03) more than pays back as soon as `n > 1`. For very tight
inner loops where a 3× per-op overhead is unacceptable, source
transformation wins; we punt to upstream XAD or hand-rolled adjoints
in that regime.

### Why a packed three-buffer tape

A naive reverse-mode tape stores one node per intermediate value, each
node holding `Vec<(operand_ptr, multiplier)>`. This is fine but wastes
memory: a node header (`Vec` metadata, possibly a vtable, possibly
padding) per intermediate value, on top of the operand list.

XAD's contribution to the AD-implementation literature is the
**packed three-buffer layout**, which `xad-rs` adopts verbatim:

- `statements: Vec<Statement>` — one entry per recorded variable
  (LHS slot). `Statement` is just `{ lhs_slot: u32, op_end: u32 }`:
  the LHS slot id and the exclusive end of this statement's operand
  range in the operations buffer. 8 bytes per intermediate.
- `operations: Vec<Operation<T>>` — a packed stream of `(multiplier:
  T, operand_slot: u32)` pairs. A statement's operand range is
  `[prev_statement.op_end, self.op_end)`, found by binary or implicit
  scan. For `T = f64` and a 4-byte slot, 12 bytes per operand.
- `derivatives: Vec<T>` — indexed directly by slot. The reverse
  sweep writes adjoints here.

Total memory per binary op: 1 statement (8 B) + 2 operations (24 B) +
1 derivative slot (8 B) = **40 B**. A non-packed alternative with one
`Vec` per node, even with small-vec optimisations, runs 64–96 B per
binary op.

The packed layout has two further wins:

1. **Cache locality on the reverse sweep.** Statements and operations
   are scanned linearly from the end to the beginning. Modern CPUs
   prefetch linear scans aggressively; an array-of-structs tape gets
   ~2× the IPC of a tree-of-nodes tape on the same data.
2. **No allocator pressure during the forward pass.** All three vecs
   double in capacity geometrically; if you pre-size the tape (via
   `Tape::with_capacity` or `Tape::new_recording` after a warm-up
   pass), the forward pass does *zero* allocations.

The bookkeeping cost is that operand ranges are computed implicitly
from consecutive `op_end` values. This is fine because the tape is
append-only during recording; the reverse sweep just walks
`statements` in reverse and uses `op_end[i-1] .. op_end[i]` as the
operand range for statement `i`.

### Slot allocation arithmetic

Each `AReal::new(...)` call does *not* allocate a slot. Slots are
allocated only when:

- `AReal::register_input(&mut [..], &mut tape)` is called on inputs;
- An *operation* produces an intermediate (`a + b`, `f(x)`, etc.).

The reason for this split is correctness: an `AReal` constructed
before the tape is active should still produce correct primal values
when used arithmetically later. If we allocated a slot on
construction, we'd have to either reject pre-tape constructions or
emit a "lazy slot" that gets fixed up — both unpleasant. Deferring
slot allocation to the first operation keeps the design clean.

The slot is a `u32`. With 32 bits we can record up to `2^32 - 1 ≈ 4.3
billion` intermediate values per tape. At 40 bytes per op, that's
~170 GB of tape, comfortably past anyone's working-set capacity, so
`u32` is the right tradeoff against `u64`.

### Why a thread-local active-tape pointer

`AReal` operators need to know *which tape* to record onto. Three
candidates:

1. **Pass the tape as an explicit argument.** `tape.mul(&a, &b)`
   instead of `a * b`. Robust, type-safe, and *unergonomic*:
   `tape.mul(&tape.add(&a, &b), &tape.sin(&c))` is unreadable. Rules
   out generic numeric code over `Real`.
2. **Store a tape pointer inside every `AReal`.** Works, but inflates
   `AReal` from 16 B (value + slot) to 24 B (value + slot + tape
   pointer), and forces every `AReal` operation to compare tape
   pointers (or fail UB-style if mixed).
3. **Thread-local active-tape pointer.** `Tape::activate()` sets a
   TLS pointer; every `AReal` operator reads it. Works with
   `a * b * c.sin()`. `AReal` stays 16 B. The cost is that you must
   activate exactly one tape per thread, and crossing thread
   boundaries with an active scalar is undefined behaviour.

`xad-rs` picks (3), matching upstream XAD and most C++ AD libraries.
The TLS pointer is read once per op, which on modern hardware is one
load from a cache-resident thread-local block — essentially free.

The downside is *liveness*. The TLS pointer points to a tape on the
stack or heap; if that tape is moved, dropped, or shadowed, the
pointer dangles. `xad-rs` has three guards against this:

- `Tape::activate_guard()` returns a RAII guard whose `Drop`
  deactivates the tape and restores the previous active pointer.
- `Tape::activate()` and `Tape::deactivate_all()` are pairs to be
  called together at scope boundaries.
- Active tape state is not `Send`: you cannot accidentally hand it to
  another thread.

The constraint that surfaces in user code is the panic on double
activation: `Tape::activate()` while a tape is already active is a
loud failure, not silent corruption.

### Why `AReal` is not `Copy`

A Rust ergonomics question: why does `AReal` move on assignment when
`f64` copies? The answer is correctness: an `AReal` carries a slot
identity. Two `AReal`s with the same slot are not two independent
variables — they are two *references* to the same node on the tape.

If we made `AReal` `Copy`, the lines

```rust
let x = AReal::new(1.0);
let y = x;        // would copy under Copy
let z = x + y;    // x and y are the same tape slot
```

would tape-record `x + x`, not `x + y` — because both operands have
the same slot. The reverse sweep would scatter `2 * ∂z/∂x` to that
slot's `derivatives` entry. This is fine if you wanted `2x`, wrong if
you wanted two independent inputs.

Rust's move semantics on non-`Copy` types make the difference
explicit: `let y = x` *moves*, and `x + y` is a compile error
("borrow of moved value"). To get two independent inputs, you must
*construct* them as `AReal::new(1.0)` twice. The non-`Copy` design
catches the bug at compile time. `f64` is `Copy` because two `f64`s
of the same value are interchangeable; two `AReal`s of the same slot
are not.

### Affine types and the tape lifecycle

Rust's affine type system (values move by default, are dropped
exactly once) is a nearly perfect fit for the AD tape lifecycle:

- A `Tape` is dropped exactly once, at which point its `Drop` impl
  clears any TLS pointer pointing at it.
- An `AReal` cannot outlive its tape, because the tape's `Drop` runs
  at scope exit and ABI-wise the borrow-checker prevents references
  surviving.
- `Tape::activate_guard()` returns a non-`Send` non-`Clone` struct
  whose `Drop` restores the previous activation state — exactly the
  RAII pattern for scoped activation.

Where the type system *constrains* us is the borrow checker's reach
into the `+`, `-`, `*`, `/` operators. To support both `&a + &b` and
`a + b`, we'd need four `impl Add` blocks (`A + B`, `A + &B`, `&A +
B`, `&A + &B`), which is tedious but works. `xad-rs` chooses to
predominantly support the `&a + &b` borrow form so that operands are
not consumed; an alternative crate-wide convention could have been to
require owned operators everywhere and rely on `Clone`. Either is
defensible; consult the API docs for the chosen convention.

### Comparison with Julia, Python, C++

For context, a quick comparison with mainstream AD libraries:

| Tool | Language | Mode | Tape representation | Notes |
|---|---|---|---|---|
| `xad-rs` (this crate) | Rust | Both | Packed 3-buffer | Operator overloading; TLS active tape |
| XAD (upstream C++) | C++ | Both | Packed 3-buffer | Operator overloading; TLS active tape; checkpointing |
| Mainstream C++ overloading AD | C++ | Both | Packed (similar) | Operator overloading; rich operation menu |
| Python reverse-mode framework | Python/C++ | Reverse | Tree of `Function` objects | Operator overloading; per-node objects |
| Julia forward-mode AD | Julia | Forward | None (dual numbers via templates) | Compile-time type composition |
| Source-transformation AD | Fortran/C | Both | Generated source code | Source transformation |
| Trace-and-JIT array AD | Python | Both | Trace-then-compile to fused kernels | Trace + JIT, not direct overloading |

The headline observation: `xad-rs` is in the same family as XAD and
other tape-based operator-overloading AD libraries, which is
intentional. `xad-rs` adopts the same data structures and complexity
profile, and adds Rust's safety guarantees on top.

## Cost model

This chapter is about implementation, not asymptotics. The numbers to
internalise:

- **40 bytes per binary op** on the tape, plus 8 bytes per
  intermediate for the derivative slot.
- **~3× primal cost** for tape recording in forward pass on `AReal`.
- **~2× primal cost** for the reverse sweep.
- **One TLS load per op** to find the active tape — single-digit
  nanoseconds.
- **Zero allocations** during the forward pass if the tape is
  pre-sized.

## Anchored API

This chapter does not introduce new types; it explains the
implementation of existing ones. See chapters 02–05 for the public
API surface.

Internal types worth knowing about for performance debugging:

- [`Tape::with_capacity(stmts, ops)`](https://docs.rs/xad-rs/latest/xad_rs/tape/struct.Tape.html#method.with_capacity) — pre-allocate.
- [`Tape::num_statements`](https://docs.rs/xad-rs/latest/xad_rs/tape/struct.Tape.html#method.num_statements),
  [`num_operations`](https://docs.rs/xad-rs/latest/xad_rs/tape/struct.Tape.html#method.num_operations),
  [`memory`](https://docs.rs/xad-rs/latest/xad_rs/tape/struct.Tape.html#method.memory) — introspection.
- [`Tape::new_recording`](https://docs.rs/xad-rs/latest/xad_rs/tape/struct.Tape.html#method.new_recording) — reset capacity in-place
  between iterations to avoid re-allocation.

## Worked example — sizing a real tape

A toy pricer with 30 inputs and a few hundred elementary operations.
Use `Tape::num_operations` and `Tape::memory` to measure, then
pre-allocate.

```rust
use xad_rs::{AReal, Tape};

fn pricer<R: xad_rs::Real>(inputs: &[R]) -> R {
    // Stand-in for a real pricer: sum-of-products plus a transcendental.
    let mut acc = R::from(0.0_f64);
    for chunk in inputs.chunks(2) {
        let a = chunk.get(0).cloned().unwrap_or(R::from(0.0_f64));
        let b = chunk.get(1).cloned().unwrap_or(R::from(1.0_f64));
        acc = acc + a.clone() * b.clone();
    }
    acc.exp()
}

fn main() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut inputs: Vec<AReal<f64>> = (0..30).map(|i| AReal::new(i as f64 * 0.1)).collect();
    AReal::register_input(&mut inputs, &mut tape);

    let mut pv = pricer(&inputs);
    AReal::register_output(std::slice::from_mut(&mut pv), &mut tape);
    pv.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();

    println!("statements: {}", tape.num_statements());
    println!("operations: {}", tape.num_operations());
    println!("memory:     {} bytes", tape.memory());

    Tape::<f64>::deactivate_all();
}
```

The reported numbers are exact: every binary op is one statement, two
operations, one derivative slot. For a Monte Carlo loop, run this
sizing pass once and use the figures to pre-allocate a future tape
with `Tape::with_capacity(stmts, ops)`.

## Common pitfalls

- **Believing a freshly allocated `Tape` is fast.** A fresh tape grows
  its buffers geometrically as you record. The *first* iteration of a
  Monte Carlo loop pays for the allocations; subsequent ones don't.
  Either pre-size with `with_capacity` or warm up once and call
  `new_recording()`.
- **Threading without re-activating.** A child thread spawned with
  `std::thread::spawn` does *not* inherit the parent's active tape
  TLS pointer. Each thread that wants to record must `activate` its
  own tape; sharing a tape across threads is unsafe.
- **Cloning `AReal` to "get a copy".** `AReal::clone()` copies the
  slot, not the value. Two clones refer to the same tape node, and
  the reverse sweep treats them as one variable. To get an
  independent variable with the same value, construct
  `AReal::new(other.value())` (and register as input if appropriate).
- **Treating the tape's `Drop` as a no-op.** It isn't: it deactivates
  the TLS pointer if it's pointing at this tape. If you `mem::forget`
  a tape, the TLS pointer dangles. There is no recovery short of
  `Tape::deactivate_all()` or the RAII guard.

## References

- **Auto-differentiation team**, *XAD: Comprehensive C++ Automatic
  Differentiation*, <https://auto-differentiation.github.io/>.
  Upstream C++ library — the tape layout `xad-rs` ports is theirs.
- **Walther, A. and Griewank, A.** *Getting started with ADOL-C*. In
  *Combinatorial Scientific Computing*, Chapman & Hall, 2011. The
  classical tape representation that motivates this family of
  designs.
- **Hogan, R. J.** *Fast reverse-mode automatic differentiation using
  expression templates in C++*. ACM TOMS 40 (2014), 26. The
  expression-template approach to operator-overloading AD; not the
  same as `xad-rs` but a useful comparison point for understanding
  why we *don't* lazily build expression trees.
- **Naumann, U.** *The Art of Differentiating Computer Programs*,
  chapters 6–9, for source-transformation implementations.
- **Reynolds, J. C.** *Definitional interpreters for higher-order
  programming languages*. Higher-Order and Symbolic Computation 11
  (1998), 363–397 (reprint of the 1972 original). Background on the
  interpreter–compiler tradeoff that mirrors the operator-overloading
  vs source-transformation choice in AD.
