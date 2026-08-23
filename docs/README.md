# `xad-rs` documentation

Long-form theory and worked examples for the `xad-rs` automatic
differentiation library. This is the companion to the crate-level
rustdoc on [docs.rs](https://docs.rs/xad-rs): rustdoc tells you *how* to
call the API, the chapters here tell you *why* the math works, *which*
mode you should pick, and *how* the implementation behaves under the
hood.

## Who this is for

You write numerical code — pricers, regressions, calibrations, ODE
solvers, loss functions — and you want exact, machine-precision
derivatives of it. You know undergraduate calculus and linear algebra.
You do **not** need prior automatic-differentiation experience; chapter
01 starts from "what is AD?". You also do not need to read Rust types
fluently to follow the math, but every chapter ends with a runnable
Rust snippet that demonstrates the chapter's idea against the actual
`xad-rs` API.

The chapters are written so each one can be read in roughly 15–25
minutes. Chapters 01–04 cover the API you will use day-to-day;
chapters 05–06 are deeper background that you can skip on first read
and come back to when you need to debug performance or accuracy.

## How to read this

If you have a specific problem in front of you, use the decision tree
below to jump to the right chapter. If you want a linear read-through,
the chapters are numbered in pedagogical order.

### Decision tree — which chapter answers your question?

You can characterize an AD problem by three numbers:

- **n** — number of independent inputs you want derivatives with respect
  to.
- **m** — number of outputs (often 1: a single scalar PV, log-
  likelihood, loss, etc.).
- **order** — first-order (gradient, JVP), or second-order (Hessian,
  gamma, curvature).

| Your situation | Recommended mode | Chapter |
|---|---|---|
| "I just want the function value, no derivatives" | `f64` | [01 — AD as a discipline](theory/01-automatic-differentiation.md) |
| "I want d/dx of a function along **one** input direction" (n small, m anything) | `Jet1<f64>`, or `ops::compute_derivative_fwd` / `ops::compute_directional_derivative_fwd` | [02 — Forward mode and dual numbers](theory/02-forward-mode-and-dual-numbers.md) |
| "I have many inputs, one scalar output, and I want the gradient" (n ≫ 4, m = 1) | `ops::compute_gradient_rev`, or `Tape` + `AReal<f64>` by hand | [03 — Reverse mode and taped adjoints](theory/03-reverse-mode-and-taped-adjoints.md) |
| "I want gamma / diagonal Hessian along one direction" (second order, one direction) | `Jet2<f64>` | [04 — Second-order and k-jets](theory/04-second-order-and-k-jets.md) |
| "I want the **full Hessian**", n ≲ 50 | `Jet2Vec` via `compute_full_hessian` | [04 — Second-order and k-jets](theory/04-second-order-and-k-jets.md) |
| "I want the **full Hessian**", larger n | `compute_hessian_k::<K, _>` (nested `Tape<JetK<f64, K>>`) | [04 — Second-order and k-jets](theory/04-second-order-and-k-jets.md) |
| "I want to understand the tape memory layout" | — | [05 — Implementation tradeoffs](theory/05-implementation-tradeoffs.md) |
| "I want to know the accuracy of AD vs finite differences" | — | [06 — Numerical analysis of AD](theory/06-numerical-analysis-of-ad.md) |

### Linear reading order

1. [**01 — Automatic differentiation as a discipline**](theory/01-automatic-differentiation.md).
   What AD is, why it is *exact*, how it differs from numerical
   bumping and from symbolic differentiation, the Wengert-list
   formalism that underlies every mode, branches and the piecewise-
   smooth setting, and the role of the [`Real`](https://docs.rs/xad-rs/latest/xad_rs/real/trait.Real.html)
   trait.
2. [**02 — Forward mode and dual numbers**](theory/02-forward-mode-and-dual-numbers.md).
   Dual numbers as the quotient ring `R[ε]/(ε²)` and its universal
   property, the chain rule as a one-line proof in `D`, forward-mode
   AD as a Jacobian-vector product (JVP), the multi-direction
   extension via multivariate dual numbers that gives `JetK`, and
   the differential-geometric interpretation as the tangent bundle.
   Anchors: `Jet1`, `JetK`, `ops::compute_derivative_fwd`,
   `ops::compute_directional_derivative_fwd`.
3. [**03 — Reverse mode and taped adjoints**](theory/03-reverse-mode-and-taped-adjoints.md).
   The linearised computation graph, the adjoint recurrence derived
   from the chain rule, the Baur–Strassen theorem and the cheap-
   gradient principle, a Lagrangian view of reverse mode, vertex
   elimination orderings, the packed three-buffer tape, and the
   tape-reuse model exposed by `Tape::record` / `Tape::new_recording`.
   Anchors: `Tape`, `AReal`, `ops::compute_gradient_rev`,
   `ops::compute_jacobian_rev`.
4. [**04 — Second-order and k-jets**](theory/04-second-order-and-k-jets.md).
   k-jets as truncated Taylor series and as the rings
   `R[ε]/(ε^{k+1})`, Faà di Bruno's higher-order chain rule, the
   single-direction second-order type `Jet2`, the dense full-Hessian
   type `Jet2Vec`, Hessian symmetry, forward-over-adjoint via a tape
   whose storage scalar is itself a jet, the K-wide extension that
   recovers `K` Hessian columns per sweep, and edge pushing for sparse
   Hessians. Anchors: `Jet2`, `Jet2Vec`, `JetK`,
   `ops::compute_hessian`, `ops::compute_hessian_k`,
   `ops::compute_full_hessian`.
5. [**05 — Implementation tradeoffs**](theory/05-implementation-tradeoffs.md).
   Operator overloading versus source-to-source transformation, the
   three-buffer packed tape layout `xad-rs` inherits from upstream
   XAD, the rationale for thread-local active-tape pointers, slot
   allocation arithmetic, and where Rust's affine type system helps
   the AD lifecycle.
6. [**06 — Numerical analysis of AD**](theory/06-numerical-analysis-of-ad.md).
   Round-off error in AD versus truncation + cancellation in finite
   differences, conditioning of the derivative computation, the step-
   size dilemma and the `√u` accuracy floor of FD, the complex-step
   trick, and how to read "machine precision" precisely.

## References used across the chapters

- **Griewank, A. and Walther, A.** *Evaluating Derivatives: Principles
  and Techniques of Algorithmic Differentiation*, 2nd ed. SIAM, 2008.
  The canonical AD reference; cited throughout for full complexity
  bounds, adjoint correctness proofs, and the revolve checkpointing
  algorithm.
- **Wengert, R. E.** *A simple automatic derivative evaluation program*.
  Communications of the ACM 7 (1964), 463–464. The original
  evaluation-trace formulation.
- **Baur, W. and Strassen, V.** *The complexity of partial derivatives*.
  Theoretical Computer Science 22 (1983), 317–330. The cheap-gradient
  principle in its original form.
- **Pearlmutter, B. A.** *Fast exact multiplication by the Hessian*,
  Neural Computation, 1994. Hessian-vector products at `O(P)` cost.
- **Naumann, U.** *The Art of Differentiating Computer Programs: An
  Introduction to Algorithmic Differentiation*. SIAM, 2012. Practical
  AD with extensive tape and source-transformation treatment.
- **Higham, N. J.** *Accuracy and Stability of Numerical Algorithms*,
  2nd ed. SIAM, 2002. The standard reference for round-off propagation
  in straight-line computations.
- **Auto-differentiation team.** *XAD: Comprehensive C++ Automatic
  Differentiation*. <https://auto-differentiation.github.io/>. Upstream
  C++ XAD library; the packed three-buffer tape layout `xad-rs` ports
  is theirs.
- **Hull, J.** *Options, Futures, and Other Derivatives*, 10th ed.
  Pearson, 2017. Background reference for the Black–Scholes pricing
  used in the worked examples.
- **Capriotti, L.** *Fast Greeks by algorithmic differentiation*.
  Journal of Computational Finance 14 (2011), 3–35. The industry-
  canonical motivation for AAD in quant finance.
