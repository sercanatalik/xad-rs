# Acknowledgements

`xad-rs` is a small library standing on other people's work. This page records
what it owes and to whom.

## The XAD library

`xad-rs` is an independent Rust implementation of the automatic-differentiation
architecture popularised by [XAD](https://github.com/auto-differentiation/xad),
the C++ library by the auto-differentiation team. The shape of the design is
theirs: one active scalar type that records onto a tape as the program runs,
the packed three-buffer tape layout that
[chapter 5](theory/05-implementation-tradeoffs.md) describes, and the premise
that a single generic body should evaluate in every mode.

What is written here is Rust against that design rather than a translation of
the C++ source. This project is not affiliated with the upstream authors, is
not endorsed by them, and is not their responsibility. Questions about `xad-rs`
belong on [its own tracker](https://github.com/sercanatalik/xad-rs/issues), not
on theirs.

If you want the original, in C++, from the people who designed it:
<https://auto-differentiation.github.io/>.

## The literature

The chapters argue from these, and cite them where the argument needs the
detail.

- **Griewank, A. and Walther, A.** *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation*, 2nd ed. SIAM, 2008. The canonical
  AD reference; cited throughout for full complexity bounds, adjoint
  correctness proofs, and the revolve checkpointing algorithm.
- **Wengert, R. E.** *A simple automatic derivative evaluation program*.
  Communications of the ACM 7 (1964), 463–464. The original evaluation-trace
  formulation, and the reason chapter 1 calls the trace a Wengert list.
- **Baur, W. and Strassen, V.** *The complexity of partial derivatives*.
  Theoretical Computer Science 22 (1983), 317–330. The cheap-gradient principle
  in its original form: the whole gradient for a constant multiple of the cost
  of the value.
- **Pearlmutter, B. A.** *Fast exact multiplication by the Hessian*. Neural
  Computation, 1994. Hessian-vector products at `O(P)` cost.
- **Naumann, U.** *The Art of Differentiating Computer Programs: An
  Introduction to Algorithmic Differentiation*. SIAM, 2012. Practical AD with
  extensive tape and source-transformation treatment.
- **Higham, N. J.** *Accuracy and Stability of Numerical Algorithms*, 2nd ed.
  SIAM, 2002. The standard reference for round-off propagation in
  straight-line computations, behind most of chapter 6.
- **Hull, J.** *Options, Futures, and Other Derivatives*, 10th ed. Pearson,
  2017. Background for the Black–Scholes pricing the worked examples use.
- **Capriotti, L.** *Fast Greeks by algorithmic differentiation*. Journal of
  Computational Finance 14 (2011), 3–35. The industry-canonical motivation for
  adjoint AD in quantitative finance.

## The crates it is built on

Three dependencies at run time, deliberately few:

- [`num-traits`](https://crates.io/crates/num-traits) — the numeric trait
  vocabulary the `Real` trait sits beside.
- [`ndarray`](https://crates.io/crates/ndarray) — the dense storage behind the
  Jacobian and Hessian drivers.
- [`rayon`](https://crates.io/crates/rayon) — the work-stealing pool the
  parallel Hessian passes use.

Two more for tests and examples only, not built into the published crate:
[`approx`](https://crates.io/crates/approx) for the cross-checks against
analytic answers, and [`paste`](https://crates.io/crates/paste) for the
uniformity tests stamped from the derivative table.

## This site

Built with [mdBook](https://rust-lang.github.io/mdBook/), typeset in
[STIX Two Text](https://github.com/stipub/stixfonts),
[Source Sans 3](https://github.com/adobe-fonts/source-sans) and
[JetBrains Mono](https://www.jetbrains.com/lp/mono/), with the mathematics
rendered by [KaTeX](https://katex.org/) and hosted on GitHub Pages.
