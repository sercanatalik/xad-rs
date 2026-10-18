# 06 — Numerical analysis of AD

> Round-off error in AD versus truncation + cancellation in finite
> differences, conditioning of the derivative computation, the
> step-size dilemma for FD (and why AD doesn't have it), the
> accuracy floor of complex-step differentiation, and what "machine
> precision" really means for a chained gradient.

## Overview

The marketing claim for AD is *machine precision*. This is more
specific than it sounds. AD eliminates **truncation error** —
the `O(h)` or `O(h²)` term in `(f(x+h) − f(x))/h` — but it does *not*
eliminate **round-off error**: the gradient computation involves new
floating-point operations beyond the primal, and each new op
contributes its own relative error of size `O(u)`, the unit round-off
(≈ 1.1 × 10⁻¹⁶ for `f64`).

The accuracy story for AD is therefore: AD is accurate to within a
constant multiple of the round-off in the primal, with that constant
depending on the *conditioning* of the derivative computation along
the program's evaluation trace. This chapter makes that precise and
then compares it against finite differences and the complex-step
trick.

## Theory

### Floating-point error of one elementary op

The IEEE-754 model says that every elementary op (`+`, `-`, `*`, `/`,
`sqrt`, …) returns the exactly-rounded result of the true real-number
operation, with a relative error bounded by `u = 2⁻⁵³ ≈ 1.1 × 10⁻¹⁶`
for double precision. For an op `c = a ⊕ b`,

```math
\widehat{c} \;=\; \mathrm{fl}(a \oplus b) \;=\; (a \oplus b)\,(1 + \delta),
\qquad |\delta| \le u.
```

For transcendentals the constant is larger (typically a few `u`) but
the form is the same.

### Forward-mode round-off

In `Jet1`, every elementary op does at most 4 floating-point
multiplies and adds beyond the primal: e.g. `Mul` computes `(ac, ad +
bc)` — 1 mul for the primal, 2 muls + 1 add for the tangent. Each
contributes one relative-error term of size `O(u)`.

For a Wengert list of length `P`, the standard backward-error analysis
of floating-point arithmetic (Wilkinson 1965; Higham 2002 §3) gives

```math
\bigl| \widehat{f}'(x) - f'(x) \bigr| \;\le\; \gamma_P \cdot |f'(x)|,
\qquad \gamma_P \;=\; \frac{P\,u}{1 - P\,u} \;\approx\; P\,u
```

for `Pu ≪ 1`. In words: the relative error in the AD-computed
derivative grows linearly in the *number of operations along the
evaluation path*, multiplied by unit round-off. For `P = 10⁵` and `u
≈ 10⁻¹⁶`, that is a relative error of `~10⁻¹¹`. Nowhere near the `u`
floor, but several orders of magnitude better than the best finite-
difference accuracy (see below).

The cleanest version of this statement, due to Bischof, Carle &
Khademi (1992) for forward mode: **AD inherits the same forward error
bound as the primal computation**, up to a small constant that comes
from the extra ops the chain rule introduces. So if your primal is
well-conditioned, the AD gradient is well-conditioned too.

### Reverse-mode round-off

Reverse mode has the same per-edge error story but a different
accumulation pattern: each input's adjoint is a *sum* over all paths
from the output to that input, with each path-product accumulated by
the scatter on the reverse sweep. The catastrophic case is when these
path products have large magnitudes but opposite signs, cancelling
heavily in the sum.

The relevant condition number is therefore the **adjoint condition
number**:

```math
\kappa_{\nabla f}(x) \;=\; \frac{\sum_{\text{paths }P} |\text{prod}(P)|}{|\nabla f(x)|}.
```

In words: the ratio of the absolute-value path sum to the actual
gradient norm. When this ratio is `O(1)`, reverse mode is as
accurate as forward; when it is large, the reverse sweep eats
catastrophic cancellation and the relative error inflates by
`κ_∇f · u`.

Forward mode has the same issue at every intermediate (the *tangent
condition number* applies to each propagated tangent), but because
each intermediate's tangent is *only the sum along paths from inputs
to that intermediate*, forward mode rarely sees the global path-sum
cancellation that hits reverse mode at the inputs. In practice this is
visible: reverse mode of a near-singular Jacobian (vol → 0 in
Black–Scholes, say) can blow up where the forward derivative remains
stable.

### Finite differences: the step-size dilemma

The forward-difference estimator is

```math
\widetilde{f}'(x; h) \;=\; \frac{f(x + h) - f(x)}{h}.
```

The error has two pieces:

1. **Truncation.** Taylor expansion gives `f(x + h) = f(x) + h f'(x) +
   (h²/2) f''(x) + O(h³)`. So
   `(f(x+h) − f(x))/h = f'(x) + (h/2) f''(x) + O(h²)`. The truncation
   error is `O(h)`.
2. **Round-off.** Each `f` evaluation has relative error `~u |f|`.
   Subtracting nearly equal numbers (`f(x+h) − f(x) ≈ h f'(x)` is
   small for small `h`) amplifies that error by `1/h`. So the round-
   off contribution to the FD estimator is `O(u |f| / h)`.

The total error is bounded by

```math
\bigl|\widetilde{f}'(x; h) - f'(x)\bigr|
\;\lesssim\;
\underbrace{\tfrac{h}{2}\,|f''(x)|}_{\text{truncation}}
\;+\;
\underbrace{\tfrac{2 u\,|f(x)|}{h}}_{\text{cancellation}}.
```

Optimising over `h`: the minimum is at `h_⋆ = 2 √(u |f| / |f''|)`,
which for typical `|f|, |f''| ≈ 1` gives `h_⋆ ≈ 2 √u ≈ 3 × 10⁻⁸` and
the minimum error is `~√u ≈ 10⁻⁸`.

This is the **finite-difference step-size dilemma**: smaller `h`
reduces truncation but blows up round-off, and the best you can do is
square-root-of-machine-precision accuracy. For double-precision FD
that's **8 decimal digits**, never more.

Central differences `(f(x+h) − f(x−h)) / (2h)` give `O(h²)`
truncation, so the optimal step is `h_⋆ ≈ u^{1/3} ≈ 5 × 10⁻⁶` and the
floor is `u^{2/3} ≈ 10⁻¹¹`. Better, still nowhere near machine
precision, and at twice the cost.

AD, by contrast, has *no `h`*. Its relative error is `O(P u)` — three
or four extra digits of error per chained op, not eight digits *floor*
across the board. For a 10000-op pricer, AD comfortably beats central
differences by 4–6 decimal digits.

### Complex-step differentiation

There is a clever trick that gets you `O(u)` accuracy at one
evaluation per direction *without* AD: the **complex-step**
derivative. For analytic `f`,

```math
\Im\bigl(f(x + i h)\bigr) \;=\; h\,f'(x) - \tfrac{h^3}{6}f'''(x) + O(h^5),
```

so

```math
f'(x) \;\approx\; \frac{\Im(f(x + i h))}{h}.
```

For very small `h` (e.g. `10⁻²⁰⁰`), the truncation term `O(h²)`
vanishes far below the round-off floor and there is *no cancellation*
(the imaginary part of `f(x + ih)` is itself `~h`, not the difference
of two `O(1)` numbers). The accuracy approaches `O(u)`.

Caveats:

- Requires complex arithmetic for every elementary op. For pricer-style
  programs with `sqrt`, `log`, branch cuts, this is a delicate
  rewrite.
- Requires `f` to extend analytically to a complex neighbourhood. Many
  payoffs (caps, floors, barriers) do not.
- Only yields a single directional derivative per call; for full
  gradient you need `n` calls, like FD.

AD is the right default. Complex-step is a useful sanity check for
analytic primals where you want machine-precision agreement.

### Conditioning of the derivative computation

The condition number of a function `f` at `x` is

```math
\kappa_f(x) \;=\; \frac{|x|\,|f'(x)|}{|f(x)|}.
```

In words: `κ_f` is the relative-change ratio between `f` and `x` at
`x`. A well-conditioned `f` has small `κ_f`; an ill-conditioned `f`
amplifies input perturbations.

Now consider differentiating *the AD program itself*. The condition
number of "computing `f'(x)` as a chained product of elementary
derivatives" is the product of intermediate condition numbers along
the path:

```math
\kappa_{f'}(x) \;\le\; \prod_{k=1}^{P} \kappa_{\varphi_k}(v_{j_k}).
```

This is the **multiplicative-error model** of straight-line
arithmetic. For most numerical programs that are not deliberately
constructed to cancel, this product is `O(P u)` and AD achieves
near-primal accuracy. For programs that *do* cancel (subtracting
quantities very close to equal, or evaluating at a near-singularity),
both the primal and the derivative inherit the same conditioning, and
the AD output is no worse than the primal *but no better*.

The practical takeaway: AD does not magically improve numerical
stability. It returns the derivative *of the program you wrote*,
including all the cancellation and overflow your primal exhibits. If
your primal is unstable, your AD gradient will be too. The remedies
are the usual ones — reformulate to avoid cancellation
(`log1p`, `expm1`, log-sum-exp tricks) — and they apply equally to
the AD-augmented body.

### A worked accuracy comparison

For `f(x) = exp(x²)/x` at `x = 1.5`:

```math
f(x) = \frac{e^{x^2}}{x},
\qquad
f'(x) = \frac{e^{x^2}(2x^2 - 1)}{x^2}.
```

At `x = 1.5`: `f(1.5) ≈ 6.328`, `f'(1.5) ≈ 15.62`. Computing `f'(1.5)`
three ways:

| Method | Result | Absolute error | Notes |
|---|---|---|---|
| AD (`Jet1`, forward) | `15.620 ... ` | `~10⁻¹⁵` | Several extra ops vs primal; round-off only |
| Central differences, `h = 10⁻⁶` | `15.620 ...` | `~10⁻¹¹` | Best FD step found by sweep |
| Forward differences, `h = 10⁻⁸` | `15.620 ...` | `~10⁻⁸` | Step at `√u`; truncation = round-off |
| Complex step, `h = 10⁻²⁰⁰` | `15.620 ...` | `~10⁻¹⁵` | Requires complex arithmetic |

The headline observation: AD and complex-step bottom out at the
round-off floor. Finite differences cannot, no matter the step. This
gap of 4–8 decimal digits is the practical reason quant teams switch
to AD.

### What "machine precision" really means for the gradient

A pedantic definition: an AD gradient is **machine-precision** if its
relative error against the true mathematical gradient is `O(c · u)`
for a constant `c` that depends only on the depth `P` of the
computation (not on a tunable parameter). Concretely:

- For a well-conditioned primal of length `P = 10⁴` ops, the AD
  gradient has relative error `~10⁻¹²` — about 12 correct digits.
- The same primal with central differences gives ~11 digits at the
  best step and ~8 digits at most steps.
- AD's accuracy degrades linearly in `P` (Wilkinson bound); FD's
  accuracy is *floor-bounded* at `√u` (or `u^{2/3}` for centred FD)
  regardless of how nice your primal is.

The "linear in `P`" rate matters for very long pricers (e.g.
unrolled multi-step Monte Carlo). For `P = 10⁹` you'd lose 9 of the
16 digits, leaving 7. That is still better than central differences,
but the gap narrows. Stable formulation of the primal matters more at
those depths than at the typical `P ≈ 10⁴` scale.

## Cost model

This chapter does not introduce new performance figures; it quantifies
*accuracy* per dollar of compute.

| Method | Cost per derivative | Accuracy (relative) | Tunable step? |
|---|---|---|---|
| AD forward (one direction) | `~2P` | `O(P u)` | No |
| AD reverse (one gradient) | `~5P` | `O(P u · κ_path)` | No |
| Forward differences | `~2P` | `~√u` floor | Yes (delicate) |
| Central differences | `~4P` | `~u^{2/3}` floor | Yes (delicate) |
| Complex step | `~3P` (complex ops) | `O(u)` | No |

The dominant cost saving of AD is *not* per-derivative time (FD is
sometimes cheaper per direction); it is the *cheap-gradient principle*
(chapter 03) — `O(P)` cost for the entire gradient regardless of `n`.

## Anchored API

This chapter does not introduce new types. It informs how you read
back results from existing ones. Relevant guidance:

- For very long computations where round-off accumulates, prefer
  `Jet1` / `JetK` forward mode over reverse mode when you have a
  choice — forward mode's path-summation is more stable.
- When you suspect AD numerical issues, compare against complex-step
  on a stripped-down analytic version of the pricer; the gap is
  diagnostic of *primal* conditioning, not an AD bug.
- The `smooth_max` / `smooth_min` / `smooth_abs` helpers in
  `math::ad` are necessary not only for differentiability at kinks
  (chapter 01) but also for *numerical* stability of gradients across
  the smoothed region, since hard `max` produces zero gradients on
  one side and the smoothed version produces a small but non-zero
  one.

## Worked example — AD vs FD accuracy

```rust
use xad_rs::Jet1;

fn f(x: f64) -> f64 {
    (x * x).exp() / x
}

fn f_prime(x: f64) -> f64 {
    let xx = x * x;
    xx.exp() * (2.0 * xx - 1.0) / (x * x)
}

fn fd_central(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn main() {
    let x = 1.5_f64;
    let analytic = f_prime(x);

    // AD via Jet1.
    let xj = Jet1::<f64>::new(x, 1.0);
    let yj = (xj.clone() * xj.clone()).exp() / xj.clone();
    let ad = yj.derivative();

    // Central differences sweep over h.
    let h_grid = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12];
    println!("analytic: {:.16}", analytic);
    println!("AD:       {:.16}  (err: {:.2e})", ad, (ad - analytic).abs());
    for h in h_grid {
        let v = fd_central(f, x, h);
        println!("FD h={:.0e}: {:.16}  (err: {:.2e})", h, v, (v - analytic).abs());
    }
}
```

Running this produces a U-shaped accuracy curve in `h`: too large and
truncation dominates; too small and round-off dominates. The minimum
sits around `h ≈ 10⁻⁶` with about 10–11 correct digits. AD sits at
~15 correct digits with no tunable parameter.

## Common pitfalls

- **Treating "machine precision" as a literal `u` floor.** AD is
  machine-precision *up to a constant times the number of ops*. For
  short pricers this is indistinguishable from `u`; for very long
  ones (millions of ops) the relative error drifts up.
- **Comparing AD against the wrong "truth".** The right reference is
  the *mathematical* derivative of the function the program computes,
  not the FD estimate. If you cross-check AD against FD and FD differs
  in the 8th digit, that's FD's error floor, not an AD bug.
- **Using AD to mask a numerically unstable primal.** AD inherits the
  primal's conditioning. If your primal cancels at the 6th digit, so
  does its gradient. Fix the primal first.
- **Ignoring the `smooth_*` helpers near kinks.** A hard `max(0, x)`
  has gradient 0 on one side and gradient 1 on the other; AD will
  silently return 0 if you're on the left of zero, even if you wanted
  a smoothed sensitivity. Use `math::ad::smooth_max` with an
  appropriate scale parameter.
- **Treating reverse-mode catastrophic cancellation as an AD bug.**
  Reverse mode sums path products at each input; for ill-conditioned
  problems (`vol → 0`, deep ITM/OTM options, near-singular Jacobians)
  the sum can lose precision. The remedy is to reformulate or switch
  to forward mode for the affected directions, not to change AD
  modes blindly.

## References

- **Wilkinson, J. H.** *The Algebraic Eigenvalue Problem*. Oxford
  University Press, 1965. The classical backward-error analysis of
  floating-point arithmetic.
- **Higham, N. J.** *Accuracy and Stability of Numerical Algorithms*,
  2nd ed. SIAM, 2002. Chapters 3–4 for forward and backward error
  analysis of straight-line computations; chapter 5 for the
  conditioning of derivatives.
- **Bischof, C. H., Carle, A., and Khademi, P.** *Algorithm 755:
  ADIFOR 2.0: an automatic differentiation tool*. ACM TOMS 22 (1996),
  131–167. Includes the forward-mode error analysis (AD inherits
  primal accuracy up to a small constant).
- **Squire, W. and Trapp, G.** *Using complex variables to estimate
  derivatives of real functions*. SIAM Review 40 (1998), 110–112.
  The complex-step trick.
- **Martins, J. R. R. A., Sturdza, P., and Alonso, J. J.** *The
  complex-step derivative approximation*. ACM TOMS 29 (2003),
  245–262. Practical implementation of complex-step.
- **Griewank, A. and Walther, A.** *Evaluating Derivatives*, 2nd ed.,
  chapter 4, for the numerical analysis of AD specifically (round-off
  propagation along the evaluation trace).
