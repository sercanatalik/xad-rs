//! The single source of truth for unary elementary derivatives.
//!
//! Every AD surface in this crate — `math::ad` (reverse, `AReal`),
//! `math::fwd` (forward, `Jet1`), `math::fwdk` (K-lane forward, `JetK`), and
//! the inherent methods on `Jet2` and `Jet2Vec` — stamps its unary elementaries from the one table below via
//! the X-macro [`for_each_unary_elementary`]. Adding an elementary here
//! adds it to **all five** surfaces at once, and every surface computes
//! bit-identical values and first derivatives by construction (they share
//! the same closures).
//!
//! # Entry shape
//!
//! Each entry is
//!
//! ```text
//! $stamp!(name, "doc string",
//!     |x: T| value,          // f(x)
//!     |x, r| d1,          // f'(x);  r = f(x) for reuse (e.g. exp' = r)
//!     |x, r, d| d2);      // f''(x); d = f'(x) for reuse
//! ```
//!
//! First-order consumers (`math::ad`, `math::fwd`, `math::fwdk`) simply do not expand
//! the `d2` closure; second-order consumers (`Jet2`, `Jet2Vec`) expand
//! all three.
//!
//! # `erf` / `inv_norm_cdf` value hooks
//!
//! The `erf`-family *values* go through [`Passive::erf_value`] /
//! [`Passive::inv_norm_cdf_value`](crate::passive::Passive) rather than a
//! free polynomial function. For `f32`/`f64` these are the A&S 7.1.26 /
//! Acklam approximations; for the nested forward-over-adjoint scalar
//! `Jet1<T>` they are overridden to carry the **exact** analytic tangent
//! instead of the approximation polynomial's derivative — see
//! `jet1_passive.rs`. Without the override, second-order results that
//! flow through `erf` pick up the polynomial's derivative error (~1e-6).
//!
//! [`Passive::erf_value`]: crate::passive::Passive::erf_value

/// X-macro: invoke `$stamp` once per unary elementary.
///
/// See the module docs for the entry shape. `$stamp` must be a
/// `macro_rules!` accepting `(name, doc, val, d1, d2)`.
macro_rules! for_each_unary_elementary {
    ($stamp:ident) => {
        // --- trigonometric ---
        $stamp!(sin, "Sine.",
            |x: T| x.sin(),
            |x: T, _r: T| x.cos(),
            |_x: T, r: T, _d: T| -r);
        $stamp!(cos, "Cosine.",
            |x: T| x.cos(),
            |x: T, _r: T| -x.sin(),
            |_x: T, r: T, _d: T| -r);
        $stamp!(tan, "Tangent. `f' = 1 + tan²`, `f'' = 2·tan·f'`.",
            |x: T| x.tan(),
            |_x: T, r: T| T::one() + r * r,
            |_x: T, r: T, d: T| T::from(2.0).unwrap() * r * d);
        $stamp!(asin, "Inverse sine. `f' = 1/√(1-x²)`, `f'' = x·f'³`.",
            |x: T| x.asin(),
            |x: T, _r: T| T::one() / (T::one() - x * x).sqrt(),
            |x: T, _r: T, d: T| x * d * d * d);
        $stamp!(acos, "Inverse cosine. `f' = -1/√(1-x²)`, `f'' = x·f'³`.",
            |x: T| x.acos(),
            |x: T, _r: T| -T::one() / (T::one() - x * x).sqrt(),
            |x: T, _r: T, d: T| x * d * d * d);
        $stamp!(atan, "Inverse tangent. `f' = 1/(1+x²)`, `f'' = -2x·f'²`.",
            |x: T| x.atan(),
            |x: T, _r: T| T::one() / (T::one() + x * x),
            |x: T, _r: T, d: T| -T::from(2.0).unwrap() * x * d * d);

        // --- hyperbolic ---
        $stamp!(sinh, "Hyperbolic sine.",
            |x: T| x.sinh(),
            |x: T, _r: T| x.cosh(),
            |_x: T, r: T, _d: T| r);
        $stamp!(cosh, "Hyperbolic cosine.",
            |x: T| x.cosh(),
            |x: T, _r: T| x.sinh(),
            |_x: T, r: T, _d: T| r);
        $stamp!(tanh, "Hyperbolic tangent. `f' = 1 - tanh²`, `f'' = -2·tanh·f'`.",
            |x: T| x.tanh(),
            |_x: T, r: T| T::one() - r * r,
            |_x: T, r: T, d: T| -T::from(2.0).unwrap() * r * d);
        $stamp!(asinh, "Inverse hyperbolic sine. `f' = 1/√(x²+1)`, `f'' = -x·f'³`.",
            |x: T| x.asinh(),
            |x: T, _r: T| T::one() / (x * x + T::one()).sqrt(),
            |x: T, _r: T, d: T| -x * d * d * d);
        $stamp!(acosh, "Inverse hyperbolic cosine. `f' = 1/√(x²-1)`, `f'' = -x·f'³`.",
            |x: T| x.acosh(),
            |x: T, _r: T| T::one() / (x * x - T::one()).sqrt(),
            |x: T, _r: T, d: T| -x * d * d * d);
        $stamp!(atanh, "Inverse hyperbolic tangent. `f' = 1/(1-x²)`, `f'' = 2x·f'²`.",
            |x: T| x.atanh(),
            |x: T, _r: T| T::one() / (T::one() - x * x),
            |x: T, _r: T, d: T| T::from(2.0).unwrap() * x * d * d);

        // --- exponential / logarithmic ---
        $stamp!(exp, "Exponential. `f = f' = f''`.",
            |x: T| x.exp(),
            |_x: T, r: T| r,
            |_x: T, r: T, _d: T| r);
        $stamp!(exp2, "Base-2 exponential. `f' = f·ln 2`, `f'' = f·(ln 2)²`.",
            |x: T| x.exp2(),
            |_x: T, r: T| r * T::from(std::f64::consts::LN_2).unwrap(),
            |_x: T, r: T, _d: T| {
                let ln2 = T::from(std::f64::consts::LN_2).unwrap();
                r * ln2 * ln2
            });
        $stamp!(ln, "Natural logarithm. `f' = 1/x`, `f'' = -1/x²`.",
            |x: T| x.ln(),
            |x: T, _r: T| T::one() / x,
            |_x: T, _r: T, d: T| -d * d);
        $stamp!(log2, "Base-2 logarithm. `f' = 1/(x·ln 2)`, `f'' = -f'/x`.",
            |x: T| x.log2(),
            |x: T, _r: T| T::one() / (x * T::from(std::f64::consts::LN_2).unwrap()),
            |x: T, _r: T, d: T| -d / x);
        $stamp!(log10, "Base-10 logarithm. `f' = 1/(x·ln 10)`, `f'' = -f'/x`.",
            |x: T| x.log10(),
            |x: T, _r: T| T::one() / (x * T::from(std::f64::consts::LN_10).unwrap()),
            |x: T, _r: T, d: T| -d / x);
        $stamp!(ln_1p, "`ln(1 + x)`. `f' = 1/(1+x)`, `f'' = -f'²`.",
            |x: T| x.ln_1p(),
            |x: T, _r: T| T::one() / (T::one() + x),
            |_x: T, _r: T, d: T| -d * d);
        $stamp!(exp_m1, "`exp(x) - 1`. `f' = f'' = exp(x) = f + 1`.",
            |x: T| x.exp_m1(),
            |_x: T, r: T| r + T::one(),
            |_x: T, r: T, _d: T| r + T::one());

        // --- roots / powers / abs ---
        $stamp!(sqrt, "Square root. `f' = 1/(2√x)`, `f'' = -1/(4·x·√x)`.",
            |x: T| x.sqrt(),
            |_x: T, r: T| T::from(0.5).unwrap() / r,
            |x: T, r: T, _d: T| -T::from(0.25).unwrap() / (r * x));
        $stamp!(cbrt, "Cube root. `f' = 1/(3·f²)`, `f'' = -2/(9·x·f²)`.",
            |x: T| x.cbrt(),
            |_x: T, r: T| T::one() / (T::from(3.0).unwrap() * r * r),
            |x: T, r: T, _d: T| -T::from(2.0).unwrap()
                / (T::from(9.0).unwrap() * x * r * r));
        $stamp!(abs,
            "Absolute value. `f' = sign(x)`, `f'' = 0` (away from 0). \
             At `x = 0` the derivative is not defined; every surface takes \
             the sub-gradient `f'(0) = +1` (the `x >= 0` branch), matching \
             the winning-branch convention `max`/`min` use for payoffs.",
            |x: T| x.abs(),
            |x: T, _r: T| if x >= T::zero() { T::one() } else { -T::one() },
            |_x: T, _r: T, _d: T| T::zero());

        // --- error function / normal distribution ---
        $stamp!(erf,
            "Gaussian error function. `f' = (2/√π)·e^{-x²}`, `f'' = -2x·f'`.",
            |x: T| x.erf_value(),
            |x: T, _r: T| T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap()
                * (-x * x).exp(),
            |x: T, _r: T, d: T| -T::from(2.0).unwrap() * x * d);
        $stamp!(erfc,
            "Complementary error function `1 - erf(x)`. `f' = -(2/√π)·e^{-x²}`.",
            |x: T| T::one() - x.erf_value(),
            |x: T, _r: T| -T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap()
                * (-x * x).exp(),
            |x: T, _r: T, d: T| -T::from(2.0).unwrap() * x * d);
        $stamp!(norm_cdf,
            "Standard normal CDF `Φ(x)`. `f' = φ(x)`, `f'' = -x·φ(x)`.",
            |x: T| crate::math::norm_cdf(x),
            |x: T, _r: T| crate::math::norm_pdf(x),
            |x: T, _r: T, d: T| -x * d);
        $stamp!(norm_pdf,
            "Standard normal density `φ(x)`. `f' = -x·φ(x)`, `f'' = (x² − 1)·φ(x)`.",
            |x: T| crate::math::norm_pdf(x),
            |x: T, r: T| -x * r,
            |x: T, r: T, _d: T| (x * x - T::one()) * r);
        $stamp!(inv_norm_cdf,
            "Inverse standard normal CDF `Φ⁻¹(p)`. `f' = 1/φ(f)`, `f'' = f·f'²`.",
            |x: T| x.inv_norm_cdf_value(),
            |_x: T, r: T| T::one() / crate::math::norm_pdf(r),
            |_x: T, r: T, d: T| r * d * d);
    };
}
pub(crate) use for_each_unary_elementary;

#[cfg(test)]
mod tests {
    //! Table-stamped uniformity tests: one test per elementary, asserting
    //!
    //! 1. all five AD surfaces (`math::ad`, `math::fwd`, `math::fwdk`,
    //!    `Jet2`, `Jet2Vec`) agree **bit-exactly** with the table closures evaluated on `f64`
    //!    (they share those closures, so any disagreement is a stamping
    //!    bug), and
    //! 2. the table's `d1`/`d2` closures agree with central finite
    //!    differences of the value closure (catches a wrong hand-derived
    //!    formula in the table itself — the one error mode the
    //!    bit-equality check cannot see).

    use crate::forward::{Jet1, Jet2, Jet2Vec, JetK};
    use crate::math;
    use crate::reverse::AReal;
    use crate::tape::Tape;

    /// In-domain probe points per elementary (with margin for the ±h
    /// finite-difference stencil).
    fn points(name: &str) -> &'static [f64] {
        match name {
            "asin" | "acos" | "atanh" => &[-0.7, 0.05, 0.3],
            "acosh" => &[1.3, 2.7],
            "ln" | "log2" | "log10" | "sqrt" | "cbrt" => &[0.4, 1.7, 3.2],
            "ln_1p" => &[-0.4, 0.9],
            "inv_norm_cdf" => &[0.12, 0.5, 0.87],
            "abs" => &[-1.3, 0.8],
            _ => &[-0.9, 0.37, 1.21],
        }
    }

    macro_rules! stamp_uniformity_test {
        ($name:ident, $doc:literal, $val:expr, $d1:expr, $d2:expr) => {
            paste::paste! {
                // Reference evaluation of the table closures. Generic over
                // `T: Passive` so `T::one()` / `T::from(..)` inside the
                // closures resolve exactly as they do in the real stamps;
                // the test instantiates it at `T = f64`.
                fn [<refs_ $name>]<T: crate::passive::Passive>(x0: T) -> (T, T, T) {
                    let v = ($val)(x0);
                    let d1 = ($d1)(x0, v);
                    let d2 = ($d2)(x0, v, d1);
                    (v, d1, d2)
                }

                #[test]
                fn [<uniform_ $name>]() {
                    for &x0 in points(stringify!($name)) {
                        // Reference: the table closures on plain f64.
                        let (v_ref, d1_ref, d2_ref) = [<refs_ $name>]::<f64>(x0);

                        // --- forward Jet1 ---
                        let j1 = math::fwd::$name(&Jet1::new(x0, 1.0));
                        assert_eq!(j1.value(), v_ref, "Jet1 value");
                        assert_eq!(j1.derivative(), d1_ref, "Jet1 d1");

                        // --- forward JetK, lane 0 seeded, lane 1 not ---
                        let jk = math::fwdk::$name(&JetK::<f64, 2>::new(x0, [1.0, 0.0]));
                        assert_eq!(jk.value, v_ref, "JetK value");
                        assert_eq!(jk.tangents[0], d1_ref, "JetK lane-0 d1");
                        assert_eq!(jk.tangents[1], 0.0, "JetK unseeded lane");

                        // --- reverse AReal ---
                        let mut tape = Tape::<f64>::new(true);
                        {
                            let _rec = tape.record();
                            let x = AReal::input(x0, &mut tape);
                            let mut y = math::ad::$name(&x);
                            y.register(&mut tape);
                            y.set_adjoint(&mut tape, 1.0);
                            tape.compute_adjoints();
                            assert_eq!(y.value(), v_ref, "AReal value");
                            assert_eq!(x.adjoint(&tape), d1_ref, "AReal d1");
                        }

                        // --- Jet2 ---
                        let j2 = Jet2::variable(x0).$name();
                        assert_eq!(j2.value(), v_ref, "Jet2 value");
                        assert_eq!(j2.first_derivative(), d1_ref, "Jet2 d1");
                        assert_eq!(j2.second_derivative(), d2_ref, "Jet2 d2");

                        // --- Jet2Vec ---
                        let j2v = Jet2Vec::variable(x0, 0, 1).$name();
                        assert_eq!(j2v.value(), v_ref, "Jet2Vec value");
                        assert_eq!(j2v.gradient()[0], d1_ref, "Jet2Vec d1");
                        assert_eq!(j2v.hessian()[[0, 0]], d2_ref, "Jet2Vec d2");

                        // --- FD sanity on the table formulas themselves ---
                        // d1 tolerance covers the erf-family, whose value is an
                        // approximation polynomial: its FD slope differs from the
                        // exact analytic d1 by ~1e-6.
                        let h = 1e-6 * (1.0 + x0.abs());
                        let (vp, _, _) = [<refs_ $name>]::<f64>(x0 + h);
                        let (vm, _, _) = [<refs_ $name>]::<f64>(x0 - h);
                        let d1_fd = (vp - vm) / (2.0 * h);
                        let d2_fd = (vp - 2.0 * v_ref + vm) / (h * h);
                        let d1_tol = 1e-4 * (1.0 + d1_ref.abs());
                        // FD roundoff on d2 is ~eps·|f|/h² ≈ 2e-4·|f|.
                        let d2_tol = 5e-3 * (1.0 + v_ref.abs() + d2_ref.abs());
                        assert!(
                            (d1_fd - d1_ref).abs() <= d1_tol,
                            "{} d1 vs FD at x={x0}: table={d1_ref} fd={d1_fd}",
                            stringify!($name)
                        );
                        assert!(
                            (d2_fd - d2_ref).abs() <= d2_tol,
                            "{} d2 vs FD at x={x0}: table={d2_ref} fd={d2_fd}",
                            stringify!($name)
                        );
                    }
                }
            }
        };
    }
    super::for_each_unary_elementary!(stamp_uniformity_test);

    /// `norm_pdf`'s `d2` closure, checked against a finite difference of the
    /// **analytic** `d1` rather than a second difference of the value.
    ///
    /// The stamped `uniform_norm_pdf` test above compares `d2` with
    /// `(f(x+h) - 2f(x) + f(x-h))/h²`, whose ~2e-4 relative roundoff floor is
    /// wide enough to pass a `d2` closure that is wrong by a small factor.
    /// Differencing `d1` instead — which every second-order surface computes
    /// from the table's own `d1` closure — has an ~1e-10 floor, so a wrong
    /// `(x² − 1)·φ(x)` fails here.
    #[test]
    fn norm_pdf_second_derivative_matches_fd_of_first() {
        for &x0 in &[-2.3_f64, -0.7, 0.0, 0.37, 1.21, 2.9] {
            let h = 1e-5;
            let d1_at = |x: f64| Jet2::variable(x).norm_pdf().first_derivative();
            let d2_fd = (d1_at(x0 + h) - d1_at(x0 - h)) / (2.0 * h);

            let d2_jet2 = Jet2::variable(x0).norm_pdf().second_derivative();
            let d2_jet2vec = Jet2Vec::variable(x0, 0, 1).norm_pdf().hessian()[[0, 0]];

            // The two second-order surfaces share the closure, so they must
            // agree bit-exactly; only the FD comparison carries a tolerance.
            assert_eq!(d2_jet2, d2_jet2vec, "Jet2 vs Jet2Vec d2 at x={x0}");
            assert!(
                (d2_fd - d2_jet2).abs() <= 1e-9 * (1.0 + d2_jet2.abs()),
                "norm_pdf d2 at x={x0}: closure={d2_jet2} fd(d1)={d2_fd}"
            );
        }
    }

    /// `norm_cdf`'s first derivative is `norm_pdf` — the closure predates the
    /// `norm_pdf` table entry and must not have moved when it was added.
    #[test]
    fn norm_cdf_first_derivative_is_the_density() {
        for &x0 in &[-2.3_f64, -0.7, 0.0, 0.37, 1.21, 2.9] {
            let cdf = math::fwd::norm_cdf(&Jet1::new(x0, 1.0));
            assert_eq!(cdf.value(), crate::math::norm_cdf(x0), "norm_cdf value");
            assert_eq!(
                cdf.derivative(),
                crate::math::norm_pdf(x0),
                "norm_cdf d1 must be the passive density at x={x0}"
            );
            // And the density reached through the new table entry is the same
            // function, not a re-derivation.
            assert_eq!(
                math::fwd::norm_pdf(&Jet1::new(x0, 1.0)).value(),
                crate::math::norm_pdf(x0),
                "norm_pdf value"
            );
        }
    }
}
