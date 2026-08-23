//! The `Real` trait — the unified *active*-scalar trait that abstracts
//! over every AD mode this crate ships (reverse via [`crate::AReal`],
//! forward first-order via [`crate::Jet1`], forward second-order via
//! [`crate::Jet2`]) plus the no-AD case (plain [`f64`]).
//!
//! Conceptually, `Real` is the seam that makes one body of numerical
//! code (a pricer, a loss, an ODE step) run under every AD mode without
//! duplication: the trait abstracts the *active scalar* and the chain
//! rule lives inside its operator impls.
//!
//! See also (theory): [`docs/theory/01-automatic-differentiation.md`](https://github.com/sercanatalik/xad-rs/blob/main/docs/theory/01-automatic-differentiation.md).
//!
//! Program your numerical code against `R: Real` once; instantiate
//! against the concrete mode that matches the problem shape at the call
//! site.
//!
//! # Why `Real`?
//!
//! Aligns with the QuantLib convention (`QuantLib::Real`,
//! `QuantLibAAD::Real`, `QuantLibAdjoint::Real`), where the same name
//! denotes the user-facing scalar type regardless of whether the build
//! is plain double or AD-enabled. Same idea here: code written against
//! `Real` reads the same whether it ultimately runs over `f64`,
//! `AReal<f64>`, `Jet1<f64>`, or `Jet2<f64>`.
//!
//! The complementary trait [`crate::Passive`] is the bound on the
//! *underlying* (storage) scalar — typically `f64`. The
//! `Real::Passive` associated type ties an active type back to the
//! passive type it wraps.
//!
//! # The passive-reference rule
//!
//! **A mode determines which derivatives are available. It does not
//! determine the number that comes out.** Every operation must produce a
//! bit-identical value in every mode, and where an active mode would
//! otherwise differ, the active mode is brought to the passive result —
//! not the reverse.
//!
//! `f64` is the reference not because it is the more accurate candidate in
//! general, but because a crate offering one generic body under several
//! modes needs one of them to be the referent, and the mode without AD
//! machinery is what a caller comparing against a hand-written
//! implementation will have. It is also the only mode whose result a user
//! can reproduce without this crate.
//!
//! The rule has a corollary that is easy to violate by accident: **an
//! intermediate held only to form partial derivatives must not decide the
//! operation's value.** An operation may compute such an intermediate; it
//! may not let it change the value's rounding. Division is the case where
//! this bites. A quotient recorded as `a * (1/b)` rounds twice where
//! `a / b` rounds once, and the two land on different `f64`s for roughly a
//! quarter of operand pairs — so an active mode that reuses the reciprocal
//! it needs for `∂/∂a = 1/b` would return a value up to 1 ulp away from
//! what `f64` returns for the same inputs. Every `Div` impl in this crate
//! therefore records `a / b` and keeps the reciprocal for the partials
//! alone.
//!
//! An implementor of a future mode inherits this obligation. The property
//! is asserted in `tests/division_value_identity.rs` over a randomised
//! sweep, and `tests/real_uniformity.rs` evaluates one body containing
//! every arithmetic operation under all four implementors; a mode that
//! reintroduces the divergence fails there rather than in a caller.
//!


use crate::passive::Passive;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Stamp one unary [`Real`] method *declaration* from a crate-wide
/// derivative-table entry `(name, doc, |x| val, |x, r| d1, |x, r, d1| d2)`.
///
/// `Real` only needs the entry's name and documentation — the chain rule
/// for these methods lives in each implementor's target
/// (`math::ad`, `math::fwd`, `Jet2`'s inherent methods, or plain `f64`),
/// all of which are stamped from the same three closures. So the three
/// closures are matched and discarded here.
macro_rules! declare_real_unary {
    ($name:ident, $doc:literal, $val:expr, $d1:expr, $d2:expr) => {
        #[doc = $doc]
        fn $name(&self) -> Self;
    };
}

/// Unified active-scalar trait — the seam between mode-agnostic
/// numerical code and the concrete AD type it eventually runs over.
///
/// See the module-level docs for the alignment with QuantLib's `Real`
/// and the rationale for not depending on [`num_traits::Float`].
pub trait Real:
    Clone
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + From<f64>
    + From<i32>
    + Send
    + Sync
    + 'static
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
{
    /// The underlying passive (non-AD) scalar type — typically [`f64`].
    type Passive: Passive;

    /// Project the active scalar back to its underlying passive value,
    /// stripping any AD machinery.
    fn value(&self) -> Self::Passive;

    // ------------------------------------------------------------------
    // Numeric identities
    //
    // Declared as associated *functions* rather than associated `const`
    // items: a mode whose representation is not a compile-time constant
    // (the dense second-order mode carries heap storage) can implement a
    // function but not a `const`. Generic code uses these instead of
    // converting a float literal at every site.
    // ------------------------------------------------------------------

    /// The additive identity — the value `x + Self::zero() == x` holds for,
    /// with a zero derivative in every active mode.
    fn zero() -> Self;

    /// The multiplicative identity — the value `x * Self::one() == x` holds
    /// for, with a zero derivative in every active mode.
    fn one() -> Self;

    /// Power with an arbitrary-real exponent.
    fn powf(&self, exponent: Self) -> Self;

    /// Power with an integer exponent.
    fn powi(&self, exponent: i32) -> Self;

    // ------------------------------------------------------------------
    // Unary elementaries — stamped from the crate-wide derivative table in
    // `src/elementaries.rs`. `Real` is the fifth surface that table feeds
    // (after `math::ad`, `math::fwd`, `Jet2`, and `Jet2Vec`); it needs only
    // each entry's name, so `declare_real_unary!` discards the three
    // closures. There is deliberately no parallel list of these names
    // anywhere in the crate — adding a table entry adds the trait method
    // and every implementation of it with no other edit.
    // ------------------------------------------------------------------
    crate::elementaries::for_each_unary_elementary!(declare_real_unary);


    // ------------------------------------------------------------------
    // Fused n-ary aggregates
    //
    // Accumulation loops dominate real pricers (swap legs, Monte Carlo
    // payoff averages). These are methods with per-mode bodies, not free
    // generic functions, precisely so reverse mode can record **one** n-ary
    // tape statement where a binary-operator chain would record `n - 1` —
    // a free function cannot dispatch to a per-mode body without
    // specialisation, so genericity would silently cost the fused
    // recording. See `crate::math::ad::{sum, dot, weighted_sum,
    // weighted_dot}` for the reverse-mode cost model.
    //
    // The weighted forms take **passive** weights: accrual factors and
    // notionals are contract data, not differentiable market inputs.
    // ------------------------------------------------------------------

    /// Sum `Σᵢ xs[i]`.
    ///
    /// An empty slice returns [`Real::zero`].
    fn sum(xs: &[Self]) -> Self;

    /// Dot product `Σᵢ xs[i]·ys[i]`.
    ///
    /// Empty slices return [`Real::zero`].
    ///
    /// # Panics
    /// Panics if `xs.len() != ys.len()`.
    fn dot(xs: &[Self], ys: &[Self]) -> Self;

    /// Weighted sum `Σᵢ ws[i]·xs[i]` with passive weights — the
    /// discounted-cashflow shape.
    ///
    /// Empty slices return [`Real::zero`].
    ///
    /// # Panics
    /// Panics if `ws.len() != xs.len()`.
    fn weighted_sum(ws: &[Self::Passive], xs: &[Self]) -> Self;

    /// Weighted dot product `Σᵢ ws[i]·xs[i]·ys[i]` with passive weights —
    /// the premium-leg shape (accrual · discount · survival).
    ///
    /// Empty slices return [`Real::zero`].
    ///
    /// # Panics
    /// Panics if `ws`, `xs`, and `ys` do not all have the same length.
    fn weighted_dot(ws: &[Self::Passive], xs: &[Self], ys: &[Self]) -> Self;

    /// The larger of `self` and `other` (by value). For AD types the
    /// derivative follows the winning branch — the standard sub-gradient
    /// convention for payoff-style `max` in pricing code.
    fn max(&self, other: &Self) -> Self;

    /// The smaller of `self` and `other` (by value); derivative follows
    /// the winning branch.
    fn min(&self, other: &Self) -> Self;
}

// ----------------------------------------------------------------------------
// impl Real for f64 — the no-AD blanket so generic code over `R: Real`
// compiles for plain `f64` without special-casing.
// ----------------------------------------------------------------------------

/// Stamp one unary `Real` method *body* for `f64` from a derivative-table
/// entry.
///
/// Most entries delegate to the `f64` inherent method of the same name. The
/// Gaussian family has no `f64` inherent form, so those five route to the
/// crate's passive free functions in [`crate::math`] — which is where the
/// full-precision evaluation and the [`Passive`] value hooks already live.
/// Routing them there (rather than re-deriving) is what keeps a generic body
/// instantiated at `f64` bit-identical to the same formula written
/// non-generically.
macro_rules! impl_real_f64_unary {
    (erf, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn erf(&self) -> Self { crate::math::erf(*self) }
    };
    (erfc, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn erfc(&self) -> Self { crate::math::erfc(*self) }
    };
    (norm_pdf, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn norm_pdf(&self) -> Self { crate::math::norm_pdf(*self) }
    };
    (norm_cdf, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn norm_cdf(&self) -> Self { crate::math::norm_cdf(*self) }
    };
    (inv_norm_cdf, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn inv_norm_cdf(&self) -> Self { crate::math::inv_norm_cdf(*self) }
    };
    ($name:ident, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn $name(&self) -> Self { f64::$name(*self) }
    };
}

impl Real for f64 {
    type Passive = f64;

    #[inline]
    fn value(&self) -> f64 {
        *self
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn powf(&self, exponent: Self) -> Self {
        f64::powf(*self, exponent)
    }
    #[inline]
    fn powi(&self, exponent: i32) -> Self {
        f64::powi(*self, exponent)
    }
    // Plain loops — no AD machinery, nothing recorded even when a tape is
    // active on the thread. Shaped so the compiler can vectorise them.
    #[inline]
    fn sum(xs: &[Self]) -> Self {
        let mut acc = 0.0;
        for &x in xs {
            acc += x;
        }
        acc
    }
    #[inline]
    fn dot(xs: &[Self], ys: &[Self]) -> Self {
        assert_eq!(xs.len(), ys.len(), "dot: slice length mismatch");
        let mut acc = 0.0;
        for (&x, &y) in xs.iter().zip(ys) {
            acc += x * y;
        }
        acc
    }
    #[inline]
    fn weighted_sum(ws: &[Self::Passive], xs: &[Self]) -> Self {
        assert_eq!(ws.len(), xs.len(), "weighted_sum: slice length mismatch");
        let mut acc = 0.0;
        for (&w, &x) in ws.iter().zip(xs) {
            acc += w * x;
        }
        acc
    }
    #[inline]
    fn weighted_dot(ws: &[Self::Passive], xs: &[Self], ys: &[Self]) -> Self {
        assert_eq!(ws.len(), xs.len(), "weighted_dot: slice length mismatch");
        assert_eq!(xs.len(), ys.len(), "weighted_dot: slice length mismatch");
        let mut acc = 0.0;
        for ((&w, &x), &y) in ws.iter().zip(xs).zip(ys) {
            acc += w * x * y;
        }
        acc
    }
    #[inline]
    fn max(&self, other: &Self) -> Self {
        f64::max(*self, *other)
    }
    #[inline]
    fn min(&self, other: &Self) -> Self {
        f64::min(*self, *other)
    }

    crate::elementaries::for_each_unary_elementary!(impl_real_f64_unary);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poly<R: Real>(x: &R) -> R {
        x.clone() * x.clone() - R::from(3.0_f64) * x.clone() + R::from(2.0_f64)
    }

    #[test]
    fn poly_at_three_under_f64_equals_two() {
        // (x - 1)(x - 2) at x = 3 is 2
        assert_eq!(poly(&3.0_f64), 2.0);
    }

    #[test]
    fn ln_exp_round_trips_under_f64() {
        let x: f64 = 2.5;
        let y = <f64 as Real>::ln(&x);
        let z = <f64 as Real>::exp(&y);
        assert!((z - 2.5).abs() < 1e-12);
    }

    #[test]
    fn value_is_reflexive_on_f64() {
        let x: f64 = 3.5;
        assert_eq!(<f64 as Real>::value(&x), 3.5);
    }

    /// `Real`'s unary method set equals the crate-wide elementary table's
    /// entry set. The two directions are established separately:
    ///
    /// - **table ⊆ trait**: for every table entry, the function-pointer
    ///   coercions below only compile if `Real` declares a method of that
    ///   name *and* all four implementors provide it. A table entry that
    ///   failed to reach `Real` is a compile error, not a failed assertion.
    /// - **trait ⊆ table**: the trait body declares its unary methods
    ///   exclusively through `declare_real_unary!`, so the source assertion
    ///   in [`no_hand_written_unary_name_list_exists`] — exactly one
    ///   `(&self) -> Self;` in this file, the one inside that macro — pins
    ///   that no method was added by hand beside the stamp.
    #[test]
    fn real_unary_method_set_equals_the_elementary_table() {
        use crate::forward::{Jet1, Jet2};
        use crate::reverse::AReal;

        let mut names: Vec<&'static str> = Vec::new();

        macro_rules! check_real_has {
            ($name:ident, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {{
                let _: fn(&f64) -> f64 = <f64 as Real>::$name;
                let _: fn(&AReal<f64>) -> AReal<f64> = <AReal<f64> as Real>::$name;
                let _: fn(&Jet1<f64>) -> Jet1<f64> = <Jet1<f64> as Real>::$name;
                let _: fn(&Jet2<f64>) -> Jet2<f64> = <Jet2<f64> as Real>::$name;
                names.push(stringify!($name));
            }};
        }
        crate::elementaries::for_each_unary_elementary!(check_real_has);

        // The four names this unification added must be present — the whole
        // point of stamping `Real` from the table.
        for want in ["erf", "erfc", "norm_cdf", "norm_pdf", "inv_norm_cdf"] {
            assert!(names.contains(&want), "table entry `{want}` missing");
        }

        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate table entry names");
    }

    /// No hand-maintained parallel list of unary elementary method names
    /// exists. The per-trait name-list macro `Real` used to keep beside the
    /// table — a hand-written 22 names against the table's 27 — is gone, and
    /// the trait declares its unary methods only through the table stamp.
    ///
    /// The needle is assembled at runtime so this test's own source does not
    /// contain the token it searches for.
    #[test]
    fn no_hand_written_unary_name_list_exists() {
        let needle = format!("for_each_{}", "real_unary");
        for (file, src) in [
            ("real.rs", include_str!("real.rs")),
            ("reverse/areal.rs", include_str!("reverse/areal.rs")),
            ("forward/jet1.rs", include_str!("forward/jet1.rs")),
            ("forward/jet2.rs", include_str!("forward/jet2.rs")),
        ] {
            assert!(
                !src.contains(&needle),
                "{file} still references the deleted parallel name list"
            );
        }

        // Exactly one unary `Real` declaration line in this file: the one
        // inside `declare_real_unary!`. A unary method added by hand to the
        // trait body would make this two. Comment lines are skipped so the
        // prose above (which quotes the shape) does not count itself.
        let shape = format!("(&self) -> {};", "Self");
        let decls = include_str!("real.rs")
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("fn ") && l.ends_with(&shape))
            .count();
        assert_eq!(decls, 1, "unary `Real` methods must be declared only by the table stamp");
    }

    /// The passive trait method and the passive free function are the same
    /// evaluation, so a generic body instantiated at `f64` agrees bit-for-bit
    /// with the same formula written non-generically.
    #[test]
    fn passive_gaussian_trait_methods_are_bit_identical_to_the_free_functions() {
        for &x in &[-6.5_f64, -3.0, -0.3, 0.0, 0.77, 2.999, 3.1, 6.5] {
            assert_eq!(Real::erf(&x), crate::math::erf(x), "erf({x})");
            assert_eq!(Real::erfc(&x), crate::math::erfc(x), "erfc({x})");
            assert_eq!(Real::norm_pdf(&x), crate::math::norm_pdf(x), "norm_pdf({x})");
            assert_eq!(Real::norm_cdf(&x), crate::math::norm_cdf(x), "norm_cdf({x})");
        }
        for &p in &[1e-6_f64, 0.02, 0.2, 0.5, 0.9, 0.98, 1.0 - 1e-6] {
            assert_eq!(
                Real::inv_norm_cdf(&p),
                crate::math::inv_norm_cdf(p),
                "inv_norm_cdf({p})"
            );
        }
    }
}
