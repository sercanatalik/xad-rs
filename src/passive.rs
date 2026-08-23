//! The `Passive` trait — the bound for the underlying *passive* (non-AD)
//! scalar storage type that the active scalars (`AReal<T>`, `Jet1<T>`,
//! `Jet2<T>`) wrap.
//!
//! The name makes the passive/active distinction explicit: `Passive` is
//! the bound on the plain storage scalar, while the QuantLib-aligned
//! [`Real`](crate::real::Real) is the unified *active*-scalar trait.

use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::{Debug, Display};

/// Trait bound for the passive (non-AD) scalar storage type usable in the
/// AD machinery. Implemented for `f64` and — via `jet1_passive.rs` and
/// `jetk.rs` — for `Jet1<T>` and `JetK<T, K>`, so a forward dual can serve
/// as the tape storage scalar of the forward-over-adjoint engines. The
/// trait stays generic over `T` precisely so that nesting works.
pub trait Passive:
    Float + NumAssign + FromPrimitive + Debug + Display + Default + Send + Sync + 'static
{
    /// Value of the Gaussian error function `erf(self)`.
    ///
    /// Default: the full-precision split evaluation in
    /// [`crate::math::erf_impl`] (series below `|x| = 3`, Gauss continued
    /// fraction above; worst measured relative error `1.3e-15`). `Jet1<T>`
    /// **overrides** this to pair the value with the *exact* analytic
    /// tangent `(2/√π)·e^{-x²}·ẋ` — evaluating the implementation in `Jet1`
    /// arithmetic would propagate the approximation's own derivative, which
    /// leaks into second-order results of the forward-over-adjoint engine.
    #[inline]
    fn erf_value(self) -> Self {
        crate::math::erf_impl(self)
    }

    /// Value of the inverse standard normal CDF `Φ⁻¹(self)`.
    ///
    /// Default: Acklam's rational approximation (~1.15e-9). `Jet1<T>`
    /// overrides it to carry the exact tangent `ẋ / φ(Φ⁻¹(x))` — same
    /// rationale as [`erf_value`](Passive::erf_value).
    ///
    /// # Panics
    /// Panics if `self` is outside `(0, 1)` (exclusive).
    #[inline]
    fn inv_norm_cdf_value(self) -> Self {
        crate::math::inv_norm_cdf_poly(self)
    }
}

impl Passive for f64 {}
