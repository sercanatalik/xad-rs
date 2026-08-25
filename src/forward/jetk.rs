//! `JetK<T, K>` — const-generic K-wide first-order forward dual.
//!
//! Where [`Jet1<T>`](crate::forward::jet1::Jet1) carries one tangent,
//! `JetK<T, K>` carries `K` simultaneous tangent directions in a fixed
//! `[T; K]` array (stack-allocated, `Copy`). It plays two roles:
//!
//! - **A [`Real`](crate::Real) mode in its own right.** `JetK<f64, K>`
//!   implements `Real` for every `K`, so a body written once as
//!   `fn f<R: Real>` evaluates `K` input directions in one tape-free pass:
//!   seed input `i`'s lane `i` to one and read `∂f/∂xᵢ` from lane `i` of the
//!   output. For a scalar output with `n ≤ K` inputs that is the whole
//!   gradient in one pass — see
//!   [`compute_gradient_fwd_k`](crate::compute_gradient_fwd_k) for the
//!   `⌈n/K⌉`-pass driver.
//! - **The storage scalar of the K-direction forward-over-adjoint Hessian
//!   engine.** As `Tape<JetK<f64, K>>` (see
//!   [`compute_hessian_k`](crate::compute_hessian_k)), one recording + one
//!   sweep yields `K` Hessian columns at once, cutting the number of
//!   forward-over-adjoint passes from `n` to `⌈n/K⌉`.
//!
//! The implementation mirrors `jet1_passive.rs` with the tangent scalar
//! replaced by the array: every chain rule becomes an elementwise loop
//! over the `K` lanes, which auto-vectorizes. Per-op cost therefore grows
//! sublinearly in `K`, which is why the engine speedup tracks `K` almost
//! ideally (measured: 4.6× / 8.2× for K = 4/8 on a 48-input, 2000-op
//! Hessian kernel). Every unary elementary is stamped from the crate's
//! single derivative table through `math::fwdk`, so lane `i` evolves
//! exactly as the corresponding `Jet1` tangent would: for every table
//! entry and for `+ − ×` the lane is **bit-identical** to `Jet1`'s
//! tangent. Division's tangent is computed in a different operation order
//! from `Jet1`'s (`tₐ·inv + t_b·(−(a·inv)·inv)` here, `(tₐ·b − a·t_b)·inv²`
//! there) and agrees to a few ulp; values are the correctly rounded
//! quotient in both.
//!
//! Larger `K` grows `Operation<JetK<f64, K>>` (the tape's per-operand
//! record) linearly — 24 B at K = 2 up to 72 B at K = 8 — so tape memory
//! traffic eventually erodes the lane amortization *inside the engine*;
//! K = 4–8 is the measured sweet spot on Apple M-series there. Used as a
//! tape-free mode there is no operand record, and the sweet spot is set by
//! register pressure instead — see `examples/jetk_gradient.rs`.
//!
//! `TapeStorage` (the thread-local active-tape slot) must be stamped per
//! concrete scalar type; it is provided for `JetK<f64, K>` with
//! K ∈ {2, 4, 8}. Other lane counts need one more `jetk_tape_storage!`
//! line here. The `Real` impl has no such limit.
//!
//! `JetK<f64, K>` implements both [`Passive`] (the storage bound) and
//! `Real` (the active surface), as `Jet1<f64>` does. The method-name
//! overlap (`sin`, `cos`, …) is resolved by which trait the bound names —
//! generic code over `T: Passive` uses the `Float` methods; nothing calls a
//! `Real` method through a `Passive` bound.
//!
//! `erf` / `inv_norm_cdf` values carry **exact** analytic tangents (same
//! rationale and mechanism as `Jet1<T>` — see `jet1_passive.rs`).

use num_traits::{Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use std::cell::Cell;
use std::fmt;
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::ptr;

use crate::math;
use crate::passive::Passive;
use crate::tape::{Tape, TapeStorage};

/// K-wide forward dual: one value, `K` simultaneous tangent directions.
#[derive(Clone, Copy, Debug)]
pub struct JetK<T: Passive, const K: usize> {
    pub value: T,
    pub tangents: [T; K],
}

impl<T: Passive, const K: usize> JetK<T, K> {
    #[inline]
    pub fn new(value: T, tangents: [T; K]) -> Self {
        JetK { value, tangents }
    }

    #[inline]
    pub fn constant(value: T) -> Self {
        JetK { value, tangents: [T::zero(); K] }
    }

    /// Unary chain rule: result value `r`, derivative `d` — tangents scale
    /// elementwise.
    #[inline]
    pub(crate) fn chain(self, r: T, d: T) -> Self {
        let mut t = self.tangents;
        for x in t.iter_mut() {
            *x *= d;
        }
        JetK { value: r, tangents: t }
    }

    /// Binary chain rule: `r = f(a, b)`, `t = da·ta + db·tb`.
    #[inline]
    pub(crate) fn chain2(self, rhs: Self, r: T, da: T, db: T) -> Self {
        let mut t = self.tangents;
        for (x, &tb) in t.iter_mut().zip(rhs.tangents.iter()) {
            *x = *x * da + tb * db;
        }
        JetK { value: r, tangents: t }
    }
}

impl<T: Passive, const K: usize> Default for JetK<T, K> {
    #[inline]
    fn default() -> Self {
        Self::constant(T::zero())
    }
}

// ---- conversions: a constant, never a seeded variable ----

impl<T: Passive, const K: usize> From<T> for JetK<T, K> {
    #[inline]
    fn from(value: T) -> Self {
        Self::constant(value)
    }
}

impl<const K: usize> From<i32> for JetK<f64, K> {
    #[inline]
    fn from(value: i32) -> Self {
        Self::constant(value as f64)
    }
}

impl<T: Passive, const K: usize> fmt::Display for JetK<T, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// ---- comparison: value-only, as for Jet1 / AReal ----

impl<T: Passive, const K: usize> PartialEq for JetK<T, K> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Passive, const K: usize> PartialOrd for JetK<T, K> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

// ---- arithmetic operators (owned forms, all that Num requires) ----

impl<T: Passive, const K: usize> Add for JetK<T, K> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.chain2(rhs, self.value + rhs.value, T::one(), T::one())
    }
}

impl<T: Passive, const K: usize> Sub for JetK<T, K> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.chain2(rhs, self.value - rhs.value, T::one(), -T::one())
    }
}

impl<T: Passive, const K: usize> Mul for JetK<T, K> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.chain2(rhs, self.value * rhs.value, rhs.value, self.value)
    }
}

impl<T: Passive, const K: usize> Div for JetK<T, K> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // The value is the correctly rounded quotient, not `self.value * inv`:
        // that spelling rounds twice and would make the number this mode
        // returns differ from the passive scalar's for the same operands. See
        // the passive-reference rule in `crate::real`.
        //
        // `q_recip` is deliberately NOT the corrected quotient. The `-a/b²`
        // partial is a derivative, and this correction moves values only —
        // reusing the corrected quotient here would shift `JetK`'s b-partial
        // by up to 1 ulp on roughly a quarter of inputs, which is a separate
        // change with its own justification to make.
        let inv = T::one() / rhs.value;
        let q_recip = self.value * inv;
        self.chain2(rhs, self.value / rhs.value, inv, -q_recip * inv)
    }
}

impl<T: Passive, const K: usize> Rem for JetK<T, K> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        let q = (self.value / rhs.value).trunc();
        self.chain2(rhs, self.value % rhs.value, T::one(), -q)
    }
}

impl<T: Passive, const K: usize> Neg for JetK<T, K> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.chain(-self.value, -T::one())
    }
}

macro_rules! jetk_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Passive, const K: usize> $trait for JetK<T, K> {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                *self = *self $op rhs;
            }
        }
    };
}
jetk_assign!(AddAssign, add_assign, +);
jetk_assign!(SubAssign, sub_assign, -);
jetk_assign!(MulAssign, mul_assign, *);
jetk_assign!(DivAssign, div_assign, /);
jetk_assign!(RemAssign, rem_assign, %);

// ---- reference permutations of `JetK op JetK` ----
//
// `JetK` is `Copy`, so every ref/value permutation is the same computation;
// the by-value impls above are the bodies and these deref-and-delegate.

macro_rules! jetk_binop_refs {
    ($trait:ident, $method:ident) => {
        impl<T: Passive, const K: usize> $trait<&JetK<T, K>> for JetK<T, K> {
            type Output = JetK<T, K>;
            #[inline]
            fn $method(self, rhs: &JetK<T, K>) -> JetK<T, K> {
                $trait::$method(self, *rhs)
            }
        }
        impl<T: Passive, const K: usize> $trait<JetK<T, K>> for &JetK<T, K> {
            type Output = JetK<T, K>;
            #[inline]
            fn $method(self, rhs: JetK<T, K>) -> JetK<T, K> {
                $trait::$method(*self, rhs)
            }
        }
        impl<T: Passive, const K: usize> $trait<&JetK<T, K>> for &JetK<T, K> {
            type Output = JetK<T, K>;
            #[inline]
            fn $method(self, rhs: &JetK<T, K>) -> JetK<T, K> {
                $trait::$method(*self, *rhs)
            }
        }
    };
}
jetk_binop_refs!(Add, add);
jetk_binop_refs!(Sub, sub);
jetk_binop_refs!(Mul, mul);
jetk_binop_refs!(Div, div);

// ---- `JetK op T` (scalar RHS): value and ref LHS forms from one body ----

macro_rules! impl_jetk_binop_scalar_rhs {
    ($trait:ident, $method:ident, ($a:ident, $r:ident) => $body:expr) => {
        impl<T: Passive, const K: usize> $trait<T> for JetK<T, K> {
            type Output = JetK<T, K>;
            #[inline]
            fn $method(self, rhs: T) -> JetK<T, K> {
                let ($a, $r) = (self, rhs);
                $body
            }
        }
        impl<T: Passive, const K: usize> $trait<T> for &JetK<T, K> {
            type Output = JetK<T, K>;
            #[inline]
            fn $method(self, rhs: T) -> JetK<T, K> {
                $trait::$method(*self, rhs)
            }
        }
    };
}

impl_jetk_binop_scalar_rhs!(Add, add, (a, r) => JetK { value: a.value + r, tangents: a.tangents });
impl_jetk_binop_scalar_rhs!(Sub, sub, (a, r) => JetK { value: a.value - r, tangents: a.tangents });
impl_jetk_binop_scalar_rhs!(Mul, mul, (a, r) => a.chain(a.value * r, r));
impl_jetk_binop_scalar_rhs!(Div, div, (a, r) => {
    // Quotient for the value, reciprocal for the tangents — see `Div` above.
    let inv = T::one() / r;
    a.chain(a.value / r, inv)
});

macro_rules! jetk_scalar_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Passive, const K: usize> $trait<T> for JetK<T, K> {
            #[inline]
            fn $method(&mut self, rhs: T) {
                *self = *self $op rhs;
            }
        }
    };
}
jetk_scalar_assign!(AddAssign, add_assign, +);
jetk_scalar_assign!(SubAssign, sub_assign, -);
jetk_scalar_assign!(MulAssign, mul_assign, *);
jetk_scalar_assign!(DivAssign, div_assign, /);

// ---- `f64 op JetK<f64, K>` (scalar LHS — the orphan rule forces a concrete
// float type): both RHS ref forms from one body ----

macro_rules! impl_scalar_lhs_jetk_binop {
    ($trait:ident, $method:ident, ($l:ident, $r:ident) => $body:expr) => {
        impl<const K: usize> $trait<JetK<f64, K>> for f64 {
            type Output = JetK<f64, K>;
            #[inline]
            fn $method(self, rhs: JetK<f64, K>) -> JetK<f64, K> {
                let ($l, $r) = (self, rhs);
                $body
            }
        }
        impl<const K: usize> $trait<&JetK<f64, K>> for f64 {
            type Output = JetK<f64, K>;
            #[inline]
            fn $method(self, rhs: &JetK<f64, K>) -> JetK<f64, K> {
                $trait::$method(self, *rhs)
            }
        }
    };
}

impl_scalar_lhs_jetk_binop!(Add, add, (l, r) => JetK { value: l + r.value, tangents: r.tangents });
impl_scalar_lhs_jetk_binop!(Sub, sub, (l, r) => r.chain(l - r.value, -1.0));
impl_scalar_lhs_jetk_binop!(Mul, mul, (l, r) => r.chain(l * r.value, l));
impl_scalar_lhs_jetk_binop!(Div, div, (l, r) => {
    // d(c/x) = -c·x' / x². Quotient for the value, reciprocal for the tangents.
    let inv = 1.0 / r.value;
    r.chain(l / r.value, -l * inv * inv)
});

// ---- Zero / One / Num / casts ----

impl<T: Passive, const K: usize> Zero for JetK<T, K> {
    #[inline]
    fn zero() -> Self {
        Self::constant(T::zero())
    }
    /// Value AND all tangents zero — same rationale as `Jet1::is_zero`:
    /// the reverse sweep's zero-adjoint skip must not drop an adjoint whose
    /// value is 0 but which carries live second-order tangents.
    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero() && self.tangents.iter().all(|t| t.is_zero())
    }
}

impl<T: Passive, const K: usize> One for JetK<T, K> {
    #[inline]
    fn one() -> Self {
        Self::constant(T::one())
    }
}

impl<T: Passive, const K: usize> Num for JetK<T, K> {
    type FromStrRadixErr = T::FromStrRadixErr;
    #[inline]
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(s, radix).map(Self::constant)
    }
}

impl<T: Passive, const K: usize> ToPrimitive for JetK<T, K> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.value.to_f64()
    }
}

impl<T: Passive, const K: usize> NumCast for JetK<T, K> {
    #[inline]
    fn from<N: ToPrimitive>(n: N) -> Option<Self> {
        <T as NumCast>::from(n).map(Self::constant)
    }
}

impl<T: Passive, const K: usize> FromPrimitive for JetK<T, K> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        T::from_i64(n).map(Self::constant)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        T::from_u64(n).map(Self::constant)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        T::from_f64(n).map(Self::constant)
    }
}

// ---- Float ----

/// Stamp a unary `Float` method from `(value, derivative)` closures.
impl<T: Passive, const K: usize> Float for JetK<T, K> {
    #[inline]
    fn nan() -> Self { Self::constant(T::nan()) }
    #[inline]
    fn infinity() -> Self { Self::constant(T::infinity()) }
    #[inline]
    fn neg_infinity() -> Self { Self::constant(T::neg_infinity()) }
    #[inline]
    fn neg_zero() -> Self { Self::constant(T::neg_zero()) }
    #[inline]
    fn min_value() -> Self { Self::constant(T::min_value()) }
    #[inline]
    fn min_positive_value() -> Self { Self::constant(T::min_positive_value()) }
    #[inline]
    fn max_value() -> Self { Self::constant(T::max_value()) }
    #[inline]
    fn epsilon() -> Self { Self::constant(T::epsilon()) }

    #[inline]
    fn is_nan(self) -> bool { self.value.is_nan() }
    #[inline]
    fn is_infinite(self) -> bool { self.value.is_infinite() }
    #[inline]
    fn is_finite(self) -> bool { self.value.is_finite() }
    #[inline]
    fn is_normal(self) -> bool { self.value.is_normal() }
    #[inline]
    fn is_sign_positive(self) -> bool { self.value.is_sign_positive() }
    #[inline]
    fn is_sign_negative(self) -> bool { self.value.is_sign_negative() }
    #[inline]
    fn classify(self) -> FpCategory { self.value.classify() }
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) { self.value.integer_decode() }

    #[inline]
    fn floor(self) -> Self { Self::constant(self.value.floor()) }
    #[inline]
    fn ceil(self) -> Self { Self::constant(self.value.ceil()) }
    #[inline]
    fn round(self) -> Self { Self::constant(self.value.round()) }
    #[inline]
    fn trunc(self) -> Self { Self::constant(self.value.trunc()) }
    #[inline]
    fn signum(self) -> Self { Self::constant(self.value.signum()) }
    #[inline]
    fn fract(self) -> Self { JetK { value: self.value.fract(), tangents: self.tangents } }
    #[inline]
    fn abs(self) -> Self { math::fwdk::abs(&self) }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        if self.value.is_sign_negative() != sign.value.is_sign_negative() {
            -self
        } else {
            self
        }
    }
    #[inline]
    fn to_degrees(self) -> Self {
        let k = T::from(180.0 / std::f64::consts::PI).unwrap();
        self.chain(self.value.to_degrees(), k)
    }
    #[inline]
    fn to_radians(self) -> Self {
        let k = T::from(std::f64::consts::PI / 180.0).unwrap();
        self.chain(self.value.to_radians(), k)
    }
    #[inline]
    fn max(self, other: Self) -> Self { math::fwdk::max(&self, &other) }
    #[inline]
    fn min(self, other: Self) -> Self { math::fwdk::min(&self, &other) }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    #[inline]
    fn recip(self) -> Self {
        let inv = T::one() / self.value;
        self.chain(inv, -inv * inv)
    }

    #[inline]
    fn powi(self, n: i32) -> Self { math::fwdk::powi(&self, n) }
    #[inline]
    fn powf(self, n: Self) -> Self { math::fwdk::pow(&self, &n) }
    #[inline]
    fn hypot(self, other: Self) -> Self { math::fwdk::hypot(&self, &other) }
    #[inline]
    fn atan2(self, other: Self) -> Self { math::fwdk::atan2(&self, &other) }
    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (math::fwdk::sin(&self), math::fwdk::cos(&self))
    }
    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        if self.value <= other.value { Self::constant(T::zero()) } else { self - other }
    }

    #[inline]
    fn sqrt(self) -> Self { math::fwdk::sqrt(&self) }
    #[inline]
    fn cbrt(self) -> Self { math::fwdk::cbrt(&self) }
    #[inline]
    fn exp(self) -> Self { math::fwdk::exp(&self) }
    #[inline]
    fn exp2(self) -> Self { math::fwdk::exp2(&self) }
    #[inline]
    fn exp_m1(self) -> Self { math::fwdk::exp_m1(&self) }
    #[inline]
    fn ln(self) -> Self { math::fwdk::ln(&self) }
    #[inline]
    fn ln_1p(self) -> Self { math::fwdk::ln_1p(&self) }
    #[inline]
    fn log2(self) -> Self { math::fwdk::log2(&self) }
    #[inline]
    fn log10(self) -> Self { math::fwdk::log10(&self) }
    #[inline]
    fn sin(self) -> Self { math::fwdk::sin(&self) }
    #[inline]
    fn cos(self) -> Self { math::fwdk::cos(&self) }
    #[inline]
    fn tan(self) -> Self { math::fwdk::tan(&self) }
    #[inline]
    fn asin(self) -> Self { math::fwdk::asin(&self) }
    #[inline]
    fn acos(self) -> Self { math::fwdk::acos(&self) }
    #[inline]
    fn atan(self) -> Self { math::fwdk::atan(&self) }
    #[inline]
    fn sinh(self) -> Self { math::fwdk::sinh(&self) }
    #[inline]
    fn cosh(self) -> Self { math::fwdk::cosh(&self) }
    #[inline]
    fn tanh(self) -> Self { math::fwdk::tanh(&self) }
    #[inline]
    fn asinh(self) -> Self { math::fwdk::asinh(&self) }
    #[inline]
    fn acosh(self) -> Self { math::fwdk::acosh(&self) }
    #[inline]
    fn atanh(self) -> Self { math::fwdk::atanh(&self) }
}

// ---- Passive (with exact-tangent erf / inv_norm_cdf, as for Jet1) ----

impl<T: Passive, const K: usize> Passive for JetK<T, K> {
    #[inline]
    fn erf_value(self) -> Self {
        let v = self.value;
        let d = T::from(std::f64::consts::FRAC_2_SQRT_PI).unwrap() * (-v * v).exp();
        self.chain(v.erf_value(), d)
    }

    #[inline]
    fn inv_norm_cdf_value(self) -> Self {
        let r = self.value.inv_norm_cdf_value();
        let d = T::one() / crate::math::norm_pdf(r);
        self.chain(r, d)
    }
}

// ---- Real: the K-lane forward mode ----

/// Stamp one unary `Real` method body for `JetK<f64, K>` from a crate-wide
/// derivative-table entry: delegate to the K-lane forward free function of
/// the same name in `math::fwdk`, which the same table stamps.
macro_rules! impl_real_jetk_unary {
    ($name:ident, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn $name(&self) -> Self {
            math::fwdk::$name(self)
        }
    };
}

impl<const K: usize> crate::real::Real for JetK<f64, K> {
    type Passive = f64;

    #[inline]
    fn value(&self) -> f64 {
        self.value
    }
    /// Zero value, zero tangents — a constant, not a seeded variable.
    #[inline]
    fn zero() -> Self {
        Self::constant(0.0)
    }
    /// Unit value, zero tangents — a constant, not a seeded variable.
    #[inline]
    fn one() -> Self {
        Self::constant(1.0)
    }
    #[inline]
    fn powf(&self, exponent: Self) -> Self {
        math::fwdk::pow(self, &exponent)
    }
    #[inline]
    fn powi(&self, exponent: i32) -> Self {
        math::fwdk::powi(self, exponent)
    }
    // Lane loops in slice order — the same operation sequence `Jet1`'s
    // aggregates use, so the value equals the passive scalar's plain loop
    // bit for bit. Forward mode has no fused encoding to preserve.
    #[inline]
    fn sum(xs: &[Self]) -> Self {
        let mut acc = Self::constant(0.0);
        for x in xs {
            acc += *x;
        }
        acc
    }
    #[inline]
    fn dot(xs: &[Self], ys: &[Self]) -> Self {
        assert_eq!(xs.len(), ys.len(), "dot: slice length mismatch");
        let mut acc = Self::constant(0.0);
        for (x, y) in xs.iter().zip(ys) {
            acc += *x * *y;
        }
        acc
    }
    #[inline]
    fn weighted_sum(ws: &[Self::Passive], xs: &[Self]) -> Self {
        assert_eq!(ws.len(), xs.len(), "weighted_sum: slice length mismatch");
        let mut acc = Self::constant(0.0);
        for (&w, x) in ws.iter().zip(xs) {
            acc += Self::constant(w) * *x;
        }
        acc
    }
    #[inline]
    fn weighted_dot(ws: &[Self::Passive], xs: &[Self], ys: &[Self]) -> Self {
        assert_eq!(ws.len(), xs.len(), "weighted_dot: slice length mismatch");
        assert_eq!(xs.len(), ys.len(), "weighted_dot: slice length mismatch");
        let mut acc = Self::constant(0.0);
        for ((&w, x), y) in ws.iter().zip(xs).zip(ys) {
            acc += Self::constant(w) * *x * *y;
        }
        acc
    }
    #[inline]
    fn max(&self, other: &Self) -> Self {
        math::fwdk::max(self, other)
    }
    #[inline]
    fn min(&self, other: &Self) -> Self {
        math::fwdk::min(self, other)
    }

    crate::elementaries::for_each_unary_elementary!(impl_real_jetk_unary);
}

// ---- TapeStorage: one thread-local active-tape slot per concrete K ----

macro_rules! jetk_tape_storage {
    ($k:literal, $slot:ident) => {
        thread_local! {
            static $slot: Cell<*mut Tape<JetK<f64, $k>>> = const { Cell::new(ptr::null_mut()) };
        }

        impl TapeStorage for JetK<f64, $k> {
            #[inline]
            fn get_active_ptr() -> Option<*mut Tape<JetK<f64, $k>>> {
                let p = $slot.with(|c| c.get());
                if p.is_null() { None } else { Some(p) }
            }
            #[inline]
            fn set_active_ptr(ptr: Option<*mut Tape<JetK<f64, $k>>>) {
                $slot.with(|c| c.set(ptr.unwrap_or(std::ptr::null_mut())));
            }
        }
    };
}

jetk_tape_storage!(2, ACTIVE_TAPE_JETK2_F64);
jetk_tape_storage!(4, ACTIVE_TAPE_JETK4_F64);
jetk_tape_storage!(8, ACTIVE_TAPE_JETK8_F64);

#[cfg(test)]
mod tests {
    use super::JetK;
    use crate::real::{CopyableReal, Real};

    type J3 = JetK<f64, 3>;

    /// `x·y + sin(z)` — three inputs, three lanes.
    fn body<R: CopyableReal>(x: R, y: R, z: R) -> R {
        x * y + z.sin()
    }

    fn seeded(x: f64, y: f64, z: f64, lanes: [bool; 3]) -> J3 {
        let seed = |i: usize, v: f64| {
            let mut t = [0.0; 3];
            if lanes[i] {
                t[i] = 1.0;
            }
            JetK::new(v, t)
        };
        body(seed(0, x), seed(1, y), seed(2, z))
    }

    #[test]
    fn seeding_lane_i_yields_the_partial_in_lane_i() {
        let (x, y, z) = (1.3, -0.7, 0.4);
        let all = seeded(x, y, z, [true; 3]);
        assert_eq!(all.value, x * y + z.sin());
        assert_eq!(all.tangents, [y, x, z.cos()]);
    }

    #[test]
    fn lanes_do_not_interfere() {
        let (x, y, z) = (1.3, -0.7, 0.4);
        let all = seeded(x, y, z, [true; 3]);
        let only_y = seeded(x, y, z, [false, true, false]);
        assert_eq!(only_y.tangents[1], all.tangents[1], "lane 1 depends on other seeds");
        assert_eq!(only_y.tangents[0], 0.0);
        assert_eq!(only_y.tangents[2], 0.0);
        assert_eq!(only_y.value, all.value, "seeding moved the value");
    }

    #[test]
    fn conversions_are_constants() {
        let a: J3 = 2.5.into();
        let b: J3 = 3.into();
        assert_eq!(a.value, 2.5);
        assert_eq!(b.value, 3.0);
        assert_eq!(a.tangents, [0.0; 3]);
        assert_eq!(b.tangents, [0.0; 3]);
        assert_eq!(<J3 as Real>::zero(), JetK::constant(0.0));
        assert_eq!(<J3 as Real>::one().tangents, [0.0; 3]);
    }

    /// Each scalar-operand spelling is the lifted spelling's number, bit for
    /// bit, in value and in every lane. The `&x` forms are impls under test,
    /// not a stylistic choice, hence the lint allowance.
    #[test]
    #[allow(clippy::op_ref)]
    fn scalar_operand_spellings_equal_the_lifted_ones() {
        let x = JetK::<f64, 3>::new(1.7, [1.0, -2.0, 0.5]);
        let c = 0.3_f64;
        let lift = J3::from(c);
        let same = |a: J3, b: J3, what: &str| {
            assert_eq!(a.value, b.value, "{what}: value");
            assert_eq!(a.tangents, b.tangents, "{what}: tangents");
        };
        same(x + c, x + lift, "x + c");
        same(x - c, x - lift, "x - c");
        same(x * c, x * lift, "x * c");
        same(x / c, x / lift, "x / c");
        same(c + x, lift + x, "c + x");
        same(c - x, lift - x, "c - x");
        same(c * x, lift * x, "c * x");
        same(c / x, lift / x, "c / x");
        same(&x * c, x * c, "&x * c");
        same(c * &x, c * x, "c * &x");
        let mut y = x;
        y *= c;
        same(y, x * c, "*= c");
    }

    #[test]
    fn a_generic_body_runs_at_any_lane_count() {
        fn poly<R: Real>(x: &R) -> R {
            let two = R::from(2.0);
            x.clone() * x.clone() - two * x.clone() + R::one()
        }
        let reference = poly(&3.0_f64);
        assert_eq!(poly(&JetK::<f64, 3>::constant(3.0)).value, reference);
        assert_eq!(poly(&JetK::<f64, 8>::constant(3.0)).value, reference);
        let j16 = poly(&JetK::<f64, 16>::new(3.0, {
            let mut t = [0.0; 16];
            t[15] = 1.0;
            t
        }));
        assert_eq!(j16.value, reference);
        assert_eq!(j16.tangents[15], 2.0 * 3.0 - 2.0);
        assert_eq!(j16.tangents[..15], [0.0; 15]);
        fn copyable<R: CopyableReal>() {}
        copyable::<JetK<f64, 5>>();
    }

    /// `JetK`'s unary derivatives once lived in a local closure list that
    /// mirrored the elementary table by hand — a parallel list the table's
    /// uniformity test could not see. Every unary now delegates to
    /// `math::fwdk`, which the table stamps. The needles are assembled at
    /// runtime so this test's own source does not contain them.
    #[test]
    fn no_hand_written_derivative_list_exists() {
        let src = include_str!("jetk.rs");
        let old_macro = format!("jetk_{}!", "unary");
        assert!(!src.contains(&old_macro), "the local unary stamp macro is back");
        // The table's `d1` closure shape, in the spellings the old list used.
        for shape in [
            format!("|_x, {}", "r: T|"),
            format!("|x: T, {}", "_r|"),
            format!("|_x: T, {}", "r: T|"),
            format!("|x: T, {}", "_r: T|"),
        ] {
            assert!(
                !src.contains(&shape),
                "a hand-written derivative closure `{shape}` is back in jetk.rs"
            );
        }
    }
}
