//! `JetK<T, K>` — const-generic K-wide first-order forward dual, the tape
//! *storage* scalar of the K-direction forward-over-adjoint Hessian engine.
//!
//! Where [`Jet1<T>`](crate::forward::jet1::Jet1) carries one tangent,
//! `JetK<T, K>` carries `K` simultaneous tangent directions in a fixed
//! `[T; K]` array (stack-allocated, `Copy`). Used as the storage scalar of
//! a reverse-mode tape (`Tape<JetK<f64, K>>`, see
//! [`compute_hessian_k`](crate::compute_hessian_k)), one recording + one
//! sweep yields `K` Hessian columns at once, cutting the number of
//! forward-over-adjoint passes from `n` to `⌈n/K⌉`.
//!
//! The implementation mirrors `jet1_passive.rs` with the tangent scalar
//! replaced by the array: every chain rule becomes an elementwise loop
//! over the `K` lanes, which auto-vectorizes. Per-op cost therefore grows
//! sublinearly in `K`, which is why the engine speedup tracks `K` almost
//! ideally (measured: 4.6× / 8.2× for K = 4/8 on a 48-input, 2000-op
//! Hessian kernel) — every tangent lane evolves exactly as the
//! corresponding `Jet1` tangent would, so results are **bit-identical**
//! to the `Jet1` engine.
//!
//! Larger `K` grows `Operation<JetK<f64, K>>` (the tape's per-operand
//! record) linearly — 24 B at K = 2 up to 72 B at K = 8 — so tape memory
//! traffic eventually erodes the lane amortization; K = 4–8 is the
//! measured sweet spot on Apple M-series.
//!
//! `TapeStorage` (the thread-local active-tape slot) must be stamped per
//! concrete scalar type; it is provided for `JetK<f64, K>` with
//! K ∈ {2, 4, 8}. Other lane counts need one more `jetk_tape_storage!`
//! line here.
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
    fn chain(self, r: T, d: T) -> Self {
        let mut t = self.tangents;
        for x in t.iter_mut() {
            *x *= d;
        }
        JetK { value: r, tangents: t }
    }

    /// Binary chain rule: `r = f(a, b)`, `t = da·ta + db·tb`.
    #[inline]
    fn chain2(self, rhs: Self, r: T, da: T, db: T) -> Self {
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
macro_rules! jetk_unary {
    ($name:ident, $val:expr, $d1:expr) => {
        #[inline]
        fn $name(self) -> Self {
            let x = self.value;
            let r = ($val)(x);
            let d = ($d1)(x, r);
            self.chain(r, d)
        }
    };
}

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
    fn abs(self) -> Self {
        let s = self.value.signum();
        self.chain(self.value.abs(), s)
    }
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
    fn max(self, other: Self) -> Self {
        if self.value >= other.value { self } else { other }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self.value <= other.value { self } else { other }
    }
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
    fn powi(self, n: i32) -> Self {
        let r = self.value.powi(n);
        let d = T::from(n).unwrap() * self.value.powi(n - 1);
        self.chain(r, d)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        let bv = self.value;
        let ev = n.value;
        let r = bv.powf(ev);
        let d_base = ev * bv.powf(ev - T::one());
        let d_exp = r * bv.ln();
        self.chain2(n, r, d_base, d_exp)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        let r = self.value.hypot(other.value);
        let inv = T::one() / r;
        self.chain2(other, r, self.value * inv, other.value * inv)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let denom = self.value * self.value + other.value * other.value;
        self.chain2(
            other,
            self.value.atan2(other.value),
            other.value / denom,
            -self.value / denom,
        )
    }
    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        if self.value <= other.value { Self::constant(T::zero()) } else { self - other }
    }

    jetk_unary!(sqrt, |x: T| x.sqrt(), |_x, r: T| T::one() / (T::from(2.0).unwrap() * r));
    jetk_unary!(cbrt, |x: T| x.cbrt(), |_x, r: T| T::one() / (T::from(3.0).unwrap() * r * r));
    jetk_unary!(exp, |x: T| x.exp(), |_x, r: T| r);
    jetk_unary!(exp2, |x: T| x.exp2(), |_x, r: T| r * T::from(std::f64::consts::LN_2).unwrap());
    jetk_unary!(exp_m1, |x: T| x.exp_m1(), |_x, r: T| r + T::one());
    jetk_unary!(ln, |x: T| x.ln(), |x: T, _r| T::one() / x);
    jetk_unary!(ln_1p, |x: T| x.ln_1p(), |x: T, _r| T::one() / (T::one() + x));
    jetk_unary!(log2, |x: T| x.log2(), |x: T, _r| T::one()
        / (x * T::from(std::f64::consts::LN_2).unwrap()));
    jetk_unary!(log10, |x: T| x.log10(), |x: T, _r| T::one()
        / (x * T::from(std::f64::consts::LN_10).unwrap()));
    jetk_unary!(sin, |x: T| x.sin(), |x: T, _r| x.cos());
    jetk_unary!(cos, |x: T| x.cos(), |x: T, _r| -x.sin());
    jetk_unary!(tan, |x: T| x.tan(), |_x, r: T| T::one() + r * r);
    jetk_unary!(asin, |x: T| x.asin(), |x: T, _r| T::one() / (T::one() - x * x).sqrt());
    jetk_unary!(acos, |x: T| x.acos(), |x: T, _r| -T::one() / (T::one() - x * x).sqrt());
    jetk_unary!(atan, |x: T| x.atan(), |x: T, _r| T::one() / (T::one() + x * x));
    jetk_unary!(sinh, |x: T| x.sinh(), |x: T, _r| x.cosh());
    jetk_unary!(cosh, |x: T| x.cosh(), |x: T, _r| x.sinh());
    jetk_unary!(tanh, |x: T| x.tanh(), |_x, r: T| T::one() - r * r);
    jetk_unary!(asinh, |x: T| x.asinh(), |x: T, _r| T::one() / (x * x + T::one()).sqrt());
    jetk_unary!(acosh, |x: T| x.acosh(), |x: T, _r| T::one() / (x * x - T::one()).sqrt());
    jetk_unary!(atanh, |x: T| x.atanh(), |x: T, _r| T::one() / (T::one() - x * x));
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
