//! `AReal` — active real number for reverse-mode (adjoint) AD.
//!
//! An [`AReal<T>`] wraps a scalar value and, when a [`Tape`] is active on
//! the current thread, records every arithmetic and transcendental
//! operation onto that tape. After the forward pass completes, calling
//! [`Tape::compute_adjoints`](crate::tape::Tape::compute_adjoints) walks
//! the recorded trace in reverse and accumulates partial derivatives.
//!
//! # Workflow
//!
//! 1. Create a [`Tape`] and call [`Tape::activate`] to install it as the
//!    thread-local active tape.
//! 2. Create registered inputs with [`AReal::input`] (or wrap existing
//!    values via [`AReal::new`] + [`AReal::register_input`] for slices);
//!    this hands out tape slots for the inputs, each recorded as a
//!    zero-operand statement (slots correspond 1:1 with statements; the
//!    reverse sweep skips them, since inputs only accumulate adjoints).
//! 3. Run your computation normally — every `+`, `-`, `*`, `/`, and
//!    `crate::math::ad::*` call pushes a new statement onto the active tape
//!    via one of [`Tape::push_binary`] / [`Tape::push_unary`].
//! 4. Register outputs with [`AReal::register`] (or
//!    [`AReal::register_output`] for slices; a no-op for values
//!    already on the tape — the common case — and a fresh slot for
//!    constant outputs), seed the starting adjoint via
//!    [`AReal::set_adjoint`], then call
//!    [`Tape::compute_adjoints`](crate::tape::Tape::compute_adjoints).
//! 5. Read back each input's gradient via [`AReal::adjoint`].
//! 6. Call [`Tape::deactivate_all`](crate::tape::Tape::deactivate_all) (or
//!    simply drop the `Tape`) when you're done.
//!
//! # Hot-path design
//!
//! Every binary operator on `AReal` — `+`, `-`, `*`, `/`, and all
//! transcendentals from [`crate::math::ad`] — dispatches through a small,
//! allocation-free helper (`record_binary` / `record_unary`) that reads the
//! thread-local tape pointer and pushes the operand(s) via the fixed-arity
//! `Tape::push_*` fast paths, which hand back the fresh LHS slot. No
//! intermediate `Vec` or slice is constructed per op; the whole tape-
//! recording cost for a single binary op is a thread-local load, two
//! `operations`-buffer pushes, and one 4-byte statement push (the
//! derivatives buffer is materialized lazily at sweep time, not during
//! recording).

use std::fmt;
use std::ops::{AddAssign, DivAssign, MulAssign, Neg, SubAssign};


use crate::tape::{Tape, TapeStorage};

// ===========================================================================
// AReal<T> — positional reverse-mode active real
// ===========================================================================

/// Active real number for reverse-mode AD.
///
/// Wraps a scalar value and a tape slot. When a tape is active, all operations
/// involving `AReal` values are recorded for later adjoint computation.
///
/// `AReal` is `Copy` (a value plus a `u32` slot, same shape as
/// [`Jet1`](crate::forward::jet1::Jet1)): copying it duplicates the
/// *reference* to the recorded variable, not the variable itself — both
/// copies share one slot and therefore one adjoint.
#[derive(Clone, Copy)]
pub struct AReal<T: TapeStorage> {
    value: T,
    pub(crate) slot: u32,
}

const INVALID_SLOT: u32 = u32::MAX;

impl<T: TapeStorage> AReal<T> {
    /// Create a new AReal with the given value. Not registered on any tape.
    pub fn new(value: T) -> Self {
        AReal {
            value,
            slot: INVALID_SLOT,
        }
    }

    /// Get the underlying value.
    #[inline]
    pub fn value(&self) -> T {
        self.value
    }

    /// Set the underlying value.
    #[inline]
    pub fn set_value(&mut self, v: T) {
        self.value = v;
    }

    /// Get the tape slot (derivative index).
    #[inline]
    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// Whether this variable is registered on the tape.
    #[inline]
    pub fn should_record(&self) -> bool {
        self.slot != INVALID_SLOT
    }

    /// Get the adjoint (derivative) from the given tape.
    #[inline]
    pub fn adjoint(&self, tape: &Tape<T>) -> T {
        if self.slot == INVALID_SLOT {
            T::zero()
        } else {
            tape.derivative(self.slot)
        }
    }

    /// Set the adjoint (derivative) on the given tape.
    #[inline]
    pub fn set_adjoint(&self, tape: &mut Tape<T>, value: T) {
        if self.slot != INVALID_SLOT {
            tape.set_derivative(self.slot, value);
        }
    }

    /// Create an `AReal` from `value` and immediately register it as an
    /// input on `tape`. One-step replacement for the
    /// [`new`](AReal::new) + [`register_input`](AReal::register_input)
    /// pair on a single variable:
    ///
    /// ```
    /// use xad_rs::{AReal, Tape};
    /// let mut tape = Tape::<f64>::new(true);
    /// let _rec = tape.record();
    /// let x = AReal::input(3.0, &mut tape);
    /// assert!(x.should_record());
    /// ```
    #[inline]
    pub fn input(value: T, tape: &mut Tape<T>) -> AReal<T> {
        AReal {
            value,
            slot: tape.register_variable(),
        }
    }

    /// Register this variable on the tape if it isn't registered yet.
    ///
    /// Hands out a fresh slot, recorded as a zero-operand statement (slots
    /// correspond 1:1 with statements) — registered variables only
    /// accumulate adjoints during the reverse sweep, which skips their
    /// empty operand range. Single-value counterpart of
    /// [`register_input`](AReal::register_input) /
    /// [`register_output`](AReal::register_output).
    #[inline]
    pub fn register(&mut self, tape: &mut Tape<T>) {
        if !self.should_record() {
            self.slot = tape.register_variable();
        }
    }

    /// Register these variables as inputs on the tape.
    ///
    /// Input variables are given a fresh slot, recorded as a zero-operand
    /// statement that the reverse sweep skips — inputs have no operands to
    /// propagate to. Their derivatives accumulate in-place in the tape's
    /// derivative buffer and are read back via [`AReal::adjoint`] after
    /// `compute_adjoints`.
    pub fn register_input(vars: &mut [AReal<T>], tape: &mut Tape<T>) {
        for v in vars.iter_mut() {
            v.register(tape);
        }
    }

    /// Register these variables as outputs on the tape.
    ///
    /// In practice the final result of a forward pass is already on the
    /// tape (it was created by the last binary/unary op), so this is a
    /// no-op for most call sites. For constant outputs that never touched
    /// the tape, it allocates a fresh slot — a zero-operand statement, as
    /// in [`AReal::register_input`].
    pub fn register_output(vars: &mut [AReal<T>], tape: &mut Tape<T>) {
        Self::register_input(vars, tape);
    }
}

// Helper to record a binary operation on the active tape.
//
// Hot path: called from every AReal `+`, `-`, `*`, `/`. Uses the fixed-arity
// `Tape::push_binary` fast path so there is no `Vec` allocation and no
// intermediate slice construction per operation.
#[inline]
fn record_binary<T: TapeStorage>(
    result_value: T,
    lhs_slot: u32,
    lhs_mul: T,
    rhs_slot: u32,
    rhs_mul: T,
) -> AReal<T> {
    // Both operands passive: no derivative can flow through this op, so
    // recording it would only waste a slot and a dead nullary statement
    // (push_binary filters u32::MAX operands). Skipping also elides the
    // thread-local tape load entirely.
    if lhs_slot == INVALID_SLOT && rhs_slot == INVALID_SLOT {
        return AReal::new(result_value);
    }
    let tape_ptr = Tape::<T>::get_active();
    if let Some(ptr) = tape_ptr {
        // SAFETY: `ptr` comes from the thread-local active-tape slot, which
        // is only set via `Tape::activate` against a live `&mut Tape` and
        // cleared on drop; the tape cannot be concurrently aliased because
        // the tape pointer is thread-local.
        let tape = unsafe { &mut *ptr };
        let slot = tape.push_binary(lhs_mul, lhs_slot, rhs_mul, rhs_slot);
        AReal { value: result_value, slot }
    } else {
        AReal::new(result_value)
    }
}

#[inline]
fn record_unary<T: TapeStorage>(result_value: T, input_slot: u32, multiplier: T) -> AReal<T> {
    // Passive operand: same rationale as `record_binary` — the result is a
    // plain constant, so don't burn a slot + dead statement on it.
    if input_slot == INVALID_SLOT {
        return AReal::new(result_value);
    }
    let tape_ptr = Tape::<T>::get_active();
    if let Some(ptr) = tape_ptr {
        // SAFETY: see `record_binary`.
        let tape = unsafe { &mut *ptr };
        let slot = tape.push_unary(multiplier, input_slot);
        AReal { value: result_value, slot }
    } else {
        AReal::new(result_value)
    }
}

pub(crate) fn record_unary_op<T: TapeStorage>(
    result_value: T,
    input_slot: u32,
    multiplier: T,
) -> AReal<T> {
    record_unary(result_value, input_slot, multiplier)
}

pub(crate) fn record_binary_op<T: TapeStorage>(
    result_value: T,
    a_slot: u32,
    a_mul: T,
    b_slot: u32,
    b_mul: T,
) -> AReal<T> {
    record_binary(result_value, a_slot, a_mul, b_slot, b_mul)
}

// Record an n-ary operation as a SINGLE tape statement. `n_max` is an upper
// bound on the operand count, letting the tape reserve once and skip
// per-operand grow checks (the `push_binary` trick generalized to n
// operands); inactive slots (u32::MAX) are skipped. Backs the fused
// recorders in `math::ad` (`sum`, `dot`, `weighted_sum`, `weighted_dot`),
// which replace a chain of binary temporaries — and the statements,
// operands, and derivative slots that chain would cost — with one statement.
pub(crate) fn record_nary_op_bounded<T: TapeStorage>(
    result_value: T,
    n_max: usize,
    operands: impl IntoIterator<Item = (T, u32)>,
) -> AReal<T> {
    if let Some(ptr) = Tape::<T>::get_active() {
        // SAFETY: see `record_binary`.
        let tape = unsafe { &mut *ptr };
        tape.push_operands_bounded(n_max, operands);
        let slot = tape.end_statement();
        AReal { value: result_value, slot }
    } else {
        AReal::new(result_value)
    }
}

// Assignment from a scalar: create an unrecorded AReal.
impl<T: TapeStorage> From<T> for AReal<T> {
    fn from(value: T) -> Self {
        AReal::new(value)
    }
}

impl From<i32> for AReal<f64> {
    fn from(value: i32) -> Self {
        AReal::new(value as f64)
    }
}

// --- Operator implementations ---
//
// Every binary `AReal` op stamps three flavours via the macros below:
//
// 1. `impl_areal_binop!`           — AReal-on-both-sides (4 ref-permutations)
// 2. `impl_areal_binop_scalar_rhs!` — `AReal<T> op T` (2 ref-permutations)
// 3. `impl_scalar_lhs_areal_binop!` — `f64 op AReal<f64>` (the orphan rule
//    forces a concrete scalar type)
//
// Bodies receive bound names for the value/slot operands and produce an
// `AReal<T>` (typically via `record_binary` / `record_unary`).

macro_rules! impl_areal_binop {
    ($trait:ident, $method:ident, ($a:ident, $b:ident, $a_slot:ident, $b_slot:ident) => $body:block) => {
        impl<T: TapeStorage> ::core::ops::$trait<AReal<T>> for AReal<T> {
            type Output = AReal<T>;
            #[inline]
            fn $method(self, rhs: AReal<T>) -> AReal<T> {
                let $a = self.value;
                let $b = rhs.value;
                let $a_slot = self.slot;
                let $b_slot = rhs.slot;
                $body
            }
        }
        impl<T: TapeStorage> ::core::ops::$trait<&AReal<T>> for &AReal<T> {
            type Output = AReal<T>;
            #[inline]
            fn $method(self, rhs: &AReal<T>) -> AReal<T> {
                let $a = self.value;
                let $b = rhs.value;
                let $a_slot = self.slot;
                let $b_slot = rhs.slot;
                $body
            }
        }
        impl<T: TapeStorage> ::core::ops::$trait<&AReal<T>> for AReal<T> {
            type Output = AReal<T>;
            #[inline]
            fn $method(self, rhs: &AReal<T>) -> AReal<T> {
                let $a = self.value;
                let $b = rhs.value;
                let $a_slot = self.slot;
                let $b_slot = rhs.slot;
                $body
            }
        }
        impl<T: TapeStorage> ::core::ops::$trait<AReal<T>> for &AReal<T> {
            type Output = AReal<T>;
            #[inline]
            fn $method(self, rhs: AReal<T>) -> AReal<T> {
                let $a = self.value;
                let $b = rhs.value;
                let $a_slot = self.slot;
                let $b_slot = rhs.slot;
                $body
            }
        }
    };
}

macro_rules! impl_areal_binop_scalar_rhs {
    ($trait:ident, $method:ident, ($a:ident, $r:ident, $a_slot:ident) => $body:block) => {
        impl<T: TapeStorage> ::core::ops::$trait<T> for AReal<T> {
            type Output = AReal<T>;
            #[inline]
            fn $method(self, rhs: T) -> AReal<T> {
                let $a = self.value;
                let $r = rhs;
                let $a_slot = self.slot;
                $body
            }
        }
        impl<T: TapeStorage> ::core::ops::$trait<T> for &AReal<T> {
            type Output = AReal<T>;
            #[inline]
            fn $method(self, rhs: T) -> AReal<T> {
                let $a = self.value;
                let $r = rhs;
                let $a_slot = self.slot;
                $body
            }
        }
    };
}

macro_rules! impl_scalar_lhs_areal_binop {
    ($trait:ident, $method:ident, ($l:ident, $r:ident, $r_slot:ident) => $body:block) => {
        impl ::core::ops::$trait<AReal<f64>> for f64 {
            type Output = AReal<f64>;
            #[inline]
            fn $method(self, rhs: AReal<f64>) -> AReal<f64> {
                let $l = self;
                let $r = rhs.value;
                let $r_slot = rhs.slot;
                $body
            }
        }
        impl ::core::ops::$trait<&AReal<f64>> for f64 {
            type Output = AReal<f64>;
            #[inline]
            fn $method(self, rhs: &AReal<f64>) -> AReal<f64> {
                let $l = self;
                let $r = rhs.value;
                let $r_slot = rhs.slot;
                $body
            }
        }
    };
}

// AReal op AReal
impl_areal_binop!(Add, add, (a, b, a_s, b_s) => {
    record_binary(a + b, a_s, T::one(), b_s, T::one())
});
impl_areal_binop!(Sub, sub, (a, b, a_s, b_s) => {
    record_binary(a - b, a_s, T::one(), b_s, -T::one())
});
impl_areal_binop!(Mul, mul, (a, b, a_s, b_s) => {
    // d(a*b) = b*da + a*db
    record_binary(a * b, a_s, b, b_s, a)
});
impl_areal_binop!(Div, div, (a, b, a_s, b_s) => {
    // d(a/b) = da/b - a*db/b^2
    let inv_b = T::one() / b;
    record_binary(a * inv_b, a_s, inv_b, b_s, -a * inv_b * inv_b)
});

// AReal op T (scalar on RHS)
impl_areal_binop_scalar_rhs!(Add, add, (a, r, a_s) => {
    record_unary(a + r, a_s, T::one())
});
impl_areal_binop_scalar_rhs!(Sub, sub, (a, r, a_s) => {
    record_unary(a - r, a_s, T::one())
});
impl_areal_binop_scalar_rhs!(Mul, mul, (a, r, a_s) => {
    record_unary(a * r, a_s, r)
});
impl_areal_binop_scalar_rhs!(Div, div, (a, r, a_s) => {
    let inv = T::one() / r;
    record_unary(a * inv, a_s, inv)
});

// scalar op AReal (the orphan rule forces a concrete scalar type)
impl_scalar_lhs_areal_binop!(Add, add, (l, r, r_s) => {
    record_unary(l + r, r_s, 1.0)
});
impl_scalar_lhs_areal_binop!(Sub, sub, (l, r, r_s) => {
    record_unary(l - r, r_s, -1.0)
});
impl_scalar_lhs_areal_binop!(Mul, mul, (l, r, r_s) => {
    record_unary(l * r, r_s, l)
});
impl_scalar_lhs_areal_binop!(Div, div, (l, r, r_s) => {
    // d(a/x) = -a/x^2 * dx
    let inv = 1.0 / r;
    record_unary(l * inv, r_s, -l * inv * inv)
});

// Negation
impl<T: TapeStorage> Neg for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn neg(self) -> AReal<T> {
        record_unary(-self.value, self.slot, -T::one())
    }
}

impl<T: TapeStorage> Neg for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn neg(self) -> AReal<T> {
        record_unary(-self.value, self.slot, -T::one())
    }
}

// Compound assignment operators
impl<T: TapeStorage> AddAssign for AReal<T> {
    fn add_assign(&mut self, rhs: AReal<T>) {
        *self = *self + rhs;
    }
}

impl<T: TapeStorage> AddAssign<&AReal<T>> for AReal<T> {
    fn add_assign(&mut self, rhs: &AReal<T>) {
        *self = *self + rhs;
    }
}

impl<T: TapeStorage> AddAssign<T> for AReal<T> {
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<T: TapeStorage> SubAssign for AReal<T> {
    fn sub_assign(&mut self, rhs: AReal<T>) {
        *self = *self - rhs;
    }
}

impl<T: TapeStorage> SubAssign<&AReal<T>> for AReal<T> {
    fn sub_assign(&mut self, rhs: &AReal<T>) {
        *self = *self - rhs;
    }
}

impl<T: TapeStorage> SubAssign<T> for AReal<T> {
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl<T: TapeStorage> MulAssign for AReal<T> {
    fn mul_assign(&mut self, rhs: AReal<T>) {
        *self = *self * rhs;
    }
}

impl<T: TapeStorage> MulAssign<&AReal<T>> for AReal<T> {
    fn mul_assign(&mut self, rhs: &AReal<T>) {
        *self = *self * rhs;
    }
}

impl<T: TapeStorage> MulAssign<T> for AReal<T> {
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: TapeStorage> DivAssign for AReal<T> {
    fn div_assign(&mut self, rhs: AReal<T>) {
        *self = *self / rhs;
    }
}

impl<T: TapeStorage> DivAssign<&AReal<T>> for AReal<T> {
    fn div_assign(&mut self, rhs: &AReal<T>) {
        *self = *self / rhs;
    }
}

impl<T: TapeStorage> DivAssign<T> for AReal<T> {
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

// Comparison operators
impl<T: TapeStorage> PartialEq for AReal<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: TapeStorage> PartialEq<T> for AReal<T> {
    fn eq(&self, other: &T) -> bool {
        self.value == *other
    }
}

impl<T: TapeStorage> PartialOrd for AReal<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: TapeStorage> PartialOrd<T> for AReal<T> {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(other)
    }
}

// Display / Debug
impl<T: TapeStorage> fmt::Display for AReal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<T: TapeStorage> fmt::Debug for AReal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AReal({}, slot={})", self.value, self.slot)
    }
}

impl<T: TapeStorage> Default for AReal<T> {
    fn default() -> Self {
        AReal::new(T::zero())
    }
}

// ============================================================================
// impl Real for AReal<f64>
// ============================================================================
//
// Delegates to `crate::math::ad::*` free functions so every call records on
// the currently-active reverse-mode tape. `powf` uses the two-AReal `pow`
// (not the AReal-and-passive `powf`) because the `Real::powf` signature
// requires `exponent: Self`.

/// Stamp one unary `Real` method body for `AReal<f64>` from a crate-wide
/// derivative-table entry: delegate to the recording free function of the
/// same name in `math::ad`, which the same table stamps. Fed the table
/// directly, so this impl gains a method the moment the table does.
macro_rules! impl_real_areal_unary {
    ($name:ident, $doc:literal, $v:expr, $d1:expr, $d2:expr) => {
        #[inline]
        fn $name(&self) -> Self {
            crate::math::ad::$name(self)
        }
    };
}

impl crate::real::Real for AReal<f64> {
    type Passive = f64;

    #[inline]
    fn value(&self) -> f64 {
        self.value
    }
    /// An **unrecorded** constant zero: a literal has no antecedent on the
    /// tape, so it pushes no statement and carries a zero adjoint.
    #[inline]
    fn zero() -> Self {
        AReal::new(0.0)
    }
    /// An **unrecorded** constant one — see [`zero`](crate::Real::zero).
    #[inline]
    fn one() -> Self {
        AReal::new(1.0)
    }
    #[inline]
    fn powf(&self, exponent: Self) -> Self {
        crate::math::ad::pow(self, &exponent)
    }
    #[inline]
    fn powi(&self, exponent: i32) -> Self {
        crate::math::ad::powi(self, exponent)
    }
    // The whole reason the aggregates are trait *methods* with per-mode
    // bodies: each of these records ONE n-ary tape statement. Delegating to
    // a binary-operator chain here would still produce the right gradient
    // while quietly multiplying the tape the reverse sweep is memory-bound
    // on — see `fused_ops.rs`, which asserts these statement counts through
    // the trait method.
    #[inline]
    fn sum(xs: &[Self]) -> Self {
        crate::math::ad::sum(xs)
    }
    #[inline]
    fn dot(xs: &[Self], ys: &[Self]) -> Self {
        crate::math::ad::dot(xs, ys)
    }
    #[inline]
    fn weighted_sum(ws: &[Self::Passive], xs: &[Self]) -> Self {
        crate::math::ad::weighted_sum(ws, xs)
    }
    #[inline]
    fn weighted_dot(ws: &[Self::Passive], xs: &[Self], ys: &[Self]) -> Self {
        crate::math::ad::weighted_dot(ws, xs, ys)
    }
    #[inline]
    fn max(&self, other: &Self) -> Self {
        crate::math::ad::max(self, other)
    }
    #[inline]
    fn min(&self, other: &Self) -> Self {
        crate::math::ad::min(self, other)
    }

    crate::elementaries::for_each_unary_elementary!(impl_real_areal_unary);
}
