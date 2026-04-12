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
//! 2. Wrap your input scalars in [`AReal::new`] and register them with
//!    [`AReal::register_input`]; this hands out tape slots for the inputs
//!    without emitting any statements (they only accumulate adjoints, never
//!    appear as an LHS in the trace).
//! 3. Run your computation normally — every `+`, `-`, `*`, `/`, and
//!    `crate::math::ad::*` call pushes a new statement onto the active tape
//!    via one of [`Tape::push_binary`] / [`Tape::push_unary`].
//! 4. Register outputs with [`AReal::register_output`] (a no-op for values
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
//! thread-local tape pointer, allocates a fresh slot, and pushes the
//! operand(s) via the fixed-arity `Tape::push_*` fast paths. No
//! intermediate `Vec` or slice is constructed per op; the whole tape-
//! recording cost for a single binary op is a thread-local load, a
//! derivatives-buffer `push`, and two `operations`-buffer pushes.
//!
//! # `NamedTape` + `NamedAReal` — string-keyed reverse-mode AD
//!
//! The named reverse-mode hero type. `NamedTape` owns a `Tape<f64>` plus
//! a per-instance name -> slot map; the user calls `input()` for each named
//! input, then `freeze()` to lock the registry and activate the tape, then
//! runs the forward closure normally and reads the gradient back by name via
//! `gradient()`.
//!
//! ## The two-phase contract
//!
//! 1. **Setup phase** (`new()` -> `input()` calls): registers each named
//!    input on the inner `Tape<f64>` via `AReal::register_input`, which does
//!    NOT require an active tape — slots are assigned eagerly through the
//!    `&mut Tape` reference. The tape stays inactive throughout the setup
//!    phase, so you can construct multiple `NamedTape`s on one thread
//!    sequentially without conflict (just not concurrently).
//! 2. **Forward + readback phase** (`freeze()` onward): `freeze()` builds the
//!    final `Arc<VarRegistry>` and calls `Tape::activate`. From this point
//!    forward, all arithmetic on the `NamedAReal` handles is recorded on
//!    this tape. After the forward closure produces its output, call
//!    `gradient(&output)` to read the per-name adjoints as an
//!    `IndexMap<String, f64>` in registry insertion order.
//!
//! ## Thread-local discipline (`!Send`)
//!
//! `Tape<f64>` uses a thread-local active-tape pointer (see `src/tape.rs`).
//! `NamedTape` is structurally `!Send` via `PhantomData<*const ()>` so the
//! compiler refuses to let you move a `NamedTape` across threads — doing
//! so would corrupt the TLS contract on either side. Two `NamedTape`s on
//! two threads work fine (each thread has its own TLS pointer); two
//! `NamedTape`s on **one** thread cannot both be `freeze()`d at the same
//! time (the second `freeze()` panics with `"A tape is already active on
//! this thread"`).
//!
//! ## `std::mem::forget` and panic-during-forward hazards
//!
//! - `std::mem::forget(named_tape)` skips `Drop`, leaving the TLS pointer
//!   dangling. **Recovery:** call [`NamedTape::deactivate_all`] before
//!   constructing the next `NamedTape`.
//! - A panic inside the user's forward closure unwinds normally and runs
//!   `NamedTape::Drop`, which deactivates the tape — panic safety is
//!   preserved unless the panic itself happens inside another `Drop` (the
//!   standard double-panic abort rule).
//!
//! ## `!Send` compile-fail assertion
//!
//! ```compile_fail,E0277
//! use xad_rs::reverse::areal::NamedTape;
//! fn assert_send<T: Send>(_: T) {}
//! assert_send(NamedTape::new());
//! ```

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::Arc;

use indexmap::{IndexMap, IndexSet};

use crate::math;
use crate::registry::VarRegistry;
use crate::tape::{Tape, TapeStorage};

// ===========================================================================
// AReal<T> — positional reverse-mode active real
// ===========================================================================

/// Active real number for reverse-mode AD.
///
/// Wraps a scalar value and a tape slot. When a tape is active, all operations
/// involving `AReal` values are recorded for later adjoint computation.
#[derive(Clone)]
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

    /// Register this variable as an input on the tape.
    ///
    /// Input variables are given a fresh slot but **no** tape statement is
    /// emitted for them — inputs never appear on the LHS of any operation,
    /// so there is nothing to record during the reverse sweep. Their
    /// derivatives accumulate in-place in the tape's derivative buffer and
    /// are read back via [`AReal::adjoint`] after `compute_adjoints`.
    pub fn register_input(vars: &mut [AReal<T>], tape: &mut Tape<T>) {
        for v in vars.iter_mut() {
            if !v.should_record() {
                v.slot = tape.register_variable();
            }
        }
    }

    /// Register this variable as an output on the tape.
    ///
    /// In practice the final result of a forward pass is already on the
    /// tape (it was created by the last binary/unary op), so this is a
    /// no-op for most call sites. For constant outputs that never touched
    /// the tape, it allocates a fresh slot — again without emitting a
    /// statement, for the same reason as [`AReal::register_input`].
    pub fn register_output(vars: &mut [AReal<T>], tape: &mut Tape<T>) {
        for v in vars.iter_mut() {
            if !v.should_record() {
                v.slot = tape.register_variable();
            }
        }
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
    let tape_ptr = Tape::<T>::get_active();
    if let Some(ptr) = tape_ptr {
        // SAFETY: `ptr` comes from the thread-local active-tape slot, which
        // is only set via `Tape::activate` against a live `&mut Tape` and
        // cleared on drop; the tape cannot be concurrently aliased because
        // the tape pointer is thread-local.
        let tape = unsafe { &mut *ptr };
        let slot = tape.register_variable();
        tape.push_binary(slot, lhs_mul, lhs_slot, rhs_mul, rhs_slot);
        AReal { value: result_value, slot }
    } else {
        AReal::new(result_value)
    }
}

#[inline]
fn record_unary<T: TapeStorage>(result_value: T, input_slot: u32, multiplier: T) -> AReal<T> {
    let tape_ptr = Tape::<T>::get_active();
    if let Some(ptr) = tape_ptr {
        // SAFETY: see `record_binary`.
        let tape = unsafe { &mut *ptr };
        let slot = tape.register_variable();
        tape.push_unary(slot, multiplier, input_slot);
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

impl From<i32> for AReal<f32> {
    fn from(value: i32) -> Self {
        AReal::new(value as f32)
    }
}

// --- Operator implementations ---

// AReal + AReal
impl<T: TapeStorage> Add for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn add(self, rhs: AReal<T>) -> AReal<T> {
        record_binary(
            self.value + rhs.value,
            self.slot,
            T::one(),
            rhs.slot,
            T::one(),
        )
    }
}

impl<T: TapeStorage> Add for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn add(self, rhs: &AReal<T>) -> AReal<T> {
        record_binary(
            self.value + rhs.value,
            self.slot,
            T::one(),
            rhs.slot,
            T::one(),
        )
    }
}

impl<T: TapeStorage> Add<&AReal<T>> for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn add(self, rhs: &AReal<T>) -> AReal<T> {
        record_binary(
            self.value + rhs.value,
            self.slot,
            T::one(),
            rhs.slot,
            T::one(),
        )
    }
}

impl<T: TapeStorage> Add<AReal<T>> for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn add(self, rhs: AReal<T>) -> AReal<T> {
        record_binary(
            self.value + rhs.value,
            self.slot,
            T::one(),
            rhs.slot,
            T::one(),
        )
    }
}

// AReal + scalar
impl<T: TapeStorage> Add<T> for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn add(self, rhs: T) -> AReal<T> {
        record_unary(self.value + rhs, self.slot, T::one())
    }
}

impl<T: TapeStorage> Add<T> for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn add(self, rhs: T) -> AReal<T> {
        record_unary(self.value + rhs, self.slot, T::one())
    }
}

// scalar + AReal (only for f64 and f32 to avoid orphan rules)
impl Add<AReal<f64>> for f64 {
    type Output = AReal<f64>;
    #[inline]
    fn add(self, rhs: AReal<f64>) -> AReal<f64> {
        record_unary(self + rhs.value, rhs.slot, 1.0)
    }
}

impl Add<&AReal<f64>> for f64 {
    type Output = AReal<f64>;
    #[inline]
    fn add(self, rhs: &AReal<f64>) -> AReal<f64> {
        record_unary(self + rhs.value, rhs.slot, 1.0)
    }
}

impl Add<AReal<f32>> for f32 {
    type Output = AReal<f32>;
    #[inline]
    fn add(self, rhs: AReal<f32>) -> AReal<f32> {
        record_unary(self + rhs.value, rhs.slot, 1.0)
    }
}

impl Add<&AReal<f32>> for f32 {
    type Output = AReal<f32>;
    #[inline]
    fn add(self, rhs: &AReal<f32>) -> AReal<f32> {
        record_unary(self + rhs.value, rhs.slot, 1.0)
    }
}

// AReal - AReal
impl<T: TapeStorage> Sub for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn sub(self, rhs: AReal<T>) -> AReal<T> {
        record_binary(
            self.value - rhs.value,
            self.slot,
            T::one(),
            rhs.slot,
            -T::one(),
        )
    }
}

impl<T: TapeStorage> Sub for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn sub(self, rhs: &AReal<T>) -> AReal<T> {
        record_binary(
            self.value - rhs.value,
            self.slot,
            T::one(),
            rhs.slot,
            -T::one(),
        )
    }
}

impl<T: TapeStorage> Sub<&AReal<T>> for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn sub(self, rhs: &AReal<T>) -> AReal<T> {
        record_binary(
            self.value - rhs.value,
            self.slot,
            T::one(),
            rhs.slot,
            -T::one(),
        )
    }
}

impl<T: TapeStorage> Sub<AReal<T>> for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn sub(self, rhs: AReal<T>) -> AReal<T> {
        record_binary(
            self.value - rhs.value,
            self.slot,
            T::one(),
            rhs.slot,
            -T::one(),
        )
    }
}

// AReal - scalar
impl<T: TapeStorage> Sub<T> for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn sub(self, rhs: T) -> AReal<T> {
        record_unary(self.value - rhs, self.slot, T::one())
    }
}

impl<T: TapeStorage> Sub<T> for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn sub(self, rhs: T) -> AReal<T> {
        record_unary(self.value - rhs, self.slot, T::one())
    }
}

// scalar - AReal
impl Sub<AReal<f64>> for f64 {
    type Output = AReal<f64>;
    #[inline]
    fn sub(self, rhs: AReal<f64>) -> AReal<f64> {
        record_unary(self - rhs.value, rhs.slot, -1.0)
    }
}

impl Sub<&AReal<f64>> for f64 {
    type Output = AReal<f64>;
    #[inline]
    fn sub(self, rhs: &AReal<f64>) -> AReal<f64> {
        record_unary(self - rhs.value, rhs.slot, -1.0)
    }
}

impl Sub<AReal<f32>> for f32 {
    type Output = AReal<f32>;
    #[inline]
    fn sub(self, rhs: AReal<f32>) -> AReal<f32> {
        record_unary(self - rhs.value, rhs.slot, -1.0)
    }
}

impl Sub<&AReal<f32>> for f32 {
    type Output = AReal<f32>;
    #[inline]
    fn sub(self, rhs: &AReal<f32>) -> AReal<f32> {
        record_unary(self - rhs.value, rhs.slot, -1.0)
    }
}

// AReal * AReal
impl<T: TapeStorage> Mul for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn mul(self, rhs: AReal<T>) -> AReal<T> {
        // d(a*b) = b*da + a*db
        record_binary(
            self.value * rhs.value,
            self.slot,
            rhs.value,
            rhs.slot,
            self.value,
        )
    }
}

impl<T: TapeStorage> Mul for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn mul(self, rhs: &AReal<T>) -> AReal<T> {
        record_binary(
            self.value * rhs.value,
            self.slot,
            rhs.value,
            rhs.slot,
            self.value,
        )
    }
}

impl<T: TapeStorage> Mul<&AReal<T>> for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn mul(self, rhs: &AReal<T>) -> AReal<T> {
        record_binary(
            self.value * rhs.value,
            self.slot,
            rhs.value,
            rhs.slot,
            self.value,
        )
    }
}

impl<T: TapeStorage> Mul<AReal<T>> for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn mul(self, rhs: AReal<T>) -> AReal<T> {
        record_binary(
            self.value * rhs.value,
            self.slot,
            rhs.value,
            rhs.slot,
            self.value,
        )
    }
}

// AReal * scalar
impl<T: TapeStorage> Mul<T> for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn mul(self, rhs: T) -> AReal<T> {
        record_unary(self.value * rhs, self.slot, rhs)
    }
}

impl<T: TapeStorage> Mul<T> for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn mul(self, rhs: T) -> AReal<T> {
        record_unary(self.value * rhs, self.slot, rhs)
    }
}

// scalar * AReal
impl Mul<AReal<f64>> for f64 {
    type Output = AReal<f64>;
    #[inline]
    fn mul(self, rhs: AReal<f64>) -> AReal<f64> {
        record_unary(self * rhs.value, rhs.slot, self)
    }
}

impl Mul<&AReal<f64>> for f64 {
    type Output = AReal<f64>;
    #[inline]
    fn mul(self, rhs: &AReal<f64>) -> AReal<f64> {
        record_unary(self * rhs.value, rhs.slot, self)
    }
}

impl Mul<AReal<f32>> for f32 {
    type Output = AReal<f32>;
    #[inline]
    fn mul(self, rhs: AReal<f32>) -> AReal<f32> {
        record_unary(self * rhs.value, rhs.slot, self)
    }
}

impl Mul<&AReal<f32>> for f32 {
    type Output = AReal<f32>;
    #[inline]
    fn mul(self, rhs: &AReal<f32>) -> AReal<f32> {
        record_unary(self * rhs.value, rhs.slot, self)
    }
}

// AReal / AReal
impl<T: TapeStorage> Div for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn div(self, rhs: AReal<T>) -> AReal<T> {
        // d(a/b) = da/b - a*db/b^2
        let inv_b = T::one() / rhs.value;
        record_binary(
            self.value * inv_b,
            self.slot,
            inv_b,
            rhs.slot,
            -self.value * inv_b * inv_b,
        )
    }
}

impl<T: TapeStorage> Div for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn div(self, rhs: &AReal<T>) -> AReal<T> {
        let inv_b = T::one() / rhs.value;
        record_binary(
            self.value * inv_b,
            self.slot,
            inv_b,
            rhs.slot,
            -self.value * inv_b * inv_b,
        )
    }
}

impl<T: TapeStorage> Div<&AReal<T>> for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn div(self, rhs: &AReal<T>) -> AReal<T> {
        let inv_b = T::one() / rhs.value;
        record_binary(
            self.value * inv_b,
            self.slot,
            inv_b,
            rhs.slot,
            -self.value * inv_b * inv_b,
        )
    }
}

impl<T: TapeStorage> Div<AReal<T>> for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn div(self, rhs: AReal<T>) -> AReal<T> {
        let inv_b = T::one() / rhs.value;
        record_binary(
            self.value * inv_b,
            self.slot,
            inv_b,
            rhs.slot,
            -self.value * inv_b * inv_b,
        )
    }
}

// AReal / scalar
impl<T: TapeStorage> Div<T> for AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn div(self, rhs: T) -> AReal<T> {
        let inv = T::one() / rhs;
        record_unary(self.value * inv, self.slot, inv)
    }
}

impl<T: TapeStorage> Div<T> for &AReal<T> {
    type Output = AReal<T>;
    #[inline]
    fn div(self, rhs: T) -> AReal<T> {
        let inv = T::one() / rhs;
        record_unary(self.value * inv, self.slot, inv)
    }
}

// scalar / AReal
impl Div<AReal<f64>> for f64 {
    type Output = AReal<f64>;
    #[inline]
    fn div(self, rhs: AReal<f64>) -> AReal<f64> {
        // d(a/x) = -a/x^2 * dx
        let inv = 1.0 / rhs.value;
        record_unary(self * inv, rhs.slot, -self * inv * inv)
    }
}

impl Div<&AReal<f64>> for f64 {
    type Output = AReal<f64>;
    #[inline]
    fn div(self, rhs: &AReal<f64>) -> AReal<f64> {
        let inv = 1.0 / rhs.value;
        record_unary(self * inv, rhs.slot, -self * inv * inv)
    }
}

impl Div<AReal<f32>> for f32 {
    type Output = AReal<f32>;
    #[inline]
    fn div(self, rhs: AReal<f32>) -> AReal<f32> {
        let inv = 1.0 / rhs.value;
        record_unary(self * inv, rhs.slot, -self * inv * inv)
    }
}

impl Div<&AReal<f32>> for f32 {
    type Output = AReal<f32>;
    #[inline]
    fn div(self, rhs: &AReal<f32>) -> AReal<f32> {
        let inv = 1.0 / rhs.value;
        record_unary(self * inv, rhs.slot, -self * inv * inv)
    }
}

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
        *self = self.clone() + rhs;
    }
}

impl<T: TapeStorage> AddAssign<&AReal<T>> for AReal<T> {
    fn add_assign(&mut self, rhs: &AReal<T>) {
        *self = self.clone() + rhs;
    }
}

impl<T: TapeStorage> AddAssign<T> for AReal<T> {
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs;
    }
}

impl<T: TapeStorage> SubAssign for AReal<T> {
    fn sub_assign(&mut self, rhs: AReal<T>) {
        *self = self.clone() - rhs;
    }
}

impl<T: TapeStorage> SubAssign<&AReal<T>> for AReal<T> {
    fn sub_assign(&mut self, rhs: &AReal<T>) {
        *self = self.clone() - rhs;
    }
}

impl<T: TapeStorage> SubAssign<T> for AReal<T> {
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs;
    }
}

impl<T: TapeStorage> MulAssign for AReal<T> {
    fn mul_assign(&mut self, rhs: AReal<T>) {
        *self = self.clone() * rhs;
    }
}

impl<T: TapeStorage> MulAssign<&AReal<T>> for AReal<T> {
    fn mul_assign(&mut self, rhs: &AReal<T>) {
        *self = self.clone() * rhs;
    }
}

impl<T: TapeStorage> MulAssign<T> for AReal<T> {
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs;
    }
}

impl<T: TapeStorage> DivAssign for AReal<T> {
    fn div_assign(&mut self, rhs: AReal<T>) {
        *self = self.clone() / rhs;
    }
}

impl<T: TapeStorage> DivAssign<&AReal<T>> for AReal<T> {
    fn div_assign(&mut self, rhs: &AReal<T>) {
        *self = self.clone() / rhs;
    }
}

impl<T: TapeStorage> DivAssign<T> for AReal<T> {
    fn div_assign(&mut self, rhs: T) {
        *self = self.clone() / rhs;
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

// ===========================================================================
// NamedAReal — string-keyed wrapper around AReal<f64>
// ===========================================================================

/// Labeled wrapper around a positional [`AReal<f64>`].
///
/// **Shape A (minimal):** does NOT carry an `Arc<VarRegistry>` field. The
/// only way to construct a `NamedAReal` is via [`NamedTape::input`], so
/// every `NamedAReal` is structurally tied to exactly one tape via the
/// thread-local active pointer; cross-tape mixing is structurally
/// impossible (the second `NamedTape::freeze()` on one thread panics).
/// Skipping the `Arc` field saves one atomic increment per operator and is
/// the gate-binding choice for the named reverse-mode bench in Phase 2.
#[derive(Clone)]
pub struct NamedAReal {
    inner: AReal<f64>,
}

impl NamedAReal {
    /// Internal constructor used by `NamedTape::input` and the operator
    /// impls. Not part of the public API.
    #[inline]
    pub(crate) fn from_inner(inner: AReal<f64>) -> Self {
        Self { inner }
    }

    /// Underlying value (no derivative information).
    #[inline]
    pub fn value(&self) -> f64 {
        self.inner.value()
    }

    /// Escape hatch: read-only access to the inner `AReal<f64>`.
    #[inline]
    pub fn inner(&self) -> &AReal<f64> {
        &self.inner
    }

    // ============ Elementary math delegations (mirrors Phase 1 NamedFReal) ============
    #[inline]
    pub fn sin(&self) -> Self {
        Self {
            inner: math::ad::sin(&self.inner),
        }
    }
    #[inline]
    pub fn cos(&self) -> Self {
        Self {
            inner: math::ad::cos(&self.inner),
        }
    }
    #[inline]
    pub fn tan(&self) -> Self {
        Self {
            inner: math::ad::tan(&self.inner),
        }
    }
    #[inline]
    pub fn exp(&self) -> Self {
        Self {
            inner: math::ad::exp(&self.inner),
        }
    }
    #[inline]
    pub fn ln(&self) -> Self {
        Self {
            inner: math::ad::ln(&self.inner),
        }
    }
    #[inline]
    pub fn sqrt(&self) -> Self {
        Self {
            inner: math::ad::sqrt(&self.inner),
        }
    }
    #[inline]
    pub fn tanh(&self) -> Self {
        Self {
            inner: math::ad::tanh(&self.inner),
        }
    }
    #[inline]
    pub fn norm_cdf(&self) -> Self {
        Self {
            inner: math::ad::norm_cdf(&self.inner),
        }
    }
    #[inline]
    pub fn inv_norm_cdf(&self) -> Self {
        Self {
            inner: math::ad::inv_norm_cdf(&self.inner),
        }
    }
}

impl fmt::Debug for NamedAReal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NamedAReal")
            .field("value", &self.inner.value())
            .field("slot", &self.inner.slot())
            .finish()
    }
}

// ============ Operator overloads — hand-written, Shape A ============
// No shared op-stamping macro is used: Shape A does not carry a
// `registry: Arc<VarRegistry>` field, and the historical LBLF-07
// stamping scaffold has been deleted in Plan 02.2-02. The four
// reference variants
// (owned/owned, ref/ref, owned/ref, ref/owned) plus scalar variants are
// stamped explicitly below. The inner `AReal` operators read TLS via
// `record_binary` / `record_unary`, so the named wrapper is a pure
// pass-through with zero atomic increments.

macro_rules! __named_areal_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl ::core::ops::$trait<NamedAReal> for NamedAReal {
            type Output = NamedAReal;
            #[inline]
            fn $method(self, rhs: NamedAReal) -> NamedAReal {
                NamedAReal { inner: self.inner $op rhs.inner }
            }
        }
        impl ::core::ops::$trait<&NamedAReal> for &NamedAReal {
            type Output = NamedAReal;
            #[inline]
            fn $method(self, rhs: &NamedAReal) -> NamedAReal {
                NamedAReal { inner: &self.inner $op &rhs.inner }
            }
        }
        impl ::core::ops::$trait<&NamedAReal> for NamedAReal {
            type Output = NamedAReal;
            #[inline]
            fn $method(self, rhs: &NamedAReal) -> NamedAReal {
                NamedAReal { inner: self.inner $op &rhs.inner }
            }
        }
        impl ::core::ops::$trait<NamedAReal> for &NamedAReal {
            type Output = NamedAReal;
            #[inline]
            fn $method(self, rhs: NamedAReal) -> NamedAReal {
                NamedAReal { inner: &self.inner $op rhs.inner }
            }
        }
        impl ::core::ops::$trait<f64> for NamedAReal {
            type Output = NamedAReal;
            #[inline]
            fn $method(self, rhs: f64) -> NamedAReal {
                NamedAReal { inner: self.inner $op rhs }
            }
        }
        impl ::core::ops::$trait<f64> for &NamedAReal {
            type Output = NamedAReal;
            #[inline]
            fn $method(self, rhs: f64) -> NamedAReal {
                NamedAReal { inner: &self.inner $op rhs }
            }
        }
    };
}

__named_areal_binop!(Add, add, +);
__named_areal_binop!(Sub, sub, -);
__named_areal_binop!(Mul, mul, *);
__named_areal_binop!(Div, div, /);

impl ::core::ops::Neg for NamedAReal {
    type Output = NamedAReal;
    #[inline]
    fn neg(self) -> NamedAReal {
        NamedAReal { inner: -self.inner }
    }
}
impl ::core::ops::Neg for &NamedAReal {
    type Output = NamedAReal;
    #[inline]
    fn neg(self) -> NamedAReal {
        NamedAReal {
            inner: -&self.inner,
        }
    }
}

// ===========================================================================
// NamedTape — string-keyed reverse-mode tape
// ===========================================================================

/// Labeled reverse-mode tape. Owns a `Tape<f64>`, a name -> slot map, and
/// (after `freeze()`) an `Arc<VarRegistry>`.
///
/// Two-phase usage: see the [module-level docs](self).
///
/// `!Send` is enforced structurally via [`PhantomData<*const ()>`]; see the
/// `compile_fail` doctest in the module-level docs.
pub struct NamedTape {
    tape: Tape<f64>,
    builder: IndexSet<String>,
    inputs: Vec<(String, u32)>,
    registry: Option<Arc<VarRegistry>>,
    frozen: bool,
    _not_send: PhantomData<*const ()>,
}

impl NamedTape {
    /// Construct a new named tape. Does NOT activate the inner `Tape<f64>`
    /// — call [`freeze`](Self::freeze) to lock the registry and activate.
    pub fn new() -> Self {
        Self {
            tape: Tape::<f64>::new(true),
            builder: IndexSet::new(),
            inputs: Vec::new(),
            registry: None,
            frozen: false,
            _not_send: PhantomData,
        }
    }

    /// Register a named input and return a [`NamedAReal`] handle.
    ///
    /// Eagerly assigns a tape slot via `AReal::register_input(&mut [v], &mut self.tape)`
    /// — `register_input` does NOT require the tape to be active, so this
    /// works during the setup phase before [`freeze`](Self::freeze) runs.
    ///
    /// Panics if called after [`freeze`](Self::freeze).
    pub fn input(&mut self, name: &str, value: f64) -> NamedAReal {
        assert!(
            !self.frozen,
            "NamedTape::input({:?}) called after freeze(); add all inputs before running the forward pass",
            name
        );
        // Idempotent insertion: first wins (matches VarRegistry::from_names semantics).
        if !self.builder.contains(name) {
            self.builder.insert(name.to_string());
        }
        let mut ar = AReal::<f64>::new(value);
        AReal::register_input(std::slice::from_mut(&mut ar), &mut self.tape);
        self.inputs.push((name.to_string(), ar.slot()));
        NamedAReal::from_inner(ar)
    }

    /// Lock the registry, activate the tape, and return the shared
    /// `Arc<VarRegistry>`. Panics if already frozen, or if another tape is
    /// already active on this thread (panic message: `"A tape is already
    /// active on this thread"`).
    pub fn freeze(&mut self) -> Arc<VarRegistry> {
        assert!(
            !self.frozen,
            "NamedTape::freeze() called twice on the same tape"
        );
        let reg = Arc::new(VarRegistry::from_names(self.builder.iter().cloned()));
        self.registry = Some(Arc::clone(&reg));
        self.tape.activate();
        self.frozen = true;
        reg
    }

    /// True if [`freeze`](Self::freeze) has been called.
    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Access the frozen registry, if any. Returns `None` until
    /// [`freeze`](Self::freeze) has been called.
    #[inline]
    pub fn registry(&self) -> Option<&Arc<VarRegistry>> {
        self.registry.as_ref()
    }

    /// Compute the gradient of `output` with respect to every registered
    /// input, returning an `IndexMap<String, f64>` in registry insertion
    /// order (i.e. the order of the [`input`](Self::input) calls).
    ///
    /// Panics if called before [`freeze`](Self::freeze).
    pub fn gradient(&mut self, output: &NamedAReal) -> IndexMap<String, f64> {
        assert!(
            self.frozen,
            "NamedTape::gradient() called before freeze()"
        );
        self.tape.clear_derivatives();
        output.inner.set_adjoint(&mut self.tape, 1.0);
        self.tape.compute_adjoints();
        let mut grad = IndexMap::with_capacity(self.inputs.len());
        for (name, slot) in &self.inputs {
            grad.insert(name.clone(), self.tape.derivative(*slot));
        }
        grad
    }

    /// Static escape hatch for the `std::mem::forget` recovery path.
    ///
    /// Wraps `Tape::<f64>::deactivate_all()`. Call this before constructing
    /// a new `NamedTape` if a previous `NamedTape` was leaked via
    /// `std::mem::forget` (which would otherwise leave the thread-local
    /// pointer dangling).
    pub fn deactivate_all() {
        Tape::<f64>::deactivate_all();
    }
}

impl Default for NamedTape {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for NamedTape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NamedTape")
            .field("frozen", &self.frozen)
            .field("inputs", &self.inputs.len())
            .field("registry_len", &self.registry.as_ref().map(|r| r.len()))
            .finish()
    }
}

impl Drop for NamedTape {
    fn drop(&mut self) {
        // Belt-and-suspenders: explicitly deactivate. The inner `Tape::Drop`
        // also calls `deactivate()`, but doing it here makes the lifecycle
        // contract obvious in the source.
        if self.frozen {
            self.tape.deactivate();
        }
    }
}
