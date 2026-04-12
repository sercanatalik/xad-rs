//! Tape for recording operations in reverse-mode automatic differentiation.
//!
//! The tape records a trace of all arithmetic operations performed on
//! [`AReal`](crate::areal::AReal) values, then replays them in reverse to
//! compute adjoints (gradients). One tape covers the whole forward pass of
//! a computation; at the end, [`Tape::compute_adjoints`] walks the recorded
//! trace backwards once, accumulating partial derivatives in the
//! derivatives buffer.
//!
//! # Data layout
//!
//! The tape uses a compact three-buffer layout (classic XAD-style):
//!
//! - `statements: Vec<Statement>` — one entry per recorded *variable*
//!   (LHS slot). Each `Statement` stores (a) the LHS `slot` and (b) an
//!   `op_end` pointer into the operations buffer.
//! - `operations: Vec<Operation<T>>` — a packed stream of
//!   `(multiplier, operand_slot)` pairs. A statement's operand range is
//!   `[prev_statement.op_end, self.op_end)`, which means operand lookup
//!   is O(1) and the whole tape is a single linear scan in both
//!   directions.
//! - `derivatives: Vec<T>` — indexed directly by slot number. After
//!   [`compute_adjoints`](Tape::compute_adjoints) this is the gradient
//!   vector for all registered variables.
//!
//! Slots are handed out monotonically by [`Tape::register_variable`], so
//! `derivatives.len() == num_variables` holds as an invariant at all
//! times and no per-slot bounds-checks are needed in the reverse sweep
//! hot loop.
//!
//! # Hot-path recording helpers
//!
//! The three fixed-arity push methods — [`Tape::push_nullary`],
//! [`Tape::push_unary`], [`Tape::push_binary`] — are the zero-allocation
//! fast paths used by the AReal operator impls. They push operands
//! directly onto the operations buffer without going through any
//! intermediate slice or `Vec`. The variadic [`Tape::push_statement`] is
//! retained for less hot call sites.
//!
//! # Thread-local active tape
//!
//! Exactly one `Tape<T>` can be *active* per thread per scalar type at a
//! time. The active tape is stored as a raw pointer in a thread-local
//! `Cell<*mut Tape<T>>` (null encodes "no active tape"). Making a tape
//! active is an explicit opt-in via [`Tape::activate`]; AReal operators
//! read the active pointer via [`TapeStorage::get_active_ptr`] and
//! record only if it's non-null.

use crate::scalar::Scalar;
use std::cell::Cell;
use std::ptr;

thread_local! {
    // A null pointer encodes "no active tape" — cheaper than `Option<*mut _>`
    // because `Cell::get()` is a single load with no runtime borrow check.
    static ACTIVE_TAPE_F32: Cell<*mut Tape<f32>> = const { Cell::new(ptr::null_mut()) };
    static ACTIVE_TAPE_F64: Cell<*mut Tape<f64>> = const { Cell::new(ptr::null_mut()) };
}

/// Trait for thread-local tape storage access, implemented per concrete scalar type.
pub trait TapeStorage: Scalar {
    fn get_active_ptr() -> Option<*mut Tape<Self>>;
    fn set_active_ptr(ptr: Option<*mut Tape<Self>>);
}

impl TapeStorage for f32 {
    #[inline]
    fn get_active_ptr() -> Option<*mut Tape<f32>> {
        let p = ACTIVE_TAPE_F32.with(|c| c.get());
        if p.is_null() { None } else { Some(p) }
    }
    #[inline]
    fn set_active_ptr(ptr: Option<*mut Tape<f32>>) {
        ACTIVE_TAPE_F32.with(|c| c.set(ptr.unwrap_or(std::ptr::null_mut())));
    }
}

impl TapeStorage for f64 {
    #[inline]
    fn get_active_ptr() -> Option<*mut Tape<f64>> {
        let p = ACTIVE_TAPE_F64.with(|c| c.get());
        if p.is_null() { None } else { Some(p) }
    }
    #[inline]
    fn set_active_ptr(ptr: Option<*mut Tape<f64>>) {
        ACTIVE_TAPE_F64.with(|c| c.set(ptr.unwrap_or(std::ptr::null_mut())));
    }
}

/// A single recorded statement: a new LHS variable and the upper bound of
/// its operand range in `Tape::operations`.
///
/// The operand range of `statements[i]` is
/// `[statements[i - 1].op_end, statements[i].op_end)`, exploiting the fact
/// that operand ranges are non-overlapping and laid out in insertion order.
/// The sentinel entry at `statements[0]` has `op_end = 0` and `slot =
/// u32::MAX`, so statement[1] starts its operands at operations[0].
#[derive(Debug, Clone)]
struct Statement {
    op_end: u32,
    slot: u32,
}

/// One operand of a statement: contributes `multiplier * d/d(slot)` to the
/// LHS variable's partial derivative during the reverse sweep.
#[derive(Debug, Clone, Copy)]
struct Operation<T: TapeStorage> {
    multiplier: T,
    slot: u32,
}

/// Position marker for the tape, used for partial rollback and adjoint computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapePosition {
    pub(crate) statement_pos: u32,
    pub(crate) operation_pos: u32,
}

/// The tape records operations for reverse-mode AD and computes adjoints.
pub struct Tape<T: TapeStorage> {
    statements: Vec<Statement>,
    operations: Vec<Operation<T>>,
    derivatives: Vec<T>,
    next_slot: u32,
    num_variables: u32,
}

// SAFETY: Tape is only accessed through thread-local storage via raw pointers.
// The thread-local ensures single-thread access.
unsafe impl<T: TapeStorage> Send for Tape<T> {}

impl<T: TapeStorage> Tape<T> {
    /// Create a new tape. The `_activate` flag is accepted for backwards
    /// compatibility with the C++ XAD API shape but is **ignored**: you
    /// must always call [`Tape::activate`] explicitly after the tape
    /// reaches its final storage location, because the tape's address is
    /// stored in thread-local storage and must remain stable.
    ///
    /// ```
    /// let mut tape = xad_rs::Tape::<f64>::new(true);
    /// tape.activate();
    /// ```
    pub fn new(_activate: bool) -> Self {
        Tape {
            statements: vec![Statement { op_end: 0, slot: u32::MAX }],
            operations: Vec::new(),
            derivatives: Vec::new(),
            next_slot: 0,
            num_variables: 0,
        }
    }

    /// Activate this tape as the thread-local active tape.
    pub fn activate(&mut self) {
        if T::get_active_ptr().is_some() {
            panic!("A tape is already active on this thread");
        }
        T::set_active_ptr(Some(self as *mut Tape<T>));
    }

    /// Deactivate this tape if it is the currently active one.
    pub fn deactivate(&mut self) {
        let ptr = self as *mut Tape<T>;
        if T::get_active_ptr() == Some(ptr) {
            T::set_active_ptr(None);
        }
    }

    /// Check if this tape is the currently active one.
    pub fn is_active(&self) -> bool {
        let ptr = self as *const Tape<T> as *mut Tape<T>;
        T::get_active_ptr() == Some(ptr)
    }

    /// Get a pointer to the currently active tape.
    pub(crate) fn get_active() -> Option<*mut Tape<T>> {
        T::get_active_ptr()
    }

    /// Deactivate any active tape on this thread.
    pub fn deactivate_all() {
        T::set_active_ptr(None);
    }

    /// Register a variable on the tape and return its slot.
    ///
    /// Slots are handed out monotonically from zero, so the derivatives
    /// buffer always has `derivatives.len() == next_slot` on exit. The
    /// underlying `Vec::push` grows geometrically, giving amortised O(1)
    /// insertion with no per-call resize bookkeeping.
    #[inline]
    pub fn register_variable(&mut self) -> u32 {
        let slot = self.next_slot;
        debug_assert_eq!(
            self.derivatives.len(),
            slot as usize,
            "tape derivatives invariant: len == next_slot"
        );
        self.next_slot += 1;
        self.num_variables += 1;
        self.derivatives.push(T::zero());
        slot
    }

    /// Record a statement: a new variable (LHS slot) that depends on the given operands.
    ///
    /// This is the fully general (slice-based) entry point; callers on the
    /// hot path should prefer [`Tape::push_nullary`], [`Tape::push_unary`],
    /// or [`Tape::push_binary`], which avoid any intermediate slice/`Vec`
    /// construction.
    #[inline]
    pub fn push_statement(&mut self, lhs_slot: u32, operands: &[(T, u32)]) {
        for &(multiplier, slot) in operands {
            self.operations.push(Operation { multiplier, slot });
        }
        self.statements.push(Statement {
            op_end: self.operations.len() as u32,
            slot: lhs_slot,
        });
    }

    /// Fast path: record a statement with **zero** operands (an input or
    /// constant). Avoids any slice traversal.
    #[inline]
    pub fn push_nullary(&mut self, lhs_slot: u32) {
        self.statements.push(Statement {
            op_end: self.operations.len() as u32,
            slot: lhs_slot,
        });
    }

    /// Fast path: record a statement with **one** operand. Avoids any
    /// intermediate slice/`Vec` construction.
    ///
    /// If `operand_slot` is `u32::MAX`, the operand is treated as inactive
    /// and is *not* pushed onto the operations buffer, matching the
    /// semantics of [`Tape::push_statement`] with a filtered slice.
    #[inline]
    pub fn push_unary(&mut self, lhs_slot: u32, multiplier: T, operand_slot: u32) {
        if operand_slot != u32::MAX {
            self.operations.push(Operation { multiplier, slot: operand_slot });
        }
        self.statements.push(Statement {
            op_end: self.operations.len() as u32,
            slot: lhs_slot,
        });
    }

    /// Fast path: record a statement with **two** operands.
    ///
    /// Inactive operands (slot `u32::MAX`) are skipped, matching the
    /// semantics of [`Tape::push_statement`] with a filtered slice.
    #[inline]
    pub fn push_binary(
        &mut self,
        lhs_slot: u32,
        m1: T,
        s1: u32,
        m2: T,
        s2: u32,
    ) {
        if s1 != u32::MAX {
            self.operations.push(Operation { multiplier: m1, slot: s1 });
        }
        if s2 != u32::MAX {
            self.operations.push(Operation { multiplier: m2, slot: s2 });
        }
        self.statements.push(Statement {
            op_end: self.operations.len() as u32,
            slot: lhs_slot,
        });
    }

    /// Start a new recording, clearing previous data but keeping the tape allocated.
    pub fn new_recording(&mut self) {
        self.statements.clear();
        self.statements.push(Statement { op_end: 0, slot: u32::MAX });
        self.operations.clear();
        self.derivatives.clear();
        self.next_slot = 0;
        self.num_variables = 0;
    }

    /// Compute all adjoints from the end of the tape to the beginning.
    pub fn compute_adjoints(&mut self) {
        let end = self.statements.len() as u32;
        self.compute_adjoints_to_impl(0, end);
    }

    /// Compute adjoints from the current end down to the given position.
    pub fn compute_adjoints_to(&mut self, pos: TapePosition) {
        let end = self.statements.len() as u32;
        self.compute_adjoints_to_impl(pos.statement_pos, end);
    }

    fn compute_adjoints_to_impl(&mut self, target_pos: u32, start: u32) {
        // Local slices unlock the invariants below for LLVM and let us
        // bypass bounds-check elision hand-wringing in the inner loop.
        //
        // ---------- Invariants relied upon by the `unsafe` code below ----------
        //
        // (I1) `self.derivatives.len() == self.num_variables as usize`
        //      Maintained by `register_variable`, which does a single
        //      `derivatives.push` per slot handout.
        //
        // (I2) Every `Statement.slot` is either `u32::MAX` (the index-0
        //      sentinel installed by `new` / `new_recording`) or was handed
        //      out by `register_variable`, so `slot < num_variables`.
        //
        // (I3) Every `Operation.slot` present in `self.operations` is
        //      strictly less than `num_variables`. The fixed-arity push
        //      helpers (`push_binary`, `push_unary`) filter `u32::MAX`
        //      operands *at push time*, so no sentinel ever lands in the
        //      operations buffer — this is why the legacy
        //      `if op.slot != u32::MAX` check is absent below.
        //
        // (I4) The sweep loop's `while i > target_pos + 1` condition means
        //      we never visit `i == 0` (the sentinel), so `stmt.slot`
        //      inside the loop is always a real, live slot (never
        //      `u32::MAX`). This is why the legacy
        //      `if lhs_slot == u32::MAX { continue; }` check is absent.
        //
        // (I5) Slots are monotonically assigned to freshly created
        //      variables, so a statement's LHS slot is strictly greater
        //      than any of its operand slots. `op.slot == lhs_slot` is
        //      therefore impossible and there is no write-read aliasing
        //      hazard on `derivatives` inside the inner loop.
        //
        // -----------------------------------------------------------------------

        let stmts = self.statements.as_slice();
        let ops = self.operations.as_slice();
        let derivs = self.derivatives.as_mut_slice();

        debug_assert_eq!(derivs.len(), self.num_variables as usize);

        let mut i = start as usize;
        let stop = target_pos as usize + 1;
        while i > stop {
            i -= 1;

            // SAFETY (I2, I4): `i > stop ≥ 1`, and `i < self.statements.len()`
            // by construction (`start` was `statements.len()` at the call
            // site, and we only decrement `i`).
            let stmt = unsafe { stmts.get_unchecked(i) };
            let lhs_slot = stmt.slot as usize;

            // SAFETY (I1, I2): `lhs_slot < num_variables == derivs.len()`.
            let adjoint = unsafe { *derivs.get_unchecked(lhs_slot) };
            if adjoint == T::zero() {
                continue;
            }

            let op_end = stmt.op_end as usize;
            // SAFETY (I4): `i ≥ 1`, so `i - 1` is a valid statement index.
            let op_start = unsafe { stmts.get_unchecked(i - 1).op_end as usize };

            for j in op_start..op_end {
                // SAFETY: `op_start..op_end` is a contiguous sub-range of
                // `ops` because every previously pushed statement's `op_end`
                // was `self.operations.len()` at push time, which is
                // monotonically non-decreasing.
                let op = unsafe { *ops.get_unchecked(j) };

                // SAFETY (I1, I3, I5): `op.slot < num_variables == derivs.len()`,
                // and `op.slot != lhs_slot` so there is no aliasing with the
                // prior `derivs[lhs_slot]` read above.
                unsafe {
                    *derivs.get_unchecked_mut(op.slot as usize) +=
                        op.multiplier * adjoint;
                }
            }
        }
    }

    /// Clear all derivative values (adjoints) to zero.
    pub fn clear_derivatives(&mut self) {
        for d in self.derivatives.iter_mut() {
            *d = T::zero();
        }
    }

    /// Get the derivative (adjoint) for the given slot.
    pub fn derivative(&self, slot: u32) -> T {
        self.derivatives.get(slot as usize).copied().unwrap_or_else(T::zero)
    }

    /// Set the derivative (adjoint) for the given slot.
    pub fn set_derivative(&mut self, slot: u32, value: T) {
        if slot as usize >= self.derivatives.len() {
            self.derivatives.resize(slot as usize + 1, T::zero());
        }
        self.derivatives[slot as usize] = value;
    }

    /// Increment the adjoint for the given slot.
    pub fn increment_adjoint(&mut self, slot: u32, value: T) {
        if slot as usize >= self.derivatives.len() {
            self.derivatives.resize(slot as usize + 1, T::zero());
        }
        self.derivatives[slot as usize] += value;
    }

    /// Get the current tape position for partial rollback.
    pub fn get_position(&self) -> TapePosition {
        TapePosition {
            statement_pos: self.statements.len() as u32,
            operation_pos: self.operations.len() as u32,
        }
    }

    /// Clear derivatives for all slots after the given position.
    pub fn clear_derivatives_after(&mut self, pos: TapePosition) {
        for i in (pos.statement_pos as usize)..self.statements.len() {
            let slot = self.statements[i].slot;
            if slot != u32::MAX && (slot as usize) < self.derivatives.len() {
                self.derivatives[slot as usize] = T::zero();
            }
        }
    }

    /// Reset the tape to the given position (truncate statements and operations after it).
    pub fn reset_to(&mut self, pos: TapePosition) {
        self.statements.truncate(pos.statement_pos as usize);
        self.operations.truncate(pos.operation_pos as usize);
    }

    /// Number of registered variables.
    pub fn num_variables(&self) -> u32 {
        self.num_variables
    }

    /// Number of recorded operations.
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }

    /// Number of recorded statements.
    pub fn num_statements(&self) -> usize {
        self.statements.len().saturating_sub(1)
    }

    /// Approximate memory usage in bytes.
    pub fn memory(&self) -> usize {
        self.statements.capacity() * std::mem::size_of::<Statement>()
            + self.operations.capacity() * std::mem::size_of::<Operation<T>>()
            + self.derivatives.capacity() * std::mem::size_of::<T>()
    }
}

impl<T: TapeStorage> Drop for Tape<T> {
    fn drop(&mut self) {
        self.deactivate();
    }
}
