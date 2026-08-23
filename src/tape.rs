//! Tape for recording operations in reverse-mode automatic differentiation.
//!
//! The tape records a trace of all arithmetic operations performed on
//! [`AReal`](crate::reverse::areal::AReal) values, then replays them in reverse to
//! compute adjoints (gradients). One tape covers the whole forward pass of
//! a computation; at the end, [`Tape::compute_adjoints`] walks the recorded
//! trace backwards once, accumulating partial derivatives in the
//! derivatives buffer.
//!
//! # Data layout
//!
//! The tape uses a compact three-buffer layout (classic XAD-style):
//!
//! - `statements: Vec<u32>` — one entry per recorded *variable*, holding
//!   only that statement's `op_end` pointer into the operations buffer.
//!   **Slot numbers are implicit**: statement index `i` *is* slot
//!   `i - 1` (index 0 is a sentinel with `op_end = 0`). Every slot —
//!   including inputs, which record a zero-operand statement — has
//!   exactly one statement, so nothing but `op_end` needs storing and a
//!   statement is 4 bytes instead of 8.
//! - `operations: Vec<Operation<T>>` — a packed stream of
//!   `(multiplier, operand_slot)` pairs. Statement `i`'s operand range is
//!   `[statements[i - 1], statements[i])`, which means operand lookup
//!   is O(1) and the whole tape is a single linear scan in both
//!   directions.
//! - `derivatives: Vec<T>` — indexed directly by slot number. After
//!   [`compute_adjoints`](Tape::compute_adjoints) this is the gradient
//!   vector for all registered variables.
//!
//! Slots are handed out monotonically by the push helpers (each returns
//! the fresh LHS slot of the statement it records) and by
//! [`Tape::register_variable`] for inputs — the derivatives buffer is
//! **not** touched during recording. It is materialized in one
//! `resize(num_variables)` at the top of
//! [`compute_adjoints`](Tape::compute_adjoints) (or grown on demand by
//! [`set_derivative`](Tape::set_derivative) when seeding adjoints), so
//! `derivatives.len() >= num_variables` holds when the reverse sweep
//! runs and no per-slot bounds-checks are needed in its hot loop.
//! Keeping the per-op `push(T::zero())` out of the recording fast path
//! makes recording ~1.9× faster on op-heavy kernels.
//!
//! # Hot-path recording helpers
//!
//! The three fixed-arity push methods — [`Tape::push_nullary`],
//! [`Tape::push_unary`], [`Tape::push_binary`] — are the zero-allocation
//! fast paths used by the AReal operator impls. They push operands
//! directly onto the operations buffer without going through any
//! intermediate slice or `Vec`. For arbitrary-arity fused statements,
//! [`Tape::push_operands_bounded`] is the bulk equivalent: one capacity
//! reserve for the whole operand batch instead of a grow check per
//! operand.
//!
//! # Thread-local active tape
//!
//! Exactly one `Tape<T>` can be *active* per thread per scalar type at a
//! time. The active tape is stored as a raw pointer in a thread-local
//! `Cell<*mut Tape<T>>` (null encodes "no active tape"). Making a tape
//! active is an explicit opt-in via [`Tape::activate`]; AReal operators
//! read the active pointer via [`TapeStorage::get_active_ptr`] and
//! record only if it's non-null.

use crate::passive::Passive;
use std::cell::Cell;
use std::marker::PhantomData;
use std::ptr;

thread_local! {
    // A null pointer encodes "no active tape" — cheaper than `Option<*mut _>`
    // because `Cell::get()` is a single load with no runtime borrow check.
    static ACTIVE_TAPE_F64: Cell<*mut Tape<f64>> = const { Cell::new(ptr::null_mut()) };
    // Forward-over-adjoint engine: a tape whose scalar is a forward dual.
    static ACTIVE_TAPE_JET1_F64: Cell<*mut Tape<crate::forward::jet1::Jet1<f64>>> =
        const { Cell::new(ptr::null_mut()) };
}

/// Trait for thread-local tape storage access, implemented per concrete scalar type.
pub trait TapeStorage: Passive {
    fn get_active_ptr() -> Option<*mut Tape<Self>>;
    fn set_active_ptr(ptr: Option<*mut Tape<Self>>);
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

impl TapeStorage for crate::forward::jet1::Jet1<f64> {
    #[inline]
    fn get_active_ptr() -> Option<*mut Tape<crate::forward::jet1::Jet1<f64>>> {
        let p = ACTIVE_TAPE_JET1_F64.with(|c| c.get());
        if p.is_null() { None } else { Some(p) }
    }
    #[inline]
    fn set_active_ptr(ptr: Option<*mut Tape<crate::forward::jet1::Jet1<f64>>>) {
        ACTIVE_TAPE_JET1_F64.with(|c| c.set(ptr.unwrap_or(std::ptr::null_mut())));
    }
}

// A recorded statement is just the upper bound of its operand range in
// `Tape::operations`; its LHS slot is its statement index minus one. The
// operand range of statement `i` is `[statements[i - 1], statements[i])`,
// exploiting the fact that operand ranges are non-overlapping and laid out
// in insertion order. The sentinel entry `statements[0] == 0` makes
// statement 1 (slot 0) start its operands at operations[0].

/// One operand of a statement: contributes `multiplier * d/d(slot)` to the
/// LHS variable's partial derivative during the reverse sweep.
///
/// NOTE(perf): for `T = f64` this pads to 16 bytes (4 bytes wasted), and
/// two attempts to reclaim them both *lost* to this layout when measured
/// (Apple M-series, rustc 1.92), including on a deliberately LLC-busting
/// tape — the operand stream is not the bottleneck, so the saved bandwidth
/// never materializes as time:
///   - SoA split (`Vec<T>` multipliers + `Vec<u32>` slots, 12 B/op):
///     recording +97% (five capacity-checked pushes per binary op instead
///     of three), scalar sweep +11%, 16-direction vector sweep −2%.
///   - `repr(C, packed(4))` (12 B/op, single stream): recording +8%,
///     scalar sweep +8.5%, 16-direction vector sweep +4% — unaligned `f64`
///     traffic costs more than the padding saves.
/// Keep the naturally aligned pair; re-measure before changing it.
#[derive(Debug, Clone, Copy)]
struct Operation<T: TapeStorage> {
    multiplier: T,
    slot: u32,
}

/// The tape records operations for reverse-mode AD and computes adjoints.
pub struct Tape<T: TapeStorage> {
    // Per-statement `op_end` values; index `i` is slot `i - 1` (see the
    // module-level "Data layout" section). `statements[0]` is a sentinel 0.
    statements: Vec<u32>,
    operations: Vec<Operation<T>>,
    derivatives: Vec<T>,
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
            statements: vec![0],
            operations: Vec::new(),
            derivatives: Vec::new(),
        }
    }

    /// Create a tape with pre-reserved capacity for `n_statements` and
    /// `n_operations`, avoiding geometric-growth reallocation copies on large
    /// recordings. Capacities are hints; the tape still grows as needed.
    ///
    /// Pair with [`record`](Tape::record) to reuse one warmed tape across many
    /// valuations — on small-tape workloads this is ~2.4× faster than a fresh
    /// [`Tape::new`] per valuation (the allocation churn is amortized away).
    ///
    /// ```
    /// let mut tape = xad_rs::Tape::<f64>::with_capacity(1024, 4096);
    /// tape.activate();
    /// ```
    pub fn with_capacity(n_statements: usize, n_operations: usize) -> Self {
        let mut statements = Vec::with_capacity(n_statements + 1);
        statements.push(0);
        Tape {
            statements,
            operations: Vec::with_capacity(n_operations),
            derivatives: Vec::with_capacity(n_statements),
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

    /// Activate this tape and return an RAII guard whose `Drop`
    /// deactivates the thread-local active tape. Replaces the
    /// `activate()` + manual `deactivate_all()` pair when the
    /// activation needs to be scoped to a block.
    ///
    /// Panics with the same semantics as [`Tape::activate`] if another
    /// tape is already active on this thread.
    pub fn activate_guard(&mut self) -> TapeGuard<T> {
        let ptr = self as *mut Tape<T>;
        self.activate();
        TapeGuard {
            tape_ptr: ptr,
            _not_send: PhantomData,
        }
    }

    /// Begin a fresh recording on this **reused** tape and activate it,
    /// returning an RAII guard that deactivates on drop. Equivalent to
    /// [`new_recording`](Tape::new_recording) followed by
    /// [`activate_guard`](Tape::activate_guard).
    ///
    /// This is the ergonomic way to reuse one tape across many valuations: the
    /// previous recording's allocation is retained (`new_recording` only
    /// clears), so a loop over scenarios/positions amortizes tape allocation
    /// instead of paying a fresh [`Tape::new`] each time (~3× measured on a
    /// many-small-tapes recording workload). The returned guard holds a raw pointer, not
    /// a borrow, so the tape stays usable (e.g. for `register_input`) while the
    /// guard is alive; the guard's `Drop` deactivates the thread-local slot.
    ///
    /// ```
    /// use xad_rs::{AReal, Tape};
    /// let mut tape = Tape::<f64>::with_capacity(64, 256);
    /// for &x0 in &[1.0_f64, 2.0, 3.0] {
    ///     let _rec = tape.record();                 // reuse: clear + activate
    ///     let mut x = AReal::new(x0);
    ///     AReal::register_input(std::slice::from_mut(&mut x), &mut tape);
    ///     let mut y = &x * &x;                       // y = x^2
    ///     AReal::register_output(std::slice::from_mut(&mut y), &mut tape);
    ///     y.set_adjoint(&mut tape, 1.0);
    ///     tape.compute_adjoints();
    ///     assert!((x.adjoint(&tape) - 2.0 * x0).abs() < 1e-12);
    /// }                                             // _rec drops -> deactivate
    /// ```
    ///
    /// # Panics
    /// Panics (like [`activate`](Tape::activate)) if another tape is already
    /// active on this thread — recordings do not nest.
    pub fn record(&mut self) -> TapeGuard<T> {
        self.new_recording();
        self.activate_guard()
    }

    /// Register a variable on the tape and return its slot.
    ///
    /// Slots are handed out monotonically from zero and correspond 1:1
    /// with statements, so this records a zero-operand statement (4 bytes)
    /// for the new variable — used for inputs and constant outputs, which
    /// never appear as an op's LHS. The derivatives buffer is deliberately
    /// **not** grown here — it is materialized in a single `resize` by
    /// [`compute_adjoints`](Tape::compute_adjoints) (or on demand by
    /// [`set_derivative`](Tape::set_derivative)).
    #[inline]
    pub fn register_variable(&mut self) -> u32 {
        self.push_nullary()
    }

    /// Fast path: record a statement with **zero** operands (an input or
    /// constant) and return its fresh LHS slot.
    #[inline]
    pub fn push_nullary(&mut self) -> u32 {
        // NOTE(perf): compute the slot from `len` *before* the push. Deriving
        // it from the post-push length (`len - 2`) puts the returned slot on
        // a dependency chain through the push's length update and costs ~25%
        // when recording onto a reused tape.
        let slot = (self.statements.len() - 1) as u32;
        self.statements.push(self.operations.len() as u32);
        slot
    }

    /// Fast path: record a statement with **one** operand and return its
    /// fresh LHS slot. Avoids any intermediate slice/`Vec` construction.
    ///
    /// If `operand_slot` is `u32::MAX`, the operand is treated as inactive
    /// and is *not* pushed onto the operations buffer.
    #[inline]
    pub fn push_unary(&mut self, multiplier: T, operand_slot: u32) -> u32 {
        // NOTE(perf): a reserve-plus-unconditional-write variant (as in
        // `push_binary`) measured no better here — one conditional push is
        // already optimal for a single operand. Keep the simple form.
        if operand_slot != u32::MAX {
            self.operations.push(Operation { multiplier, slot: operand_slot });
        }
        self.push_nullary()
    }

    /// Fast path: record a statement with **two** operands and return its
    /// fresh LHS slot.
    ///
    /// Inactive operands (slot `u32::MAX`) are skipped.
    #[inline]
    pub fn push_binary(&mut self, m1: T, s1: u32, m2: T, s2: u32) -> u32 {
        // NOTE(perf): one `reserve(2)` + branchless unconditional writes
        // instead of two capacity-checked conditional pushes — worth ~15%
        // on `record/reused_tape` (~11% on `_large`): the operator hot loop
        // drops two grow-check branches and the operand stores no longer
        // depend on the activity tests.
        let len = self.operations.len();
        self.operations.reserve(2);
        // SAFETY: `reserve(2)` guarantees room for both writes past `len`;
        // `set_len` only exposes the `n` operands that are actually active
        // (an inactive first operand's write is overwritten or excluded).
        unsafe {
            let p = self.operations.as_mut_ptr().add(len);
            p.write(Operation { multiplier: m1, slot: s1 });
            let n1 = (s1 != u32::MAX) as usize;
            p.add(n1).write(Operation { multiplier: m2, slot: s2 });
            let n = n1 + (s2 != u32::MAX) as usize;
            self.operations.set_len(len + n);
        }
        self.push_nullary()
    }

    /// Push a single operand of an in-progress **n-ary** statement.
    ///
    /// Call once per operand, then close the statement with
    /// [`end_statement`](Tape::end_statement). Inactive operands (slot
    /// `u32::MAX`) are skipped, mirroring [`push_unary`](Tape::push_unary) /
    /// [`push_binary`](Tape::push_binary). Fused recorders
    /// ([`math::ad::sum`](crate::math::ad::sum),
    /// [`math::ad::dot`](crate::math::ad::dot), ...) record an
    /// arbitrary-arity op as **one** statement instead of a chain of binary
    /// temporaries — shrinking both the statement and operand streams the
    /// reverse sweep must walk; they push their operands in bulk via
    /// [`push_operands_bounded`](Tape::push_operands_bounded), keeping this
    /// per-operand entry point for callers that don't know their operand
    /// count up front.
    #[inline]
    pub fn push_operand(&mut self, multiplier: T, operand_slot: u32) {
        if operand_slot != u32::MAX {
            self.operations.push(Operation { multiplier, slot: operand_slot });
        }
    }

    /// Bulk variant of [`push_operand`](Tape::push_operand): push the
    /// operands of an in-progress n-ary statement with a **single**
    /// capacity reserve and no per-operand grow checks — the
    /// [`push_binary`](Tape::push_binary) reserve-plus-unconditional-write
    /// trick generalized to n operands. Inactive operands (slot `u32::MAX`)
    /// are skipped, as in the fixed-arity helpers.
    ///
    /// `n_max` must be an upper bound on the number of yielded operands
    /// (the slice-based fused recorders in [`crate::math::ad`] know it
    /// exactly). Yielding more than `n_max` items is a logic error: it is
    /// caught by a `debug_assert!`, and in release builds the excess
    /// operands are dropped.
    ///
    /// NOTE(perf): worth −3.3% on a fused `sum` and −2.4% on a fused
    /// `weighted_dot`, end-to-end versus per-operand `push_operand` calls
    /// (Apple M-series, rustc 1.92, fat LTO) — those measurements also record
    /// the input legs and run the sweep, so the fused statement's recording
    /// itself speeds up by more than the headline.
    #[inline]
    pub fn push_operands_bounded(
        &mut self,
        n_max: usize,
        operands: impl IntoIterator<Item = (T, u32)>,
    ) {
        let len = self.operations.len();
        self.operations.reserve(n_max);
        let mut iter = operands.into_iter();
        // SAFETY: `reserve(n_max)` guarantees room for `n_max` writes past
        // `len`, and `take(n_max)` caps the number of writes at `n_max`;
        // `set_len` only exposes the operands that are actually active (an
        // inactive operand's write is overwritten by the next active one or
        // excluded by the final length, as in `push_binary`).
        unsafe {
            let p = self.operations.as_mut_ptr().add(len);
            let mut written = 0usize;
            for (multiplier, slot) in iter.by_ref().take(n_max) {
                p.add(written).write(Operation { multiplier, slot });
                written += (slot != u32::MAX) as usize;
            }
            self.operations.set_len(len + written);
        }
        debug_assert!(
            iter.next().is_none(),
            "push_operands_bounded: iterator yielded more than n_max operands"
        );
    }

    /// Close an n-ary statement and return its fresh LHS slot: the operand
    /// range covers everything pushed via
    /// [`push_operand`](Tape::push_operand) since the previous statement.
    /// With no preceding `push_operand` calls this is identical to
    /// [`push_nullary`](Tape::push_nullary).
    #[inline]
    pub fn end_statement(&mut self) -> u32 {
        self.push_nullary()
    }

    /// Start a new recording, clearing previous data but keeping the tape allocated.
    pub fn new_recording(&mut self) {
        self.statements.clear();
        self.statements.push(0);
        self.operations.clear();
        self.derivatives.clear();
    }

    /// Compute all adjoints from the end of the tape to the beginning.
    pub fn compute_adjoints(&mut self) {
        // Materialize the derivatives buffer for every registered slot.
        // Recording never grows it (see `register_variable`); seeding via
        // `set_derivative` may have grown it partially.
        let num_vars = self.num_variables() as usize;
        if self.derivatives.len() < num_vars {
            self.derivatives.resize(num_vars, T::zero());
        }
        let end = self.statements.len() as u32;
        self.compute_adjoints_to_impl(0, end);
    }

    /// Vector reverse sweep — propagate `n_dir` adjoint directions through the
    /// recorded tape in a **single pass**. A Jacobian of `f: Rⁿ → Rᵐ` is one
    /// recording plus one call here with `n_dir = m`, versus `m` scalar sweeps
    /// (see [`compute_jacobian_rev`](crate::compute_jacobian_rev)).
    ///
    /// `derivs` is a caller-owned, **slot-major** buffer of length
    /// `num_variables * n_dir` (slot `s`, direction `d` at index
    /// `s * n_dir + d`), pre-seeded with the output adjoints (typically each
    /// output seeded to `1` in its own direction). On return it holds the
    /// adjoints of every recorded variable in all `n_dir` directions; read an
    /// input's gradient row at `input_slot * n_dir + d`.
    ///
    /// The scalar [`compute_adjoints`](Tape::compute_adjoints) path is left
    /// entirely untouched, so single-direction users pay nothing for the
    /// existence of vector mode.
    ///
    /// # Panics
    /// Panics if `derivs.len() != num_variables * n_dir`.
    pub fn compute_adjoints_vector(&self, n_dir: usize, derivs: &mut [T]) {
        assert_eq!(
            derivs.len(),
            self.num_variables() as usize * n_dir,
            "compute_adjoints_vector: derivs.len() must equal num_variables * n_dir"
        );
        if n_dir == 0 {
            return;
        }

        // NOTE(perf): three rewrites of this loop have been measured and all
        // lost to the safe version below — keep it unless a fresh measurement
        // (including on an LLC-busting tape) says otherwise:
        //   - `unsafe`/`get_unchecked` (mirroring the scalar sweep), at
        //     n_dir ∈ {4, 16, 64} over a 768k-operand tape: no measurable
        //     win — LLVM already elides these checks.
        //   - raw-pointer rows (`from_raw_parts(_mut)` instead of
        //     `split_at_mut` + subslices): +2-3% — the slice bounds
        //     knowledge evidently *helps* the vectorizer.
        //   - fixed-width direction tiling (`chunks_exact(4)` over the
        //     row): +56% — the iterator machinery wrecked codegen that was
        //     already auto-vectorizing cleanly with a runtime trip count.
        let stmts = self.statements.as_slice();
        let mut i = stmts.len();
        while i > 1 {
            i -= 1;
            let op_end = stmts[i] as usize;
            let op_start = stmts[i - 1] as usize;
            if op_start == op_end {
                continue;
            }
            // NOTE(perf): `black_box` hides from LLVM that `lhs_slot` is an
            // affine function of the loop counter (before slots were implicit
            // in statement order, it was a value loaded from the statement,
            // equally opaque). Letting LLVM see the relation costs ~17% on
            // a 16-direction vector sweep and ~9% on the scalar sweep,
            // reproducibly (A/B measured, Apple M-series, rustc 1.92
            // fat-LTO) — the extra register round-trip is far cheaper than
            // whatever the optimizer does with the extra knowledge.
            let lhs_slot = std::hint::black_box(i - 1);

            // The lhs adjoint row is `[lhs_slot*n_dir, lhs_slot*n_dir+n_dir)`.
            // Every operand slot is strictly less than `lhs_slot` (invariant I5
            // in `compute_adjoints_to_impl`), so all operand rows lie strictly
            // below `lhs_slot*n_dir`. Splitting there cleanly separates the read
            // (lhs row) from the writes (operand rows) — no `unsafe` needed.
            let split = lhs_slot * n_dir;
            let (below, from) = derivs.split_at_mut(split);
            let adj = &from[..n_dir];

            if adj.iter().all(|a| a.is_zero()) {
                continue;
            }

            for op in &self.operations[op_start..op_end] {
                let mult = op.multiplier;
                let base = op.slot as usize * n_dir;
                let target = &mut below[base..base + n_dir];
                for d in 0..n_dir {
                    target[d] += mult * adj[d];
                }
            }
        }
    }

    fn compute_adjoints_to_impl(&mut self, target_pos: u32, start: u32) {
        // Local slices unlock the invariants below for LLVM and let us
        // bypass bounds-check elision hand-wringing in the inner loop.
        //
        // ---------- Invariants relied upon by the `unsafe` code below ----------
        //
        // (I1) `self.derivatives.len() >= num_variables`
        //      (`num_variables == statements.len() - 1`). Established by the
        //      `resize` in `compute_adjoints` (the only caller) before this
        //      runs, and re-checked by the hard `assert!` below so the
        //      `get_unchecked` accesses stay bounded even if a future caller
        //      skips the resize.
        //
        // (I2) Statement index `i` corresponds 1:1 to LHS slot `i - 1`
        //      (index 0 is the sentinel), so for every visited statement
        //      `lhs_slot = i - 1 < statements.len() - 1 == num_variables`.
        //
        // (I3) Every `Operation.slot` present in `self.operations` is
        //      strictly less than `num_variables`. The fixed-arity push
        //      helpers (`push_binary`, `push_unary`) filter `u32::MAX`
        //      operands *at push time*, so no sentinel ever lands in the
        //      operations buffer — this is why the legacy
        //      `if op.slot != u32::MAX` check is absent below.
        //
        // (I4) The sweep loop's `while i > target_pos + 1` condition means
        //      we never visit `i == 0` (the sentinel), so `i - 1` is always
        //      a valid statement index and a real, live slot.
        //
        // (I5) An operand's statement was recorded before the statement that
        //      consumes it, so a statement's LHS slot is strictly greater
        //      than any of its operand slots. `op.slot == lhs_slot` is
        //      therefore impossible and there is no write-read aliasing
        //      hazard on `derivatives` inside the inner loop.
        //
        // -----------------------------------------------------------------------

        let stmts = self.statements.as_slice();
        let ops = self.operations.as_slice();
        let derivs = self.derivatives.as_mut_slice();

        // Hard assert (not debug): the unchecked derivative accesses below
        // are only sound if every registered slot has backing storage.
        assert!(
            derivs.len() >= stmts.len() - 1,
            "derivatives buffer must be materialized before the reverse sweep"
        );

        let mut i = start as usize;
        let stop = target_pos as usize + 1;
        while i > stop {
            i -= 1;

            // SAFETY (I4): `i > stop ≥ 1`, and `i < self.statements.len()`
            // by construction (`start` was `statements.len()` at the call
            // site, and we only decrement `i`).
            let op_end = unsafe { *stmts.get_unchecked(i) as usize };
            // SAFETY (I4): `i ≥ 1`, so `i - 1` is a valid statement index.
            let op_start = unsafe { *stmts.get_unchecked(i - 1) as usize };
            // Nullary statements (inputs, constant outputs) have nothing to
            // propagate — skip before touching the derivatives buffer.
            if op_start == op_end {
                continue;
            }

            // NOTE(perf): see the matching `black_box` in
            // `compute_adjoints_vector` — hiding the affine relation between
            // `lhs_slot` and the loop counter is worth ~9% on `sweep/scalar`.
            let lhs_slot = std::hint::black_box(i - 1);
            // SAFETY (I1, I2): `lhs_slot < num_variables <= derivs.len()`.
            let adjoint = unsafe { *derivs.get_unchecked(lhs_slot) };
            // `is_zero()` (not `== T::zero()`): for the dual-scalar engine
            // (`T = Jet1<f64>`) an adjoint can have a zero value but a nonzero
            // tangent (a real second-order contribution) that must not be
            // skipped; `Jet1::is_zero` checks both parts. Identical to `== 0`
            // for `f64`.
            if adjoint.is_zero() {
                continue;
            }

            for j in op_start..op_end {
                // SAFETY: `op_start..op_end` is a contiguous sub-range of
                // `ops` because every previously pushed statement's `op_end`
                // was `self.operations.len()` at push time, which is
                // monotonically non-decreasing.
                let op = unsafe { *ops.get_unchecked(j) };

                // SAFETY (I1, I3, I5): `op.slot < num_variables <= derivs.len()`,
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

    /// Number of registered variables.
    pub fn num_variables(&self) -> u32 {
        (self.statements.len() - 1) as u32
    }

    /// Number of recorded operations.
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }

    /// Number of recorded statements (one per registered variable,
    /// including the zero-operand statements recorded for inputs).
    pub fn num_statements(&self) -> usize {
        self.statements.len() - 1
    }

    /// Approximate memory usage in bytes.
    pub fn memory(&self) -> usize {
        self.statements.capacity() * std::mem::size_of::<u32>()
            + self.operations.capacity() * std::mem::size_of::<Operation<T>>()
            + self.derivatives.capacity() * std::mem::size_of::<T>()
    }
}

impl<T: TapeStorage> Drop for Tape<T> {
    fn drop(&mut self) {
        self.deactivate();
    }
}

/// RAII guard returned by [`Tape::activate_guard`]. On drop, if the
/// guard's tape is still the active tape on this thread, the
/// thread-local active-tape pointer is cleared.
///
/// The guard is `!Send` (and therefore `!Sync`) because tape activation
/// is thread-local. If the guard outlives the `Tape<T>` it was created
/// from, the tape's own `Drop` will have already cleared the active
/// pointer, and the guard's `Drop` becomes a no-op — safe by
/// construction. Tape A's guard cannot accidentally deactivate tape B.
pub struct TapeGuard<T: TapeStorage> {
    tape_ptr: *mut Tape<T>,
    _not_send: PhantomData<*mut ()>,
}

impl<T: TapeStorage> Drop for TapeGuard<T> {
    fn drop(&mut self) {
        if T::get_active_ptr() == Some(self.tape_ptr) {
            T::set_active_ptr(None);
        }
    }
}
