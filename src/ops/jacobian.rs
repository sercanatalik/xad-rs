//! Full-Jacobian helper for vector-valued functions.
//!
//! [`compute_jacobian_rev`] computes the Jacobian matrix
//! `J[i][j] = ∂output_i / ∂input_j` for a function `f: Rⁿ → Rᵐ` given as a
//! closure: the function is recorded **once**, and the whole matrix comes
//! back from a **single** vector reverse sweep with each output seeded into
//! its own adjoint direction — see [`Tape::compute_adjoints_vector`]. It is
//! the driver that demonstrates that sweep; a caller with its own recording
//! can call the sweep directly.

use ndarray::Array2;

use crate::reverse::areal::AReal;
use crate::tape::{Tape, TapeStorage};

/// Compute the Jacobian using reverse (adjoint) mode.
///
/// Records the function **once** and recovers the full `m × n` Jacobian
/// (`J[[i, j]] = ∂output_i/∂input_j`) in a **single vector reverse sweep**
/// (each output seeded into its own adjoint direction), rather than one
/// scalar sweep per output. See [`Tape::compute_adjoints_vector`].
/// Efficient when `m ≤ n`.
pub fn compute_jacobian_rev<T, F>(inputs: &[T], func: F) -> Array2<T>
where
    T: TapeStorage,
    F: Fn(&[AReal<T>]) -> Vec<AReal<T>>,
{
    let mut tape = Tape::<T>::new(true);
    compute_jacobian_rev_with(&mut tape, inputs, func)
}

/// [`compute_jacobian_rev`] on a tape the caller owns.
///
/// Begins a fresh recording on `tape`, retaining its allocation, and returns
/// with the tape inactive and still allocated for the next call. Constructs
/// no tape; computes exactly what the bare driver computes.
///
/// # Panics
/// Panics if a tape is already active on this thread — recordings do not
/// nest.
///
/// # Example
///
/// ```
/// use xad_rs::ops::compute_jacobian_rev_with;
/// use xad_rs::Tape;
///
/// let mut tape = Tape::<f64>::new(true);
/// let j = compute_jacobian_rev_with(&mut tape, &[2.0_f64, 3.0], |v| {
///     vec![v[0].clone() * v[1].clone(), v[0].clone() + v[1].clone()]
/// });
/// assert_eq!(j[[0, 0]], 3.0);
/// assert_eq!(j[[1, 1]], 1.0);
/// ```
pub fn compute_jacobian_rev_with<T, F>(tape: &mut Tape<T>, inputs: &[T], func: F) -> Array2<T>
where
    T: TapeStorage,
    F: Fn(&[AReal<T>]) -> Vec<AReal<T>>,
{
    // RAII: deactivated when `_rec` drops, including on unwind from a panic
    // inside `func` — which the former `activate` / `deactivate_all` pair did
    // not survive.
    let _rec = tape.record();

    let mut ad_inputs: Vec<AReal<T>> = inputs.iter().map(|&v| AReal::new(v)).collect();
    AReal::register_input(&mut ad_inputs, tape);

    let mut ad_outputs = func(&ad_inputs);
    AReal::register_output(&mut ad_outputs, tape);

    let in_slots: Vec<u32> = ad_inputs.iter().map(|i| i.slot()).collect();
    let out_slots: Vec<u32> = ad_outputs.iter().map(|o| o.slot()).collect();
    jacobian_rows_serial(tape, &in_slots, &out_slots)
}

/// One vector sweep over a finished recording: seed output `o` to 1 in
/// direction `o`, sweep all `out_slots.len()` directions at once, and read
/// each input's gradient row back out. Shared by [`compute_jacobian_rev`]
/// and the small-tape fallback of [`compute_jacobian_rev_par`].
fn jacobian_rows_serial<T: TapeStorage>(
    tape: &Tape<T>,
    in_slots: &[u32],
    out_slots: &[u32],
) -> Array2<T> {
    let num_outputs = out_slots.len();
    let mut jacobian = Array2::<T>::zeros((num_outputs, in_slots.len()));
    if num_outputs == 0 {
        return jacobian;
    }
    let n_dir = num_outputs;
    let num_vars = tape.num_variables() as usize;
    let mut derivs = vec![T::zero(); num_vars * n_dir];
    for (o, &s) in out_slots.iter().enumerate() {
        derivs[s as usize * n_dir + o] = T::one();
    }
    tape.compute_adjoints_vector(n_dir, &mut derivs);

    for i in 0..num_outputs {
        for (j, &s) in in_slots.iter().enumerate() {
            jacobian[[i, j]] = derivs[s as usize * n_dir + i];
        }
    }
    jacobian
}
