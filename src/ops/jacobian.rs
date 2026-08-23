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
    tape.activate();

    let mut ad_inputs: Vec<AReal<T>> = inputs.iter().map(|&v| AReal::new(v)).collect();
    AReal::register_input(&mut ad_inputs, &mut tape);

    let mut ad_outputs = func(&ad_inputs);
    AReal::register_output(&mut ad_outputs, &mut tape);

    let in_slots: Vec<u32> = ad_inputs.iter().map(|i| i.slot()).collect();
    let out_slots: Vec<u32> = ad_outputs.iter().map(|o| o.slot()).collect();
    let jacobian = jacobian_rows_serial(&tape, &in_slots, &out_slots);

    Tape::<T>::deactivate_all();
    jacobian
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
