//! Full-Jacobian helpers for vector-valued functions.
//!
//! Computes the Jacobian matrix `J[i][j] = ∂output_i / ∂input_j` for a
//! function `f: Rⁿ → Rᵐ` given as a closure. Two mode-specific entry
//! points are provided and you should pick whichever is cheaper for your
//! problem shape:
//!
//! - [`compute_jacobian_rev`] — reverse-mode; one tape pass per output
//!   row. Efficient when `m ≪ n` (many inputs, few outputs).
//! - [`compute_jacobian_fwd`] — forward-mode; one forward pass per input
//!   column. Efficient when `n ≪ m` (few inputs, many outputs).

use std::sync::Arc;

use ndarray::Array2;

use crate::reverse::areal::AReal;
use crate::forward::freal::FReal;
use crate::scalar::Scalar;
use crate::tape::{Tape, TapeStorage};
use crate::registry::VarRegistry;

/// Compute the Jacobian using reverse (adjoint) mode.
///
/// Efficient when `num_outputs << num_inputs`. Requires one reverse pass per output.
pub fn compute_jacobian_rev<T, F>(inputs: &[T], func: F) -> Vec<Vec<T>>
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

    let num_inputs = ad_inputs.len();
    let num_outputs = ad_outputs.len();
    let mut jacobian = vec![vec![T::zero(); num_inputs]; num_outputs];

    for i in 0..num_outputs {
        tape.clear_derivatives();
        ad_outputs[i].set_adjoint(&mut tape, T::one());
        tape.compute_adjoints();

        for j in 0..num_inputs {
            jacobian[i][j] = ad_inputs[j].adjoint(&tape);
        }
    }

    Tape::<T>::deactivate_all();
    jacobian
}

/// Compute the Jacobian using forward (tangent) mode.
///
/// Efficient when `num_inputs << num_outputs`. Requires one forward pass per input.
pub fn compute_jacobian_fwd<T, F>(inputs: &[T], func: F) -> Vec<Vec<T>>
where
    T: Scalar,
    F: Fn(&[FReal<T>]) -> Vec<FReal<T>>,
{
    let zero_inputs: Vec<FReal<T>> = inputs
        .iter()
        .map(|&v| FReal::constant(v))
        .collect();
    let num_outputs = func(&zero_inputs).len();
    let num_inputs = inputs.len();
    let mut jacobian = vec![vec![T::zero(); num_inputs]; num_outputs];

    for j in 0..num_inputs {
        let mut fwd_inputs: Vec<FReal<T>> = inputs
            .iter()
            .map(|&v| FReal::constant(v))
            .collect();
        fwd_inputs[j].set_derivative(T::one());

        let fwd_outputs = func(&fwd_inputs);

        for i in 0..num_outputs {
            jacobian[i][j] = fwd_outputs[i].derivative();
        }
    }

    jacobian
}

// ============================================================================
// Labeled Jacobian
// ============================================================================

/// Row-named, column-named Jacobian matrix.
///
/// `matrix[[i, j]] = ∂outputs[i] / ∂inputs[j]`.
pub struct NamedJacobian {
    pub rows: Vec<String>,
    pub cols: Arc<VarRegistry>,
    pub matrix: Array2<f64>,
}

/// Compute a named Jacobian via reverse-mode AD.
pub fn compute_named_jacobian<F>(
    inputs: &[(String, f64)],
    outputs: &[String],
    f: F,
) -> NamedJacobian
where
    F: Fn(&[AReal<f64>]) -> Vec<AReal<f64>>,
{
    let input_values: Vec<f64> = inputs.iter().map(|(_, v)| *v).collect();
    let positional = compute_jacobian_rev(&input_values, f);

    let m = outputs.len();
    let n = inputs.len();
    assert_eq!(
        positional.len(),
        m,
        "compute_named_jacobian: closure produced {} outputs but {} output labels were given",
        positional.len(),
        m
    );

    let mut matrix = Array2::<f64>::zeros((m, n));
    for (i, row) in positional.iter().enumerate() {
        debug_assert_eq!(row.len(), n);
        for (j, &v) in row.iter().enumerate() {
            matrix[[i, j]] = v;
        }
    }

    let input_names: Vec<String> = inputs.iter().map(|(name, _)| name.clone()).collect();
    NamedJacobian {
        rows: outputs.to_vec(),
        cols: Arc::new(VarRegistry::from_names(input_names)),
        matrix,
    }
}
