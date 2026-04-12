//! Hessian computation via reverse-mode AD and dense forward-mode Dual2Vec.

use ndarray::Array2;

use crate::reverse::areal::AReal;
use crate::forward::dual2vec::Dual2Vec;
use crate::tape::{Tape, TapeStorage};
use crate::registry::VarRegistry;

/// Compute the Hessian of a scalar-valued function using repeated reverse-mode passes
/// with finite-difference perturbation of the gradient.
///
/// For each input direction i, computes the gradient at `x` and at `x + eps * e_i`,
/// then approximates `H[i][j] = (grad_j(x + eps*e_i) - grad_j(x)) / eps`.
#[allow(clippy::needless_range_loop)]
pub fn compute_hessian<T, F>(inputs: &[T], func: F) -> Vec<Vec<T>>
where
    T: TapeStorage,
    F: Fn(&[AReal<T>]) -> AReal<T>,
{
    let n = inputs.len();
    let eps = T::from(1e-7).unwrap();
    let inv_eps = T::one() / eps;
    let half = T::from(0.5).unwrap();
    let mut hessian = vec![vec![T::zero(); n]; n];

    // Compute base gradient
    let base_grad = compute_gradient(inputs, &func);

    // For each direction, perturb and compute gradient
    for i in 0..n {
        let mut perturbed = inputs.to_vec();
        perturbed[i] += eps;
        let perturbed_grad = compute_gradient(&perturbed, &func);

        for (j, h_ij) in hessian[i].iter_mut().enumerate() {
            *h_ij = (perturbed_grad[j] - base_grad[j]) * inv_eps;
        }
    }

    // Symmetrize off-diagonal entries.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (hessian[i][j] + hessian[j][i]) * half;
            hessian[i][j] = avg;
            hessian[j][i] = avg;
        }
    }

    hessian
}

/// Compute the gradient of a scalar-valued function using reverse-mode AD.
fn compute_gradient<T, F>(inputs: &[T], func: &F) -> Vec<T>
where
    T: TapeStorage,
    F: Fn(&[AReal<T>]) -> AReal<T>,
{
    let mut tape = Tape::<T>::new(true);
    tape.activate();

    let mut ad_inputs: Vec<AReal<T>> = inputs.iter().map(|&v| AReal::new(v)).collect();
    AReal::register_input(&mut ad_inputs, &mut tape);

    let mut output = func(&ad_inputs);
    AReal::register_output(std::slice::from_mut(&mut output), &mut tape);

    output.set_adjoint(&mut tape, T::one());
    tape.compute_adjoints();

    let grad: Vec<T> = ad_inputs.iter().map(|x| x.adjoint(&tape)).collect();

    Tape::<T>::deactivate_all();
    grad
}

// ============================================================================
// Labeled Hessian (dense full Hessian via Dual2Vec)
// ============================================================================

/// Labeled dense full Hessian produced by [`compute_full_hessian`].
pub struct NamedHessian {
    pub vars: VarRegistry,
    pub value: f64,
    pub gradient: Vec<f64>,
    pub hessian: Array2<f64>,
}

/// Compute a named dense full Hessian via a single `Dual2Vec` forward pass.
pub fn compute_full_hessian<F>(inputs: &[(String, f64)], f: F) -> NamedHessian
where
    F: Fn(&[Dual2Vec]) -> Dual2Vec,
{
    let n = inputs.len();

    let names: Vec<String> = inputs.iter().map(|(name, _)| name.clone()).collect();
    let vars = VarRegistry::from_names(names);

    let d2v_inputs: Vec<Dual2Vec> = inputs
        .iter()
        .enumerate()
        .map(|(i, (_, value))| Dual2Vec::variable(*value, i, n))
        .collect();

    let output = f(&d2v_inputs);

    let value = output.value();
    let gradient: Vec<f64> = output.gradient().to_vec();
    let hessian: Array2<f64> = output.hessian().clone();

    NamedHessian {
        vars,
        value,
        gradient,
        hessian,
    }
}
