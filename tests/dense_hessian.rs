//! Tests for the positional dense full Hessian (`compute_full_hessian` →
//! `DenseHessian`), replacing the former name-keyed `NamedHessian`.

use xad_rs::{compute_full_hessian, Jet2Vec};

const TOL: f64 = 1e-12;

#[test]
fn dense_hessian_x2y_plus_y3() {
    // f(x, y) = x²·y + y³  →  ∇ = [2xy, x² + 3y²], H = [[2y, 2x], [2x, 6y]].
    // At (2, 3): value 12+27=39, ∇=[12, 4+27=31], H=[[6,4],[4,18]].
    let r = compute_full_hessian(&[2.0_f64, 3.0], |v: &[Jet2Vec]| {
        let x2y = &(&v[0] * &v[0]) * &v[1];
        let y3 = &(&v[1] * &v[1]) * &v[1];
        &x2y + &y3
    });

    assert!((r.value - 39.0).abs() < TOL);
    assert!((r.gradient[0] - 12.0).abs() < TOL);
    assert!((r.gradient[1] - 31.0).abs() < TOL);
    assert!((r.hessian[[0, 0]] - 6.0).abs() < TOL);
    assert!((r.hessian[[0, 1]] - 4.0).abs() < TOL);
    assert!((r.hessian[[1, 0]] - 4.0).abs() < TOL);
    assert!((r.hessian[[1, 1]] - 18.0).abs() < TOL);
}

#[test]
fn dense_hessian_is_symmetric() {
    // f(x, y) = sin(x·y): the Hessian is genuinely dense and symmetric.
    let r = compute_full_hessian(&[0.7_f64, 1.3], |v: &[Jet2Vec]| (&v[0] * &v[1]).sin());
    let n = r.gradient.len();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (r.hessian[[i, j]] - r.hessian[[j, i]]).abs() < TOL,
                "asymmetry at [{i}][{j}]"
            );
        }
    }
}

#[test]
fn dense_hessian_is_positional_no_names() {
    // Inputs are plain &[f64]; readback is by index. (Compile-level proof that
    // the helper no longer requires names / a VarRegistry.)
    let inputs = [1.1_f64, 0.9, 1.7];
    let r = compute_full_hessian(&inputs, |v: &[Jet2Vec]| {
        &(&(&v[0] * &v[1]) * &v[2]) + &v[0].clone().exp()
    });
    assert_eq!(r.gradient.len(), 3);
    assert_eq!(r.hessian.shape(), &[3, 3]);
}
