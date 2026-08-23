//! Tests for reverse vector mode: `Tape::compute_adjoints_vector` and the
//! one-sweep `compute_jacobian_rev` built on it.

use xad_rs::{compute_jacobian_rev, AReal, Real, Tape};

const TOL: f64 = 1e-12;

/// f: R^3 -> R^2:  f1 = x0*x1 + x2,  f2 = sin(x0) - x1*x2
fn f(v: &[AReal<f64>]) -> Vec<AReal<f64>> {
    let f1 = &(&v[0] * &v[1]) + &v[2];
    let s = v[0].sin();
    let xy = &v[1] * &v[2];
    let f2 = &s - &xy;
    vec![f1, f2]
}

fn analytic(x: &[f64]) -> [[f64; 3]; 2] {
    [
        [x[1], x[0], 1.0],
        [x[0].cos(), -x[2], -x[1]],
    ]
}

#[test]
fn jacobian_one_sweep_matches_analytic() {
    let x = [0.7_f64, 1.3, 2.1];
    let jac = compute_jacobian_rev(&x, f);
    let a = analytic(&x);
    assert_eq!(jac.nrows(), 2);
    for i in 0..2 {
        for j in 0..3 {
            assert!((jac[[i, j]] - a[i][j]).abs() < TOL, "J[{i}][{j}]");
        }
    }
}

#[test]
fn vector_sweep_equals_per_row_sweeps() {
    let x = [0.4_f64, 1.1, 0.9];
    let full = compute_jacobian_rev(&x, f);
    let row0 = compute_jacobian_rev(&x, |v| vec![&(&v[0] * &v[1]) + &v[2]]);
    let row1 = compute_jacobian_rev(&x, |v| {
        let s = v[0].sin();
        let xy = &v[1] * &v[2];
        vec![&s - &xy]
    });
    for j in 0..3 {
        assert!((full[[0, j]] - row0[[0, j]]).abs() < TOL);
        assert!((full[[1, j]] - row1[[0, j]]).abs() < TOL);
    }
}

#[test]
fn primitive_direction_isolation() {
    // Seed only direction 1 (output 0); directions 0 and 2 stay exactly zero.
    let x = [1.0_f64, 2.0, 3.0];
    let mut tape = Tape::<f64>::new(true);
    tape.activate();
    let mut inputs: Vec<AReal<f64>> = x.iter().map(|&v| AReal::new(v)).collect();
    AReal::register_input(&mut inputs, &mut tape);
    let mut outputs = f(&inputs);
    AReal::register_output(&mut outputs, &mut tape);

    let n_dir = 3;
    let num_vars = tape.num_variables() as usize;
    let mut derivs = vec![0.0_f64; num_vars * n_dir];
    derivs[outputs[0].slot() as usize * n_dir + 1] = 1.0;
    tape.compute_adjoints_vector(n_dir, &mut derivs);

    let a = analytic(&x);
    for (j, inp) in inputs.iter().enumerate() {
        let row = inp.slot() as usize * n_dir;
        assert_eq!(derivs[row], 0.0);       // direction 0 unseeded
        assert_eq!(derivs[row + 2], 0.0);   // direction 2 unseeded
        assert!((derivs[row + 1] - a[0][j]).abs() < TOL); // direction 1 = df1
    }
    Tape::<f64>::deactivate_all();
}
