//! Tests for the exact forward-over-adjoint Hessian engines.

use xad_rs::{
    compute_full_hessian, compute_hessian, compute_hessian_k, compute_hessian_k_par, math,
    Jet1, Jet2Vec, JetK,
};

/// f(x, y, z) = x²·y + x·sin(z) + y·z²  (generic over the active scalar).
fn f_areal<T: xad_rs::TapeStorage>(v: &[xad_rs::AReal<T>]) -> xad_rs::AReal<T> {
    &(&(&v[0] * &v[0]) * &v[1])
        + &(&(&v[0] * math::ad::sin(&v[2])) + &(&v[1] * &(&v[2] * &v[2])))
}

fn analytic(x: f64, y: f64, z: f64) -> [[f64; 3]; 3] {
    // H_xx=2y, H_xy=2x, H_xz=cos z, H_yy=0, H_yz=2z, H_zz=-x sin z + 2y
    [
        [2.0 * y, 2.0 * x, z.cos()],
        [2.0 * x, 0.0, 2.0 * z],
        [z.cos(), 2.0 * z, -x * z.sin() + 2.0 * y],
    ]
}

#[test]
fn matches_analytic_and_is_symmetric() {
    let (x, y, z) = (1.0_f64, 2.0, 0.5);
    let h = compute_hessian(&[x, y, z], f_areal);
    let a = analytic(x, y, z);
    for i in 0..3 {
        for j in 0..3 {
            assert!((h[[i, j]] - a[i][j]).abs() < 1e-10, "H[{i}][{j}]");
            assert!((h[[i, j]] - h[[j, i]]).abs() < 1e-12, "asymmetry [{i}][{j}]");
        }
    }
}

#[test]
fn exact_on_quadratic() {
    // q(x, y) = 3x² + 2xy + 5y²  ->  H = [[6, 2], [2, 10]] everywhere, constant.
    // Finite differences would carry a step-size error term; exact AD must
    // reproduce the constant matrix to full f64 precision.
    let h = compute_hessian(&[0.37_f64, -1.9], |v| {
        let q1 = (&v[0] * &v[0]) * Jet1::constant(3.0);
        let q2 = (&v[0] * &v[1]) * Jet1::constant(2.0);
        let q3 = (&v[1] * &v[1]) * Jet1::constant(5.0);
        q1 + q2 + q3
    });
    assert_eq!(h[[0, 0]], 6.0);
    assert_eq!(h[[0, 1]], 2.0);
    assert_eq!(h[[1, 0]], 2.0);
    assert_eq!(h[[1, 1]], 10.0);
}

#[test]
fn jetk_matches_jet1_engine_bit_exact() {
    // Every JetK tangent lane evolves exactly as the corresponding Jet1
    // tangent, so the K-wide engine must reproduce compute_hessian
    // bit-for-bit — including partial final blocks (n = 3 with K = 2,
    // n = 7 with K = 4/8).
    let x3 = [1.0_f64, 2.0, 0.5];
    let h_ref = compute_hessian(&x3, f_areal);
    assert_eq!(h_ref, compute_hessian_k::<2, _>(&x3, f_areal));
    assert_eq!(h_ref, compute_hessian_k::<4, _>(&x3, f_areal));

    let x7: Vec<f64> = (0..7).map(|i| 0.8 + i as f64 * 0.1).collect();
    let kernel = |v: &[xad_rs::AReal<JetK<f64, 4>>]| {
        let n = v.len();
        let mut acc = v[0];
        for r in 0..40 {
            let a = &v[r % n];
            let b = &v[(r + 1) % n];
            let t = &(a * b) * JetK::constant(1e-2);
            acc = &acc + &(math::ad::exp(&t) * b);
        }
        acc
    };
    let kernel_jet1 = |v: &[xad_rs::AReal<Jet1<f64>>]| {
        let n = v.len();
        let mut acc = v[0];
        for r in 0..40 {
            let a = &v[r % n];
            let b = &v[(r + 1) % n];
            let t = &(a * b) * Jet1::constant(1e-2);
            acc = &acc + &(math::ad::exp(&t) * b);
        }
        acc
    };
    assert_eq!(
        compute_hessian(&x7, kernel_jet1),
        compute_hessian_k::<4, _>(&x7, kernel)
    );
}

#[test]
fn jetk_par_matches_serial_across_the_crossover() {
    // Small problem: exercises the serial-finish fallback inside
    // compute_hessian_k_par.
    let x3 = [1.0_f64, 2.0, 0.5];
    assert_eq!(
        compute_hessian_k::<2, _>(&x3, f_areal),
        compute_hessian_k_par::<2, _>(&x3, f_areal)
    );

    // 32 inputs × 2000-op kernel: past the crossover, exercises the
    // genuinely parallel path (per-worker tapes).
    let x32: Vec<f64> = (0..32).map(|i| 1.0 + i as f64 * 0.02).collect();
    let kernel = |v: &[xad_rs::AReal<JetK<f64, 4>>]| {
        let n = v.len();
        let mut acc = v[0];
        for r in 0..2000 {
            let a = &v[r % n];
            let b = &v[(r + 1) % n];
            let t = &(a * b) * JetK::constant(1e-3);
            acc = &acc + &(math::ad::exp(&t) * b);
        }
        acc
    };
    assert_eq!(
        compute_hessian_k::<4, _>(&x32, kernel),
        compute_hessian_k_par::<4, _>(&x32, kernel)
    );
}

#[test]
fn agrees_with_jet2vec_dense_path() {
    let (x, y, z) = (1.0_f64, 2.0, 0.5);
    let h_foa = compute_hessian(&[x, y, z], f_areal);
    let dense = compute_full_hessian(&[x, y, z], |v: &[Jet2Vec]| {
        &(&(&v[0] * &v[0]) * &v[1])
            + &(&(&v[0] * &v[2].clone().sin()) + &(&v[1] * &(&v[2] * &v[2])))
    });
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (h_foa[[i, j]] - dense.hessian[[i, j]]).abs() < 1e-10,
                "FOA vs Jet2Vec disagree at [{i}][{j}]"
            );
        }
    }
}
