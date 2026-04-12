//! Integration tests for `compute_named_jacobian` (LBLJ-01..03).

#![allow(clippy::needless_range_loop)]

use xad_rs::AReal;
use xad_rs::compute_jacobian_rev;
use xad_rs::compute_named_jacobian;
use xad_rs::math;

#[test]
fn test_named_jacobian_2x2_shape_and_values() {
    // f(x, y) = [x · y, x + y] at (3, 5).
    // J = [[y, x], [1, 1]] = [[5, 3], [1, 1]]
    let inputs = vec![("x".to_string(), 3.0_f64), ("y".to_string(), 5.0)];
    let outputs = vec!["prod".to_string(), "sum".to_string()];

    let j = compute_named_jacobian(&inputs, &outputs, |v: &[AReal<f64>]| {
        vec![&v[0] * &v[1], &v[0] + &v[1]]
    });

    assert_eq!(j.matrix.shape(), &[2, 2]);
    assert!((j.matrix[[0, 0]] - 5.0).abs() < 1e-12);
    assert!((j.matrix[[0, 1]] - 3.0).abs() < 1e-12);
    assert!((j.matrix[[1, 0]] - 1.0).abs() < 1e-12);
    assert!((j.matrix[[1, 1]] - 1.0).abs() < 1e-12);

    assert_eq!(j.rows, vec!["prod".to_string(), "sum".to_string()]);
    assert_eq!(j.cols.len(), 2);
    assert_eq!(j.cols.name(0), Some("x"));
    assert_eq!(j.cols.name(1), Some("y"));
}

#[test]
fn test_named_jacobian_4x4_matches_examples_jacobian() {
    // Reproduces the case from `examples/jacobian.rs` with labels.
    // f(x, y, z, w) = [sin(x+y), sin(y+z), cos(z+w), cos(w+x)]
    let inputs = vec![
        ("x".to_string(), 1.0_f64),
        ("y".to_string(), 1.5),
        ("z".to_string(), 1.3),
        ("w".to_string(), 1.2),
    ];
    let outputs = vec![
        "f0".to_string(),
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
    ];

    let f = |v: &[AReal<f64>]| -> Vec<AReal<f64>> {
        vec![
            math::ad::sin(&(&v[0] + &v[1])),
            math::ad::sin(&(&v[1] + &v[2])),
            math::ad::cos(&(&v[2] + &v[3])),
            math::ad::cos(&(&v[3] + &v[0])),
        ]
    };

    // Cross-check 1: named vs positional helper (the proper named-vs-positional comparison).
    let named = compute_named_jacobian(&inputs, &outputs, f);
    let positional = compute_jacobian_rev(&[1.0_f64, 1.5, 1.3, 1.2], f);

    for i in 0..4 {
        for j in 0..4 {
            let diff = (named.matrix[[i, j]] - positional[i][j]).abs();
            assert!(
                diff < 1e-12,
                "named[{},{}] = {} vs positional[{},{}] = {} (diff = {})",
                i, j, named.matrix[[i, j]],
                i, j, positional[i][j],
                diff
            );
        }
    }

    // Cross-check 2: named vs analytic (mirrors examples/jacobian.rs lines 60-74).
    let (x, y, z, w) = (1.0_f64, 1.5, 1.3, 1.2);
    let cxy = (x + y).cos();
    let cyz = (y + z).cos();
    let szw = (z + w).sin();
    let swx = (w + x).sin();
    let expected = [
        [ cxy,  cxy, 0.0, 0.0],
        [ 0.0,  cyz, cyz, 0.0],
        [ 0.0,  0.0, -szw, -szw],
        [-swx,  0.0, 0.0, -swx],
    ];
    for i in 0..4 {
        for j in 0..4 {
            assert!((named.matrix[[i, j]] - expected[i][j]).abs() < 1e-12);
        }
    }

    // Verify row + column labels are preserved.
    assert_eq!(named.rows, vec!["f0", "f1", "f2", "f3"]);
    assert_eq!(named.cols.name(0), Some("x"));
    assert_eq!(named.cols.name(1), Some("y"));
    assert_eq!(named.cols.name(2), Some("z"));
    assert_eq!(named.cols.name(3), Some("w"));
}
