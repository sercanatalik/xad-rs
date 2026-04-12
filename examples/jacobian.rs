//! Jacobian - Rust port of XAD's Jacobian sample.
//!
//! Computes
//!     f(x, y, z, w) = [sin(x + y), sin(y + z), cos(z + w), cos(w + x)]
//! and its 4x4 Jacobian matrix using reverse-mode (adjoint) automatic
//! differentiation.
//!
//! Original C++ source:
//!   <https://github.com/auto-differentiation/xad/blob/main/samples/Jacobian>
//!
//! Analytic Jacobian (for cross-checking the AD result):
//!
//!     J = [ cos(x+y)  cos(x+y)      0         0      ]
//!         [   0       cos(y+z)  cos(y+z)      0      ]
//!         [   0         0      -sin(z+w) -sin(z+w)   ]
//!         [-sin(w+x)    0         0      -sin(w+x)   ]

use xad_rs::AReal;
use xad_rs::compute_jacobian_rev;
use xad_rs::math;

fn main() {
    // Input vector: [x, y, z, w]
    let inputs = [1.0_f64, 1.5, 1.3, 1.2];

    // Many-input, many-output function: R^4 -> R^4
    //   f0 = sin(x + y)
    //   f1 = sin(y + z)
    //   f2 = cos(z + w)
    //   f3 = cos(w + x)
    let f = |x: &[AReal<f64>]| -> Vec<AReal<f64>> {
        vec![
            math::ad::sin(&(&x[0] + &x[1])),
            math::ad::sin(&(&x[1] + &x[2])),
            math::ad::cos(&(&x[2] + &x[3])),
            math::ad::cos(&(&x[3] + &x[0])),
        ]
    };

    // Compute the Jacobian using reverse-mode AD (one tape pass per output).
    let jacobian = compute_jacobian_rev(&inputs, f);

    // -------- Output --------
    println!("Jacobian - Rust port of XAD sample");
    println!("==================================");
    println!(
        "Inputs: x={}, y={}, z={}, w={}",
        inputs[0], inputs[1], inputs[2], inputs[3]
    );
    println!();
    println!("f(x, y, z, w) = [sin(x+y), sin(y+z), cos(z+w), cos(w+x)]");
    println!();
    println!("Jacobian (computed via adjoint mode):");
    for row in &jacobian {
        for elem in row {
            print!("{:>12.6} ", elem);
        }
        println!();
    }

    // -------- Cross-check against analytic derivatives --------
    let (x, y, z, w) = (inputs[0], inputs[1], inputs[2], inputs[3]);
    let cxy = (x + y).cos();
    let cyz = (y + z).cos();
    let szw = (z + w).sin();
    let swx = (w + x).sin();

    #[rustfmt::skip]
    let expected = [
        [ cxy,  cxy, 0.0, 0.0],
        [ 0.0,  cyz, cyz, 0.0],
        [ 0.0,  0.0, -szw, -szw],
        [-swx,  0.0, 0.0, -swx],
    ];

    println!();
    println!("Analytic Jacobian (reference):");
    for row in &expected {
        for elem in row {
            print!("{:>12.6} ", elem);
        }
        println!();
    }

    // Max absolute error across all entries
    let mut max_err = 0.0_f64;
    for i in 0..4 {
        for j in 0..4 {
            let err = (jacobian[i][j] - expected[i][j]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    println!();
    println!("Max |AD - analytic| = {:.2e}", max_err);
    assert!(max_err < 1e-12, "Jacobian mismatch vs. analytic: {}", max_err);
}
