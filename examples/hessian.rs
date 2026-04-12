//! Hessian - Rust port of XAD's Hessian sample.
//!
//! Computes
//!     f(x, y, z, w) = sin(x*y) - cos(y*z) - sin(z*w) - cos(w*x)
//! and its 4x4 Hessian matrix (second-order partial derivatives).
//!
//! Original C++ source:
//!   <https://github.com/auto-differentiation/xad/blob/main/samples/Hessian>
//!
//! The C++ sample uses forward-over-adjoint (fwd_adj) mode for efficient
//! exact second derivatives. This Rust port uses the library's
//! `compute_hessian` helper, which approximates the Hessian by perturbing
//! the reverse-mode gradient (central trade-off: simpler, but only as
//! accurate as the finite-difference step allows).
//!
//! Analytic Hessian (symbolic reference for verification):
//!
//!   f = sin(xy) - cos(yz) - sin(zw) - cos(wx)
//!
//!   Let  A = xy, B = yz, C = zw, D = wx
//!
//!   df/dx = y cos(A) + w sin(D)
//!   df/dy = x cos(A) + z sin(B)
//!   df/dz = y sin(B) - w cos(C)
//!   df/dw = x sin(D) - z cos(C)
//!
//!   d²f/dx²  = -y² sin(A) + w² cos(D)
//!   d²f/dy²  = -x² sin(A) + z² cos(B)
//!   d²f/dz²  =  y² cos(B) + w² sin(C)
//!   d²f/dw²  =  x² cos(D) + z² sin(C)
//!
//!   d²f/dxdy = cos(A) - xy sin(A)             ( = d²f/dydx )
//!   d²f/dxdz = 0
//!   d²f/dxdw = sin(D) + xw cos(D)             ( = d²f/dwdx )
//!   d²f/dydz = sin(B) + yz cos(B)             ( = d²f/dzdy )
//!   d²f/dydw = 0
//!   d²f/dzdw = -cos(C) + zw sin(C)            ( = d²f/dwdz )

// The max-error and symmetry loops are naturally paired-index accesses on
// both `hessian` and `expected`, which do not translate cleanly to iterators.
#![allow(clippy::needless_range_loop)]

use xad_rs::AReal;
use xad_rs::compute_hessian;
use xad_rs::math;

fn main() {
    // Input vector: [x, y, z, w]
    let inputs = [1.0_f64, 1.5, 1.3, 1.2];

    // Scalar-valued function of 4 inputs: R^4 -> R
    let f = |x: &[AReal<f64>]| -> AReal<f64> {
        math::ad::sin(&(&x[0] * &x[1]))
            - math::ad::cos(&(&x[1] * &x[2]))
            - math::ad::sin(&(&x[2] * &x[3]))
            - math::ad::cos(&(&x[3] * &x[0]))
    };

    // Compute the Hessian. (Symmetric 4x4 matrix.)
    let hessian = compute_hessian(&inputs, f);

    // -------- Output --------
    println!("Hessian - Rust port of XAD sample");
    println!("=================================");
    println!(
        "Inputs: x={}, y={}, z={}, w={}",
        inputs[0], inputs[1], inputs[2], inputs[3]
    );
    println!();
    println!("f(x, y, z, w) = sin(x*y) - cos(y*z) - sin(z*w) - cos(w*x)");
    println!();
    println!("Hessian (computed via reverse-mode AD + finite diff):");
    for row in &hessian {
        for elem in row {
            print!("{:>12.6} ", elem);
        }
        println!();
    }

    // -------- Analytic cross-check --------
    let (x, y, z, w) = (inputs[0], inputs[1], inputs[2], inputs[3]);
    let a = x * y;
    let b = y * z;
    let c = z * w;
    let d = w * x;

    let (sa, ca) = (a.sin(), a.cos());
    let (sb, cb) = (b.sin(), b.cos());
    let (sc, cc) = (c.sin(), c.cos());
    let (sd, cd) = (d.sin(), d.cos());

    let hxx = -y * y * sa + w * w * cd;
    let hyy = -x * x * sa + z * z * cb;
    let hzz = y * y * cb + w * w * sc;
    let hww = x * x * cd + z * z * sc;

    let hxy = ca - x * y * sa;
    let hxz = 0.0;
    let hxw = sd + x * w * cd;
    let hyz = sb + y * z * cb;
    let hyw = 0.0;
    let hzw = -cc + z * w * sc;

    #[rustfmt::skip]
    let expected = [
        [hxx, hxy, hxz, hxw],
        [hxy, hyy, hyz, hyw],
        [hxz, hyz, hzz, hzw],
        [hxw, hyw, hzw, hww],
    ];

    println!();
    println!("Analytic Hessian (reference):");
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
            let err = (hessian[i][j] - expected[i][j]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    println!();
    println!("Max |AD - analytic| = {:.2e}", max_err);

    // The Rust library's compute_hessian uses finite differences on the
    // reverse-mode gradient (eps = 1e-7), so accuracy is limited to ~1e-6.
    assert!(
        max_err < 1e-5,
        "Hessian mismatch vs. analytic: {}",
        max_err
    );

    // Verify symmetry of the computed Hessian.
    let mut max_asym = 0.0_f64;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let asym = (hessian[i][j] - hessian[j][i]).abs();
            if asym > max_asym {
                max_asym = asym;
            }
        }
    }
    println!("Max |H[i,j] - H[j,i]| (symmetry)    = {:.2e}", max_asym);
}
