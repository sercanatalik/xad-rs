//! Hessian - exact 4x4 Hessian in a single forward pass, via `Jet2Vec`.
//!
//! Computes
//!     f(x, y, z, w) = sin(x*y) - cos(y*z) - sin(z*w) - cos(w*x)
//! and its 4x4 Hessian matrix (second-order partial derivatives).
//!
//! Forward-over-adjoint is the usual route to exact second derivatives, and
//! `xad-rs` offers it as `compute_hessian` / `compute_hessian_k`
//! (demonstrated in `fwd_adj_second_order.rs`). This example instead shows
//! the *other* exact route: `compute_full_hessian`
//! computes the dense Hessian in a **single** `Jet2Vec` forward pass, with
//! value, gradient, and Hessian propagating in lock-step at machine
//! precision (no finite differences, no truncation error).
//!
//! The two routes are complementary, and the crossover is roughly
//! `n ~ 50`: `Jet2Vec` costs `O(n²)` per operation but only one pass, while
//! the forward-over-adjoint engines cost `ceil(n/K)` passes over a tape. At
//! the `n = 4` here, one forward pass is clearly the cheaper shape.
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

use xad_rs::Jet2Vec;
use xad_rs::compute_full_hessian;

fn main() {
    // Input vector: [x, y, z, w]. Names are applied at the call site for
    // display; the helper itself is positional.
    let names = ["x", "y", "z", "w"];
    let input_values: [f64; 4] = [1.0, 1.5, 1.3, 1.2];

    // Scalar-valued function of 4 inputs: R^4 -> R, computed on `Jet2Vec`
    // so a single forward pass produces value, full gradient, and full
    // dense Hessian at machine precision.
    let f = |v: &[Jet2Vec]| -> Jet2Vec {
        let xy = &v[0] * &v[1];
        let yz = &v[1] * &v[2];
        let zw = &v[2] * &v[3];
        let wx = &v[3] * &v[0];
        let term1 = xy.sin();
        let term2 = yz.cos();
        let term3 = zw.sin();
        let term4 = wx.cos();
        &(&(&term1 - &term2) - &term3) - &term4
    };

    // One forward pass yields value, gradient, and exact Hessian.
    let result = compute_full_hessian(&input_values, f);
    let hessian = &result.hessian;

    // -------- Output --------
    println!("Hessian - exact 4x4 Hessian in one forward pass, via Jet2Vec");
    println!("======================================================");
    println!(
        "Inputs: x={}, y={}, z={}, w={}",
        input_values[0], input_values[1], input_values[2], input_values[3]
    );
    println!();
    println!("f(x, y, z, w) = sin(x*y) - cos(y*z) - sin(z*w) - cos(w*x)");
    println!();
    println!("Hessian (computed via Jet2Vec forward pass — exact):");
    for i in 0..4 {
        for j in 0..4 {
            print!("{:>12.6} ", hessian[[i, j]]);
        }
        println!();
    }

    // -------- Analytic cross-check --------
    let (x, y, z, w) = (input_values[0], input_values[1], input_values[2], input_values[3]);
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
            let err = (hessian[[i, j]] - expected[i][j]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    println!();
    println!("Max |AD - analytic| = {:.2e}", max_err);

    // Jet2Vec propagates value, grad, and Hessian at machine precision
    // — agreement is at the few-ulps level (~1e-15), nothing like the
    // ~1e-6 floor of finite-difference reverse-mode.
    assert!(
        max_err < 1e-12,
        "Hessian mismatch vs. analytic: {}",
        max_err
    );

    // Verify symmetry of the computed Hessian.
    let mut max_asym = 0.0_f64;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let asym = (hessian[[i, j]] - hessian[[j, i]]).abs();
            if asym > max_asym {
                max_asym = asym;
            }
        }
    }
    println!("Max |H[i,j] - H[j,i]| (symmetry)    = {:.2e}", max_asym);

    // -------- Show gradient + value too (free with Jet2Vec) --------
    println!();
    println!("Function value: {:.10}", result.value);
    println!("Gradient (free side-output of compute_full_hessian):");
    for (name, g) in names.iter().zip(result.gradient.iter()) {
        println!("  df/d{} = {:>14.10}", name, g);
    }
}
