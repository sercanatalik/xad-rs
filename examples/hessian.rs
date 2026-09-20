//! Hessian - every exact route to a full Hessian, cross-checked and timed.
//!
//! Computes
//!     f(x, y, z, w) = sin(x*y) - cos(y*z) - sin(z*w) - cos(w*x)
//! and its 4x4 Hessian matrix (second-order partial derivatives), first via
//! `compute_full_hessian` — the dense `Jet2Vec` forward pass, in which value,
//! gradient, and Hessian propagate in lock-step through **one** evaluation
//! with no tape — verified against the analytic Hessian below.
//!
//! It then times every exact route the crate offers on that body and on a
//! 12-input body, and asserts they agree to 1e-12:
//!
//!   - `compute_full_hessian`: one `Jet2Vec` pass, `O(n²)` per operation.
//!   - `compute_hessian`: `n` forward-over-adjoint passes on `Tape<Jet1<f64>>`.
//!   - `compute_hessian_k::<4>` / `::<8>`: `⌈n/K⌉` passes on `Tape<JetK<f64, K>>`.
//!
//! This is the measurement the README's second-order guidance cites. The cost
//! model says `Jet2Vec` should win for small `n`; the constants say otherwise:
//! a `Jet2Vec` operation allocates and walks `n(n+1)/2` Hessian cells, while a
//! K-lane engine operation is one tape record plus a lane-vectorised sweep
//! step. `Jet2Vec` remains the route that needs no tape and returns value,
//! gradient, and Hessian from a single evaluation.
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

use std::hint::black_box;
use std::time::{Duration, Instant};

use ndarray::Array2;
use xad_rs::math::ad;
use xad_rs::{AReal, Jet2Vec, TapeStorage, compute_full_hessian, compute_hessian, compute_hessian_k};

/// Timed iterations per measurement.
const N_TRIALS: usize = 2_000;
/// Measurements per route; the minimum average is reported.
const REPEATS: usize = 5;

// ---- Body 1: the 4-input function above, spelled for both active families ----
// (`Jet2Vec` has no `Real` impl — `From<f64>` cannot know the input dimension —
// so each body is written once for `&[Jet2Vec]` and once for `&[AReal<T>]`; the
// latter serves the `Jet1` engine and every `JetK<K>` engine.)

fn f4_vec(v: &[Jet2Vec]) -> Jet2Vec {
    let xy = &v[0] * &v[1];
    let yz = &v[1] * &v[2];
    let zw = &v[2] * &v[3];
    let wx = &v[3] * &v[0];
    &(&(&xy.sin() - &yz.cos()) - &zw.sin()) - &wx.cos()
}

fn f4_areal<T: TapeStorage>(v: &[AReal<T>]) -> AReal<T> {
    let xy = &v[0] * &v[1];
    let yz = &v[1] * &v[2];
    let zw = &v[2] * &v[3];
    let wx = &v[3] * &v[0];
    ad::sin(&xy) - ad::cos(&yz) - ad::sin(&zw) - ad::cos(&wx)
}

// ---- Body 2: 12 inputs, Σᵢ sin(xᵢ·xᵢ₊₁)·xᵢ — a chain with every pair coupled ----

fn f12_vec(v: &[Jet2Vec]) -> Jet2Vec {
    let n = v.len();
    let mut acc = Jet2Vec::constant(0.0, n);
    for i in 0..n - 1 {
        acc = &acc + &(&(&v[i] * &v[i + 1]).sin() * &v[i]);
    }
    acc
}

fn f12_areal<T: TapeStorage>(v: &[AReal<T>]) -> AReal<T> {
    let mut acc = AReal::new(T::zero());
    for i in 0..v.len() - 1 {
        acc = acc + ad::sin(&(&v[i] * &v[i + 1])) * &v[i];
    }
    acc
}

struct Route {
    name: &'static str,
    hessian: Array2<f64>,
    per_call: Duration,
}

/// Runs `f` `N_TRIALS` times, `REPEATS` times over, keeping the fastest average.
fn measure(name: &'static str, mut f: impl FnMut() -> Array2<f64>) -> Route {
    let mut best = Duration::MAX;
    let mut last = Array2::zeros((0, 0));
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..N_TRIALS {
            last = black_box(f());
        }
        best = best.min(start.elapsed() / N_TRIALS as u32);
    }
    Route { name, hessian: last, per_call: best }
}

fn routes(x: &[f64], vec_body: fn(&[Jet2Vec]) -> Jet2Vec, areal_body: &'static dyn Fn(&[AReal<xad_rs::Jet1<f64>>]) -> AReal<xad_rs::Jet1<f64>>, k4_body: &'static dyn Fn(&[AReal<xad_rs::JetK<f64, 4>>]) -> AReal<xad_rs::JetK<f64, 4>>, k8_body: &'static dyn Fn(&[AReal<xad_rs::JetK<f64, 8>>]) -> AReal<xad_rs::JetK<f64, 8>>) -> Vec<Route> {
    vec![
        measure("compute_full_hessian (Jet2Vec, 1 pass)", || compute_full_hessian(black_box(x), vec_body).hessian),
        measure("compute_hessian (Jet1 engine, n passes)", || compute_hessian(black_box(x), areal_body)),
        measure("compute_hessian_k::<4> (⌈n/4⌉ passes)", || compute_hessian_k::<4, _>(black_box(x), k4_body)),
        measure("compute_hessian_k::<8> (⌈n/8⌉ passes)", || compute_hessian_k::<8, _>(black_box(x), k8_body)),
    ]
}

fn check_and_report(title: &str, n: usize, routes: &[Route]) {
    let reference = &routes[0].hessian;
    for r in routes {
        let mut max_diff = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                max_diff = max_diff.max((r.hessian[[i, j]] - reference[[i, j]]).abs());
            }
        }
        assert!(max_diff < 1e-12, "{title}: {} disagrees with Jet2Vec by {max_diff:.2e}", r.name);
    }
    println!();
    println!("{title}  (n = {n}, min of {REPEATS} averages over {N_TRIALS} calls; all routes agree to 1e-12)");
    println!("  {:<44} {:>12} {:>12}", "route", "per call", "vs Jet2Vec");
    println!("  {:-<44} {:->12} {:->12}", "", "", "");
    let base = routes[0].per_call.as_secs_f64();
    for r in routes {
        println!("  {:<44} {:>12.3?} {:>11.2}×", r.name, r.per_call, base / r.per_call.as_secs_f64());
    }
}

fn main() {
    // Input vector: [x, y, z, w]. Names are applied at the call site for
    // display; the helper itself is positional.
    let names = ["x", "y", "z", "w"];
    let input_values: [f64; 4] = [1.0, 1.5, 1.3, 1.2];

    // Scalar-valued function of 4 inputs: R^4 -> R, computed on `Jet2Vec`
    // so a single forward pass produces value, full gradient, and full
    // dense Hessian at machine precision.
    // One forward pass yields value, gradient, and exact Hessian.
    let result = compute_full_hessian(&input_values, f4_vec);
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
    // — agreement is at the few-ulps level (~1e-15).
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

    // -------- Every exact route, timed and cross-checked --------
    println!();
    println!("Every exact route to the full Hessian, timed");
    println!("============================================");
    let r4 = routes(&input_values, f4_vec, &f4_areal, &f4_areal, &f4_areal);
    check_and_report("4-input body (the function above)", 4, &r4);

    let x12: Vec<f64> = (0..12).map(|i| 0.5 + 0.1 * i as f64).collect();
    let r12 = routes(&x12, f12_vec, &f12_areal, &f12_areal, &f12_areal);
    check_and_report("12-input body Σ sin(xᵢ·xᵢ₊₁)·xᵢ", 12, &r12);
    println!();
    println!("Reading the table: `vs Jet2Vec` is the one-pass Jet2Vec time divided by");
    println!("the route's time, so > 1× means the route beats the dense forward pass.");
}
