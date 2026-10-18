//! Hot-path benchmark for the reverse-mode tape: recording, scalar sweep,
//! vector sweep, and end-to-end gradient drivers. Hand-rolled harness
//! (no criterion): `R` repeats of `n` iterations, reporting the minimum and
//! median ns per iteration. Run with `cargo bench --bench tape_bench`.
use std::hint::black_box;
use std::time::Instant;
use xad_rs::{
    AReal, Real, Tape, compute_gradient_rev, compute_gradient_rev_with, compute_jacobian_rev_with,
    math,
};

const R: usize = 15;

fn bench(name: &str, n: usize, mut f: impl FnMut()) {
    for _ in 0..n / 4 {
        f();
    } // warm-up
    let mut samples = Vec::with_capacity(R);
    for _ in 0..R {
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        samples.push(t.elapsed().as_nanos() as f64 / n as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{name:<34} min {:>9.1} ns   median {:>9.1} ns",
        samples[0],
        samples[R / 2]
    );
}

/// 2000-op chain `acc = acc * x + c`, two binary ops per step.
fn chain<R: Real>(x: &R, c: &R) -> R {
    let mut acc = x.clone();
    for _ in 0..1000 {
        acc = acc * x.clone() + c.clone();
    }
    acc
}

/// 30-input swap-like body: discounted fixed leg against floating leg.
fn swap<R: Real>(v: &[R]) -> R {
    let (rates, dfs) = v.split_at(15);
    let mut pv = R::zero();
    let mut prev = R::one();
    for i in 0..15 {
        let tau = 0.5_f64;
        let fwd = (prev.clone() / dfs[i].clone() - R::one()) * R::from(1.0 / tau);
        pv = pv + (fwd - rates[i].clone()) * dfs[i].clone() * R::from(tau);
        prev = dfs[i].clone();
    }
    pv
}

/// `Σ_{k=0..8} t^k` by `powi` under a `max` floor, 100 times: the
/// polynomial-curve shape the zero-base `powi` guard and `f64::max` touch.
fn poly<R: Real>(t: &R) -> R {
    let mut acc = R::zero();
    for _ in 0..100 {
        for k in 0..8 {
            acc = acc + t.powi(k);
        }
        acc = acc.max(&R::zero());
    }
    acc
}

/// 6-input Garman–Kohlhagen call.
fn gk<R: Real>(v: &[R]) -> R {
    let (s, k, rd, rf, vol, t) = (&v[0], &v[1], &v[2], &v[3], &v[4], &v[5]);
    let sqrt_t = t.sqrt();
    let d1 = ((s.clone() / k.clone()).ln()
        + (rd.clone() - rf.clone() + vol.clone() * vol.clone() * R::from(0.5_f64)) * t.clone())
        / (vol.clone() * sqrt_t.clone());
    let d2 = d1.clone() - vol.clone() * sqrt_t;
    s.clone() * (-rf.clone() * t.clone()).exp() * d1.norm_cdf()
        - k.clone() * (-rd.clone() * t.clone()).exp() * d2.norm_cdf()
}

fn main() {
    let swap_in: Vec<f64> = (0..15)
        .map(|i| 0.02 + 0.001 * i as f64)
        .chain((0..15).map(|i| 0.99 - 0.01 * i as f64))
        .collect();
    let gk_in = [1.10_f64, 1.12, 0.05, 0.03, 0.12, 1.0];

    // --- recording only: reused tape, 2000-op chain ---
    let mut tape = Tape::<f64>::with_capacity(4096, 8192);
    bench("record/chain_2000ops_reused", 2000, || {
        let _rec = tape.record();
        let x = AReal::input(black_box(1.0001), &mut tape);
        let c = AReal::input(black_box(0.5), &mut tape);
        black_box(chain(&x, &c));
    });

    bench("record/poly_powi_max_900ops", 5000, || {
        let _rec = tape.record();
        let t = AReal::input(black_box(0.7), &mut tape);
        black_box(poly(&t));
    });
    bench("passive/poly_powi_max f64", 20000, || {
        black_box(poly(&black_box(0.7_f64)));
    });

    // --- scalar sweep only: record once, sweep repeatedly ---
    let mut tape = Tape::<f64>::with_capacity(4096, 8192);
    let (x, out) = {
        let _rec = tape.record();
        let x = AReal::input(1.0001, &mut tape);
        let c = AReal::input(0.5, &mut tape);
        let out = chain(&x, &c);
        (x, out)
    };
    bench("sweep/scalar_2000ops", 2000, || {
        tape.clear_derivatives();
        out.set_adjoint(&mut tape, 1.0);
        tape.compute_adjoints();
        black_box(x.adjoint(&tape));
    });

    // --- vector sweep, 8 directions ---
    let n_vars = tape.num_variables() as usize;
    let mut derivs = vec![0.0; n_vars * 8];
    bench("sweep/vector_8dir_2000ops", 500, || {
        derivs.iter_mut().for_each(|d| *d = 0.0);
        for d in 0..8 {
            derivs[out.slot() as usize * 8 + d] = 1.0;
        }
        tape.compute_adjoints_vector(8, &mut derivs);
        black_box(derivs[x.slot() as usize * 8]);
    });
    drop(tape);

    // --- end to end ---
    let mut tape = Tape::<f64>::new(true);
    bench("e2e/gradient_rev_with swap30", 20000, || {
        black_box(compute_gradient_rev_with(
            &mut tape,
            black_box(&swap_in),
            |v| swap(v),
        ));
    });
    bench("e2e/gradient_rev_with gk6", 50000, || {
        black_box(compute_gradient_rev_with(
            &mut tape,
            black_box(&gk_in),
            |v| gk(v),
        ));
    });
    bench("e2e/gradient_rev fresh-tape gk6", 50000, || {
        black_box(compute_gradient_rev(black_box(&gk_in), |v| gk(v)));
    });
    bench("e2e/jacobian_rev_with 4x4", 50000, || {
        black_box(compute_jacobian_rev_with(
            &mut tape,
            black_box(&[1.0, 1.5, 1.3, 1.2]),
            |x: &[AReal<f64>]| {
                vec![
                    math::ad::sin(&(&x[0] + &x[1])),
                    math::ad::sin(&(&x[1] + &x[2])),
                    math::ad::cos(&(&x[2] + &x[3])),
                    math::ad::cos(&(&x[3] + &x[0])),
                ]
            },
        ));
    });
    bench("e2e/fused weighted_sum 64", 50000, || {
        black_box(compute_gradient_rev_with(
            &mut tape,
            black_box(&swap_in),
            |v| {
                let ws: Vec<f64> = (0..30).map(|i| 0.5 + i as f64 * 0.01).collect();
                R::weighted_sum(&ws, v)
            },
        ));
    });
}
type R = AReal<f64>;
