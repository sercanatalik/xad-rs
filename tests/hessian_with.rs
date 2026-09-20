//! The Hessian drivers on a tape the caller owns, and the fresh-tape reserve.
//!
//! Mirrors `first_order_drivers.rs` for `compute_hessian_with` /
//! `compute_hessian_k_with`: bit identity with the bare form on a fresh tape
//! and across reuse over different functions, the tape inactive and its
//! capacity retained afterwards, and unwind safety. Plus: `Tape::new` records
//! a swap-sized body with no buffer growth.

use ndarray::Array2;
use xad_rs::math::ad;
use xad_rs::{
    AReal, Jet1, JetK, Tape, TapeStorage, compute_hessian, compute_hessian_k,
    compute_hessian_k_with, compute_hessian_with,
};

fn f1<T: TapeStorage>(v: &[AReal<T>]) -> AReal<T> {
    v[0] * v[0] * v[1] + ad::sin(&v[2])
}
fn f2<T: TapeStorage>(v: &[AReal<T>]) -> AReal<T> {
    ad::exp(&(v[0] * v[1])) / (v[2] + T::one())
}
fn f3<T: TapeStorage>(v: &[AReal<T>]) -> AReal<T> {
    // Larger tape than f1/f2 so reuse in the middle of the sequence grows it.
    let mut acc = AReal::new(T::zero());
    for _ in 0..10 {
        acc = acc + ad::sqrt(&(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
    }
    acc
}

const X: [f64; 3] = [1.3, 0.7, -0.4];

fn eq(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
    assert_eq!(a.dim(), b.dim());
    for ((i, j), x) in a.indexed_iter() {
        assert!(x == &b[[i, j]], "{what} [{i},{j}]: {x} vs {}", b[[i, j]]);
    }
}

#[test]
fn hessian_with_matches_bare_across_reuse() {
    let mut tape = Tape::<Jet1<f64>>::new(true);
    for (name, f) in [("f1", f1 as fn(&[AReal<Jet1<f64>>]) -> AReal<Jet1<f64>>), ("f3", f3), ("f2", f2), ("f1", f1)] {
        let bare = compute_hessian(&X, f);
        let with = compute_hessian_with(&mut tape, &X, f);
        eq(&bare, &with, name);
        assert!(!tape.is_active(), "{name}: tape left active");
    }
}

#[test]
fn hessian_k_with_matches_bare_across_reuse_and_retains_capacity() {
    let mut tape = Tape::<JetK<f64, 2>>::new(true);
    let mut max_mem = 0;
    for (name, f) in [("f1", f1 as fn(&[AReal<JetK<f64, 2>>]) -> AReal<JetK<f64, 2>>), ("f3", f3), ("f2", f2), ("f1", f1)] {
        let bare = compute_hessian_k::<2, _>(&X, f);
        let with = compute_hessian_k_with::<2, _>(&mut tape, &X, f);
        eq(&bare, &with, name);
        assert!(!tape.is_active(), "{name}: tape left active");
        max_mem = max_mem.max(tape.memory());
        assert_eq!(tape.memory(), max_mem, "{name}: retained capacity shrank");
    }
    // And the K engine still equals the Jet1 engine bit for bit through the `_with` forms.
    let mut t1 = Tape::<Jet1<f64>>::new(true);
    eq(&compute_hessian_with(&mut t1, &X, f2), &compute_hessian_k_with::<2, _>(&mut tape, &X, f2), "engines");
}

#[test]
fn a_panicking_function_leaves_no_tape_active() {
    let mut tape = Tape::<Jet1<f64>>::new(true);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute_hessian_with(&mut tape, &X, |v: &[AReal<Jet1<f64>>]| {
            if v[0].value().value() > 1.0 {
                panic!("boom");
            }
            v[0] * v[1]
        })
    }));
    assert!(r.is_err());
    assert!(!tape.is_active());
    // The tape is usable again.
    let h = compute_hessian_with(&mut tape, &X, f1);
    eq(&h, &compute_hessian(&X, f1), "after panic");

    let mut tk = Tape::<JetK<f64, 4>>::new(true);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute_hessian_k_with::<4, _>(&mut tk, &X, |_: &[AReal<JetK<f64, 4>>]| -> AReal<JetK<f64, 4>> { panic!("boom") })
    }));
    assert!(r.is_err());
    assert!(!tk.is_active());
}

#[test]
fn a_fresh_tape_records_a_swap_sized_body_with_no_growth() {
    let mut tape = Tape::<f64>::new(true);
    assert_eq!(tape.memory(), Tape::<f64>::with_capacity(Tape::<f64>::DEFAULT_STATEMENTS, Tape::<f64>::DEFAULT_OPERATIONS).memory());
    let before = tape.memory();
    let _rec = tape.record();
    // 30 inputs, ~215 statements / ~246 operands: the shape of `examples/swap_pricer.rs`.
    let mut r: Vec<AReal<f64>> = (0..30).map(|i| AReal::new(0.01 + 0.002 * i as f64)).collect();
    AReal::register_input(&mut r, &mut tape);
    let mut fix = AReal::new(0.0);
    let mut flt = AReal::new(0.0);
    for (t, x) in r.iter().enumerate() {
        let d = ad::powf(&(x + 1.0), (t + 1) as f64);
        fix += AReal::new(3e5) / &d;
        flt += AReal::new(1e6) / &d;
    }
    let mut v = flt - fix;
    v.register(&mut tape);
    v.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert!(tape.num_statements() <= Tape::<f64>::DEFAULT_STATEMENTS);
    assert!(tape.num_operations() <= Tape::<f64>::DEFAULT_OPERATIONS);
    assert_eq!(tape.memory(), before, "a fresh tape grew on a small body");
}
