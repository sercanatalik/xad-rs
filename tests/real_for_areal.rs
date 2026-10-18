use xad_rs::{AReal, Real, Tape};

#[test]
fn areal_satisfies_real_with_ln_exp_identity() {
    let mut tape = Tape::<f64>::new(true);
    tape.activate();

    let mut x = AReal::new(2.0);
    AReal::register_input(std::slice::from_mut(&mut x), &mut tape);

    fn f<R: Real>(x: &R) -> R {
        x.ln().exp()
    }
    let y = f(&x);

    assert!((y.value() - 2.0).abs() < 1e-12);
    Tape::<f64>::deactivate_all();
}

#[test]
fn areal_real_value_matches_inherent_value() {
    let a = AReal::new(3.5_f64);
    assert_eq!(<AReal<f64> as Real>::value(&a), a.value());
}
