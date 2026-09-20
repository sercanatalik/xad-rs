//! Unit operands: `+`, `−`, negation, `max`/`min`, and the fused `sum`
//! record their `±1` multipliers as flag bits in the operand slot word, and
//! both sweeps accumulate them by add / subtract. `d ± a` is the same `f64`
//! as `d + (±1)·a`, so nothing here needs a tolerance.

use xad_rs::math::ad;
use xad_rs::{AReal, JetK, Tape, TapeStorage};

/// Every unit-producing operator in every operand position, mixed with two
/// non-unit ops so the sweep's branch alternates. Gradient is exactly
/// `(2, 2, 4)` and the value `14` at `(1.5, 0.5, 2)`.
fn body<T: TapeStorage>(x: &AReal<T>, y: &AReal<T>, z: &AReal<T>, two: T, three: T, one: T) -> AReal<T> {
    let a = ad::max(x, y); //       1 op
    let b = ad::min(x, y); //       1
    let c = x - y; //               2
    let d = -&c; //                 1
    let e = x + two; //             1
    let g = AReal::new(three) - z; // 1 (passive lhs: Sub<T> on the value, unary on z)
    let h = ad::sum(&[*x, *y, *z]); // 3
    let i = y - one; //             1
    let k = z * z; //               2 (non-unit)
    let s = a + b; //               2
    let s = s + d; //               2
    let s = s + e; //               2
    let s = s + g; //               2
    let s = s + h; //               2
    let s = s - i; //               2
    s + k //                        2
}

#[test]
fn gradient_is_the_closed_form_and_counts_are_unchanged() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let x = AReal::input(1.5, &mut tape);
    let y = AReal::input(0.5, &mut tape);
    let z = AReal::input(2.0, &mut tape);
    let mut out = body(&x, &y, &z, 2.0, 3.0, 1.0);
    out.register(&mut tape);
    assert_eq!(out.value(), 14.0);
    // 3 inputs + 16 op statements; 27 operands, exactly as under the
    // multiplied encoding (a unit operand still occupies one record).
    assert_eq!(tape.num_statements(), 19);
    assert_eq!(tape.num_operations(), 27);
    out.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    assert_eq!([x.adjoint(&tape), y.adjoint(&tape), z.adjoint(&tape)], [2.0, 2.0, 4.0]);
}

#[test]
fn flag_bits_never_leak_into_a_variable_slot() {
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let x = AReal::input(1.5, &mut tape);
    let y = AReal::input(0.5, &mut tape);
    let z = AReal::input(2.0, &mut tape);
    let out = body(&x, &y, &z, 2.0, 3.0, 1.0);
    let probes = [x, y, z, ad::max(&x, &y), -&x, &x + 1.0, 1.0 - &x, &x - &y, ad::sum(&[x, y]), out];
    for v in probes {
        assert!(v.slot() < Tape::<f64>::MAX_VARIABLES, "slot {} carries a flag bit", v.slot());
        assert_eq!(v.slot() & (Tape::<f64>::UNIT | Tape::<f64>::NEG), 0);
    }
    // The passive short-cuts still return unrecorded values.
    let c = AReal::<f64>::new(1.0);
    assert!(!(-&c).should_record());
    assert!(!(&c + 1.0).should_record());
    assert!(!(&c - &c).should_record());
}

#[test]
fn vector_sweep_agrees_with_the_scalar_sweep_bit_for_bit() {
    // Non-integer inputs so the sweep produces non-trivial floats.
    let inputs = [1.37_f64, -0.61, 2.09];
    let mut tape = Tape::<f64>::new(true);
    let _rec = tape.record();
    let mut v: Vec<AReal<f64>> = inputs.iter().map(|&a| AReal::new(a)).collect();
    AReal::register_input(&mut v, &mut tape);
    let mut out = body(&v[0], &v[1], &v[2], 2.0, 3.0, 1.0);
    out.register(&mut tape);
    let out_slot = out.slot();

    out.set_adjoint(&mut tape, 1.0);
    tape.compute_adjoints();
    let scalar: Vec<f64> = v.iter().map(|a| a.adjoint(&tape)).collect();

    let n_dir = 2;
    let nv = tape.num_variables() as usize;
    let mut derivs = vec![0.0; nv * n_dir];
    derivs[out_slot as usize * n_dir] = 1.0;
    derivs[out_slot as usize * n_dir + 1] = 1.0;
    tape.compute_adjoints_vector(n_dir, &mut derivs);
    for (i, a) in v.iter().enumerate() {
        let s = a.slot() as usize;
        assert_eq!(derivs[s * n_dir], scalar[i], "dir 0, input {i}");
        assert_eq!(derivs[s * n_dir + 1], scalar[i], "dir 1, input {i}");
    }
}

#[test]
fn the_k_lane_engine_value_part_equals_the_scalar_gradient_bit_for_bit() {
    let inputs = [1.37_f64, -0.61, 2.09];
    let scalar = {
        let mut tape = Tape::<f64>::new(true);
        let _rec = tape.record();
        let mut v: Vec<AReal<f64>> = inputs.iter().map(|&a| AReal::new(a)).collect();
        AReal::register_input(&mut v, &mut tape);
        let mut out = body(&v[0], &v[1], &v[2], 2.0, 3.0, 1.0);
        out.register(&mut tape);
        out.set_adjoint(&mut tape, 1.0);
        tape.compute_adjoints();
        v.iter().map(|a| a.adjoint(&tape)).collect::<Vec<_>>()
    };
    type J = JetK<f64, 4>;
    let mut tape = Tape::<J>::new(true);
    let _rec = tape.record();
    let mut v: Vec<AReal<J>> = inputs
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let mut t = [0.0; 4];
            t[i] = 1.0;
            AReal::new(JetK::new(a, t))
        })
        .collect();
    AReal::register_input(&mut v, &mut tape);
    let mut out = body(&v[0], &v[1], &v[2], J::constant(2.0), J::constant(3.0), J::constant(1.0));
    out.register(&mut tape);
    out.set_adjoint(&mut tape, J::constant(1.0));
    tape.compute_adjoints();
    for (i, a) in v.iter().enumerate() {
        let adj = a.adjoint(&tape);
        assert_eq!(adj.value, scalar[i], "input {i} value part");
    }
    // Second-order sanity: only `z·z` is nonlinear, so H = diag(0, 0, 2).
    assert_eq!(v[2].adjoint(&tape).tangents, [0.0, 0.0, 2.0, 0.0]);
    assert_eq!(v[0].adjoint(&tape).tangents, [0.0; 4]);
}
