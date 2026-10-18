//! `powi`/`powf` at a zero base, and `max`/`min` on NaN and signed zeros:
//! the edges where `n·u^{n-1}` is `0·∞` and `f64::max` skips a NaN. Every
//! mode must give the true derivative and the passive scalar's value.

use xad_rs::math::{ad, fwd, fwdk};
use xad_rs::{Jet1, Jet2, Jet2Vec, JetK, Real, compute_gradient_rev};

/// `(value, d/du)` of a `Real` kernel at `u` in reverse, Jet1, JetK and
/// Jet2, plus Jet2's `d²/du²`, checked against `want = (value, d, d²)`.
macro_rules! check_modes {
    ($u:expr, $want:expr, |$x:ident| $body:expr) => {{
        fn k<R: Real>($x: &R) -> R {
            $body
        }
        let (u, (v, d, d2)): (f64, (f64, f64, f64)) = ($u, $want);
        let (rv, rg) = compute_gradient_rev(&[u], |x| k(&x[0]));
        assert_eq!((rv, rg[0]), (v, d), "reverse at {u}");
        let j1 = k(&Jet1::new(u, 1.0));
        assert_eq!((j1.value(), j1.derivative()), (v, d), "Jet1 at {u}");
        let jk = k(&JetK::<f64, 2>::new(u, [1.0, 0.0]));
        assert_eq!((jk.value, jk.tangents[0]), (v, d), "JetK at {u}");
        let j2 = k(&Jet2::variable(u));
        assert_eq!(
            (j2.value(), j2.first_derivative(), j2.second_derivative()),
            (v, d, d2),
            "Jet2 at {u}"
        );
        assert_eq!(k(&u), v, "f64 at {u}");
    }};
}

#[test]
fn powi_at_zero_has_finite_derivatives() {
    check_modes!(0.0, (1.0, 0.0, 0.0), |x| x.powi(0));
    check_modes!(0.0, (0.0, 1.0, 0.0), |x| x.powi(1));
    check_modes!(0.0, (0.0, 0.0, 2.0), |x| x.powi(2));
    check_modes!(0.0, (0.0, 0.0, 0.0), |x| x.powi(3));
}

#[test]
fn powf_with_a_passive_integer_exponent_at_zero_has_finite_derivatives() {
    check_modes!(0.0, (1.0, 0.0, 0.0), |x| x.powf(R::from(0.0_f64)));
    check_modes!(0.0, (0.0, 1.0, 0.0), |x| x.powf(R::from(1.0_f64)));
    check_modes!(0.0, (0.0, 0.0, 2.0), |x| x.powf(R::from(2.0_f64)));

    // The free `powf(base, T)` forms and Jet2Vec's inherent `powf`.
    for (e, d) in [(0.0, 0.0), (1.0, 1.0)] {
        let (_, g) = compute_gradient_rev(&[0.0], |x| ad::powf(&x[0], e));
        assert_eq!(g[0], d, "ad::powf e={e}");
        assert_eq!(
            fwd::powf(&Jet1::new(0.0, 1.0), e).derivative(),
            d,
            "fwd e={e}"
        );
        let k = fwdk::powf(&JetK::<f64, 2>::new(0.0, [1.0, 0.0]), e);
        assert_eq!(k.tangents[0], d, "fwdk e={e}");
        let j2 = Jet2::variable(0.0).powf(e);
        assert_eq!(
            (j2.first_derivative(), j2.second_derivative()),
            (d, 0.0),
            "Jet2 e={e}"
        );
        let jv = Jet2Vec::variable(0.0, 0, 1).powf(e);
        assert_eq!(
            (jv.gradient()[0], jv.hessian_at(0, 0)),
            (d, 0.0),
            "Jet2Vec e={e}"
        );
    }
}

#[test]
fn polynomial_at_its_reference_point_has_a_finite_gradient() {
    // Σ_{k=0..4} t^k at t = 0: value 1, slope 1 (k = 1), curvature 2 (k = 2).
    check_modes!(0.0, (1.0, 1.0, 2.0), |x| {
        let mut acc = R::zero();
        for k in 0..=4 {
            acc = acc + x.powi(k);
        }
        acc
    });
}

#[test]
fn powi_values_stay_the_passive_scalars() {
    for &t in &[0.0_f64, -0.0, 0.3, -1.7, 2.5] {
        for n in -3..=5 {
            let want = t.powi(n);
            let same = |got: f64| got.to_bits() == want.to_bits();
            assert!(
                same(compute_gradient_rev(&[t], |x| x[0].powi(n)).0),
                "rev {t}^{n}"
            );
            assert!(same(Jet1::new(t, 1.0).powi(n).value()), "Jet1 {t}^{n}");
            assert!(
                same(Real::powi(&Jet2::variable(t), n).value()),
                "Jet2 {t}^{n}"
            );
            assert!(
                same(JetK::<f64, 2>::new(t, [1.0, 0.0]).powi(n).value),
                "JetK {t}^{n}"
            );
        }
    }
}

#[test]
fn active_exponent_at_a_zero_base_is_flat_in_the_exponent() {
    // ∂(0^e)/∂e = 0 for e > 0; ∂/∂b is the direct-form partial.
    for (e, db) in [(2.0, 0.0), (1.0, 1.0), (0.5, f64::INFINITY)] {
        let (v, g) = compute_gradient_rev(&[0.0, e], |x| x[0].powf(x[1]));
        assert_eq!((v, g[0], g[1]), (0.0, db, 0.0), "reverse e={e}");
        if db.is_infinite() {
            // Forward modes multiply the infinite base partial by a zero
            // tangent (∞·0 = NaN) like every other op at a pole; reverse only.
            continue;
        }
        let j = Jet1::new(0.0, 0.0).powf(Jet1::new(e, 1.0));
        assert_eq!(j.derivative(), 0.0, "Jet1 d/de e={e}");
        let k = JetK::<f64, 2>::new(0.0, [1.0, 0.0]).powf(JetK::new(e, [0.0, 1.0]));
        assert_eq!((k.tangents[0], k.tangents[1]), (db, 0.0), "JetK e={e}");
    }
    // Jet2 along (u, e) = (t, 2 + t): d²/dt² = g_uu + 2 g_ue + g_ee = 2 + 0 + 0.
    let u = Jet2::variable(0.0);
    let y = Real::powf(&u, Jet2::variable(2.0));
    assert_eq!(
        (y.value(), y.first_derivative(), y.second_derivative()),
        (0.0, 0.0, 2.0)
    );
    // Exponent-only direction: 0^e stays 0, all derivatives 0.
    let y = Real::powf(&Jet2::constant(0.0), Jet2::variable(2.0));
    assert_eq!((y.first_derivative(), y.second_derivative()), (0.0, 0.0));
}

#[test]
fn max_and_min_agree_bit_for_bit_on_nan_and_signed_zero() {
    let pairs = [
        (0.0, f64::NAN),
        (f64::NAN, 0.0),
        (f64::NAN, f64::NAN),
        (0.0, -0.0),
        (-0.0, 0.0),
        (1.0, 2.0),
    ];
    for (a, b) in pairs {
        for (name, op) in [("max", 0), ("min", 1)] {
            fn pick<R: Real>(a: &R, b: &R, op: i32) -> R {
                if op == 0 { a.max(b) } else { a.min(b) }
            }
            let want = pick(&a, &b, op).to_bits();
            let rev = compute_gradient_rev(&[a, b], |x| pick(&x[0], &x[1], op)).0;
            let j1 = pick(&Jet1::new(a, 1.0), &Jet1::new(b, 0.0), op).value();
            let j2 = pick(&Jet2::variable(a), &Jet2::constant(b), op).value();
            let jk = pick(
                &JetK::<f64, 2>::new(a, [1.0, 0.0]),
                &JetK::new(b, [0.0, 1.0]),
                op,
            )
            .value;
            for (mode, got) in [("AReal", rev), ("Jet1", j1), ("Jet2", j2), ("JetK", jk)] {
                assert_eq!(got.to_bits(), want, "{name}({a}, {b}) {mode} vs f64");
            }
        }
    }
    // The documented rule: a NaN second operand wins, a NaN first is dropped.
    assert!(Real::max(&0.0_f64, &f64::NAN).is_nan());
    assert_eq!(Real::max(&f64::NAN, &0.0_f64), 0.0);
}
