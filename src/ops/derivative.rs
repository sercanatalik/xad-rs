//! First-order derivative drivers — the two shapes the Jacobian and Hessian
//! entry points do not serve.
//!
//! [`compute_jacobian_rev`](crate::compute_jacobian_rev) covers `f: Rⁿ → Rᵐ`
//! and the Hessian drivers cover second order, but the everyday first-order
//! shapes — "give me `df/dx`" for a sequential root-solve, "give me the
//! gradient" for a calibration objective — had no entry point at all, which
//! pushed callers into hand-written tape management.
//!
//! Each driver here does all the mode wiring:
//!
//! | Driver | Mode | Shape | Cost |
//! |---|---|---|---|
//! | [`compute_derivative_fwd`] | forward | `R → R` | one evaluation, no tape |
//! | [`compute_directional_derivative_fwd`] | forward | `Rⁿ → R` along one seed | one evaluation, no tape |
//! | [`compute_gradient_rev`] | reverse | `Rⁿ → R` | one recording + one sweep |
//!
//! The forward drivers create and activate **no tape**: `Jet1` carries its
//! tangent in the value itself. The reverse driver does all the registration,
//! adjoint seeding, and sweeping internally, so the caller supplies only the
//! function and the point.
//!
//! Which to reach for is the usual forward/reverse trade-off: the forward
//! directional driver costs one evaluation per direction, so recovering all
//! `n` partials that way costs `n` evaluations, while
//! [`compute_gradient_rev`] returns all `n` from a single sweep. Reverse
//! breaks even around `n ~ 4`.

use crate::forward::jet1::Jet1;
use crate::passive::Passive;
use crate::reverse::areal::AReal;
use crate::tape::{Tape, TapeStorage};

/// Value and first derivative of a single-input, single-output function, in
/// forward mode.
///
/// Seeds the input's tangent to `1` and reads `f'(x)` straight off the
/// result. No tape is created or activated, and nothing is recorded even if
/// a tape happens to be active on the thread.
///
/// # Example
///
/// ```
/// use xad_rs::ops::compute_derivative_fwd;
/// use xad_rs::Real;
///
/// // f(x) = x·ln(x); f'(x) = ln(x) + 1
/// let (v, d) = compute_derivative_fwd(2.0_f64, |x| x.clone() * x.ln());
/// assert!((v - 2.0 * 2.0_f64.ln()).abs() < 1e-15);
/// assert!((d - (2.0_f64.ln() + 1.0)).abs() < 1e-15);
/// ```
pub fn compute_derivative_fwd<T, F>(x: T, func: F) -> (T, T)
where
    T: Passive,
    F: FnOnce(&Jet1<T>) -> Jet1<T>,
{
    let out = func(&Jet1::new(x, T::one()));
    (out.value(), out.derivative())
}

/// Value and directional derivative `∇f(x)·v` of a many-input,
/// single-output function, in forward mode.
///
/// Each input `i` is seeded with tangent `seed[i]`, so one evaluation yields
/// the derivative along the whole direction. A unit seed (`1` at index `i`,
/// `0` elsewhere) recovers the single partial `∂f/∂xᵢ`. No tape is created
/// or activated.
///
/// # Panics
/// Panics if `inputs.len() != seed.len()`.
///
/// # Example
///
/// ```
/// use xad_rs::ops::compute_directional_derivative_fwd;
/// use xad_rs::Real;
///
/// // f(x, y) = x²·y; ∇f = (2xy, x²) = (24, 9) at (3, 4).
/// let f = |v: &[xad_rs::Jet1<f64>]| v[0].clone() * v[0].clone() * v[1].clone();
/// let (val, d) = compute_directional_derivative_fwd(&[3.0, 4.0], &[1.0, 0.0], f);
/// assert_eq!(val, 36.0);
/// assert_eq!(d, 24.0);              // unit seed -> one partial
///
/// let (_, d) = compute_directional_derivative_fwd(&[3.0, 4.0], &[0.5, 2.0], f);
/// assert_eq!(d, 0.5 * 24.0 + 2.0 * 9.0);   // general seed -> ∇f·v
/// ```
pub fn compute_directional_derivative_fwd<T, F>(inputs: &[T], seed: &[T], func: F) -> (T, T)
where
    T: Passive,
    F: FnOnce(&[Jet1<T>]) -> Jet1<T>,
{
    assert_eq!(
        inputs.len(),
        seed.len(),
        "compute_directional_derivative_fwd: seed length must match inputs"
    );
    let jets: Vec<Jet1<T>> = inputs
        .iter()
        .zip(seed)
        .map(|(&v, &d)| Jet1::new(v, d))
        .collect();
    let out = func(&jets);
    (out.value(), out.derivative())
}

/// Value and full gradient of a many-input, single-output function, from a
/// **single** reverse sweep.
///
/// Records the function once on a private tape, seeds the output adjoint to
/// `1`, sweeps, and reads every input's adjoint. The caller performs no
/// registration, seeding, or sweep, and the tape is deactivated before the
/// function returns.
///
/// This is the single-output specialisation of
/// [`compute_jacobian_rev`](crate::compute_jacobian_rev): the returned
/// gradient is that driver's single row, computed by the scalar sweep rather
/// than the one-direction vector sweep.
///
/// # Panics
/// Panics if a tape is already active on this thread — recordings do not
/// nest.
///
/// # Example
///
/// ```
/// use xad_rs::ops::compute_gradient_rev;
/// use xad_rs::Real;
///
/// // f(x, y) = x²·y + sin(x) at (3, 4)
/// let (v, g) = compute_gradient_rev(&[3.0_f64, 4.0], |v| {
///     v[0].clone() * v[0].clone() * v[1].clone() + v[0].sin()
/// });
/// assert!((v - (36.0 + 3.0_f64.sin())).abs() < 1e-12);
/// assert!((g[0] - (24.0 + 3.0_f64.cos())).abs() < 1e-12);  // 2xy + cos x
/// assert!((g[1] - 9.0).abs() < 1e-12);                     // x²
/// ```
pub fn compute_gradient_rev<T, F>(inputs: &[T], func: F) -> (T, Vec<T>)
where
    T: TapeStorage,
    F: FnOnce(&[AReal<T>]) -> AReal<T>,
{
    let mut tape = Tape::<T>::new(true);
    // RAII: the tape is deactivated when `_rec` drops, including on unwind
    // from a panic inside `func`.
    let _rec = tape.record();

    let mut ad_inputs: Vec<AReal<T>> = inputs.iter().map(|&v| AReal::new(v)).collect();
    AReal::register_input(&mut ad_inputs, &mut tape);

    let mut output = func(&ad_inputs);
    AReal::register_output(std::slice::from_mut(&mut output), &mut tape);
    output.set_adjoint(&mut tape, T::one());
    tape.compute_adjoints();

    let gradient = ad_inputs.iter().map(|x| x.adjoint(&tape)).collect();
    (output.value(), gradient)
}
