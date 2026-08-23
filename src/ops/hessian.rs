//! Hessian computation via reverse-mode AD and dense forward-mode Jet2Vec.

use ndarray::Array2;

use crate::reverse::areal::AReal;
use crate::forward::jet1::Jet1;
use crate::forward::jet2vec::Jet2Vec;
use crate::forward::jetk::JetK;
use crate::tape::Tape;

/// Compute the exact Hessian of a scalar-valued function via **forward-over-
/// adjoint** automatic differentiation — no finite-difference error.
///
/// The function is written in the nested active type `AReal<Jet1<f64>>`: a
/// reverse-mode adjoint whose *storage* scalar is itself a forward dual. For
/// each input direction `k`, input `k`'s tangent is seeded to `1`, the function
/// is recorded, and one reverse sweep is run; each input `j`'s adjoint is then a
/// dual whose **value** is `∂f/∂x_j` and whose **tangent** is `∂²f/∂x_j∂x_k` —
/// column `k` of the Hessian. `n` sweeps give the full `n × n` Hessian exactly,
/// in `O(n)` reverse passes. This replaces the previous finite-difference
/// implementation.
///
/// Closures written generically over the active scalar (operators + `math::ad`)
/// compile unchanged against the nested type; closures naming `AReal<f64>` or
/// building `AReal` from `f64` literals are updated to be generic over
/// `TapeStorage` or to use the nested scalar.
pub fn compute_hessian<F>(inputs: &[f64], func: F) -> Array2<f64>
where
    F: Fn(&[AReal<Jet1<f64>>]) -> AReal<Jet1<f64>>,
{
    let n = inputs.len();
    let mut hessian = Array2::<f64>::zeros((n, n));

    // One tape reused across all n direction sweeps: direction k's recording
    // replays the same op sequence as direction 0, so after the first sweep
    // every subsequent `record()` runs entirely in already-grown buffers —
    // no per-direction allocation churn.
    let mut tape = Tape::<Jet1<f64>>::new(true);

    for k in 0..n {
        let _rec = tape.record();

        // Seed direction k: input k carries tangent 1, all others tangent 0.
        let mut ad_inputs: Vec<AReal<Jet1<f64>>> = inputs
            .iter()
            .enumerate()
            .map(|(i, &v)| AReal::new(Jet1::new(v, if i == k { 1.0 } else { 0.0 })))
            .collect();
        AReal::register_input(&mut ad_inputs, &mut tape);

        let mut output = func(&ad_inputs);
        AReal::register_output(std::slice::from_mut(&mut output), &mut tape);
        output.set_adjoint(&mut tape, Jet1::constant(1.0));
        tape.compute_adjoints();

        // The tangent of input j's adjoint is ∂²f/∂x_j∂x_k = H[[j, k]].
        for (j, inp) in ad_inputs.iter().enumerate() {
            hessian[[j, k]] = inp.adjoint(&tape).derivative();
        }
    }

    hessian
}

// ============================================================================
// K-wide forward-over-adjoint via JetK (⌈n/K⌉ recordings)
// ============================================================================

/// One K-wide forward-over-adjoint pass: seed tangent lanes
/// `block .. block+K` (one input per lane), record, sweep, and return each
/// input's adjoint tangent array — input `j`'s lane `k` is
/// `H[[j, block + k]]`. Shared by the serial and parallel K-wide drivers.
fn hessian_k_block<const K: usize, F>(
    tape: &mut Tape<JetK<f64, K>>,
    inputs: &[f64],
    block: usize,
    func: &F,
) -> Vec<[f64; K]>
where
    JetK<f64, K>: crate::tape::TapeStorage,
    F: Fn(&[AReal<JetK<f64, K>>]) -> AReal<JetK<f64, K>>,
{
    let _rec = tape.record();
    let mut ad_inputs: Vec<AReal<JetK<f64, K>>> = inputs
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let mut t = [0.0; K];
            if i >= block && i < block + K {
                t[i - block] = 1.0;
            }
            AReal::new(JetK::new(v, t))
        })
        .collect();
    AReal::register_input(&mut ad_inputs, tape);

    let mut output = func(&ad_inputs);
    AReal::register_output(std::slice::from_mut(&mut output), tape);
    output.set_adjoint(tape, JetK::constant(1.0));
    tape.compute_adjoints();

    ad_inputs.iter().map(|inp| inp.adjoint(tape).tangents).collect()
}

/// Scatter one block's K tangent columns into the Hessian, clipping the
/// final partial block when `K` does not divide `n`.
fn write_hessian_k_block<const K: usize>(
    hessian: &mut Array2<f64>,
    block: usize,
    tangents: &[[f64; K]],
) {
    let n = tangents.len();
    for (j, t) in tangents.iter().enumerate() {
        for (k, &v) in t.iter().take(n - block).enumerate() {
            hessian[[j, block + k]] = v;
        }
    }
}

/// K-wide exact forward-over-adjoint Hessian — [`compute_hessian`] with the
/// storage scalar widened from `Jet1<f64>` to [`JetK<f64, K>`]: each
/// recording seeds `K` input tangent lanes at once, so the full `n × n`
/// Hessian takes `⌈n/K⌉` record-and-sweep passes instead of `n`.
///
/// Every tangent lane evolves exactly as the corresponding `Jet1` tangent
/// would, so the result is **bit-identical** to [`compute_hessian`]; the
/// lanes are elementwise array ops that auto-vectorize, so per-pass cost
/// grows sublinearly in `K` and the speedup tracks `K` nearly ideally.
/// Measured on Apple M-series: 3.9× at K = 8 on a forward-over-adjoint
/// shape with n = 12, and 4.6× / 8.2× at K = 4/8 on a 48-input, 2000-op
/// kernel — single-threaded.
///
/// `K` must be one of the lane counts with a stamped tape slot
/// (K ∈ {2, 4, 8}; see `src/forward/jetk.rs`). K = 4–8 is the measured
/// sweet spot — beyond that the tape's per-operand record (`8 + 8K` bytes)
/// erodes the amortization.
///
/// The closure is written against `AReal<JetK<f64, K>>`; code that is
/// generic over [`TapeStorage`](crate::tape::TapeStorage) (operators +
/// `math::ad`) compiles unchanged.
pub fn compute_hessian_k<const K: usize, F>(inputs: &[f64], func: F) -> Array2<f64>
where
    JetK<f64, K>: crate::tape::TapeStorage,
    F: Fn(&[AReal<JetK<f64, K>>]) -> AReal<JetK<f64, K>>,
{
    const { assert!(K > 0, "compute_hessian_k: K must be at least 1") };
    let n = inputs.len();
    let mut hessian = Array2::<f64>::zeros((n, n));
    // One tape reused across all ⌈n/K⌉ block sweeps (see compute_hessian).
    let mut tape = Tape::<JetK<f64, K>>::new(true);
    for block in (0..n).step_by(K) {
        let tangents = hessian_k_block(&mut tape, inputs, block, &func);
        write_hessian_k_block(&mut hessian, block, &tangents);
    }
    hessian
}

/// Below this many recorded operations remaining after the first block
/// (`ops_per_block × remaining_blocks`), [`compute_hessian_k_par`] finishes
/// serially instead of spawning the pool: rayon dispatch (~40–60 µs)
/// outweighs the remaining sweeps on small problems. Measured on Apple
/// M-series — at n = 12 / K = 4 (3 blocks, ~2k ops each) the pool loses
/// 2.3× to the serial loop; at n = 48 / 2000-op kernel it wins 3×.
const K_PAR_MIN_REMAINING_OPERATIONS: usize = 50_000;

/// Parallel [`compute_hessian_k`] — the `⌈n/K⌉` independent block passes
/// distributed across a rayon pool, each worker reusing its own
/// `Tape<JetK<f64, K>>`. Results are bit-identical to [`compute_hessian`].
///
/// The first block is recorded serially; its tape size then decides
/// whether the remaining blocks are worth the pool (below a measured
/// crossover the serial loop finishes the job), so this is safe to call
/// unconditionally. Requires `F: Sync`. Measured on Apple M-series
/// (14 workers), n = 48 / 2000-op kernel: 11.9× at K = 4, 14.7× at K = 8
/// over the serial `Jet1` engine (vs 8.2× for serial K = 8 alone).
pub fn compute_hessian_k_par<const K: usize, F>(inputs: &[f64], func: F) -> Array2<f64>
where
    JetK<f64, K>: crate::tape::TapeStorage,
    F: Fn(&[AReal<JetK<f64, K>>]) -> AReal<JetK<f64, K>> + Sync,
{
    use rayon::prelude::*;

    const { assert!(K > 0, "compute_hessian_k_par: K must be at least 1") };
    let n = inputs.len();
    let mut hessian = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return hessian;
    }

    // First block serial: establishes ops-per-block for the crossover test.
    let mut tape = Tape::<JetK<f64, K>>::new(true);
    let tangents = hessian_k_block(&mut tape, inputs, 0, &func);
    write_hessian_k_block(&mut hessian, 0, &tangents);

    let blocks: Vec<usize> = (K..n).step_by(K).collect();
    if tape.num_operations() * blocks.len() < K_PAR_MIN_REMAINING_OPERATIONS {
        for block in blocks {
            let tangents = hessian_k_block(&mut tape, inputs, block, &func);
            write_hessian_k_block(&mut hessian, block, &tangents);
        }
        return hessian;
    }

    let cols: Vec<(usize, Vec<[f64; K]>)> = blocks
        .into_par_iter()
        .map_init(
            || Tape::<JetK<f64, K>>::new(true),
            |tape, block| (block, hessian_k_block(tape, inputs, block, &func)),
        )
        .collect();
    for (block, tangents) in &cols {
        write_hessian_k_block(&mut hessian, *block, tangents);
    }
    hessian
}

// ============================================================================
// Dense full Hessian via Jet2Vec (positional)
// ============================================================================

/// Dense full Hessian produced by [`compute_full_hessian`] — value, gradient,
/// and the full `n × n` Hessian, all read back positionally.
pub struct DenseHessian {
    pub value: f64,
    pub gradient: Vec<f64>,
    pub hessian: Array2<f64>,
}

/// Compute a dense full Hessian via a single `Jet2Vec` forward pass.
///
/// Inputs are positional (`&[f64]`); read the gradient as `gradient[i]` and the
/// Hessian as `hessian[[i, j]]`. Apply human-readable names at the call site if
/// you need them.
pub fn compute_full_hessian<F>(inputs: &[f64], f: F) -> DenseHessian
where
    F: Fn(&[Jet2Vec]) -> Jet2Vec,
{
    let n = inputs.len();

    let d2v_inputs: Vec<Jet2Vec> = inputs
        .iter()
        .enumerate()
        .map(|(i, &value)| Jet2Vec::variable(value, i, n))
        .collect();

    let output = f(&d2v_inputs);

    DenseHessian {
        value: output.value(),
        gradient: output.gradient().to_vec(),
        hessian: output.hessian(),
    }
}
