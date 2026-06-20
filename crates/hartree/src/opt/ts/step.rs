//! Pure step-construction math for the [`super::prfo`] driver: the partitioned
//! RFO step in the mass-weighted eigenbasis (with a restricted-step scaling that
//! sizes it to the trust radius), followed-mode selection, the Bofill Hessian
//! update, trust-radius adaption, and the step-acceptance test. Split out of
//! `prfo.rs` so the control flow (the iteration loop, Hessian maintenance, and
//! step backtracking) and the numerics each stay small and independently testable.

use super::TsOptions;
use super::numerics::{MwSpectrum, dot, matvec, overlap};
use crate::linalg::{mat_from_row_major, symmetric_eigh};

/// One partitioned RFO step in the mass-weighted eigenbasis: the `followed` mode is
/// driven to a maximum, the rest to a minimum. Returns the trust-limited,
/// mass-weighted Cartesian step (flat, length `ndof`).
///
/// The unrestricted (α = 1) rational-function step is taken whenever it already
/// falls inside the trust radius — byte-identical to the historical step there. When
/// it overshoots, the step is *restricted* rather than clipped: a single scaling
/// parameter α ≥ 1 raises the RFO level shift in **both** partitions (the 2×2 climbed
/// block and the (m+1)×(m+1) minimized block) until the step norm meets the trust
/// radius exactly. This preserves the rational-model direction — each mode is damped
/// in proportion to how stiff it is — where a uniform clip (`step · trust/‖step‖`)
/// rescales every mode by one common factor, over-shrinking the well-conditioned
/// modes only to rein in a single soft one.
pub(super) fn prfo_step(
    spec: &MwSpectrum,
    g_mw: &[f64],
    non_null: &[usize],
    followed: usize,
    trust: f64,
) -> Vec<f64> {
    let ndof = spec.eigenvalues.len();
    let modes = partition_modes(spec, g_mw, non_null, followed, ndof);

    // α = 1 is the unrestricted RFO step; restrict it only when it overshoots the
    // trust radius. The common near-saddle case keeps the unrestricted step (and a
    // single eigendecomposition), exactly as before.
    let (mut step_p, mut steps_n) = step_components(&modes, 1.0);
    if components_norm(step_p, &steps_n) > trust {
        let alpha = restrict_alpha(&modes, trust);
        let restricted = step_components(&modes, alpha);
        step_p = restricted.0;
        steps_n = restricted.1;
        // The restricted scaling lands the norm on the trust radius for any realistic
        // trust; only an extreme trust/gradient ratio (far below the `min_trust` a real
        // search floors at) could leave `restrict_alpha` unable to bracket it. Guard
        // that unreachable case with a uniform clamp so the returned step is never
        // larger than the trust radius.
        let n = components_norm(step_p, &steps_n);
        if n > trust {
            let scale = trust / n;
            step_p *= scale;
            steps_n.iter_mut().for_each(|s| *s *= scale);
        }
    }
    assemble_step(spec, &modes, followed, step_p, &steps_n, ndof)
}

/// The followed (climbed) mode and the minimized modes for the partitioned RFO step,
/// each reduced to its projected-Hessian eigenvalue `b` and the gradient component
/// `F` along it in the eigenbasis. Extracted once per step so the restricted-step
/// search can re-evaluate the step at many trial scalings without re-touching the
/// `ndof`-length eigenvectors.
struct PartitionedModes {
    /// Followed mode: (eigenvalue `b_p`, gradient component `F_p`).
    followed: (f64, f64),
    /// Minimized modes: (mode index `k`, eigenvalue `b_k`, gradient component `F_k`).
    minimized: Vec<(usize, f64, f64)>,
}

fn partition_modes(
    spec: &MwSpectrum,
    g_mw: &[f64],
    non_null: &[usize],
    followed: usize,
    ndof: usize,
) -> PartitionedModes {
    let fcomp = |k: usize| -> f64 {
        (0..ndof)
            .map(|i| spec.eigenvectors[i * ndof + k] * g_mw[i])
            .sum()
    };
    let minimized = non_null
        .iter()
        .copied()
        .filter(|&k| k != followed)
        .map(|k| (k, spec.eigenvalues[k], fcomp(k)))
        .collect();
    PartitionedModes {
        followed: (spec.eigenvalues[followed], fcomp(followed)),
        minimized,
    }
}

/// The partitioned RFO step components at restricted-step scaling `alpha`: the
/// climbing step along the followed mode and the descending step along each
/// minimized mode (aligned with [`PartitionedModes::minimized`]). Because the modes
/// are orthonormal, the step's mass-weighted norm is the Euclidean norm of these
/// components ([`components_norm`]), so the restricted-step search can size the step
/// without assembling the full `ndof` vector.
fn step_components(modes: &PartitionedModes, alpha: f64) -> (f64, Vec<f64>) {
    let (bp, fp) = modes.followed;
    let step_p = step_ratio(fp, bp - followed_shift(bp, fp, alpha));
    let sigma_n = minimized_shift(&modes.minimized, alpha);
    let steps_n = modes
        .minimized
        .iter()
        .map(|&(_, bk, fk)| match sigma_n {
            Some(sigma) => step_ratio(fk, bk - sigma),
            None => 0.0,
        })
        .collect();
    (step_p, steps_n)
}

/// The level shift σ for the climbed 2×2 block at scaling `alpha`: the upper root of
/// the α-scaled augmented matrix `[[b_p, F_p], [F_p, 0]]`, written directly as the
/// shift σ = αλ so the step along the followed mode is `−F_p/(b_p − σ)`. σ ≥ b_p, so
/// the step climbs uphill; raising α raises σ and shortens the climb. At α = 1 this
/// reduces to the historical `½(b_p + √(b_p² + 4·F_p²))`.
fn followed_shift(bp: f64, fp: f64, alpha: f64) -> f64 {
    0.5 * (bp + (bp * bp + 4.0 * alpha * fp * fp).sqrt())
}

/// The level shift σ for the minimized (m+1)×(m+1) block at scaling `alpha`: the
/// lowest eigenvalue of the α-scaled augmented matrix
/// `[[diag(b)/α, F/√α], [Fᵀ/√α, 0]]`, returned as the shift σ = αλ (so each minimized
/// step is `−F_k/(b_k − σ)`). σ ≤ min b_k — the lowest root lies below every diagonal —
/// so every minimized mode descends; raising α pushes σ further below the diagonals
/// and shortens the descent. At α = 1 the scaled matrix is the plain augmented matrix,
/// reproducing the historical minimized-block shift. `None` when there are no
/// minimized modes.
fn minimized_shift(minimized: &[(usize, f64, f64)], alpha: f64) -> Option<f64> {
    let m = minimized.len();
    if m == 0 {
        return None;
    }
    let sqrt_alpha = alpha.sqrt();
    let mut aug = vec![0.0f64; (m + 1) * (m + 1)];
    for (a, &(_, bk, fk)) in minimized.iter().enumerate() {
        aug[a * (m + 1) + a] = bk / alpha;
        aug[a * (m + 1) + m] = fk / sqrt_alpha;
        aug[m * (m + 1) + a] = fk / sqrt_alpha;
    }
    let lambda = symmetric_eigh(&mat_from_row_major(m + 1, &aug)).values[0];
    Some(lambda * alpha)
}

/// `−numer/denom`, guarded against a vanishing denominator (a mode whose level shift
/// sits on its eigenvalue), matching the historical step construction.
fn step_ratio(numer: f64, denom: f64) -> f64 {
    if denom.abs() > 1e-12 {
        -numer / denom
    } else {
        0.0
    }
}

/// The mass-weighted norm of a step from its orthonormal-mode components.
fn components_norm(step_p: f64, steps_n: &[f64]) -> f64 {
    (step_p * step_p + steps_n.iter().map(|s| s * s).sum::<f64>()).sqrt()
}

/// Expand the partitioned step components back onto the `ndof` mass-weighted
/// Cartesian axes: `step_p` along the followed eigenvector plus each minimized
/// component along its eigenvector.
fn assemble_step(
    spec: &MwSpectrum,
    modes: &PartitionedModes,
    followed: usize,
    step_p: f64,
    steps_n: &[f64],
    ndof: usize,
) -> Vec<f64> {
    let mut dxi = vec![0.0f64; ndof];
    for (i, slot) in dxi.iter_mut().enumerate() {
        *slot += step_p * spec.eigenvectors[i * ndof + followed];
    }
    for (&(k, _, _), &step_k) in modes.minimized.iter().zip(steps_n) {
        for (i, slot) in dxi.iter_mut().enumerate() {
            *slot += step_k * spec.eigenvectors[i * ndof + k];
        }
    }
    dxi
}

/// Relative tolerance to which the restricted step's norm is driven onto the trust
/// radius, and the iteration cap on the bisection that gets it there. The 0.1 % band
/// keeps the capped step within the ~1 % of the trust radius the restricted step
/// promises, and is reached in far fewer than `RS_MAX_MICRO` bisection steps.
const RS_TRUST_TOL: f64 = 1e-3;
const RS_MAX_MICRO: usize = 60;
/// Geometric factor by which the scaling α is grown to bracket the trust radius
/// before bisection.
const RS_ALPHA_GROWTH: f64 = 4.0;
/// Finite ceiling on the bracketing scaling α — a termination guard, not the normal
/// stopping condition. The step norm decays like `1/√α`, so the bracket closes at
/// α ≈ `(‖step‖/trust)²`, far below this ceiling for any trust at or above the
/// `min_trust` a real search floors at. Should an extreme trust/gradient ratio reach
/// the ceiling without bracketing, [`prfo_step`] clamps the result, so the returned
/// step still never exceeds the trust radius.
const RS_ALPHA_MAX: f64 = 1e30;

/// Find the restricted-step scaling α ≥ 1 whose partitioned RFO step norm meets the
/// trust radius. The step norm decreases monotonically from the unrestricted RFO step
/// at α = 1 toward zero as α grows (raising the level shift in both partitions), so a
/// geometric bracketing followed by bisection lands the norm on the trust radius
/// while preserving the rational-model direction. Only called when the α = 1 step
/// overshoots, so α = 1 is a valid lower bracket (its norm exceeds the trust radius).
fn restrict_alpha(modes: &PartitionedModes, trust: f64) -> f64 {
    let norm_at = |alpha: f64| {
        let (step_p, steps_n) = step_components(modes, alpha);
        components_norm(step_p, &steps_n)
    };
    // Bracket: `lo` keeps a norm above the trust radius (α = 1 on entry), `hi` grows
    // until its norm falls to or below it. The `1/√α` decay guarantees this terminates
    // for any positive trust; `RS_ALPHA_MAX` bounds it against overflow in the
    // (unreachable-in-practice) extreme-ratio case, which `prfo_step` then clamps.
    let mut lo = 1.0;
    let mut hi = RS_ALPHA_GROWTH;
    while norm_at(hi) > trust && hi < RS_ALPHA_MAX {
        lo = hi;
        hi *= RS_ALPHA_GROWTH;
    }
    // Bisect, holding the invariant norm(lo) > trust ≥ norm(hi) (once bracketed) so the
    // returned α never lets the step exceed the trust radius.
    for _ in 0..RS_MAX_MICRO {
        let mid = 0.5 * (lo + hi);
        let n = norm_at(mid);
        if n > trust {
            lo = mid;
        } else {
            hi = mid;
            if trust - n <= RS_TRUST_TOL * trust {
                break;
            }
        }
    }
    hi
}

/// First step: the non-null mode of maximum overlap with the reaction-coordinate
/// `seed` if one is supplied (anchoring the climb to the forming/breaking-bond
/// direction, immune to an avoided-crossing reordering of the soft modes), else
/// the `follow_mode`-th non-null mode by ascending eigenvalue. Later steps (once
/// `previous` is set): the non-null mode of maximum overlap with the previous one.
/// `seed` and `previous` are both in the mass-weighted frame of `spec`.
pub(super) fn select_followed(
    spec: &MwSpectrum,
    non_null: &[usize],
    follow_mode: usize,
    previous: &Option<Vec<f64>>,
    seed: Option<&[f64]>,
) -> usize {
    let ndof = spec.eigenvalues.len();
    let max_overlap_with = |reference: &[f64]| -> usize {
        non_null
            .iter()
            .copied()
            .max_by(|&a, &b| {
                overlap(spec, ndof, a, reference)
                    .partial_cmp(&overlap(spec, ndof, b, reference))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    };
    match (previous, seed) {
        // Tracking the climbed mode dominates once a step has been taken.
        (Some(reference), _) => max_overlap_with(reference),
        // First step with a seed: anchor to the reaction coordinate.
        (None, Some(seed)) => max_overlap_with(seed),
        // First step, no seed: the requested softest mode.
        (None, None) => non_null[follow_mode.min(non_null.len() - 1)],
    }
}

/// Bofill (1994) Hessian update — an SR1/PSB blend that, unlike BFGS, preserves
/// the indefiniteness the reaction mode needs. `s` is the Cartesian step, `y` the
/// gradient change.
pub(super) fn bofill_update(hess: &mut [f64], s: &[f64], y: &[f64], n: usize) {
    let ss = dot(s, s);
    if ss < 1e-14 {
        return;
    }
    let hs = matvec(hess, s, n);
    let delta: Vec<f64> = y.iter().zip(&hs).map(|(yi, hsi)| yi - hsi).collect();
    let sd = dot(s, &delta);
    let dd = dot(&delta, &delta);
    let phi = if dd > 1e-14 {
        (sd * sd) / (ss * dd)
    } else {
        0.0
    };
    let ms_ok = sd.abs() > 1e-12;
    for i in 0..n {
        for j in 0..n {
            let ms = if ms_ok { delta[i] * delta[j] / sd } else { 0.0 };
            let psb = (delta[i] * s[j] + s[i] * delta[j]) / ss - sd * s[i] * s[j] / (ss * ss);
            hess[i * n + j] += phi * ms + (1.0 - phi) * psb;
        }
    }
}

/// Adapt the trust radius from how well the quadratic model predicted the energy
/// change. `step_norm` is mass-weighted, matching `prfo_step`'s cap.
pub(super) fn update_trust_ts(
    trust: f64,
    actual: f64,
    predicted: f64,
    step_norm: f64,
    opts: &TsOptions,
) -> f64 {
    if predicted.abs() < 1e-14 {
        return trust;
    }
    let ratio = actual / predicted;
    if (0.75..=1.25).contains(&ratio) && step_norm > 0.8 * trust {
        (2.0 * trust).min(opts.max_trust)
    } else if !(0.25..=1.75).contains(&ratio) {
        (0.5 * trust).max(opts.min_trust)
    } else {
        trust
    }
}

/// Factor by which the actual energy change must exceed the model's prediction
/// (and in the opposite direction) for a step to count as a gross overshoot.
const OVERSHOOT_FACTOR: f64 = 2.0;

/// Whether a trial step should be rejected and retried at a smaller trust radius.
///
/// Rejection is reserved for a *gross overshoot*: the surface moved opposite to a
/// non-trivial model prediction **and** by substantially more than predicted, so
/// the step clearly left the region the quadratic describes. The opposite-direction
/// test alone is not enough — P-RFO deliberately drives the followed mode *uphill*,
/// so a positive predicted change is routine, and a small genuine drop along the
/// reaction coordinate (`|actual| ≲ |predicted|`) is benign progress, not a model
/// failure. Only when the drop dwarfs the prediction (`|actual| > 2·|predicted|`)
/// has the step overshot into a different region. A merely poor magnitude (same
/// sign, ratio off from 1) is left to [`update_trust_ts`], which shrinks the radius
/// for the *next* step without discarding this one. The magnitude floor keeps
/// finite-difference noise near convergence from tripping a spurious rejection.
pub(super) fn is_pathological(actual: f64, predicted: f64) -> bool {
    predicted.abs() > 1e-7
        && actual * predicted < 0.0
        && actual.abs() > OVERSHOOT_FACTOR * predicted.abs()
}
