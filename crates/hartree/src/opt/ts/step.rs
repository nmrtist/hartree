//! Pure step-construction math for the [`super::prfo`] driver: the partitioned
//! RFO step in the mass-weighted eigenbasis, followed-mode selection, the Bofill
//! Hessian update, trust-radius adaption, and the step-acceptance test. Split out
//! of `prfo.rs` so the control flow (the iteration loop, Hessian maintenance, and
//! step backtracking) and the numerics each stay small and independently testable.

use super::TsOptions;
use super::numerics::{MwSpectrum, dot, matvec, norm, overlap};
use crate::linalg::{mat_from_row_major, symmetric_eigh};

/// One partitioned RFO step in the mass-weighted eigenbasis: the `followed` mode
/// is driven to a maximum, the rest to a minimum. Returns the trust-limited,
/// mass-weighted Cartesian step (flat, length `ndof`).
pub(super) fn prfo_step(
    spec: &MwSpectrum,
    g_mw: &[f64],
    non_null: &[usize],
    followed: usize,
    trust: f64,
) -> Vec<f64> {
    let ndof = spec.eigenvalues.len();
    let fcomp = |k: usize| -> f64 {
        (0..ndof)
            .map(|i| spec.eigenvectors[i * ndof + k] * g_mw[i])
            .sum()
    };
    let b = |k: usize| spec.eigenvalues[k];

    // Followed mode: upper root of [[b_p, F_p], [F_p, 0]] gives an uphill step.
    let fp = fcomp(followed);
    let bp = b(followed);
    let lambda_p = 0.5 * (bp + (bp * bp + 4.0 * fp * fp).sqrt());
    let denom_p = bp - lambda_p;
    let step_p = if denom_p.abs() > 1e-12 {
        -fp / denom_p
    } else {
        0.0
    };

    // Remaining modes: lowest root of [[diag(b_N), F_N], [F_N^T, 0]] gives downhill steps.
    let minimized: Vec<usize> = non_null
        .iter()
        .copied()
        .filter(|&k| k != followed)
        .collect();
    let m = minimized.len();
    let lambda_n = if m > 0 {
        let mut aug = vec![0.0f64; (m + 1) * (m + 1)];
        for (a, &k) in minimized.iter().enumerate() {
            aug[a * (m + 1) + a] = b(k);
            let fk = fcomp(k);
            aug[a * (m + 1) + m] = fk;
            aug[m * (m + 1) + a] = fk;
        }
        symmetric_eigh(&mat_from_row_major(m + 1, &aug)).values[0]
    } else {
        0.0
    };

    let mut dxi = vec![0.0f64; ndof];
    for (i, slot) in dxi.iter_mut().enumerate() {
        *slot += step_p * spec.eigenvectors[i * ndof + followed];
    }
    for &k in &minimized {
        let denom = b(k) - lambda_n;
        let step_k = if denom.abs() > 1e-12 {
            -fcomp(k) / denom
        } else {
            0.0
        };
        for (i, slot) in dxi.iter_mut().enumerate() {
            *slot += step_k * spec.eigenvectors[i * ndof + k];
        }
    }

    let n = norm(&dxi);
    if n > trust && n > 0.0 {
        let scale = trust / n;
        for v in &mut dxi {
            *v *= scale;
        }
    }
    dxi
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
                    .unwrap()
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
