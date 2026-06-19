//! The [`TsResult::diagnostic`] reason populated for non-converged outcomes: each
//! distinct stopping cause should yield a `Some`, non-empty, cause-specific note.

use super::*;
use crate::opt::ts::{TsOptions, TsStatus, find_transition_state};

/// A large displacement with `max_iter` capped low cannot reach the saddle, so the
/// run stops `NotConverged` and the diagnostic names the iteration-cap cause.
#[test]
fn not_converged_max_iter_sets_reason() {
    let x_ref = h3_positions();
    let basis = internal_basis(&x_ref);
    let mut start = x_ref.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.35 * basis[0][i] + 0.15 * basis[1][i] - 0.10 * basis[2][i];
        }
    }
    let mut surf = Anharmonic {
        x_ref: x_ref.clone(),
        w: basis,
        a: 0.5,
        b: 1.0,
        k2: 0.7,
        k3: 0.9,
    };
    let mut opts = TsOptions::default();
    opts.max_iter = 2;
    let result = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(result.status, TsStatus::NotConverged);
    let reason = result
        .diagnostic
        .as_deref()
        .expect("a non-converged run carries a diagnostic reason");
    assert!(!reason.is_empty(), "the reason should be non-empty");
    assert!(
        reason.contains("max_iter"),
        "the max-iter reason should name the cause, got {reason:?}"
    );
}

/// A converged first-order saddle carries no diagnostic — the field is reserved for
/// non-success outcomes.
#[test]
fn converged_saddle_has_no_diagnostic() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.06 * basis[0][i] + 0.04 * basis[1][i];
        }
    }
    let mut surf = Quadratic { x0, h };
    let result =
        find_transition_state(&h3_molecule(&start), &mut surf, &TsOptions::default(), None)
            .unwrap();
    assert_eq!(result.status, TsStatus::Converged);
    assert!(result.diagnostic.is_none());
}

/// A geometry that converges to a second-order saddle carries a diagnostic that
/// names the wrong-mode-count cause.
#[test]
fn wrong_mode_count_sets_reason() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, -0.3, 0.9]);
    let mut surf = Quadratic { x0: x0.clone(), h };
    let result =
        find_transition_state(&h3_molecule(&x0), &mut surf, &TsOptions::default(), None).unwrap();
    assert_eq!(result.status, TsStatus::WrongImaginaryModeCount);
    let reason = result
        .diagnostic
        .as_deref()
        .expect("a wrong-mode-count run carries a diagnostic reason");
    assert!(!reason.is_empty());
    assert!(
        reason.contains("saddle"),
        "the reason should describe the mode count, got {reason:?}"
    );
}
