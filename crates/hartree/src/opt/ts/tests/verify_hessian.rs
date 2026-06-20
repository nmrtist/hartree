//! Two-tier verification (`TsOptions::verify_hessian`): `Strict` always finite-
//! differences a fresh Hessian for the post-convergence check; `Maintained`/`Auto`
//! reuse the maintained Bofill Hessian P-RFO already carries, skipping that
//! ≈6N-gradient build when the spectrum is unambiguous.

use super::*;
use crate::opt::ts::{TsOptions, TsStatus, VerifyHessian, find_transition_state};
use std::cell::Cell;

/// A surface decorator that counts analytic-gradient evaluations. Each
/// finite-difference Hessian column is one such call, so the count exposes whether a
/// verification finite-differenced a fresh Hessian.
struct CountingSurface<S: Surface> {
    inner: S,
    grad_calls: Cell<usize>,
}
impl<S: Surface> Surface for CountingSurface<S> {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        self.inner.energy(x)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        self.grad_calls.set(self.grad_calls.get() + 1);
        self.inner.analytic_gradient(x)
    }
}

/// Run the same quadratic-saddle search under a given verification mode, returning the
/// status and the number of gradient evaluations.
fn run(mode: VerifyHessian) -> (TsStatus, usize) {
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
    let mut surf = CountingSurface {
        inner: Quadratic { x0: x0.clone(), h },
        grad_calls: Cell::new(0),
    };
    let mut opts = TsOptions::default();
    opts.verify_hessian = mode;
    let r = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    (r.status, surf.grad_calls.get())
}

/// On a cleanly-classified saddle, `Maintained` and `Auto` skip the fresh
/// verification Hessian that `Strict` builds — saving exactly `2·ndof` gradient calls
/// (18 for three atoms) — while all three reach the same first-order saddle.
#[test]
fn maintained_and_auto_skip_the_fresh_verification_hessian() {
    let (s_status, s_calls) = run(VerifyHessian::Strict);
    let (m_status, m_calls) = run(VerifyHessian::Maintained);
    let (a_status, a_calls) = run(VerifyHessian::Auto);

    assert_eq!(s_status, TsStatus::Converged);
    assert_eq!(m_status, TsStatus::Converged);
    assert_eq!(a_status, TsStatus::Converged);

    // The only gradient-count difference is the post-convergence Hessian: Strict
    // finite-differences a fresh one (2·ndof = 18 calls for 3 atoms), the others
    // reuse the maintained Bofill Hessian.
    assert!(
        m_calls < s_calls,
        "maintained {m_calls} should be < strict {s_calls}"
    );
    assert_eq!(
        s_calls - m_calls,
        18,
        "expected the 2·ndof verification Hessian"
    );
    // The spectrum here is clean, so Auto behaves like Maintained (no fall-back).
    assert_eq!(a_calls, m_calls);
}

/// Backward compatibility: a `TsOptions` serialized before `verify_hessian` existed
/// deserializes with the default `Strict` (the historical behaviour).
#[test]
fn options_round_trip_defaults_verify_hessian() {
    let opts = TsOptions::default();
    assert_eq!(opts.verify_hessian, VerifyHessian::Strict);
    let json = serde_json::to_string(&opts).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value.as_object_mut().unwrap().remove("verify_hessian");
    let legacy: TsOptions = serde_json::from_value(value).unwrap();
    assert_eq!(legacy.verify_hessian, VerifyHessian::Strict);
}
