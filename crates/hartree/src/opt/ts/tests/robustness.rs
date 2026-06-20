//! Numerical-fault robustness: non-finite surface values are surfaced as the
//! recoverable [`TsError::Numerical`] (never a panic), and the step-acceptance
//! overshoot test only fires on a genuine gross overshoot — not on the benign
//! energy *drop* that an uphill eigenvector-following step routinely produces.

use super::*;
use crate::opt::ts::step::is_pathological;
use crate::opt::ts::{TsError, TsOptions, find_transition_state};

/// L2 distance between two geometries (Bohr).
fn dist(x: &[[f64; 3]], y: &[[f64; 3]]) -> f64 {
    x.iter()
        .zip(y)
        .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).powi(2)))
        .sum::<f64>()
        .sqrt()
}

/// A surface whose gradient is non-finite at the starting point but finite at the
/// (displaced) finite-difference probe points, so the Hessian builds cleanly. The
/// driver must reject the non-finite gradient before it reaches the panicking inner
/// eigensolver in `prfo_step`; without that guard this geometry panics rather than
/// returning [`TsError::Numerical`].
struct NanGradAtPoint<S: Surface> {
    inner: S,
    start: Vec<[f64; 3]>,
    eps: f64,
}
impl<S: Surface> Surface for NanGradAtPoint<S> {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        self.inner.energy(x)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        if dist(x, &self.start) < self.eps {
            Some(Ok(vec![[f64::NAN; 3]; x.len()]))
        } else {
            self.inner.analytic_gradient(x)
        }
    }
}

#[test]
fn nonfinite_gradient_at_a_point_is_numerical_not_panic() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut surf = NanGradAtPoint {
        inner: Quadratic { x0: x0.clone(), h },
        start: x0.clone(),
        // Smaller than the 5e-3 finite-difference step, so only the central point
        // (where the driver reads the followed gradient) is poisoned.
        eps: 1e-3,
    };
    let err = find_transition_state(&h3_molecule(&x0), &mut surf, &TsOptions::default(), None)
        .unwrap_err();
    assert!(
        matches!(err, TsError::Numerical(_)),
        "expected TsError::Numerical, got {err:?}"
    );
}

/// A surface that returns a non-finite *energy* (as `Ok`, not `Err`) once a step
/// moves beyond `radius` — modelling an SCF that reports success but yields garbage.
/// `min_trust` is floored above `radius` so the backtracked step can never land back
/// inside the finite region; the run must report [`TsError::Numerical`] rather than
/// committing a `NaN` energy and reporting a clean status.
struct NanEnergyOnStep<S: Surface> {
    inner: S,
    start: Vec<[f64; 3]>,
    radius: f64,
}
impl<S: Surface> Surface for NanEnergyOnStep<S> {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        if dist(x, &self.start) > self.radius {
            return Ok(f64::NAN);
        }
        self.inner.energy(x)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        self.inner.analytic_gradient(x)
    }
}

#[test]
fn nonfinite_energy_on_a_step_is_numerical_not_committed() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.06 * basis[0][i];
        }
    }
    let mut surf = NanEnergyOnStep {
        inner: Quadratic { x0: x0.clone(), h },
        start: start.clone(),
        radius: 0.02,
    };
    let mut opts = TsOptions::default();
    opts.min_trust = 0.05;
    opts.max_step_retries = 3;
    let err = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap_err();
    assert!(
        matches!(err, TsError::Numerical(_)),
        "expected TsError::Numerical, got {err:?}"
    );
}

/// The overshoot criterion: P-RFO climbs the followed mode *uphill*, so a positive
/// predicted change and a small genuine energy drop are routine, not failures. Only
/// a gross opposite-direction overshoot (|actual| beyond twice |predicted|) counts.
#[test]
fn is_pathological_only_flags_gross_overshoots() {
    // Exact model agreement (same sign): never pathological.
    assert!(!is_pathological(1e-3, 1e-3));
    // Model predicts an uphill rise, surface drops only slightly (|actual| <
    // |predicted|): benign progress, NOT a rejection (the false-reject this fixes).
    assert!(!is_pathological(-0.8e-3, 1e-3));
    assert!(!is_pathological(-1.9e-3, 1e-3));
    // A gross opposite-direction overshoot (|actual| > 2|predicted|): pathological.
    assert!(is_pathological(-3e-3, 1e-3));
    // Below the noise floor: ignored.
    assert!(!is_pathological(-1e-9, 1e-9));
}
