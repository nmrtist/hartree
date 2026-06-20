use super::*;
use crate::ext::kabsch::kabsch_rmsd;
use crate::opt::OptStep;
use crate::opt::ts::{Flow, Progress, TsOptions, TsStatus, find_transition_state};
use std::cell::Cell;

#[test]
fn prfo_converges_on_quadratic_saddle() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    // Start displaced off the saddle (along the reaction mode w1 and a transverse
    // mode w2) so the P-RFO actually has to climb back to x0.
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.06 * basis[0][i] + 0.04 * basis[1][i];
        }
    }
    let mut surf = Quadratic { x0: x0.clone(), h };
    let result =
        find_transition_state(&h3_molecule(&start), &mut surf, &TsOptions::default(), None)
            .unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "status {:?} after {} iters",
        result.status,
        result.iterations
    );
    // The surface depends only on internal coordinates, so its saddle is the
    // whole rigid-body manifold {x0 + translation/rotation}; compare invariantly.
    let rmsd = kabsch_rmsd(&result.positions, &x0).unwrap();
    assert!(rmsd < 1e-3, "RMSD to saddle = {rmsd:e}");
    assert_eq!(result.verification.unwrap().negative_eigenvalues.len(), 1);
}

#[test]
fn prfo_converges_on_anharmonic_saddle() {
    let x_ref = h3_positions();
    let basis = internal_basis(&x_ref);
    // A sizeable displacement (where the quartic term matters) so the run
    // exercises several P-RFO steps with Bofill/recompute Hessian maintenance.
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
    opts.recalc_hessian = 5;
    let result = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "status {:?} after {} iters",
        result.status,
        result.iterations
    );
    // Rigid-body-invariant comparison: the converged saddle may be a rotated /
    // translated copy of x_ref (the surface depends only on internal coordinates).
    let rmsd = kabsch_rmsd(&result.positions, &x_ref).unwrap();
    assert!(
        rmsd < 1e-3,
        "RMSD to saddle = {rmsd:e} after {} iters",
        result.iterations
    );
}

/// Pure Bofill quasi-Newton (recalc_hessian == 0, the default): no Hessian
/// recompute, so the run leans entirely on the indefinite-preserving update from a
/// single starting Hessian. A moderate displacement keeps the Bofill model valid.
#[test]
fn prfo_pure_bofill_on_curved_surface() {
    let x_ref = h3_positions();
    let basis = internal_basis(&x_ref);
    let mut start = x_ref.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.20 * basis[0][i] + 0.10 * basis[1][i] - 0.05 * basis[2][i];
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
    // Default options: recalc_hessian == 0 (pure Bofill).
    let opts = TsOptions::default();
    assert_eq!(opts.recalc_hessian, 0);
    let result = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "status {:?} after {} iters",
        result.status,
        result.iterations
    );
    let rmsd = kabsch_rmsd(&result.positions, &x_ref).unwrap();
    assert!(
        rmsd < 1e-3,
        "RMSD to saddle = {rmsd:e} after {} iters",
        result.iterations
    );
}

/// A second-order saddle (two negative curvatures): starting exactly at the
/// stationary point converges geometrically at iteration 1, and `verify_saddle`
/// must report the wrong imaginary-mode count.
#[test]
fn wrong_imaginary_mode_count_end_to_end() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, -0.3, 0.9]);
    let mut surf = Quadratic { x0: x0.clone(), h };
    // Start exactly at the stationary point: gradient is zero, so it converges
    // immediately and runs verification on a genuine second-order saddle.
    let result =
        find_transition_state(&h3_molecule(&x0), &mut surf, &TsOptions::default(), None).unwrap();
    assert_eq!(result.status, TsStatus::WrongImaginaryModeCount);
    assert!(result.verification.is_some());
    assert_eq!(
        result
            .verification
            .as_ref()
            .unwrap()
            .negative_eigenvalues
            .len(),
        2
    );
    assert!(result.irc.is_none());
}

/// Capping `max_iter` at 2 from a large displacement: too few steps to reach the
/// saddle, so the run gives up with [`TsStatus::NotConverged`] and never verifies.
#[test]
fn not_converged_hits_max_iter() {
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
    assert!(result.verification.is_none());
    assert!(result.irc.is_none());
}

struct StopAfter {
    seen: Cell<usize>,
    limit: usize,
}
impl Progress for StopAfter {
    fn step(&self, _s: &OptStep) -> Flow {
        let n = self.seen.get() + 1;
        self.seen.set(n);
        if n >= self.limit {
            Flow::Stop
        } else {
            Flow::Continue
        }
    }
}

/// An observer that returns [`Flow::Stop`] after two iterations halts a run that
/// would otherwise need many more steps, yielding [`TsStatus::StoppedEarly`].
#[test]
fn observer_stop_yields_stopped_early() {
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
    let observer = StopAfter {
        seen: Cell::new(0),
        limit: 2,
    };
    let result = find_transition_state(
        &h3_molecule(&start),
        &mut surf,
        &TsOptions::default(),
        Some(&observer),
    )
    .unwrap();
    assert_eq!(result.status, TsStatus::StoppedEarly);
    assert!(result.verification.is_none());
}

/// A SCF non-convergence on the real `HfSurface` survives the `TsError` wrapping:
/// it arrives as `TsError::SurfaceEvaluation(OptError::ScfNotConverged { .. })`, so
/// the SCF-failure type is branchable without parsing prose. Water/sto-3g cannot
/// converge in one SCF iteration; the very first surface evaluation in the driver
/// raises the error.
#[test]
fn scf_non_convergence_propagates_through_find_transition_state() {
    use crate::opt::ts::TsError;
    use crate::scf::Reference;
    use crate::surface::HfSurface;

    let mol = crate::core::Molecule::from_xyz(
        "3\nwater\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n",
    )
    .unwrap();
    let mut surface = HfSurface::new(&mol, "sto-3g", Reference::Rhf).unwrap();
    surface.set_scf_max_iter(1);

    let err = find_transition_state(&mol, &mut surface, &TsOptions::default(), None).unwrap_err();
    assert!(
        matches!(
            err,
            TsError::SurfaceEvaluation(OptError::ScfNotConverged { iterations: 1 })
        ),
        "expected SurfaceEvaluation(ScfNotConverged {{ iterations: 1 }}), got {err:?}"
    );
}

#[test]
fn convergence_force_ignores_rigid_body_residue() {
    use crate::opt::ts::numerics::{force_norms, masses_of, projected_force_norms};

    let x = h3_positions();
    let masses = masses_of(&h3_molecule(&x));
    let internal = internal_basis(&x);

    // A pure internal force along one internal direction: the real progress signal.
    let c = 3.0e-3;
    let w = &internal[0];
    let g_internal: Vec<[f64; 3]> = (0..x.len())
        .map(|a| [c * w[3 * a], c * w[3 * a + 1], c * w[3 * a + 2]])
        .collect();

    // Add a rigid-body translation (a net force that no real step removes), the
    // kind of residue finite-difference gradients carry.
    let t = [5.0e-4, -3.0e-4, 2.0e-4];
    let g_contaminated: Vec<[f64; 3]> = g_internal
        .iter()
        .map(|gi| [gi[0] + t[0], gi[1] + t[1], gi[2] + t[2]])
        .collect();

    let (raw_max, _) = force_norms(&g_contaminated);
    let (proj_max, proj_rms) = projected_force_norms(&g_contaminated, &masses, &x);
    let (int_max, int_rms) = force_norms(&g_internal);

    // The raw force is inflated by the translation; projecting it out recovers the
    // internal force the convergence test should actually be measuring.
    assert!(
        raw_max > proj_max + 1e-6,
        "raw {raw_max} should exceed projected {proj_max}"
    );
    assert!(
        (proj_max - int_max).abs() < 1e-9 && (proj_rms - int_rms).abs() < 1e-9,
        "projected ({proj_max}, {proj_rms}) should match internal ({int_max}, {int_rms})"
    );
}

/// L2 distance between two geometries (Bohr).
fn geom_distance(x: &[[f64; 3]], y: &[[f64; 3]]) -> f64 {
    x.iter()
        .zip(y)
        .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).powi(2)))
        .sum::<f64>()
        .sqrt()
}

/// A surface decorator that returns [`OptError::ScfNotConverged`] whenever a trial
/// step moves more than `max_step` from the last geometry whose energy it accepted
/// — modelling an SCF that fails when a climbing step overshoots. Only `energy` is
/// gated; the gradient stays valid so the finite-difference Hessian probes (which
/// take small displacements) are unaffected. The driver must shrink the trust
/// radius and retry to make progress; aborting on the first failure could never
/// reach the saddle.
struct BoundedStep<S: Surface> {
    inner: S,
    last_ok: Vec<[f64; 3]>,
    max_step: f64,
    rejections: usize,
}
impl<S: Surface> Surface for BoundedStep<S> {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        if geom_distance(x, &self.last_ok) > self.max_step {
            self.rejections += 1;
            return Err(OptError::ScfNotConverged { iterations: 1 });
        }
        let e = self.inner.energy(x)?;
        self.last_ok = x.to_vec();
        Ok(e)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        self.inner.analytic_gradient(x)
    }
}

/// Step backtracking recovers from a recoverable SCF failure: when the full
/// climbing step overshoots into the surface's failure region, the driver shrinks
/// the trust radius and retries from the same point, and still converges to the
/// saddle. Without backtracking the first overshoot would abort the search.
#[test]
fn step_backtracking_recovers_from_scf_failure() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.10 * basis[0][i] + 0.06 * basis[1][i];
        }
    }
    let mut surf = BoundedStep {
        inner: Quadratic { x0: x0.clone(), h },
        last_ok: start.clone(),
        max_step: 0.07,
        rejections: 0,
    };
    let result =
        find_transition_state(&h3_molecule(&start), &mut surf, &TsOptions::default(), None)
            .unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "status {:?} after {} iters",
        result.status,
        result.iterations
    );
    let rmsd = kabsch_rmsd(&result.positions, &x0).unwrap();
    assert!(rmsd < 1e-3, "RMSD to saddle = {rmsd:e}");
    assert!(
        surf.rejections > 0,
        "backtracking was never exercised (no SCF rejections recorded)"
    );
}

/// A surface decorator whose SCF fails for any geometry more than `radius` from the
/// fixed `start` (only `energy` is gated, so the start-point Hessian still builds).
/// No real step can ever be accepted.
struct ScfWall<S: Surface> {
    inner: S,
    start: Vec<[f64; 3]>,
    radius: f64,
}
impl<S: Surface> Surface for ScfWall<S> {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        if geom_distance(x, &self.start) > self.radius {
            return Err(OptError::ScfNotConverged { iterations: 9 });
        }
        self.inner.energy(x)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        self.inner.analytic_gradient(x)
    }
}

/// The soft-failure contract: when every trial step's SCF fails and the retries are
/// exhausted, the search returns `Ok(NotConverged)` with the best-so-far geometry
/// rather than propagating the failure as a `TsError`. `min_trust` is floored above
/// the failure radius so the backtracked step never shrinks small enough to land
/// inside the reachable region.
#[test]
fn scf_failure_on_every_step_yields_soft_not_converged() {
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
    let mut surf = ScfWall {
        inner: Quadratic { x0: x0.clone(), h },
        start: start.clone(),
        radius: 0.02,
    };
    let mut opts = TsOptions::default();
    opts.min_trust = 0.05;
    opts.max_step_retries = 3;
    let result = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(result.status, TsStatus::NotConverged);
    assert!(result.verification.is_none());
    // Best-so-far is the (only reachable) starting geometry, not garbage.
    let rmsd = kabsch_rmsd(&result.positions, &start).unwrap();
    assert!(
        rmsd < 1e-9,
        "best-so-far should be the start, RMSD {rmsd:e}"
    );
}

/// Backward compatibility: a `TsOptions` serialized before `max_step_retries`
/// existed (no such key) still deserializes, defaulting the field — the
/// `#[serde(default)]` round-trip the new knob promises.
#[test]
fn options_round_trip_defaults_new_step_retries_field() {
    let opts = TsOptions::default();
    let json = serde_json::to_string(&opts).unwrap();
    let back: TsOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(back.max_step_retries, opts.max_step_retries);

    // Drop the key to mimic an options object written before the field was added.
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value.as_object_mut().unwrap().remove("max_step_retries");
    let legacy: TsOptions = serde_json::from_value(value).unwrap();
    assert_eq!(legacy.max_step_retries, 6);
}

/// A surface whose gradient is always non-finite, so the finite-difference Hessian
/// is `NaN` and its eigendecomposition cannot be formed.
struct NanGradient {
    inner: Quadratic,
}
impl Surface for NanGradient {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        self.inner.energy(x)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        Some(Ok(vec![[f64::NAN; 3]; x.len()]))
    }
}

/// A non-finite Hessian surfaces as a recoverable [`TsError::Numerical`] — the
/// checked eigensolver and its one self-healing recompute both reject it — instead
/// of panicking out of the eigendecomposition.
#[test]
fn nonfinite_hessian_surfaces_as_numerical_error_not_panic() {
    use crate::opt::ts::TsError;
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut surf = NanGradient {
        inner: Quadratic { x0: x0.clone(), h },
    };
    let err = find_transition_state(&h3_molecule(&x0), &mut surf, &TsOptions::default(), None)
        .unwrap_err();
    assert!(
        matches!(err, TsError::Numerical(_)),
        "expected TsError::Numerical, got {err:?}"
    );
}
