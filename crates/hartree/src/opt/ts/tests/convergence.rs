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
