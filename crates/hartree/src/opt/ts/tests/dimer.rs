//! Dimer-method convergence tests on the shared analytic surfaces.
//!
//! These pin the two sign-sensitive halves of the algorithm — the rotation's
//! minimum-curvature branch selection and the translation's *positive*
//! parallel-step sign — by requiring the search to climb to the right saddle and
//! verify exactly one negative mode. The displacements / `dimer_delta` chosen
//! here are documented per test; the RMSD tolerance is looser than P-RFO's
//! (`5e-3` vs `1e-3`) because the dimer estimates curvature from finite
//! differences and so converges to slightly lower precision.

use super::*;
use crate::core::{Atom, Element, Molecule};
use crate::ext::kabsch::kabsch_rmsd;
use crate::opt::OptStep;
use crate::opt::ts::{
    Flow, Progress, TsAlgorithm, TsError, TsOptions, TsStatus, find_transition_state,
};
use std::cell::Cell;

/// The reaction mode the dimer converged to (from verification) must align with
/// the surface's reaction direction `w1`, and the lowest eigenvalue must be
/// negative — i.e. the dimer followed the *softest* mode, not the highest. This
/// is the test that fails if the rotation picks the maximum-curvature branch.
#[test]
fn dimer_curvature_is_negative_at_aligned_axis() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.05 * basis[0][i] + 0.03 * basis[1][i];
        }
    }
    let mut surf = Quadratic { x0: x0.clone(), h };
    let mut opts = TsOptions::default();
    opts.algorithm = TsAlgorithm::Dimer;
    let result = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "status {:?} after {} iters",
        result.status,
        result.iterations
    );
    let v = result.verification.unwrap();
    assert_eq!(v.negative_eigenvalues.len(), 1);
    assert!(v.negative_eigenvalues[0] < 0.0);
    let overlap = mode_overlap(v.reaction_mode.as_ref().unwrap(), &basis[0]);
    assert!(overlap > 0.99, "reaction mode overlap with w1 = {overlap}");
}

/// A displaced quadratic saddle (along the reaction mode w1 and a transverse
/// mode w2): the dimer must rotate onto w1 and climb back to x0.
#[test]
fn dimer_converges_on_quadratic_saddle() {
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
    let mut surf = Quadratic { x0: x0.clone(), h };
    let mut opts = TsOptions::default();
    opts.algorithm = TsAlgorithm::Dimer;
    let result = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "status {:?} after {} iters",
        result.status,
        result.iterations
    );
    let rmsd = kabsch_rmsd(&result.positions, &x0).unwrap();
    assert!(rmsd < 5e-3, "RMSD to saddle = {rmsd:e}");
    assert_eq!(result.verification.unwrap().negative_eigenvalues.len(), 1);
}

/// The anharmonic double well: a moderate displacement (where the quartic term
/// is active) that the dimer still reliably climbs within `max_iter`.
#[test]
fn dimer_converges_on_anharmonic_saddle() {
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
    let mut opts = TsOptions::default();
    opts.algorithm = TsAlgorithm::Dimer;
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
        rmsd < 5e-3,
        "RMSD to saddle = {rmsd:e} after {} iters",
        result.iterations
    );
}

/// Heteronuclear (H, C, O): the distinct masses make the mass-weighting and the
/// internal-subspace projection non-trivial, stressing those code paths.
#[test]
fn dimer_converges_heteronuclear() {
    let x0 = h3_positions();
    let atoms = [1u32, 6, 8]
        .iter()
        .zip(&x0)
        .map(|(&z, &p)| Atom::new(Element::from_z(z).unwrap(), p))
        .collect();
    let mol = Molecule::new(atoms, 0, 1);
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.03 * basis[0][i] + 0.02 * basis[1][i];
        }
    }
    let mut moved = mol.clone();
    for (atom, p) in moved.atoms.iter_mut().zip(&start) {
        atom.position = *p;
    }
    let mut surf = Quadratic { x0: x0.clone(), h };
    let mut opts = TsOptions::default();
    opts.algorithm = TsAlgorithm::Dimer;
    let result = find_transition_state(&moved, &mut surf, &opts, None).unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "status {:?} after {} iters",
        result.status,
        result.iterations
    );
    let v = result.verification.unwrap();
    assert_eq!(v.negative_eigenvalues.len(), 1);
    assert!(v.negative_eigenvalues[0] < 0.0);
}

/// Seeding the dimer axis with the reaction-coordinate direction (the handoff a
/// two-endpoint/NEB guess provides) climbs to the saddle: the first axis starts
/// aligned with the reaction mode `w0` rather than the gradient, and the search
/// converges to the single-imaginary saddle. The seed is consumed up front (it is
/// not silently ignored).
#[test]
fn dimer_consumes_reaction_mode_seed() {
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
    let mol = h3_molecule(&start);

    // The reaction coordinate w0 as a Cartesian per-atom seed.
    let seed: Vec<[f64; 3]> = (0..mol.len())
        .map(|a| [basis[0][3 * a], basis[0][3 * a + 1], basis[0][3 * a + 2]])
        .collect();
    let mut opts = TsOptions::default();
    opts.algorithm = TsAlgorithm::Dimer;
    opts.reaction_mode_seed = Some(seed);

    let mut surf = Quadratic { x0: x0.clone(), h };
    let result = find_transition_state(&mol, &mut surf, &opts, None).unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "seeded dimer status {:?} after {} iters",
        result.status,
        result.iterations
    );
    let rmsd = kabsch_rmsd(&result.positions, &x0).unwrap();
    assert!(rmsd < 5e-3, "RMSD to saddle = {rmsd:e}");
    let v = result.verification.unwrap();
    assert_eq!(v.negative_eigenvalues.len(), 1);
    let overlap = mode_overlap(v.reaction_mode.as_ref().unwrap(), &basis[0]);
    assert!(overlap > 0.99, "reaction mode overlap with w0 = {overlap}");
}

/// A reaction-coordinate seed whose length does not match the molecule cannot be a
/// reaction coordinate; the dimer rejects it up front as a bad initial guess
/// (before any surface evaluation), exactly as P-RFO does.
#[test]
fn dimer_wrong_length_seed_is_bad_initial_guess() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut surf = Quadratic { x0: x0.clone(), h };
    let mut opts = TsOptions::default();
    opts.algorithm = TsAlgorithm::Dimer;
    // One atom direction, but the molecule has three atoms.
    opts.reaction_mode_seed = Some(vec![[1.0, 0.0, 0.0]]);
    let err = find_transition_state(&h3_molecule(&x0), &mut surf, &opts, None).unwrap_err();
    assert!(matches!(err, TsError::BadInitialGuess(_)), "got {err:?}");
}

/// A surface whose gradient is non-finite at the dimer's starting midpoint, but
/// finite at the displaced finite-difference probe points (so the verification
/// Hessian would build cleanly). The driver must reject the poisoned midpoint
/// gradient with a clear [`TsError::Numerical`] rather than churning through
/// `max_iter` and reporting a silent non-convergence.
struct NanGradAtPoint {
    inner: Quadratic,
    start: Vec<[f64; 3]>,
    eps: f64,
}
impl Surface for NanGradAtPoint {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        self.inner.energy(x)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        let d: f64 = x
            .iter()
            .zip(&self.start)
            .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).powi(2)))
            .sum::<f64>()
            .sqrt();
        if d < self.eps {
            Some(Ok(vec![[f64::NAN; 3]; x.len()]))
        } else {
            self.inner.analytic_gradient(x)
        }
    }
}

#[test]
fn dimer_nonfinite_gradient_is_numerical_not_silent() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut surf = NanGradAtPoint {
        inner: Quadratic { x0: x0.clone(), h },
        start: x0.clone(),
        // Smaller than the 5e-3 finite-difference step, so only the central
        // midpoint (where the driver reads the followed gradient) is poisoned.
        eps: 1e-3,
    };
    let mut opts = TsOptions::default();
    opts.algorithm = TsAlgorithm::Dimer;
    let err = find_transition_state(&h3_molecule(&x0), &mut surf, &opts, None).unwrap_err();
    assert!(
        matches!(err, TsError::Numerical(_)),
        "expected TsError::Numerical, got {err:?}"
    );
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

/// A `Flow::Stop`-after-2 observer halts the dimer before convergence, yielding
/// [`TsStatus::StoppedEarly`] with no verification.
#[test]
fn dimer_observer_stop_yields_stopped_early() {
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
    opts.algorithm = TsAlgorithm::Dimer;
    let observer = StopAfter {
        seen: Cell::new(0),
        limit: 2,
    };
    let result =
        find_transition_state(&h3_molecule(&start), &mut surf, &opts, Some(&observer)).unwrap();
    assert_eq!(result.status, TsStatus::StoppedEarly);
    assert!(result.verification.is_none());
}
