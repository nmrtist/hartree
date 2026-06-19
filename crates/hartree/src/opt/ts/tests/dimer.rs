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
use crate::opt::ts::{Flow, Progress, TsAlgorithm, TsOptions, TsStatus, find_transition_state};
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
