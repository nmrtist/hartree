use super::*;
use crate::core::{Atom, Element, Molecule};
use crate::ext::kabsch::kabsch_rmsd;
use crate::opt::ts::numerics::{
    gram_schmidt, mw_projected_hessian, non_null_modes, trans_rot_vectors,
};
use crate::opt::ts::{TsOptions, TsStatus, find_transition_state, verify_saddle};

/// Three H along the z-axis (Bohr).
fn linear_positions() -> Vec<[f64; 3]> {
    vec![[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]
}

fn linear_molecule(x: &[[f64; 3]]) -> Molecule {
    let atoms = x
        .iter()
        .map(|&p| Atom::new(Element::from_z(1).unwrap(), p))
        .collect();
    Molecule::new(atoms, 0, 2)
}

/// Exercises the 3N-5 projection and the gram_schmidt redundant-rotation drop.
/// Equal masses keep the analytic expectation exact: the mass-COM coincides with
/// the geometric centroid, so the real-mass trans/rot projector matches the
/// kernel exactly (a heteronuclear linear case would leak through the
/// rotation-center mismatch and make the non-null count flaky).
#[test]
fn linear_saddle_3n_minus_5_projection() {
    let x = linear_positions();
    // Equal masses; masses = vec![1.0; 3] suffices since the species is uniform.
    let masses = vec![1.0; 3];

    // Redundant-rotation drop: linear loses one rotation (5 trans/rot), bent keeps 6.
    assert_eq!(gram_schmidt(&trans_rot_vectors(&x, &masses)).len(), 5);
    assert_eq!(
        gram_schmidt(&trans_rot_vectors(&h3_positions(), &masses)).len(),
        6
    );

    // 3N - 5 = 4 internal coordinates for the linear molecule.
    let internal = internal_basis(&x);
    assert_eq!(internal.len(), 4);
    let h = hessian_from(&internal, &[-0.4, 0.5, 0.7, 0.9]);

    // Exactly 3N-5 internal modes survive the trans/rot projection.
    assert_eq!(
        non_null_modes(&mw_projected_hessian(&x, &masses, &h)).len(),
        4
    );

    let mol = linear_molecule(&x);
    let opts = TsOptions::default();
    let mut surf = Quadratic {
        x0: x.clone(),
        h: h.clone(),
    };
    let v = verify_saddle(&mol, &mut surf, &x, &opts).unwrap();
    assert!(
        v.is_first_order_saddle(),
        "expected one negative mode, got {:?}",
        v.negative_eigenvalues
    );
    assert_eq!(v.negative_eigenvalues.len(), 1);
    assert!(v.imaginary_frequency_cm1.unwrap() < 0.0);
    let overlap = mode_overlap(&v.reaction_mode.unwrap(), &internal[0]);
    assert!(overlap > 0.99, "reaction mode overlap with w1 = {overlap}");

    // Climb back to the saddle from a displaced start.
    let mut start = x.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.05 * internal[0][i] + 0.03 * internal[1][i];
        }
    }
    let mut surf = Quadratic {
        x0: x.clone(),
        h: h.clone(),
    };
    let result = find_transition_state(&linear_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "status {:?} after {} iters",
        result.status,
        result.iterations
    );
    let rmsd = kabsch_rmsd(&result.positions, &x).unwrap();
    assert!(rmsd < 1e-3, "RMSD to saddle = {rmsd:e}");
}
