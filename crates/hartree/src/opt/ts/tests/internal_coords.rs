//! Internal-coordinate climb frame ([`Coordinates::Internal`]) tests: the
//! redundant-internal Hessian transform, the completeness/rank guard that drives the
//! Cartesian fallback, and an end-to-end saddle search that must reach the same
//! stationary point the Cartesian frame does.

use super::*;
use crate::opt::internals::{self, Internal};
use crate::opt::ts::{Coordinates, TsOptions, TsStatus, find_transition_state};

/// The transform `H_q = G⁻ B Hₓ Bᵀ G⁻` carries a Cartesian curvature into the internal
/// metric. A Cartesian Hessian that is `k` along a diatomic's normalized stretch maps
/// to a bond-coordinate curvature of `k/2`: a unit change in the bond *length* is a
/// Cartesian displacement of each atom by `½` along the axis (Euclidean norm `1/√2`),
/// so the energy `½k‖Δx‖²` reads as `½(k/2)Δq²` in the bond coordinate. Pinning the
/// `k/2` proves the metric factor is applied, not dropped or doubled.
#[test]
fn internal_hessian_carries_bond_curvature_with_metric() {
    let x = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]];
    let defs = vec![Internal::Bond(0, 1)];
    let k = 0.37;
    // Normalized Cartesian bond-stretch direction (atoms move oppositely along z).
    let inv = 1.0 / 2.0_f64.sqrt();
    let e = [0.0, 0.0, -inv, 0.0, 0.0, inv];
    let mut hx = vec![0.0; 36];
    for i in 0..6 {
        for j in 0..6 {
            hx[i * 6 + j] = k * e[i] * e[j];
        }
    }
    let hq = internals::internal_hessian(&defs, &x, &hx);
    assert_eq!(hq.len(), 1);
    assert!(
        (hq[0] - k / 2.0).abs() < 1e-9,
        "internal bond curvature {} != expected {}",
        hq[0],
        k / 2.0
    );
}

/// The transform sends the redundant null space to ~zero eigenvalues and preserves the
/// negative reaction mode: a Cartesian saddle Hessian (one negative mode) on a complete
/// internal set yields an internal Hessian with exactly one negative eigenvalue and no
/// spurious negative from redundancy.
#[test]
fn internal_hessian_preserves_one_negative_mode() {
    let x = h3_positions();
    let basis = internal_basis(&x);
    let hx = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let defs = internals::generate(&h3_molecule(&x));
    let hq = internals::internal_hessian(&defs, &x, &hx);
    let nq = defs.len();
    let eig = crate::linalg::symmetric_eigh(&crate::linalg::mat_from_row_major(nq, &hq));
    let neg = eig.values.iter().filter(|&&l| l < -1e-6).count();
    assert_eq!(neg, 1, "internal spectrum {:?}", eig.values);
}

/// The rank guard recognizes a complete redundant set (`rank G ≥ 3N − 6`) and rejects
/// an incomplete one, which is what makes the internal frame fall back to Cartesian
/// rather than run a search that cannot move along a missing coordinate.
#[test]
fn internal_rank_flags_complete_and_incomplete_sets() {
    let x = h3_positions();
    let full = internals::generate(&h3_molecule(&x));
    assert_eq!(
        internals::internal_rank(&full, &x),
        3,
        "H3 (bent) spans 3N−6 = 3 internal DOF"
    );
    let partial = vec![Internal::Bond(0, 1)];
    assert!(
        internals::internal_rank(&partial, &x) < 3,
        "a single bond cannot span 3 internal DOF"
    );
}

/// End-to-end: a P-RFO search in internal coordinates reaches the same quadratic
/// saddle the Cartesian frame does. The stationary point of `½(x−x₀)ᵀH(x−x₀)` is `x₀`
/// with energy 0, so convergence to a verified first-order saddle at ~zero energy
/// proves the internal climb (transform → eigenvector following → B-matrix
/// back-transformation) lands on the saddle.
#[test]
fn internal_frame_finds_quadratic_saddle() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    start[0][0] += 0.15;
    start[1][1] -= 0.10;
    start[2][0] -= 0.08;
    let start_mol = h3_molecule(&start);
    let mut surf = Quadratic { x0: x0.clone(), h };

    let mut opts = TsOptions::default();
    opts.coordinates = Coordinates::Internal;
    let res = find_transition_state(&start_mol, &mut surf, &opts, None).unwrap();

    assert_eq!(
        res.status,
        TsStatus::Converged,
        "internal-frame search did not converge: {:?}",
        res.diagnostic
    );
    assert!(
        res.energy.abs() < 1e-6,
        "converged off the quadratic saddle (energy {})",
        res.energy
    );
    assert!(res.verification.unwrap().is_first_order_saddle());
}
