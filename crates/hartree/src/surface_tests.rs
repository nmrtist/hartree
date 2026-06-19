//! Tests for the private finite-difference Hessian helper on `HfSurface`. Relocated
//! out of `surface.rs` (via `#[path]`) to keep that file short; the module is still a
//! child of `surface`, so it can reach the private `fd_hessian_parallel`.

use super::*;
use crate::core::Molecule;
use crate::opt::{OptError, Surface};
use crate::scf::Reference;

/// Equilibrium water (Angstrom); paired with `set_scf_max_iter(1)` it cannot reach
/// self-consistency in a single iteration, so the SCF reports non-convergence.
fn water() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n").unwrap()
}

/// Capping the SCF at one iteration makes `energy` surface the structured
/// `OptError::ScfNotConverged` rather than a prose `Evaluation` string.
#[test]
fn energy_reports_scf_not_converged_when_iterations_capped() {
    let mol = water();
    let mut s = HfSurface::new(&mol, "sto-3g", Reference::Rhf).unwrap();
    s.set_scf_max_iter(1);
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();

    let err = s.energy(&pos).unwrap_err();
    assert!(
        matches!(err, OptError::ScfNotConverged { iterations: 1 }),
        "expected ScfNotConverged {{ iterations: 1 }}, got {err:?}"
    );
    // The surface really failed to converge (not a sham): the cached SCF, if any,
    // is non-convergent — but `energy` errors before caching, so confirm via a
    // fresh evaluation that the public contract is the typed error.
    assert!(
        s.last_scf().is_none(),
        "non-convergent point must not cache"
    );
}

/// The analytic-gradient path also short-circuits a capped SCF as
/// `OptError::ScfNotConverged` (the failure flows through `eval`'s `and_then`,
/// never re-wrapped as `Evaluation`).
#[test]
fn analytic_gradient_reports_scf_not_converged_when_iterations_capped() {
    let mol = water();
    let mut s = HfSurface::new(&mol, "sto-3g", Reference::Rhf).unwrap();
    s.set_scf_max_iter(1);
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();

    let err = s
        .analytic_gradient(&pos)
        .expect("RHF exposes an analytic gradient")
        .unwrap_err();
    assert!(
        matches!(err, OptError::ScfNotConverged { iterations: 1 }),
        "expected ScfNotConverged {{ iterations: 1 }}, got {err:?}"
    );
}

#[test]
fn fd_hessian_parallel_matches_serial_central_difference() {
    let mol = Molecule::from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
    let mut s = HfSurface::new(&mol, "sto-3g", Reference::Rhf).unwrap();

    let positions: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    let ndof = 3 * positions.len();
    let fd = 1e-3;

    let par = s.fd_hessian_parallel(&positions, fd).unwrap();

    // Serial reference: central differences of the analytic gradient, one dof at a
    // time. Column `dof` holds ∂g/∂x_dof.
    let mut serial = vec![0.0f64; ndof * ndof];
    for dof in 0..ndof {
        let mut plus = positions.clone();
        plus[dof / 3][dof % 3] += fd;
        let mut minus = positions.clone();
        minus[dof / 3][dof % 3] -= fd;

        let g_plus = s.analytic_gradient(&plus).unwrap().unwrap();
        let g_minus = s.analytic_gradient(&minus).unwrap().unwrap();

        for i in 0..ndof {
            let gp = g_plus[i / 3][i % 3];
            let gm = g_minus[i / 3][i % 3];
            // serial[i, dof] = column dof, row i.
            serial[i * ndof + dof] = (gp - gm) / (2.0 * fd);
        }
    }
    // Symmetrize to match the parallel routine.
    for i in 0..ndof {
        for j in (i + 1)..ndof {
            let avg = 0.5 * (serial[i * ndof + j] + serial[j * ndof + i]);
            serial[i * ndof + j] = avg;
            serial[j * ndof + i] = avg;
        }
    }

    let mut max_diff = 0.0f64;
    for k in 0..ndof * ndof {
        max_diff = max_diff.max((par[k] - serial[k]).abs());
    }
    assert!(
        max_diff < 1e-8,
        "parallel vs serial FD Hessian disagree: max abs diff {max_diff}"
    );
}
