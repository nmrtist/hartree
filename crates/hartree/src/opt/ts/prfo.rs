//! Partitioned rational-function optimization (P-RFO / eigenvector following).
//!
//! The search runs in mass-weighted Cartesian coordinates with translations and
//! rotations projected out (the [`crate::props::frequencies`] frame). A single
//! climb — diagonalize the projected Hessian, follow one mode uphill while
//! minimizing the rest, maintain the Hessian by a Bofill update, with step
//! backtracking — lives in [`super::climb`]; this file orchestrates it. After a
//! climb converges geometrically the shared [`verify_saddle`](super::verify_saddle)
//! check counts negative modes; if the count is wrong and a reaction-coordinate
//! seed is available, the search displaces off that point along the seed (descending
//! the spurious negative modes) and re-climbs, up to [`TsOptions::max_recover`] times.

use super::climb::{ClimbStop, run_climb};
use super::numerics::{
    add_step, column, fd_hessian, gram_schmidt, masses_of, mw_projected_hessian, norm, overlap,
    positions_of, trans_rot_vectors, unmass_weight_step,
};
use super::{Progress, TsError, TsOptions, TsResult, TsStatus, verify_saddle};
use crate::core::Molecule;
use crate::opt::{OptError, OptStep, Surface};

pub(super) fn run_prfo<S: Surface>(
    molecule: &Molecule,
    surface: &mut S,
    options: &TsOptions,
    progress: Option<&dyn Progress>,
) -> Result<TsResult, TsError> {
    let masses = masses_of(molecule);
    let natom = molecule.len();
    let ndof = 3 * natom;

    let x0 = positions_of(molecule);
    let n_tr = gram_schmidt(&trans_rot_vectors(&x0, &masses)).len();
    if ndof < n_tr + 1 {
        return Err(TsError::BadInitialGuess(format!(
            "{natom} atom(s) leave no internal coordinate to follow ({ndof} \
             Cartesian DOF, {n_tr} translation/rotation modes)"
        )));
    }

    let seed_mw = mass_weighted_seed(options, &masses, natom)?;

    let mut history: Vec<OptStep> = Vec::new();
    let mut iter_counter = 0usize;
    let mut x_start = x0;

    // Climb, then (if it converged to the wrong negative-mode count and a seed is
    // available) re-seed off that point and climb again, up to `max_recover` times.
    for attempt in 0..=options.max_recover {
        let climb = run_climb(
            surface,
            options,
            progress,
            &masses,
            &x_start,
            seed_mw.as_deref(),
            &mut history,
            &mut iter_counter,
        )?;

        // Only a geometric convergence proceeds to verification; the soft outcomes
        // return their best-so-far geometry directly.
        let (x, energy) = match climb.stop {
            ClimbStop::StoppedEarly => {
                return Ok(soft_result(
                    TsStatus::StoppedEarly,
                    climb.x,
                    climb.energy,
                    iter_counter,
                    history,
                ));
            }
            ClimbStop::NotConverged => {
                return Ok(soft_result(
                    TsStatus::NotConverged,
                    climb.x,
                    climb.energy,
                    iter_counter,
                    history,
                ));
            }
            ClimbStop::ConvergedGeom => (climb.x, climb.energy),
        };

        // Verify with a fresh finite-difference Hessian, not the maintained one.
        let verification = verify_saddle(molecule, surface, &x, options)?;
        if verification.is_first_order_saddle() {
            let irc = if options.confirm_irc {
                match &verification.reaction_mode {
                    Some(mode) => {
                        match super::irc::irc_endpoints(surface, &x, mode, &masses, energy, options)
                        {
                            Ok(endpoints) => Some(endpoints),
                            // A recoverable SCF failure during the (purely confirmatory)
                            // IRC trace must not discard the converged saddle: report it
                            // without endpoints rather than turning success into an error.
                            Err(TsError::SurfaceEvaluation(OptError::ScfNotConverged {
                                ..
                            })) => None,
                            Err(e) => return Err(e),
                        }
                    }
                    None => None,
                }
            } else {
                None
            };
            // Leave the surface cache at the returned saddle, not the last verification
            // / IRC displacement, so a caller's `last_scf()` is the saddle wavefunction.
            let _ = surface.energy(&x)?;
            return Ok(TsResult {
                positions: x,
                energy,
                status: TsStatus::Converged,
                iterations: iter_counter,
                history,
                verification: Some(verification),
                irc,
            });
        }

        // Wrong negative-mode count. If a seed pins the reaction coordinate and the
        // recovery budget is not spent, displace off this point (descending the
        // spurious negative modes, climbing the seed) and try again.
        if attempt < options.max_recover {
            if let Some(seed) = seed_mw.as_deref() {
                x_start = recovery_perturbation(surface, &x, &masses, options, seed)?;
                continue;
            }
        }
        // Recovery is unavailable or exhausted: report the wrong-mode point, leaving
        // the surface cache at it (not the last verification displacement) so a
        // caller's `last_scf()` matches the returned geometry.
        let _ = surface.energy(&x)?;
        return Ok(TsResult {
            positions: x,
            energy,
            status: TsStatus::WrongImaginaryModeCount,
            iterations: iter_counter,
            history,
            verification: Some(verification),
            irc: None,
        });
    }
    unreachable!("the recovery loop returns on every branch");
}

/// A soft (non-verified) outcome carried on `Ok`: best-so-far geometry, no
/// verification or IRC. Shared by the [`StoppedEarly`](TsStatus::StoppedEarly) and
/// [`NotConverged`](TsStatus::NotConverged) exits.
fn soft_result(
    status: TsStatus,
    positions: Vec<[f64; 3]>,
    energy: f64,
    iterations: usize,
    history: Vec<OptStep>,
) -> TsResult {
    TsResult {
        positions,
        energy,
        status,
        iterations,
        history,
        verification: None,
        irc: None,
    }
}

/// Bring the optional Cartesian reaction-coordinate seed into the mass-weighted
/// frame the Hessian spectrum lives in: a displacement mass-weights by √m (the
/// inverse of a gradient) and is then normalized. An all-zero seed carries no
/// direction (`None`); a seed of the wrong length cannot be a reaction coordinate.
fn mass_weighted_seed(
    options: &TsOptions,
    masses: &[f64],
    natom: usize,
) -> Result<Option<Vec<f64>>, TsError> {
    let Some(seed) = &options.reaction_mode_seed else {
        return Ok(None);
    };
    if seed.len() != natom {
        return Err(TsError::BadInitialGuess(format!(
            "reaction_mode_seed has {} atom direction(s) but the molecule has {natom}",
            seed.len()
        )));
    }
    let mut q = vec![0.0f64; 3 * natom];
    for (a, v) in seed.iter().enumerate() {
        let s = masses[a].sqrt();
        for c in 0..3 {
            q[3 * a + c] = v[c] * s;
        }
    }
    let nrm = norm(&q);
    Ok((nrm > 1e-12).then(|| {
        for v in &mut q {
            *v /= nrm;
        }
        q
    }))
}

/// Build the geometry to restart a climb from after the search converged to a point
/// with the wrong number of negative modes. Given the known reaction-coordinate
/// `seed` (mass-weighted, normalized), descend every *spurious* negative mode at
/// `x_wrong` — every negative mode except the one most aligned with the seed — and
/// add a component along the seed itself, so the re-climb pushes the extra imaginary
/// directions into minima while climbing the intended reaction coordinate. (For a
/// minimum, with no negative modes, the seed alone is the whole displacement.) The
/// combined direction is scaled to the trust radius in the mass-weighted frame, then
/// un-mass-weighted into the Cartesian displacement applied to `x_wrong`.
fn recovery_perturbation<S: Surface>(
    surface: &mut S,
    x_wrong: &[[f64; 3]],
    masses: &[f64],
    options: &TsOptions,
    seed_mw: &[f64],
) -> Result<Vec<[f64; 3]>, TsError> {
    let ndof = 3 * x_wrong.len();
    let hess = fd_hessian(surface, x_wrong, options.fd_step)?;
    let spec = mw_projected_hessian(x_wrong, masses, &hess).map_err(TsError::Numerical)?;

    let negatives: Vec<usize> = (0..ndof)
        .filter(|&k| spec.eigenvalues[k] < -options.negative_mode_tol)
        .collect();
    // The reaction-coordinate negative mode is the one most aligned with the seed;
    // every other negative mode is spurious and gets descended.
    let reaction = negatives.iter().copied().max_by(|&a, &b| {
        overlap(&spec, ndof, a, seed_mw)
            .partial_cmp(&overlap(&spec, ndof, b, seed_mw))
            .unwrap()
    });

    let mut dir = vec![0.0f64; ndof];
    for &k in &negatives {
        if Some(k) != reaction {
            // Either sign descends a maximum; take the eigensolver's sign.
            let col = column(&spec.eigenvectors, ndof, k);
            for (d, c) in dir.iter_mut().zip(&col) {
                *d += c;
            }
        }
    }
    for (d, s) in dir.iter_mut().zip(seed_mw) {
        *d += s;
    }

    let nrm = norm(&dir);
    // The seed is a unit vector, so `dir` is degenerate only if the spurious-mode
    // contributions exactly cancel it; fall back to the seed alone.
    let basis = if nrm > 1e-12 { &dir } else { seed_mw };
    let scale = options.trust_radius / norm(basis);
    let scaled: Vec<f64> = basis.iter().map(|v| v * scale).collect();
    Ok(add_step(x_wrong, &unmass_weight_step(&scaled, masses)))
}
