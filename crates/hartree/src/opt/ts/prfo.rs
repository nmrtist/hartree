//! Partitioned rational-function optimization (P-RFO / eigenvector following)
//! and the lightweight IRC endpoint confirmation.
//!
//! The search runs in mass-weighted Cartesian coordinates with translations and
//! rotations projected out (the [`crate::props::frequencies`] frame). Each step
//! diagonalizes the projected Hessian, follows one mode uphill while minimizing
//! the rest (the partitioned RFO step), and maintains the Hessian by a Bofill
//! update. Convergence is tested on the trans/rot-projected Cartesian force and
//! the step; after it, the shared [`verify_saddle`](super::verify_saddle) check
//! confirms one negative mode.
//!
//! The per-step math (the RFO step, mode selection, the Bofill update, the trust
//! adaption, and the acceptance test) lives in [`super::step`]; this file is the
//! driver loop, the Hessian maintenance, and the step backtracking around them.

use super::numerics::{
    add_step, column, disp_norms, fd_hessian, flatten, force_norms, gradient, gram_schmidt,
    mass_weight_grad, masses_of, mw_projected_hessian, non_null_modes, norm, overlap, positions_of,
    predicted_change_cart, projected_force_norms, trans_rot_vectors, unmass_weight_step,
};
use super::step::{bofill_update, is_pathological, prfo_step, select_followed, update_trust_ts};
use super::{Flow, IrcEndpoints, Progress, TsError, TsOptions, TsResult, TsStatus, verify_saddle};
use crate::core::Molecule;
use crate::opt::{OptError, OptStep, Surface};

/// Overlap with the previously followed eigenvector below which mode tracking is
/// taken to have failed (and the Hessian is recomputed).
const TRACK_TOL: f64 = 0.5;

/// One accepted P-RFO step, threaded out of the backtracking loop in one piece:
/// the data the trust adaption, Hessian update, and commit all consume. The
/// gradient at `x_new` is *not* carried — it is evaluated once after acceptance, so
/// a backtracking retry (which only needs the energy) does not pay for a discarded
/// finite-difference gradient.
struct AcceptedStep {
    /// Mass-weighted step (flat), for the trust adaption's `step_norm`.
    dxi: Vec<f64>,
    /// Cartesian step, for the Bofill `s` vector.
    dx: Vec<[f64; 3]>,
    /// Model-predicted energy change, for the trust adaption and Hessian guard.
    predicted: f64,
    /// Trial geometry reached.
    x_new: Vec<[f64; 3]>,
    /// Energy at `x_new`.
    energy_new: f64,
    /// Actual energy change `energy_new - energy`, for the trust adaption.
    actual: f64,
}

pub(super) fn run_prfo<S: Surface>(
    molecule: &Molecule,
    surface: &mut S,
    options: &TsOptions,
    progress: Option<&dyn Progress>,
) -> Result<TsResult, TsError> {
    let masses = masses_of(molecule);
    let natom = molecule.len();
    let ndof = 3 * natom;

    let mut x = positions_of(molecule);

    let n_tr = gram_schmidt(&trans_rot_vectors(&x, &masses)).len();
    if ndof < n_tr + 1 {
        return Err(TsError::BadInitialGuess(format!(
            "{natom} atom(s) leave no internal coordinate to follow ({ndof} \
             Cartesian DOF, {n_tr} translation/rotation modes)"
        )));
    }

    let mut energy = surface.energy(&x)?;
    let mut g = gradient(surface, &x, options.fd_step)?;
    // A non-finite energy/gradient at an accepted point yields no usable step (and
    // a non-finite gradient would otherwise reach the panicking inner eigensolver
    // in `prfo_step`); surface it as the documented soft fault.
    if !energy.is_finite() || !finite_grad(&g) {
        return Err(TsError::Numerical(
            "surface returned a non-finite energy or gradient at the initial geometry".to_string(),
        ));
    }
    let mut hess = fd_hessian(surface, &x, options.fd_step)?;

    let mut trust = options
        .trust_radius
        .min(options.max_trust)
        .max(options.min_trust);
    let mut history: Vec<OptStep> = Vec::new();
    let mut x_prev: Option<Vec<[f64; 3]>> = None;
    let mut followed_vec: Option<Vec<f64>> = None;
    let mut steps_since_hess = 0usize;

    let mut best_x = x.clone();
    let mut best_energy = energy;
    let mut best_force = f64::INFINITY;

    let mut iterations = 0usize;
    let mut converged_geom = false;
    let mut stopped_early = false;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let (max_force, rms_force) = force_norms(&g);
        // Convergence (and best-so-far) are judged on the trans/rot-projected force,
        // the frame the step lives in; the history record keeps the raw force so the
        // reported trace is unchanged.
        let (conv_force, conv_rms) = projected_force_norms(&g, &masses, &x);
        let (max_disp, rms_disp) = match &x_prev {
            Some(xp) => disp_norms(&x, xp),
            None => (0.0, 0.0),
        };
        if conv_force < best_force {
            best_force = conv_force;
            best_x = x.clone();
            best_energy = energy;
        }

        let record = OptStep {
            iteration: iter,
            energy,
            max_force,
            rms_force,
            max_disp,
            rms_disp,
        };
        history.push(record);
        if let Some(observer) = progress {
            if observer.step(&record) == Flow::Stop {
                stopped_early = true;
                break;
            }
        }

        let force_ok = conv_force < options.max_force && conv_rms < options.rms_force;
        let disp_ok =
            x_prev.is_none() || (max_disp < options.max_disp && rms_disp < options.rms_disp);
        if force_ok && disp_ok {
            converged_geom = true;
            break;
        }
        if iter == options.max_iter {
            break;
        }

        // Diagonalize the maintained Hessian's projected spectrum, self-healing a
        // numerical failure once: a stale quasi-Newton (Bofill) Hessian can drift to
        // a non-finite or ill-conditioned state the checked eigensolver rejects, so
        // rebuild it from finite differences and retry before giving up.
        let mut spec = match mw_projected_hessian(&x, &masses, &hess) {
            Ok(s) => s,
            Err(_) => {
                hess = fd_hessian(surface, &x, options.fd_step)?;
                steps_since_hess = 0;
                mw_projected_hessian(&x, &masses, &hess).map_err(TsError::Numerical)?
            }
        };
        let mut non_null = non_null_modes(&spec);
        if non_null.is_empty() {
            break;
        }
        let mut followed = select_followed(&spec, &non_null, options.follow_mode, &followed_vec);

        // Mode tracking lost the climbed mode (usually a stale quasi-Newton
        // Hessian): recompute once from finite differences and re-pick.
        if let Some(reference) = &followed_vec {
            if overlap(&spec, ndof, followed, reference) < TRACK_TOL && steps_since_hess > 0 {
                hess = fd_hessian(surface, &x, options.fd_step)?;
                steps_since_hess = 0;
                spec = mw_projected_hessian(&x, &masses, &hess).map_err(TsError::Numerical)?;
                non_null = non_null_modes(&spec);
                if non_null.is_empty() {
                    break;
                }
                followed = select_followed(&spec, &non_null, options.follow_mode, &followed_vec);
            }
        }
        followed_vec = Some(column(&spec.eigenvectors, ndof, followed));

        let g_mw = mass_weight_grad(&g, &masses);

        // Step with backtracking: shrink the trust radius and retry from the same
        // point when a trial step's SCF fails to converge, returns a non-finite
        // energy, or grossly overshoots the quadratic model (see `is_pathological`).
        // Only the energy is evaluated here — the decision needs nothing else — so a
        // rejected retry does not pay for a (finite-difference) gradient; the
        // gradient is taken once, below, after a step is accepted. A persistent SCF
        // failure is a soft stop (best-so-far, NotConverged) rather than an abort.
        let mut attempt_trust = trust;
        let mut retries = 0usize;
        let accepted = loop {
            let mut dxi = prfo_step(&spec, &g_mw, &non_null, followed, attempt_trust);
            if norm(&dxi) < 1e-10 {
                // RFO produced no step (e.g. a symmetric guess with no gradient along
                // the climbed mode): take a trust-sized step along it to break the stall.
                for (i, slot) in dxi.iter_mut().enumerate() {
                    *slot = attempt_trust * spec.eigenvectors[i * ndof + followed];
                }
            }
            let dx = unmass_weight_step(&dxi, &masses);
            let predicted = predicted_change_cart(&g, &hess, &dx);
            let x_new = add_step(&x, &dx);

            let shrink_allowed =
                retries < options.max_step_retries && attempt_trust > options.min_trust * 1.0001;
            match surface.energy(&x_new) {
                Ok(energy_new) => {
                    if !energy_new.is_finite() {
                        // A non-finite trial energy: retry smaller if we still can,
                        // else it is a genuine numerical fault.
                        if shrink_allowed {
                            retries += 1;
                            attempt_trust = (0.25 * attempt_trust).max(options.min_trust);
                            continue;
                        }
                        return Err(TsError::Numerical(
                            "surface returned a non-finite energy".to_string(),
                        ));
                    }
                    let actual = energy_new - energy;
                    if is_pathological(actual, predicted) && shrink_allowed {
                        retries += 1;
                        attempt_trust = (0.25 * attempt_trust).max(options.min_trust);
                        continue;
                    }
                    break Some(AcceptedStep {
                        dxi,
                        dx,
                        predicted,
                        x_new,
                        energy_new,
                        actual,
                    });
                }
                Err(OptError::ScfNotConverged { .. }) if shrink_allowed => {
                    retries += 1;
                    attempt_trust = (0.25 * attempt_trust).max(options.min_trust);
                }
                // SCF failed and no retry budget remains: stop softly with best-so-far.
                Err(OptError::ScfNotConverged { .. }) => break None,
                Err(e) => return Err(TsError::SurfaceEvaluation(e)),
            }
        };

        let Some(step) = accepted else {
            // The SCF would not converge from this point even after backtracking;
            // stop with the best geometry so far rather than aborting the search.
            break;
        };

        // The step is accepted: now (and only now) evaluate the gradient at it.
        let g_new = gradient(surface, &step.x_new, options.fd_step)?;
        if !finite_grad(&g_new) {
            return Err(TsError::Numerical(
                "surface returned a non-finite gradient at an accepted step".to_string(),
            ));
        }

        // Judge the trust step in the frame it was capped in (‖dxi‖, mass-weighted):
        // the Cartesian ‖dx‖ shrinks as dxi/√m and would stop the radius growing.
        trust = update_trust_ts(
            attempt_trust,
            step.actual,
            step.predicted,
            norm(&step.dxi),
            options,
        );

        steps_since_hess += 1;
        // A step that was force-accepted despite overshooting the model (retries or
        // the trust floor exhausted) carries a curvature sample the model could not
        // describe; rebuild a fresh Hessian instead of feeding it to Bofill.
        let forced_overshoot = is_pathological(step.actual, step.predicted);
        let recalc_due = options.recalc_hessian != 0 && steps_since_hess >= options.recalc_hessian;
        if forced_overshoot || recalc_due {
            hess = fd_hessian(surface, &step.x_new, options.fd_step)?;
            steps_since_hess = 0;
        } else {
            let s = flatten(&step.dx);
            let gf_new = flatten(&g_new);
            let gf_old = flatten(&g);
            let y: Vec<f64> = gf_new.iter().zip(&gf_old).map(|(a, b)| a - b).collect();
            bofill_update(&mut hess, &s, &y, ndof);
        }

        x_prev = Some(x.clone());
        x = step.x_new;
        energy = step.energy_new;
        g = g_new;
    }

    if stopped_early {
        return Ok(TsResult {
            positions: best_x,
            energy: best_energy,
            status: TsStatus::StoppedEarly,
            iterations,
            history,
            verification: None,
            irc: None,
        });
    }
    if !converged_geom {
        return Ok(TsResult {
            positions: best_x,
            energy: best_energy,
            status: TsStatus::NotConverged,
            iterations,
            history,
            verification: None,
            irc: None,
        });
    }

    // Verify with a fresh finite-difference Hessian, not the maintained one.
    let verification = verify_saddle(molecule, surface, &x, options)?;
    let status = if verification.is_first_order_saddle() {
        TsStatus::Converged
    } else {
        TsStatus::WrongImaginaryModeCount
    };

    let irc = if status == TsStatus::Converged && options.confirm_irc {
        match &verification.reaction_mode {
            Some(mode) => Some(irc_endpoints(surface, &x, mode, options)?),
            None => None,
        }
    } else {
        None
    };

    // Leave the surface cache at the returned saddle, not the last verification /
    // IRC displacement, so a caller's `last_scf()` is the saddle wavefunction.
    let _ = surface.energy(&x)?;

    Ok(TsResult {
        positions: x,
        energy,
        status,
        iterations,
        history,
        verification: Some(verification),
        irc,
    })
}

/// Whether every component of a Cartesian gradient is finite. A non-finite
/// gradient would otherwise flow into the (panicking) inner eigensolver of
/// [`prfo_step`]; the driver maps it to [`TsError::Numerical`] instead.
fn finite_grad(g: &[[f64; 3]]) -> bool {
    g.iter().all(|v| v.iter().all(|c| c.is_finite()))
}

const IRC_DISPLACE: f64 = 0.2;
const IRC_MAX_STEPS: usize = 20;
const IRC_MAX_STEP: f64 = 0.2;
const IRC_GTOL: f64 = 1e-3;

/// Confirm the saddle joins two distinct basins by relaxing a short way downhill
/// from `saddle ± reaction mode`. A cheap damped-descent endpoint check, not a
/// mass-weighted IRC integrator.
pub(super) fn irc_endpoints<S: Surface>(
    surface: &mut S,
    saddle: &[[f64; 3]],
    reaction_mode: &[[f64; 3]],
    options: &TsOptions,
) -> Result<IrcEndpoints, TsError> {
    let mut dir = reaction_mode.to_vec();
    let dnorm = norm(&flatten(&dir));
    if dnorm > 0.0 {
        for d in &mut dir {
            for c in d.iter_mut() {
                *c /= dnorm;
            }
        }
    }

    let (forward, forward_energy) =
        relax_downhill(surface, &add_scaled(saddle, &dir, IRC_DISPLACE), options)?;
    let (reverse, reverse_energy) =
        relax_downhill(surface, &add_scaled(saddle, &dir, -IRC_DISPLACE), options)?;
    Ok(IrcEndpoints {
        forward,
        forward_energy,
        reverse,
        reverse_energy,
    })
}

fn add_scaled(x: &[[f64; 3]], dir: &[[f64; 3]], scale: f64) -> Vec<[f64; 3]> {
    x.iter()
        .zip(dir)
        .map(|(a, d)| {
            [
                a[0] + scale * d[0],
                a[1] + scale * d[1],
                a[2] + scale * d[2],
            ]
        })
        .collect()
}

fn relax_downhill<S: Surface>(
    surface: &mut S,
    start: &[[f64; 3]],
    options: &TsOptions,
) -> Result<(Vec<[f64; 3]>, f64), TsError> {
    let mut x = start.to_vec();
    let mut energy = surface.energy(&x)?;
    for _ in 0..IRC_MAX_STEPS {
        let g = gradient(surface, &x, options.fd_step)?;
        let gnorm = norm(&flatten(&g));
        if gnorm < IRC_GTOL {
            break;
        }
        let mut trial = (IRC_MAX_STEP / gnorm).min(1.0);
        let mut accepted = false;
        for _ in 0..5 {
            let x_new: Vec<[f64; 3]> = x
                .iter()
                .zip(&g)
                .map(|(xi, gi)| {
                    [
                        xi[0] - trial * gi[0],
                        xi[1] - trial * gi[1],
                        xi[2] - trial * gi[2],
                    ]
                })
                .collect();
            let e_new = surface.energy(&x_new)?;
            if e_new <= energy {
                x = x_new;
                energy = e_new;
                accepted = true;
                break;
            }
            trial *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    Ok((x, energy))
}
