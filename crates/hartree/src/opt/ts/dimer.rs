//! Dimer method for transition-state search — a Hessian-free saddle-point
//! optimizer (Henkelman-Jónsson 1999; Heyden 2005; Kästner-Sherwood 2008).
//!
//! Like [`super::prfo`] the search runs in mass-weighted Cartesian coordinates
//! with translations and rotations projected out (the
//! [`crate::props::frequencies`] frame). Where P-RFO forms and maintains a full
//! Hessian, the dimer estimates only the lowest-curvature mode from a pair of
//! nearby gradient evaluations: each outer iteration *rotates* a unit axis `N` to
//! align it with the softest mode (finite-differencing the curvature along `N`),
//! then *translates* the midpoint by inverting the force component along that
//! mode — climbing uphill toward the saddle along `N` while descending the rest.
//! Convergence is tested on the trans/rot-projected Cartesian force/step exactly
//! as in P-RFO; after it, the shared [`verify_saddle`](super::verify_saddle)
//! check confirms one negative mode and the [`super::irc`] tracer (reused, not
//! duplicated) optionally confirms the basins the saddle joins.

use super::dimer_rotate::{
    AXIS_DEGEN_EPS, endpoint_curvature, initial_axis, normalize, project_internal, require_finite,
    rotate_to_min_mode, seed_axis,
};
use super::numerics::{
    add_step, disp_norms, dot, force_norms, gradient, gram_schmidt, mass_weight_grad, masses_of,
    norm, positions_of, projected_force_norms, trans_rot_vectors, unmass_weight_step,
};
use super::{Flow, Progress, TsError, TsOptions, TsResult, TsStatus, verify_with_hessian};
use crate::core::Molecule;
use crate::opt::{OptError, OptStep, Surface};

/// Floor on |curvature| in the parallel-step denominator, so a near-flat mode
/// does not blow the step up.
const C_MIN: f64 = 1e-3;
/// Weight of the perpendicular (steepest-descent) component once an unstable
/// mode is aligned; for `C >= 0` the step moves along the axis only. Not
/// curvature-scaled, so stiff transverse modes rely on the trust radius to bound
/// the step.
const ALPHA: f64 = 1.0;

pub(super) fn run_dimer<S: Surface>(
    molecule: &Molecule,
    surface: &mut S,
    options: &TsOptions,
    progress: Option<&dyn Progress>,
) -> Result<TsResult, TsError> {
    let masses = masses_of(molecule);
    let natom = molecule.len();
    let ndof = 3 * natom;
    let delta = options.dimer_delta;

    let mut x = positions_of(molecule);

    let n_tr = gram_schmidt(&trans_rot_vectors(&x, &masses)).len();
    if ndof < n_tr + 1 {
        return Err(TsError::BadInitialGuess(format!(
            "{natom} atom(s) leave no internal coordinate to follow ({ndof} \
             Cartesian DOF, {n_tr} translation/rotation modes)"
        )));
    }

    // The reaction-coordinate seed (if any) is validated up front — a length
    // mismatch is a bad guess before any surface evaluation — then used to
    // initialize the dimer axis (see the axis selection below). Mass-weighted the
    // same way P-RFO seeds its climb.
    let seed_mw = super::prfo::mass_weighted_seed(options, &masses, natom)?;

    let mut energy = surface.energy(&x)?;
    let mut trust = options
        .trust_radius
        .min(options.max_trust)
        .max(options.min_trust);
    let mut history: Vec<OptStep> = Vec::new();
    let mut x_prev: Option<Vec<[f64; 3]>> = None;

    // Carried across iterations: the dimer axis, the previous projected gradient
    // and step norm for the force-based trust update.
    let mut axis: Option<Vec<f64>> = None;
    let mut g_prev: Option<Vec<f64>> = None;
    let mut last_step_norm = 0.0;
    let mut trust_prev = trust;

    let mut best_x = x.clone();
    let mut best_energy = energy;
    let mut best_force = f64::INFINITY;

    let mut iterations = 0usize;
    let mut converged_geom = false;
    let mut stopped_early = false;
    let mut not_converged_reason: Option<String> = None;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let basis = gram_schmidt(&trans_rot_vectors(&x, &masses));

        let g0_cart = gradient(surface, &x, options.fd_step)?;
        require_finite(&g0_cart, "dimer midpoint gradient")?;
        let (max_force, rms_force) = force_norms(&g0_cart);
        // Judge convergence (and best-so-far) on the trans/rot-projected force, the
        // frame the rotation/translation runs in; keep the raw force in the record.
        let (conv_force, conv_rms) = projected_force_norms(&g0_cart, &masses, &x);
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
            not_converged_reason = Some(format!(
                "reached max_iter ({}) with max projected force {:.2e} a.u.",
                options.max_iter, conv_force
            ));
            break;
        }

        let g0 = project_internal(&mass_weight_grad(&g0_cart, &masses), &basis);

        // Force-based trust update (energy is non-monotonic in a saddle search):
        // compare this midpoint's projected gradient with the previous one.
        if let Some(prev) = &g_prev {
            let np = norm(prev);
            if np > 0.0 {
                let r_mag = norm(&g0) / np;
                let rho = dot(&g0, prev) / (norm(&g0) * np);
                if r_mag < 1.0 && rho > 0.0 && last_step_norm >= 0.8 * trust_prev {
                    trust = (2.0 * trust).min(options.max_trust);
                } else if r_mag > 1.0 || rho < -0.2 {
                    trust = (0.5 * trust).max(options.min_trust);
                }
            }
        }

        // Axis: initialize on the first iteration. When a reaction-coordinate seed
        // is supplied (e.g. the tangent carried over from a two-endpoint/NEB
        // handoff), build the first axis from it — projected into the internal
        // subspace and normalized — so the dimer climbs the reaction coordinate
        // from the outset; otherwise follow the projected gradient, falling back to
        // a canonical internal direction. Reuse + reproject the axis later.
        let mut n_axis = match &axis {
            None => seed_axis(seed_mw.as_deref(), &basis)
                .unwrap_or_else(|| initial_axis(&g0, &basis, ndof)),
            Some(prev) => {
                let mut a = project_internal(prev, &basis);
                if normalize(&mut a) < AXIS_DEGEN_EPS {
                    initial_axis(&g0, &basis, ndof)
                } else {
                    a
                }
            }
        };

        // ROTATION: align `n_axis` with the lowest-curvature mode, then take the
        // curvature there (reusing the rotation loop's value when it converged
        // early, else finite-differencing it afresh) for the translation below.
        let aligned_curvature = rotate_to_min_mode(
            surface,
            &x,
            &mut n_axis,
            &g0,
            delta,
            &masses,
            &basis,
            options.fd_step,
        )?;
        let curvature = match aligned_curvature {
            Some(c) => c,
            None => endpoint_curvature(
                surface,
                &x,
                &n_axis,
                &g0,
                delta,
                &masses,
                &basis,
                options.fd_step,
            )?,
        };

        // TRANSLATION (note the PLUS sign on step_par — it climbs uphill along N),
        // with SCF backtracking: a climbing step can overshoot into a region where
        // the SCF will not converge, so on `ScfNotConverged` shrink the trust radius
        // and recompute the step from the same midpoint instead of propagating the
        // failure. The dimer keeps no quadratic energy model, so unlike P-RFO it has
        // no actual/predicted ratio to reject on — SCF recovery is the win here. A
        // failure that persists through every retry is a soft stop, not an abort.
        let gpar = dot(&g0, &n_axis);
        let gperp_vec: Vec<f64> = g0
            .iter()
            .zip(&n_axis)
            .map(|(g, ni)| g - gpar * ni)
            .collect();
        let step_par = gpar / curvature.abs().max(C_MIN);

        let mut attempt_trust = trust;
        let mut retries = 0usize;
        let committed = loop {
            let mut dq: Vec<f64> = n_axis.iter().map(|ni| step_par * ni).collect();
            if curvature < 0.0 {
                for (dqi, &gp) in dq.iter_mut().zip(&gperp_vec) {
                    *dqi -= ALPHA * gp;
                }
            }
            let mut dq_norm = norm(&dq);
            if dq_norm < 1e-10 {
                // The gradient is (near-)orthogonal to the soft mode, so the Newton
                // step vanishes; take a trust-sized step along the axis to keep
                // climbing rather than stall in place.
                for (v, &ni) in dq.iter_mut().zip(&n_axis) {
                    *v = attempt_trust * ni;
                }
                dq_norm = norm(&dq);
            }
            if dq_norm > attempt_trust && dq_norm > 0.0 {
                let scale = attempt_trust / dq_norm;
                for v in &mut dq {
                    *v *= scale;
                }
            }
            let step_norm = norm(&dq);
            let x_new = add_step(&x, &unmass_weight_step(&dq, &masses));

            let shrink_allowed =
                retries < options.max_step_retries && attempt_trust > options.min_trust * 1.0001;
            match surface.energy(&x_new) {
                Ok(e) if e.is_finite() => break Some((x_new, e, step_norm)),
                // A non-finite trial energy: retry smaller if we still can, else it
                // is a genuine numerical fault.
                Ok(_) if shrink_allowed => {
                    retries += 1;
                    attempt_trust = (0.25 * attempt_trust).max(options.min_trust);
                }
                Ok(_) => {
                    return Err(TsError::Numerical(
                        "surface returned a non-finite energy".to_string(),
                    ));
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

        let Some((x_new, energy_new, step_norm)) = committed else {
            // SCF would not converge from this midpoint even after backtracking;
            // stop with the best geometry so far rather than aborting the search.
            not_converged_reason = Some(
                "step reduced to the trust floor without a usable SCF (backtracking exhausted)"
                    .to_string(),
            );
            break;
        };

        x_prev = Some(x.clone());
        x = x_new;
        energy = energy_new;
        axis = Some(n_axis);
        g_prev = Some(g0);
        last_step_norm = step_norm;
        trust = attempt_trust;
        trust_prev = trust;
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
            diagnostic: None,
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
            diagnostic: not_converged_reason,
        });
    }

    // Keep the verification Hessian so a Hessian-corrector IRC reuses it (the dimer
    // is Hessian-free, so this verify is its only saddle Hessian).
    let (verification, hessian) = verify_with_hessian(molecule, surface, &x, options)?;
    let status = if verification.is_first_order_saddle() {
        TsStatus::Converged
    } else {
        TsStatus::WrongImaginaryModeCount
    };

    let irc = if status == TsStatus::Converged {
        super::irc::confirm_irc_endpoints(
            surface,
            &x,
            &verification,
            &masses,
            energy,
            options,
            &hessian,
        )?
    } else {
        None
    };

    // Leave the surface cache at the returned saddle (as P-RFO does).
    let _ = surface.energy(&x)?;

    let diagnostic = if status == TsStatus::WrongImaginaryModeCount {
        Some(super::prfo::wrong_mode_reason(
            verification.negative_eigenvalues.len(),
        ))
    } else {
        None
    };
    Ok(TsResult {
        positions: x,
        energy,
        status,
        iterations,
        history,
        verification: Some(verification),
        irc,
        diagnostic,
    })
}
