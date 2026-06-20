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

use std::f64::consts::PI;

use super::numerics::{
    add_step, disp_norms, dot, force_norms, gradient, gram_schmidt, mass_weight_grad, masses_of,
    norm, positions_of, projected_force_norms, trans_rot_vectors, unmass_weight_step,
};
use super::{Flow, Progress, TsError, TsOptions, TsResult, TsStatus, verify_with_hessian};
use crate::core::Molecule;
use crate::opt::{OptError, OptStep, Surface};

/// Maximum dimer rotations per outer (translation) step.
const MAX_ROT: usize = 4;
/// Floor on |curvature| in the parallel-step denominator, so a near-flat mode
/// does not blow the step up.
const C_MIN: f64 = 1e-3;
/// Weight of the perpendicular (steepest-descent) component once an unstable
/// mode is aligned; for `C >= 0` the step moves along the axis only. Not
/// curvature-scaled, so stiff transverse modes rely on the trust radius to bound
/// the step.
const ALPHA: f64 = 1.0;

/// Below this projected-axis norm the carried dimer axis is treated as degenerate
/// and reseeded.
const AXIS_DEGEN_EPS: f64 = 1e-8;
/// Perpendicular gradient-difference norm below which the axis is taken to be
/// aligned with the lowest-curvature mode (rotation converged).
const GPERP_TOL: f64 = 1e-6;
/// Rotational-force magnitude below which a further rotation is not worthwhile.
const FROT_TOL: f64 = 1e-3;
/// Rotation angle (radians) below which the trial rotation is treated as a no-op.
const ROT_ANGLE_TOL: f64 = 1e-3;

/// Project a flat mass-weighted vector onto the internal subspace: subtract its
/// components along each (orthonormal) translation/rotation basis vector.
fn project_internal(v: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
    let mut out = v.to_vec();
    for b in basis {
        let c = dot(v, b);
        for (o, &bi) in out.iter_mut().zip(b) {
            *o -= c * bi;
        }
    }
    out
}

/// Normalize in place; returns the original norm.
fn normalize(v: &mut [f64]) -> f64 {
    let n = norm(v);
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    n
}

/// The mass-weighted, internal-subspace-projected gradient at the dimer endpoint
/// `x + Δ·N` (in mass-weighted space): displace the Cartesian midpoint by
/// un-mass-weighting `Δ N`, evaluate the gradient, mass-weight and project it.
/// The projection uses the trans/rot `basis` built at the midpoint `x` (the
/// endpoint frame differs by O(Δ)).
fn endpoint_grad<S: Surface>(
    surface: &mut S,
    x: &[[f64; 3]],
    axis: &[f64],
    delta: f64,
    masses: &[f64],
    basis: &[Vec<f64>],
    fd_step: f64,
) -> Result<Vec<f64>, TsError> {
    let scaled: Vec<f64> = axis.iter().map(|a| delta * a).collect();
    let endpoint = add_step(x, &unmass_weight_step(&scaled, masses));
    let g_cart = gradient(surface, &endpoint, fd_step)?;
    Ok(project_internal(&mass_weight_grad(&g_cart, masses), basis))
}

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

    for iter in 1..=options.max_iter {
        iterations = iter;
        let basis = gram_schmidt(&trans_rot_vectors(&x, &masses));

        let g0_cart = gradient(surface, &x, options.fd_step)?;
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

        // Axis: initialize on the first iteration (along the projected gradient,
        // falling back to a canonical internal direction); reuse + reproject later.
        let mut n_axis = match &axis {
            None => initial_axis(&g0, &basis, ndof),
            Some(prev) => {
                let mut a = project_internal(prev, &basis);
                if normalize(&mut a) < AXIS_DEGEN_EPS {
                    initial_axis(&g0, &basis, ndof)
                } else {
                    a
                }
            }
        };

        // ROTATION inner loop: align `n_axis` with the lowest-curvature mode.
        let mut aligned_curvature: Option<f64> = None;
        for _ in 0..MAX_ROT {
            let g1 = endpoint_grad(
                surface,
                &x,
                &n_axis,
                delta,
                &masses,
                &basis,
                options.fd_step,
            )?;
            let d: Vec<f64> = g1.iter().zip(&g0).map(|(a, b)| a - b).collect();
            let gpar = dot(&d, &n_axis);
            let curvature = gpar / delta;
            let gperp: Vec<f64> = d
                .iter()
                .zip(&n_axis)
                .map(|(di, ni)| di - gpar * ni)
                .collect();
            let gperp_norm = norm(&gperp);
            if gperp_norm < GPERP_TOL {
                aligned_curvature = Some(curvature);
                break;
            }
            let theta: Vec<f64> = gperp.iter().map(|g| g / gperp_norm).collect();

            let frot_norm = 2.0 * gperp_norm;
            if frot_norm < FROT_TOL {
                aligned_curvature = Some(curvature);
                break;
            }
            let b1 = dot(&d, &theta) / delta;

            // Trial rotation by π/4 to estimate the curvature's Fourier model.
            let phi1 = PI / 4.0;
            let mut nt = rotate(&n_axis, &theta, phi1.cos(), phi1.sin());
            nt = project_internal(&nt, &basis);
            normalize(&mut nt);
            let c1 = endpoint_curvature(
                surface,
                &x,
                &nt,
                &g0,
                delta,
                &masses,
                &basis,
                options.fd_step,
            )?;

            let a1 = (curvature - c1 + b1 * (2.0 * phi1).sin()) / (1.0 - (2.0 * phi1).cos());
            let phi_e = 0.5 * b1.atan2(a1);
            // Select the MINIMUM-curvature branch, then wrap into (-π/2, π/2].
            let mut phi_min = if a1 * (2.0 * phi_e).cos() + b1 * (2.0 * phi_e).sin() < 0.0 {
                phi_e
            } else {
                phi_e + PI / 2.0
            };
            if phi_min > PI / 2.0 {
                phi_min -= PI;
            } else if phi_min <= -PI / 2.0 {
                phi_min += PI;
            }

            n_axis = rotate(&n_axis, &theta, phi_min.cos(), phi_min.sin());
            n_axis = project_internal(&n_axis, &basis);
            normalize(&mut n_axis);

            // Early break if the rotation barely moves the axis.
            if phi_min.abs() < ROT_ANGLE_TOL {
                break;
            }
        }
        // Curvature at the converged axis (used by the translation step below).
        // Reuse the value from the rotation loop when the axis did not move on its
        // final pass; otherwise finite-difference it afresh.
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

/// Curvature `C ≈ NᵀHN` from one endpoint gradient: `(g1 - g0)·N / Δ`.
#[allow(clippy::too_many_arguments)]
fn endpoint_curvature<S: Surface>(
    surface: &mut S,
    x: &[[f64; 3]],
    axis: &[f64],
    g0: &[f64],
    delta: f64,
    masses: &[f64],
    basis: &[Vec<f64>],
    fd_step: f64,
) -> Result<f64, TsError> {
    let g1 = endpoint_grad(surface, x, axis, delta, masses, basis, fd_step)?;
    let d: Vec<f64> = g1.iter().zip(g0).map(|(a, b)| a - b).collect();
    Ok(dot(&d, axis) / delta)
}

/// `cos·N + sin·Θ` (unnormalized rotation of `N` toward `Θ`).
fn rotate(n: &[f64], theta: &[f64], cos: f64, sin: f64) -> Vec<f64> {
    n.iter()
        .zip(theta)
        .map(|(ni, ti)| cos * ni + sin * ti)
        .collect()
}

/// First-iteration dimer axis: the projected gradient direction; if it is
/// (near-)zero, the first canonical internal-subspace unit vector.
fn initial_axis(g0: &[f64], basis: &[Vec<f64>], ndof: usize) -> Vec<f64> {
    let mut n = project_internal(g0, basis);
    if normalize(&mut n) >= AXIS_DEGEN_EPS {
        return n;
    }
    for i in 0..ndof {
        let mut e = vec![0.0f64; ndof];
        e[i] = 1.0;
        let mut p = project_internal(&e, basis);
        if normalize(&mut p) > 1e-6 {
            return p;
        }
    }
    // Unreachable for a molecule with an internal DOF (guarded at setup).
    let mut e = vec![0.0f64; ndof];
    e[0] = 1.0;
    e
}
