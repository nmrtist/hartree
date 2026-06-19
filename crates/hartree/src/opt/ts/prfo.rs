//! Partitioned rational-function optimization (P-RFO / eigenvector following)
//! and the lightweight IRC endpoint confirmation.
//!
//! The search runs in mass-weighted Cartesian coordinates with translations and
//! rotations projected out (the [`crate::props::frequencies`] frame). Each step
//! diagonalizes the projected Hessian, follows one mode uphill while minimizing
//! the rest (the partitioned RFO step), and maintains the Hessian by a Bofill
//! update. Convergence is tested on the Cartesian force/step; after it, the
//! shared [`verify_saddle`](super::verify_saddle) check confirms one negative mode.

use super::numerics::{
    MwSpectrum, add_step, column, disp_norms, dot, fd_hessian, flatten, force_norms, gradient,
    gram_schmidt, mass_weight_grad, masses_of, matvec, mw_projected_hessian, non_null_modes, norm,
    overlap, positions_of, predicted_change_cart, trans_rot_vectors, unmass_weight_step,
};
use super::{Flow, IrcEndpoints, Progress, TsError, TsOptions, TsResult, TsStatus, verify_saddle};
use crate::core::Molecule;
use crate::linalg::{mat_from_row_major, symmetric_eigh};
use crate::opt::{OptStep, Surface};

/// Overlap with the previously followed eigenvector below which mode tracking is
/// taken to have failed (and the Hessian is recomputed).
const TRACK_TOL: f64 = 0.5;

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
        let (max_disp, rms_disp) = match &x_prev {
            Some(xp) => disp_norms(&x, xp),
            None => (0.0, 0.0),
        };
        if max_force < best_force {
            best_force = max_force;
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

        let force_ok = max_force < options.max_force && rms_force < options.rms_force;
        let disp_ok =
            x_prev.is_none() || (max_disp < options.max_disp && rms_disp < options.rms_disp);
        if force_ok && disp_ok {
            converged_geom = true;
            break;
        }
        if iter == options.max_iter {
            break;
        }

        let mut spec = mw_projected_hessian(&x, &masses, &hess);
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
                spec = mw_projected_hessian(&x, &masses, &hess);
                non_null = non_null_modes(&spec);
                if non_null.is_empty() {
                    break;
                }
                followed = select_followed(&spec, &non_null, options.follow_mode, &followed_vec);
            }
        }
        followed_vec = Some(column(&spec.eigenvectors, ndof, followed));

        let g_mw = mass_weight_grad(&g, &masses);
        let mut dxi = prfo_step(&spec, &g_mw, &non_null, followed, trust);
        if norm(&dxi) < 1e-10 {
            // RFO produced no step (e.g. a symmetric guess with no gradient along
            // the climbed mode): take a trust-sized step along it to break the stall.
            for (i, slot) in dxi.iter_mut().enumerate() {
                *slot = trust * spec.eigenvectors[i * ndof + followed];
            }
        }
        let dx = unmass_weight_step(&dxi, &masses);

        let predicted = predicted_change_cart(&g, &hess, &dx);
        let x_new = add_step(&x, &dx);
        let energy_new = surface.energy(&x_new)?;
        let g_new = gradient(surface, &x_new, options.fd_step)?;
        let actual = energy_new - energy;
        // Judge the trust step in the frame it was capped in (‖dxi‖, mass-weighted):
        // the Cartesian ‖dx‖ shrinks as dxi/√m and would stop the radius growing.
        trust = update_trust_ts(trust, actual, predicted, norm(&dxi), options);

        steps_since_hess += 1;
        if options.recalc_hessian != 0 && steps_since_hess >= options.recalc_hessian {
            hess = fd_hessian(surface, &x_new, options.fd_step)?;
            steps_since_hess = 0;
        } else {
            let s = flatten(&dx);
            let gf_new = flatten(&g_new);
            let gf_old = flatten(&g);
            let y: Vec<f64> = gf_new.iter().zip(&gf_old).map(|(a, b)| a - b).collect();
            bofill_update(&mut hess, &s, &y, ndof);
        }

        x_prev = Some(x.clone());
        x = x_new;
        energy = energy_new;
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

/// One partitioned RFO step in the mass-weighted eigenbasis: the `followed` mode
/// is driven to a maximum, the rest to a minimum. Returns the trust-limited,
/// mass-weighted Cartesian step (flat, length `ndof`).
fn prfo_step(
    spec: &MwSpectrum,
    g_mw: &[f64],
    non_null: &[usize],
    followed: usize,
    trust: f64,
) -> Vec<f64> {
    let ndof = spec.eigenvalues.len();
    let fcomp = |k: usize| -> f64 {
        (0..ndof)
            .map(|i| spec.eigenvectors[i * ndof + k] * g_mw[i])
            .sum()
    };
    let b = |k: usize| spec.eigenvalues[k];

    // Followed mode: upper root of [[b_p, F_p], [F_p, 0]] gives an uphill step.
    let fp = fcomp(followed);
    let bp = b(followed);
    let lambda_p = 0.5 * (bp + (bp * bp + 4.0 * fp * fp).sqrt());
    let denom_p = bp - lambda_p;
    let step_p = if denom_p.abs() > 1e-12 {
        -fp / denom_p
    } else {
        0.0
    };

    // Remaining modes: lowest root of [[diag(b_N), F_N], [F_N^T, 0]] gives downhill steps.
    let minimized: Vec<usize> = non_null
        .iter()
        .copied()
        .filter(|&k| k != followed)
        .collect();
    let m = minimized.len();
    let lambda_n = if m > 0 {
        let mut aug = vec![0.0f64; (m + 1) * (m + 1)];
        for (a, &k) in minimized.iter().enumerate() {
            aug[a * (m + 1) + a] = b(k);
            let fk = fcomp(k);
            aug[a * (m + 1) + m] = fk;
            aug[m * (m + 1) + a] = fk;
        }
        symmetric_eigh(&mat_from_row_major(m + 1, &aug)).values[0]
    } else {
        0.0
    };

    let mut dxi = vec![0.0f64; ndof];
    for (i, slot) in dxi.iter_mut().enumerate() {
        *slot += step_p * spec.eigenvectors[i * ndof + followed];
    }
    for &k in &minimized {
        let denom = b(k) - lambda_n;
        let step_k = if denom.abs() > 1e-12 {
            -fcomp(k) / denom
        } else {
            0.0
        };
        for (i, slot) in dxi.iter_mut().enumerate() {
            *slot += step_k * spec.eigenvectors[i * ndof + k];
        }
    }

    let n = norm(&dxi);
    if n > trust && n > 0.0 {
        let scale = trust / n;
        for v in &mut dxi {
            *v *= scale;
        }
    }
    dxi
}

/// First step: the `follow_mode`-th non-null mode by ascending eigenvalue.
/// Later steps: the non-null mode of maximum overlap with the previous one.
fn select_followed(
    spec: &MwSpectrum,
    non_null: &[usize],
    follow_mode: usize,
    previous: &Option<Vec<f64>>,
) -> usize {
    let ndof = spec.eigenvalues.len();
    match previous {
        None => non_null[follow_mode.min(non_null.len() - 1)],
        Some(reference) => non_null
            .iter()
            .copied()
            .max_by(|&a, &b| {
                overlap(spec, ndof, a, reference)
                    .partial_cmp(&overlap(spec, ndof, b, reference))
                    .unwrap()
            })
            .unwrap(),
    }
}

/// Bofill (1994) Hessian update — an SR1/PSB blend that, unlike BFGS, preserves
/// the indefiniteness the reaction mode needs. `s` is the Cartesian step, `y` the
/// gradient change.
fn bofill_update(hess: &mut [f64], s: &[f64], y: &[f64], n: usize) {
    let ss = dot(s, s);
    if ss < 1e-14 {
        return;
    }
    let hs = matvec(hess, s, n);
    let delta: Vec<f64> = y.iter().zip(&hs).map(|(yi, hsi)| yi - hsi).collect();
    let sd = dot(s, &delta);
    let dd = dot(&delta, &delta);
    let phi = if dd > 1e-14 {
        (sd * sd) / (ss * dd)
    } else {
        0.0
    };
    let ms_ok = sd.abs() > 1e-12;
    for i in 0..n {
        for j in 0..n {
            let ms = if ms_ok { delta[i] * delta[j] / sd } else { 0.0 };
            let psb = (delta[i] * s[j] + s[i] * delta[j]) / ss - sd * s[i] * s[j] / (ss * ss);
            hess[i * n + j] += phi * ms + (1.0 - phi) * psb;
        }
    }
}

/// Adapt the trust radius from how well the quadratic model predicted the energy
/// change. `step_norm` is mass-weighted, matching `prfo_step`'s cap.
fn update_trust_ts(
    trust: f64,
    actual: f64,
    predicted: f64,
    step_norm: f64,
    opts: &TsOptions,
) -> f64 {
    if predicted.abs() < 1e-14 {
        return trust;
    }
    let ratio = actual / predicted;
    if (0.75..=1.25).contains(&ratio) && step_norm > 0.8 * trust {
        (2.0 * trust).min(opts.max_trust)
    } else if !(0.25..=1.75).contains(&ratio) {
        (0.5 * trust).max(opts.min_trust)
    } else {
        trust
    }
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
