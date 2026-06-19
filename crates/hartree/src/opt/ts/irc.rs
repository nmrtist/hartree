//! Intrinsic-reaction-coordinate (IRC) integration from a converged saddle.
//!
//! Where the saddle search climbs *to* the first-order saddle, the IRC traces the
//! minimum-energy path *away* from it — the steepest-descent path in **mass-weighted
//! Cartesian coordinates** (`q = √m · x`), the frame in which the reaction coordinate
//! is physically meaningful. Starting just off the saddle along the (mass-weighted)
//! transition vector, the path is integrated in both senses until each side relaxes
//! into a distinct basin; the two endpoints are the evidence that the saddle joins
//! two minima rather than sitting on a shoulder.
//!
//! Three integrators are offered through [`IrcMethod`], trading cost for fidelity:
//! [`Dvv`](IrcMethod::Dvv) (the default) is Hessian-free, integrating a velocity-
//! damped trajectory down the valley floor; [`EulerPc`](IrcMethod::EulerPc) is a
//! predictor-corrector that reuses one cached Hessian to take second-order-accurate
//! steps at one gradient per step; [`GonzalezSchlegel`](IrcMethod::GonzalezSchlegel)
//! constrains each step to a hypersphere and refines it against the true gradient,
//! the most accurate and most expensive. All three share the driver below, which
//! handles the initial displacement, both-senses integration, convergence on the
//! trans/rot-projected force, and per-endpoint reporting.

use serde::{Deserialize, Serialize};

use super::numerics::{
    add_step, dot, fd_hessian, flatten, gradient, gram_schmidt, mass_weight_grad, matvec, norm,
    projected_force_norms, trans_rot_vectors, unmass_weight_step,
};
use super::{SaddleVerification, TsError, TsOptions};
use crate::opt::{OptError, Surface};

/// Pseudo-time step of the [`Dvv`](IrcMethod::Dvv) integrator (reduced units; the
/// mass is unity in mass-weighted coordinates).
const IRC_DT: f64 = 1.0;
/// Per-step velocity damping of the [`Dvv`](IrcMethod::Dvv) integrator: < 1 so the
/// trajectory continuously bleeds the kinetic energy it gains descending, settling
/// onto the valley floor (the steepest-descent path) and into the basin minimum.
const IRC_DAMP: f64 = 0.85;
/// Maximum hypersphere micro-iterations per [`GonzalezSchlegel`](IrcMethod::GonzalezSchlegel) step.
const GS2_INNER: usize = 8;
/// Perpendicular-gradient norm below which a Gonzalez–Schlegel constrained step is
/// taken to lie on the reaction path (the micro-iteration has converged).
const GS2_TOL: f64 = 1e-4;
/// Steepest-descent rate (atomic units, length per force) of the settle step the
/// gradient-following integrators take once inside a basin: a step `−rate·g_mw`,
/// capped at the arc length. `1.0` is the natural a.u. (unit-Newton) scale — stable
/// for the soft mass-weighted curvatures reaction paths carry.
const IRC_SD_RATE: f64 = 1.0;
/// Energy (hartree) a trace must drop below the saddle before its endpoint counts as
/// having entered a basin — so a tiny residual force a short step off a flat/floppy
/// saddle ridge is not mistaken for a converged minimum.
const IRC_BASIN_EPS: f64 = 1e-4;

/// IRC integrator selected by [`TsOptions::irc_method`](super::TsOptions::irc_method).
///
/// The variants form a cost/accuracy ladder; see the module docs. `#[non_exhaustive]`
/// so further integrators can be added without a breaking change; the default is
/// [`Dvv`](IrcMethod::Dvv), the Hessian-free workhorse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IrcMethod {
    /// Damped-velocity-Verlet: integrate a velocity-damped classical trajectory down
    /// the mass-weighted gradient, restarting the velocity whenever a step would climb
    /// (the standard safeguard that keeps a stiff transverse mode from oscillating
    /// under the fixed pseudo-time step). Hessian-free, one gradient per step.
    #[default]
    Dvv,
    /// Gonzalez–Schlegel (second order): each step is constrained to a hypersphere
    /// about a pivot point and refined against the true gradient. Most accurate,
    /// most gradient evaluations per step.
    GonzalezSchlegel,
    /// Euler predictor / Hessian corrector: a steepest-descent predictor followed by
    /// a corrector that reuses one cached Hessian for second-order accuracy at one
    /// true gradient per step.
    EulerPc,
}

/// Endpoints found by tracing the intrinsic reaction coordinate (IRC) downhill from
/// the converged saddle, in the `+` and `-` senses of the reaction mode
/// ([`SaddleVerification::reaction_mode`](super::SaddleVerification::reaction_mode)).
/// Present in [`TsResult::irc`](super::TsResult::irc) only when
/// [`TsOptions::confirm_irc`](super::TsOptions::confirm_irc) was set and the trace
/// ran. `#[non_exhaustive]` so further per-endpoint diagnostics can be added later;
/// the convergence/step fields carry `#[serde(default)]` so a record serialized
/// before they existed still deserializes (defaulting to `false`/`0`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IrcEndpoints {
    /// Geometry reached following the reaction mode in the `+` sense
    /// (Cartesian, atomic units, input atom order).
    pub forward: Vec<[f64; 3]>,
    /// Energy at the forward endpoint.
    pub forward_energy: f64,
    /// Whether the forward trace reached a minimum (the projected force fell below
    /// [`TsOptions::irc_gtol`](super::TsOptions::irc_gtol) after descending into a
    /// basin) rather than exhausting [`TsOptions::irc_max_steps`](super::TsOptions::irc_max_steps).
    #[serde(default)]
    pub forward_converged: bool,
    /// Integration steps taken to reach the forward endpoint.
    #[serde(default)]
    pub forward_steps: usize,
    /// Geometry reached following the reaction mode in the `-` sense.
    pub reverse: Vec<[f64; 3]>,
    /// Energy at the reverse endpoint.
    pub reverse_energy: f64,
    /// Whether the reverse trace reached a minimum (see [`forward_converged`](Self::forward_converged)).
    #[serde(default)]
    pub reverse_converged: bool,
    /// Integration steps taken to reach the reverse endpoint.
    #[serde(default)]
    pub reverse_steps: usize,
}

/// Trace the IRC downhill from `saddle` in both senses of `reaction_mode`.
///
/// `reaction_mode` is the normalized **Cartesian** reaction-mode eigenvector
/// ([`SaddleVerification::reaction_mode`](super::SaddleVerification::reaction_mode));
/// it is converted to the mass-weighted transition direction here (the IRC is a
/// steepest descent in mass-weighted coordinates, so the path direction must be
/// mass-weighted, not the raw Cartesian mode). One Hessian is computed and cached
/// for the whole run only when the integrator needs it ([`EulerPc`](IrcMethod::EulerPc)).
///
/// `saddle_energy` is the converged saddle energy; a trace must drop below it before
/// its endpoint is accepted as a minimum (so a flat saddle is not mistaken for a basin).
///
/// `cached_hessian`, when supplied, is the Cartesian Hessian already computed at
/// `saddle` (by the post-convergence verification); the [`EulerPc`](IrcMethod::EulerPc)
/// integrator reuses it instead of finite-differencing a second one. The other
/// integrators never form a Hessian and ignore it.
fn irc_endpoints<S: Surface>(
    surface: &mut S,
    saddle: &[[f64; 3]],
    reaction_mode: &[[f64; 3]],
    masses: &[f64],
    saddle_energy: f64,
    options: &TsOptions,
    cached_hessian: Option<&[f64]>,
) -> Result<IrcEndpoints, TsError> {
    let dir = mw_transition_dir(reaction_mode, masses);
    let neg: Vec<f64> = dir.iter().map(|c| -c).collect();

    // The Hessian-corrector method reuses one Hessian across both endpoints: the
    // verification's cached Hessian at the saddle when available, else a fresh one.
    // The other integrators never form one.
    let hess = if matches!(options.irc_method, IrcMethod::EulerPc) {
        match cached_hessian {
            Some(h) => Some(h.to_vec()),
            None => Some(fd_hessian(surface, saddle, options.fd_step)?),
        }
    } else {
        None
    };
    let hess = hess.as_deref();

    let fwd = integrate_endpoint(surface, saddle, &dir, masses, saddle_energy, hess, options)?;
    let rev = integrate_endpoint(surface, saddle, &neg, masses, saddle_energy, hess, options)?;
    Ok(IrcEndpoints {
        forward: fwd.positions,
        forward_energy: fwd.energy,
        forward_converged: fwd.converged,
        forward_steps: fwd.steps,
        reverse: rev.positions,
        reverse_energy: rev.energy,
        reverse_converged: rev.converged,
        reverse_steps: rev.steps,
    })
}

/// The post-convergence IRC confirmation shared by both drivers: when
/// [`confirm_irc`](TsOptions::confirm_irc) is set and `verification` carries a
/// reaction mode (a first-order saddle), trace the path off `saddle` and return the
/// endpoints, reusing the verification's `hessian` for a Hessian-corrector run.
/// Returns `None` when confirmation is off, there is no reaction mode, or the
/// (purely confirmatory) trace hits a recoverable SCF failure — the last case must
/// not turn a converged saddle into a hard error.
pub(super) fn confirm_irc_endpoints<S: Surface>(
    surface: &mut S,
    saddle: &[[f64; 3]],
    verification: &SaddleVerification,
    masses: &[f64],
    saddle_energy: f64,
    options: &TsOptions,
    hessian: &[f64],
) -> Result<Option<IrcEndpoints>, TsError> {
    if !options.confirm_irc {
        return Ok(None);
    }
    let Some(mode) = &verification.reaction_mode else {
        return Ok(None);
    };
    match irc_endpoints(
        surface,
        saddle,
        mode,
        masses,
        saddle_energy,
        options,
        Some(hessian),
    ) {
        Ok(endpoints) => Ok(Some(endpoints)),
        Err(TsError::SurfaceEvaluation(OptError::ScfNotConverged { .. })) => Ok(None),
        Err(e) => Err(e),
    }
}

/// One traced reaction-path endpoint: where the integration stopped, its energy,
/// whether it reached a minimum, and how many steps it took.
struct EndpointTrace {
    positions: Vec<[f64; 3]>,
    energy: f64,
    converged: bool,
    steps: usize,
}

/// Integrate one sense of the reaction path until it reaches a minimum or exhausts
/// the step budget.
fn integrate_endpoint<S: Surface>(
    surface: &mut S,
    saddle: &[[f64; 3]],
    dir: &[f64],
    masses: &[f64],
    saddle_energy: f64,
    hess: Option<&[f64]>,
    options: &TsOptions,
) -> Result<EndpointTrace, TsError> {
    let step = options.irc_step;
    // At least one integration step (mirrors the CLI's `--ts-irc-max-steps >= 1`, but
    // guards the library/serde path too), so the reported endpoint is never the raw,
    // un-integrated seed.
    let max_steps = options.irc_max_steps.max(1);
    // Leave the saddle ridge along the (signed) transition direction: the gradient
    // is ~0 at the saddle, so the gradient-following integrators need a finite kick
    // to pick up the downhill direction.
    let mut x = add_step(saddle, &unmass_weight_step(&scale(dir, step), masses));
    // Velocity state for the damped trajectory (unused by the other integrators).
    let mut velocity = scale(dir, step);
    let mut prev_energy = f64::INFINITY;

    let mut converged = false;
    let mut taken = 0usize;
    for s in 1..=max_steps {
        taken = s;
        let g_cart = gradient(surface, &x, options.fd_step)?;
        if !finite(&g_cart) {
            return Err(TsError::Numerical(
                "surface returned a non-finite gradient during IRC integration".to_string(),
            ));
        }
        // Energy at the *same* geometry as the gradient — a cache hit for a surface
        // that caches its last SCF (`HfSurface`), so this costs no extra evaluation.
        let energy = surface.energy(&x)?;
        let (_, rms) = projected_force_norms(&g_cart, masses, &x);
        // A trace has reached a minimum only once it is both force-converged *and*
        // has descended into a basin: the second clause stops a flat/low-barrier
        // saddle (tiny force a step off the ridge) being reported as a converged
        // minimum sitting on the saddle.
        let in_basin = energy < saddle_energy - IRC_BASIN_EPS;
        if rms < options.irc_gtol && in_basin {
            converged = true;
            break;
        }
        let g_mw = mw_grad_projected(&g_cart, masses, &x);
        // Take full arc-length strides down the ridge/shoulder until the trace has
        // dropped into a basin, then settle with a unit-rate steepest-descent step
        // (capped at the arc length) so it homes onto the minimum instead of orbiting
        // it at the fixed arc length. The regime switch is on energy, not a force/length
        // comparison. The damped trajectory (Dvv) settles on its own and ignores this.
        let step_eff = if in_basin {
            (IRC_SD_RATE * norm(&g_mw)).min(step)
        } else {
            step
        };
        let mut dq = match options.irc_method {
            IrcMethod::Dvv => {
                // Damped leapfrog v ← γ·(v − Δt·g_mw); restart the velocity if the last
                // step climbed, so a stiff mode cannot oscillate under the fixed Δt.
                if energy > prev_energy {
                    velocity.iter_mut().for_each(|v| *v = 0.0);
                }
                for (vi, &gi) in velocity.iter_mut().zip(&g_mw) {
                    *vi = IRC_DAMP * (*vi - IRC_DT * gi);
                }
                cap_norm(&mut velocity, step);
                scale(&velocity, IRC_DT)
            }
            IrcMethod::EulerPc => eulerpc_step(&g_mw, &g_cart, hess, &x, masses, step_eff),
            IrcMethod::GonzalezSchlegel => gs2_step(surface, &x, &g_mw, masses, step_eff, options)?,
            // `IrcMethod` is `#[non_exhaustive]`: fall back to a plain steepest-descent
            // step so a future variant still integrates rather than panicking.
            #[allow(unreachable_patterns)]
            _ => scale(&unit(&g_mw), -step_eff),
        };
        cap_norm(&mut dq, step);
        prev_energy = energy;
        x = add_step(&x, &unmass_weight_step(&dq, masses));
    }

    let energy = surface.energy(&x)?;
    if !energy.is_finite() {
        return Err(TsError::Numerical(
            "surface returned a non-finite energy at an IRC endpoint".to_string(),
        ));
    }
    Ok(EndpointTrace {
        positions: x,
        energy,
        converged,
        steps: taken,
    })
}

/// Euler predictor + Hessian corrector (Heun's method for the steepest-descent ODE).
/// The predictor gradient is modeled from the cached Hessian, so the corrected step
/// is second-order accurate at no extra surface evaluation. Returns the mass-weighted
/// step.
fn eulerpc_step(
    g_mw: &[f64],
    g_cart: &[[f64; 3]],
    hess: Option<&[f64]>,
    x: &[[f64; 3]],
    masses: &[f64],
    step: f64,
) -> Vec<f64> {
    let Some(hess) = hess else {
        // No cached Hessian: degrade to the Euler predictor alone.
        return scale(&unit(g_mw), -step);
    };
    let ndof = g_mw.len();
    let ghat = unit(g_mw);
    // Predictor: an Euler step −step·ĝ in mass-weighted space; its Cartesian form
    // feeds the Hessian-vector product for the modeled predictor gradient.
    let dx_pred_cart = unmass_weight_step(&scale(&ghat, -step), masses);
    let x_pred = add_step(x, &dx_pred_cart);
    let hdx = matvec(hess, &flatten(&dx_pred_cart), ndof);
    let g_pred: Vec<f64> = flatten(g_cart)
        .iter()
        .zip(&hdx)
        .map(|(a, b)| a + b)
        .collect();
    // Project the predicted gradient in the predictor geometry's own rigid-body frame.
    let ghat_pred = unit(&mw_grad_projected(&unflatten(&g_pred), masses, &x_pred));
    // Corrector: average the current and predicted descent directions.
    (0..ndof)
        .map(|i| -0.5 * step * (ghat[i] + ghat_pred[i]))
        .collect()
}

/// Gonzalez–Schlegel constrained step: place a pivot half a step back along the
/// descent direction, then find the point on the hypersphere of radius `step/2`
/// about it whose true gradient is parallel to the radius (a point on the reaction
/// path). Returns the mass-weighted step.
///
/// The hypersphere meets the path at two points: the *descent* root, where the path
/// gradient points back toward the pivot (the outward radial component `gn < 0`), and
/// an *uphill* root essentially back at `x` (`gn > 0`). Only the descent root is a
/// valid forward step. The constrained point is returned only when the micro-iteration
/// converges to it; if the iteration instead exhausts its budget without converging (a
/// silent stall) or settles on the uphill root, the step degrades to the plain
/// steepest-descent (Euler) step — always a descent direction — rather than emitting
/// the last half-rotated, off-path point.
pub(super) fn gs2_step<S: Surface>(
    surface: &mut S,
    x: &[[f64; 3]],
    g_mw: &[f64],
    masses: &[f64],
    step: f64,
    options: &TsOptions,
) -> Result<Vec<f64>, TsError> {
    let r = 0.5 * step;
    let ghat = unit(g_mw);
    let pivot = scale(&ghat, -r); // mass-weighted offset from x to the pivot
    // The steepest-descent (Euler) step: both the initial guess and the
    // guaranteed-descent fallback used when the constrained search fails to reach a
    // valid on-path point.
    let sd = scale(&ghat, -step);
    let mut s = sd.clone();
    // Set true only once the micro-iteration reaches the descent root (`gn < 0`).
    let mut on_path = false;
    for _ in 0..GS2_INNER {
        let xp = add_step(x, &unmass_weight_step(&s, masses));
        let gp = mw_grad_projected(&gradient(surface, &xp, options.fd_step)?, masses, &xp);
        let from_pivot: Vec<f64> = s.iter().zip(&pivot).map(|(a, b)| a - b).collect();
        let n = unit(&from_pivot);
        let gn = dot(&gp, &n);
        let gperp: Vec<f64> = gp.iter().zip(&n).map(|(g, ni)| g - gn * ni).collect();
        if norm(&gperp) < GS2_TOL {
            // Accept the constrained point only at the descent root; the uphill root
            // (gn > 0, ~back at x) would stall the trace, so reject it like a non-step.
            on_path = gn < 0.0;
            break;
        }
        // Rotate the point around the pivot toward −g_perp, staying on the sphere.
        let dirv: Vec<f64> = from_pivot
            .iter()
            .zip(&gperp)
            .map(|(f, gp)| f - r * gp)
            .collect();
        let nd = unit(&dirv);
        s = pivot.iter().zip(&nd).map(|(p, d)| p + r * d).collect();
    }
    // The constrained point only when the micro-iteration actually reached the path;
    // otherwise the steepest-descent step, never a half-rotated, off-path point.
    Ok(if on_path { s } else { sd })
}

/// The normalized mass-weighted transition direction from the normalized Cartesian
/// reaction mode: a coordinate displacement mass-weights as `√m · Δx` (the opposite
/// of a gradient), so multiply each component by `√m` and renormalize.
pub(super) fn mw_transition_dir(mode: &[[f64; 3]], masses: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0f64; 3 * masses.len()];
    for (a, mi) in masses.iter().enumerate() {
        let sm = mi.sqrt();
        for c in 0..3 {
            v[3 * a + c] = mode[a][c] * sm;
        }
    }
    unit(&v)
}

/// The mass-weighted gradient with the rigid-body (translation/rotation) component
/// projected out — the descent direction lives in this internal subspace.
fn mw_grad_projected(g_cart: &[[f64; 3]], masses: &[f64], x: &[[f64; 3]]) -> Vec<f64> {
    let basis = gram_schmidt(&trans_rot_vectors(x, masses));
    let mut v = mass_weight_grad(g_cart, masses);
    for b in &basis {
        let p = dot(&v, b);
        for (vi, &bi) in v.iter_mut().zip(b) {
            *vi -= p * bi;
        }
    }
    v
}

fn unflatten(v: &[f64]) -> Vec<[f64; 3]> {
    (0..v.len() / 3)
        .map(|a| [v[3 * a], v[3 * a + 1], v[3 * a + 2]])
        .collect()
}

fn scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

/// Normalized copy of `v` (unchanged if `v` is the zero vector).
fn unit(v: &[f64]) -> Vec<f64> {
    let n = norm(v);
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

/// Scale `v` in place so its norm does not exceed `cap`.
fn cap_norm(v: &mut [f64], cap: f64) {
    let n = norm(v);
    if n > cap && n > 0.0 {
        let s = cap / n;
        for x in v.iter_mut() {
            *x *= s;
        }
    }
}

fn finite(g: &[[f64; 3]]) -> bool {
    g.iter().all(|v| v.iter().all(|c| c.is_finite()))
}
