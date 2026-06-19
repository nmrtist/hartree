//! The dimer's rotation step and the vector primitives it shares with
//! [`super::dimer`]'s translation step.
//!
//! Each outer dimer iteration first *rotates* the unit axis `N` toward the
//! lowest-curvature mode — finite-differencing the curvature along `N` from a
//! single nearby endpoint gradient and aligning by a Fourier model of the
//! rotation force ([`rotate_to_min_mode`]) — before the caller translates the
//! midpoint along the aligned axis. Everything here works in the mass-weighted,
//! translation/rotation-projected frame the search runs in.

use std::f64::consts::PI;

use super::TsError;
use super::numerics::{add_step, dot, gradient, mass_weight_grad, norm, unmass_weight_step};
use crate::opt::Surface;

/// Maximum dimer rotations per outer (translation) step.
const MAX_ROT: usize = 4;
/// Below this projected-axis norm the carried dimer axis is treated as degenerate
/// and reseeded.
pub(super) const AXIS_DEGEN_EPS: f64 = 1e-8;
/// Perpendicular gradient-difference norm below which the axis is taken to be
/// aligned with the lowest-curvature mode (rotation converged).
const GPERP_TOL: f64 = 1e-6;
/// Rotational-force magnitude below which a further rotation is not worthwhile.
const FROT_TOL: f64 = 1e-3;
/// Rotation angle (radians) below which the trial rotation is treated as a no-op.
const ROT_ANGLE_TOL: f64 = 1e-3;

/// Project a flat mass-weighted vector onto the internal subspace: subtract its
/// components along each (orthonormal) translation/rotation basis vector.
pub(super) fn project_internal(v: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
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
pub(super) fn normalize(v: &mut [f64]) -> f64 {
    let n = norm(v);
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    n
}

/// Fail fast if any component of a per-atom gradient is non-finite, attributing
/// the fault to `what` (e.g. the midpoint or an endpoint gradient). Returning a
/// clear [`TsError::Numerical`] here stops a poisoned gradient from churning
/// silently through the iteration budget.
pub(super) fn require_finite(g: &[[f64; 3]], what: &str) -> Result<(), TsError> {
    if g.iter().flatten().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(TsError::Numerical(format!(
            "{what} carried a non-finite value"
        )))
    }
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
    require_finite(&g_cart, "dimer endpoint gradient")?;
    Ok(project_internal(&mass_weight_grad(&g_cart, masses), basis))
}

/// Curvature `C ≈ NᵀHN` from one endpoint gradient: `(g1 - g0)·N / Δ`. Used both
/// inside the rotation loop and by [`super::dimer`] for the curvature at the
/// converged axis (when the rotation did not return one because the axis moved on
/// its final pass).
#[allow(clippy::too_many_arguments)]
pub(super) fn endpoint_curvature<S: Surface>(
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

/// Rotate `n_axis` (in place) toward the lowest-curvature mode at midpoint `x`,
/// returning the curvature along the aligned axis when the rotation converges
/// within the inner-loop budget (`None` if it ran out of rotations without an
/// early-exit alignment, leaving the caller to finite-difference the curvature
/// afresh). The axis is kept projected into the internal subspace and normalized.
#[allow(clippy::too_many_arguments)]
pub(super) fn rotate_to_min_mode<S: Surface>(
    surface: &mut S,
    x: &[[f64; 3]],
    n_axis: &mut Vec<f64>,
    g0: &[f64],
    delta: f64,
    masses: &[f64],
    basis: &[Vec<f64>],
    fd_step: f64,
) -> Result<Option<f64>, TsError> {
    for _ in 0..MAX_ROT {
        let g1 = endpoint_grad(surface, x, n_axis, delta, masses, basis, fd_step)?;
        let d: Vec<f64> = g1.iter().zip(g0).map(|(a, b)| a - b).collect();
        let gpar = dot(&d, n_axis);
        let curvature = gpar / delta;
        let gperp: Vec<f64> = d
            .iter()
            .zip(n_axis.iter())
            .map(|(di, ni)| di - gpar * ni)
            .collect();
        let gperp_norm = norm(&gperp);
        if gperp_norm < GPERP_TOL {
            return Ok(Some(curvature));
        }
        let theta: Vec<f64> = gperp.iter().map(|g| g / gperp_norm).collect();

        let frot_norm = 2.0 * gperp_norm;
        if frot_norm < FROT_TOL {
            return Ok(Some(curvature));
        }
        let b1 = dot(&d, &theta) / delta;

        // Trial rotation by π/4 to estimate the curvature's Fourier model.
        let phi1 = PI / 4.0;
        let mut nt = rotate(n_axis, &theta, phi1.cos(), phi1.sin());
        nt = project_internal(&nt, basis);
        normalize(&mut nt);
        let c1 = endpoint_curvature(surface, x, &nt, g0, delta, masses, basis, fd_step)?;

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

        *n_axis = rotate(n_axis, &theta, phi_min.cos(), phi_min.sin());
        *n_axis = project_internal(n_axis, basis);
        normalize(n_axis);

        // Early break if the rotation barely moves the axis.
        if phi_min.abs() < ROT_ANGLE_TOL {
            break;
        }
    }
    Ok(None)
}

/// First-iteration dimer axis from the mass-weighted reaction-coordinate seed:
/// project it into the internal subspace and normalize. Returns `None` when no
/// seed was given or it projects to (near-)zero in the internal subspace, so the
/// caller falls back to [`initial_axis`].
pub(super) fn seed_axis(seed_mw: Option<&[f64]>, basis: &[Vec<f64>]) -> Option<Vec<f64>> {
    let seed = seed_mw?;
    let mut a = project_internal(seed, basis);
    (normalize(&mut a) >= AXIS_DEGEN_EPS).then_some(a)
}

/// First-iteration dimer axis: the projected gradient direction; if it is
/// (near-)zero, the first canonical internal-subspace unit vector.
pub(super) fn initial_axis(g0: &[f64], basis: &[Vec<f64>], ndof: usize) -> Vec<f64> {
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
