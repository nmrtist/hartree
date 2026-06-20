//! Geometry optimization: redundant internal coordinates, BFGS/RFO, and finite-difference drivers.

pub mod constrained;
pub mod fd;
pub mod internals;
pub mod ts;

use crate::core::Molecule;
use crate::linalg::{mat_from_row_major, mat_to_row_major, symmetric_eigh_checked};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use internals::Internal;

// `#[non_exhaustive]` (matching `TsError`) so future surface-failure modes can be
// added non-breakingly; the only matches on `OptError` (the job error flatteners)
// carry a wildcard arm, so this stays safe.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OptError {
    #[error("surface evaluation failed: {0}")]
    Evaluation(String),
    /// The SCF did not reach self-consistency within its iteration cap. A distinct,
    /// branchable signal (rather than prose inside `Evaluation`) so a caller can
    /// respond programmatically — tighten the SCF, change the initial guess, or
    /// raise the level shift — and retry. `iterations` is the SCF iteration count
    /// reached (the cap, when the cap was exhausted).
    #[error("SCF did not converge in {iterations} iterations")]
    ScfNotConverged { iterations: usize },
    /// A numerical kernel (e.g. the RFO eigensolver) hit non-finite or
    /// ill-conditioned data. Surfaced as `Err` — rather than a panic — so a caller
    /// can recover (retry, rebuild the matrix) instead of the process aborting.
    #[error("numerical failure: {0}")]
    Numerical(String),
}

pub trait Surface {
    fn energy(&mut self, positions: &[[f64; 3]]) -> Result<f64, OptError>;

    fn analytic_gradient(
        &mut self,
        positions: &[[f64; 3]],
    ) -> Option<Result<Vec<[f64; 3]>, OptError>>;

    /// Optional fast Cartesian finite-difference Hessian (row-major, symmetrized);
    /// an implementation may run the `2·ndof` independent gradient evaluations in
    /// parallel. `None` (the default) selects the driver's serial finite difference.
    fn fd_hessian(
        &mut self,
        _positions: &[[f64; 3]],
        _fd_step: f64,
    ) -> Option<Result<Vec<f64>, OptError>> {
        None
    }

    /// Optional model/approximate Cartesian Hessian (row-major, `ndof×ndof`) to
    /// *seed* a saddle search's initial climbing Hessian — e.g. from a cheap force
    /// field or a learned surrogate. `None` (the default) makes the driver build the
    /// initial Hessian by finite difference. Unlike [`fd_hessian`](Self::fd_hessian)
    /// (a fast path for the *exact* finite-difference Hessian), a seed need not be
    /// accurate: P-RFO refines it by quasi-Newton (Bofill) updates and the
    /// post-convergence verification is independent of it, so it only has to point
    /// the climb in roughly the right curvature directions. The optimizer consults
    /// it according to the caller's `hessian_init` policy.
    fn seed_hessian(&mut self, _positions: &[[f64; 3]]) -> Option<Result<Vec<f64>, OptError>> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct OptOptions {
    pub max_iter: usize,
    pub trust_radius: f64,
    pub max_trust: f64,
    pub min_trust: f64,
    pub fd_step: f64,
    pub max_force: f64,
    pub rms_force: f64,
    pub max_disp: f64,
    pub rms_disp: f64,
}

impl Default for OptOptions {
    fn default() -> Self {
        Self {
            max_iter: 150,
            trust_radius: 0.3,
            max_trust: 1.0,
            min_trust: 1e-4,
            fd_step: 5e-3,
            max_force: 3.0e-6,
            rms_force: 2.0e-6,
            max_disp: 3.0e-5,
            rms_disp: 2.0e-5,
        }
    }
}

// `Serialize`/`Deserialize` so the trace can travel inside the (serde) TS
// result objects the agent consumes; `OptResult` reuses the same step record.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OptStep {
    pub iteration: usize,
    pub energy: f64,
    pub max_force: f64,
    pub rms_force: f64,
    pub max_disp: f64,
    pub rms_disp: f64,
}

#[derive(Debug, Clone)]
pub struct OptResult {
    pub positions: Vec<[f64; 3]>,
    pub energy: f64,
    pub converged: bool,
    pub iterations: usize,
    pub history: Vec<OptStep>,
}

pub(crate) const MAX_TRUST_RETRIES: usize = 8;

pub fn optimize<S: Surface>(
    molecule: &Molecule,
    surface: &mut S,
    options: &OptOptions,
) -> Result<OptResult, OptError> {
    let defs = internals::generate(molecule);
    let natom = molecule.len();
    let ndof = 3 * natom;
    let nq = defs.len();

    let mut x: Vec<[f64; 3]> = molecule.atoms.iter().map(|a| a.position).collect();
    let mut energy = surface.energy(&x)?;
    let mut gx = eval_gradient(surface, &x, options)?;
    let mut gq = {
        let b = internals::wilson_b(&defs, &x);
        internals::internal_gradient(&b, &flatten(&gx), nq, ndof)
    };
    let mut hessian = init_hessian(&defs);
    let mut trust = options.trust_radius;

    let mut history = Vec::new();
    let mut x_prev: Option<Vec<[f64; 3]>> = None;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let (max_force, rms_force) = force_norms(&gx);
        let (max_disp, rms_disp) = match &x_prev {
            Some(xp) => disp_norms(&x, xp),
            None => (0.0, 0.0),
        };
        history.push(OptStep {
            iteration: iter,
            energy,
            max_force,
            rms_force,
            max_disp,
            rms_disp,
        });

        if max_force < options.max_force
            && rms_force < options.rms_force
            && max_disp < options.max_disp
            && rms_disp < options.rms_disp
        {
            converged = true;
            break;
        }
        if iter == options.max_iter {
            break;
        }

        let q = internals::values(&defs, &x);
        let mut retries = 0;
        loop {
            retries += 1;
            let dq = rfo_step(&hessian, &gq, nq, trust)?;
            let predicted = predicted_change(&gq, &hessian, &dq, nq);
            let x_new = internals::back_transform(&defs, &x, &dq);
            let energy_new = surface.energy(&x_new)?;
            let actual = energy_new - energy;

            let force_accept = energy_new <= energy + 1e-12;
            let force_anyway = retries >= MAX_TRUST_RETRIES || trust <= options.min_trust * 1.0001;

            if force_accept || force_anyway {
                let gx_new = eval_gradient(surface, &x_new, options)?;
                let b_new = internals::wilson_b(&defs, &x_new);
                let gq_new = internals::internal_gradient(&b_new, &flatten(&gx_new), nq, ndof);

                if actual < 0.0 {
                    let q_new = internals::values(&defs, &x_new);
                    // Dihedral components are wrapped into (−π, π] so a torsion
                    // crossing the ±π branch yields its true step, not a near-2π jump.
                    let s = internals::displacement(&defs, &q_new, &q);
                    let y: Vec<f64> = gq_new.iter().zip(&gq).map(|(a, b)| a - b).collect();
                    bfgs_update(&mut hessian, &s, &y, nq);
                }

                trust = update_trust(trust, actual, predicted, norm(&dq), options);

                x_prev = Some(x.clone());
                x = x_new;
                energy = energy_new;
                gx = gx_new;
                gq = gq_new;
                break;
            }

            trust = (0.25 * trust).max(options.min_trust);
        }
    }

    Ok(OptResult {
        positions: x,
        energy,
        converged,
        iterations,
        history,
    })
}

pub(crate) fn eval_gradient<S: Surface>(
    surface: &mut S,
    x: &[[f64; 3]],
    options: &OptOptions,
) -> Result<Vec<[f64; 3]>, OptError> {
    match surface.analytic_gradient(x) {
        Some(result) => result,
        None => fd::central_difference(surface, x, options.fd_step),
    }
}

pub(crate) fn init_hessian(defs: &[Internal]) -> Vec<f64> {
    let nq = defs.len();
    let mut h = vec![0.0; nq * nq];
    for (i, d) in defs.iter().enumerate() {
        h[i * nq + i] = match d {
            Internal::Bond(..) => 0.5,
            Internal::Angle(..) => 0.2,
            // Torsions are the softest internal mode; a small positive seed keeps the
            // initial Hessian positive-definite without over-restraining the rotation.
            Internal::Dihedral(..) => 0.1,
            // A co-linear bend is a bending mode like a valence angle; seed it likewise.
            Internal::LinearBend(..) => 0.2,
        };
    }
    h
}

fn rfo_step(hessian: &[f64], grad: &[f64], nq: usize, trust: f64) -> Result<Vec<f64>, OptError> {
    if nq == 0 {
        return Ok(Vec::new());
    }
    let m = nq + 1;
    let mut aug = vec![0.0; m * m];
    for i in 0..nq {
        for j in 0..nq {
            aug[i * m + j] = hessian[i * nq + j];
        }
        aug[i * m + nq] = grad[i];
        aug[nq * m + i] = grad[i];
    }
    let eig = symmetric_eigh_checked(&mat_from_row_major(m, &aug)).map_err(OptError::Numerical)?;
    let vectors = mat_to_row_major(&eig.vectors); // column 0 = lowest-eigenvalue vector
    let last = vectors[nq * m]; // row nq, column 0

    let mut dq = vec![0.0; nq];
    if last.abs() > 1e-8 {
        for (i, slot) in dq.iter_mut().enumerate() {
            *slot = vectors[i * m] / last;
        }
    } else {
        for (i, slot) in dq.iter_mut().enumerate() {
            *slot = -grad[i];
        }
    }

    let n = norm(&dq);
    if n > trust {
        let scale = trust / n;
        for v in &mut dq {
            *v *= scale;
        }
    }
    Ok(dq)
}

pub(crate) fn predicted_change(grad: &[f64], hessian: &[f64], dq: &[f64], nq: usize) -> f64 {
    let mut p = 0.0;
    for i in 0..nq {
        p += grad[i] * dq[i];
        let mut hd = 0.0;
        for j in 0..nq {
            hd += hessian[i * nq + j] * dq[j];
        }
        p += 0.5 * dq[i] * hd;
    }
    p
}

pub(crate) fn bfgs_update(hessian: &mut [f64], s: &[f64], y: &[f64], nq: usize) {
    let sy = dot(s, y);
    if sy <= 1e-10 {
        return;
    }
    let mut hs = vec![0.0; nq];
    for i in 0..nq {
        let mut acc = 0.0;
        for j in 0..nq {
            acc += hessian[i * nq + j] * s[j];
        }
        hs[i] = acc;
    }
    let shs = dot(s, &hs);
    if shs <= 1e-12 {
        return;
    }
    for i in 0..nq {
        for j in 0..nq {
            hessian[i * nq + j] += y[i] * y[j] / sy - hs[i] * hs[j] / shs;
        }
    }
}

pub(crate) fn update_trust(
    trust: f64,
    actual: f64,
    predicted: f64,
    step_norm: f64,
    opts: &OptOptions,
) -> f64 {
    if actual >= 0.0 {
        return (0.25 * trust).max(opts.min_trust);
    }
    let ratio = if predicted.abs() > 1e-14 {
        actual / predicted
    } else {
        1.0
    };
    if ratio > 0.75 && step_norm > 0.8 * trust {
        (2.0 * trust).min(opts.max_trust)
    } else if ratio < 0.25 {
        (0.25 * trust).max(opts.min_trust)
    } else {
        trust
    }
}

pub(crate) fn force_norms(gx: &[[f64; 3]]) -> (f64, f64) {
    let mut max = 0.0_f64;
    let mut sum_sq = 0.0;
    let mut count = 0;
    for v in gx {
        for &c in v {
            max = max.max(c.abs());
            sum_sq += c * c;
            count += 1;
        }
    }
    (max, (sum_sq / count as f64).sqrt())
}

pub(crate) fn disp_norms(x: &[[f64; 3]], x_prev: &[[f64; 3]]) -> (f64, f64) {
    let mut max = 0.0_f64;
    let mut sum_sq = 0.0;
    let mut count = 0;
    for (a, b) in x.iter().zip(x_prev) {
        for k in 0..3 {
            let d = a[k] - b[k];
            max = max.max(d.abs());
            sum_sq += d * d;
            count += 1;
        }
    }
    (max, (sum_sq / count as f64).sqrt())
}

pub(crate) fn flatten(g: &[[f64; 3]]) -> Vec<f64> {
    let mut out = Vec::with_capacity(g.len() * 3);
    for v in g {
        out.extend_from_slice(v);
    }
    out
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests;
