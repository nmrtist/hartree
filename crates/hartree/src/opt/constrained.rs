//! Constrained minimization in redundant internal coordinates: hold (or drive) a
//! chosen set of internal coordinates at target values while every other internal
//! coordinate relaxes to a minimum.
//!
//! The method follows Baker, "Constrained optimization in delocalized internal
//! coordinates" (J. Comput. Chem. 18, 1080 (1997)) and the projection of
//! Bakken–Helgaker (J. Chem. Phys. 117, 9160 (2002)): the redundant-internal step
//! is split into a *free* block and a *constrained* block. Each iteration takes a
//! quasi-Newton (RFO) step on the free subspace alone — the reduced gradient and
//! reduced Hessian with the constrained rows and columns removed — while the
//! constrained coordinates are driven straight toward their targets,
//! `dq_C = target_C − q_C`. The coupling term `H_FC · dq_C` is folded into the free
//! gradient so the free relaxation anticipates the drive. The two blocks are
//! reassembled into a full internal step and back-transformed to Cartesians by the
//! same Wilson-B iteration the unconstrained minimizer uses.
//!
//! At convergence the drive vanishes (`dq_C → 0`, so each constrained coordinate
//! sits *exactly* at its target) and the reduced gradient vanishes (`g_F → 0`, the
//! constrained minimum), which is the exact KKT stationarity condition for the
//! equality-constrained problem. This is a NEW entry point: it does not touch
//! [`optimize`](super::optimize) or its default behaviour, and reuses that module's
//! shared step helpers.

use crate::core::Molecule;
use crate::linalg::{mat_from_row_major, mat_to_row_major, symmetric_eigh_checked};
use crate::opt::internals::{self, Internal};
use crate::opt::{
    MAX_TRUST_RETRIES, OptError, OptOptions, OptResult, OptStep, Surface, bfgs_update,
    eval_gradient, flatten, force_norms, init_hessian, norm, predicted_change, update_trust,
};

/// One internal coordinate held (or driven) at a target value during a constrained
/// minimization. `coordinate` must be one of the coordinates the redundant set
/// [`generate`](internals::generate)s for the molecule (matched by its atom indices);
/// `target` is the value (Bohr for a bond, radians for an angle/dihedral, the
/// dimensionless projection for a linear bend) the coordinate is held at or driven to.
#[derive(Debug, Clone, Copy)]
pub struct Constraint {
    pub coordinate: Internal,
    pub target: f64,
}

/// Minimize `surface` subject to holding each [`Constraint`] at its target, working in
/// redundant internal coordinates.
///
/// The redundant set is the same one [`optimize`](super::optimize) builds, augmented
/// with any constrained coordinate that the automatic generator did not already include
/// (so an arbitrary driven coordinate is always representable). Convergence requires the
/// free forces to fall below [`OptOptions`]'s force tolerances *and* every constraint
/// residual to fall below a fixed `1e-8` internal tolerance (tighter than the displacement
/// tolerance, so on return every constrained coordinate sits at its target to within
/// numerical noise). The constraint residual is reported in the result's final history
/// step as the `max_disp`/`rms_disp` of the driven block when it dominates.
///
/// # Errors
/// [`OptError`] if a surface evaluation, gradient, or the RFO eigensolver fails.
pub fn optimize_constrained<S: Surface>(
    molecule: &Molecule,
    surface: &mut S,
    constraints: &[Constraint],
    options: &OptOptions,
) -> Result<OptResult, OptError> {
    let natom = molecule.len();
    let ndof = 3 * natom;

    let mut x: Vec<[f64; 3]> = molecule.atoms.iter().map(|a| a.position).collect();

    // Start from the automatic redundant set and append any constrained coordinate it
    // does not already contain, so every driven coordinate has a row in the internal
    // representation regardless of the covalent-radius connectivity.
    let mut defs = internals::generate(molecule);
    let mut constrained_idx = Vec::with_capacity(constraints.len());
    let mut targets = Vec::with_capacity(constraints.len());
    for c in constraints {
        let idx = match defs.iter().position(|d| same_coordinate(d, &c.coordinate)) {
            Some(i) => i,
            None => {
                defs.push(c.coordinate);
                defs.len() - 1
            }
        };
        constrained_idx.push(idx);
        targets.push(c.target);
    }
    let nq = defs.len();
    let is_constrained = |i: usize| constrained_idx.contains(&i);

    let mut energy = surface.energy(&x)?;
    let mut gx = eval_gradient(surface, &x, options)?;
    let mut gq = {
        let b = internals::wilson_b(&defs, &x);
        internals::internal_gradient(&b, &flatten(&gx), nq, ndof)
    };
    let mut hessian = init_hessian(&defs);
    let mut trust = options.trust_radius;

    let mut history = Vec::new();
    let mut converged = false;
    let mut iterations = 0;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let q = internals::values(&defs, &x);

        // Constraint residual: how far each constrained coordinate is from its target,
        // wrapped for torsions so a drive across the ±π branch takes the short arc.
        let mut con_residual = vec![0.0; nq];
        let mut max_con = 0.0_f64;
        for (slot, &ci) in constrained_idx.iter().enumerate() {
            let r = match defs[ci] {
                Internal::Dihedral(..) => internals::wrap_to_pi(targets[slot] - q[ci]),
                _ => targets[slot] - q[ci],
            };
            con_residual[ci] = r;
            max_con = max_con.max(r.abs());
        }

        let (max_force, rms_force) = free_force_norms(&gx, &gq, &is_constrained, nq);
        history.push(OptStep {
            iteration: iter,
            energy,
            max_force,
            rms_force,
            max_disp: max_con,
            rms_disp: max_con,
        });

        if max_force < options.max_force && rms_force < options.rms_force && max_con < 1e-8 {
            converged = true;
            break;
        }
        if iter == options.max_iter {
            break;
        }

        let mut retries = 0;
        loop {
            retries += 1;
            // Free step: RFO on the free subspace, with the constraint-coupling term
            // H_FC·dq_C folded into the free gradient (Baker's reduced gradient).
            let dq = constrained_step(
                &hessian,
                &gq,
                &con_residual,
                &is_constrained,
                nq,
                trust,
                options,
            )?;
            let predicted = predicted_change(&gq, &hessian, &dq, nq);
            let x_new = internals::back_transform(&defs, &x, &dq);
            let energy_new = surface.energy(&x_new)?;
            let actual = energy_new - energy;

            // Accept when the energy does not increase, or when the trust region has
            // shrunk to its floor / the retry budget is spent. The energy may rise while
            // the constraint is still being driven, so also accept any step that reduces
            // the constraint residual — otherwise the drive could stall short of target.
            let q_new = internals::values(&defs, &x_new);
            let new_con = max_constraint_residual(&defs, &q_new, &constrained_idx, &targets);
            let force_accept = energy_new <= energy + 1e-12 || new_con < max_con - 1e-12;
            let force_anyway = retries >= MAX_TRUST_RETRIES || trust <= options.min_trust * 1.0001;

            if force_accept || force_anyway {
                let gx_new = eval_gradient(surface, &x_new, options)?;
                let b_new = internals::wilson_b(&defs, &x_new);
                let gq_new = internals::internal_gradient(&b_new, &flatten(&gx_new), nq, ndof);

                // Quasi-Newton update on the FREE block only: the constrained rows/cols
                // are driven explicitly, so their curvature does not enter the step.
                let s = internals::displacement(&defs, &q_new, &q);
                let y: Vec<f64> = gq_new.iter().zip(&gq).map(|(a, b)| a - b).collect();
                bfgs_update_free(&mut hessian, &s, &y, &is_constrained, nq);

                trust = update_trust(trust, actual, predicted, norm(&dq), options);

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

/// Assemble the full internal step: the free block from an RFO step on the reduced
/// (free-only) gradient and Hessian — with the coupling term `H_FC·dq_C` added to the
/// free gradient — and the constrained block from the drive `dq_C = target_C − q_C`.
fn constrained_step(
    hessian: &[f64],
    grad: &[f64],
    con_residual: &[f64],
    is_constrained: &impl Fn(usize) -> bool,
    nq: usize,
    trust: f64,
    options: &OptOptions,
) -> Result<Vec<f64>, OptError> {
    // Map full indices → reduced (free) indices.
    let free: Vec<usize> = (0..nq).filter(|&i| !is_constrained(i)).collect();
    let nf = free.len();

    // The constrained drive actually applied this step: each constrained coordinate moved
    // toward its target, clamped to the trust bound. The free block's coupling term must
    // anticipate this *same* clamped move (not the raw residual) so the model and the step
    // it scores stay consistent when a large residual is clamped.
    let drive: Vec<f64> = (0..nq)
        .map(|i| {
            if is_constrained(i) {
                con_residual[i].clamp(-options.max_trust, options.max_trust)
            } else {
                0.0
            }
        })
        .collect();

    // Reduced gradient g_F + H_FC·dq_C, and reduced Hessian H_FF.
    let mut g_red = vec![0.0; nf];
    for (a, &i) in free.iter().enumerate() {
        let mut coupling = 0.0;
        for j in 0..nq {
            if is_constrained(j) {
                coupling += hessian[i * nq + j] * drive[j];
            }
        }
        g_red[a] = grad[i] + coupling;
    }
    let mut h_red = vec![0.0; nf * nf];
    for (a, &i) in free.iter().enumerate() {
        for (b, &j) in free.iter().enumerate() {
            h_red[a * nf + b] = hessian[i * nq + j];
        }
    }

    let dq_free = rfo_step(&h_red, &g_red, nf, trust)?;

    // Reassemble: free entries from the RFO step, constrained entries from the clamped
    // drive. The drive is bounded only by the ±max_trust clamp above; it shrinks to zero
    // as the coordinate reaches target, so the clamp matters only on the first steps of a
    // large drive.
    let mut dq = vec![0.0; nq];
    for (a, &i) in free.iter().enumerate() {
        dq[i] = dq_free[a];
    }
    for i in 0..nq {
        if is_constrained(i) {
            dq[i] = drive[i];
        }
    }
    Ok(dq)
}

/// The rational-function-optimization (Newton-with-shift) step on a square block,
/// identical in form to the unconstrained minimizer's `rfo_step` but operating on the
/// supplied (reduced) gradient and Hessian. Returned step is trust-limited.
fn rfo_step(hessian: &[f64], grad: &[f64], n: usize, trust: f64) -> Result<Vec<f64>, OptError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = n + 1;
    let mut aug = vec![0.0; m * m];
    for i in 0..n {
        for j in 0..n {
            aug[i * m + j] = hessian[i * n + j];
        }
        aug[i * m + n] = grad[i];
        aug[n * m + i] = grad[i];
    }
    let eig = symmetric_eigh_checked(&mat_from_row_major(m, &aug)).map_err(OptError::Numerical)?;
    let vectors = mat_to_row_major(&eig.vectors); // column 0 = lowest-eigenvalue vector
    let last = vectors[n * m]; // row n, column 0

    let mut dq = vec![0.0; n];
    if last.abs() > 1e-8 {
        for (i, slot) in dq.iter_mut().enumerate() {
            *slot = vectors[i * m] / last;
        }
    } else {
        for (i, slot) in dq.iter_mut().enumerate() {
            *slot = -grad[i];
        }
    }

    let nrm = norm(&dq);
    if nrm > trust {
        let scale = trust / nrm;
        for v in &mut dq {
            *v *= scale;
        }
    }
    Ok(dq)
}

/// BFGS update restricted to the free block: zero out the constrained components of the
/// step `s` and gradient change `y` before updating, so the maintained curvature is the
/// free Hessian only.
fn bfgs_update_free(
    hessian: &mut [f64],
    s: &[f64],
    y: &[f64],
    is_constrained: &impl Fn(usize) -> bool,
    nq: usize,
) {
    let mut sf = s.to_vec();
    let mut yf = y.to_vec();
    for i in 0..nq {
        if is_constrained(i) {
            sf[i] = 0.0;
            yf[i] = 0.0;
        }
    }
    bfgs_update(hessian, &sf, &yf, nq);
}

/// Force convergence norms over the FREE coordinates only: the constrained block is
/// driven explicitly, so its (generally non-zero) gradient does not gate convergence.
/// Reported in Cartesian terms (`gx`) when nothing is constrained — preserving the
/// unconstrained meaning — and over the free internal gradient `gq` otherwise.
fn free_force_norms(
    gx: &[[f64; 3]],
    gq: &[f64],
    is_constrained: &impl Fn(usize) -> bool,
    nq: usize,
) -> (f64, f64) {
    let any_constrained = (0..nq).any(is_constrained);
    if !any_constrained {
        return force_norms(gx);
    }
    let mut max = 0.0_f64;
    let mut sum_sq = 0.0;
    let mut count = 0;
    for (i, &g) in gq.iter().enumerate() {
        if is_constrained(i) {
            continue;
        }
        max = max.max(g.abs());
        sum_sq += g * g;
        count += 1;
    }
    if count == 0 {
        return (0.0, 0.0);
    }
    (max, (sum_sq / count as f64).sqrt())
}

fn max_constraint_residual(
    defs: &[Internal],
    q: &[f64],
    constrained_idx: &[usize],
    targets: &[f64],
) -> f64 {
    let mut max = 0.0_f64;
    for (slot, &ci) in constrained_idx.iter().enumerate() {
        let r = match defs[ci] {
            Internal::Dihedral(..) => internals::wrap_to_pi(targets[slot] - q[ci]),
            _ => targets[slot] - q[ci],
        };
        max = max.max(r.abs());
    }
    max
}

/// Whether two internal coordinates refer to the same physical coordinate, treating the
/// two index orderings a definition may carry as equal (a bond `i–j` ≡ `j–i`; a valence
/// angle / linear bend with end atoms swapped; a dihedral read end-to-end either way).
fn same_coordinate(a: &Internal, b: &Internal) -> bool {
    match (*a, *b) {
        (Internal::Bond(i, j), Internal::Bond(p, q)) => (i, j) == (p, q) || (i, j) == (q, p),
        (Internal::Angle(i, k, j), Internal::Angle(p, m, q)) => {
            k == m && ((i, j) == (p, q) || (i, j) == (q, p))
        }
        (Internal::Dihedral(i, j, k, l), Internal::Dihedral(p, q, r, s)) => {
            (i, j, k, l) == (p, q, r, s) || (i, j, k, l) == (s, r, q, p)
        }
        (Internal::LinearBend(i, k, j, ax), Internal::LinearBend(p, m, q, ay)) => {
            k == m && ax == ay && ((i, j) == (p, q) || (i, j) == (q, p))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests;
