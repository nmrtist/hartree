use super::*;
use crate::opt::Surface;
use crate::opt::ts::numerics::{
    MwSpectrum, add_step, column, dot, mass_weight_grad, masses_of, mw_projected_hessian,
    non_null_modes, norm, projected_force_norms, unmass_weight_step,
};
use crate::opt::ts::step::{prfo_step, select_followed};

/// How the trust-radius cap is applied to the unrestricted partitioned-RFO step.
#[derive(Clone, Copy)]
enum Cap {
    /// The production restricted step: scale the augmented RFO problem so the step
    /// norm meets the trust radius while preserving the rational-model direction.
    Restricted,
    /// The historical hard clip: take the full RFO step (here via an unbounded trust)
    /// and uniformly rescale it down to the trust radius.
    Clip,
}

/// One capped mass-weighted P-RFO step under either trust strategy, both built from
/// the same spectrum and gradient so the comparison isolates the cap alone.
fn capped_step(
    cap: Cap,
    spec: &MwSpectrum,
    g_mw: &[f64],
    non_null: &[usize],
    followed: usize,
    trust: f64,
) -> Vec<f64> {
    match cap {
        Cap::Restricted => prfo_step(spec, g_mw, non_null, followed, trust),
        Cap::Clip => {
            let mut s = prfo_step(spec, g_mw, non_null, followed, f64::INFINITY);
            let n = norm(&s);
            if n > trust {
                let scale = trust / n;
                for v in &mut s {
                    *v *= scale;
                }
            }
            s
        }
    }
}

/// Fixed trust radius, projected-force convergence threshold, and iteration cap for
/// the comparison climbs. The threshold is the production `max_force` default (1e-4);
/// the surface below is conditioned so the very soft transverse mode's force stays
/// below it throughout, so that mode never gates convergence and the iteration count
/// reflects how each cap drives the force-carrying reaction and stiff modes. The cap
/// is generous so neither strategy is cut off prematurely.
const TRUST: f64 = 0.2;
const GTOL: f64 = 1.0e-4;
const MAX_ITER: usize = 2000;

/// Drive a fixed-trust P-RFO climb on an exact quadratic surface (constant analytic
/// Hessian `h`, so no Bofill maintenance is needed) from `start` and return the
/// iteration count to reach [`GTOL`], or [`MAX_ITER`] if it never does. Every step
/// reuses the production spectrum, mode-selection, and step-assembly helpers; the only
/// thing that varies between runs is `cap`.
fn climb_iterations(x0: &[[f64; 3]], h: &[f64], start: &[[f64; 3]], cap: Cap) -> usize {
    let mut surf = Quadratic {
        x0: x0.to_vec(),
        h: h.to_vec(),
    };
    let masses = masses_of(&h3_molecule(x0));
    let mut x = start.to_vec();
    for iter in 1..=MAX_ITER {
        let g = surf.analytic_gradient(&x).unwrap().unwrap();
        let (max_force, _) = projected_force_norms(&g, &masses, &x);
        if max_force < GTOL {
            return iter;
        }
        let spec = mw_projected_hessian(&x, &masses, h).unwrap();
        let non_null = non_null_modes(&spec);
        let followed = select_followed(&spec, &non_null, 0, &None, None);
        let g_mw = mass_weight_grad(&g, &masses);
        let dxi = capped_step(cap, &spec, &g_mw, &non_null, followed, TRUST);
        let dx = unmass_weight_step(&dxi, &masses);
        x = add_step(&x, &dx);
    }
    MAX_ITER
}

/// A strongly ill-conditioned (≈1:10000) quadratic saddle: a reaction mode and a stiff
/// transverse mode of curvature ±0.5, plus a very soft transverse mode (5e-5). The soft
/// mode's full Newton step is huge (≈ its displacement), so the unrestricted RFO step
/// badly overshoots the trust radius — the regime where a uniform clip throttles the
/// well-conditioned reaction/stiff modes to rein in the one soft direction. Its tiny
/// curvature also keeps the soft mode's force below the convergence threshold, so it
/// never gates: the comparison turns on how each cap converges the reaction and stiff
/// modes, which is where the restricted step's reshaping pays off.
fn ill_conditioned() -> (Vec<[f64; 3]>, Vec<Vec<f64>>, Vec<f64>) {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.5, 5.0e-5, 0.5]);
    (x0, basis, h)
}

#[test]
fn restricted_step_needs_fewer_iterations_than_hard_clip() {
    let (x0, basis, h) = ill_conditioned();
    // Displace along all three internal modes: a moderate climb along the reaction
    // mode and the stiff transverse mode, and a large excursion along the soft mode
    // (whose tiny curvature keeps its force sub-threshold throughout, so it inflates
    // the step without gating convergence).
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.5 * basis[0][i] + 1.5 * basis[1][i] + 0.4 * basis[2][i];
        }
    }
    let clip_iters = climb_iterations(&x0, &h, &start, Cap::Clip);
    let rs_iters = climb_iterations(&x0, &h, &start, Cap::Restricted);

    assert!(clip_iters < MAX_ITER, "clip path did not converge");
    assert!(rs_iters < MAX_ITER, "restricted path did not converge");
    // At least ~20% fewer iterations for the restricted step.
    assert!(
        rs_iters * 5 <= clip_iters * 4,
        "restricted step ({rs_iters}) not ≥20% fewer than hard clip ({clip_iters})"
    );
}

#[test]
fn restricted_step_meets_trust_without_clip_distortion() {
    let (x0, basis, h) = ill_conditioned();
    // A point where the unrestricted RFO step overshoots the trust radius (driven by
    // the large soft-mode component), so the restricted scaling actually engages.
    let mut x = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            x[a][c] += 0.5 * basis[0][i] + 1.5 * basis[1][i] + 0.4 * basis[2][i];
        }
    }
    let masses = masses_of(&h3_molecule(&x0));
    let trust = TRUST;

    let mut surf = Quadratic { x0: x0.clone(), h };
    let g = surf.analytic_gradient(&x).unwrap().unwrap();
    let spec = mw_projected_hessian(&x, &masses, &surf.h).unwrap();
    let non_null = non_null_modes(&spec);
    let followed = select_followed(&spec, &non_null, 0, &None, None);
    let g_mw = mass_weight_grad(&g, &masses);

    let uncapped = prfo_step(&spec, &g_mw, &non_null, followed, f64::INFINITY);
    let rs = capped_step(Cap::Restricted, &spec, &g_mw, &non_null, followed, trust);
    let clip = capped_step(Cap::Clip, &spec, &g_mw, &non_null, followed, trust);

    // Precondition: the unrestricted step really does overshoot, so this exercises the
    // restricted branch rather than passing the step through untouched.
    assert!(
        norm(&uncapped) > 1.5 * trust,
        "unrestricted step ({}) should exceed the trust radius for this test",
        norm(&uncapped)
    );
    // The restricted step lands on the trust radius to within ~1% — unlike the clip,
    // which hits it exactly only by construction.
    assert!(
        (norm(&rs) - trust).abs() <= 0.01 * trust,
        "restricted step norm {} not within 1% of trust {trust}",
        norm(&rs)
    );
    // And it reaches the trust radius by *reshaping* the step, not by uniformly
    // scaling it. Measure each mode's surviving fraction of the unrestricted RFO step
    // (its component along that eigenvector, relative to the unrestricted one). The
    // clip multiplies every mode by the same factor `trust/‖uncapped‖`, so every mode's
    // surviving fraction is identical; the restricted step instead damps the soft
    // transverse mode *harder* than that factor and the stiff transverse mode *softer*,
    // preserving travel along the well-conditioned direction at the soft mode's expense.
    // (That per-mode reshaping is what wins iterations; see the iteration-count test.)
    let ndof = spec.eigenvalues.len();
    let minimized: Vec<usize> = non_null
        .iter()
        .copied()
        .filter(|&k| k != followed)
        .collect();
    let by_abs_eig = |k: &usize| spec.eigenvalues[*k].abs();
    let soft = *minimized
        .iter()
        .min_by(|a, b| by_abs_eig(a).partial_cmp(&by_abs_eig(b)).unwrap())
        .unwrap();
    let stiff = *minimized
        .iter()
        .max_by(|a, b| by_abs_eig(a).partial_cmp(&by_abs_eig(b)).unwrap())
        .unwrap();
    let surviving = |step: &[f64], k: usize| {
        let c = column(&spec.eigenvectors, ndof, k);
        dot(step, &c).abs() / dot(&uncapped, &c).abs()
    };
    let uniform = trust / norm(&uncapped);
    // The clip really is uniform: every mode keeps the same fraction of its component.
    assert!(
        (surviving(&clip, soft) - uniform).abs() < 1e-9
            && (surviving(&clip, stiff) - uniform).abs() < 1e-9,
        "clip is not uniform across modes (soft {}, stiff {}, factor {uniform})",
        surviving(&clip, soft),
        surviving(&clip, stiff)
    );
    // The restricted step reshapes: it damps the soft mode below the uniform factor and
    // keeps the stiff mode above it.
    assert!(
        surviving(&rs, soft) < uniform && surviving(&rs, stiff) > uniform,
        "restricted step did not reshape across stiffness (soft {}, uniform {uniform}, stiff {})",
        surviving(&rs, soft),
        surviving(&rs, stiff)
    );
}
