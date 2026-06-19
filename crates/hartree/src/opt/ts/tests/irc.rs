//! IRC integration: each integrator traces the reaction path off the converged
//! saddle into the two distinct basins and reaches the true minima.
//!
//! These run on the anharmonic double-well [`Anharmonic`] (a quartic maximum along
//! the reaction mode, harmonic minima transverse), whose minima are analytic: along
//! the reaction direction `q1` the energy is `−½a·q1² + ¼b·q1⁴`, with minima at
//! `q1 = ±√(a/b)` and energy `−a²/(4b)`.

use super::*;
use crate::core::{Atom, Element, Molecule};
use crate::opt::ts::{IrcMethod, TsOptions, TsStatus, find_transition_state};

const A: f64 = 0.5;
const B: f64 = 1.0;
const K2: f64 = 0.7;
const K3: f64 = 0.9;

/// The analytic minimum energy of the double well, `−a²/(4b)`.
fn well_min_energy() -> f64 {
    -A * A / (4.0 * B)
}

/// Run a full saddle search on the anharmonic double well (for the given atomic
/// numbers) then trace the IRC with `method`. Asserts the saddle converged and
/// returns the result.
fn double_well_irc(method: IrcMethod, z: &[u32; 3]) -> crate::opt::ts::TsResult {
    let x_ref = h3_positions();
    let basis = internal_basis(&x_ref);
    // A modest displacement off the saddle along the reaction mode and a transverse
    // mode, small enough that the search converges under the default controls.
    let mut start = x_ref.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.15 * basis[0][i] + 0.07 * basis[1][i] - 0.04 * basis[2][i];
        }
    }
    let mol = Molecule::new(
        z.iter()
            .zip(&start)
            .map(|(&zi, &p)| Atom::new(Element::from_z(zi).unwrap(), p))
            .collect(),
        0,
        if z.iter().all(|&zi| zi == 1) { 2 } else { 1 },
    );
    let mut surf = Anharmonic {
        x_ref,
        w: basis,
        a: A,
        b: B,
        k2: K2,
        k3: K3,
    };
    let mut opts = TsOptions::default();
    opts.confirm_irc = true;
    opts.irc_method = method;
    // A fresh Hessian every few steps keeps the curved-surface climb robust.
    opts.recalc_hessian = 5;
    let result = find_transition_state(&mol, &mut surf, &opts, None).unwrap();
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "saddle search status {:?} ({:?})",
        result.status,
        method
    );
    result
}

/// Signed projection of an endpoint displacement from the saddle onto the reaction
/// mode — the two endpoints must land on opposite sides.
fn projection(end: &[[f64; 3]], saddle: &[[f64; 3]], mode: &[[f64; 3]]) -> f64 {
    end.iter()
        .zip(saddle)
        .zip(mode)
        .map(|((e, s), m)| (0..3).map(|c| (e[c] - s[c]) * m[c]).sum::<f64>())
        .sum()
}

/// Shared assertions: both endpoints converged to the analytic minimum, below the
/// saddle, and on opposite sides of it along the reaction mode.
fn assert_reaches_both_minima(result: &crate::opt::ts::TsResult, method: IrcMethod) {
    let irc = result.irc.as_ref().expect("IRC requested");
    let e_min = well_min_energy();

    assert!(
        irc.forward_converged && irc.reverse_converged,
        "{method:?}: an endpoint did not converge (fwd {}, rev {}; steps {}/{})",
        irc.forward_converged,
        irc.reverse_converged,
        irc.forward_steps,
        irc.reverse_steps
    );
    assert!(
        (irc.forward_energy - e_min).abs() < 2e-3,
        "{method:?}: forward endpoint energy {:.6} not at the well minimum {e_min:.6}",
        irc.forward_energy
    );
    assert!(
        (irc.reverse_energy - e_min).abs() < 2e-3,
        "{method:?}: reverse endpoint energy {:.6} not at the well minimum {e_min:.6}",
        irc.reverse_energy
    );

    let mode = result
        .verification
        .as_ref()
        .unwrap()
        .reaction_mode
        .as_ref()
        .unwrap();
    let pf = projection(&irc.forward, &result.positions, mode);
    let pr = projection(&irc.reverse, &result.positions, mode);
    assert!(
        pf * pr < 0.0,
        "{method:?}: endpoints on the same side (fwd {pf}, rev {pr})"
    );
}

#[test]
fn dvv_reaches_double_well_minima() {
    let result = double_well_irc(IrcMethod::Dvv, &[1, 1, 1]);
    assert_reaches_both_minima(&result, IrcMethod::Dvv);
}

#[test]
fn gonzalez_schlegel_reaches_double_well_minima() {
    let result = double_well_irc(IrcMethod::GonzalezSchlegel, &[1, 1, 1]);
    assert_reaches_both_minima(&result, IrcMethod::GonzalezSchlegel);
}

#[test]
fn eulerpc_reaches_double_well_minima() {
    let result = double_well_irc(IrcMethod::EulerPc, &[1, 1, 1]);
    assert_reaches_both_minima(&result, IrcMethod::EulerPc);
}

/// End to end on a distinct-mass system (H, C, O): the saddle search, the
/// mass-weighted transition direction, and the integrator plumbing all run with
/// non-uniform masses and still relax into the two analytic minima. (The
/// mass-weighting *arithmetic* is pinned directly by
/// [`mw_transition_dir_reweights_a_coordinate_by_sqrt_mass`]; on this separable
/// surface the endpoint geometries themselves are mass-independent, so this case
/// exercises the multi-mass code path rather than isolating the weighting.)
#[test]
fn dvv_mass_weighted_path_reaches_minima_heteronuclear() {
    let result = double_well_irc(IrcMethod::Dvv, &[1, 6, 8]);
    assert_reaches_both_minima(&result, IrcMethod::Dvv);
}

/// The mass-weighting arithmetic itself: a coordinate displacement transforms as
/// `√m · Δx`, so mass-weighting a Cartesian mode multiplies each atom's component by
/// `√m` and renormalizes. A mass-blind integrator (treating `√m = 1`) would give a
/// different direction — this is the assertion the separable double-well cannot make.
#[test]
fn mw_transition_dir_reweights_a_coordinate_by_sqrt_mass() {
    use crate::opt::ts::irc::mw_transition_dir;
    // Equal Cartesian displacement on an H and an O atom along x.
    let mode = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let m_h = Element::from_z(1).unwrap().mass();
    let m_o = Element::from_z(8).unwrap().mass();
    let dir = mw_transition_dir(&mode, &[m_h, m_o]);

    // O's component grows relative to H's by exactly √(m_O/m_H)...
    let ratio = dir[3] / dir[0]; // O_x / H_x
    assert!(
        (ratio - (m_o / m_h).sqrt()).abs() < 1e-12,
        "O/H component ratio {ratio} != sqrt(m_O/m_H) {}",
        (m_o / m_h).sqrt()
    );
    // ...the result is a unit vector...
    let norm: f64 = dir.iter().map(|c| c * c).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-12, "not normalized: {norm}");
    // ...and the reweighting is non-trivial (a mass-blind version would give ratio 1).
    assert!((ratio - 1.0).abs() > 1.0, "mass weighting had no effect");
}

/// Backward compatibility: an [`IrcEndpoints`](crate::opt::ts::IrcEndpoints) record
/// serialized before the convergence/step fields existed (only the four
/// geometry/energy keys) still deserializes, defaulting the new fields.
#[test]
fn irc_endpoints_round_trip_defaults_new_fields() {
    let legacy = r#"{"forward":[[0.0,0.0,0.0]],"forward_energy":-1.5,
                     "reverse":[[0.0,0.0,0.0]],"reverse_energy":-2.5}"#;
    let ep: crate::opt::ts::IrcEndpoints = serde_json::from_str(legacy).unwrap();
    assert_eq!(ep.forward_energy, -1.5);
    assert_eq!(ep.reverse_energy, -2.5);
    assert!(!ep.forward_converged && !ep.reverse_converged);
    assert_eq!(ep.forward_steps, 0);
    assert_eq!(ep.reverse_steps, 0);
}

/// A soft reaction mode: the force a short step off the saddle is already below
/// `irc_gtol`, so a bare force test would accept the seed itself — sitting essentially
/// on the saddle — as a converged minimum. The basin guard requires an endpoint to
/// descend a clear margin below the saddle before it counts as converged, so the trace
/// moves off the ridge into a basin. (With a soft mode the looser `irc_gtol` still
/// halts the trace before the exact well bottom; the guarantee under test is that the
/// endpoint is a genuine basin point below the saddle, not the near-saddle seed.)
#[test]
fn soft_mode_irc_descends_into_the_basin_not_the_seed() {
    let x_ref = h3_positions();
    let basis = internal_basis(&x_ref);
    // Soft reaction-mode curvature `a`: the seed (0.1 off the ridge) has force ~a·0.1
    // below irc_gtol yet sits only ~7e-5 below the saddle — short of a basin, so without
    // the guard the trace would converge right there.
    let a = 0.015;
    let b = 0.0306;
    let mut start = x_ref.clone();
    for at in 0..3 {
        for c in 0..3 {
            let i = 3 * at + c;
            start[at][c] += 0.10 * basis[0][i] + 0.05 * basis[1][i];
        }
    }
    let mut surf = Anharmonic {
        x_ref: x_ref.clone(),
        w: basis,
        a,
        b,
        k2: 0.7,
        k3: 0.9,
    };
    let mut opts = TsOptions::default();
    opts.confirm_irc = true;
    opts.recalc_hessian = 5;
    let result = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(result.status, TsStatus::Converged);
    let irc = result.irc.expect("IRC requested");

    assert!(
        irc.forward_converged && irc.reverse_converged,
        "endpoints not converged (fwd {} rev {})",
        irc.forward_converged,
        irc.reverse_converged
    );
    // Both endpoints descended a clear margin below the saddle into a basin — not the
    // ~7e-5-below-saddle seed a bare force test would have accepted.
    assert!(
        irc.forward_energy < result.energy - 1.0e-4 && irc.reverse_energy < result.energy - 1.0e-4,
        "endpoints did not descend below the saddle {:.6}: fwd {:.6} rev {:.6}",
        result.energy,
        irc.forward_energy,
        irc.reverse_energy
    );
    // ...and on opposite sides of it along the reaction mode.
    let mode = result
        .verification
        .as_ref()
        .unwrap()
        .reaction_mode
        .as_ref()
        .unwrap();
    let pf = projection(&irc.forward, &result.positions, mode);
    let pr = projection(&irc.reverse, &result.positions, mode);
    assert!(
        pf * pr < 0.0,
        "endpoints on the same side (fwd {pf}, rev {pr})"
    );
}

fn flat(x: &[[f64; 3]]) -> Vec<f64> {
    x.iter().flat_map(|a| a.iter().copied()).collect()
}
fn dotp(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn unitv(v: &[f64]) -> Vec<f64> {
    let n = dotp(v, v).sqrt();
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

/// A surface whose (unit-mass, trans/rot-clean) gradient is everywhere tangent to the
/// Gonzalez–Schlegel hypersphere — perpendicular to the radius from the pivot. The
/// constrained micro-iteration can therefore never null its perpendicular component, so
/// it runs its whole budget without converging: a deterministic *silent stall*. The
/// gradient is the in-plane 90° rotation of the outward radius `n` (`ghat → t`,
/// `t → −ghat`), which is `⟂ n` at every point on the sphere.
///
/// `gs2_step` re-projects trans/rot at the *displaced* probe geometry (not `x0`), which
/// leaks a few percent of this gradient out of the `{ghat, t}` plane; that is harmless
/// here because the tangential component stays `‖g⊥‖ ≈ 1`, orders of magnitude above
/// `GS2_TOL`, so the stall premise holds with wide margin.
struct TangentialStall {
    x0: Vec<[f64; 3]>,
    ghat: Vec<f64>, // mass-weighted reaction direction (unit, internal)
    t: Vec<f64>,    // a transverse unit direction (internal, ⟂ ghat)
    r: f64,         // hypersphere radius = step/2
}
impl Surface for TangentialStall {
    fn energy(&mut self, _x: &[[f64; 3]]) -> Result<f64, OptError> {
        Ok(0.0)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        // s = displacement from the saddle (unit masses ⇒ mass-weighted == Cartesian).
        let s: Vec<f64> = flat(x)
            .iter()
            .zip(flat(&self.x0))
            .map(|(a, b)| a - b)
            .collect();
        // Outward radius of the sphere centred at the pivot (−r·ghat).
        let from_pivot: Vec<f64> = s
            .iter()
            .zip(&self.ghat)
            .map(|(si, gi)| si + self.r * gi)
            .collect();
        let n = unitv(&from_pivot);
        let a = dotp(&n, &self.ghat);
        let b = dotp(&n, &self.t);
        let g: Vec<f64> = (0..self.ghat.len())
            .map(|i| a * self.t[i] - b * self.ghat[i])
            .collect();
        Some(Ok((0..x.len())
            .map(|k| [g[3 * k], g[3 * k + 1], g[3 * k + 2]])
            .collect()))
    }
}

/// A Gonzalez–Schlegel micro-iteration that cannot converge (its perpendicular gradient
/// is unkillable) must not silently emit the last half-rotated, on-sphere point: it
/// falls back to the steepest-descent step — exactly `−step·ĝ`, a descent direction of
/// the requested length with no transverse leak.
#[test]
fn gs2_step_falls_back_to_steepest_descent_on_a_stall() {
    use crate::opt::ts::irc::gs2_step;
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let ghat = basis[0].clone();
    let t = basis[1].clone();
    let masses = vec![1.0; x0.len()];
    let step = 0.1;
    let mut surf = TangentialStall {
        x0: x0.clone(),
        ghat: ghat.clone(),
        t: t.clone(),
        r: 0.5 * step,
    };
    let opts = TsOptions::default();
    // g_mw = ghat (unit) ⇒ the search direction is ghat and the steepest-descent
    // fallback is exactly −step·ghat.
    let s = gs2_step(&mut surf, &x0, &ghat, &masses, step, &opts).unwrap();

    // The returned step is the steepest-descent step: −step along ghat with no
    // transverse (t) component — i.e. NOT a rotated on-sphere point (which carries a
    // large t component).
    assert!(
        (dotp(&s, &ghat) + step).abs() < 1e-9,
        "ghat component {} != -step {}",
        dotp(&s, &ghat),
        -step
    );
    assert!(
        dotp(&s, &t).abs() < 1e-9,
        "unexpected transverse component {}",
        dotp(&s, &t)
    );
    // ...and it is a genuine descent step.
    assert!(dotp(&s, &ghat) < 0.0, "step is not a descent direction");
}

/// A converged endpoint records the steps it took (a nonzero, bounded count) — the
/// per-endpoint diagnostics the result now carries.
#[test]
fn endpoint_step_counts_are_recorded() {
    let result = double_well_irc(IrcMethod::Dvv, &[1, 1, 1]);
    let irc = result.irc.as_ref().unwrap();
    let opts = TsOptions::default();
    for (label, steps, converged) in [
        ("forward", irc.forward_steps, irc.forward_converged),
        ("reverse", irc.reverse_steps, irc.reverse_converged),
    ] {
        assert!(steps > 0, "{label}: no integration steps recorded");
        assert!(
            steps <= opts.irc_max_steps,
            "{label}: steps {steps} exceeds the cap"
        );
        // A converged endpoint stopped before the cap.
        assert!(
            !converged || steps < opts.irc_max_steps,
            "{label}: converged yet ran to the step cap"
        );
    }
}
