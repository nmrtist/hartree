//! Climbing-image NEB driver tests on analytic surfaces.
//!
//! The [`Anharmonic`](super::Anharmonic) fixture is a quartic double well along one
//! internal direction `w0` (minima at `q1 = ±√(a/b)`, a maximum — the saddle — at
//! `q1 = 0`) and harmonic along the others, so its minimum-energy path is the straight
//! `q1` line through the origin. A NEB between the two wells should trace that path and
//! land its climbing image on the origin saddle, with a barrier equal to the analytic
//! well depth `a²/4b`. These run on a closed-form surface (fast, deterministic); the
//! real-SCF end-to-end check lives in `tests/neb_reference.rs`.

use std::cell::Cell;

use super::{Anharmonic, h3_molecule, h3_positions, internal_basis, mode_overlap};
use crate::core::{Atom, Element, Molecule};
use crate::opt::ts::{
    Flow, NebOptions, NebStatus, Progress, TsOptions, TsStatus, find_minimum_energy_path,
    find_transition_state_from_endpoints,
};
use crate::opt::{OptError, OptStep, Surface};

/// Build the double-well surface and its two endpoint minima from the shared H3
/// geometry: the saddle sits at `x_ref`, the minima at `x_ref ± √(a/b)·w0`.
fn double_well() -> (Anharmonic, Molecule, Molecule, Vec<f64>) {
    let x_ref = h3_positions();
    let basis = internal_basis(&x_ref);
    let (a, b, k2, k3): (f64, f64, f64, f64) = (0.4, 0.4, 0.6, 0.6);
    let q = (a / b).sqrt();
    let w0 = basis[0].clone();
    let displace = |sign: f64| -> Vec<[f64; 3]> {
        (0..x_ref.len())
            .map(|atom| {
                [
                    x_ref[atom][0] + sign * q * w0[3 * atom],
                    x_ref[atom][1] + sign * q * w0[3 * atom + 1],
                    x_ref[atom][2] + sign * q * w0[3 * atom + 2],
                ]
            })
            .collect()
    };
    let reactant = h3_molecule(&displace(1.0));
    let product = h3_molecule(&displace(-1.0));
    let surface = Anharmonic {
        x_ref,
        w: basis,
        a,
        b,
        k2,
        k3,
    };
    (surface, reactant, product, w0)
}

/// Max per-coordinate deviation of `pos` from the (saddle) `x_ref`.
fn max_dev(pos: &[[f64; 3]], x_ref: &[[f64; 3]]) -> f64 {
    let mut m = 0.0f64;
    for (p, r) in pos.iter().zip(x_ref) {
        for k in 0..3 {
            m = m.max((p[k] - r[k]).abs());
        }
    }
    m
}

#[test]
fn ci_neb_finds_the_double_well_saddle() {
    let (mut surface, reactant, product, w0) = double_well();
    let mut opts = NebOptions::default();
    opts.n_images = 7;
    opts.max_iter = 600;
    opts.spring_k = 0.1;

    let result = find_minimum_energy_path(&reactant, &product, &mut surface, &opts, None)
        .expect("surface evaluation");

    assert_eq!(
        result.status,
        NebStatus::Converged,
        "CI-NEB did not converge ({} iters, max force trace tail {:?})",
        result.iterations,
        result.history.last().map(|s| s.max_force),
    );

    // The climbing image is an interior image and the band brackets it with the two
    // fixed endpoints.
    assert_eq!(result.images.len(), opts.n_images + 2);
    assert!(result.climbing_image >= 1 && result.climbing_image <= opts.n_images);

    // The climbing image sits on the origin saddle (x_ref).
    let x_ref = h3_positions();
    let dev = max_dev(&result.peak_geometry, &x_ref);
    assert!(dev < 0.02, "climbing image off the saddle by {dev:.4} Bohr");

    // Barrier equals the analytic well depth a²/4b = 0.1 Ha.
    assert!(
        (result.barrier - 0.1).abs() < 0.01,
        "barrier {:.5} Ha drifted from analytic 0.1",
        result.barrier
    );

    // The peak tangent is the reaction coordinate (the w0 double-well direction).
    let overlap = mode_overlap(&result.peak_tangent, &w0);
    assert!(overlap > 0.9, "peak tangent overlap with w0 = {overlap:.3}");

    // The result is serializable (the agent-facing contract).
    let json = serde_json::to_string(&result).expect("serialize NebResult");
    let back: crate::opt::ts::NebResult = serde_json::from_str(&json).expect("round-trip");
    assert_eq!(back.images.len(), result.images.len());
}

#[test]
fn neb_ts_pipeline_refines_the_climbing_image_to_the_saddle() {
    // The end-to-end convenience entry: relax the band, then let the local refiner
    // converge the tight saddle from the climbing image + reaction-coordinate tangent.
    // One surface drives both stages. (The real-SCF counterpart is the HCN↔HNC band in
    // tests/neb_reference.rs; this is the fast, deterministic analytic check.)
    let (mut surface, reactant, product, w0) = double_well();
    let mut neb_opts = NebOptions::default();
    neb_opts.n_images = 7;
    neb_opts.max_iter = 600;
    neb_opts.spring_k = 0.1;

    let out = find_transition_state_from_endpoints(
        &reactant,
        &product,
        &mut surface,
        &neb_opts,
        &TsOptions::default(),
        None,
    )
    .expect("surface evaluation");

    // The band got into the saddle basin and the refiner converged a first-order saddle.
    assert_eq!(
        out.neb.status,
        NebStatus::Converged,
        "band did not converge"
    );
    assert_eq!(
        out.transition_state.status,
        TsStatus::Converged,
        "refiner did not converge from the NEB peak"
    );
    let v = out
        .transition_state
        .verification
        .as_ref()
        .expect("verification present");
    assert!(
        v.is_first_order_saddle(),
        "expected one imaginary mode, got {:?}",
        v.negative_eigenvalues
    );

    // The refined saddle sits on the origin (x_ref), tighter than the on-band climbing
    // image the NEB alone produced.
    let x_ref = h3_positions();
    let refined_dev = max_dev(&out.transition_state.positions, &x_ref);
    let band_dev = max_dev(&out.neb.peak_geometry, &x_ref);
    assert!(
        refined_dev < 5e-3,
        "refined saddle off x_ref by {refined_dev:.5} Bohr"
    );
    assert!(
        refined_dev <= band_dev + 1e-9,
        "refinement did not tighten the guess (band {band_dev:.5}, refined {refined_dev:.5})"
    );

    // The reaction coordinate the helper seeded is the double-well direction w0.
    let overlap = mode_overlap(&out.neb.peak_tangent, &w0);
    assert!(overlap > 0.9, "seed tangent overlap with w0 = {overlap:.3}");

    // The combined result round-trips through serde (the agent-facing contract).
    let json = serde_json::to_string(&out).expect("serialize NebTsResult");
    let back: crate::opt::ts::NebTsResult = serde_json::from_str(&json).expect("round-trip");
    assert_eq!(back.transition_state.status, TsStatus::Converged);
}

#[test]
fn plain_neb_relaxes_without_climbing() {
    let (mut surface, reactant, product, _w0) = double_well();
    let mut opts = NebOptions::default();
    opts.n_images = 7;
    opts.climbing = false;
    opts.max_iter = 600;

    let result = find_minimum_energy_path(&reactant, &product, &mut surface, &opts, None)
        .expect("surface evaluation");
    assert_eq!(
        result.status,
        NebStatus::Converged,
        "plain NEB did not converge"
    );

    // There is a real barrier: the peak image lies above both endpoints.
    let peak_e = result.energies[result.climbing_image];
    assert!(
        peak_e > result.energies[0] + 0.02 && peak_e > result.energies[opts.n_images + 1] + 0.02,
        "no barrier on the relaxed band (peak {peak_e:.4})"
    );
    // Band energies are sane (every image at least as deep as the saddle top).
    assert!(result.energies.iter().all(|&e| e <= peak_e + 1e-9));
}

#[test]
fn history_records_the_actual_step_displacement() {
    // The displacement trace must reflect the geometry change actually applied each
    // iteration: the first step has no predecessor (zero), but a moving band must then
    // record nonzero displacements (it is measured against the pre-step geometry, not
    // a copy of the current one).
    let (mut surface, reactant, product, _w0) = double_well();
    let mut opts = NebOptions::default();
    opts.n_images = 5;
    opts.max_iter = 30;

    let result = find_minimum_energy_path(&reactant, &product, &mut surface, &opts, None)
        .expect("surface evaluation");
    assert_eq!(
        result.history[0].max_disp, 0.0,
        "first step has no predecessor"
    );
    assert!(
        result.history.iter().skip(1).any(|s| s.max_disp > 1e-9),
        "displacement trace is flat while the band is moving"
    );
}

#[test]
fn observer_can_stop_the_search_early() {
    let (mut surface, reactant, product, _w0) = double_well();
    let opts = NebOptions::default();

    // Stop after the second observed iteration.
    struct StopAfter(Cell<usize>);
    impl Progress for StopAfter {
        fn step(&self, _s: &OptStep) -> Flow {
            self.0.set(self.0.get() + 1);
            if self.0.get() >= 2 {
                Flow::Stop
            } else {
                Flow::Continue
            }
        }
    }
    let obs = StopAfter(Cell::new(0));
    let result = find_minimum_energy_path(&reactant, &product, &mut surface, &opts, Some(&obs))
        .expect("surface evaluation");
    assert_eq!(result.status, NebStatus::StoppedEarly);
    assert_eq!(result.iterations, 2);
}

/// A surface that never moves the band — only used to reach the endpoint validation,
/// which runs before any evaluation.
struct Flat;
impl Surface for Flat {
    fn energy(&mut self, _x: &[[f64; 3]]) -> Result<f64, OptError> {
        Ok(0.0)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        Some(Ok(vec![[0.0; 3]; x.len()]))
    }
}

#[test]
fn rejects_bad_endpoints() {
    use crate::opt::ts::NebError;
    let mut surface = Flat;
    let opts = NebOptions::default();

    let three_h = h3_molecule(&h3_positions());

    // Mismatched atom count.
    let two = Molecule::new(
        vec![
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [1.5, 0.0, 0.0]),
        ],
        0,
        1,
    );
    let err = find_minimum_energy_path(&three_h, &two, &mut surface, &opts, None).unwrap_err();
    assert!(matches!(err, NebError::BadEndpoints(_)), "got {err:?}");

    // Same count, different element ordering at index 1.
    let x = h3_positions();
    let mixed = Molecule::new(
        vec![
            Atom::new(Element::from_z(1).unwrap(), x[0]),
            Atom::new(Element::from_z(6).unwrap(), x[1]),
            Atom::new(Element::from_z(1).unwrap(), x[2]),
        ],
        0,
        1,
    );
    let err = find_minimum_energy_path(&three_h, &mixed, &mut surface, &opts, None).unwrap_err();
    assert!(matches!(err, NebError::BadEndpoints(_)), "got {err:?}");

    // Zero interior images.
    let mut zero = NebOptions::default();
    zero.n_images = 0;
    let err = find_minimum_energy_path(&three_h, &three_h, &mut surface, &zero, None).unwrap_err();
    assert!(matches!(err, NebError::BadEndpoints(_)), "got {err:?}");

    // Zero iterations (a zero-iteration band would carry unevaluated interior energies).
    let mut no_iters = NebOptions::default();
    no_iters.max_iter = 0;
    let err =
        find_minimum_energy_path(&three_h, &three_h, &mut surface, &no_iters, None).unwrap_err();
    assert!(matches!(err, NebError::BadEndpoints(_)), "got {err:?}");

    // Fewer than two atoms.
    let one = Molecule::new(
        vec![Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0])],
        0,
        2,
    );
    let err = find_minimum_energy_path(&one, &one, &mut surface, &opts, None).unwrap_err();
    assert!(matches!(err, NebError::BadEndpoints(_)), "got {err:?}");
}

#[test]
fn options_serde_round_trip_and_defaults() {
    // Full round-trip.
    let opts = NebOptions::default();
    let json = serde_json::to_string(&opts).unwrap();
    let back: NebOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(back.n_images, opts.n_images);
    assert_eq!(back.spring_k, opts.spring_k);
    assert_eq!(back.fire_alpha_start, opts.fire_alpha_start);

    // A partial record fills every missing field from the default (container
    // `#[serde(default)]`), so options serialized before a field existed still load.
    let partial: NebOptions = serde_json::from_str(r#"{"n_images": 4}"#).unwrap();
    assert_eq!(partial.n_images, 4);
    assert_eq!(partial.spring_k, NebOptions::default().spring_k);
    assert!(partial.climbing);
}
