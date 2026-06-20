//! Double-ended climbing-image NEB on a real self-consistent-field surface, then the
//! production hand-off to the local P-RFO refiner.
//!
//! This exercises the full NEB-TS workflow end to end on HCN ⇌ HNC (RHF/STO-3G): a
//! band between the two isomer basins relaxes toward the minimum-energy path, the
//! climbing image rides up to an approximate saddle, and its geometry + reaction-
//! coordinate tangent seed a P-RFO search that converges the tight transition state.
//! The refined saddle is pinned against the same baseline as `tests/ts_reference.rs`
//! (energy ≈ −91.56485 Ha, one imaginary mode ≈ −1248.5 cm⁻¹), so the NEB peak only
//! has to land the refiner in the right basin — the converged saddle is the
//! invariant. Compute-heavy: run with `--release`.

use hartree::HfSurface;
use hartree::core::{Atom, Element, Molecule};
use hartree::opt::ts::{
    NebOptions, TsOptions, TsStatus, find_minimum_energy_path, find_transition_state,
};
use hartree::scf::Reference;

fn mol(atoms: &[(u32, [f64; 3])], charge: i32, multiplicity: u32) -> Molecule {
    Molecule::new(
        atoms
            .iter()
            .map(|&(z, p)| Atom::new(Element::from_z(z).unwrap(), p))
            .collect(),
        charge,
        multiplicity,
    )
}

fn bond(p: &[[f64; 3]], i: usize, j: usize) -> f64 {
    let (a, b) = (p[i], p[j]);
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The SCF settings the saddle path applies (small-gap TS geometries need them).
fn prepare(surface: &mut HfSurface) {
    surface.set_scf_max_iter(400);
    surface.set_scf_level_shift(0.3);
}

/// Atom order is `[C, N, H]` in both endpoints. The hydrogen sits a little off the
/// C–N axis (same side in both) so the band bows through the bridging region rather
/// than dragging H straight through the C≡N bond — the path swaps which heavy atom H
/// is bound to. These bent endpoints are basin representatives, not exact minima; the
/// converged saddle (below) is what is pinned.
fn hcn_basin() -> Molecule {
    mol(
        &[
            (6, [0.0, 0.0, 0.0]),  // C
            (7, [2.19, 0.0, 0.0]), // N
            (1, [-1.5, 1.2, 0.0]), // H, bound to C
        ],
        0,
        1,
    )
}

fn hnc_basin() -> Molecule {
    mol(
        &[
            (6, [0.0, 0.0, 0.0]),  // C
            (7, [2.30, 0.0, 0.0]), // N
            (1, [3.7, 1.2, 0.0]),  // H, bound to N
        ],
        0,
        1,
    )
}

#[test]
fn hcn_hnc_neb_then_prfo_rhf_sto3g() {
    let reactant = hcn_basin();
    let product = hnc_basin();

    // --- Double-ended CI-NEB to get into the saddle basin. ---
    let mut neb_surface = HfSurface::new(&reactant, "sto-3g", Reference::Rhf).unwrap();
    prepare(&mut neb_surface);

    let mut neb_opts = NebOptions::default();
    neb_opts.n_images = 6;
    neb_opts.gtol = 5e-3; // loose: the band only has to seed the refiner
    neb_opts.max_iter = 200;
    neb_opts.climb_after = 10;
    neb_opts.spring_k = 0.05;

    let neb = find_minimum_energy_path(&reactant, &product, &mut neb_surface, &neb_opts, None)
        .expect("NEB surface evaluation");

    // The band made progress (the max NEB force fell over the relaxation).
    let first = neb.history.first().expect("history").max_force;
    let last = neb.history.last().expect("history").max_force;
    assert!(
        last < first,
        "NEB force did not decrease (first {first:.4}, last {last:.4})"
    );

    // The climbing image is a genuine bridging geometry: H roughly equidistant from
    // C and N and well off the C–N axis, with a positive forward barrier.
    let peak = &neb.peak_geometry;
    let r_ch = bond(peak, 0, 2);
    let r_nh = bond(peak, 1, 2);
    assert!(
        r_ch > 1.8 && r_ch < 3.2 && r_nh > 1.8 && r_nh < 3.2,
        "peak is not bridging: r(C–H)={r_ch:.3}, r(N–H)={r_nh:.3} Bohr"
    );
    assert!(
        peak[2][1].abs() > 0.5,
        "peak H is on the C–N axis (y={:.3})",
        peak[2][1]
    );
    assert!(
        neb.barrier > 0.0,
        "non-positive barrier {:.5} Ha",
        neb.barrier
    );

    // --- Hand the climbing image + tangent to the local refiner (the NEB-TS pattern). ---
    let guess = mol(&[(6, peak[0]), (7, peak[1]), (1, peak[2])], 0, 1);
    let mut ts_surface = HfSurface::new(&guess, "sto-3g", Reference::Rhf).unwrap();
    prepare(&mut ts_surface);

    let mut ts_opts = TsOptions::default();
    ts_opts.reaction_mode_seed = Some(neb.peak_tangent.clone());

    let ts = find_transition_state(&guess, &mut ts_surface, &ts_opts, None)
        .expect("P-RFO surface evaluation");

    assert_eq!(
        ts.status,
        TsStatus::Converged,
        "P-RFO did not converge from the NEB peak"
    );
    let v = ts.verification.as_ref().expect("verification present");
    assert!(
        v.is_first_order_saddle(),
        "expected exactly one negative mode, got {:?}",
        v.negative_eigenvalues
    );
    let imag = v.imaginary_frequency_cm1.expect("imaginary frequency");

    // Pinned RHF/STO-3G saddle baselines, shared with tests/ts_reference.rs.
    assert!(
        (ts.energy - (-91.5648510302)).abs() < 1e-4,
        "saddle energy {:.8} Ha drifted from baseline",
        ts.energy
    );
    let r_cn = bond(&ts.positions, 0, 1);
    let r_ch = bond(&ts.positions, 0, 2);
    let r_nh = bond(&ts.positions, 1, 2);
    assert!((r_cn - 2.3080).abs() < 3e-2, "r(C–N) = {r_cn:.4} Bohr");
    assert!((r_ch - 2.2714).abs() < 3e-2, "r(C–H) = {r_ch:.4} Bohr");
    assert!((r_nh - 2.7168).abs() < 3e-2, "r(N–H) = {r_nh:.4} Bohr");
    assert!(
        (imag - (-1248.5)).abs() < 40.0,
        "imaginary frequency {imag:.1} cm⁻¹ drifted from baseline"
    );
}
