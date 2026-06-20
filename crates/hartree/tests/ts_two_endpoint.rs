//! Two-endpoint transition-state search through the public `Job` API.
//!
//! Exercises `JobOptions::ts_guess`: the job takes a reactant (`Job::molecule`) and a
//! product, builds a near-saddle guess between them — a single IDPP guess, or a
//! climbing-image NEB band — seeds the reaction coordinate, and refines the saddle. The
//! converged HCN ⇌ HNC saddle (RHF/STO-3G) is pinned against the same baseline as
//! `tests/ts_reference.rs` / `tests/neb_reference.rs` (energy ≈ −91.56485 Ha, one
//! imaginary mode ≈ −1248.5 cm⁻¹), so each guess route only has to land the refiner in
//! the right basin. Compute-heavy: run with `--release`.

use hartree::core::{Atom, Element, Molecule};
use hartree::opt::ts::TsStatus;
use hartree::{Job, JobOptions, Method, TsGuessInput};

fn mol(atoms: &[(u32, [f64; 3])]) -> Molecule {
    Molecule::new(
        atoms
            .iter()
            .map(|&(z, p)| Atom::new(Element::from_z(z).unwrap(), p))
            .collect(),
        0,
        1,
    )
}

fn bond(p: &[[f64; 3]], i: usize, j: usize) -> f64 {
    let (a, b) = (p[i], p[j]);
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Atom order `[C, N, H]` in both endpoints; H bound to C (reactant) / N (product),
/// bent off the C–N axis so the path bridges rather than dragging H through the bond.
/// These are basin representatives — the converged saddle is what is pinned.
fn hcn_basin() -> Molecule {
    mol(&[
        (6, [0.0, 0.0, 0.0]),
        (7, [2.19, 0.0, 0.0]),
        (1, [-1.5, 1.2, 0.0]),
    ])
}

fn hnc_basin() -> Molecule {
    mol(&[
        (6, [0.0, 0.0, 0.0]),
        (7, [2.30, 0.0, 0.0]),
        (1, [3.7, 1.2, 0.0]),
    ])
}

/// Assert a converged first-order saddle pinned to the shared RHF/STO-3G baseline.
fn assert_pinned_saddle(result: &hartree::JobResult) {
    let ts = result
        .transition_state
        .as_ref()
        .expect("a transition-state result");
    assert_eq!(
        ts.status,
        TsStatus::Converged,
        "two-endpoint saddle search did not converge"
    );
    let v = ts.verification.as_ref().expect("verification present");
    assert!(
        v.is_first_order_saddle(),
        "expected one imaginary mode, got {:?}",
        v.negative_eigenvalues
    );
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
    let imag = v.imaginary_frequency_cm1.expect("imaginary frequency");
    assert!(
        (imag - (-1248.5)).abs() < 40.0,
        "imaginary frequency {imag:.1} cm⁻¹ drifted from baseline"
    );
}

fn two_endpoint_job(guess: TsGuessInput) -> Job {
    Job {
        molecule: hcn_basin(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            transition_state: true,
            ts_guess: Some(guess),
            ..JobOptions::default()
        },
    }
}

#[test]
fn idpp_two_endpoint_job_converges_the_pinned_saddle() {
    // The default route: a single IDPP guess between the endpoints, seeded with the
    // forming/breaking-bond reaction coordinate (H–C breaking, H–N forming).
    let job = two_endpoint_job(TsGuessInput::new(hnc_basin()));
    let result = job.run().expect("job ran");
    assert_pinned_saddle(&result);
}

#[test]
fn scanned_idpp_two_endpoint_job_converges_the_pinned_saddle() {
    // The IDPP route with an energy-peaked scan: evaluate the SCF surface along the path
    // and place the guess at the parabola-fitted barrier top before refining.
    let mut guess = TsGuessInput::new(hnc_basin());
    guess.scan_points = Some(7);
    let job = two_endpoint_job(guess);
    let result = job.run().expect("job ran");
    assert_pinned_saddle(&result);
}

#[test]
fn neb_two_endpoint_job_converges_the_pinned_saddle() {
    // The robust route: relax a climbing-image NEB band, then refine its climbing image.
    let mut guess = TsGuessInput::new(hnc_basin());
    guess.use_neb = true;
    guess.neb_options.n_images = 6;
    guess.neb_options.gtol = 5e-3;
    guess.neb_options.max_iter = 200;
    guess.neb_options.climb_after = 10;
    guess.neb_options.spring_k = 0.05;

    let job = two_endpoint_job(guess);
    let result = job.run().expect("job ran");
    assert_pinned_saddle(&result);
}
