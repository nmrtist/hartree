//! Integration tests for the resource-control surface added for downstream
//! embedders: `estimate_memory` (pre-flight sizing), `JobOptions::n_threads`
//! (scoped thread pool), and `JobOptions::mem_budget_bytes` (budget guard with
//! a safe integral-direct downgrade).

use hartree::dft::FunctionalSpec;
use hartree::{Atom, Element, EstimateBackend, Job, JobOptions, Method, Molecule, estimate_memory};

fn dft(name: &str) -> Method {
    Method::Dft(FunctionalSpec::parse(name).unwrap())
}

fn atom(symbol: &str, charge: i32, multiplicity: u32) -> Molecule {
    let a = Atom::new(Element::from_symbol(symbol).unwrap(), [0.0, 0.0, 0.0]);
    Molecule::new(vec![a], charge, multiplicity)
}

fn labels(est: &hartree::MemoryEstimate) -> Vec<&str> {
    est.breakdown.iter().map(|t| t.label.as_str()).collect()
}

fn water() -> Molecule {
    let atoms = vec![
        Atom::new(Element::from_symbol("O").unwrap(), [0.0, 0.0, 0.0]),
        Atom::new(Element::from_symbol("H").unwrap(), [0.0, 0.757, 0.587]),
        Atom::new(Element::from_symbol("H").unwrap(), [0.0, -0.757, 0.587]),
    ];
    Molecule::new(atoms, 0, 1)
}

fn hydrogen() -> Molecule {
    let atoms = vec![
        Atom::new(Element::from_symbol("H").unwrap(), [0.0, 0.0, 0.0]),
        Atom::new(Element::from_symbol("H").unwrap(), [0.0, 0.0, 0.74]),
    ];
    Molecule::new(atoms, 0, 1)
}

fn job(mol: Molecule, basis: &str, method: Method, options: JobOptions) -> Job {
    Job {
        molecule: mol,
        basis: basis.to_string(),
        method,
        options,
    }
}

#[test]
fn estimate_reports_conventional_backend_and_eri() {
    let est = estimate_memory(&job(water(), "sto-3g", Method::Rhf, JobOptions::default())).unwrap();
    assert_eq!(est.backend, EstimateBackend::Conventional);
    assert!(est.peak_bytes > 0);
    // The in-core ERI is the dominant, and therefore first, breakdown term.
    assert_eq!(est.breakdown[0].label, "eri_in_core");
    assert_eq!(
        est.peak_bytes,
        est.breakdown.iter().map(|t| t.bytes).sum::<u64>()
    );
}

#[test]
fn estimate_grows_with_basis_size() {
    let small =
        estimate_memory(&job(water(), "sto-3g", Method::Rhf, JobOptions::default())).unwrap();
    let large = estimate_memory(&job(
        water(),
        "def2-svp",
        Method::Rhf,
        JobOptions::default(),
    ))
    .unwrap();
    assert!(
        large.peak_bytes > small.peak_bytes,
        "def2-svp ({}) should estimate larger than sto-3g ({})",
        large.peak_bytes,
        small.peak_bytes
    );
}

#[test]
fn estimate_backends_track_options() {
    let direct = estimate_memory(&job(
        water(),
        "sto-3g",
        Method::Rhf,
        JobOptions {
            direct: true,
            ..JobOptions::default()
        },
    ))
    .unwrap();
    assert_eq!(direct.backend, EstimateBackend::Direct);

    let conventional =
        estimate_memory(&job(water(), "sto-3g", Method::Rhf, JobOptions::default())).unwrap();
    // Integral-direct stores no ERI, so it must estimate well below in-core.
    assert!(direct.peak_bytes < conventional.peak_bytes);
}

#[test]
fn n_threads_is_deterministic() {
    let reference = job(hydrogen(), "sto-3g", Method::Rhf, JobOptions::default())
        .run()
        .unwrap();
    for threads in [1usize, 2, 4] {
        let capped = job(
            hydrogen(),
            "sto-3g",
            Method::Rhf,
            JobOptions {
                n_threads: Some(threads),
                ..JobOptions::default()
            },
        )
        .run()
        .unwrap();
        assert!(
            (capped.scf.energy - reference.scf.energy).abs() < 1e-9,
            "n_threads={threads} changed the energy: {} vs {}",
            capped.scf.energy,
            reference.scf.energy
        );
    }
}

#[test]
fn n_threads_zero_falls_back_to_default() {
    // Some(0) must be treated as "use the default pool", not "a zero-thread pool".
    let result = job(
        hydrogen(),
        "sto-3g",
        Method::Rhf,
        JobOptions {
            n_threads: Some(0),
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap();
    assert!(result.scf.converged);
}

#[test]
fn generous_budget_runs_unchanged() {
    let baseline = job(water(), "sto-3g", Method::Rhf, JobOptions::default())
        .run()
        .unwrap();
    let budgeted = job(
        water(),
        "sto-3g",
        Method::Rhf,
        JobOptions {
            mem_budget_bytes: Some(u64::MAX),
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap();
    assert!((budgeted.scf.energy - baseline.scf.energy).abs() < 1e-10);
    // No downgrade happened, so the structured report is absent.
    assert!(budgeted.backend_downgrade.is_none());
}

#[test]
fn tight_budget_downgrades_to_direct() {
    // A budget that fits integral-direct but not the in-core ERI must trigger a
    // transparent downgrade rather than an out-of-memory run.
    let direct_peak = estimate_memory(&job(
        water(),
        "sto-3g",
        Method::Rhf,
        JobOptions {
            direct: true,
            ..JobOptions::default()
        },
    ))
    .unwrap()
    .peak_bytes;

    let result = job(
        water(),
        "sto-3g",
        Method::Rhf,
        JobOptions {
            mem_budget_bytes: Some(direct_peak),
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap();

    assert!(result.scf.converged);
    // The downgrade is reported, not silent: a structured field AND a warning.
    let report = result
        .backend_downgrade
        .as_ref()
        .expect("a budget downgrade should be reported structurally");
    assert_eq!(report.from, EstimateBackend::Conventional);
    assert_eq!(report.to, EstimateBackend::Direct);
    assert_eq!(report.budget_bytes, direct_peak);
    assert!(report.estimated_bytes > direct_peak);
    assert!(
        result
            .method_warnings
            .iter()
            .any(|w| w.contains("switched to the direct backend")),
        "expected a downgrade warning, got {:?}",
        result.method_warnings
    );
}

#[test]
fn downgraded_energy_matches_conventional() {
    // The integral-direct fallback must reproduce the in-core energy.
    let conventional = job(water(), "sto-3g", Method::Rhf, JobOptions::default())
        .run()
        .unwrap();
    let direct_peak = estimate_memory(&job(
        water(),
        "sto-3g",
        Method::Rhf,
        JobOptions {
            direct: true,
            ..JobOptions::default()
        },
    ))
    .unwrap()
    .peak_bytes;
    let downgraded = job(
        water(),
        "sto-3g",
        Method::Rhf,
        JobOptions {
            mem_budget_bytes: Some(direct_peak),
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap();
    assert!((downgraded.scf.energy - conventional.scf.energy).abs() < 1e-7);
}

#[test]
fn dft_estimate_includes_grid_term() {
    let est =
        estimate_memory(&job(water(), "sto-3g", dft("b3lyp"), JobOptions::default())).unwrap();
    assert_eq!(est.backend, EstimateBackend::Conventional);
    assert!(labels(&est).contains(&"dft_grid"));
    assert!(labels(&est).contains(&"eri_in_core"));
    // b3lyp is a plain hybrid (no range separation), so no long-range tensor.
    assert!(!labels(&est).contains(&"eri_long_range"));
}

#[test]
fn range_separated_functional_adds_long_range_tensor() {
    // wb97x-v is range-separated (CAM): a second nao⁴ erf-attenuated tensor.
    let est = estimate_memory(&job(
        water(),
        "sto-3g",
        dft("wb97x-v"),
        JobOptions::default(),
    ))
    .unwrap();
    let eri = est
        .breakdown
        .iter()
        .find(|t| t.label == "eri_in_core")
        .unwrap();
    let lr = est
        .breakdown
        .iter()
        .find(|t| t.label == "eri_long_range")
        .expect("range-separated functional should add eri_long_range");
    assert_eq!(eri.bytes, lr.bytes);
}

#[test]
fn ccsd_and_ccsdt_estimates_grow_with_method() {
    let mp2 = estimate_memory(&job(water(), "sto-3g", Method::Mp2, JobOptions::default())).unwrap();
    let ccsd =
        estimate_memory(&job(water(), "sto-3g", Method::Ccsd, JobOptions::default())).unwrap();
    let ccsdt = estimate_memory(&job(
        water(),
        "sto-3g",
        Method::CcsdT,
        JobOptions::default(),
    ))
    .unwrap();
    assert!(labels(&ccsd).contains(&"ccsd_mo_integrals"));
    assert!(labels(&ccsd).contains(&"ccsd_vvvv_intermediate"));
    assert!(!labels(&ccsd).contains(&"ccsdt_triples_blocks"));
    assert!(labels(&ccsdt).contains(&"ccsdt_triples_blocks"));
    // CCSD(T) carries every CCSD block plus the triples, so it estimates larger;
    // CCSD in turn exceeds conventional MP2.
    assert!(ccsdt.peak_bytes > ccsd.peak_bytes);
    assert!(ccsd.peak_bytes > mp2.peak_bytes);
}

#[test]
fn ri_mp2_estimate_uses_fitted_tensors_not_a_full_clone() {
    // def2-svp has a bundled def2-svp/c MP2-fit auxiliary partner.
    let opts = JobOptions {
        ri_mp2: true,
        ..JobOptions::default()
    };
    let est = estimate_memory(&job(water(), "def2-svp", Method::Mp2, opts)).unwrap();
    assert!(labels(&est).contains(&"rimp2_mo_integrals"));
    assert!(labels(&est).contains(&"rimp2_3c_scratch"));
    // RI-MP2 must NOT model the conventional nao⁴ transform clone.
    assert!(!labels(&est).contains(&"mp2_transform_scratch"));
}

#[test]
fn double_hybrid_estimate_adds_pt2_terms_on_conventional_backend() {
    // b2plyp is a (conventional-PT2) double hybrid.
    let est = estimate_memory(&job(
        water(),
        "sto-3g",
        dft("b2plyp"),
        JobOptions::default(),
    ))
    .unwrap();
    assert_eq!(est.backend, EstimateBackend::Conventional);
    assert!(labels(&est).contains(&"mp2_mo_integrals"));
    assert!(labels(&est).contains(&"dft_grid"));
}

#[test]
fn open_shell_occupancy_is_handled() {
    // A lithium-atom doublet exercises n_alpha != n_beta through the shared
    // alpha_beta_electrons derivation.
    let est = estimate_memory(&job(
        atom("Li", 0, 2),
        "sto-3g",
        Method::Mp2,
        JobOptions::default(),
    ))
    .unwrap();
    assert!(est.peak_bytes > 0);
    assert!(labels(&est).contains(&"mp2_mo_integrals"));
}

#[test]
fn invalid_multiplicity_errors_through_estimate() {
    // Two electrons cannot support a quintet: the shared validation must surface.
    let mol = Molecule::new(hydrogen().atoms, 0, 5);
    let err = estimate_memory(&job(mol, "sto-3g", Method::Rhf, JobOptions::default())).unwrap_err();
    assert!(err.contains("multiplicity"), "unexpected error: {err}");
}

#[test]
fn over_budget_ineligible_job_is_refused_not_downgraded() {
    // compute_properties is unsupported by integral-direct, so an over-budget
    // properties job must refuse rather than silently downgrade.
    let opts = JobOptions {
        compute_properties: true,
        mem_budget_bytes: Some(1),
        ..JobOptions::default()
    };
    let err = job(water(), "sto-3g", Method::Rhf, opts).run().unwrap_err();
    assert!(
        err.contains("exceeds") && err.contains("budget"),
        "unexpected error: {err}"
    );
}

#[test]
fn over_budget_eligible_but_direct_also_too_small_is_refused() {
    // RHF is downgrade-eligible, but a 1-byte budget cannot fit even integral-direct.
    let opts = JobOptions {
        mem_budget_bytes: Some(1),
        ..JobOptions::default()
    };
    let err = job(water(), "sto-3g", Method::Rhf, opts).run().unwrap_err();
    assert!(err.contains("exceeds"), "unexpected error: {err}");
}

#[test]
fn unsatisfiable_budget_is_refused() {
    // MP2 cannot run integral-direct, so an impossibly tight budget must error
    // out cleanly rather than downgrade or OOM.
    let err = job(
        water(),
        "sto-3g",
        Method::Mp2,
        JobOptions {
            mem_budget_bytes: Some(1),
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap_err();
    assert!(
        err.contains("exceeds") && err.contains("budget"),
        "unexpected error: {err}"
    );
}
