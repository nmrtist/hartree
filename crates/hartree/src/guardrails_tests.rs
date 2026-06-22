//! Unit tests for [`super`] (per-job model/basis guardrails and the `--recommend`
//! table), split out via `#[path]` so `guardrails.rs` stays under the module-size
//! limit. As a child module it still sees the crate-private items through `super`.

use super::*;
use crate::{JobOptions, Molecule};

fn h2() -> Molecule {
    Molecule::from_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n").unwrap()
}

fn job(method: Method, basis: &str, options: JobOptions) -> Job {
    Job {
        molecule: h2(),
        basis: basis.into(),
        method,
        options,
    }
}

fn dft(name: &str) -> Method {
    Method::Dft(crate::dft::FunctionalSpec::parse(name).unwrap())
}

fn has(warnings: &[String], needle: &str) -> bool {
    warnings.iter().any(|w| w.contains(needle))
}

#[test]
fn pople_and_cc_bases_warn_with_dft_only() {
    for basis in ["6-31g", "cc-pvdz"] {
        let w = assess_job(&job(dft("pbe"), basis, JobOptions::default()));
        assert!(has(&w, "def2 family"), "{basis}: {w:?}");
    }
    let w = assess_job(&job(Method::Mp2, "cc-pvdz", JobOptions::default()));
    assert!(!has(&w, "def2 family"), "{w:?}");
    let w = assess_job(&job(dft("pbe"), "def2-svp", JobOptions::default()));
    assert!(!has(&w, "def2 family"), "{w:?}");
}

#[test]
fn small_basis_warns_and_polarized_does_not() {
    let w = assess_job(&job(Method::Rhf, "sto-3g", JobOptions::default()));
    assert!(has(&w, "minimal/unpolarized"), "{w:?}");
    let w = assess_job(&job(Method::Rhf, "def2-svp", JobOptions::default()));
    assert!(!has(&w, "minimal/unpolarized"), "{w:?}");
}

#[test]
fn pure_gga_carries_barrier_note_hybrids_do_not() {
    let w = assess_job(&job(dft("pbe"), "def2-svp", JobOptions::default()));
    assert!(has(&w, "underestimate"), "{w:?}");
    for name in ["b3lyp", "wb97m-v", "tpss"] {
        let w = assess_job(&job(dft(name), "def2-svp", JobOptions::default()));
        assert!(!has(&w, "underestimate"), "{name}: {w:?}");
    }
}

#[test]
fn missing_dispersion_warns_from_metadata() {
    let w = assess_job(&job(dft("pbe"), "def2-svp", JobOptions::default()));
    assert!(has(&w, "without a dispersion correction"), "{w:?}");
    assert!(has(&w, "\"pbe\""), "{w:?}");

    let with_d4 = JobOptions {
        dispersion: crate::disp::Dispersion::for_method(true, "pbe"),
        ..JobOptions::default()
    };
    let w = assess_job(&job(dft("pbe"), "def2-svp", with_d4));
    assert!(!has(&w, "without a dispersion correction"), "{w:?}");

    let w = assess_job(&job(dft("wb97m-v"), "def2-svp", JobOptions::default()));
    assert!(!has(&w, "without a dispersion correction"), "{w:?}");
}

#[test]
fn hf_notes_missing_correlation_dft_does_not() {
    for m in [Method::Rhf, Method::Uhf, Method::Rohf] {
        let w = assess_job(&job(m, "def2-svp", JobOptions::default()));
        assert!(has(&w, "neglects electron correlation"), "{w:?}");
    }
    let w = assess_job(&job(dft("pbe0"), "def2-svp", JobOptions::default()));
    assert!(!has(&w, "neglects electron correlation"), "{w:?}");
}

#[test]
fn coarse_grid_under_grid_sensitive_functional_warns() {
    let coarse = JobOptions {
        grid_level: 2,
        ..JobOptions::default()
    };
    let w = assess_job(&job(dft("m06-2x"), "def2-svp", coarse));
    assert!(has(&w, "grid-sensitive"), "{w:?}");
    let fine = JobOptions {
        grid_level: 4,
        ..JobOptions::default()
    };
    let w = assess_job(&job(dft("m06-2x"), "def2-svp", fine));
    assert!(!has(&w, "grid-sensitive"), "{w:?}");
    let w = assess_job(&job(
        dft("pbe0"),
        "def2-svp",
        JobOptions {
            grid_level: 1,
            ..JobOptions::default()
        },
    ));
    assert!(!has(&w, "grid-sensitive"), "{w:?}");
}

#[test]
fn r2scan_3c_job_is_warning_clean() {
    let c = crate::composite::composite("r2scan-3c").unwrap();
    let options = JobOptions {
        grid_level: c.grid_level,
        dispersion: Some(c.dispersion),
        gcp: c.gcp,
        srb: c.srb,
        ..JobOptions::default()
    };
    let w = assess_job(&job(dft(c.functional), c.basis, options));
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn recommendation_table_is_consistent() {
    assert!(!RECOMMENDATIONS.is_empty());
    for r in RECOMMENDATIONS {
        assert!(!r.level.is_empty() && !r.rationale.is_empty());
        assert!(!r.invocation.is_empty(), "{}: no invocation", r.task);
        for inv in r.invocation {
            assert!(inv.starts_with("hartree "), "{}: {inv}", r.task);
        }
    }
    assert_eq!(recommend("GENERAL").unwrap().task, "general");
    assert_eq!(recommend("kinetics").unwrap().task, "barriers");
    assert_eq!(recommend("thermo").unwrap().task, "thermochemistry");
    assert!(recommend("nope").is_none());
    assert_eq!(
        recommendation_tasks(),
        vec![
            "general",
            "optimization",
            "barriers",
            "nci",
            "anions",
            "reference",
            "multireference",
            "thermochemistry",
        ]
    );
    assert!(recommend("general").unwrap().level.contains("r2scan-3c"));
    assert!(recommend("barriers").unwrap().level.contains("wb97m-v"));
    assert!(recommend("nci").unwrap().level.contains("wb97m-v"));
}

#[test]
fn task_names_and_aliases_are_unique_and_resolvable() {
    // No name/alias collides with another entry's name or aliases, and every
    // token resolves back to its own entry.
    let mut seen: Vec<&str> = Vec::new();
    for r in RECOMMENDATIONS {
        for tok in std::iter::once(&r.task).chain(r.aliases) {
            assert!(
                !seen.contains(tok),
                "duplicate task/alias token {tok:?} (in {})",
                r.task
            );
            seen.push(tok);
            assert_eq!(recommend(tok).unwrap().task, r.task, "{tok} -> {}", r.task);
            // Lookups are case-insensitive.
            assert_eq!(recommend(&tok.to_uppercase()).unwrap().task, r.task);
        }
    }
}

#[test]
fn new_tasks_resolve_and_stay_on_supported_methods() {
    // The four original tasks keep their core level strings.
    assert!(
        recommend("optimization")
            .unwrap()
            .level
            .contains("r2scan-3c")
    );
    assert_eq!(recommend("geometry-only").unwrap().task, "optimization");
    assert_eq!(recommend("opt").unwrap().task, "optimization");

    // Anions get a diffuse Karlsruhe basis that hartree actually bundles.
    let anions = recommend("anions").unwrap();
    assert_eq!(recommend("diffuse").unwrap().task, "anions");
    assert!(anions.level.contains("def2-TZVPD"));
    assert!(
        crate::basis::BasisSet::load("def2-tzvpd").is_ok(),
        "anions basis must be loadable"
    );

    // Reference ladder names a supported post-HF method and the W1 protocol.
    let reference = recommend("reference").unwrap();
    assert_eq!(recommend("ccsdt").unwrap().task, "reference");
    assert!(reference.level.contains("CCSD(T)"));
    assert!(
        reference
            .invocation
            .iter()
            .any(|i| i.contains("--protocol w1"))
    );

    // The multireference entry is the FOD diagnostic, not a (nonexistent) CASSCF.
    let mr = recommend("multireference").unwrap();
    assert_eq!(recommend("fod").unwrap().task, "multireference");
    assert!(mr.invocation.iter().any(|i| i.contains("--fod")));
    assert!(mr.notes.iter().any(|n| n.contains("CASSCF")));
}

#[test]
fn thermochemistry_note_states_actual_heavy_element_coverage() {
    // The heavy-element note must reflect what hartree actually vendors (def2-ECP for
    // Ag/Sn/I/Au on def2-SVP/def2-TZVP), not claim blanket automatic ECP coverage for
    // every Z > 36 — only those bases carry the heavy orbital split.
    let notes = recommend("thermochemistry").unwrap().notes.join(" ");
    assert!(
        notes.contains("Z > 36"),
        "mentions the heavy-element regime: {notes}"
    );
    // The note must state the actual coverage: the full def2-ECP Rb–Rn range on the
    // heavy-capable def2-SVP/def2-TZVP orbital bases.
    assert!(
        notes.contains("Rb") && notes.contains("Rn"),
        "names the supported def2-ECP range: {notes}"
    );
    assert!(
        notes.contains("def2-SVP") || notes.contains("def2-TZVP"),
        "names the heavy-capable orbital bases: {notes}"
    );
}

#[test]
fn transition_state_note_is_algorithm_aware() {
    use crate::opt::ts::{TsAlgorithm, TsOptions};

    // P-RFO (the default): the shared FD-Hessian cost line plus the guess
    // requirement, the dimer alternative, and the --ts-recalc-hessian knob.
    let prfo = JobOptions {
        transition_state: true,
        ..JobOptions::default()
    };
    let w = assess_job(&job(Method::Rhf, "def2-svp", prfo));
    assert!(has(&w, "transition-state search"), "{w:?}");
    assert!(has(&w, "6·natom"), "{w:?}");
    assert!(has(&w, "P-RFO"), "{w:?}");
    assert!(has(&w, "--ts-algo dimer"), "{w:?}");
    assert!(has(&w, "--ts-recalc-hessian"), "{w:?}");

    // Dimer: Hessian-free framing, and it says --ts-recalc-hessian does not apply.
    let dimer = JobOptions {
        transition_state: true,
        ts_options: TsOptions {
            algorithm: TsAlgorithm::Dimer,
            ..TsOptions::default()
        },
        ..JobOptions::default()
    };
    let w = assess_job(&job(Method::Rhf, "def2-svp", dimer));
    assert!(has(&w, "transition-state search"), "{w:?}");
    assert!(has(&w, "Hessian-free"), "{w:?}");
    assert!(has(&w, "does not apply"), "{w:?}");
    assert!(!has(&w, "P-RFO"), "{w:?}");

    // Without --ts there is no transition-state note at all.
    let plain = assess_job(&job(Method::Rhf, "def2-svp", JobOptions::default()));
    assert!(!has(&plain, "transition-state search"), "{plain:?}");
}
