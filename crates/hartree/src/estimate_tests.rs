//! Unit tests for [`super`] (pre-flight peak-memory estimation), split out via
//! `#[path]` so `estimate.rs` stays under the module-size limit. As a child module it
//! still sees the crate-private helpers (`term`, `doubles`, …) through `super`.

use super::*;
use crate::JobOptions;
use crate::core::{Atom, Element};

fn water() -> Molecule {
    let atoms = vec![
        Atom::new(Element::from_symbol("O").unwrap(), [0.0, 0.0, 0.0]),
        Atom::new(Element::from_symbol("H").unwrap(), [0.0, 0.757, 0.587]),
        Atom::new(Element::from_symbol("H").unwrap(), [0.0, -0.757, 0.587]),
    ];
    Molecule::new(atoms, 0, 1)
}

fn job(method: Method, opts: JobOptions) -> Job {
    Job {
        molecule: water(),
        basis: "sto-3g".to_string(),
        method,
        options: opts,
    }
}

#[test]
fn doubles_saturates() {
    assert_eq!(doubles(0), 0);
    assert_eq!(doubles(10), 80);
    assert_eq!(doubles(u128::MAX), u64::MAX);
}

#[test]
fn human_bytes_scales() {
    assert_eq!(human_bytes(512), "512 B");
    assert_eq!(human_bytes(1024), "1.00 KiB");
    assert_eq!(human_bytes(1024 * 1024), "1.00 MiB");
}

#[test]
fn term_drops_zero() {
    assert!(term("x", 0).is_none());
    assert_eq!(term("x", 1).unwrap().bytes, 8);
}

#[test]
fn conventional_hf_dominated_by_eri() {
    // Water/STO-3G has 7 AOs: the ERI tensor is 7⁴·8 bytes.
    let est = estimate_memory(&job(Method::Rhf, JobOptions::default())).unwrap();
    assert_eq!(est.backend, EstimateBackend::Conventional);
    let eri = est
        .breakdown
        .iter()
        .find(|t| t.label == "eri_in_core")
        .unwrap();
    assert_eq!(eri.bytes, 7u64.pow(4) * 8);
    // peak_bytes is the sum of the breakdown, and the breakdown is sorted.
    let sum: u64 = est.breakdown.iter().map(|t| t.bytes).sum();
    assert_eq!(est.peak_bytes, sum);
    assert!(est.breakdown.windows(2).all(|w| w[0].bytes >= w[1].bytes));
}

#[test]
fn mp2_adds_correlation_terms() {
    let hf = estimate_memory(&job(Method::Rhf, JobOptions::default())).unwrap();
    let mp2 = estimate_memory(&job(Method::Mp2, JobOptions::default())).unwrap();
    assert!(mp2.peak_bytes > hf.peak_bytes);
    assert!(mp2.breakdown.iter().any(|t| t.label == "mp2_mo_integrals"));
    assert!(
        mp2.breakdown
            .iter()
            .any(|t| t.label == "mp2_transform_scratch")
    );
}

#[test]
fn direct_backend_has_no_eri() {
    let opts = JobOptions {
        direct: true,
        ..JobOptions::default()
    };
    let est = estimate_memory(&job(Method::Rhf, opts)).unwrap();
    assert_eq!(est.backend, EstimateBackend::Direct);
    assert!(!est.breakdown.iter().any(|t| t.label == "eri_in_core"));
    assert!(est.breakdown.iter().any(|t| t.label == "schwarz_table"));
}

#[test]
fn ri_backend_reports_fitted_tensor() {
    let opts = JobOptions {
        ri: true,
        ..JobOptions::default()
    };
    let est = estimate_memory(&job(Method::Rhf, opts)).unwrap();
    assert_eq!(est.backend, EstimateBackend::Ri);
    assert!(est.breakdown.iter().any(|t| t.label == "df_b_tensor"));
}

#[test]
fn transition_state_adds_hessian_and_concurrency_terms() {
    let ts = JobOptions {
        transition_state: true,
        ..JobOptions::default()
    };
    let base = estimate_memory(&job(Method::Rhf, JobOptions::default())).unwrap();
    let est = estimate_memory(&job(Method::Rhf, ts)).unwrap();

    // The Hessian phase pushes the peak above the bare SCF estimate.
    assert!(
        est.peak_bytes > base.peak_bytes,
        "TS peak {} should exceed plain SCF peak {}",
        est.peak_bytes,
        base.peak_bytes
    );
    for label in [
        "ts_hessian",
        "ts_eigensolver_scratch",
        "ts_fd_hessian_concurrency",
    ] {
        assert!(
            est.breakdown.iter().any(|t| t.label == label),
            "missing {label}: {:?}",
            est.breakdown
        );
    }
    // Water has 3 atoms ⇒ ndof = 9, so the dense Hessian is 9²·8 bytes.
    let hess = est
        .breakdown
        .iter()
        .find(|t| t.label == "ts_hessian")
        .unwrap();
    assert_eq!(hess.bytes, 9 * 9 * 8);
    // The concurrency term reuses the per-evaluation SCF working set (the ERI plus
    // SCF matrices for water/STO-3G), scaled by the live evaluation count (≥ 1).
    let conc = est
        .breakdown
        .iter()
        .find(|t| t.label == "ts_fd_hessian_concurrency")
        .unwrap();
    let scf_working_set = 7u64.pow(4) * 8 + 6 * 7u64.pow(2) * 8;
    assert!(
        conc.bytes >= scf_working_set,
        "concurrency {} should be at least one SCF working set {scf_working_set}",
        conc.bytes
    );
}

#[test]
fn two_endpoint_guess_does_not_change_the_estimate() {
    // A two-endpoint TS search builds its guess from a product that shares the
    // reactant's atom count and composition, so the saddle search still runs on a
    // 3-atom molecule — the memory estimate is byte-identical to the single-geometry
    // TS estimate, and the product carried in `ts_guess` adds no SCF working set.
    let single = JobOptions {
        transition_state: true,
        ..JobOptions::default()
    };
    let two_endpoint = JobOptions {
        transition_state: true,
        ts_guess: Some(crate::TsGuessInput::new(water())),
        ..JobOptions::default()
    };
    let single_est = estimate_memory(&job(Method::Rhf, single)).unwrap();
    let two_est = estimate_memory(&job(Method::Rhf, two_endpoint)).unwrap();
    assert_eq!(
        single_est.peak_bytes, two_est.peak_bytes,
        "two-endpoint guess changed the peak estimate"
    );
}

#[test]
fn unknown_basis_errors() {
    let j = Job {
        molecule: water(),
        basis: "not-a-basis".to_string(),
        method: Method::Rhf,
        options: JobOptions::default(),
    };
    assert!(estimate_memory(&j).is_err());
}

#[test]
fn estimate_is_serde_round_trippable() {
    let est = estimate_memory(&job(Method::Rhf, JobOptions::default())).unwrap();
    let json = serde_json::to_string(&est).unwrap();
    let back: MemoryEstimate = serde_json::from_str(&json).unwrap();
    assert_eq!(est, back);
}
