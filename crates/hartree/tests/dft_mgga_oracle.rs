mod dft_common;

use dft_common::{geometries, run_ks_error_tol};
use serde::Deserialize;

const MGGA_REFERENCES_JSON: &str = include_str!("../../../tests/ref/mgga_references.json");

#[derive(Deserialize)]
struct MggaReferences {
    provenance: Provenance,
    entries: Vec<MggaRefEntry>,
}

#[derive(Deserialize)]
struct Provenance {
    engine: String,
}

#[derive(Deserialize)]
struct MggaRefEntry {
    molecule: String,
    basis: String,
    functional: String,
    reference: String,
    energy: f64,
}

const MGGA_GATE_TPSS: f64 = 3.0e-6;
const MGGA_GATE_R2SCAN: f64 = 1.3e-5;

fn level_and_gate(functional: &str) -> (usize, f64) {
    match functional {
        "tpss" => (3, MGGA_GATE_TPSS),
        "r2scan" => (4, MGGA_GATE_R2SCAN),
        other => panic!("unexpected functional {other} in mgga_references.json"),
    }
}

fn check_subset(refs: &MggaReferences, want_6_31g: bool) -> usize {
    let geoms = geometries();
    let mut checked = 0;
    let mut worst = 0.0_f64;
    let mut over_gate: Vec<String> = Vec::new();
    for entry in &refs.entries {
        if (entry.basis == "6-31g") != want_6_31g {
            continue;
        }
        let (level, gate) = level_and_gate(&entry.functional);
        let geom = &geoms.molecules[&entry.molecule];
        let r = run_ks_error_tol(geom, &entry.basis, &entry.functional, level, 3e-6);
        assert!(
            r.converged,
            "{}/{}/{}: KS did not converge",
            entry.molecule, entry.basis, entry.functional
        );
        let delta = r.energy - entry.energy;
        worst = worst.max(delta.abs());
        eprintln!(
            "  {:8}/{:8}/{:7} [{}] L{} hartree {:.8}  PySCF {:.8}  Δ={:+.2e}  (gate {:.1e})",
            entry.molecule,
            entry.basis,
            entry.functional,
            entry.reference,
            level,
            r.energy,
            entry.energy,
            delta,
            gate
        );
        if delta.abs() >= gate {
            over_gate.push(format!(
                "{}/{}/{}: Δ = {:.2e} ≥ gate {:.1e}",
                entry.molecule, entry.basis, entry.functional, delta, gate
            ));
        }
        checked += 1;
    }
    eprintln!("{checked} meta-GGA references matched PySCF; worst Δ = {worst:.2e}");
    assert!(
        over_gate.is_empty(),
        "{} entries over gate:\n  {}",
        over_gate.len(),
        over_gate.join("\n  ")
    );
    checked
}

fn mgga_references() -> MggaReferences {
    serde_json::from_str(MGGA_REFERENCES_JSON).expect("parse mgga_references.json")
}

#[test]
fn mgga_fast_subset_matches_pyscf() {
    let refs = mgga_references();
    assert_eq!(refs.provenance.engine, "PySCF");
    let checked = check_subset(&refs, true);
    assert!(
        checked >= 6,
        "expected ≥6 fast 6-31g references, checked {checked}"
    );
}

#[test]
#[ignore = "cc-pVDZ tier; slow — run with --release -- --ignored"]
fn mgga_full_set_matches_pyscf() {
    let refs = mgga_references();
    assert_eq!(refs.provenance.engine, "PySCF");
    let checked = check_subset(&refs, false);
    assert!(
        checked >= 6,
        "expected ≥6 cc-pVDZ references, checked {checked}"
    );
}
