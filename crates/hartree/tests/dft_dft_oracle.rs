mod dft_common;

use dft_common::{DftReferences, dft_references, geometries, run_ks};
use rayon::prelude::*;

const DFT_GATE_CLEAN: f64 = 3.0e-6;
const DFT_GATE_PBE: f64 = 7.5e-5;

fn gate_for(functional: &str) -> f64 {
    if functional == "pbe" || functional == "pbe0" {
        DFT_GATE_PBE
    } else {
        DFT_GATE_CLEAN
    }
}

fn check_subset(refs: &DftReferences, want_6_31g: bool, level: usize) -> usize {
    let geoms = geometries();

    struct Row {
        line: String,
        delta_abs: f64,
        is_pbe: bool,
        over_gate: Option<String>,
    }
    let rows: Vec<Row> = refs
        .entries
        .par_iter()
        .filter(|entry| (entry.basis == "6-31g") == want_6_31g)
        .map(|entry| {
            let geom = &geoms.molecules[&entry.molecule];
            let r = run_ks(geom, &entry.basis, &entry.functional, level);
            assert!(
                r.converged,
                "{}/{}/{}: KS did not converge",
                entry.molecule, entry.basis, entry.functional
            );
            let delta = r.energy - entry.energy;
            let gate = gate_for(&entry.functional);
            let line = format!(
                "  {:9}/{:8}/{:5} [{}] hartree {:.8}  ORCA {:.8}  Δ={:+.2e}  (gate {:.1e})",
                entry.molecule,
                entry.basis,
                entry.functional,
                entry.reference,
                r.energy,
                entry.energy,
                delta,
                gate
            );
            let over_gate = (delta.abs() >= gate).then(|| {
                format!(
                    "{}/{}/{}: Δ = {:.2e} ≥ gate {:.1e}",
                    entry.molecule, entry.basis, entry.functional, delta, gate
                )
            });
            Row {
                line,
                delta_abs: delta.abs(),
                is_pbe: gate == DFT_GATE_PBE,
                over_gate,
            }
        })
        .collect();

    let mut worst_clean = 0.0_f64;
    let mut worst_pbe = 0.0_f64;
    let mut over_gate: Vec<String> = Vec::new();
    for row in &rows {
        eprintln!("{}", row.line);
        if row.is_pbe {
            worst_pbe = worst_pbe.max(row.delta_abs);
        } else {
            worst_clean = worst_clean.max(row.delta_abs);
        }
        if let Some(o) = &row.over_gate {
            over_gate.push(o.clone());
        }
    }
    let checked = rows.len();
    eprintln!(
        "level {level}: {checked} references matched ORCA; worst clean Δ = {worst_clean:.2e} \
         (gate {DFT_GATE_CLEAN:.1e}); worst pbe-family Δ = {worst_pbe:.2e} (gate {DFT_GATE_PBE:.1e})"
    );
    assert!(
        over_gate.is_empty(),
        "level {level}: {} entries over gate:\n  {}",
        over_gate.len(),
        over_gate.join("\n  ")
    );
    checked
}

#[test]
fn dft_fast_subset_matches_orca() {
    let refs = dft_references();
    assert_eq!(refs.provenance.engine, "ORCA");
    let checked = check_subset(&refs, true, 3);
    assert!(
        checked >= 19,
        "expected ≥19 fast 6-31g references, checked {checked}"
    );
}

#[test]
#[ignore = "cc-pVDZ/def2-SVP + ethylene at grid level 4; slow — run with --release -- --ignored"]
fn dft_full_set_matches_orca() {
    let refs = dft_references();
    assert_eq!(refs.provenance.engine, "ORCA");
    let checked = check_subset(&refs, false, 4);
    assert!(
        checked >= 30,
        "expected ≥30 cc-pVDZ/def2-SVP references, checked {checked}"
    );
}
