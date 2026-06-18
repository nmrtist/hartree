use hartree::basis::BasisSet;
use hartree::cc::{CcsdOptions, frozen_core_orbitals, rccsd_spin_orbital, rhf_mp2};
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::ConventionalProvider;
use hartree::scf::{ScfOptions, ScfResult, run_rhf};
use serde::Deserialize;

use std::collections::HashMap;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");
const CCSD_REFERENCES_JSON: &str = include_str!("../../../tests/ref/ccsd_references.json");

const CRAWFORD_WATER_STO3G_CCSD_CORR: f64 = -0.070680088372;

#[derive(Deserialize)]
struct Geometries {
    molecules: HashMap<String, GeomEntry>,
}

#[derive(Deserialize)]
struct GeomEntry {
    charge: i32,
    multiplicity: u32,
    atoms: Vec<(String, f64, f64, f64)>,
}

#[derive(Deserialize)]
struct CcsdReferences {
    entries: Vec<CcsdRefEntry>,
}

#[derive(Deserialize)]
struct CcsdRefEntry {
    molecule: String,
    basis: String,
    frozen_core: bool,
    ccsd_correlation: f64,
    ccsd_total_energy: f64,
}

fn molecule(g: &GeomEntry) -> Molecule {
    let atoms = g
        .atoms
        .iter()
        .map(|(s, x, y, z)| Atom::new(Element::from_symbol(s).unwrap(), [*x, *y, *z]))
        .collect();
    Molecule::new(atoms, g.charge, g.multiplicity)
}

fn provider_for(mol: &Molecule, basis: &str) -> ConventionalProvider {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    ConventionalProvider::new(ao.into_integral(), charges)
}

fn converged_rhf(provider: &ConventionalProvider, mol: &Molecule) -> ScfResult {
    let opts = ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        ..ScfOptions::default()
    };
    let scf = run_rhf(
        provider,
        mol.n_electrons() as usize,
        mol.nuclear_repulsion(),
        &opts,
    )
    .unwrap();
    assert!(scf.converged, "SCF did not converge");
    scf
}

#[test]
fn ccsd_iteration0_equals_mp2() {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let opts = CcsdOptions::default();
    for (name, basis) in [("h2", "6-31g"), ("water", "sto-3g"), ("water", "6-31g")] {
        let mol = molecule(&geoms.molecules[name]);
        let provider = provider_for(&mol, basis);
        let scf = converged_rhf(&provider, &mol);
        for n_frozen in [0, frozen_core_orbitals(&mol)] {
            let mp2 = rhf_mp2(&provider, &scf, n_frozen);
            let cc = rccsd_spin_orbital(&provider, &scf, n_frozen, &opts);
            let d = cc.mp2_correlation - mp2.correlation_energy;
            assert!(
                d.abs() < 1e-10,
                "{name}/{basis} (frozen {n_frozen}): CCSD iter-0 {:.12} vs MP2 {:.12} (Δ {:.2e})",
                cc.mp2_correlation,
                mp2.correlation_energy,
                d
            );
        }
    }
}

#[test]
fn ccsd_crawford_anchor() {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let mol = molecule(&geoms.molecules["water"]);
    let provider = provider_for(&mol, "sto-3g");
    let scf = converged_rhf(&provider, &mol);
    let cc = rccsd_spin_orbital(&provider, &scf, 0, &CcsdOptions::default());
    assert!(cc.converged, "CCSD did not converge");
    let d = cc.correlation_energy - CRAWFORD_WATER_STO3G_CCSD_CORR;
    eprintln!(
        "Crawford anchor: hartree {:.12} vs canonical {:.12} (Δ {:.2e})",
        cc.correlation_energy, CRAWFORD_WATER_STO3G_CCSD_CORR, d
    );
    assert!(d.abs() < 1e-9, "Crawford anchor Δ {d:.2e} exceeds 1e-9");
}

fn check_oracle(bases: &[&str]) -> usize {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let refs: CcsdReferences = serde_json::from_str(CCSD_REFERENCES_JSON).unwrap();
    let opts = CcsdOptions::default();

    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in &refs.entries {
        if !bases.contains(&entry.basis.as_str()) {
            continue;
        }
        if entry.molecule == "ethylene" {
            continue;
        }
        let mol = molecule(&geoms.molecules[&entry.molecule]);
        let provider = provider_for(&mol, &entry.basis);
        let scf = converged_rhf(&provider, &mol);
        let n_frozen = if entry.frozen_core {
            frozen_core_orbitals(&mol)
        } else {
            0
        };
        let cc = rccsd_spin_orbital(&provider, &scf, n_frozen, &opts);
        assert!(
            cc.converged,
            "{}/{} CCSD not converged",
            entry.molecule, entry.basis
        );
        let d_corr = cc.correlation_energy - entry.ccsd_correlation;
        let d_total = cc.total_energy - entry.ccsd_total_energy;
        worst = worst.max(d_corr.abs());
        let fc = if entry.frozen_core { "fc" } else { "ae" };
        eprintln!(
            "CCSD {}/{}/{fc}: Ecorr {:.10} (Δ {:.2e}), Etot Δ {:.2e}  [{} iters]",
            entry.molecule, entry.basis, cc.correlation_energy, d_corr, d_total, cc.iterations
        );
        assert!(
            d_corr.abs() < 1e-6,
            "CCSD {}/{}/{fc} correlation: hartree {:.10} vs ORCA {:.10} (Δ {:.2e})",
            entry.molecule,
            entry.basis,
            cc.correlation_energy,
            entry.ccsd_correlation,
            d_corr
        );
        assert!(
            d_total.abs() < 1e-6,
            "CCSD {}/{}/{fc} total Δ {:.2e}",
            entry.molecule,
            entry.basis,
            d_total
        );
        checked += 1;
    }
    eprintln!("CCSD: {checked} oracle entries matched; worst ΔEcorr = {worst:.2e}");
    checked
}

#[test]
fn rccsd_spin_orbital_matches_oracle_fast() {
    let checked = check_oracle(&["sto-3g", "6-31g"]);
    assert!(
        checked >= 3,
        "expected ≥3 fast oracle entries, checked {checked}"
    );
}
