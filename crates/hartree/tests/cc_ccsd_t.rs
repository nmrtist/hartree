use hartree::basis::BasisSet;
use hartree::cc::{CcsdOptions, frozen_core_orbitals, rccsd_t_spin_adapted};
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::ConventionalProvider;
use hartree::scf::{ScfOptions, ScfResult, run_rhf};
use serde::Deserialize;
use std::collections::HashMap;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");
const CCSD_T_REFERENCES_JSON: &str = include_str!("../../../tests/ref/ccsd_t_references.json");

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
struct CcsdTReferences {
    entries: Vec<CcsdTRefEntry>,
}
#[derive(Deserialize)]
struct CcsdTRefEntry {
    molecule: String,
    basis: String,
    frozen_core: bool,
    triples_correction: f64,
    ccsd_t_total_energy: f64,
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

fn check_oracle(bases: &[&str]) -> usize {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let refs: CcsdTReferences = serde_json::from_str(CCSD_T_REFERENCES_JSON).unwrap();
    let opts = CcsdOptions::default();

    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in &refs.entries {
        if !bases.contains(&entry.basis.as_str()) {
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
        let r = rccsd_t_spin_adapted(&provider, &scf, n_frozen, &opts);
        let d_t = r.triples_energy - entry.triples_correction;
        let d_tot = r.total_energy - entry.ccsd_t_total_energy;
        worst = worst.max(d_t.abs()).max(d_tot.abs());
        let fc = if entry.frozen_core { "fc" } else { "ae" };
        eprintln!(
            "CCSD(T) {}/{}/{fc}: (T) {:.10} (Δ {:.2e}), Etot {:.10} (Δ {:.2e})",
            entry.molecule, entry.basis, r.triples_energy, d_t, r.total_energy, d_tot
        );
        assert!(
            d_t.abs() < 1e-6,
            "{}/{}/{fc} (T): hartree {:.10} vs ORCA {:.10} (Δ {:.2e})",
            entry.molecule,
            entry.basis,
            r.triples_energy,
            entry.triples_correction,
            d_t
        );
        assert!(
            d_tot.abs() < 1e-6,
            "{}/{}/{fc} CCSD(T) total Δ {:.2e}",
            entry.molecule,
            entry.basis,
            d_tot
        );
        checked += 1;
    }
    eprintln!("CCSD(T): {checked} oracle entries matched; worst Δ = {worst:.2e}");
    checked
}

#[test]
fn rccsd_t_matches_oracle_fast() {
    let checked = check_oracle(&["sto-3g", "6-31g"]);
    assert!(
        checked >= 4,
        "expected ≥4 fast oracle entries, checked {checked}"
    );
}

#[test]
#[ignore = "cc-pVDZ CCSD(T) is slow in debug; run with --release"]
fn rccsd_t_matches_oracle_ccpvdz() {
    let checked = check_oracle(&["cc-pvdz"]);
    assert!(
        checked >= 2,
        "expected ≥2 cc-pVDZ oracle entries, checked {checked}"
    );
}
