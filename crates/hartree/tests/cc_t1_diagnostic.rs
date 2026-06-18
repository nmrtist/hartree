use hartree::basis::BasisSet;
use hartree::cc::{CcsdOptions, rccsd_spin_adapted, rccsd_spin_orbital};
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::ConventionalProvider;
use hartree::scf::{ScfOptions, ScfResult, run_rhf};
use serde::Deserialize;

use std::collections::HashMap;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");
const T1_REFERENCES_JSON: &str = include_str!("../../../tests/ref/t1_references.json");

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
struct T1References {
    entries: Vec<T1RefEntry>,
}

#[derive(Deserialize)]
struct T1RefEntry {
    molecule: String,
    basis: String,
    n_frozen: usize,
    correlation_energy: f64,
    t1_diagnostic: f64,
}

fn setup(molecule: &str, basis: &str) -> (ConventionalProvider, ScfResult, Molecule) {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let g = &geoms.molecules[molecule];
    let atoms = g
        .atoms
        .iter()
        .map(|(s, x, y, z)| Atom::new(Element::from_symbol(s).unwrap(), [*x, *y, *z]))
        .collect();
    let mol = Molecule::new(atoms, g.charge, g.multiplicity);
    let ao = BasisSet::load(basis).unwrap().build(&mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let provider = ConventionalProvider::new(ao.into_integral(), charges);
    let opts = ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        ..ScfOptions::default()
    };
    let scf = run_rhf(
        &provider,
        mol.n_electrons() as usize,
        mol.nuclear_repulsion(),
        &opts,
    )
    .unwrap();
    assert!(scf.converged);
    (provider, scf, mol)
}

#[test]
fn t1_diagnostic_matches_pyscf() {
    let refs: T1References = serde_json::from_str(T1_REFERENCES_JSON).unwrap();
    let fast = ["6-31g"];
    let mut checked = 0;
    for entry in refs
        .entries
        .iter()
        .filter(|e| fast.contains(&e.basis.as_str()))
    {
        let (provider, scf, _) = setup(&entry.molecule, &entry.basis);
        let cc = rccsd_spin_adapted(&provider, &scf, entry.n_frozen, &CcsdOptions::default());
        assert!(cc.converged, "{}/{} CCSD", entry.molecule, entry.basis);
        let de = cc.correlation_energy - entry.correlation_energy;
        assert!(
            de.abs() < 1e-7,
            "{}/{} E_corr: hartree {:.10} vs PySCF {:.10}",
            entry.molecule,
            entry.basis,
            cc.correlation_energy,
            entry.correlation_energy
        );
        let dt = cc.t1_diagnostic - entry.t1_diagnostic;
        assert!(
            dt.abs() < 1e-8,
            "{}/{} (frozen {}): T1 hartree {:.10} vs PySCF {:.10} (Δ = {:.2e})",
            entry.molecule,
            entry.basis,
            entry.n_frozen,
            cc.t1_diagnostic,
            entry.t1_diagnostic,
            dt
        );
        checked += 1;
    }
    assert_eq!(checked, 2, "expected 2 fast-tier T1 references");
}

#[test]
fn t1_diagnostic_is_convention_invariant() {
    let (provider, scf, _) = setup("water", "sto-3g");
    let opts = CcsdOptions::default();
    let sa = rccsd_spin_adapted(&provider, &scf, 1, &opts);
    let so = rccsd_spin_orbital(&provider, &scf, 1, &opts);
    assert!(sa.converged && so.converged);
    assert!(sa.t1_diagnostic >= 0.0);
    assert!(
        (sa.t1_diagnostic - so.t1_diagnostic).abs() < 1e-9,
        "spin-adapted T1 {} vs spin-orbital T1 {}",
        sa.t1_diagnostic,
        so.t1_diagnostic
    );
    assert!(sa.t1_diagnostic > 0.0 && sa.t1_diagnostic < 0.02);
}
