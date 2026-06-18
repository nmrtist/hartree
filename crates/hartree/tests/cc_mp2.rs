use hartree::basis::BasisSet;
use hartree::cc::{frozen_core_orbitals, rhf_mp2};
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::ConventionalProvider;
use hartree::scf::{ScfOptions, run_rhf};
use serde::Deserialize;

use std::collections::HashMap;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");
const MP2_REFERENCES_JSON: &str = include_str!("../../../tests/ref/mp2_references.json");

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
struct Mp2References {
    entries: Vec<Mp2RefEntry>,
}

#[derive(Deserialize)]
struct Mp2RefEntry {
    molecule: String,
    basis: String,
    frozen_core: bool,
    mp2_correlation: f64,
    mp2_total_energy: f64,
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

#[test]
fn frozen_core_convention_is_pinned() {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    assert_eq!(
        frozen_core_orbitals(&molecule(&geoms.molecules["water"])),
        1
    );
    assert_eq!(frozen_core_orbitals(&molecule(&geoms.molecules["h2"])), 0);
}

#[test]
fn rhf_mp2_matches_orca_references() {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let refs: Mp2References = serde_json::from_str(MP2_REFERENCES_JSON).unwrap();

    let opts = ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        ..ScfOptions::default()
    };

    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in &refs.entries {
        let mol = molecule(&geoms.molecules[&entry.molecule]);
        let provider = provider_for(&mol, &entry.basis);
        let n_elec = mol.n_electrons() as usize;
        let scf = run_rhf(&provider, n_elec, mol.nuclear_repulsion(), &opts).unwrap();
        assert!(scf.converged, "{}/{} SCF", entry.molecule, entry.basis);

        let n_frozen = if entry.frozen_core {
            frozen_core_orbitals(&mol)
        } else {
            0
        };
        let mp2 = rhf_mp2(&provider, &scf, n_frozen);

        let d_corr = mp2.correlation_energy - entry.mp2_correlation;
        let d_total = mp2.total_energy - entry.mp2_total_energy;
        worst = worst.max(d_corr.abs());
        let fc = if entry.frozen_core { "fc" } else { "ae" };
        eprintln!(
            "MP2 {}/{}/{fc}: Ecorr {:.10} (Δ {:.2e}), Etot Δ {:.2e}  [OS {:.6} SS {:.6}]",
            entry.molecule,
            entry.basis,
            mp2.correlation_energy,
            d_corr,
            d_total,
            mp2.opposite_spin,
            mp2.same_spin
        );

        assert!(
            d_corr.abs() < 1e-6,
            "MP2 {}/{}/{fc} correlation: hartree {:.10} vs ORCA {:.10} (Δ {:.2e})",
            entry.molecule,
            entry.basis,
            mp2.correlation_energy,
            entry.mp2_correlation,
            d_corr
        );
        assert!(
            d_total.abs() < 1e-6,
            "MP2 {}/{}/{fc} total: hartree {:.10} vs ORCA {:.10} (Δ {:.2e})",
            entry.molecule,
            entry.basis,
            mp2.total_energy,
            entry.mp2_total_energy,
            d_total
        );
        checked += 1;
    }
    eprintln!("MP2: {checked} references matched ORCA; worst ΔEcorr = {worst:.2e}");
    assert!(
        checked >= 12,
        "expected ≥12 MP2 references, checked {checked}"
    );
}
