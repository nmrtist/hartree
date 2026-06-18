use hartree::core::{Atom, Element, Molecule};
use hartree::disp::{D4Params, d4_energy, d4_energy_gradient, eeq_charges};
use serde::Deserialize;
use std::collections::HashMap;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");
const D4_JSON: &str = include_str!("../../../tests/ref/d4.json");

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

impl GeomEntry {
    fn molecule(&self) -> Molecule {
        let atoms = self
            .atoms
            .iter()
            .map(|(sym, x, y, z)| Atom::new(Element::from_symbol(sym).unwrap(), [*x, *y, *z]))
            .collect();
        Molecule::new(atoms, self.charge, self.multiplicity)
    }
}

#[derive(Deserialize)]
struct References {
    entries: Vec<RefEntry>,
}

#[derive(Deserialize)]
struct RefEntry {
    molecule: String,
    charge: i32,
    functional: String,
    energy: f64,
    gradient: Vec<[f64; 3]>,
    eeq_charges: Option<Vec<f64>>,
}

fn load() -> (Geometries, References) {
    (
        serde_json::from_str(GEOMETRIES_JSON).expect("geometries.json parses"),
        serde_json::from_str(D4_JSON).expect("d4.json parses"),
    )
}

#[test]
fn energies_match_oracle() {
    let (geoms, refs) = load();
    assert!(!refs.entries.is_empty());
    for entry in &refs.entries {
        let mol = geoms.molecules[&entry.molecule].molecule();
        assert_eq!(mol.charge, entry.charge, "{}", entry.molecule);
        let params = D4Params::for_method(&entry.functional)
            .unwrap_or_else(|| panic!("params for {}", entry.functional));
        let energy = d4_energy(&mol, &params);
        assert!(
            (energy - entry.energy).abs() <= 1e-9,
            "{} / {}: hartree {energy:.12e} vs oracle {:.12e}",
            entry.molecule,
            entry.functional,
            entry.energy
        );
    }
}

#[test]
fn gradients_match_oracle() {
    let (geoms, refs) = load();
    let mut checked = 0;
    for entry in &refs.entries {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let params = D4Params::for_method(&entry.functional).unwrap();
        let (energy, grad) = d4_energy_gradient(&mol, &params);
        assert!((energy - entry.energy).abs() <= 1e-9);
        for (iat, (g, gref)) in grad.iter().zip(&entry.gradient).enumerate() {
            for k in 0..3 {
                assert!(
                    (g[k] - gref[k]).abs() <= 1e-9,
                    "{} / {} atom {iat} component {k}: {:.3e} vs {:.3e}",
                    entry.molecule,
                    entry.functional,
                    g[k],
                    gref[k]
                );
            }
        }
        checked += 1;
    }
    assert!(checked >= 60, "expected oracle gradients for every entry");
}

#[test]
fn eeq_charges_match_oracle() {
    let (geoms, refs) = load();
    let mut checked = 0;
    let mut charged = 0;
    for entry in refs.entries.iter().filter(|e| e.eeq_charges.is_some()) {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let q = eeq_charges(&mol);
        let qref = entry.eeq_charges.as_ref().unwrap();
        let total: f64 = q.iter().sum();
        assert!(
            (total - entry.charge as f64).abs() <= 1e-10,
            "{}: EEQ charges sum to {total}, expected {}",
            entry.molecule,
            entry.charge
        );
        for (iat, (a, b)) in q.iter().zip(qref).enumerate() {
            assert!(
                (a - b).abs() <= 1e-9,
                "{} atom {iat}: q {a:.12e} vs oracle {b:.12e}",
                entry.molecule
            );
        }
        if entry.charge != 0 {
            charged += 1;
        }
        checked += 1;
    }
    assert!(checked >= 6, "expected EEQ charges for every system");
    assert!(charged >= 2, "expected charged species among EEQ checks");
}

#[test]
fn for_method_lookup() {
    for m in [
        "pbe", "blyp", "b3lyp", "pbe0", "tpss", "r2scan", "hf", "PBE", "B3LYP5",
    ] {
        assert!(
            D4Params::for_method(m).is_some(),
            "{m} should have D4 params"
        );
    }
    assert_eq!(
        D4Params::for_method("b3lyp5"),
        D4Params::for_method("b3lyp")
    );
    assert!(D4Params::for_method("svwn").is_none());
    assert!(D4Params::for_method("not-a-method").is_none());
}
