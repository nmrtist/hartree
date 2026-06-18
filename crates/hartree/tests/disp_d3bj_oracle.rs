use hartree::core::{Atom, Element, Molecule};
use hartree::disp::{D3Params, coordination_numbers, d3bj_energy, d3bj_energy_gradient};
use serde::Deserialize;
use std::collections::HashMap;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");
const D3BJ_JSON: &str = include_str!("../../../tests/ref/d3bj.json");

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
    functional: String,
    energy: f64,
    gradient: Option<Vec<[f64; 3]>>,
    cn: Option<Vec<f64>>,
}

fn load() -> (Geometries, References) {
    (
        serde_json::from_str(GEOMETRIES_JSON).expect("geometries.json parses"),
        serde_json::from_str(D3BJ_JSON).expect("d3bj.json parses"),
    )
}

#[test]
fn energies_match_oracle() {
    let (geoms, refs) = load();
    assert!(!refs.entries.is_empty());
    for entry in &refs.entries {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let params = D3Params::for_method(&entry.functional)
            .unwrap_or_else(|| panic!("params for {}", entry.functional));
        let energy = d3bj_energy(&mol, &params);
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
    for entry in refs.entries.iter().filter(|e| e.gradient.is_some()) {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let params = D3Params::for_method(&entry.functional).unwrap();
        let (energy, grad) = d3bj_energy_gradient(&mol, &params);
        assert!((energy - entry.energy).abs() <= 1e-9);
        for (iat, (g, gref)) in grad
            .iter()
            .zip(entry.gradient.as_ref().unwrap())
            .enumerate()
        {
            for k in 0..3 {
                assert!(
                    (g[k] - gref[k]).abs() <= 1e-9,
                    "{} atom {iat} component {k}: {:.3e} vs {:.3e}",
                    entry.molecule,
                    g[k],
                    gref[k]
                );
            }
        }
        checked += 1;
    }
    assert!(checked >= 5, "expected oracle gradients for every system");
}

#[test]
fn coordination_numbers_match_oracle() {
    let (geoms, refs) = load();
    let mut checked = 0;
    for entry in refs.entries.iter().filter(|e| e.cn.is_some()) {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let cn = coordination_numbers(&mol);
        for (a, b) in cn.iter().zip(entry.cn.as_ref().unwrap()) {
            assert!((a - b).abs() <= 1e-10, "{}: CN {a} vs {b}", entry.molecule);
        }
        checked += 1;
    }
    assert!(checked >= 5);
}

#[test]
fn for_method_lookup() {
    for m in ["pbe", "blyp", "b3lyp", "pbe0", "hf", "PBE", "B3LYP5"] {
        assert!(
            D3Params::for_method(m).is_some(),
            "{m} should have D3(BJ) params"
        );
    }
    assert_eq!(
        D3Params::for_method("b3lyp5"),
        D3Params::for_method("b3lyp")
    );
    assert!(D3Params::for_method("svwn").is_none());
    assert!(D3Params::for_method("not-a-method").is_none());
}
