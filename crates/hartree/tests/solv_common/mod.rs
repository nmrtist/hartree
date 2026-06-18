#![allow(dead_code)]

use std::collections::HashMap;

use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::ConventionalProvider;
use serde::Deserialize;

const GEOMETRIES_JSON: &str = include_str!("../../../../tests/ref/geometries.json");
const CPCM_REFS_JSON: &str = include_str!("../../../../tests/ref/cpcm_references.json");

#[derive(Deserialize)]
pub struct Geometries {
    pub molecules: HashMap<String, GeomEntry>,
}

#[derive(Deserialize)]
pub struct GeomEntry {
    pub charge: i32,
    pub multiplicity: u32,
    pub atoms: Vec<(String, f64, f64, f64)>,
}

impl GeomEntry {
    pub fn molecule(&self) -> Molecule {
        let atoms = self
            .atoms
            .iter()
            .map(|(sym, x, y, z)| Atom::new(Element::from_symbol(sym).unwrap(), [*x, *y, *z]))
            .collect();
        Molecule::new(atoms, self.charge, self.multiplicity)
    }
}

#[derive(Deserialize)]
pub struct CpcmRefs {
    pub entries: Vec<CpcmEntry>,
}

#[derive(Deserialize)]
pub struct CpcmEntry {
    pub molecule: String,
    pub basis: String,
    pub method: String,
    pub reference: String,
    pub eps: f64,
    pub energy: f64,
    pub e_solv: f64,
}

pub fn geometries() -> Geometries {
    serde_json::from_str(GEOMETRIES_JSON).expect("parse geometries.json")
}

pub fn cpcm_refs() -> CpcmRefs {
    serde_json::from_str(CPCM_REFS_JSON).expect("parse cpcm_references.json")
}

pub fn provider_for(mol: &Molecule, basis: &str) -> ConventionalProvider {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    ConventionalProvider::new(ao.into_integral(), charges)
}
