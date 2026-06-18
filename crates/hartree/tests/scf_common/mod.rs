#![allow(dead_code)]

use std::collections::HashMap;

use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::{ConventionalProvider, DfProvider, DirectProvider};
use serde::Deserialize;

const GEOMETRIES_JSON: &str = include_str!("../../../../tests/ref/geometries.json");
const REFERENCES_JSON: &str = include_str!("../../../../tests/ref/scf_references.json");

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
pub struct References {
    pub provenance: Provenance,
    pub entries: Vec<RefEntry>,
}

#[derive(Deserialize)]
pub struct Provenance {
    pub engine: String,
    pub version: String,
}

#[derive(Deserialize)]
pub struct RefEntry {
    pub molecule: String,
    pub basis: String,
    pub method: String,
    pub energy: f64,
    pub s2: Option<f64>,
}

pub fn geometries() -> Geometries {
    serde_json::from_str(GEOMETRIES_JSON).expect("parse geometries.json")
}

pub fn references() -> References {
    serde_json::from_str(REFERENCES_JSON).expect("parse scf_references.json")
}

fn charges_of(mol: &Molecule) -> Vec<([f64; 3], f64)> {
    mol.atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect()
}

pub fn provider_for(mol: &Molecule, basis: &str) -> ConventionalProvider {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    ConventionalProvider::new(ao.into_integral(), charges_of(mol))
}

pub fn direct_provider_for(mol: &Molecule, basis: &str) -> DirectProvider {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    DirectProvider::new(ao.into_integral(), charges_of(mol))
}

pub fn df_provider_for(mol: &Molecule, basis: &str) -> DfProvider {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let aux = BasisSet::load_aux("def2-universal-jkfit")
        .unwrap()
        .build(mol)
        .unwrap()
        .into_integral();
    DfProvider::new(ao.into_integral(), &aux, charges_of(mol)).unwrap()
}

pub fn trace_ds(d: &[f64], s: &[f64], n: usize) -> f64 {
    let mut t = 0.0;
    for i in 0..n {
        for k in 0..n {
            t += d[i * n + k] * s[k * n + i];
        }
    }
    t
}
