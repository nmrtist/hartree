#![allow(dead_code)]

use std::collections::HashMap;

use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc};
use hartree::integrals::ConventionalProvider;
use hartree::scf::{Reference, ScfOptions, ScfResult, XcContributor, run_scf_with_xc};
use serde::Deserialize;

const GEOMETRIES_JSON: &str = include_str!("../../../../tests/ref/geometries.json");
const DFT_REFERENCES_JSON: &str = include_str!("../../../../tests/ref/dft_references.json");

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
pub struct DftReferences {
    pub provenance: Provenance,
    pub entries: Vec<DftRefEntry>,
}

#[derive(Deserialize)]
pub struct Provenance {
    pub engine: String,
    pub version: String,
}

#[derive(Deserialize)]
pub struct DftRefEntry {
    pub molecule: String,
    pub basis: String,
    pub functional: String,
    pub reference: String,
    pub multiplicity: u32,
    pub energy: f64,
    pub s2: Option<f64>,
}

pub fn geometries() -> Geometries {
    serde_json::from_str(GEOMETRIES_JSON).expect("parse geometries.json")
}

pub fn dft_references() -> DftReferences {
    serde_json::from_str(DFT_REFERENCES_JSON).expect("parse dft_references.json")
}

fn charges_of(mol: &Molecule) -> Vec<([f64; 3], f64)> {
    mol.atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect()
}

fn occ(mol: &Molecule) -> (usize, usize) {
    let n = mol.n_electrons() as usize;
    let two_s = (mol.multiplicity - 1) as usize;
    ((n + two_s) / 2, (n - two_s) / 2)
}

pub fn run_ks(geom: &GeomEntry, basis: &str, functional: &str, level: usize) -> ScfResult {
    run_ks_error_tol(geom, basis, functional, level, 1e-6)
}

pub fn run_ks_error_tol(
    geom: &GeomEntry,
    basis: &str,
    functional: &str,
    level: usize,
    error_tol: f64,
) -> ScfResult {
    let mol = geom.molecule();
    let ao = BasisSet::load(basis).unwrap().build(&mol).unwrap();
    let spec = FunctionalSpec::parse(functional).unwrap();
    let xc = GridXc::new(&mol, &ao, &spec, level).unwrap();
    let provider = ConventionalProvider::new(ao.into_integral(), charges_of(&mol));
    let (na, nb) = occ(&mol);
    let reference = if mol.multiplicity > 1 {
        Reference::Uhf
    } else {
        Reference::Rhf
    };
    let opts = ScfOptions {
        error_tol,
        energy_tol: 1e-9,
        ..ScfOptions::default()
    };
    run_scf_with_xc(
        &provider,
        na,
        nb,
        reference,
        mol.nuclear_repulsion(),
        &opts,
        Some(&xc as &dyn XcContributor),
    )
    .unwrap_or_else(|err| panic!("{}/{basis}/{functional}: {err}", geom_label(geom)))
}

fn geom_label(geom: &GeomEntry) -> String {
    geom.atoms
        .iter()
        .map(|(s, ..)| s.as_str())
        .collect::<Vec<_>>()
        .join("")
}
