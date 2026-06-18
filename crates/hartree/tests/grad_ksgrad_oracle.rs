use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc, XcContributor};
use hartree::grad::ks_gradient;
use hartree::integrals::ConventionalProvider;
use hartree::scf::{Reference, ScfOptions, run_scf_with_xc};
use serde::Deserialize;
use std::collections::HashMap;

const REFS_JSON: &str = include_str!("../../../tests/ref/ksgrad_references.json");
const GEOMS_JSON: &str = include_str!("../../../tests/ref/geometries.json");

const GATE: f64 = 4.0e-6;

#[derive(Deserialize)]
struct References {
    provenance: Provenance,
    entries: Vec<RefEntry>,
}

#[derive(Deserialize)]
struct Provenance {
    engine: String,
}

#[derive(Deserialize)]
struct RefEntry {
    molecule: String,
    basis: String,
    functional: String,
    reference: String,
    multiplicity: u32,
    charge: i32,
    gradient: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
struct Geometries {
    molecules: HashMap<String, Geometry>,
}

#[derive(Deserialize)]
struct Geometry {
    charge: i32,
    multiplicity: u32,
    atoms: Vec<(String, f64, f64, f64)>,
}

fn molecule(geom: &Geometry) -> Molecule {
    let atoms = geom
        .atoms
        .iter()
        .map(|(sym, x, y, z)| Atom::new(Element::from_symbol(sym).unwrap(), [*x, *y, *z]))
        .collect();
    Molecule::new(atoms, geom.charge, geom.multiplicity)
}

fn check_subset(want_6_31g: bool) -> usize {
    let refs: References = serde_json::from_str(REFS_JSON).expect("parse ksgrad_references.json");
    assert_eq!(refs.provenance.engine, "PySCF");
    let geoms: Geometries = serde_json::from_str(GEOMS_JSON).expect("parse geometries.json");

    let mut checked = 0;
    let mut worst = 0.0_f64;
    let mut over: Vec<String> = Vec::new();
    for entry in &refs.entries {
        if (entry.basis == "6-31g") != want_6_31g {
            continue;
        }
        let geom = &geoms.molecules[&entry.molecule];
        assert_eq!(geom.charge, entry.charge);
        assert_eq!(geom.multiplicity, entry.multiplicity);
        let mol = molecule(geom);
        let n_elec = mol.n_electrons() as usize;
        let two_s = (mol.multiplicity - 1) as usize;
        let (na, nb) = ((n_elec + two_s) / 2, (n_elec - two_s) / 2);
        let reference = if entry.reference == "uks" {
            Reference::Uhf
        } else {
            Reference::Rhf
        };

        let ao = BasisSet::load(&entry.basis).unwrap().build(&mol).unwrap();
        let ao2 = BasisSet::load(&entry.basis).unwrap().build(&mol).unwrap();
        let charges: Vec<([f64; 3], f64)> = mol
            .atoms
            .iter()
            .map(|a| (a.position, a.element.z() as f64))
            .collect();
        let provider = ConventionalProvider::new(ao.into_integral(), charges);
        let spec = FunctionalSpec::parse(&entry.functional).unwrap();
        let xc = GridXc::new(&mol, &ao2, &spec, 3).unwrap();
        let opts = ScfOptions {
            energy_tol: 1e-11,
            error_tol: 1e-7,
            max_iter: 512,
            level_shift: if reference == Reference::Uhf {
                0.3
            } else {
                0.0
            },
            ..ScfOptions::default()
        };
        let r = run_scf_with_xc(
            &provider,
            na,
            nb,
            reference,
            mol.nuclear_repulsion(),
            &opts,
            Some(&xc as &dyn XcContributor),
        )
        .unwrap();
        assert!(
            r.converged,
            "{}/{}/{} did not converge",
            entry.molecule, entry.basis, entry.functional
        );
        let g = ks_gradient(
            &provider,
            &mol,
            &xc as &dyn XcContributor,
            &r.density_alpha,
            &r.density_beta,
            reference == Reference::Rhf,
        )
        .unwrap();

        let mut delta = 0.0_f64;
        for (ga, gr) in g.iter().zip(&entry.gradient) {
            for k in 0..3 {
                delta = delta.max((ga[k] - gr[k]).abs());
            }
        }
        worst = worst.max(delta);
        eprintln!(
            "  {:8}/{:9}/{:6} [{}] max comp delta = {delta:.2e} (gate {GATE:.1e})",
            entry.molecule, entry.basis, entry.functional, entry.reference
        );
        if delta >= GATE {
            over.push(format!(
                "{}/{}/{}: delta = {delta:.2e} >= gate {GATE:.1e}",
                entry.molecule, entry.basis, entry.functional
            ));
        }
        checked += 1;
    }
    eprintln!("{checked} KS gradient references matched PySCF; worst delta = {worst:.2e}");
    assert!(
        over.is_empty(),
        "{} entries over gate:\n  {}",
        over.len(),
        over.join("\n  ")
    );
    checked
}

#[test]
fn ksgrad_fast_subset_matches_pyscf() {
    let checked = check_subset(true);
    assert!(checked >= 3, "expected >=3 fast references, got {checked}");
}

#[test]
#[ignore = "def2-SVP tier; run with --release -- --ignored"]
fn ksgrad_full_set_matches_pyscf() {
    let checked = check_subset(false);
    assert!(
        checked >= 3,
        "expected >=3 def2-svp references, got {checked}"
    );
}
