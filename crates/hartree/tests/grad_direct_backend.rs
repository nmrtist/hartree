use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc, XcContributor};
use hartree::grad::{hf_gradient, ks_gradient};
use hartree::integrals::{ConventionalProvider, DirectProvider};
use hartree::scf::{Reference, ScfOptions, run_scf, run_scf_with_xc};

fn tight_options() -> ScfOptions {
    ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        max_iter: 256,
        ..ScfOptions::default()
    }
}

fn charges(mol: &Molecule) -> Vec<([f64; 3], f64)> {
    mol.atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect()
}

fn providers(mol: &Molecule, basis: &str) -> (ConventionalProvider, DirectProvider) {
    let build = || {
        BasisSet::load(basis)
            .unwrap()
            .build(mol)
            .unwrap()
            .into_integral()
    };
    (
        ConventionalProvider::new(build(), charges(mol)),
        DirectProvider::new(build(), charges(mol)),
    )
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.12]),
            Atom::new(Element::from_z(1).unwrap(), [0.0, 1.50, -0.95]),
            Atom::new(Element::from_z(1).unwrap(), [0.1, -1.38, -1.02]),
        ],
        0,
        1,
    )
}

fn methyl() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(Element::from_z(6).unwrap(), [0.0, 0.0, 0.05]),
            Atom::new(Element::from_z(1).unwrap(), [2.02, 0.0, -0.1]),
            Atom::new(Element::from_z(1).unwrap(), [-1.03, 1.77, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [-1.0, -1.75, 0.1]),
        ],
        0,
        2,
    )
}

fn max_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn rhf_gradient_matches_conventional() {
    let mol = water();
    let (conv, direct) = providers(&mol, "6-31g");
    let r = run_scf(
        &conv,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &tight_options(),
    )
    .unwrap();
    assert!(r.converged);
    let g_conv = hf_gradient(&conv, &mol, &r.density_alpha, &r.density_beta).unwrap();
    let g_direct = hf_gradient(&direct, &mol, &r.density_alpha, &r.density_beta).unwrap();
    assert!(g_conv.iter().flatten().any(|c| c.abs() > 1e-3));
    assert!(max_diff(&g_conv, &g_direct) < 1e-9);
}

#[test]
fn uhf_gradient_matches_conventional() {
    let mol = methyl();
    let (conv, direct) = providers(&mol, "6-31g");
    let r = run_scf(
        &conv,
        5,
        4,
        Reference::Uhf,
        mol.nuclear_repulsion(),
        &tight_options(),
    )
    .unwrap();
    assert!(r.converged);
    let g_conv = hf_gradient(&conv, &mol, &r.density_alpha, &r.density_beta).unwrap();
    let g_direct = hf_gradient(&direct, &mol, &r.density_alpha, &r.density_beta).unwrap();
    assert!(max_diff(&g_conv, &g_direct) < 1e-9);
}

#[test]
fn hybrid_ks_gradient_matches_conventional() {
    let mol = water();
    let (conv, direct) = providers(&mol, "6-31g");
    let ao = BasisSet::load("6-31g").unwrap().build(&mol).unwrap();
    let spec = FunctionalSpec::parse("b3lyp").unwrap();
    let xc = GridXc::new(&mol, &ao, &spec, 3).unwrap();
    let options = ScfOptions {
        energy_tol: 1e-10,
        error_tol: 1e-7,
        max_iter: 256,
        ..ScfOptions::default()
    };
    let r = run_scf_with_xc(
        &conv,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &options,
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(r.converged);
    let g_conv = ks_gradient(&conv, &mol, &xc, &r.density_alpha, &r.density_beta, true).unwrap();
    let g_direct =
        ks_gradient(&direct, &mol, &xc, &r.density_alpha, &r.density_beta, true).unwrap();
    assert!(max_diff(&g_conv, &g_direct) < 1e-9);
}
