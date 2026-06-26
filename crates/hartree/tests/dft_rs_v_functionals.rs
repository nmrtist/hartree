use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc};
use hartree::integrals::ConventionalProvider;
use hartree::scf::{Reference, ScfOptions, ScfResult, XcContributor, run_scf_with_xc};

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(
                Element::from_symbol("O").unwrap(),
                [0.0, -0.143225816552, 0.0],
            ),
            Atom::new(
                Element::from_symbol("H").unwrap(),
                [1.638036840407, 1.136548822547, 0.0],
            ),
            Atom::new(
                Element::from_symbol("H").unwrap(),
                [-1.638036840407, 1.136548822547, 0.0],
            ),
        ],
        0,
        1,
    )
}

fn run(func: &str, reference: Reference) -> ScfResult {
    let mol = water();
    let ao = BasisSet::load("def2-svp").unwrap().build(&mol).unwrap();
    let spec = FunctionalSpec::parse(func).unwrap();
    let xc = GridXc::new(&mol, &ao, &spec, 3).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let provider = ConventionalProvider::new(ao.into_integral(), charges);
    let opts = ScfOptions {
        energy_tol: 1e-9,
        error_tol: 1e-6,
        ..ScfOptions::default()
    };
    run_scf_with_xc(
        &provider,
        5,
        5,
        reference,
        mol.nuclear_repulsion(),
        &opts,
        Some(&xc as &dyn XcContributor),
    )
    .unwrap()
}

// Self-pinned hartree RKS energies (water/def2-svp/grid 3) on the default
// point-efficient pruned grid (a moderated Treutler–Ahlrichs radial set with the
// full five-zone angular pruning); the grid-sensitive m06-2x meta-GGA is the loosest.
const FIXTURES: &[(&str, f64)] = &[
    ("m06-2x", -76.295766206199),
    ("pw6b95", -76.409242001923),
    ("b97m-v", -76.344774687374),
    ("wb97x-v", -76.345153292028),
    ("wb97m-v", -76.341760482814),
];

#[test]
fn rks_water_def2svp_regression_fixtures() {
    for &(func, e_ref) in FIXTURES {
        let r = run(func, Reference::Rhf);
        assert!(r.converged, "{func}: RKS did not converge");
        assert!(
            (r.n_elec_grid.unwrap() - 10.0).abs() < 1e-3,
            "{func}: grid electrons {}",
            r.n_elec_grid.unwrap()
        );
        println!("{func}: RKS E = {:.12}  ({} iters)", r.energy, r.iterations);
        assert!(
            (r.energy - e_ref).abs() < 1e-8,
            "{func}: E = {:.12} vs fixture {:.12}",
            r.energy,
            e_ref
        );
    }
}

#[test]
fn uks_equals_rks_closed_shell() {
    for &(func, e_ref) in FIXTURES {
        let r = run(func, Reference::Uhf);
        assert!(r.converged, "{func}: UKS did not converge");
        assert!(
            (r.energy - e_ref).abs() < 1e-7,
            "{func}: UKS E = {:.12} vs RKS fixture {:.12}",
            r.energy,
            e_ref
        );
    }
}
