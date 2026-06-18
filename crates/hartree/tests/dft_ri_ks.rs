use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc};
use hartree::integrals::{ConventionalProvider, DfProvider};
use hartree::scf::{Reference, ScfOptions, XcContributor, run_scf_with_xc};

fn atom(sym: &str, pos: [f64; 3]) -> Atom {
    Atom::new(Element::from_symbol(sym).unwrap(), pos)
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            atom("O", [0.0, -0.143225816552, 0.0]),
            atom("H", [1.638036840407, 1.136548822547, 0.0]),
            atom("H", [-1.638036840407, 1.136548822547, 0.0]),
        ],
        0,
        1,
    )
}

fn charges(mol: &Molecule) -> Vec<([f64; 3], f64)> {
    mol.atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect()
}

#[test]
fn ri_pbe0_matches_conventional_to_fitting_error() {
    let mol = water();
    let basis = "6-31g";
    let ao = BasisSet::load(basis).unwrap().build(&mol).unwrap();
    let spec = FunctionalSpec::parse("pbe0").unwrap();
    let xc = GridXc::new(&mol, &ao, &spec, 3).unwrap();
    let aux = BasisSet::load_aux("def2-universal-jkfit")
        .unwrap()
        .build(&mol)
        .unwrap()
        .into_integral();

    let n_elec = mol.n_electrons() as usize;
    let opts = ScfOptions::default();
    macro_rules! run {
        ($provider:expr) => {
            run_scf_with_xc(
                $provider,
                n_elec / 2,
                n_elec / 2,
                Reference::Rhf,
                mol.nuclear_repulsion(),
                &opts,
                Some(&xc as &dyn XcContributor),
            )
            .unwrap()
        };
    }

    let conv = ConventionalProvider::new(ao.integral().clone(), charges(&mol));
    let r_conv = run!(&conv);
    let df = DfProvider::new(ao.integral().clone(), &aux, charges(&mol)).unwrap();
    let r_df = run!(&df);
    assert!(r_conv.converged && r_df.converged);

    let delta = (r_df.energy - r_conv.energy).abs();
    eprintln!(
        "RI-PBE0 water/{basis}: E_df = {:.10}, E_conv = {:.10}, fitting error = {delta:.2e}",
        r_df.energy, r_conv.energy
    );
    assert!(
        delta < 1e-4,
        "RI-PBE0 fitting error {delta:.2e} exceeds 1e-4 Eh"
    );
    assert!(
        delta > 1e-9,
        "RI-PBE0 ΔE = {delta:.2e} is implausibly small for a fitted run"
    );
}
