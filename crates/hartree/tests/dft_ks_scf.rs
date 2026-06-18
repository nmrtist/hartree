use hartree::basis::{AoBasis, BasisSet};
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc};
use hartree::integrals::{ConventionalProvider, DirectProvider};
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

fn oh() -> Molecule {
    Molecule::new(
        vec![atom("O", [0.0, 0.0, 0.0]), atom("H", [0.0, 0.0, 1.8344])],
        0,
        2,
    )
}

fn charges(mol: &Molecule) -> Vec<([f64; 3], f64)> {
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

fn setup(mol: &Molecule, basis: &str, func: &str, level: usize) -> (AoBasis, GridXc) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let spec = FunctionalSpec::parse(func).unwrap();
    let xc = GridXc::new(mol, &ao, &spec, level).unwrap();
    (ao, xc)
}

#[test]
fn rks_water_pbe_converges() {
    let mol = water();
    let (ao, xc) = setup(&mol, "6-31g", "pbe", 3);
    let provider = ConventionalProvider::new(ao.into_integral(), charges(&mol));
    let (na, nb) = occ(&mol);

    let run = || {
        run_scf_with_xc(
            &provider,
            na,
            nb,
            Reference::Rhf,
            mol.nuclear_repulsion(),
            &ScfOptions::default(),
            Some(&xc as &dyn XcContributor),
        )
        .unwrap()
    };
    let r = run();
    assert!(r.converged, "RKS did not converge");
    assert!(r.iterations < 40, "RKS took {} iterations", r.iterations);
    assert!(r.xc_energy.unwrap() < 0.0, "E_xc should be negative");
    assert!(
        (r.n_elec_grid.unwrap() - 10.0).abs() < 1e-4,
        "∫ρ = {}",
        r.n_elec_grid.unwrap()
    );
    println!(
        "RKS water/pbe/6-31g L3: E = {:.10}  E_xc = {:.10}  iters = {}",
        r.energy,
        r.xc_energy.unwrap(),
        r.iterations
    );
    let r2 = run();
    assert!(
        (r.energy - r2.energy).abs() < 1e-12,
        "energy not reproducible"
    );
}

#[test]
fn uks_oh_pbe_converges() {
    let mol = oh();
    let (ao, xc) = setup(&mol, "6-31g", "pbe", 3);
    let provider = ConventionalProvider::new(ao.into_integral(), charges(&mol));
    let (na, nb) = occ(&mol);
    let opts = ScfOptions {
        error_tol: 1e-6,
        ..ScfOptions::default()
    };
    let r = run_scf_with_xc(
        &provider,
        na,
        nb,
        Reference::Uhf,
        mol.nuclear_repulsion(),
        &opts,
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(r.converged, "UKS did not converge");
    assert!(r.iterations < 60, "UKS took {} iterations", r.iterations);
    assert!(
        (r.n_elec_grid.unwrap() - 9.0).abs() < 1e-4,
        "∫ρ = {}",
        r.n_elec_grid.unwrap()
    );
    println!(
        "UKS OH/pbe/6-31g L3: E = {:.10}  <S^2> = {:.4}  iters = {}",
        r.energy, r.spin_squared, r.iterations
    );
}

#[test]
fn rks_water_b3lyp_smoke() {
    let mol = water();
    let (ao, xc) = setup(&mol, "6-31g", "b3lyp", 3);
    assert!((xc.exx_fraction() - 0.20).abs() < 1e-12);
    let provider = ConventionalProvider::new(ao.into_integral(), charges(&mol));
    let (na, nb) = occ(&mol);
    let r = run_scf_with_xc(
        &provider,
        na,
        nb,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(r.converged, "b3lyp RKS did not converge");
    println!("RKS water/b3lyp/6-31g L3: E = {:.10}", r.energy);
}

#[test]
fn rks_equals_uks_closed_shell() {
    let mol = water();
    let (ao, xc) = setup(&mol, "6-31g", "pbe", 3);
    let provider = ConventionalProvider::new(ao.into_integral(), charges(&mol));
    let (na, nb) = occ(&mol);

    let rks = run_scf_with_xc(
        &provider,
        na,
        nb,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    let uks = run_scf_with_xc(
        &provider,
        na,
        nb,
        Reference::Uhf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(rks.converged && uks.converged);
    assert!(
        (rks.energy - uks.energy).abs() < 1e-9,
        "RKS {} vs UKS {}",
        rks.energy,
        uks.energy
    );
}

#[test]
fn rks_mgga_converges_and_equals_uks() {
    let mol = water();
    for func in ["tpss", "r2scan"] {
        let (ao, xc) = setup(&mol, "6-31g", func, 3);
        assert_eq!(xc.exx_fraction(), 0.0, "{func} is pure");
        let provider = ConventionalProvider::new(ao.into_integral(), charges(&mol));
        let (na, nb) = occ(&mol);
        let run = |reference| {
            run_scf_with_xc(
                &provider,
                na,
                nb,
                reference,
                mol.nuclear_repulsion(),
                &ScfOptions::default(),
                Some(&xc as &dyn XcContributor),
            )
            .unwrap()
        };
        let rks = run(Reference::Rhf);
        assert!(rks.converged, "{func}: RKS did not converge");
        assert!(
            (rks.n_elec_grid.unwrap() - 10.0).abs() < 1e-4,
            "{func}: ∫ρ = {}",
            rks.n_elec_grid.unwrap()
        );
        let uks = run(Reference::Uhf);
        assert!(uks.converged, "{func}: UKS did not converge");
        println!(
            "RKS water/{func}/6-31g L3: E = {:.10}  (UKS Δ = {:+.2e})",
            rks.energy,
            rks.energy - uks.energy
        );
        assert!(
            (rks.energy - uks.energy).abs() < 1e-9,
            "{func}: RKS {} vs UKS {}",
            rks.energy,
            uks.energy
        );
    }
}

#[test]
fn uks_oh_mgga_converges() {
    let mol = oh();
    for func in ["tpss", "r2scan"] {
        let (ao, xc) = setup(&mol, "6-31g", func, 3);
        let provider = ConventionalProvider::new(ao.into_integral(), charges(&mol));
        let (na, nb) = occ(&mol);
        let opts = ScfOptions {
            error_tol: 1e-6,
            ..ScfOptions::default()
        };
        let r = run_scf_with_xc(
            &provider,
            na,
            nb,
            Reference::Uhf,
            mol.nuclear_repulsion(),
            &opts,
            Some(&xc as &dyn XcContributor),
        )
        .unwrap();
        assert!(r.converged, "{func}: UKS did not converge");
        assert!(
            (r.n_elec_grid.unwrap() - 9.0).abs() < 1e-4,
            "{func}: ∫ρ = {}",
            r.n_elec_grid.unwrap()
        );
        println!(
            "UKS OH/{func}/6-31g L3: E = {:.10}  <S^2> = {:.4}  iters = {}",
            r.energy, r.spin_squared, r.iterations
        );
    }
}

#[test]
fn conventional_equals_direct_pbe() {
    let mol = water();
    let spec = FunctionalSpec::parse("pbe").unwrap();
    let (na, nb) = occ(&mol);

    let ao1 = BasisSet::load("6-31g").unwrap().build(&mol).unwrap();
    let xc1 = GridXc::new(&mol, &ao1, &spec, 3).unwrap();
    let conv = ConventionalProvider::new(ao1.into_integral(), charges(&mol));
    let e_conv = run_scf_with_xc(
        &conv,
        na,
        nb,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
        Some(&xc1 as &dyn XcContributor),
    )
    .unwrap()
    .energy;

    let ao2 = BasisSet::load("6-31g").unwrap().build(&mol).unwrap();
    let xc2 = GridXc::new(&mol, &ao2, &spec, 3).unwrap();
    let direct = DirectProvider::new(ao2.into_integral(), charges(&mol));
    let e_direct = run_scf_with_xc(
        &direct,
        na,
        nb,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
        Some(&xc2 as &dyn XcContributor),
    )
    .unwrap()
    .energy;

    assert!(
        (e_conv - e_direct).abs() < 1e-9,
        "conv {e_conv} vs direct {e_direct}"
    );
}

#[test]
fn incremental_equals_full_build_pbe() {
    let mol = water();
    let (ao, xc) = setup(&mol, "6-31g", "pbe", 3);
    let provider = ConventionalProvider::new(ao.into_integral(), charges(&mol));
    let (na, nb) = occ(&mol);

    let full = run_scf_with_xc(
        &provider,
        na,
        nb,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions {
            incremental_fock: false,
            ..ScfOptions::default()
        },
        Some(&xc as &dyn XcContributor),
    )
    .unwrap()
    .energy;
    let incr = run_scf_with_xc(
        &provider,
        na,
        nb,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions {
            incremental_fock: true,
            ..ScfOptions::default()
        },
        Some(&xc as &dyn XcContributor),
    )
    .unwrap()
    .energy;
    assert!(
        (full - incr).abs() < 1e-9,
        "full {full} vs incremental {incr}"
    );
}

#[test]
fn rohf_kohn_sham_rejected() {
    let mol = water();
    let (ao, xc) = setup(&mol, "sto-3g", "pbe", 1);
    let provider = ConventionalProvider::new(ao.into_integral(), charges(&mol));
    let (na, nb) = occ(&mol);
    let err = run_scf_with_xc(
        &provider,
        na,
        nb,
        Reference::Rohf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
        Some(&xc as &dyn XcContributor),
    );
    assert!(err.is_err(), "ROHF×xc should be rejected");
}
