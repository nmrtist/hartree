mod dft_common;

use hartree::basis::BasisSet;
use hartree::core::Molecule;
use hartree::dft::cosx::{COSX_DEFAULT_GRID, CosxExchange, CosxProvider};
use hartree::dft::{FunctionalSpec, GridXc};
use hartree::integrals::{ConventionalProvider, IntegralProvider};
use hartree::linalg::{Mat, mat_from_row_major, mat_to_row_major};
use hartree::scf::{Reference, ScfOptions, ScfResult, XcContributor, run_scf, run_scf_with_xc};

fn water() -> Molecule {
    dft_common::geometries().molecules["water"].molecule()
}

fn charges_of(mol: &Molecule) -> Vec<([f64; 3], f64)> {
    mol.atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect()
}

fn water_rhf() -> (Molecule, ConventionalProvider, Vec<f64>, usize, ScfResult) {
    let mol = water();
    let ao = BasisSet::load("def2-svp").unwrap().build(&mol).unwrap();
    let nao = ao.n_ao();
    let provider = ConventionalProvider::new(ao.into_integral(), charges_of(&mol));
    let scf = run_scf(
        &provider,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
    )
    .unwrap();
    assert!(scf.converged);
    let s = mat_to_row_major(&provider.overlap());
    (mol, provider, s, nao, scf)
}

fn max_abs_diff(a: &Mat, b: &Mat) -> f64 {
    let (ar, br) = (mat_to_row_major(a), mat_to_row_major(b));
    ar.iter()
        .zip(&br)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

fn trace_dk(d: &[f64], k: &Mat) -> f64 {
    let kr = mat_to_row_major(k);
    d.iter().zip(&kr).map(|(x, y)| x * y).sum()
}

#[test]
fn unfitted_dense_grid_matches_exact_k() {
    let mol = water();
    let ao = BasisSet::load("def2-svp").unwrap().build(&mol).unwrap();
    let shells = ao.shells().to_vec();
    let nao = ao.n_ao();
    let provider = ConventionalProvider::new(ao.into_integral(), charges_of(&mol));
    let scf = run_scf(
        &provider,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
    )
    .unwrap();
    assert!(scf.converged);

    let d = mat_from_row_major(nao, &scf.density_alpha);
    let k_exact = provider
        .build_jk(std::slice::from_ref(&d))
        .exchange
        .remove(0);

    let cosx = CosxExchange::with_grid_level(&mol, &shells, nao, None, 3, "dense").unwrap();
    assert!(!cosx.fitted());
    let k_cosx = cosx
        .build_k(&provider, std::slice::from_ref(&d))
        .expect("in-core backend supplies grid_coulomb")
        .remove(0);

    let kr = mat_to_row_major(&k_cosx);
    for mu in 0..nao {
        for nu in 0..nao {
            assert_eq!(kr[mu * nao + nu], kr[nu * nao + mu], "K not symmetric");
        }
    }

    let dk = max_abs_diff(&k_cosx, &k_exact);
    let de = (trace_dk(&scf.density_alpha, &k_cosx) - trace_dk(&scf.density_alpha, &k_exact)).abs();
    println!("unfitted dense COSX: max|ΔK| = {dk:.3e}, |ΔE_x| = {de:.3e} Eh");
    assert!(dk <= 1e-5, "max |ΔK| = {dk:e} > 1e-5");
    assert!(de <= 1e-5, "exchange-energy error = {de:e} > 1e-5 Eh");
}

#[test]
fn fitted_default_grid_rhf_energy() {
    let (mol, provider, s, nao, scf_ref) = water_rhf();
    let ao = BasisSet::load("def2-svp").unwrap().build(&mol).unwrap();
    let cosx = CosxExchange::new(&mol, ao.shells(), nao, &s, COSX_DEFAULT_GRID).unwrap();
    assert!(cosx.fitted());
    let wrapped = CosxProvider::new(&provider, cosx).unwrap();
    let scf_cosx = run_scf(
        &wrapped,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
    )
    .unwrap();
    assert!(scf_cosx.converged);
    let de = (scf_cosx.energy - scf_ref.energy).abs();
    println!(
        "RHF/def2-SVP water: in-core {:.10} Eh, COSX {:.10} Eh, |ΔE| = {de:.3e} Eh",
        scf_ref.energy, scf_cosx.energy
    );
    assert!(de <= 5e-5, "RHF COSX energy error {de:e} > 5e-5 Eh");
}

#[test]
fn fitted_default_grid_pbe0_energy() {
    let mol = water();
    let ao = BasisSet::load("def2-svp").unwrap().build(&mol).unwrap();
    let shells = ao.shells().to_vec();
    let nao = ao.n_ao();
    let spec = FunctionalSpec::parse("pbe0").unwrap();
    let xc = GridXc::new(&mol, &ao, &spec, 3).unwrap();
    let provider = ConventionalProvider::new(ao.into_integral(), charges_of(&mol));
    let opts = ScfOptions {
        energy_tol: 1e-9,
        error_tol: 1e-6,
        ..ScfOptions::default()
    };
    let scf_ref = run_scf_with_xc(
        &provider,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &opts,
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(scf_ref.converged);

    let s = mat_to_row_major(&provider.overlap());
    let cosx = CosxExchange::new(&mol, &shells, nao, &s, COSX_DEFAULT_GRID).unwrap();
    let wrapped = CosxProvider::new(&provider, cosx).unwrap();
    let scf_cosx = run_scf_with_xc(
        &wrapped,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &opts,
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(scf_cosx.converged);
    let de = (scf_cosx.energy - scf_ref.energy).abs();
    println!(
        "PBE0/def2-SVP water: in-core {:.10} Eh, COSX {:.10} Eh, |ΔE| = {de:.3e} Eh",
        scf_ref.energy, scf_cosx.energy
    );
    assert!(de <= 5e-5, "PBE0 COSX energy error {de:e} > 5e-5 Eh");
}

#[test]
fn unfitted_dense_grid_matches_exact_k_lr() {
    const OMEGA: f64 = 0.3;
    let mol = water();
    let ao = BasisSet::load("def2-svp").unwrap().build(&mol).unwrap();
    let shells = ao.shells().to_vec();
    let nao = ao.n_ao();
    let provider = ConventionalProvider::new(ao.into_integral(), charges_of(&mol));
    let scf = run_scf(
        &provider,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
    )
    .unwrap();
    assert!(scf.converged);

    let d = mat_from_row_major(nao, &scf.density_alpha);
    let k_exact = provider
        .build_k_erf(std::slice::from_ref(&d), OMEGA)
        .expect("in-core backend supplies erf-attenuated K")
        .remove(0);

    let cosx = CosxExchange::with_grid_level(&mol, &shells, nao, None, 3, "dense").unwrap();
    let k_cosx = cosx
        .build_k_erf(&provider, std::slice::from_ref(&d), OMEGA)
        .expect("in-core backend supplies grid_coulomb_erf")
        .remove(0);

    let dk = max_abs_diff(&k_cosx, &k_exact);
    let de = (trace_dk(&scf.density_alpha, &k_cosx) - trace_dk(&scf.density_alpha, &k_exact)).abs();
    println!(
        "unfitted dense RS-COSX (omega = {OMEGA}): max|dK_LR| = {dk:.3e}, |dE_x| = {de:.3e} Eh"
    );
    assert!(dk <= 1e-5, "max |dK_LR| = {dk:e} > 1e-5");
    assert!(de <= 1e-5, "LR exchange-energy error = {de:e} > 1e-5 Eh");
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn fitted_default_grid_wb97xv_rs_energy() {
    let mol = water();
    let ao = BasisSet::load("def2-svp").unwrap().build(&mol).unwrap();
    let shells = ao.shells().to_vec();
    let nao = ao.n_ao();
    let spec = FunctionalSpec::parse("wb97x-v").unwrap();
    let cam = spec.cam().expect("wb97x-v is range-separated");
    let xc = GridXc::new(&mol, &ao, &spec, 3).unwrap();
    let provider = ConventionalProvider::new(ao.into_integral(), charges_of(&mol));
    let opts = ScfOptions {
        energy_tol: 1e-9,
        error_tol: 1e-6,
        ..ScfOptions::default()
    };
    let scf_ref = run_scf_with_xc(
        &provider,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &opts,
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(scf_ref.converged);

    let s = mat_to_row_major(&provider.overlap());
    let cosx = CosxExchange::new(&mol, &shells, nao, &s, COSX_DEFAULT_GRID).unwrap();

    let d = mat_from_row_major(nao, &scf_ref.density_alpha);
    let (kc, klr) = cosx
        .build_k_rs(&provider, std::slice::from_ref(&d), cam.omega)
        .unwrap();
    let kc_single = cosx.build_k(&provider, std::slice::from_ref(&d)).unwrap();
    let klr_single = cosx
        .build_k_erf(&provider, std::slice::from_ref(&d), cam.omega)
        .unwrap();
    assert_eq!(mat_to_row_major(&kc[0]), mat_to_row_major(&kc_single[0]));
    assert_eq!(mat_to_row_major(&klr[0]), mat_to_row_major(&klr_single[0]));

    let wrapped = CosxProvider::with_range_separation(&provider, cosx, cam.omega).unwrap();
    assert_eq!(wrapped.range_separation_omega(), Some(cam.omega));
    let scf_cosx = run_scf_with_xc(
        &wrapped,
        5,
        5,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &opts,
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(scf_cosx.converged);
    let de = (scf_cosx.energy - scf_ref.energy).abs();
    println!(
        "wb97x-v/def2-SVP water: in-core {:.10} Eh, RS-COSX {:.10} Eh, |dE| = {de:.3e} Eh",
        scf_ref.energy, scf_cosx.energy
    );
    assert!(de <= 5e-5, "RS-COSX wb97x-v energy error {de:e} > 5e-5 Eh");

    let scf_uks = run_scf_with_xc(
        &wrapped,
        5,
        5,
        Reference::Uhf,
        mol.nuclear_repulsion(),
        &opts,
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(scf_uks.converged);
    let de_uks = (scf_uks.energy - scf_cosx.energy).abs();
    assert!(
        de_uks <= 1e-6,
        "UKS RS-COSX vs RKS RS-COSX: |dE| = {de_uks:e} > 1e-6 Eh"
    );
}

#[test]
fn declining_backend_is_rejected() {
    struct NoGrid;
    impl IntegralProvider for NoGrid {
        fn n_basis(&self) -> usize {
            1
        }
        fn overlap(&self) -> Mat {
            unimplemented!()
        }
        fn kinetic(&self) -> Mat {
            unimplemented!()
        }
        fn nuclear(&self) -> Mat {
            unimplemented!()
        }
        fn build_jk(&self, _: &[Mat]) -> hartree::integrals::JkResult {
            unimplemented!()
        }
        fn dipole_integrals(&self, _: [f64; 3]) -> [Vec<f64>; 3] {
            unimplemented!()
        }
        fn ao_atom_indices(&self) -> Vec<usize> {
            unimplemented!()
        }
        fn charge_potential_3c(&self, _: &[([f64; 3], f64)]) -> Vec<f64> {
            unimplemented!()
        }
    }
    assert!(NoGrid.grid_coulomb(&[[0.0; 3]]).is_none());
    assert!(NoGrid.grid_coulomb_erf(&[[0.0; 3]], 0.3).is_none());

    let mol = water();
    let ao = BasisSet::load("sto-3g").unwrap().build(&mol).unwrap();
    let cosx =
        CosxExchange::with_grid_level(&mol, ao.shells(), ao.n_ao(), None, 0, "small").unwrap();
    let err = CosxProvider::new(&NoGrid, cosx)
        .err()
        .expect("must decline");
    assert!(err.to_string().contains("grid_coulomb"), "{err}");

    let cosx =
        CosxExchange::with_grid_level(&mol, ao.shells(), ao.n_ao(), None, 0, "small").unwrap();
    let err = CosxProvider::with_range_separation(&NoGrid, cosx, 0.3)
        .err()
        .expect("must decline");
    assert!(err.to_string().contains("grid_coulomb"), "{err}");
}
