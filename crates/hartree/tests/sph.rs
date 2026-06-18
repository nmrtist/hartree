use hartree::core::Molecule;
use hartree::optimize_geometry;
use hartree::scf::Reference;
use hartree::{Job, JobOptions, Method};

fn water_start() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n")
        .unwrap()
}

fn freq_cm1(mol: &Molecule, sph: bool) -> (Vec<f64>, usize) {
    let result = Job {
        molecule: mol.clone(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            compute_frequencies: true,
            single_point_hessian: sph,
            symmetry_number: 2,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    let f = &result.frequencies.as_ref().unwrap().frequencies;
    (f.frequencies_cm1.clone(), f.n_imaginary)
}

#[test]
#[ignore = "slow (opt + two Hessians); run with --include-ignored"]
fn sph_at_minimum_matches_ordinary() {
    let opt = optimize_geometry(
        &water_start(),
        "sto-3g",
        Reference::Rhf,
        &Default::default(),
    )
    .unwrap();
    assert!(
        opt.converged,
        "water RHF/STO-3G optimization did not converge"
    );
    let mut mol = water_start();
    for (atom, pos) in mol.atoms.iter_mut().zip(&opt.positions) {
        atom.position = *pos;
    }

    let (ord, ord_imag) = freq_cm1(&mol, false);
    let (sph, sph_imag) = freq_cm1(&mol, true);
    assert_eq!(ord_imag, 0);
    assert_eq!(sph_imag, 0);
    assert_eq!(ord.len(), sph.len());
    for (a, b) in ord.iter().zip(&sph) {
        assert!(
            (a - b).abs() < 1.0,
            "SPH {b:.3} vs ordinary {a:.3} cm⁻¹ at the minimum"
        );
    }
}

#[test]
#[ignore = "slow (Hessian); run with --include-ignored"]
fn sph_at_displaced_geometry_no_spurious_imaginary() {
    let displaced = Molecule::from_xyz(
        "3\ndisplaced water\nO 0.0 0.0 0.117\nH 0.0 0.857 -0.520\nH 0.0 -0.757 -0.470\n",
    )
    .unwrap();
    let (sph, sph_imag) = freq_cm1(&displaced, true);
    assert!(sph.iter().all(|f| f.is_finite()), "NaN/inf: {sph:?}");
    assert_eq!(
        sph_imag, 0,
        "SPH produced spurious imaginary modes: {sph:?}"
    );
    let n_real = sph.iter().filter(|&&f| f >= 10.0).count();
    assert_eq!(
        n_real, 2,
        "expected 2 real modes after SPH gradient projection: {sph:?}"
    );
}
