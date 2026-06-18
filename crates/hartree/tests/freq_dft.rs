use hartree::core::{Atom, Element, Molecule};
use hartree::dft::FunctionalSpec;
use hartree::{Job, JobOptions, Method};

fn mol(atoms: &[(u32, [f64; 3])], charge: i32, multiplicity: u32) -> Molecule {
    Molecule::new(
        atoms
            .iter()
            .map(|&(z, p)| Atom::new(Element::from_z(z).unwrap(), p))
            .collect(),
        charge,
        multiplicity,
    )
}

fn check_freqs(result: &hartree::JobResult, n_tr: usize, pinned: &[f64], tol: f64) {
    let freq = &result
        .frequencies
        .as_ref()
        .expect("frequencies")
        .frequencies;
    let f = &freq.frequencies_cm1;
    assert!(f.iter().all(|v| v.is_finite()), "NaN/inf in {f:?}");
    assert_eq!(f.len(), n_tr + pinned.len());
    for (k, &v) in f[..n_tr].iter().enumerate() {
        assert!(
            v.abs() < 10.0,
            "trans/rot mode {k} = {v} cm⁻¹ after projection (expected < 10)"
        );
    }
    assert_eq!(freq.n_imaginary, 0, "imaginary modes at a minimum: {f:?}");
    for (k, (&got, &want)) in f[n_tr..].iter().zip(pinned).enumerate() {
        assert!(
            (got - want).abs() < tol,
            "vib mode {k}: {got:.4} vs pinned {want:.4} cm⁻¹ (tol {tol})"
        );
    }
    let th = &result.frequencies.as_ref().unwrap().thermochemistry;
    assert!(th.zpe > 0.0 && th.entropy.is_finite() && th.gibbs.is_finite());
    assert!(th.entropy_qrrho.is_finite() && th.gibbs_qrrho.is_finite());
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn pbe_rks_water_sto3g_frequencies() {
    let water = Molecule::from_xyz(
        "3\nPBE/STO-3G optimized water\n\
         O 0.0 0.0 0.18776140\n\
         H 0.0 0.77196482 -0.50538070\n\
         H 0.0 -0.77196482 -0.50538070\n",
    )
    .unwrap();
    let result = Job {
        molecule: water,
        basis: "sto-3g".into(),
        method: Method::Dft(FunctionalSpec::parse("pbe").unwrap()),
        options: JobOptions {
            compute_frequencies: true,
            symmetry_number: 2,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    check_freqs(&result, 6, &[1984.50, 3495.40, 3709.88], 1.0);
}

#[test]
fn uhf_oh_radical_def2svp_frequencies() {
    let oh = mol(
        &[(8, [0.0, 0.0, 0.0132023265]), (1, [0.0, 0.0, 1.8198320143])],
        0,
        2,
    );
    let result = Job {
        molecule: oh,
        basis: "def2-svp".into(),
        method: Method::Uhf,
        options: JobOptions {
            compute_frequencies: true,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    check_freqs(&result, 5, &[4053.22], 1.0);
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn r2scan_3c_hf_frequencies() {
    let c = hartree::composite::composite("r2scan-3c").unwrap();
    let hf = Molecule::from_xyz(
        "2\nr2scan-3c optimized hydrogen fluoride\n\
         F 0.0 0.0 -0.00173416\n\
         H 0.0 0.0 0.92173416\n",
    )
    .unwrap();
    let result = Job {
        molecule: hf,
        basis: c.basis.into(),
        method: Method::Dft(FunctionalSpec::parse(c.functional).unwrap()),
        options: JobOptions {
            compute_frequencies: true,
            dispersion: Some(c.dispersion),
            gcp: c.gcp,
            srb: c.srb,
            grid_level: c.grid_level,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    check_freqs(&result, 5, &[4109.38], 1.0);
    assert!(result.gcp_energy.is_some() && result.dispersion_energy.is_some());
    let th = &result.frequencies.as_ref().unwrap().thermochemistry;
    assert!((result.best_energy() - -100.4438572991).abs() < 1e-6);
    assert!((th.enthalpy - (result.best_energy() + th.enthalpy_corr)).abs() < 1e-12);
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn water_dimer_mrrho_direction() {
    let dimer = mol(
        &[
            (8, [-2.8556396897, -0.2744820068, 0.0009186396]),
            (1, [-3.3775717399, 1.5182762886, -0.0000146865]),
            (1, [-0.9954425983, -0.0838054685, -0.0005479748]),
            (8, [2.2949060482, 0.2524835363, -0.0004451581]),
            (1, [3.0596770651, -0.6565075152, -1.4385128709]),
            (1, [3.0579427577, -0.6564102829, 1.4386020507]),
        ],
        0,
        1,
    );
    let result = Job {
        molecule: dimer,
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            compute_frequencies: true,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    let fd = result.frequencies.as_ref().unwrap();
    let f = &fd.frequencies.frequencies_cm1;
    assert_eq!(fd.frequencies.n_imaginary, 0, "{f:?}");
    let lowest_vib = f[6];
    assert!(
        (lowest_vib - 113.3).abs() < 2.0,
        "lowest dimer mode {lowest_vib:.2} cm⁻¹ (pinned 113.3)"
    );
    let th = &fd.thermochemistry;
    assert!(
        th.entropy_qrrho < th.entropy,
        "S(mRRHO) {} must be below S(RRHO) {}",
        th.entropy_qrrho,
        th.entropy
    );
    assert!(
        th.gibbs_qrrho > th.gibbs,
        "G(mRRHO) {} must be above G(RRHO) {}",
        th.gibbs_qrrho,
        th.gibbs
    );
    let dg_kcal = (th.gibbs_qrrho - th.gibbs) * 627.509_474;
    assert!(
        (0.005..2.0).contains(&dg_kcal),
        "ΔG(mRRHO−RRHO) = {dg_kcal:.4} kcal/mol"
    );
}
