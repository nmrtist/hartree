use hartree::core::Molecule;
use hartree::scf::{Reference, ScfOptions, run_scf};
use hartree::{BasisSet, Job, JobOptions, Method};

fn agh() -> Molecule {
    Molecule::from_xyz("2\nAgH\nAg 0 0 0\nH 0 0 1.62\n").unwrap()
}

fn hi() -> Molecule {
    Molecule::from_xyz("2\nHI\nI 0 0 0\nH 0 0 1.609\n").unwrap()
}

fn job(mol: Molecule, basis: &str, method: Method) -> Job {
    Job {
        molecule: mol,
        basis: basis.into(),
        method,
        options: JobOptions::default(),
    }
}

#[test]
fn ecp_electron_counting() {
    let svp = BasisSet::load("def2-svp").unwrap();
    assert_eq!(svp.ecp_for(47).unwrap().n_core, 28);
    assert_eq!(svp.ecp_core_electrons(&agh()), 28);
    assert_eq!(svp.ecp_core_electrons(&hi()), 28);
    assert_eq!(
        svp.ecp_core_electrons(&Molecule::from_xyz("1\nAu\nAu 0 0 0\n").unwrap()),
        60
    );
    assert_eq!(
        svp.ecp_core_electrons(&Molecule::from_xyz("1\nO\nO 0 0 0\n").unwrap()),
        0
    );

    let r = job(agh(), "def2-svp", Method::Rhf).run().unwrap();
    assert!(r.scf.converged);
    assert_eq!((r.scf.n_alpha, r.scf.n_beta), (10, 10));

    let ag = Molecule::from_xyz("1\nAg\nAg 0 0 0\n")
        .unwrap()
        .with_multiplicity(2);
    let r = job(ag, "def2-svp", Method::Uhf).run().unwrap();
    assert!(r.scf.converged);
    assert_eq!((r.scf.n_alpha, r.scf.n_beta), (10, 9));
    assert!(
        (r.scf.spin_squared - 0.75).abs() < 0.01,
        "{}",
        r.scf.spin_squared
    );
}

#[test]
fn ecp_scf_regression() {
    let r = job(agh(), "def2-svp", Method::Rhf).run().unwrap();
    assert!(r.scf.converged);
    let vnn_expect = 19.0 / (1.62 * hartree::core::units::ANGSTROM_TO_BOHR);
    assert!((r.scf.nuclear_repulsion - vnn_expect).abs() < 1e-12);
    assert!(
        (r.scf.energy - (-146.624409861919)).abs() < 1e-8,
        "AgH/def2-SVP drifted: {:.12}",
        r.scf.energy
    );

    let r = job(hi(), "def2-svp", Method::Rhf).run().unwrap();
    assert!(r.scf.converged);
    assert!(
        (r.scf.energy - (-297.231531663360)).abs() < 1e-8,
        "HI/def2-SVP drifted: {:.12}",
        r.scf.energy
    );
}

#[test]
fn ecp_scf_stability_and_backends() {
    let mol = agh();
    let e_default = job(mol.clone(), "def2-svp", Method::Rhf)
        .run()
        .unwrap()
        .scf
        .energy;

    let ao = BasisSet::load("def2-svp").unwrap().build(&mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .zip(ao.ecp_core())
        .map(|(a, &c)| (a.position, a.element.z() as f64 - c as f64))
        .collect();
    let zeff: Vec<f64> = charges.iter().map(|&(_, q)| q).collect();
    let vnn = mol.nuclear_repulsion_with(&zeff);
    let ecps = ao.ecps().to_vec();
    let provider =
        hartree::integrals::ConventionalProvider::new(ao.into_integral(), charges).with_ecps(ecps);
    let tight = ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        max_iter: 256,
        ..ScfOptions::default()
    };
    let r = run_scf(&provider, 10, 10, Reference::Rhf, vnn, &tight).unwrap();
    assert!(r.converged);
    assert!(
        (r.energy - e_default).abs() < 1e-8,
        "tight vs default: {:.3e}",
        (r.energy - e_default).abs()
    );

    let direct = Job {
        molecule: mol,
        basis: "def2-svp".into(),
        method: Method::Rhf,
        options: JobOptions {
            direct: true,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    assert!(direct.scf.converged);
    assert!(
        (direct.scf.energy - e_default).abs() < 1e-8,
        "direct vs in-core: {:.3e}",
        (direct.scf.energy - e_default).abs()
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn ecp_frequencies() {
    let result = Job {
        molecule: agh(),
        basis: "def2-svp".into(),
        method: Method::Rhf,
        options: JobOptions {
            compute_frequencies: true,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    let freq = &result.frequencies.as_ref().unwrap().frequencies;
    let f = &freq.frequencies_cm1;
    assert!(f.iter().all(|v| v.is_finite()), "NaN in {f:?}");
    assert_eq!(f.len(), 6);
    for (k, &v) in f[..5].iter().enumerate() {
        assert!(v.abs() < 10.0, "AgH trans/rot mode {k} = {v} cm⁻¹");
    }
    assert!(
        (500.0..3000.0).contains(&f[5]),
        "Ag–H stretch {} cm⁻¹ outside the physical window",
        f[5]
    );
}

#[test]
fn ecp_guards() {
    let expect_err = |opts: JobOptions, method: Method, needle: &str| {
        let err = Job {
            molecule: agh(),
            basis: "def2-svp".into(),
            method,
            options: opts,
        }
        .run()
        .unwrap_err();
        assert!(err.contains("ECP"), "missing ECP mention: {err}");
        assert!(err.contains(needle), "missing {needle:?}: {err}");
    };
    expect_err(
        JobOptions {
            compute_properties: true,
            ..JobOptions::default()
        },
        Method::Rhf,
        "properties",
    );
    expect_err(JobOptions::default(), Method::Mp2, "post-HF");
    expect_err(
        JobOptions {
            ri: true,
            ..JobOptions::default()
        },
        Method::Rhf,
        "RI-JK",
    );
    expect_err(
        JobOptions {
            solvent_eps: Some(78.4),
            ..JobOptions::default()
        },
        Method::Rhf,
        "C-PCM",
    );

    let err = job(agh(), "cc-pvdz", Method::Rhf).run().unwrap_err();
    assert!(err.contains("Z=47"), "{err}");
}
