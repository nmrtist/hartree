use hartree::core::Molecule;
use hartree::core::units::ANGSTROM_TO_BOHR;
use hartree::dft::{
    CubeParams, MolecularGrid, fod_density_matrices, fod_grid_integral, write_fod_cube,
};
use hartree::scf::Smearing;
use hartree::{BasisSet, Job, JobOptions, Method};

fn water() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n").unwrap()
}

fn stretched_h2() -> Molecule {
    Molecule::from_xyz("2\nstretched H2\nH 0 0 0\nH 0 0 3.0\n").unwrap()
}

fn fod_job(mol: Molecule, basis: &str, method: Method, options: JobOptions) -> hartree::JobResult {
    Job {
        molecule: mol,
        basis: basis.into(),
        method,
        options,
    }
    .run()
    .unwrap()
}

#[test]
fn equilibrium_water_has_negligible_fod() {
    let result = fod_job(
        water(),
        "sto-3g",
        Method::Rhf,
        JobOptions {
            fod: true,
            smearing: Some(Smearing::Fermi {
                temperature_k: 5000.0,
            }),
            ..JobOptions::default()
        },
    );
    assert!(result.scf.converged);
    let fod = result.fod.expect("fod requested");
    assert_eq!(fod.temperature_k, 5000.0, "explicit --smear pins T_el");
    assert!(
        fod.n_fod <= 0.01,
        "equilibrium water N_FOD = {} (expected ≈ 0)",
        fod.n_fod
    );
    assert!(fod.n_fod >= 0.0 && fod.n_fod_alpha >= 0.0 && fod.n_fod_beta >= 0.0);
}

#[test]
fn stretched_h2_fod_is_large_and_grid_matches_analytic() {
    let mol = stretched_h2();
    let result = fod_job(
        mol.clone(),
        "def2-tzvp",
        Method::Dft(hartree::dft::FunctionalSpec::parse("tpss").unwrap()),
        JobOptions {
            fod: true,
            ..JobOptions::default()
        },
    );
    assert!(result.scf.converged);
    let fod = result.fod.expect("fod requested");
    assert_eq!(fod.temperature_k, 5000.0, "TPSS: a_x = 0 ⇒ T_el = 5000 K");
    assert!(
        fod.n_fod > 0.5,
        "stretched H2 N_FOD = {} (expected substantial static correlation)",
        fod.n_fod
    );
    assert!((fod.n_fod - (fod.n_fod_alpha + fod.n_fod_beta)).abs() < 1e-14);

    let ao = BasisSet::load("def2-tzvp").unwrap().build(&mol).unwrap();
    let grid = MolecularGrid::build(&mol, 3).unwrap();
    let (da, db) = fod_density_matrices(&result.scf).unwrap();
    let d_tot: Vec<f64> = da.iter().zip(&db).map(|(a, b)| a + b).collect();
    let n_grid = fod_grid_integral(ao.shells(), ao.n_ao(), &grid, &d_tot).unwrap();
    assert!(
        (n_grid - fod.n_fod).abs() <= 1e-3,
        "grid N_FOD = {n_grid} vs analytic {}",
        fod.n_fod
    );
}

#[test]
fn fod_cube_roundtrip() {
    let mol = stretched_h2();
    let cube_path = std::env::temp_dir().join("hartree_fod_h2.cube");
    let result = fod_job(
        mol.clone(),
        "def2-tzvp",
        Method::Dft(hartree::dft::FunctionalSpec::parse("tpss").unwrap()),
        JobOptions {
            fod: true,
            fod_cube: Some(cube_path.clone()),
            ..JobOptions::default()
        },
    );
    let fod = result.fod.expect("fod requested");

    let (natoms, npts, values) = read_cube(&std::fs::read_to_string(&cube_path).unwrap());
    assert_eq!(natoms, 2);
    assert_eq!(values.len(), npts[0] * npts[1] * npts[2]);
    let d = 3.0 * ANGSTROM_TO_BOHR;
    assert_eq!(npts[0], 41);
    assert_eq!(npts[1], 41);
    assert_eq!(npts[2], (((d + 8.0) / 0.2).ceil() as usize) + 1);

    let wide = std::env::temp_dir().join("hartree_fod_h2_wide.cube");
    let ao = BasisSet::load("def2-tzvp").unwrap().build(&mol).unwrap();
    let (da, db) = fod_density_matrices(&result.scf).unwrap();
    let d_tot: Vec<f64> = da.iter().zip(&db).map(|(a, b)| a + b).collect();
    let params = CubeParams {
        margin: 7.0,
        spacing: 0.2,
    };
    write_fod_cube(&wide, &mol, ao.shells(), ao.n_ao(), &d_tot, &params).unwrap();
    let (_, npts_w, values_w) = read_cube(&std::fs::read_to_string(&wide).unwrap());
    assert_eq!(values_w.len(), npts_w[0] * npts_w[1] * npts_w[2]);
    assert!(values_w.iter().all(|&v| v >= -1e-12), "ρ_FOD must be ≥ 0");
    let integral: f64 = values_w.iter().sum::<f64>() * params.spacing.powi(3);
    let rel = (integral - fod.n_fod).abs() / fod.n_fod;
    assert!(
        rel <= 0.02,
        "cube integral {integral} vs N_FOD {} (rel err {rel:.4})",
        fod.n_fod
    );

    let _ = std::fs::remove_file(&cube_path);
    let _ = std::fs::remove_file(&wide);
}

#[test]
fn fod_guards() {
    let post_hf = Job {
        molecule: water(),
        basis: "sto-3g".into(),
        method: Method::Mp2,
        options: JobOptions {
            fod: true,
            ..JobOptions::default()
        },
    }
    .run();
    let err = post_hf.expect_err("FOD with MP2 must be rejected");
    assert!(err.contains("post-HF"), "{err}");

    let cube_only = Job {
        molecule: water(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            fod_cube: Some("unused.cube".into()),
            ..JobOptions::default()
        },
    }
    .run();
    let err = cube_only.expect_err("fod_cube without fod must be rejected");
    assert!(err.contains("fod"), "{err}");

    let direct = Job {
        molecule: water(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            fod: true,
            direct: true,
            ..JobOptions::default()
        },
    }
    .run();
    let err = direct.expect_err("FOD with --direct must be rejected");
    assert!(err.contains("in-core"), "{err}");
}

fn read_cube(text: &str) -> (usize, [usize; 3], Vec<f64>) {
    let mut lines = text.lines();
    lines.next().unwrap(); // comment 1
    lines.next().unwrap(); // comment 2
    let head: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let natoms: usize = head[0].parse().unwrap();
    let mut npts = [0usize; 3];
    for n in &mut npts {
        let axis: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
        *n = axis[0].parse().unwrap();
    }
    for _ in 0..natoms {
        lines.next().unwrap(); // atom records
    }
    let values: Vec<f64> = lines
        .flat_map(|l| l.split_whitespace())
        .map(|t| t.parse().unwrap())
        .collect();
    (natoms, npts, values)
}
