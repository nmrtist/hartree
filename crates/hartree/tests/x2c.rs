use hartree::core::Molecule;
use hartree::integrals::ConventionalProvider;
use hartree::scf::x2c::{SPEED_OF_LIGHT_AU, X2cTransform, x2c1e_hcore};
use hartree::scf::{Reference, ScfOptions, run_scf};
use hartree::{BasisSet, Job, JobOptions, Method};

fn water() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n").unwrap()
}

fn ao_matrices(mol: &Molecule, basis: &str) -> (hartree::basis::AoBasis, Vec<([f64; 3], f64)>) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    (ao, charges)
}

fn transform(mol: &Molecule, basis: &str, c: f64, lindep: f64) -> X2cTransform {
    let (ao, charges) = ao_matrices(mol, basis);
    let b = ao.integral();
    x2c1e_hcore(
        &b.overlap(),
        &b.kinetic(),
        &b.nuclear(&charges),
        &b.pvp_charges(&charges),
        b.nao(),
        c,
        lindep,
    )
    .unwrap()
}

fn rhf_energy(mol: &Molecule, basis: &str, c: Option<f64>) -> f64 {
    let (ao, charges) = ao_matrices(mol, basis);
    let opts = ScfOptions::default();
    let hcore_override = c.map(|c| {
        let b = ao.integral();
        x2c1e_hcore(
            &b.overlap(),
            &b.kinetic(),
            &b.nuclear(&charges),
            &b.pvp_charges(&charges),
            b.nao(),
            c,
            opts.lindep_thresh,
        )
        .unwrap()
        .h
    });
    let n_elec = mol.n_electrons() as usize;
    let provider = ConventionalProvider::new(ao.into_integral(), charges);
    let scf = run_scf(
        &provider,
        n_elec / 2,
        n_elec / 2,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &ScfOptions {
            hcore_override,
            ..opts
        },
    )
    .unwrap();
    assert!(scf.converged, "SCF must converge");
    scf.energy
}

#[test]
fn nonrelativistic_limit_h2o() {
    let mol = water();
    let e_nr = rhf_energy(&mol, "def2-svp", None);
    let e_x2c = rhf_energy(&mol, "def2-svp", Some(1e6));
    assert!(
        (e_x2c - e_nr).abs() <= 1e-8,
        "NR limit violated: E_X2C(c=1e6) = {e_x2c:.12}, E_NR = {e_nr:.12}, \
         diff = {:.3e}",
        e_x2c - e_nr
    );
}

#[test]
fn h2o_physical_c_small_stabilization() {
    let mol = water();
    let e_nr = rhf_energy(&mol, "def2-svp", None);
    let e_x2c = rhf_energy(&mol, "def2-svp", Some(SPEED_OF_LIGHT_AU));
    let shift = e_nr - e_x2c;
    assert!(
        shift > 1e-3 && shift < 1.0,
        "H2O X2C shift out of the expected light-element range: \
         E_NR = {e_nr:.10}, E_X2C = {e_x2c:.10}, shift = {shift:.6e}"
    );
}

#[test]
fn kr_atom_pinned_fixture() {
    let mol = Molecule::from_xyz("1\nKr\nKr 0 0 0\n").unwrap();
    let e_nr = rhf_energy(&mol, "def2-svp", None);
    let e_x2c = rhf_energy(&mol, "def2-svp", Some(SPEED_OF_LIGHT_AU));
    assert!(
        e_nr - e_x2c > 10.0,
        "Kr should be substantially stabilized: E_NR = {e_nr:.8}, E_X2C = {e_x2c:.8}"
    );
    assert!(
        (e_nr - (-2751.66987861)).abs() < 1e-5,
        "Kr NR energy drifted: {e_nr:.8}"
    );
    assert!(
        (e_x2c - (-2786.48484292)).abs() < 1e-5,
        "Kr X2C energy drifted: {e_x2c:.8}"
    );
}

#[test]
fn job_api_x2c_matches_low_level() {
    let mol = water();
    let r = Job {
        molecule: mol.clone(),
        basis: "def2-svp".into(),
        method: Method::Rhf,
        options: JobOptions {
            x2c: true,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    assert!(r.scf.converged);
    let e_x2c = rhf_energy(&mol, "def2-svp", Some(SPEED_OF_LIGHT_AU));
    assert!(
        (r.scf.energy - e_x2c).abs() < 1e-12,
        "Job x2c path disagrees with the low-level path: {} vs {e_x2c}",
        r.scf.energy
    );
}

#[test]
fn job_api_x2c_dft() {
    let mol = water();
    let run = |x2c: bool| {
        Job {
            molecule: mol.clone(),
            basis: "def2-svp".into(),
            method: Method::Dft(hartree::dft::FunctionalSpec::parse("pbe").unwrap()),
            options: JobOptions {
                x2c,
                grid_level: 1,
                ..JobOptions::default()
            },
        }
        .run()
        .unwrap()
    };
    let nr = run(false);
    let rel = run(true);
    assert!(nr.scf.converged && rel.scf.converged);
    let shift = nr.scf.energy - rel.scf.energy;
    assert!(
        shift > 1e-3 && shift < 1.0,
        "PBE X2C shift out of range: {shift:.6e}"
    );
}

#[test]
fn decoupling_consistency_h2o() {
    let mol = water();
    let out = transform(&mol, "def2-svp", SPEED_OF_LIGHT_AU, 1e-6);
    let n = (out.h.len() as f64).sqrt() as usize;
    assert_eq!(n * n, out.h.len());
    for i in 0..n {
        for j in 0..n {
            let hij = out.h[i * n + j];
            assert!(hij.is_finite(), "h[{i},{j}] is not finite");
            assert_eq!(hij, out.h[j * n + i], "h_X2C must be exactly symmetric");
        }
    }

    let (ao, _) = ao_matrices(&mol, "def2-svp");
    let s = hartree::linalg::mat_from_row_major(n, &ao.integral().overlap());
    let h = hartree::linalg::mat_from_row_major(n, &out.h);
    let se = hartree::linalg::symmetric_eigh(&s);
    let m = out.n_orbitals;
    assert_eq!(m, n, "def2-SVP water has no linear dependence");
    let x = hartree::linalg::Mat::from_fn(n, m, |i, k| se.vectors[(i, k)] / se.values[k].sqrt());
    let hx = hartree::linalg::matmul(
        &hartree::linalg::matmul(&hartree::linalg::transpose(&x), &h),
        &x,
    );
    let he = hartree::linalg::symmetric_eigh(&hx);
    for (got, want) in he.values.iter().zip(&out.electronic_eigenvalues) {
        assert!(
            (got - want).abs() < 1e-9,
            "decoupled spectrum mismatch: {got} vs Dirac {want}"
        );
    }
}

#[test]
fn lindep_threshold_stability() {
    let mol = water();
    let loose = transform(&mol, "def2-svp", SPEED_OF_LIGHT_AU, 1e-6);
    let tight = transform(&mol, "def2-svp", SPEED_OF_LIGHT_AU, 1e-10);
    assert_eq!(loose.n_orbitals, tight.n_orbitals);
    for (a, b) in loose.h.iter().zip(&tight.h) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() < 1e-10, "h drifts with lindep: {a} vs {b}");
    }
}

#[test]
fn guard_x2c_with_ecp_rejected() {
    let ag = Molecule::from_xyz("1\nAg\nAg 0 0 0\n")
        .unwrap()
        .with_multiplicity(2);
    let err = Job {
        molecule: ag,
        basis: "def2-svp".into(),
        method: Method::Uhf,
        options: JobOptions {
            x2c: true,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap_err();
    assert!(
        err.contains("X2C") && err.contains("ECP"),
        "unexpected guard message: {err}"
    );
}

#[test]
fn guard_x2c_gradient_paths_rejected() {
    let run = |method: Method, opt: bool, freq: bool| {
        Job {
            molecule: water(),
            basis: "def2-svp".into(),
            method,
            options: JobOptions {
                x2c: true,
                optimize_geometry: opt,
                compute_frequencies: freq,
                ..JobOptions::default()
            },
        }
        .run()
        .unwrap_err()
    };
    let err = run(Method::Rhf, true, false);
    assert!(
        err.contains("X2C") && err.contains("optimization"),
        "unexpected --opt guard: {err}"
    );
    let err = run(Method::Rhf, false, true);
    assert!(
        err.contains("X2C") && err.contains("frequencies"),
        "unexpected --freq guard: {err}"
    );
    let err = run(Method::Mp2, false, false);
    assert!(
        err.contains("X2C") && err.contains("post-HF"),
        "unexpected post-HF guard: {err}"
    );
}
