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

/// External validation of the def2-ECP heavy-element coverage. Closed-shell (1S0) heavy
/// atoms spanning the range: 4d (Pd), 5p (Xe), 5d/ECP60 (Hg), and the 4f lanthanide
/// Yb -- whose h local channel exercises the L=5 ECP path the raised parser cap admits.
/// The references are independent PySCF RHF/def2-SVP runs over the same raw def2-ECP,
/// so these are external anchors, not self-pins.
/// Tolerance 1e-5: two independent codes (the diatomic AgH/HI pins agree to ~1e-12).
#[test]
fn heavy_atoms_match_external_pyscf_reference() {
    for (sym, e_ref) in [
        ("Pd", -127.0003510202),
        ("Xe", -328.2983936756),
        ("Hg", -152.5503994976),
        ("Yb", -1155.6519630909),
    ] {
        let mol = Molecule::from_xyz(&format!("1\n{sym}\n{sym} 0 0 0\n")).unwrap();
        let r = job(mol, "def2-svp", Method::Rhf).run().unwrap();
        assert!(r.scf.converged, "{sym}: SCF did not converge");
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-5,
            "{sym}/def2-SVP RHF = {:.10} vs PySCF {:.10} (d = {:.2e})",
            r.scf.energy,
            e_ref,
            r.scf.energy - e_ref
        );
    }
}

/// External validation of Kohn-Sham DFT on the def2-ECP path. Plain (non-double-hybrid)
/// KS runs on ECP atoms; the integration grid is built
/// from the full nuclear charge Z while the density carries only valence electrons, so the
/// XC quadrature over an ECP density is its own code path. PBE (pure GGA) and PBE0 (global
/// hybrid → exercises exact exchange on the ECP) on the 4d/5p/5d/4f closed-shell atoms are
/// pinned to independent PySCF RKS/def2-SVP references (grids.level 5). hartree runs its
/// default grid (level 3); the residual is
/// grid-resolution only (≤7e-7, confirmed by the L3↔L4 convergence test in heavy_grid.rs).
#[test]
fn heavy_atoms_dft_match_external_pyscf_reference() {
    for (sym, xc, e_ref) in [
        ("Pd", "pbe", -127.8475034429),
        ("Pd", "pbe0", -127.8251192787),
        ("Xe", "pbe", -329.3428716167),
        ("Xe", "pbe0", -329.3776809901),
        ("Hg", "pbe", -153.5515859678),
        ("Hg", "pbe0", -153.5101501239),
        ("Yb", "pbe", -1159.5350682049),
        ("Yb", "pbe0", -1159.1539942942),
    ] {
        let mol = Molecule::from_xyz(&format!("1\n{sym}\n{sym} 0 0 0\n")).unwrap();
        let method = Method::Dft(hartree::dft::FunctionalSpec::parse(xc).unwrap());
        let r = job(mol, "def2-svp", method).run().unwrap();
        assert!(r.scf.converged, "{sym}/{xc}: KS did not converge");
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-5,
            "{sym}/def2-SVP {xc} = {:.10} vs PySCF {:.10} (d = {:.2e})",
            r.scf.energy,
            e_ref,
            r.scf.energy - e_ref
        );
    }
}

/// External validation of the *second* vendored heavy basis, def2-TZVP. The coverage tests
/// above all use def2-SVP; this pins def2-TZVP energies too. Pd (4d) and Hg
/// (5d/ECP60) are pinned to independent PySCF RHF/def2-TZVP references — both have a genuinely
/// larger valence basis than def2-SVP (energies differ from the def2-SVP pins above), so this
/// exercises the def2-TZVP heavy basis data and the ECP path together. (Xe is deliberately
/// omitted: its def2-TZVP entry is byte-identical to def2-SVP in the BSE source — 16 shells /
/// 31 primitives — so it would not test anything def2-TZVP-specific.)
#[test]
fn heavy_atoms_def2tzvp_match_external_pyscf_reference() {
    for (sym, e_ref) in [("Pd", -127.0301941786), ("Hg", -152.5543268793)] {
        let mol = Molecule::from_xyz(&format!("1\n{sym}\n{sym} 0 0 0\n")).unwrap();
        let r = job(mol, "def2-tzvp", Method::Rhf).run().unwrap();
        assert!(r.scf.converged, "{sym}: SCF did not converge");
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-5,
            "{sym}/def2-TZVP RHF = {:.10} vs PySCF {:.10} (d = {:.2e})",
            r.scf.energy,
            e_ref,
            r.scf.energy - e_ref
        );
    }
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
