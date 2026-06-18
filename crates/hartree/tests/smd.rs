use hartree::core::Molecule;
use hartree::{Job, JobOptions, Method};

fn water() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n").unwrap()
}

fn methanol() -> Molecule {
    Molecule::from_xyz(
        "6\nmethanol\nC -0.0469 0.6635 0.0000\nO -0.0469 -0.7556 0.0000\n\
         H -1.0801 0.9991 0.0000\nH 0.4366 1.0813 0.8839\nH 0.4366 1.0813 -0.8839\n\
         H 0.8821 -1.0431 0.0000\n",
    )
    .unwrap()
}

fn smd_job(mol: Molecule, solvent: &str) -> hartree::JobResult {
    Job {
        molecule: mol,
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            smd: Some(solvent.into()),
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap()
}

#[test]
fn smd_water_reports_components() {
    let r = smd_job(water(), "water");
    let smd = r.smd.clone().expect("SMD breakdown present");
    assert_eq!(smd.solvent, "water");
    assert!((smd.epsilon - 78.355).abs() < 1e-2);
    assert!(smd.g_ep < 0.0, "ΔG_EP = {}", smd.g_ep);
    assert!((smd.dg_solv - (smd.g_ep + smd.g_cds)).abs() < 1e-12);
    assert!((smd.g_ep - (smd.e_solution - smd.e_gas)).abs() < 1e-12);
    assert!((r.best_energy() - (smd.e_solution + smd.g_cds)).abs() < 1e-12);
    let dg_kcal = smd.dg_solv * 627.509_451;
    assert!(
        (-30.0..5.0).contains(&dg_kcal),
        "ΔG_solv {dg_kcal} kcal/mol out of sane range"
    );
}

#[test]
fn smd_radii_differ_from_bare_cpcm() {
    let smd = smd_job(water(), "water").smd.unwrap();
    let cpcm = Job {
        molecule: water(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            solvent_eps: Some(78.355),
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    let e_solv_cpcm = cpcm.scf.solvation_energy.unwrap();
    assert!(
        (smd.g_ep - e_solv_cpcm).abs() > 1e-5,
        "SMD ΔG_EP {} should differ from bare C-PCM E_solv {}",
        smd.g_ep,
        e_solv_cpcm
    );
}

#[test]
fn smd_second_solvent_differs() {
    let in_water = smd_job(water(), "water").smd.unwrap();
    let in_toluene = smd_job(water(), "toluene").smd.unwrap();
    assert!((in_water.epsilon - in_toluene.epsilon).abs() > 1.0);
    assert!(
        (in_water.dg_solv - in_toluene.dg_solv).abs() > 1e-4,
        "water {} vs toluene {} ΔG_solv too close",
        in_water.dg_solv,
        in_toluene.dg_solv
    );
    assert!(in_water.g_ep < in_toluene.g_ep);
}

#[test]
fn smd_methanol_in_water() {
    let r = smd_job(methanol(), "methanol");
    let smd = r.smd.expect("SMD present");
    assert_eq!(smd.solvent, "methanol");
    assert!(smd.g_ep < 0.0);
}

#[test]
fn smd_and_cpcm_mutually_exclusive() {
    let err = Job {
        molecule: water(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            smd: Some("water".into()),
            solvent_eps: Some(78.355),
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap_err();
    assert!(err.contains("mutually exclusive"), "{err}");
}

#[test]
fn smd_unknown_solvent_rejected() {
    let err = Job {
        molecule: water(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            smd: Some("liquid-unobtainium".into()),
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap_err();
    assert!(err.contains("unknown SMD solvent"), "{err}");
}

#[test]
fn smd_rejects_post_hf() {
    let err = Job {
        molecule: water(),
        basis: "sto-3g".into(),
        method: Method::Mp2,
        options: JobOptions {
            smd: Some("water".into()),
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap_err();
    assert!(err.contains("SCF-level"), "{err}");
}

#[test]
fn smd_pinned_fixtures() {
    let cases: [(Molecule, &str, f64, f64, f64); 4] = [
        (
            water(),
            "water",
            -74.972634758591,
            -0.009571604453,
            0.002290269941,
        ),
        (
            methanol(),
            "water",
            -113.555232621376,
            -0.007065495191,
            0.003960488593,
        ),
        (
            water(),
            "toluene",
            -74.965120361454,
            -0.002057207316,
            -0.001347581357,
        ),
        (
            methanol(),
            "chloroform",
            -113.551531871717,
            -0.003364745532,
            0.000470132553,
        ),
    ];
    for (mol, solvent, e_solution, g_ep, g_cds) in cases {
        let r = smd_job(mol, solvent);
        let d = r.smd.clone().unwrap();
        let tol = 1e-8;
        assert!(
            (d.e_solution - e_solution).abs() < tol,
            "{solvent}: e_solution {} vs pinned {e_solution}",
            d.e_solution
        );
        assert!(
            (d.g_ep - g_ep).abs() < tol,
            "{solvent}: g_ep {} vs pinned {g_ep}",
            d.g_ep
        );
        assert!(
            (d.g_cds - g_cds).abs() < tol,
            "{solvent}: g_cds {} vs pinned {g_cds}",
            d.g_cds
        );
        assert!(
            (r.best_energy() - (e_solution + g_cds)).abs() < tol,
            "{solvent}: best_energy {} vs pinned {}",
            r.best_energy(),
            e_solution + g_cds
        );
    }
}
