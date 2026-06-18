use hartree::core::Molecule;
use hartree::ext::ensemble::{Conformer, Ensemble};
use hartree::multilevel::{MultiLevelOptions, parse_spec, rerank_ensemble, run_multilevel};
use hartree::{Job, JobOptions, Method};

fn h2() -> Molecule {
    Molecule::from_xyz("2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.78\n").unwrap()
}

fn water() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n")
        .unwrap()
}

#[test]
fn same_level_h2_matches_manual_opt_plus_sp_bitwise() {
    let spec = parse_spec("hf/sto-3g // hf/sto-3g", 1).unwrap().unwrap();
    let res = run_multilevel(&h2(), &spec, &MultiLevelOptions::default()).unwrap();

    let opt = Job {
        molecule: h2(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            optimize_geometry: true,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    let positions = &opt.optimized_geometry.as_ref().unwrap().positions;
    let mut mol = h2();
    for (a, p) in mol.atoms.iter_mut().zip(positions) {
        a.position = *p;
    }
    let sp = Job {
        molecule: mol,
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions::default(),
    }
    .run()
    .unwrap();

    assert_eq!(res.e_low, opt.best_energy());
    assert_eq!(res.e_high, sp.best_energy());
    for (a, b) in res
        .geometry
        .atoms
        .iter()
        .zip(&opt.optimized_geometry.unwrap().positions)
    {
        assert_eq!(a.position, *b);
    }
    assert!(res.thermo.is_none());
}

#[test]
fn heterogeneous_h2_pinned() {
    let spec = parse_spec("hf/6-31g // hf/sto-3g", 1).unwrap().unwrap();
    let res = run_multilevel(&h2(), &spec, &MultiLevelOptions::default()).unwrap();
    assert!(
        (res.e_low - (-1.117505884197)).abs() < 1e-8,
        "E_low = {:.12}",
        res.e_low
    );
    assert!(
        (res.e_high - (-1.126587648154)).abs() < 1e-8,
        "E_high//low = {:.12}",
        res.e_high
    );
    assert!((res.e_high - res.e_low).abs() > 1e-4);
}

#[test]
fn composite_free_energy_composition_water() {
    let spec = parse_spec("hf/6-31g // hf/sto-3g", 1).unwrap().unwrap();
    let opts = MultiLevelOptions {
        compute_frequencies: true,
        symmetry_number: 2,
        ..MultiLevelOptions::default()
    };
    let res = run_multilevel(&water(), &spec, &opts).unwrap();
    let t = res.thermo.as_ref().expect("freq requested");

    let freq_job = Job {
        molecule: res.geometry.clone(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions {
            compute_frequencies: true,
            symmetry_number: 2,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    let e_low = freq_job.best_energy();
    let thermo = &freq_job.frequencies.as_ref().unwrap().thermochemistry;

    let tol = 1e-9;
    assert!(
        (t.gibbs - (res.e_high + (thermo.gibbs - e_low))).abs() < tol,
        "G_composite {:.12} vs E_high + (G_low - E_low) {:.12}",
        t.gibbs,
        res.e_high + (thermo.gibbs - e_low)
    );
    assert!(
        (t.enthalpy - (res.e_high + (thermo.enthalpy - e_low))).abs() < tol,
        "H_composite mismatch"
    );
    assert!(
        (t.gibbs_qrrho - (res.e_high + (thermo.gibbs_qrrho - e_low))).abs() < tol,
        "G_composite(mRRHO) mismatch"
    );
    assert!(t.g_corr.abs() < 0.1 && t.h_corr > 0.0);
    assert_eq!(t.freq.frequencies.n_imaginary, 0);
}

#[test]
fn guards_fire_with_the_existing_named_errors() {
    let spec = parse_spec("hf/sto-3g // mp2/sto-3g", 1).unwrap().unwrap();
    let err = run_multilevel(&h2(), &spec, &MultiLevelOptions::default()).unwrap_err();
    assert!(
        err.contains("geometry optimization is not supported for post-HF"),
        "unexpected error: {err}"
    );

    let spec = parse_spec("hf/sto-3g // hf/sto-3g", 1).unwrap().unwrap();
    let opts = MultiLevelOptions {
        alpb: Some("water".into()),
        ..MultiLevelOptions::default()
    };
    let err = run_multilevel(&h2(), &spec, &opts).unwrap_err();
    assert!(
        err.contains("ALPB") && err.contains("geometry"),
        "unexpected error: {err}"
    );

    let opts = MultiLevelOptions {
        compute_frequencies: true,
        solvent_eps: Some(78.4),
        ..MultiLevelOptions::default()
    };
    let err = run_multilevel(&h2(), &spec, &opts).unwrap_err();
    assert!(
        err.contains("frequencies in solvent are not supported"),
        "unexpected error: {err}"
    );
}

#[test]
fn rerank_ensemble_weights_and_cap() {
    let stretched = Molecule::from_xyz("2\nh2 stretched\nH 0.0 0.0 0.0\nH 0.0 0.0 0.90\n").unwrap();
    let ensemble = Ensemble::new(vec![
        Conformer {
            molecule: h2(),
            energy: -1.0,
        },
        Conformer {
            molecule: stretched,
            energy: -0.9,
        },
    ]);
    let spec = parse_spec("hf/6-31g // hf/sto-3g", 1).unwrap().unwrap();
    let opts = MultiLevelOptions::default();

    let ranked = rerank_ensemble(&ensemble, &spec, &opts, 6).unwrap();
    assert_eq!(ranked.len(), 2);
    let wsum: f64 = ranked.iter().map(|r| r.weight).sum();
    assert!((wsum - 1.0).abs() < 1e-12, "weights sum to {wsum}");
    assert!(
        (ranked[0].e_high - ranked[1].e_high).abs() < 1e-6,
        "both conformers reach the same minimum: {:.12} vs {:.12}",
        ranked[0].e_high,
        ranked[1].e_high
    );
    assert!((ranked[0].weight - 0.5).abs() < 1e-3);
    assert!(ranked[0].e_high <= ranked[1].e_high);

    let capped = rerank_ensemble(&ensemble, &spec, &opts, 1).unwrap();
    assert_eq!(capped.len(), 1);
    assert!((capped[0].weight - 1.0).abs() < 1e-12);

    let err = rerank_ensemble(&Ensemble::new(vec![]), &spec, &opts, 6).unwrap_err();
    assert!(err.contains("non-empty"), "unexpected error: {err}");
}
