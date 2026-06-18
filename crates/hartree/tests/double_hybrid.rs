use hartree::core::{Atom, Element, Molecule};
use hartree::dft::FunctionalSpec;
use hartree::{Job, JobOptions, Method};

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(
                Element::from_symbol("O").unwrap(),
                [0.0, -0.143225816552, 0.0],
            ),
            Atom::new(
                Element::from_symbol("H").unwrap(),
                [1.638036840407, 1.136548822547, 0.0],
            ),
            Atom::new(
                Element::from_symbol("H").unwrap(),
                [-1.638036840407, 1.136548822547, 0.0],
            ),
        ],
        0,
        1,
    )
}

fn job(func: &str, basis: &str, opts: JobOptions) -> Job {
    Job {
        molecule: water(),
        basis: basis.into(),
        method: Method::Dft(FunctionalSpec::parse(func).unwrap()),
        options: opts,
    }
}

#[test]
fn b2plyp_water_def2svp_regression() {
    let r = job("b2plyp", "def2-svp", JobOptions::default())
        .run()
        .unwrap();
    assert!(r.converged());
    let dh = r.double_hybrid.as_ref().expect("b2plyp is a double hybrid");

    assert_eq!(dh.functional_name, "b2plyp");
    assert_eq!(dh.scf_functional_name, "b2plyp");
    assert!((dh.e_scf - r.scf.energy).abs() < 1e-14);
    assert_eq!(dh.pt2_energy_with(0.0, 0.0), 0.0);
    assert!((dh.e_scf + dh.pt2_energy_with(0.0, 0.0) - r.scf.energy).abs() < 1e-14);

    assert!((dh.c_os - 0.27).abs() < 1e-12 && (dh.c_ss - 0.27).abs() < 1e-12);

    let e0 = dh.pt2_energy();
    for t in [0.25, 0.5, 2.0] {
        let et = dh.pt2_energy_with(t * dh.c_os, t * dh.c_ss);
        assert!(
            (et - t * e0).abs() < 1e-13,
            "PT2 scaling is not linear: t = {t}, {et} vs {}",
            t * e0
        );
        let eos = dh.pt2_energy_with(t * dh.c_os, dh.c_ss);
        assert!((eos - (t * dh.c_os * dh.e_os + dh.c_ss * dh.e_ss)).abs() < 1e-13);
    }

    assert!(dh.e_os < 0.0 && dh.e_ss < 0.0);
    assert!(dh.pt2_energy() < 0.0);
    assert!(r.best_energy() < r.scf.energy);
    assert!((r.best_energy() - (dh.e_scf + dh.pt2_energy())).abs() < 1e-14);

    assert_eq!(dh.n_frozen, 1);

    println!(
        "b2plyp: E_scf = {:.12}  E_os = {:.12}  E_ss = {:.12}  total = {:.12}",
        dh.e_scf,
        dh.e_os,
        dh.e_ss,
        r.best_energy()
    );
    assert!((dh.e_scf - FIXTURE_B2PLYP_E_SCF).abs() < 1e-8);
    assert!((dh.e_os - FIXTURE_B2PLYP_E_OS).abs() < 1e-8);
    assert!((dh.e_ss - FIXTURE_B2PLYP_E_SS).abs() < 1e-8);
    assert!((r.best_energy() - FIXTURE_B2PLYP_TOTAL).abs() < 1e-8);
}

const FIXTURE_B2PLYP_E_SCF: f64 = -76.193742914397;
const FIXTURE_B2PLYP_E_OS: f64 = -0.193250885735;
const FIXTURE_B2PLYP_E_SS: f64 = -0.062474960497;
const FIXTURE_B2PLYP_TOTAL: f64 = -76.262788892879;

#[test]
fn revdsd_and_pwpb95_sto3g() {
    let r = job("revdsd-pbep86", "sto-3g", JobOptions::default())
        .run()
        .unwrap();
    assert!(r.converged());
    let dh = r.double_hybrid.as_ref().unwrap();
    assert!((dh.c_os - 0.5922).abs() < 1e-12 && (dh.c_ss - 0.0636).abs() < 1e-12);
    assert!(dh.pt2_energy() < 0.0 && r.best_energy() < r.scf.energy);

    let r = job("pwpb95", "sto-3g", JobOptions::default())
        .run()
        .unwrap();
    assert!(r.converged());
    let dh = r.double_hybrid.as_ref().unwrap();
    assert!((dh.c_os - 0.269).abs() < 1e-12 && dh.c_ss == 0.0);
    assert!((dh.pt2_energy() - dh.c_os * dh.e_os).abs() < 1e-15);
    assert!(r.best_energy() < r.scf.energy);
}

#[test]
fn wb97m2_xdh_water_sto3g() {
    let r = job("wb97m(2)", "sto-3g", JobOptions::default())
        .run()
        .unwrap();
    assert!(r.converged());
    let dh = r
        .double_hybrid
        .as_ref()
        .expect("ωB97M(2) is a double hybrid");
    assert_eq!(dh.functional_name, "wb97m(2)");
    assert_eq!(dh.scf_functional_name, "wb97m-v");
    assert!((dh.c_os - 0.34096).abs() < 1e-12 && (dh.c_ss - 0.34096).abs() < 1e-12);
    assert!((dh.vv10_scale - 0.65904).abs() < 1e-12);

    assert!(
        (dh.e_scf - r.scf.energy).abs() > 1e-6,
        "ωB97M(2) energy expression should differ from the ωB97M-V SCF energy"
    );

    let e_nl = r.vv10_energy.expect("ωB97M(2) retains scaled VV10");
    assert!(dh.pt2_energy() < 0.0);
    assert!((r.best_energy() - (dh.e_scf + dh.pt2_energy() + e_nl)).abs() < 1e-14);

    assert_eq!(
        FunctionalSpec::parse("wb97m(2)").unwrap().d4_param_set(),
        None
    );
}

#[test]
fn dh_d4_metadata_keys() {
    assert_eq!(
        FunctionalSpec::parse("b2plyp").unwrap().d4_param_set(),
        Some("b2plyp")
    );
    assert_eq!(
        FunctionalSpec::parse("revdsd-pbep86")
            .unwrap()
            .d4_param_set(),
        Some("revdsdpbep86")
    );
    assert_eq!(
        FunctionalSpec::parse("pwpb95").unwrap().d4_param_set(),
        Some("pwpb95")
    );
    assert_eq!(
        FunctionalSpec::parse("pbe0").unwrap().d4_param_set(),
        Some("pbe0")
    );
    assert_eq!(
        FunctionalSpec::parse("wb97m-v").unwrap().d4_param_set(),
        None
    );
}

#[test]
fn dh_ri_mp2_matches_conventional_all_functionals() {
    for func in ["b2plyp", "revdsd-pbep86", "pwpb95", "wb97m(2)"] {
        let conv = job(func, "def2-svp", JobOptions::default()).run().unwrap();
        let ri = job(
            func,
            "def2-svp",
            JobOptions {
                ri_mp2: true,
                ..JobOptions::default()
            },
        )
        .run()
        .unwrap();
        assert!(conv.converged() && ri.converged());
        let dh_conv = conv.double_hybrid.as_ref().unwrap();
        let dh_ri = ri.double_hybrid.as_ref().unwrap();
        assert_eq!(dh_conv.pt2_aux_basis, None, "{func}: conventional backend");
        assert_eq!(
            dh_ri.pt2_aux_basis.as_deref(),
            Some("def2-svp/c"),
            "{func}: RI backend"
        );
        assert_eq!(dh_conv.n_frozen, dh_ri.n_frozen);
        assert!((dh_conv.e_scf - dh_ri.e_scf).abs() < 1e-8, "{func}: E_SCF");
        let de = (ri.best_energy() - conv.best_energy()).abs();
        println!(
            "{func}: conv = {:.12}  ri = {:.12}  |dE| = {de:.3e} Eh",
            conv.best_energy(),
            ri.best_energy()
        );
        assert!(
            de <= 5e-4,
            "{func}: |E_DH(RI) - E_DH(conv)| = {de:e} > 5e-4"
        );
    }
}

#[test]
fn b2plyp_ri_mp2_water_def2svp_regression() {
    let r = job(
        "b2plyp",
        "def2-svp",
        JobOptions {
            ri_mp2: true,
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap();
    assert!(r.converged());
    let dh = r.double_hybrid.as_ref().unwrap();
    assert_eq!(dh.pt2_aux_basis.as_deref(), Some("def2-svp/c"));
    assert_eq!(dh.n_frozen, 1);
    println!(
        "b2plyp (RI-MP2): E_scf = {:.12}  E_os = {:.12}  E_ss = {:.12}  total = {:.12}",
        dh.e_scf,
        dh.e_os,
        dh.e_ss,
        r.best_energy()
    );
    assert!((r.best_energy() - FIXTURE_B2PLYP_TOTAL).abs() <= 5e-4);
    assert!((r.best_energy() - FIXTURE_B2PLYP_RI_TOTAL).abs() < 1e-8);
}

const FIXTURE_B2PLYP_RI_TOTAL: f64 = -76.262763229381;

#[test]
fn dh_ri_mp2_missing_aux_errors() {
    let err = job(
        "b2plyp",
        "cc-pvdz",
        JobOptions {
            ri_mp2: true,
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap_err();
    assert!(err.contains("cc-pvdz/c"), "{err}");
    assert!(err.contains("no silent fallback"), "{err}");
}

#[test]
fn dh_guards() {
    let assert_rejects = |opts: JobOptions, what: &str| {
        let err = job("b2plyp", "sto-3g", opts).run().unwrap_err();
        assert!(
            err.contains("double hybrid"),
            "{what}: unexpected error {err:?}"
        );
    };
    assert_rejects(
        JobOptions {
            optimize_geometry: true,
            ..JobOptions::default()
        },
        "--opt",
    );
    assert_rejects(
        JobOptions {
            compute_frequencies: true,
            ..JobOptions::default()
        },
        "--freq",
    );
    assert_rejects(
        JobOptions {
            direct: true,
            ..JobOptions::default()
        },
        "--direct",
    );
    assert_rejects(
        JobOptions {
            ri: true,
            ..JobOptions::default()
        },
        "--ri",
    );
    assert_rejects(
        JobOptions {
            smearing: Some(hartree::scf::Smearing::Fermi {
                temperature_k: 5000.0,
            }),
            ..JobOptions::default()
        },
        "--smear",
    );
    assert_rejects(
        JobOptions {
            fod: true,
            ..JobOptions::default()
        },
        "--fod",
    );
    assert_rejects(
        JobOptions {
            solvent_eps: Some(78.4),
            ..JobOptions::default()
        },
        "--eps",
    );

    let oh = Molecule::new(
        vec![
            Atom::new(Element::from_symbol("O").unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_symbol("H").unwrap(), [0.0, 0.0, 1.83]),
        ],
        0,
        2,
    );
    let err = Job {
        molecule: oh,
        basis: "sto-3g".into(),
        method: Method::Dft(FunctionalSpec::parse("b2plyp").unwrap()),
        options: JobOptions::default(),
    }
    .run()
    .unwrap_err();
    assert!(err.contains("closed-shell"), "open shell: {err:?}");
}
