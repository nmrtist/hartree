use hartree::core::Molecule;
use hartree::dft::FunctionalSpec;
use hartree::{Job, JobOptions, Method};

fn water() -> Molecule {
    use hartree::core::{Atom, Element};
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

fn job(func: &str, opts: JobOptions) -> Job {
    Job {
        molecule: water(),
        basis: "def2-svp".into(),
        method: Method::Dft(FunctionalSpec::parse(func).unwrap()),
        options: opts,
    }
}

#[test]
fn wb97xv_water_vv10_total() {
    let r = job("wb97x-v", JobOptions::default()).run().unwrap();
    assert!(r.scf.converged);
    let e_nl = r.vv10_energy.expect("ωB97X-V carries VV10");
    assert!(
        e_nl > 0.0 && e_nl < 0.1,
        "E_nl = {e_nl} outside the sane range"
    );
    assert!((r.best_energy() - (r.scf.energy + e_nl)).abs() < 1e-14);
    println!("wb97x-v: E_scf = {:.12}  E_nl = {:.12}", r.scf.energy, e_nl);
    assert!((r.scf.energy - -76.345153292028).abs() < 1e-8);
    assert!((e_nl - 0.047365112642).abs() < 1e-9);
}

#[test]
fn b97mv_water_vv10_total() {
    let r = job("b97m-v", JobOptions::default()).run().unwrap();
    assert!(r.scf.converged);
    let e_nl = r.vv10_energy.expect("B97M-V carries VV10");
    assert!(e_nl > 0.0 && e_nl < 0.1);
    println!("b97m-v: E_scf = {:.12}  E_nl = {:.12}", r.scf.energy, e_nl);
    assert!((r.scf.energy - -76.344774687374).abs() < 1e-8);
    assert!((e_nl - 0.047367026618).abs() < 1e-9);
}

#[test]
fn b97mv_def2_svpd_protocol_single_point() {
    let r = Job {
        molecule: water(),
        basis: "def2-svpd".into(),
        method: Method::Dft(FunctionalSpec::parse("b97m-v").unwrap()),
        options: JobOptions::default(),
    }
    .run()
    .unwrap();
    assert!(r.scf.converged);
    let e_nl = r.vv10_energy.expect("B97M-V carries VV10");
    println!(
        "b97m-v/def2-svpd: E_scf = {:.12}  E_nl = {:.12}  total = {:.12}",
        r.scf.energy,
        e_nl,
        r.best_energy()
    );
    assert!((r.scf.energy - -76.362229246367).abs() < 1e-8);
    assert!((e_nl - 0.047338493691).abs() < 1e-9);
}

#[test]
fn m062x_has_no_vv10() {
    let r = job("m06-2x", JobOptions::default()).run().unwrap();
    assert!(r.scf.converged);
    assert!(r.vv10_energy.is_none());
    assert_eq!(r.best_energy(), r.scf.energy);
}

#[test]
fn rs_and_v_functional_guards() {
    type OptSetter = fn(&mut JobOptions);
    let cases: &[(&str, OptSetter)] = &[
        ("--direct", |o| o.direct = true),
        ("--opt", |o| o.optimize_geometry = true),
        ("--freq", |o| o.compute_frequencies = true),
    ];
    for func in ["wb97x-v", "wb97m-v", "b97m-v"] {
        for (label, set) in cases {
            let mut opts = JobOptions::default();
            set(&mut opts);
            let err = job(func, opts)
                .run()
                .expect_err(&format!("{func} with {label} should be rejected"));
            assert!(
                err.contains("not supported"),
                "{func} {label}: unexpected message {err:?}"
            );
            if *label != "--direct" {
                assert!(
                    err.contains("VV10") && err.contains("E_nl"),
                    "{func} {label}: guard must name the VV10 E_nl gradient gap, got {err:?}"
                );
            }
        }
    }
    for func in ["wb97x-v", "wb97m-v"] {
        let err = job(
            func,
            JobOptions {
                ri: true,
                ..JobOptions::default()
            },
        )
        .run()
        .expect_err("RS + --ri without --cosx must be rejected");
        assert!(
            err.contains("--ri alone is not supported") && err.contains("--cosx"),
            "{func} --ri: unexpected message {err:?}"
        );
    }
    let r = job(
        "m06-2x",
        JobOptions {
            ri: true,
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap();
    assert!(r.scf.converged);
}

#[test]
fn b97mv_ri_matches_incore_within_fit_error() {
    let conv = job("b97m-v", JobOptions::default()).run().unwrap();
    let ri = job(
        "b97m-v",
        JobOptions {
            ri: true,
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap();
    assert!(conv.scf.converged && ri.scf.converged);
    let e_nl_ri = ri.vv10_energy.expect("B97M-V carries VV10 on --ri too");
    let de = (ri.best_energy() - conv.best_energy()).abs();
    println!(
        "b97m-v --ri: total {:.10} vs in-core {:.10} (|dE| = {de:.2e}), E_nl {e_nl_ri:.10}",
        ri.best_energy(),
        conv.best_energy()
    );
    assert!(de <= 2e-4, "RI-JK vs in-core |dE| = {de:e} > 2e-4 Eh");
}

#[test]
#[ignore = "three wb97x-v/def2-SVP SCFs (~18 s alone, >30 s under parallel load); run via --run-ignored all"]
fn wb97xv_ri_cosx_matches_incore() {
    let with_grid = |f: fn(&mut JobOptions)| {
        let mut o = JobOptions {
            grid_level: 1,
            ..JobOptions::default()
        };
        f(&mut o);
        o
    };
    let conv = job("wb97x-v", with_grid(|_| {})).run().unwrap();
    let cosx = job("wb97x-v", with_grid(|o| o.cosx = true)).run().unwrap();
    let ri_cosx = job(
        "wb97x-v",
        with_grid(|o| {
            o.ri = true;
            o.cosx = true;
        }),
    )
    .run()
    .unwrap();
    assert!(conv.scf.converged && cosx.scf.converged && ri_cosx.scf.converged);

    let ri_diag = ri_cosx.ri.as_ref().expect("RI diagnostics");
    assert_eq!(ri_diag.aux_basis, "def2-universal-jkfit");
    let cosx_diag = ri_cosx.cosx.as_ref().expect("COSX diagnostics");
    assert_eq!(cosx_diag.rs_omega, Some(0.3), "ωB97X-V has ω = 0.3");
    ri_cosx
        .vv10_energy
        .expect("ωB97X-V carries VV10 on --ri --cosx too");

    let d_cosx = (ri_cosx.best_energy() - cosx.best_energy()).abs();
    let d_conv = (ri_cosx.best_energy() - conv.best_energy()).abs();
    println!(
        "wb97x-v --ri --cosx: total {:.10}; vs RS-COSX {:.10} (|dE| = {d_cosx:.2e}); \
         vs in-core {:.10} (|dE| = {d_conv:.2e})",
        ri_cosx.best_energy(),
        cosx.best_energy(),
        conv.best_energy()
    );
    assert!(
        d_cosx <= 1.5e-4,
        "RI-J fit error vs RS-COSX: {d_cosx:e} > 1.5e-4 Eh"
    );
    assert!(
        d_conv <= 2.5e-4,
        "combined RI-J + COSX error vs in-core: {d_conv:e} > 2.5e-4 Eh"
    );
}
