use hartree::core::{Atom, Element, Molecule};
use hartree::{Job, JobOptions, Method, PostHfResult};

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

fn mp2_job(basis: &str, opts: JobOptions) -> Job {
    Job {
        molecule: water(),
        basis: basis.into(),
        method: Method::Mp2,
        options: opts,
    }
}

#[test]
fn ri_mp2_job_matches_conventional_mp2() {
    let conv = mp2_job("def2-svp", JobOptions::default()).run().unwrap();
    let ri = mp2_job(
        "def2-svp",
        JobOptions {
            ri_mp2: true,
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap();
    assert!(conv.converged() && ri.converged());

    let Some(PostHfResult::Mp2 { result: c, .. }) = &conv.post_hf else {
        panic!("expected conventional MP2 result");
    };
    let Some(PostHfResult::RiMp2 {
        result: r,
        n_frozen,
        aux_basis,
    }) = &ri.post_hf
    else {
        panic!("expected RI-MP2 result");
    };
    assert_eq!(aux_basis, "def2-svp/c");
    assert_eq!(*n_frozen, c.n_frozen, "same frozen-core convention");
    assert_eq!(*n_frozen, 1, "water freezes the O 1s");
    assert!((r.opposite_spin - c.opposite_spin).abs() <= 2e-4);
    assert!((r.same_spin - c.same_spin).abs() <= 2e-4);
    assert!((r.total_energy - c.total_energy).abs() <= 2e-4);
    assert!((ri.best_energy() - r.total_energy).abs() < 1e-12);
}

#[test]
fn ri_mp2_on_ri_jk_scf() {
    let conv = mp2_job("def2-svp", JobOptions::default()).run().unwrap();
    let ri = mp2_job(
        "def2-svp",
        JobOptions {
            ri: true,
            ri_mp2: true,
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap();
    assert!(ri.converged());
    let Some(PostHfResult::RiMp2 {
        result, aux_basis, ..
    }) = &ri.post_hf
    else {
        panic!("expected RI-MP2 result");
    };
    assert_eq!(aux_basis, "def2-svp/c");
    assert_eq!(
        ri.ri.as_ref().map(|d| d.aux_basis.as_str()),
        Some("def2-universal-jkfit"),
        "the SCF step keeps the JK fitting set"
    );
    assert!((result.total_energy - conv.best_energy()).abs() <= 1e-3);
}

fn oh_radical() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(Element::from_symbol("O").unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_symbol("H").unwrap(), [0.0, 0.0, 1.8344]),
        ],
        0,
        2,
    )
}

#[test]
fn uhf_ri_mp2_job_matches_conventional_uhf_mp2() {
    let job = |opts: JobOptions| Job {
        molecule: oh_radical(),
        basis: "def2-svp".into(),
        method: Method::Mp2,
        options: opts,
    };
    let conv = job(JobOptions::default()).run().unwrap();
    let ri = job(JobOptions {
        ri_mp2: true,
        ..JobOptions::default()
    })
    .run()
    .unwrap();
    assert!(conv.converged() && ri.converged());
    assert_eq!(conv.scf.reference, hartree::scf::Reference::Uhf);
    assert_eq!(ri.scf.reference, hartree::scf::Reference::Uhf);

    let Some(PostHfResult::Mp2 { result: c, .. }) = &conv.post_hf else {
        panic!("expected conventional MP2 result");
    };
    let Some(PostHfResult::RiMp2 {
        result: r,
        n_frozen,
        aux_basis,
    }) = &ri.post_hf
    else {
        panic!("expected RI-MP2 result");
    };
    assert_eq!(aux_basis, "def2-svp/c");
    assert_eq!(*n_frozen, c.n_frozen, "same frozen-core convention");
    assert_eq!(*n_frozen, 1, "OH freezes the O 1s (both spin channels)");
    assert!((r.opposite_spin - c.opposite_spin).abs() <= 2e-4);
    assert!((r.same_spin - c.same_spin).abs() <= 2e-4);
    assert!((r.total_energy - c.total_energy).abs() <= 2e-4);
    assert!((ri.best_energy() - r.total_energy).abs() < 1e-12);
}

#[test]
fn ri_mp2_guards() {
    let err = Job {
        molecule: water(),
        basis: "def2-svp".into(),
        method: Method::Rhf,
        options: JobOptions {
            ri_mp2: true,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap_err();
    assert!(err.contains("MP2 method only"), "{err}");

    let err = mp2_job(
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

    let err = mp2_job(
        "def2-svp",
        JobOptions {
            ri_mp2: true,
            direct: true,
            ..JobOptions::default()
        },
    )
    .run()
    .unwrap_err();
    assert!(err.contains("--direct"), "{err}");
}
