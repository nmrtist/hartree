use hartree::core::Molecule;
use hartree::{Atom, CpFragments, Job, JobOptions, Method, counterpoise};

const WATER_DIMER: &str = "6\nwater dimer\n\
O -1.551007 -0.114520  0.000000\n\
H -1.934259  0.762503  0.000000\n\
H -0.599677  0.040712  0.000000\n\
O  1.350625  0.111469  0.000000\n\
H  1.680398 -0.373741 -0.758561\n\
H  1.680398 -0.373741  0.758561\n";

const WATER_LI: &str = "4\nwater-Li+\n\
O 0 0 0.117790\n\
H 0 0.755453 -0.471161\n\
H 0 -0.755453 -0.471161\n\
Li 0 0 2.017790\n";

fn job(mol: Molecule, basis: &str, method: Method) -> Job {
    Job {
        molecule: mol,
        basis: basis.into(),
        method,
        options: JobOptions::default(),
    }
}

#[test]
fn water_dimer_rhf_def2svp_counterpoise() {
    let dimer = Molecule::from_xyz(WATER_DIMER).unwrap();
    let cp = counterpoise(
        &job(dimer, "def2-svp", Method::Rhf),
        &CpFragments::new(vec![0, 1, 2]),
    )
    .unwrap();

    let de = cp.interaction_uncorrected();
    let de_cp = cp.interaction_cp();
    let bsse = cp.bsse();
    assert!(de < 0.0, "dimer not bound uncorrected: {de}");
    assert!(de_cp < 0.0, "dimer not bound after CP: {de_cp}");
    assert!(
        de_cp > de,
        "CP must remove overbinding: dE_int^CP = {de_cp} !> dE_int = {de}"
    );
    assert!(bsse <= 0.0, "BSSE estimate must be <= 0: {bsse}");
    assert!(
        (de_cp + bsse - de).abs() < 1e-12,
        "identity dE = dE^CP + bsse"
    );
    let kcal = 627.509474063;
    assert!(
        (-8.0..-2.0).contains(&(de_cp * kcal)),
        "dE_int^CP = {} kcal/mol",
        de_cp * kcal
    );
    assert!(
        (-2.0..0.0).contains(&(bsse * kcal)),
        "bsse = {} kcal/mol",
        bsse * kcal
    );
}

#[test]
fn cp_pieces_match_independent_single_points() {
    let dimer = Molecule::from_xyz(WATER_DIMER).unwrap();
    let cp = counterpoise(
        &job(dimer.clone(), "sto-3g", Method::Rhf),
        &CpFragments::new(vec![0, 1, 2]),
    )
    .unwrap();

    let run = |mol: Molecule| job(mol, "sto-3g", Method::Rhf).run().unwrap().best_energy();

    assert_eq!(cp.e_complex, run(dimer.clone()));
    assert_eq!(cp.e_a, run(Molecule::new(dimer.atoms[..3].to_vec(), 0, 1)));
    assert_eq!(cp.e_b, run(Molecule::new(dimer.atoms[3..].to_vec(), 0, 1)));
    let mut a_ghosted = dimer.atoms[..3].to_vec();
    a_ghosted.extend(
        dimer.atoms[3..]
            .iter()
            .map(|a| Atom::new_ghost(a.element, a.position)),
    );
    assert_eq!(cp.e_a_in_dimer_basis, run(Molecule::new(a_ghosted, 0, 1)));
}

#[test]
fn water_dimer_mp2_counterpoise() {
    let dimer = Molecule::from_xyz(WATER_DIMER).unwrap();
    let cp = counterpoise(
        &job(dimer, "sto-3g", Method::Mp2),
        &CpFragments::new(vec![0, 1, 2]),
    )
    .unwrap();
    assert!(cp.bsse() <= 0.0);
    assert!(cp.interaction_cp() > cp.interaction_uncorrected());
}

#[test]
fn water_lithium_cation_counterpoise() {
    let complex = Molecule::from_xyz(WATER_LI).unwrap().with_charge(1);
    let frags = CpFragments {
        fragment_a: vec![0, 1, 2],
        charge_a: 0,
        multiplicity_a: 1,
        charge_b: 1,
        multiplicity_b: 1,
    };
    let cp = counterpoise(&job(complex.clone(), "sto-3g", Method::Rhf), &frags).unwrap();
    assert!(cp.interaction_cp() < 0.0, "Li+ - water must bind");
    assert!(cp.bsse() <= 0.0);

    let bad = CpFragments {
        charge_b: 0,
        ..frags.clone()
    };
    let err = counterpoise(&job(complex, "sto-3g", Method::Rhf), &bad).unwrap_err();
    assert!(err.contains("inconsistent"), "unexpected error: {err}");
}

#[test]
fn cp_input_validation() {
    let dimer = Molecule::from_xyz(WATER_DIMER).unwrap();
    let j = |mol: Molecule| job(mol, "sto-3g", Method::Rhf);

    assert!(
        counterpoise(&j(dimer.clone()), &CpFragments::new(vec![])).is_err(),
        "empty fragment A"
    );
    assert!(
        counterpoise(&j(dimer.clone()), &CpFragments::new((0..6).collect())).is_err(),
        "empty fragment B"
    );
    assert!(
        counterpoise(&j(dimer.clone()), &CpFragments::new(vec![0, 0])).is_err(),
        "duplicate index"
    );
    assert!(
        counterpoise(&j(dimer.clone()), &CpFragments::new(vec![0, 99])).is_err(),
        "out-of-range index"
    );

    let mut opt_job = j(dimer);
    opt_job.options.optimize_geometry = true;
    assert!(counterpoise(&opt_job, &CpFragments::new(vec![0, 1, 2])).is_err());
}

#[test]
fn standalone_gcp_matches_composite_value() {
    use hartree::disp::GcpParams;
    let dimer = Molecule::from_xyz(WATER_DIMER).unwrap();
    let mut j = job(dimer.clone(), "sto-3g", Method::Rhf);
    j.options.gcp = Some(GcpParams::R2SCAN_3C);
    let result = j.run().unwrap();
    let direct = hartree::disp::gcp_energy(&dimer, &GcpParams::R2SCAN_3C);
    assert_eq!(result.gcp_energy, Some(direct));
    assert!((result.best_energy() - (result.scf.energy + direct)).abs() < 1e-14);
}
