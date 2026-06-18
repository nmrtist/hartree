use hartree::core::Molecule;
use hartree::ext::confgen::{ConfGenOptions, generate_conformers};
use hartree::{Job, JobOptions, Method};

fn butane() -> Molecule {
    Molecule::from_xyz(
        "14
n-butane anti
C    -1.9255    -0.2545     0.0000
C    -0.6586     0.5872     0.0000
C     0.6586    -0.5872     0.0000
C     1.9255     0.2545     0.0000
H    -2.8190     0.3727     0.0000
H    -1.9698    -0.8907     0.8870
H    -1.9698    -0.8907    -0.8870
H    -0.6285     1.2335     0.8835
H    -0.6285     1.2335    -0.8835
H     0.6285    -1.2335     0.8835
H     0.6285    -1.2335    -0.8835
H     2.8190    -0.3727     0.0000
H     1.9698     0.8907     0.8870
H     1.9698     0.8907    -0.8870
",
    )
    .unwrap()
}

#[test]
#[ignore = "slow (HF/STO-3G single points per rotamer); run with --include-ignored"]
fn butane_anti_below_gauche_hf_sto3g() {
    let mol = butane();
    let res = generate_conformers(&mol, &ConfGenOptions::default(), |m| {
        let job = Job {
            molecule: m.clone(),
            basis: "sto-3g".into(),
            method: Method::Rhf,
            options: JobOptions::default(),
        };
        job.run()
            .ok()
            .and_then(|r| r.converged().then(|| r.best_energy()))
    })
    .unwrap();

    assert_eq!(res.rotatable_bonds.len(), 1);
    assert_eq!(res.n_candidates, 3);
    assert_eq!(res.ensemble.len(), 3, "ensemble: {}", res.ensemble.len());

    let rel = res.ensemble.relative_energies_kcal();
    assert!(rel[0].abs() < 1e-6);
    assert!(rel[1] > 0.0, "second conformer not above anti: {rel:?}");
    assert!(
        (rel[1] - rel[2]).abs() < 0.2,
        "gauche pair should be ~degenerate: {rel:?}"
    );
    assert!(rel[1] < 10.0, "gauche gap implausibly large: {rel:?}");
}
