mod solv_common;

use hartree::core::{Atom, Molecule};
use hartree::scf::{Reference, ScfOptions, run_scf_with_env};
use hartree::solv::{Cpcm, DEFAULT_GRID};

use solv_common::{geometries, provider_for};

fn solvated_hf(mol: &Molecule, basis: &str, eps: f64) -> (f64, f64) {
    let provider = provider_for(mol, basis);
    let cpcm = Cpcm::new(&provider, mol, eps, DEFAULT_GRID).unwrap();
    let n_elec = mol.n_electrons() as usize;
    let two_s = (mol.multiplicity - 1) as usize;
    let (na, nb) = ((n_elec + two_s) / 2, (n_elec - two_s) / 2);
    let reference = if two_s == 0 {
        Reference::Rhf
    } else {
        Reference::Uhf
    };
    let r = run_scf_with_env(
        &provider,
        na,
        nb,
        reference,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
        None,
        Some(&cpcm),
    )
    .unwrap();
    assert!(r.converged);
    (r.energy, r.solvation_energy.unwrap())
}

#[test]
fn neutral_polar_molecule_is_stabilized() {
    let mol = geometries().molecules["water"].molecule();
    for eps in [2.0, 10.0, 78.3553] {
        let (_, e_solv) = solvated_hf(&mol, "sto-3g", eps);
        assert!(e_solv < 0.0, "E_solv = {e_solv} at eps = {eps}");
    }
}

#[test]
fn solvation_vanishes_as_epsilon_approaches_one() {
    let mol = geometries().molecules["water"].molecule();
    let (_, e_far) = solvated_hf(&mol, "sto-3g", 78.3553);
    let (_, e_near) = solvated_hf(&mol, "sto-3g", 1.0 + 1e-8);
    assert!(e_near.abs() < 1e-9, "E_solv(eps→1⁺) = {e_near:.3e}");
    assert!(e_near.abs() < 1e-7 * e_far.abs());
}

#[test]
fn anion_solvates_much_more_strongly_than_neutral() {
    let geoms = geometries();
    let (_, e_water) = solvated_hf(&geoms.molecules["water"].molecule(), "6-31g", 78.3553);
    let (_, e_oh) = solvated_hf(&geoms.molecules["oh_minus"].molecule(), "6-31g", 78.3553);
    assert!(
        e_oh < 5.0 * e_water,
        "OH⁻ ({e_oh}) should bind much more strongly than water ({e_water})"
    );
}

#[test]
fn rigid_motion_invariance() {
    let mol = geometries().molecules["water"].molecule();
    let (e0, s0) = solvated_hf(&mol, "sto-3g", 78.3553);

    let t = [1.3, -0.7, 2.1];
    let translated = Molecule::new(
        mol.atoms
            .iter()
            .map(|a| {
                Atom::new(
                    a.element,
                    [
                        a.position[0] + t[0],
                        a.position[1] + t[1],
                        a.position[2] + t[2],
                    ],
                )
            })
            .collect(),
        mol.charge,
        mol.multiplicity,
    );
    let (e1, s1) = solvated_hf(&translated, "sto-3g", 78.3553);
    assert!((e1 - e0).abs() < 1e-9, "translation: ΔE = {:.2e}", e1 - e0);
    assert!(
        (s1 - s0).abs() < 1e-9,
        "translation: ΔE_solv = {:.2e}",
        s1 - s0
    );

    let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
    let rotated = Molecule::new(
        mol.atoms
            .iter()
            .map(|a| {
                let [x, y, z] = a.position;
                Atom::new(a.element, [c * x - s * y, s * x + c * y, z])
            })
            .collect(),
        mol.charge,
        mol.multiplicity,
    );
    let (e2, s2) = solvated_hf(&rotated, "sto-3g", 78.3553);
    assert!((e2 - e0).abs() < 3e-6, "rotation: ΔE = {:.2e}", e2 - e0);
    assert!(
        (s2 - s0).abs() < 3e-6,
        "rotation: ΔE_solv = {:.2e}",
        s2 - s0
    );
}

#[test]
fn no_solvent_is_bit_identical_to_gas_phase() {
    let mol = geometries().molecules["water"].molecule();
    let provider = provider_for(&mol, "sto-3g");
    let n = mol.n_electrons() as usize;
    let opts = ScfOptions::default();
    let gas = hartree::scf::run_rhf(&provider, n, mol.nuclear_repulsion(), &opts).unwrap();
    let env = run_scf_with_env(
        &provider,
        n / 2,
        n / 2,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &opts,
        None,
        None,
    )
    .unwrap();
    assert_eq!(gas.energy, env.energy);
    assert_eq!(gas.density, env.density);
    assert!(env.solvation_energy.is_none());

    let cpcm = Cpcm::new(&provider, &mol, 78.3553, DEFAULT_GRID).unwrap();
    let solvated = run_scf_with_env(
        &provider,
        n / 2,
        n / 2,
        Reference::Rhf,
        mol.nuclear_repulsion(),
        &opts,
        None,
        Some(&cpcm),
    )
    .unwrap();
    assert!((solvated.energy - gas.energy).abs() > 1e-4);
}
