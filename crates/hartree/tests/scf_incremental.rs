mod scf_common;

use hartree::scf::{Reference, ScfOptions, run_rhf, run_scf};

use scf_common::{direct_provider_for, geometries, provider_for, references};

fn occupations(n_electrons: usize, multiplicity: u32) -> (usize, usize) {
    let two_s = (multiplicity - 1) as usize;
    ((n_electrons + two_s) / 2, (n_electrons - two_s) / 2)
}

fn opts(incremental: bool) -> ScfOptions {
    ScfOptions {
        incremental_fock: incremental,
        fock_rebuild_period: 3,
        ..ScfOptions::default()
    }
}

fn orca_energy(molecule: &str, basis: &str, method: &str) -> Option<f64> {
    references()
        .entries
        .into_iter()
        .find(|e| e.molecule == molecule && e.basis == basis && e.method == method)
        .map(|e| e.energy)
}

#[test]
fn incremental_rhf_matches_full_conventional() {
    let geoms = geometries();
    for (name, basis) in [("water", "6-31g"), ("h2", "cc-pvdz")] {
        let mol = geoms.molecules[name].molecule();
        let provider = provider_for(&mol, basis);
        let n_elec = mol.n_electrons() as usize;
        let nr = mol.nuclear_repulsion();

        let full = run_rhf(&provider, n_elec, nr, &opts(false)).unwrap();
        let incr = run_rhf(&provider, n_elec, nr, &opts(true)).unwrap();
        assert!(
            full.converged && incr.converged,
            "{name}/{basis} did not converge"
        );
        let delta = (full.energy - incr.energy).abs();
        assert!(
            delta < 1e-10,
            "{name}/{basis}: incremental vs full Δ = {delta:.2e}"
        );
    }
}

#[test]
fn incremental_rhf_matches_full_direct() {
    let geoms = geometries();
    for (name, basis) in [("water", "sto-3g"), ("water", "6-31g")] {
        let mol = geoms.molecules[name].molecule();
        let provider = direct_provider_for(&mol, basis);
        let n_elec = mol.n_electrons() as usize;
        let nr = mol.nuclear_repulsion();

        let full = run_rhf(&provider, n_elec, nr, &opts(false)).unwrap();
        let incr = run_rhf(&provider, n_elec, nr, &opts(true)).unwrap();
        assert!(
            full.converged && incr.converged,
            "{name}/{basis} did not converge"
        );

        let delta = (full.energy - incr.energy).abs();
        assert!(
            delta < 1e-10,
            "{name}/{basis}: incremental-direct vs full-direct Δ = {delta:.2e}"
        );

        if let Some(e_orca) = orca_energy(name, basis, "rhf") {
            let de = (incr.energy - e_orca).abs();
            assert!(
                de < 1e-7,
                "{name}/{basis}: incremental-direct {:.10} vs ORCA {e_orca:.10} (Δ = {de:.2e})",
                incr.energy
            );
        }
    }
}

#[test]
fn incremental_uhf_matches_full_direct() {
    let geoms = geometries();
    let mol = geoms.molecules["oh"].molecule();
    let provider = direct_provider_for(&mol, "sto-3g");
    let n_elec = mol.n_electrons() as usize;
    let (na, nb) = occupations(n_elec, mol.multiplicity);
    let nr = mol.nuclear_repulsion();

    let full = run_scf(&provider, na, nb, Reference::Uhf, nr, &opts(false)).unwrap();
    let incr = run_scf(&provider, na, nb, Reference::Uhf, nr, &opts(true)).unwrap();
    assert!(
        full.converged && incr.converged,
        "oh/sto-3g UHF did not converge"
    );
    let delta = (full.energy - incr.energy).abs();
    assert!(
        delta < 1e-10,
        "oh/sto-3g UHF: incremental-direct vs full-direct Δ = {delta:.2e}"
    );
}
