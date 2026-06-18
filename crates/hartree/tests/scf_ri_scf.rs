mod scf_common;

use hartree::scf::{Reference, ScfOptions, run_rhf, run_scf};
use serde::Deserialize;

use scf_common::{df_provider_for, geometries, provider_for};

const RI_REFS_JSON: &str = include_str!("../../../tests/ref/ri_references.json");

#[derive(Deserialize)]
struct RiRefs {
    entries: Vec<RiEntry>,
}

#[derive(Deserialize)]
struct RiEntry {
    molecule: String,
    basis: String,
    method: String,
    energy: f64,
}

fn ri_refs() -> RiRefs {
    serde_json::from_str(RI_REFS_JSON).expect("parse ri_references.json")
}

#[test]
fn ri_hf_matches_conventional_to_fitting_error() {
    let cases = [("water", "sto-3g", 4e-4), ("water", "6-31g", 1e-4)];
    let geoms = geometries();
    for (molecule, basis, gate) in cases {
        let mol = geoms.molecules[molecule].molecule();
        let n_elec = mol.n_electrons() as usize;
        let opts = ScfOptions::default();

        let conv = provider_for(&mol, basis);
        let r_conv = run_rhf(&conv, n_elec, mol.nuclear_repulsion(), &opts).unwrap();
        let df = df_provider_for(&mol, basis);
        let r_df = run_rhf(&df, n_elec, mol.nuclear_repulsion(), &opts).unwrap();
        assert!(r_conv.converged && r_df.converged, "{molecule}/{basis}");

        let delta = (r_df.energy - r_conv.energy).abs();
        eprintln!(
            "RI-HF {molecule}/{basis}: E_df = {:.10}, E_conv = {:.10}, fitting error = {delta:.2e}",
            r_df.energy, r_conv.energy
        );
        assert!(
            delta < gate,
            "{molecule}/{basis}: fitting error {delta:.2e} exceeds gate {gate:.0e}"
        );
        assert!(
            delta > 1e-9,
            "{molecule}/{basis}: ΔE = {delta:.2e} is implausibly small for a fitted run"
        );
    }
}

#[test]
fn ri_uhf_equals_rhf_closed_shell() {
    let geoms = geometries();
    let mol = geoms.molecules["water"].molecule();
    let n_elec = mol.n_electrons() as usize;
    let df = df_provider_for(&mol, "6-31g");
    let opts = ScfOptions::default();

    let rhf = run_rhf(&df, n_elec, mol.nuclear_repulsion(), &opts).unwrap();
    let uhf = run_scf(
        &df,
        n_elec / 2,
        n_elec / 2,
        Reference::Uhf,
        mol.nuclear_repulsion(),
        &opts,
    )
    .unwrap();
    assert!(rhf.converged && uhf.converged);
    let delta = (uhf.energy - rhf.energy).abs();
    eprintln!(
        "RI water/6-31g: RHF {:.12} vs closed-shell UHF {:.12} (Δ = {delta:.2e})",
        rhf.energy, uhf.energy
    );
    assert!(delta < 1e-9, "closed-shell UHF/RHF gap {delta:.2e}");
}

fn check_oracle(select: impl Fn(&RiEntry) -> bool, expect: usize) {
    let geoms = geometries();
    const TOL: f64 = 1e-9;
    let mut checked = 0;
    for entry in ri_refs()
        .entries
        .iter()
        .filter(|e| e.method == "ri-rhf" && select(e))
    {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let n_elec = mol.n_electrons() as usize;
        let df = df_provider_for(&mol, &entry.basis);
        let r = run_rhf(&df, n_elec, mol.nuclear_repulsion(), &ScfOptions::default()).unwrap();
        assert!(
            r.converged,
            "{}/{}: SCF did not converge",
            entry.molecule, entry.basis
        );
        let delta = r.energy - entry.energy;
        eprintln!(
            "RI-RHF {}/{}: hartree {:.10} vs PySCF {:.10} (Δ = {delta:.2e})",
            entry.molecule, entry.basis, r.energy, entry.energy,
        );
        assert!(
            delta.abs() < TOL,
            "RI-RHF {}/{}: hartree {:.12} vs PySCF {:.12} (Δ = {delta:.2e})",
            entry.molecule,
            entry.basis,
            r.energy,
            entry.energy,
        );
        checked += 1;
    }
    assert_eq!(checked, expect, "expected {expect} references");
}

#[test]
fn ri_rhf_water_def2_svp_matches_pyscf() {
    check_oracle(|e| e.basis == "def2-svp", 1);
}

#[test]
#[ignore = "TZ RI oracle (def2-TZVP); run with --release -- --ignored"]
fn ri_rhf_water_def2_tzvp_matches_pyscf() {
    check_oracle(|e| e.basis == "def2-tzvp", 1);
}
