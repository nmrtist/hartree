mod scf_common;

use hartree::integrals::IntegralProvider;
use hartree::linalg::mat_to_row_major;
use hartree::scf::{ScfOptions, run_rhf};
use serde::Deserialize;

use scf_common::{direct_provider_for, geometries, provider_for, trace_ds};

const HF_BASIS_REFS_JSON: &str = include_str!("../../../tests/ref/hf_basis_references.json");

#[derive(Deserialize)]
struct HfBasisRefs {
    entries: Vec<HfBasisEntry>,
}

#[derive(Deserialize)]
struct HfBasisEntry {
    molecule: String,
    basis: String,
    method: String,
    energy: f64,
}

fn refs() -> HfBasisRefs {
    serde_json::from_str(HF_BASIS_REFS_JSON).expect("parse hf_basis_references.json")
}

fn check(select: impl Fn(&HfBasisEntry) -> bool, direct: bool, expect: usize) {
    let geoms = geometries();
    const TOL: f64 = 1e-9;
    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in refs()
        .entries
        .iter()
        .filter(|e| e.method == "rhf" && select(e))
    {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let n_elec = mol.n_electrons() as usize;
        let options = ScfOptions {
            incremental_fock: direct,
            ..ScfOptions::default()
        };
        let (energy, density, n_basis, converged, s) = if direct {
            let provider = direct_provider_for(&mol, &entry.basis);
            let r = run_rhf(&provider, n_elec, mol.nuclear_repulsion(), &options)
                .unwrap_or_else(|err| panic!("{}/{}: {err}", entry.molecule, entry.basis));
            let s = mat_to_row_major(&provider.overlap());
            (r.energy, r.density, r.n_basis, r.converged, s)
        } else {
            let provider = provider_for(&mol, &entry.basis);
            let r = run_rhf(&provider, n_elec, mol.nuclear_repulsion(), &options)
                .unwrap_or_else(|err| panic!("{}/{}: {err}", entry.molecule, entry.basis));
            let s = mat_to_row_major(&provider.overlap());
            (r.energy, r.density, r.n_basis, r.converged, s)
        };
        assert!(
            converged,
            "{}/{}: SCF did not converge",
            entry.molecule, entry.basis
        );

        let n_check = trace_ds(&density, &s, n_basis);
        assert!(
            (n_check - n_elec as f64).abs() < 1e-9,
            "{}/{}: Tr(DS) = {n_check}, expected {n_elec}",
            entry.molecule,
            entry.basis
        );

        let delta = energy - entry.energy;
        worst = worst.max(delta.abs());
        eprintln!(
            "RHF {}/{}: hartree {:.10} vs PySCF {:.10} (Δ = {delta:.2e})",
            entry.molecule, entry.basis, energy, entry.energy,
        );
        assert!(
            delta.abs() < TOL,
            "RHF {}/{}: hartree {:.12} vs PySCF {:.12} (Δ = {:.2e})",
            entry.molecule,
            entry.basis,
            energy,
            entry.energy,
            delta
        );
        checked += 1;
    }
    eprintln!("HF Karlsruhe basis refs: {checked} matched PySCF; worst Δ = {worst:.2e}");
    assert_eq!(checked, expect, "expected {expect} references");
}

#[test]
#[ignore = "TZ/diffuse HF oracle (def2-TZVPP/TZVPD/SVPD, in-core nao⁴); run with --release -- --ignored"]
fn rhf_def2_tzvpp_and_diffuse_match_pyscf() {
    check(
        |e| matches!(e.basis.as_str(), "def2-tzvpp" | "def2-tzvpd" | "def2-svpd"),
        false,
        5,
    );
}

#[test]
#[ignore = "TZ-class HF oracle (def2-mTZVPP, in-core nao⁴); run with --release -- --ignored"]
fn rhf_def2_mtzvpp_matches_pyscf() {
    check(|e| e.basis == "def2-mtzvpp", false, 2);
}

#[test]
#[ignore = "QZ HF via DirectProvider (def2-QZVPP, minutes-class); run with --release -- --ignored"]
fn rhf_def2_qzvpp_matches_pyscf() {
    check(|e| e.basis == "def2-qzvpp", true, 1);
}
