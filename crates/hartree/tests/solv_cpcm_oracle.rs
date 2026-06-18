mod solv_common;

use hartree::basis::BasisSet;
use hartree::core::Molecule;
use hartree::dft::{FunctionalSpec, GridXc};
use hartree::integrals::ConventionalProvider;
use hartree::scf::{Reference, ScfOptions, ScfResult, XcContributor, run_scf_with_env};
use hartree::solv::{Cpcm, DEFAULT_GRID};

use solv_common::{CpcmEntry, cpcm_refs, geometries};

fn run_entry(mol: &Molecule, entry: &CpcmEntry) -> ScfResult {
    let ao = BasisSet::load(&entry.basis).unwrap().build(mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let (xc, opts) = if entry.method == "hf" {
        (None, ScfOptions::default())
    } else {
        let spec = FunctionalSpec::parse(&entry.method).unwrap();
        let xc = GridXc::new(mol, &ao, &spec, 3).unwrap();
        (
            Some(xc),
            ScfOptions {
                energy_tol: 1e-9,
                error_tol: 1e-6,
                ..ScfOptions::default()
            },
        )
    };
    let provider = ConventionalProvider::new(ao.into_integral(), charges);
    let cpcm = Cpcm::new(&provider, mol, entry.eps, DEFAULT_GRID).unwrap();

    let n_elec = mol.n_electrons() as usize;
    let two_s = (mol.multiplicity - 1) as usize;
    let n_alpha = (n_elec + two_s) / 2;
    let n_beta = (n_elec - two_s) / 2;
    let reference = match entry.reference.as_str() {
        "rhf" | "rks" => Reference::Rhf,
        _ => Reference::Uhf,
    };
    run_scf_with_env(
        &provider,
        n_alpha,
        n_beta,
        reference,
        mol.nuclear_repulsion(),
        &opts,
        xc.as_ref().map(|x| x as &dyn XcContributor),
        Some(&cpcm),
    )
    .unwrap()
}

fn check(select: impl Fn(&CpcmEntry) -> bool, tol_e: f64, tol_solv: f64, expect: usize) {
    let geoms = geometries();
    let mut checked = 0;
    for entry in cpcm_refs().entries.iter().filter(|e| select(e)) {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let r = run_entry(&mol, entry);
        assert!(
            r.converged,
            "{}/{}/{}: SCF did not converge",
            entry.molecule, entry.basis, entry.method
        );
        let de = r.energy - entry.energy;
        let ds = r.solvation_energy.unwrap() - entry.e_solv;
        eprintln!(
            "C-PCM {}/{}/{} (eps {}): E {:.10} vs {:.10} (Δ {de:.2e}); E_solv {:.10} vs {:.10} (Δ {ds:.2e})",
            entry.molecule,
            entry.basis,
            entry.method,
            entry.eps,
            r.energy,
            entry.energy,
            r.solvation_energy.unwrap(),
            entry.e_solv,
        );
        assert!(
            de.abs() < tol_e,
            "{}/{}/{}: ΔE_tot = {de:.2e} exceeds {tol_e:.0e}",
            entry.molecule,
            entry.basis,
            entry.method
        );
        assert!(
            ds.abs() < tol_solv,
            "{}/{}/{}: ΔE_solv = {ds:.2e} exceeds {tol_solv:.0e}",
            entry.molecule,
            entry.basis,
            entry.method
        );
        checked += 1;
    }
    assert_eq!(checked, expect, "expected {expect} matching references");
}

#[test]
fn cpcm_hf_small_bases_match_pyscf() {
    check(
        |e| e.method == "hf" && matches!(e.basis.as_str(), "sto-3g" | "6-31g"),
        1e-8,
        1e-8,
        4,
    );
}

#[test]
fn cpcm_uks_pbe_6_31g_matches_pyscf() {
    check(|e| e.method == "pbe" && e.basis == "6-31g", 5e-6, 1e-7, 1);
}

#[test]
#[ignore = "def2-SVP oracle; run with --release -- --ignored"]
fn cpcm_hf_def2_svp_matches_pyscf() {
    check(|e| e.method == "hf" && e.basis == "def2-svp", 1e-8, 1e-8, 2);
}

#[test]
#[ignore = "def2-TZVP oracle; run with --release -- --ignored"]
fn cpcm_hf_def2_tzvp_matches_pyscf() {
    check(
        |e| e.method == "hf" && e.basis == "def2-tzvp",
        1e-8,
        1e-8,
        1,
    );
}

#[test]
#[ignore = "def2-SVP UKS oracle; run with --release -- --ignored"]
fn cpcm_uks_pbe_def2_svp_matches_pyscf() {
    check(
        |e| e.method == "pbe" && e.basis == "def2-svp",
        5e-6,
        1e-7,
        1,
    );
}
