mod scf_common;

use std::time::Instant;

use hartree::integrals::IntegralProvider;
use hartree::linalg::mat_to_row_major;
use hartree::scf::{Reference, ScfOptions, run_rhf, run_scf};

use scf_common::{direct_provider_for, geometries, references, trace_ds};

fn occupations(n_electrons: usize, multiplicity: u32) -> (usize, usize) {
    let two_s = (multiplicity - 1) as usize;
    ((n_electrons + two_s) / 2, (n_electrons - two_s) / 2)
}

fn check_rhf_direct(basis_filter: impl Fn(&str) -> bool) -> usize {
    let geoms = geometries();
    let refs = references();

    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in refs
        .entries
        .iter()
        .filter(|e| e.method == "rhf" && e.molecule != "benzene" && basis_filter(&e.basis))
    {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let provider = direct_provider_for(&mol, &entry.basis);
        let n_elec = mol.n_electrons() as usize;
        let result = run_rhf(
            &provider,
            n_elec,
            mol.nuclear_repulsion(),
            &ScfOptions::default(),
        )
        .unwrap_or_else(|err| panic!("{}/{}: {err}", entry.molecule, entry.basis));

        assert!(
            result.converged,
            "{}/{}: direct SCF did not converge",
            entry.molecule, entry.basis
        );

        let s = mat_to_row_major(&provider.overlap());
        let n_check = trace_ds(&result.density, &s, result.n_basis);
        assert!(
            (n_check - n_elec as f64).abs() < 1e-9,
            "{}/{}: Tr(DS) = {n_check}, expected {n_elec}",
            entry.molecule,
            entry.basis
        );

        let delta = result.energy - entry.energy;
        worst = worst.max(delta.abs());
        assert!(
            delta.abs() < 1e-7,
            "RHF-direct {}/{}: hartree {:.10} vs ORCA {:.10} (Δ = {:.2e})",
            entry.molecule,
            entry.basis,
            result.energy,
            entry.energy,
            delta
        );
        checked += 1;
    }
    eprintln!("RHF (direct): {checked} references matched ORCA; worst Δ = {worst:.2e}");
    checked
}

fn check_uhf_direct(basis_filter: impl Fn(&str) -> bool) -> usize {
    let geoms = geometries();
    let refs = references();

    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in refs
        .entries
        .iter()
        .filter(|e| e.method == "uhf" && basis_filter(&e.basis))
    {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let n_elec = mol.n_electrons() as usize;
        let (na, nb) = occupations(n_elec, mol.multiplicity);
        let provider = direct_provider_for(&mol, &entry.basis);
        let result = run_scf(
            &provider,
            na,
            nb,
            Reference::Uhf,
            mol.nuclear_repulsion(),
            &ScfOptions::default(),
        )
        .unwrap_or_else(|err| panic!("{}/{}: {err}", entry.molecule, entry.basis));

        assert!(
            result.converged,
            "{}/{} UHF-direct did not converge",
            entry.molecule, entry.basis
        );

        let de = result.energy - entry.energy;
        worst = worst.max(de.abs());
        assert!(
            de.abs() < 1e-6,
            "UHF-direct {}/{}: hartree {:.10} vs ORCA {:.10} (Δ = {:.2e})",
            entry.molecule,
            entry.basis,
            result.energy,
            entry.energy,
            de
        );

        if let Some(ref_s2) = entry.s2 {
            assert!(
                (result.spin_squared - ref_s2).abs() < 1e-4,
                "UHF-direct {}/{} <S²>: hartree {:.5} vs ORCA {:.5}",
                entry.molecule,
                entry.basis,
                result.spin_squared,
                ref_s2
            );
        }
        checked += 1;
    }
    eprintln!("UHF (direct): {checked} references matched ORCA; worst Δ = {worst:.2e}");
    checked
}

fn is_fast_direct_basis(b: &str) -> bool {
    matches!(b, "sto-3g" | "6-31g" | "def2-svp")
}

#[test]
fn rhf_direct_matches_orca_references() {
    let checked = check_rhf_direct(is_fast_direct_basis);
    assert!(
        checked >= 6,
        "expected ≥6 small-basis RHF references, checked {checked}"
    );
}

#[test]
fn uhf_direct_matches_orca_references() {
    let checked = check_uhf_direct(is_fast_direct_basis);
    assert!(
        checked >= 1,
        "expected ≥1 small-basis UHF reference, checked {checked}"
    );
}

#[test]
#[ignore = "direct cc-pVDZ recomputes ERIs each iteration; slow in debug, run --release"]
fn rhf_direct_ccpvdz() {
    let checked = check_rhf_direct(|b| b == "cc-pvdz");
    assert!(
        checked >= 1,
        "expected ≥1 cc-pVDZ RHF reference, checked {checked}"
    );
}

#[test]
#[ignore = "direct cc-pVDZ recomputes ERIs each iteration; slow in debug, run --release"]
fn uhf_direct_ccpvdz() {
    let checked = check_uhf_direct(|b| b == "cc-pvdz");
    assert!(
        checked >= 1,
        "expected ≥1 cc-pVDZ UHF reference, checked {checked}"
    );
}

#[test]
#[ignore = "QZ HF via DirectProvider (cc-pVQZ/def2-QZVP, minutes-class); run with --release --ignored"]
fn rhf_qz_spherical_g_matches_orca() {
    let geoms = geometries();
    let refs = references();

    const TOL: f64 = 1e-9;
    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in refs.entries.iter().filter(|e| {
        e.method == "rhf"
            && e.molecule == "water"
            && matches!(e.basis.as_str(), "cc-pvqz" | "def2-qzvp")
    }) {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let provider = direct_provider_for(&mol, &entry.basis);
        let n_elec = mol.n_electrons() as usize;
        let n_bf = provider.n_basis();

        let options = ScfOptions {
            incremental_fock: true,
            ..ScfOptions::default()
        };
        let t0 = Instant::now();
        let result = run_rhf(&provider, n_elec, mol.nuclear_repulsion(), &options)
            .unwrap_or_else(|err| panic!("{}/{}: {err}", entry.molecule, entry.basis));
        let secs = t0.elapsed().as_secs_f64();
        assert!(
            result.converged,
            "{}/{}: direct SCF did not converge",
            entry.molecule, entry.basis
        );

        let s = mat_to_row_major(&provider.overlap());
        let n_check = trace_ds(&result.density, &s, result.n_basis);
        assert!(
            (n_check - n_elec as f64).abs() < 1e-9,
            "{}/{}: Tr(DS) = {n_check}, expected {n_elec}",
            entry.molecule,
            entry.basis
        );

        let delta = result.energy - entry.energy;
        worst = worst.max(delta.abs());
        eprintln!(
            "RHF-direct {}/{} ({n_bf} bf, {secs:.1} s): hartree {:.10} vs ORCA {:.10} (Δ = {delta:.2e})",
            entry.molecule, entry.basis, result.energy, entry.energy,
        );
        assert!(
            delta.abs() < TOL,
            "RHF-direct {}/{}: hartree {:.12} vs ORCA {:.12} (Δ = {:.2e})",
            entry.molecule,
            entry.basis,
            result.energy,
            entry.energy,
            delta
        );
        checked += 1;
    }

    eprintln!(
        "RHF QZ spherical-g (direct): {checked} references matched ORCA; worst Δ = {worst:.2e}"
    );
    assert_eq!(
        checked, 2,
        "expected 2 water QZ g-tier references (cc-pVQZ + def2-QZVP; \
         h2s QZ HF oracle bounded out per the nao⁴ ceiling, cross-checked ORCA↔PySCF)"
    );
}
