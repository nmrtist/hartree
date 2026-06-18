mod scf_common;

use hartree::integrals::IntegralProvider;
use hartree::linalg::mat_to_row_major;
use hartree::scf::{ScfOptions, run_rhf};

use scf_common::{geometries, provider_for, references, trace_ds};

fn check_rhf(basis_ok: impl Fn(&str) -> bool) -> usize {
    let geoms = geometries();
    let refs = references();
    assert_eq!(refs.provenance.engine, "ORCA");

    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in refs
        .entries
        .iter()
        .filter(|e| e.method == "rhf" && e.molecule != "benzene" && basis_ok(&e.basis))
    {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let provider = provider_for(&mol, &entry.basis);
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
            "{}/{}: SCF did not converge",
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
            "RHF {}/{}: hartree {:.10} vs ORCA {:.10} (Δ = {:.2e})",
            entry.molecule,
            entry.basis,
            result.energy,
            entry.energy,
            delta
        );
        checked += 1;
    }

    eprintln!("RHF: {checked} references matched ORCA; worst Δ = {worst:.2e}");
    checked
}

#[test]
fn rhf_matches_orca_references() {
    let checked = check_rhf(|b| matches!(b, "sto-3g" | "6-31g"));
    assert!(
        checked >= 4,
        "expected ≥4 s/p RHF references, checked {checked}"
    );
}

#[test]
#[ignore = "cc-pVDZ/def2-SVP RHF (slow tier per the tiering policy); run with --release -- --ignored"]
fn rhf_matches_orca_references_dz() {
    let checked = check_rhf(|b| matches!(b, "cc-pvdz" | "def2-svp"));
    assert!(
        checked >= 4,
        "expected ≥4 double-zeta RHF references, checked {checked}"
    );
}

#[test]
#[ignore = "benzene/6-31G (66 bf) is slow in debug; run with --release"]
fn rhf_benzene_6_31g() {
    let geoms = geometries();
    let refs = references();
    let entry = refs
        .entries
        .iter()
        .find(|e| e.molecule == "benzene" && e.basis == "6-31g" && e.method == "rhf")
        .expect("benzene/6-31g/rhf reference present");

    let mol = geoms.molecules["benzene"].molecule();
    let provider = provider_for(&mol, "6-31g");
    let n_elec = mol.n_electrons() as usize;
    let result = run_rhf(
        &provider,
        n_elec,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
    )
    .unwrap();
    assert!(result.converged, "benzene/6-31g SCF did not converge");

    let s = mat_to_row_major(&provider.overlap());
    let n_check = trace_ds(&result.density, &s, result.n_basis);
    assert!((n_check - n_elec as f64).abs() < 1e-9, "Tr(DS) = {n_check}");

    let delta = result.energy - entry.energy;
    eprintln!(
        "RHF benzene/6-31g: hartree {:.10} vs ORCA {:.10} (Δ {:.2e})",
        result.energy, entry.energy, delta
    );
    assert!(
        delta.abs() < 1e-7,
        "RHF benzene/6-31g: hartree {:.10} vs ORCA {:.10} (Δ {:.2e})",
        result.energy,
        entry.energy,
        delta
    );
}

#[test]
#[ignore = "6-311G triple-zeta family (slow tier per the tiering policy); run with --release -- --ignored"]
fn rhf_pople_6311_matches_orca() {
    let geoms = geometries();
    let refs = references();

    const TOL: f64 = 1e-9;
    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in refs
        .entries
        .iter()
        .filter(|e| e.method == "rhf" && e.basis.starts_with("6-311"))
    {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let provider = provider_for(&mol, &entry.basis);
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
            "{}/{}: SCF did not converge",
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
            delta.abs() < TOL,
            "RHF {}/{}: hartree {:.12} vs ORCA {:.12} (Δ = {:.2e})",
            entry.molecule,
            entry.basis,
            result.energy,
            entry.energy,
            delta
        );
        checked += 1;
    }

    eprintln!("RHF 6-311G family: {checked} references matched ORCA; worst Δ = {worst:.2e}");
    assert_eq!(
        checked, 8,
        "expected 8 6-311G-family references (water+h2s × 4 sets)"
    );
}

#[test]
#[ignore = "TZ HF oracle (cc-pVTZ/def2-TZVP, in-core nao⁴) is slow; run with --release --ignored"]
fn rhf_tz_spherical_f_matches_orca() {
    let geoms = geometries();
    let refs = references();

    const TOL: f64 = 1e-9;
    let mut checked = 0;
    let mut worst = 0.0_f64;
    for entry in refs
        .entries
        .iter()
        .filter(|e| e.method == "rhf" && matches!(e.basis.as_str(), "cc-pvtz" | "def2-tzvp"))
    {
        let mol = geoms.molecules[&entry.molecule].molecule();
        let provider = provider_for(&mol, &entry.basis);
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
            "{}/{}: SCF did not converge",
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
            delta.abs() < TOL,
            "RHF {}/{}: hartree {:.12} vs ORCA {:.12} (Δ = {:.2e})",
            entry.molecule,
            entry.basis,
            result.energy,
            entry.energy,
            delta
        );
        checked += 1;
    }

    eprintln!("RHF TZ spherical-f: {checked} references matched ORCA; worst Δ = {worst:.2e}");
    assert_eq!(
        checked, 4,
        "expected 4 compact TZ f-tier references (water+h2s × cc-pVTZ/def2-TZVP; \
         aug-cc-pVTZ HF oracle deliberately dropped from the fast tier)"
    );
}

#[test]
fn homo_lumo_gap_consistent_with_orbital_energies() {
    let mol = geometries().molecules["water"].molecule();
    let provider = provider_for(&mol, "sto-3g");
    let result = run_rhf(
        &provider,
        10,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
    )
    .unwrap();
    assert!(result.converged);

    let (gap_a, gap_b) = result.homo_lumo_gap();
    let n_occ = result.n_alpha;
    let expect = result.orbital_energies_alpha[n_occ] - result.orbital_energies_alpha[n_occ - 1];
    assert_eq!(gap_a, Some(expect));
    assert_eq!(gap_b, Some(expect), "RHF β gap mirrors α");
    assert!(
        expect > 0.0,
        "water/STO-3G gap should be positive: {expect}"
    );
}

#[test]
fn rhf_water_sto3g_crawford_anchor() {
    let mol = geometries().molecules["water"].molecule();
    assert!(
        (mol.nuclear_repulsion() - 8.002367061810).abs() < 1e-6,
        "nuclear repulsion = {}",
        mol.nuclear_repulsion()
    );

    let provider = provider_for(&mol, "sto-3g");
    let result = run_rhf(
        &provider,
        10,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
    )
    .unwrap();

    assert!(
        (result.energy - (-74.942079928192)).abs() < 1e-6,
        "RHF/STO-3G water = {}, Crawford -74.942079928192",
        result.energy
    );
}
