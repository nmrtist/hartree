use std::path::PathBuf;
use std::process::Command;

use hartree::BasisSet;
use hartree::cc::{CcsdOptions, frozen_core_orbitals, rccsd_spin_adapted, rhf_mp2};
use hartree::core::units::ANGSTROM_TO_BOHR;
use hartree::integrals::ConventionalProvider;
use hartree::scf::{ScfOptions, run_rhf};
use hartree::{Atom, Element, Molecule};

const WATER_XYZ: &str = "\
3
water
O   0.0000000000   0.0000000000   0.0000000000
H   0.7570000000   0.5860000000   0.0000000000
H  -0.7570000000   0.5860000000   0.0000000000
";

fn water_molecule() -> Molecule {
    let a = ANGSTROM_TO_BOHR;
    let o = Element::from_symbol("O").unwrap();
    let h = Element::from_symbol("H").unwrap();
    let atoms = vec![
        Atom::new(o, [0.0, 0.0, 0.0]),
        Atom::new(h, [0.757 * a, 0.586 * a, 0.0]),
        Atom::new(h, [-0.757 * a, 0.586 * a, 0.0]),
    ];
    Molecule::new(atoms, 0, 1)
}

fn library_reference() -> (f64, f64) {
    let mol = water_molecule();
    let ao = BasisSet::load("sto-3g").unwrap().build(&mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let provider = ConventionalProvider::new(ao.into_integral(), charges);
    let scf = run_rhf(
        &provider,
        mol.n_electrons() as usize,
        mol.nuclear_repulsion(),
        &ScfOptions {
            energy_tol: 1e-12,
            error_tol: 1e-10,
            ..ScfOptions::default()
        },
    )
    .unwrap();
    let n_frozen = frozen_core_orbitals(&mol); // O 1s → 1 (CLI default)
    let mp2 = rhf_mp2(&provider, &scf, n_frozen).correlation_energy;
    let ccsd =
        rccsd_spin_adapted(&provider, &scf, n_frozen, &CcsdOptions::default()).correlation_energy;
    (mp2, ccsd)
}

fn write_water_xyz() -> PathBuf {
    let path = std::env::temp_dir().join("hartree_cli_smoke_water.xyz");
    std::fs::write(&path, WATER_XYZ).unwrap();
    path
}

fn run_hartree(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .args(args)
        .output()
        .expect("failed to spawn hartree binary");
    assert!(
        out.status.success(),
        "hartree {:?} exited {:?}\nstderr:\n{}",
        args,
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("non-UTF8 stdout")
}

fn parse_correlation(stdout: &str) -> f64 {
    let line = stdout
        .lines()
        .find(|l| l.contains("correlation energy"))
        .unwrap_or_else(|| panic!("no 'correlation energy' line in:\n{stdout}"));
    line.split_whitespace()
        .rev()
        .nth(1) // ... <value> Eh
        .and_then(|tok| tok.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("could not parse correlation energy from: {line:?}"))
}

#[test]
fn cli_mp2_and_ccsd_match_library() {
    let xyz = write_water_xyz();
    let xyz = xyz.to_str().unwrap();
    let (mp2_ref, ccsd_ref) = library_reference();

    let mp2_out = run_hartree(&[xyz, "--basis", "sto-3g", "--method", "mp2"]);
    let mp2 = parse_correlation(&mp2_out);
    assert!(
        (mp2 - mp2_ref).abs() < 1e-9,
        "CLI MP2 {mp2:.12} vs library {mp2_ref:.12}"
    );

    let ccsd_out = run_hartree(&[xyz, "--basis", "sto-3g", "--method", "ccsd"]);
    let ccsd = parse_correlation(&ccsd_out);
    assert!(
        ccsd_out.contains("RHF-CCSD"),
        "CCSD report header missing:\n{ccsd_out}"
    );
    assert!(
        (ccsd - ccsd_ref).abs() < 1e-9,
        "CLI CCSD {ccsd:.12} vs library {ccsd_ref:.12}"
    );
}
