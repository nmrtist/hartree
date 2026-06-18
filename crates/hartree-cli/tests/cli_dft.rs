use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static XYZ_SEQ: AtomicU32 = AtomicU32::new(0);

fn write_xyz(name: &str, body: &str) -> std::path::PathBuf {
    let seq = XYZ_SEQ.fetch_add(1, Ordering::Relaxed);
    let stem = name.strip_suffix(".xyz").unwrap_or(name);
    let path = std::env::temp_dir().join(format!("{stem}_{}_{seq}.xyz", std::process::id()));
    std::fs::write(&path, body).unwrap();
    path
}

fn water() -> std::path::PathBuf {
    write_xyz(
        "hartree_dft_water.xyz",
        "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n",
    )
}

fn oh() -> std::path::PathBuf {
    write_xyz(
        "hartree_dft_oh.xyz",
        "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n",
    )
}

fn hartree(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_hartree"))
        .args(args)
        .output()
        .expect("run hartree")
}

#[test]
fn pbe_rks_water_converges_and_reports() {
    let xyz = water();
    let out = hartree(&[xyz.to_str().unwrap(), "--basis", "6-31g", "--method", "pbe"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("converged in"), "not converged:\n{stdout}");
    assert!(
        stdout.contains("Kohn-Sham DFT: pbe (RKS)"),
        "no KS line:\n{stdout}"
    );
    assert!(stdout.contains("E_xc"), "no E_xc:\n{stdout}");
    assert!(stdout.contains("grid level 3"), "no grid line:\n{stdout}");
}

#[test]
fn pbe_open_shell_is_uks() {
    let xyz = oh();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--basis",
        "6-31g",
        "--method",
        "pbe",
        "--spin",
        "2",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exit failure:\n{stdout}");
    assert!(stdout.contains("(UKS)"), "open shell not UKS:\n{stdout}");
    assert!(stdout.contains("converged in"), "not converged:\n{stdout}");
}

#[test]
fn b3lyp_reports_exact_exchange() {
    let xyz = water();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--basis",
        "6-31g",
        "--method",
        "b3lyp",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exit failure:\n{stdout}");
    assert!(
        stdout.contains("exact exchange c_x"),
        "hybrid did not report c_x:\n{stdout}"
    );
}

#[test]
fn properties_on_rks_density() {
    let xyz = water();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--basis",
        "6-31g",
        "--method",
        "pbe",
        "--properties",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "RKS properties run failed:\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("converged in"), "not converged:\n{stdout}");
    assert!(
        stdout.contains("Kohn-Sham DFT: pbe (RKS)"),
        "no KS line:\n{stdout}"
    );
    assert!(
        stdout.contains("dipole moment"),
        "no dipole on RKS density:\n{stdout}"
    );
    assert!(
        stdout.contains("Mayer bond orders"),
        "no populations on RKS density:\n{stdout}"
    );
}

#[test]
fn functional_with_direct_is_allowed() {
    let xyz = water();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--basis",
        "6-31g",
        "--method",
        "pbe",
        "--direct",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "direct DFT should be allowed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("converged in"), "not converged:\n{stdout}");
}

#[test]
fn grid_without_functional_errors() {
    let xyz = water();
    let out = hartree(&[xyz.to_str().unwrap(), "--method", "rhf", "--grid", "3"]);
    assert!(
        !out.status.success(),
        "--grid without functional should fail"
    );
}

#[test]
fn grid_out_of_range_errors() {
    let xyz = water();
    let out = hartree(&[xyz.to_str().unwrap(), "--method", "pbe", "--grid", "9"]);
    assert!(!out.status.success(), "--grid 9 should fail");
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn freq_with_functional_works() {
    let xyz = water();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "pbe",
        "--freq",
        "--symmetry-number",
        "2",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "--freq with PBE should work:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("harmonic vibrational frequencies"),
        "no frequency section:\n{stdout}"
    );
    assert!(
        stdout.contains("RRHO thermochemistry"),
        "no RRHO section:\n{stdout}"
    );
    assert!(
        stdout.contains("quasi-RRHO (mRRHO)"),
        "no mRRHO section:\n{stdout}"
    );
}

#[test]
fn freq_uhf_radical_works_with_qrrho_w0() {
    let xyz = oh();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "uhf",
        "--spin",
        "2",
        "--freq",
        "--qrrho-w0",
        "50",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "--freq with UHF should work:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("harmonic vibrational frequencies"),
        "no frequency section:\n{stdout}"
    );
    assert!(
        stdout.contains("quasi-RRHO (mRRHO)") && stdout.contains("w0 = 50.0"),
        "no mRRHO section with w0 = 50:\n{stdout}"
    );
    assert!(
        stdout.contains("G(mRRHO)"),
        "no mRRHO Gibbs line:\n{stdout}"
    );
}

#[test]
fn qrrho_w0_must_be_positive() {
    let xyz = oh();
    let out = hartree(&[xyz.to_str().unwrap(), "--freq", "--qrrho-w0", "-5"]);
    assert!(!out.status.success(), "--qrrho-w0 -5 should fail");
}

#[test]
fn freq_with_post_hf_errors() {
    let xyz = water();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "mp2",
        "--freq",
    ]);
    assert!(!out.status.success(), "--freq with MP2 should fail");
}

#[test]
fn unknown_method_still_errors() {
    let xyz = water();
    let out = hartree(&[xyz.to_str().unwrap(), "--method", "not_a_method"]);
    assert!(!out.status.success(), "unknown method should fail");
}
