use std::process::Command;

const WATER_DIMER: &str = "6\nwater dimer\n\
O -1.551007 -0.114520  0.000000\n\
H -1.934259  0.762503  0.000000\n\
H -0.599677  0.040712  0.000000\n\
O  1.350625  0.111469  0.000000\n\
H  1.680398 -0.373741 -0.758561\n\
H  1.680398 -0.373741  0.758561\n";

fn run(xyz_name: &str, xyz: &str, args: &[&str]) -> (bool, String, String) {
    let path = std::env::temp_dir().join(xyz_name);
    std::fs::write(&path, xyz).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&path)
        .args(args)
        .output()
        .expect("run hartree");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn cp_water_dimer_prints_full_table() {
    let (ok, stdout, stderr) = run(
        "hartree_cli_cp_dimer.xyz",
        WATER_DIMER,
        &["--basis", "sto-3g", "--method", "rhf", "--cp", "3"],
    );
    assert!(ok, "exit failure.\nstdout:\n{stdout}\nstderr:\n{stderr}");
    for needle in [
        "counterpoise (Boys-Bernardi)",
        "E_AB^(AB)",
        "E_A^(AB)",
        "E_B^(AB)",
        "E_A^(A)",
        "E_B^(B)",
        "dE_int (uncorrected)",
        "dE_int^CP (corrected)",
        "delta_BSSE",
    ] {
        assert!(stdout.contains(needle), "missing {needle:?}:\n{stdout}");
    }
}

#[test]
fn ghost_xyz_runs_single_point() {
    let xyz =
        "4\nwater + ghost\nO 0 0 0.1178\nH 0 0.7555 -0.4712\nH 0 -0.7555 -0.4712\n@O 0 0 3.0\n";
    let (ok, stdout, stderr) = run(
        "hartree_cli_ghost.xyz",
        xyz,
        &["--basis", "sto-3g", "--method", "rhf"],
    );
    assert!(ok, "exit failure.\nstdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(
        stdout.contains("electrons: 10"),
        "ghost added electrons:\n{stdout}"
    );
    assert!(stdout.contains("converged in"));
}

#[test]
fn standalone_gcp_prints_breakdown() {
    let (ok, stdout, stderr) = run(
        "hartree_cli_gcp_dimer.xyz",
        WATER_DIMER,
        &["--basis", "sto-3g", "--method", "rhf", "--gcp", "r2scan-3c"],
    );
    assert!(ok, "exit failure.\nstdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(
        stdout.contains("gCP (def2-mTZVPP)"),
        "no gCP line:\n{stdout}"
    );
    assert!(stdout.contains("total energy incl. corrections"));
}

#[test]
fn gcp_misuse_rejected() {
    let (ok, _, stderr) = run(
        "hartree_cli_gcp_bad1.xyz",
        WATER_DIMER,
        &["--method", "r2scan-3c", "--gcp", "r2scan-3c"],
    );
    assert!(!ok && stderr.contains("composite"), "stderr:\n{stderr}");
    let (ok, _, stderr) = run(
        "hartree_cli_gcp_bad2.xyz",
        WATER_DIMER,
        &["--basis", "sto-3g", "--gcp", "nosuchset"],
    );
    assert!(!ok && stderr.contains("unknown gCP"), "stderr:\n{stderr}");
}

#[test]
fn dftc_keyword_cleanly_blocked() {
    for args in [
        &["--basis", "def2-svpd", "--method", "b97m-v-dftc"][..],
        &["--basis", "def2-svpd", "--method", "dft-c"][..],
        &["--basis", "def2-svpd", "--method", "rhf", "--gcp", "dft-c"][..],
    ] {
        let (ok, stdout, stderr) = run("hartree_cli_dftc.xyz", WATER_DIMER, args);
        assert!(!ok, "DFT-C must be refused.\nstdout:\n{stdout}");
        assert!(
            stderr.contains("DFT-C is not implemented")
                && stderr.contains("supplementary material")
                && stderr.contains("10.1063/1.4986962"),
            "blocked message missing for {args:?}:\n{stderr}"
        );
    }
}
