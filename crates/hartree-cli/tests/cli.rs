use std::process::Command;

#[test]
fn runs_rhf_on_water() {
    let xyz = std::env::temp_dir().join("hartree_cli_water.xyz");
    std::fs::write(
        &xyz,
        "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "sto-3g", "--method", "rhf"])
        .output()
        .expect("run hartree");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("converged in"),
        "no convergence line:\n{stdout}"
    );
    assert!(stdout.contains("total energy"), "no energy line:\n{stdout}");
}

#[test]
fn reports_homo_lumo_gap() {
    let xyz = std::env::temp_dir().join("hartree_cli_water_gap.xyz");
    std::fs::write(
        &xyz,
        "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "sto-3g", "--method", "rhf"])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(
        stdout.contains("HOMO-LUMO gap"),
        "no HOMO-LUMO gap line:\n{stdout}"
    );
    assert!(
        !stdout.contains("warning: small HOMO-LUMO gap"),
        "unexpected gap warning for equilibrium water:\n{stdout}"
    );
}

#[test]
fn warns_on_small_gap_stretched_h2() {
    let xyz = std::env::temp_dir().join("hartree_cli_h2_stretched.xyz");
    std::fs::write(&xyz, "2\nstretched H2\nH 0.0 0.0 0.0\nH 0.0 0.0 6.0\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "sto-3g", "--method", "rhf"])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("warning: small HOMO-LUMO gap"),
        "no small-gap warning for stretched H2:\n{stdout}"
    );
}

#[test]
fn warns_on_large_t1_stretched_water() {
    let xyz = std::env::temp_dir().join("hartree_cli_water_stretched.xyz");
    std::fs::write(
        &xyz,
        "3\nwater, one O-H at 2.1x\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -1.5897 -0.987\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "6-31g", "--method", "ccsd"])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("T1 diagnostic"),
        "no T1 diagnostic line:\n{stdout}"
    );
    assert!(
        stdout.contains("warning: T1 diagnostic"),
        "no T1 warning for stretched water:\n{stdout}"
    );
}

#[test]
fn rejects_open_shell_rhf() {
    let xyz = std::env::temp_dir().join("hartree_cli_oh.xyz");
    std::fs::write(&xyz, "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--method", "rhf", "--spin", "2"])
        .output()
        .expect("run hartree");
    assert!(!output.status.success());
}
