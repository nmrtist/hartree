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

/// `--ts-product` is the two-endpoint TS input; without `--ts` it is a user error,
/// caught before any SCF (so this stays fast).
#[test]
fn ts_product_without_ts_is_rejected() {
    let xyz = std::env::temp_dir().join("hartree_cli_react.xyz");
    let prod = std::env::temp_dir().join("hartree_cli_prod.xyz");
    std::fs::write(&xyz, "2\nh2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
    std::fs::write(&prod, "2\nh2\nH 0 0 0\nH 0 0 0.90\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--method", "rhf", "--ts-product"])
        .arg(&prod)
        .output()
        .expect("run hartree");
    assert!(
        !output.status.success(),
        "should reject --ts-product sans --ts"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("add --ts"),
        "expected a '--ts' hint, got:\n{stderr}"
    );
}

/// `--ts-scan` also needs a second endpoint; without `--ts-product` it is rejected.
#[test]
fn ts_scan_without_product_is_rejected() {
    let xyz = std::env::temp_dir().join("hartree_cli_react3.xyz");
    std::fs::write(&xyz, "2\nh2\nH 0 0 0\nH 0 0 0.74\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--method", "rhf", "--ts", "--ts-scan", "7"])
        .output()
        .expect("run hartree");
    assert!(
        !output.status.success(),
        "should reject --ts-scan sans product"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--ts-product"),
        "expected a '--ts-product' hint, got:\n{stderr}"
    );
}

/// `--ts-neb` needs a second endpoint; without `--ts-product` it is rejected.
#[test]
fn ts_neb_without_product_is_rejected() {
    let xyz = std::env::temp_dir().join("hartree_cli_react2.xyz");
    std::fs::write(&xyz, "2\nh2\nH 0 0 0\nH 0 0 0.74\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--method", "rhf", "--ts", "--ts-neb"])
        .output()
        .expect("run hartree");
    assert!(
        !output.status.success(),
        "should reject --ts-neb sans product"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--ts-product"),
        "expected a '--ts-product' hint, got:\n{stderr}"
    );
}
