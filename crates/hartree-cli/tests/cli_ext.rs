use std::process::Command;

fn write_tmp(name: &str, body: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(name);
    std::fs::write(&p, body).unwrap();
    p
}

const BUTANE: &str = "14
n-butane
C    -1.9255    -0.2545     0.0000
C    -0.6586     0.5872     0.0000
C     0.6586    -0.5872     0.0000
C     1.9255     0.2545     0.0000
H    -2.8190     0.3727     0.0000
H    -1.9698    -0.8907     0.8870
H    -1.9698    -0.8907    -0.8870
H    -0.6285     1.2335     0.8835
H    -0.6285     1.2335    -0.8835
H     0.6285    -1.2335     0.8835
H     0.6285    -1.2335    -0.8835
H     2.8190    -0.3727     0.0000
H     1.9698     0.8907     0.8870
H     1.9698     0.8907    -0.8870
";

#[test]
fn gfn2_xtb_absent_reports_named_error() {
    let xyz = write_tmp("hartree_ext_h2.xyz", "2\nh2\nH 0 0 0\nH 0 0 0.74\n");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--method", "gfn2-xtb"])
        .env_remove("HARTREE_XTB_PATH")
        .env("PATH", "") // ensure xtb is not found
        .output()
        .expect("run hartree");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "expected failure when xtb absent");
    assert!(
        stderr.contains("xtb binary not found"),
        "expected named error, got: {stderr}"
    );
    assert!(
        stderr.contains("HARTREE_XTB_PATH"),
        "error should mention the env override: {stderr}"
    );
}

#[test]
fn sph_requires_freq() {
    let xyz = write_tmp("hartree_ext_sph.xyz", "2\nh2\nH 0 0 0\nH 0 0 0.74\n");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--method", "rhf", "--sph"])
        .output()
        .expect("run hartree");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(stderr.contains("--sph"), "got: {stderr}");
}

#[test]
fn fallback_conformers_butane() {
    let xyz = write_tmp("hartree_ext_butane.xyz", BUTANE);
    let out_ens = std::env::temp_dir().join("hartree_ext_butane_ens.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args([
            "--conformers",
            "--method",
            "rhf",
            "--basis",
            "sto-3g",
            "--conformers-out",
            out_ens.to_str().unwrap(),
        ])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("conformer ensemble"),
        "no ensemble header:\n{stdout}"
    );
    assert!(
        stdout.contains("rotatable bonds: 1"),
        "wrong bond count:\n{stdout}"
    );
    assert!(
        stdout.contains("Boltzmann weights"),
        "no weights:\n{stdout}"
    );
    let ens = std::fs::read_to_string(&out_ens).unwrap();
    assert!(ens.starts_with("14"), "ensemble xyz malformed:\n{ens}");
}

#[test]
fn crest_absent_reports_named_error() {
    let xyz = write_tmp("hartree_ext_crest.xyz", "2\nh2\nH 0 0 0\nH 0 0 0.74\n");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--conformers", "crest", "--method", "gfn2-xtb"])
        .env_remove("HARTREE_CREST_PATH")
        .env("PATH", "")
        .output()
        .expect("run hartree");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("crest binary not found"),
        "expected named error, got: {stderr}"
    );
}
