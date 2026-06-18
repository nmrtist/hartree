use std::process::Command;

fn water_xyz(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    std::fs::write(
        &path,
        "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n",
    )
    .unwrap();
    path
}

fn run(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
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
fn fod_with_explicit_level_warns_and_reports() {
    let xyz = water_xyz("hartree_cli_fod_rhf.xyz");
    let (ok, stdout, stderr) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "rhf",
        "--fod",
        "--smear",
        "5000",
    ]);
    assert!(ok, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("FOD analysis (Grimme)"), "{stdout}");
    assert!(stdout.contains("N_FOD"), "{stdout}");
    assert!(
        stdout.contains("no significant static correlation"),
        "{stdout}"
    );
    assert!(
        stderr.contains("TPSS/def2-TZVP"),
        "expected the calibration warning:\n{stderr}"
    );
}

#[test]
fn fod_guards_reject_unsupported_combinations() {
    let xyz = water_xyz("hartree_cli_fod_guard.xyz");
    let path = xyz.to_str().unwrap();

    let (ok, _, stderr) = run(&[path, "--fod-cube", "x.cube"]);
    assert!(!ok);
    assert!(stderr.contains("--fod-cube requires --fod"), "{stderr}");

    for method in ["mp2", "ccsd"] {
        let (ok, stdout, stderr) = run(&[path, "--basis", "sto-3g", "--method", method, "--fod"]);
        assert!(!ok, "expected failure for {method}:\n{stdout}");
        assert!(stderr.contains("post-HF"), "{method}: {stderr}");
    }

    for backend in ["--direct", "--ri"] {
        let (ok, stdout, stderr) = run(&[
            path, "--basis", "sto-3g", "--method", "rhf", backend, "--fod",
        ]);
        assert!(!ok, "expected failure for {backend}:\n{stdout}");
        assert!(stderr.contains("in-core"), "{backend}: {stderr}");
    }
}
