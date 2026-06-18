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
fn smeared_rhf_prints_summary() {
    let xyz = water_xyz("hartree_cli_smear_rhf.xyz");
    let (ok, stdout, stderr) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "rhf",
        "--smear",
        "5000",
    ]);
    assert!(ok, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("Fermi smearing (T = 5000.0 K)"), "{stdout}");
    assert!(stdout.contains("T*S_el"), "{stdout}");
    assert!(stdout.contains("free energy F=E-T*S"), "{stdout}");
}

#[test]
fn smeared_ks_runs() {
    let xyz = water_xyz("hartree_cli_smear_ks.xyz");
    let (ok, stdout, stderr) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "svwn",
        "--grid",
        "0",
        "--smear",
        "2000",
    ]);
    assert!(ok, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("Fermi smearing"), "{stdout}");
    assert!(stdout.contains("Kohn-Sham"), "{stdout}");
}

#[test]
fn smearing_guards_reject_unsupported_combinations() {
    let xyz = water_xyz("hartree_cli_smear_guard.xyz");
    let path = xyz.to_str().unwrap();
    for (extra, needle) in [
        (vec!["--method", "rhf", "--opt"], "geometry optimization"),
        (vec!["--method", "rhf", "--freq"], "frequencies"),
        (vec!["--method", "mp2"], "not post-HF"),
        (vec!["--method", "ccsd"], "not post-HF"),
        (vec!["--method", "rohf"], "ROHF"),
    ] {
        let mut args = vec![path, "--basis", "sto-3g", "--smear", "5000"];
        args.extend(&extra);
        let (ok, stdout, stderr) = run(&args);
        assert!(!ok, "expected failure for {extra:?}:\n{stdout}");
        assert!(
            stderr.contains(needle),
            "guard message for {extra:?} should mention {needle:?}:\n{stderr}"
        );
    }

    let (ok, _, stderr) = run(&[path, "--smear", "-5"]);
    assert!(!ok);
    assert!(stderr.contains("positive"), "{stderr}");
}
