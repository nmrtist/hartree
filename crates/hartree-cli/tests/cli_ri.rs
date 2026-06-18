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

#[test]
fn ri_hf_runs_and_prints_aux_header() {
    let xyz = water_xyz("hartree_cli_ri_water.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "sto-3g", "--method", "rhf", "--ri"])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("RI-JK density fitting: aux basis def2-universal-jkfit"),
        "no RI aux header:\n{stdout}"
    );
    assert!(stdout.contains("aux fns"), "no naux in header:\n{stdout}");
    assert!(stdout.contains("converged in"), "not converged:\n{stdout}");
}

#[test]
fn ri_pbe0_runs() {
    let xyz = water_xyz("hartree_cli_ri_pbe0.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args([
            "--basis", "sto-3g", "--method", "pbe0", "--ri", "--grid", "1",
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
        stdout.contains("RI-JK density fitting"),
        "no RI header:\n{stdout}"
    );
    assert!(
        stdout.contains("exact exchange c_x"),
        "no hybrid c_x line:\n{stdout}"
    );
}

#[test]
fn ri_guards_reject_incompatible_flags() {
    let xyz = water_xyz("hartree_cli_ri_guards.xyz");
    let cases: &[(&[&str], &str)] = &[
        (&["--ri", "--direct"], "contradictory"),
        (&["--ri", "--opt"], "geometry optimization"),
        (&["--ri", "--properties"], "properties"),
        (&["--ri", "--freq"], "properties or frequencies"),
        (&["--ri", "--method", "mp2"], "post-HF"),
        (&["--ri", "--method", "ccsd"], "post-HF"),
        (&["--ri", "--method", "ccsd(t)"], "post-HF"),
    ];
    for (flags, needle) in cases {
        let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
            .arg(&xyz)
            .args(["--basis", "sto-3g"])
            .args(*flags)
            .output()
            .expect("run hartree");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{flags:?} unexpectedly succeeded");
        assert!(
            stderr.contains(needle),
            "{flags:?}: expected {needle:?} in stderr:\n{stderr}"
        );
    }
}

#[test]
fn aux_set_rejected_as_orbital_basis() {
    let xyz = water_xyz("hartree_cli_ri_auxbasis.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "def2-universal-jkfit", "--method", "rhf"])
        .output()
        .expect("run hartree");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("auxiliary fitting set"),
        "expected aux-set rejection message:\n{stderr}"
    );
}
