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
fn solvent_hf_prints_solvation_block() {
    let xyz = water_xyz("hartree_cli_solv_water.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "sto-3g", "--method", "rhf", "--solvent", "water"])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("C-PCM solvation"),
        "no solvation block:\n{stdout}"
    );
    assert!(stdout.contains("E_solv"), "no E_solv line:\n{stdout}");
    assert!(stdout.contains("78.3553"), "no epsilon line:\n{stdout}");
    assert!(stdout.contains("converged in"), "not converged:\n{stdout}");
}

#[test]
fn eps_with_ri_dft_d3_runs() {
    let xyz = water_xyz("hartree_cli_solv_ri.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args([
            "--basis", "sto-3g", "--method", "pbe-d3", "--ri", "--grid", "1", "--eps", "36.7",
        ])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("C-PCM solvation"), "{stdout}");
    assert!(stdout.contains("36.7"), "no eps echo:\n{stdout}");
    assert!(stdout.contains("dispersion D3(BJ)"), "{stdout}");
    assert!(stdout.contains("RI-JK density fitting"), "{stdout}");
}

#[test]
fn smd_prints_solvation_block() {
    let xyz = water_xyz("hartree_cli_smd_water.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "sto-3g", "--method", "rhf", "--smd", "water"])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("SMD solvation"), "no SMD block:\n{stdout}");
    assert!(stdout.contains("ΔG_EP"), "no ΔG_EP line:\n{stdout}");
    assert!(stdout.contains("G_CDS"), "no G_CDS line:\n{stdout}");
    assert!(stdout.contains("ΔG_solv"), "no ΔG_solv line:\n{stdout}");
    assert!(
        stdout.contains("standard state"),
        "no std-state note:\n{stdout}"
    );
}

#[test]
fn smd_guards_reject_incompatible_flags() {
    let xyz = water_xyz("hartree_cli_smd_guards.xyz");
    let cases: &[(&[&str], &str)] = &[
        (
            &["--smd", "water", "--solvent", "water"],
            "mutually exclusive",
        ),
        (&["--smd", "water", "--eps", "78.0"], "mutually exclusive"),
        (&["--smd", "olive-oil"], "unknown SMD solvent"),
        (&["--smd", "water", "--method", "mp2"], "SCF-level"),
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
fn solvent_guards_reject_incompatible_flags() {
    let xyz = water_xyz("hartree_cli_solv_guards.xyz");
    let cases: &[(&[&str], &str)] = &[
        (
            &["--solvent", "water", "--eps", "78.0"],
            "mutually exclusive",
        ),
        (&["--solvent", "olive-oil"], "unknown solvent"),
        (&["--solvent", "water", "--freq"], "gas phase"),
        (&["--solvent", "water", "--method", "mp2"], "SCF-level"),
        (&["--solvent", "water", "--method", "ccsd"], "SCF-level"),
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
#[ignore = "slow; run with --include-ignored"]
fn opt_in_solvent_runs_through_fd() {
    let xyz = water_xyz("hartree_cli_solv_opt.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args([
            "--basis",
            "sto-3g",
            "--method",
            "rhf",
            "--solvent",
            "water",
            "--opt",
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
        stdout.contains("optimization converged"),
        "opt did not converge:\n{stdout}"
    );
}

#[test]
fn alpb_water_prints_breakdown() {
    let xyz = water_xyz("hartree_cli_alpb_water.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "sto-3g", "--method", "rhf", "--alpb", "water"])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "exit failure:\n{stdout}");
    assert!(
        stdout.contains("ALPB solvation"),
        "no ALPB block:\n{stdout}"
    );
    assert!(stdout.contains("G_born"), "no G_born:\n{stdout}");
    assert!(stdout.contains("G_sasa"), "no G_sasa:\n{stdout}");
    assert!(stdout.contains("ΔG_solv"), "no dG_solv:\n{stdout}");
}

#[test]
fn gbsa_water_prints_breakdown() {
    let xyz = water_xyz("hartree_cli_gbsa_water.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "sto-3g", "--method", "rhf", "--gbsa", "water"])
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "exit failure:\n{stdout}");
    assert!(
        stdout.contains("GBSA solvation"),
        "no GBSA block:\n{stdout}"
    );
}

#[test]
fn cosmo_file_written() {
    let xyz = water_xyz("hartree_cli_cosmo.xyz");
    let cosmo = std::env::temp_dir().join("hartree_cli_water.cosmo");
    let _ = std::fs::remove_file(&cosmo);
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--basis", "sto-3g", "--method", "rhf"])
        .arg("--cosmo-file")
        .arg(&cosmo)
        .output()
        .expect("run hartree");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "exit failure:\n{stdout}");
    assert!(
        stdout.contains("ideal conductor"),
        "no COSMO note:\n{stdout}"
    );
    let written = std::fs::read_to_string(&cosmo).expect("cosmo file");
    assert!(
        written.contains("$segment_information"),
        "bad cosmo:\n{written}"
    );
    assert!(
        written.contains("epsilon=infinity"),
        "no eps=inf:\n{written}"
    );
}

#[test]
fn alpb_and_smd_mutually_exclusive() {
    let xyz = water_xyz("hartree_cli_alpb_smd.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--alpb", "water", "--smd", "water"])
        .output()
        .expect("run hartree");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("mutually exclusive"), "stderr:\n{stderr}");
}

#[test]
fn gbsa_and_solvent_mutually_exclusive() {
    let xyz = water_xyz("hartree_cli_gbsa_solv.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--gbsa", "water", "--solvent", "water"])
        .output()
        .expect("run hartree");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("mutually exclusive"), "stderr:\n{stderr}");
}

#[test]
fn unknown_alpb_solvent_rejected() {
    let xyz = water_xyz("hartree_cli_alpb_unknown.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--alpb", "unobtainium"])
        .output()
        .expect("run hartree");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown ALPB solvent"), "stderr:\n{stderr}");
}

#[test]
fn alpb_opt_rejected() {
    let xyz = water_xyz("hartree_cli_alpb_opt.xyz");
    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--method", "rhf", "--alpb", "water", "--opt"])
        .output()
        .expect("run hartree");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("post-SCF single-point"),
        "stderr:\n{stderr}"
    );
}
