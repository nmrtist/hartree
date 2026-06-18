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

fn run_hartree(xyz: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(xyz)
        .args(args)
        .output()
        .expect("run hartree")
}

fn total_energy(stdout: &str) -> f64 {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with("total energy"))
        .and_then(|l| l.split_whitespace().nth(2))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no total-energy line in:\n{stdout}"))
}

#[test]
fn cosx_rhf_def2svp_matches_incore() {
    let xyz = water_xyz("hartree_cli_cosx_water.xyz");
    let reference = run_hartree(&xyz, &["--basis", "def2-svp", "--method", "rhf"]);
    assert!(reference.status.success());
    let e_ref = total_energy(&String::from_utf8_lossy(&reference.stdout));

    let output = run_hartree(&xyz, &["--basis", "def2-svp", "--method", "rhf", "--cosx"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("COSX semi-numerical exchange: grid medium"),
        "no COSX header:\n{stdout}"
    );
    assert!(stdout.contains("overlap-fitted"), "no fit note:\n{stdout}");
    assert!(stdout.contains("converged in"), "not converged:\n{stdout}");

    let e_cosx = total_energy(&stdout);
    let de = (e_cosx - e_ref).abs();
    assert!(
        de <= 5e-5,
        "COSX RHF energy {e_cosx} vs in-core {e_ref}: |ΔE| = {de:e} > 5e-5 Eh"
    );
}

#[test]
fn cosx_pbe0_runs() {
    let xyz = water_xyz("hartree_cli_cosx_pbe0.xyz");
    let output = run_hartree(
        &xyz,
        &[
            "--basis", "sto-3g", "--method", "pbe0", "--cosx", "--grid", "1",
        ],
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("COSX semi-numerical exchange"),
        "no COSX header:\n{stdout}"
    );
    assert!(
        stdout.contains("exact exchange c_x"),
        "no hybrid c_x line:\n{stdout}"
    );
}

#[test]
fn cosx_with_ri_j_runs() {
    let xyz = water_xyz("hartree_cli_cosx_ri.xyz");
    let output = run_hartree(
        &xyz,
        &["--basis", "sto-3g", "--method", "rhf", "--ri", "--cosx"],
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("RI-JK density fitting"), "{stdout}");
    assert!(stdout.contains("COSX semi-numerical exchange"), "{stdout}");
    assert!(stdout.contains("converged in"), "{stdout}");
}

#[test]
fn cosx_runs_range_separated_functionals() {
    let xyz = water_xyz("hartree_cli_cosx_rs.xyz");
    let reference = run_hartree(
        &xyz,
        &["--basis", "sto-3g", "--method", "wb97m-v", "--grid", "1"],
    );
    assert!(reference.status.success());
    let e_ref = total_energy(&String::from_utf8_lossy(&reference.stdout));

    let output = run_hartree(
        &xyz,
        &[
            "--basis", "sto-3g", "--method", "wb97m-v", "--cosx", "--grid", "1",
        ],
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("COSX semi-numerical exchange")
            && stdout.contains("RS Coulomb + erf(omega=0.3) kernels"),
        "no RS-COSX header:\n{stdout}"
    );
    assert!(
        stdout.contains("VV10"),
        "VV10 must compose with RS-COSX:\n{stdout}"
    );
    let e_cosx = total_energy(&stdout);
    let de = (e_cosx - e_ref).abs();
    assert!(
        de <= 5e-5,
        "RS-COSX wb97m-v energy {e_cosx} vs in-core {e_ref}: |dE| = {de:e} > 5e-5 Eh"
    );
}

#[test]
fn cosx_guards_reject_incompatible_flags() {
    let xyz = water_xyz("hartree_cli_cosx_guards.xyz");
    let cases: &[(&[&str], &str)] = &[
        (&["--cosx", "--direct"], "--cosx with --direct"),
        (&["--cosx", "--method", "mp2"], "SCF-level methods only"),
        (&["--cosx", "--method", "ccsd"], "SCF-level methods only"),
        (&["--cosx", "--opt"], "geometry optimization"),
        (&["--cosx", "--freq"], "frequencies"),
        (&["--cosx", "--fod"], "FOD"),
        (&["--cosx", "--method", "b2plyp"], "double hybrid"),
    ];
    for (args, needle) in cases {
        let output = run_hartree(&xyz, &[&["--basis", "sto-3g"][..], args].concat());
        assert!(!output.status.success(), "{args:?} must fail");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(needle),
            "{args:?}: expected {needle:?} in:\n{stderr}"
        );
    }
}
