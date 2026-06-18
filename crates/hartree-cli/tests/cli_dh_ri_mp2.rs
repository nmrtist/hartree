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

#[test]
fn dh_ri_mp2_reports_backend() {
    let xyz = water_xyz("hartree_cli_dh_ri_mp2.xyz");
    let output = run_hartree(
        &xyz,
        &["--basis", "def2-svp", "--method", "b2plyp", "--ri-mp2"],
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("double hybrid b2plyp"), "{stdout}");
    assert!(
        stdout.contains("RI-MP2 (def2-svp/c)"),
        "PT2 backend line missing:\n{stdout}"
    );

    let output = run_hartree(&xyz, &["--basis", "def2-svp", "--method", "b2plyp"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("conventional MP2"),
        "conventional backend line missing:\n{stdout}"
    );
}
