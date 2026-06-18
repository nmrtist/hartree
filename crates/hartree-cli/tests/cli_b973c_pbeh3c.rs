use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static XYZ_SEQ: AtomicU32 = AtomicU32::new(0);

fn write_xyz(name: &str, body: &str) -> std::path::PathBuf {
    let seq = XYZ_SEQ.fetch_add(1, Ordering::Relaxed);
    let stem = name.strip_suffix(".xyz").unwrap_or(name);
    let path = std::env::temp_dir().join(format!("{stem}_{}_{seq}.xyz", std::process::id()));
    std::fs::write(&path, body).unwrap();
    path
}

fn water() -> std::path::PathBuf {
    write_xyz(
        "hartree_3c_water.xyz",
        "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n",
    )
}

fn hartree(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_hartree"))
        .args(args)
        .output()
        .expect("run hartree")
}

fn grab(stdout: &str, label: &str) -> f64 {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with(label))
        .unwrap_or_else(|| panic!("no '{label}' line in:\n{stdout}"))
        .split_whitespace()
        .rev()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn conflicting_basis_is_rejected_for_both() {
    let xyz = water();
    for (method, implied) in [("b97-3c", "mTZVP"), ("pbeh-3c", "def2-mSVP")] {
        let out = hartree(&[
            xyz.to_str().unwrap(),
            "--method",
            method,
            "--basis",
            "cc-pvdz",
        ]);
        assert!(!out.status.success(), "{method}: conflicting --basis");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(&format!("implies the {implied}")),
            "{method} unexpected error:\n{stderr}"
        );
    }
}

#[test]
fn dispersion_suffixes_are_rejected_for_both() {
    let xyz = water();
    for method in ["b97-3c-d3", "b97-3c-d4", "pbeh-3c-d3", "pbeh-3c-d4"] {
        let out = hartree(&[xyz.to_str().unwrap(), "--method", method]);
        assert!(!out.status.success(), "{method} should fail");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("defines its own D3(BJ) and short-range corrections"),
            "{method} unexpected error:\n{stderr}"
        );
    }
}

#[test]
fn b973c_runs_and_reports_srb_breakdown() {
    let xyz = water();
    let out = hartree(&[xyz.to_str().unwrap(), "--method", "b97-3c"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failure:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("def2-mtzvp"),
        "implied basis not reported:\n{stdout}"
    );
    assert!(
        stdout.contains("b97-3c composite:"),
        "breakdown header missing:\n{stdout}"
    );
    assert!(
        !stdout.contains("E_gCP"),
        "B97-3c must not print a gCP line:\n{stdout}"
    );
    let scf = grab(&stdout, "E_SCF (gga_xc_b97_3c)");
    let d3 = grab(&stdout, "E_D3(BJ)-ATM");
    let srb = grab(&stdout, "E_SRB");
    let total = grab(&stdout, "composite total");
    assert!(d3 < 0.0 && srb < 0.0, "D3 and SRB are attractive");
    assert!(
        (scf + d3 + srb - total).abs() < 1e-10,
        "total must be the sum: {scf} + {d3} + {srb} != {total}"
    );
}

#[test]
fn pbeh3c_runs_as_hybrid_and_reports_gcp_breakdown() {
    let xyz = water();
    let out = hartree(&[xyz.to_str().unwrap(), "--method", "pbeh-3c"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failure:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("def2-msvp"),
        "implied basis not reported:\n{stdout}"
    );
    assert!(
        stdout.contains("pbeh-3c composite:"),
        "breakdown header missing:\n{stdout}"
    );
    assert!(
        !stdout.contains("E_SRB"),
        "PBEh-3c must not print an SRB line:\n{stdout}"
    );
    assert!(
        stdout.contains("0.42") || stdout.contains("42"),
        "hybrid EXX fraction not reported:\n{stdout}"
    );
    let scf = grab(&stdout, "E_SCF (hyb_gga_xc_pbeh_3c)");
    let d3 = grab(&stdout, "E_D3(BJ)-ATM");
    let gcp = grab(&stdout, "E_gCP");
    let total = grab(&stdout, "composite total");
    assert!(d3 < 0.0 && gcp > 0.0, "D3 attractive, gCP repulsive");
    assert!(
        (scf + d3 + gcp - total).abs() < 1e-10,
        "total must be the sum: {scf} + {d3} + {gcp} != {total}"
    );
}

#[test]
fn blocked_stub_messages_are_gone() {
    for method in ["b97-3c", "pbeh-3c"] {
        let out = hartree(&["nonexistent_file.xyz", "--method", method]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("not available"),
            "{method} still emits a blocked-stub message:\n{stderr}"
        );
    }
}
