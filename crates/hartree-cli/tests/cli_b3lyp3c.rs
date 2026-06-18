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
        "hartree_b3lyp3c_water.xyz",
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
fn conflicting_basis_is_rejected() {
    let xyz = water();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--method",
        "b3lyp-3c",
        "--basis",
        "cc-pvdz",
    ]);
    assert!(!out.status.success(), "conflicting --basis should fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("implies the def2-mSVP basis"),
        "unexpected error:\n{stderr}"
    );
}

#[test]
fn dispersion_suffixes_are_rejected() {
    let xyz = water();
    for method in ["b3lyp-3c-d3", "b3lyp-3c-d4"] {
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
fn composite_runs_without_basis_flag_and_reports_components() {
    let xyz = water();
    let out = hartree(&[xyz.to_str().unwrap(), "--method", "b3lyp-3c"]);
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
        stdout.contains("b3lyp-3c composite:"),
        "breakdown header missing:\n{stdout}"
    );
    let scf = grab(&stdout, "E_SCF (b3lyp5)");
    let d3 = grab(&stdout, "E_D3(BJ)-ATM");
    let gcp = grab(&stdout, "E_gCP (DFT/SV(P))");
    let total = grab(&stdout, "composite total");
    assert!(gcp > 0.0, "gCP is a repulsive BSSE correction: {gcp}");
    assert!(d3 < 0.0, "D3 should be attractive here: {d3}");
    assert!(
        (scf + d3 + gcp - total).abs() < 1e-11,
        "components do not sum: {scf} + {d3} + {gcp} != {total}"
    );
    let out2 = hartree(&[
        xyz.to_str().unwrap(),
        "--method",
        "b3lyp-3c",
        "--basis",
        "def2-msvp",
    ]);
    assert!(out2.status.success(), "matching --basis must be accepted");
}
