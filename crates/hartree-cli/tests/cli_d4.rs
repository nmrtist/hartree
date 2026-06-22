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
        "hartree_d4_water.xyz",
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
        .find(|l| l.starts_with(label))
        .unwrap_or_else(|| panic!("no '{label}' line in:\n{stdout}"))
        .split_whitespace()
        .rev()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn pbe_d4_reports_dispersion_and_consistent_total() {
    let xyz = water();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "pbe-d4",
        "--grid",
        "1",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failure:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("Kohn-Sham DFT: pbe"),
        "no KS line:\n{stdout}"
    );
    let scf = grab(&stdout, "total energy  ");
    let disp = grab(&stdout, "dispersion D4");
    let total = grab(&stdout, "total energy + disp");
    assert!(disp < 0.0, "dispersion should be attractive: {disp}");
    assert!(
        (scf + disp - total).abs() < 1e-12,
        "total mismatch: {scf} + {disp} != {total}"
    );
}

#[test]
fn r2scan_d4_runs_and_reports() {
    let xyz = water();
    let out = hartree(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "r2scan-d4",
        "--grid",
        "1",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failure:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("dispersion D4"),
        "no dispersion line:\n{stdout}"
    );
}

#[test]
fn d4_energy_responds_to_net_charge() {
    let xyz = water();
    let run = |charge: &str| {
        let out = hartree(&[
            xyz.to_str().unwrap(),
            "--basis",
            "sto-3g",
            "--method",
            "hf-d4",
            "--charge",
            charge,
            "--multiplicity",
            if charge == "0" { "1" } else { "2" },
        ]);
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(
            out.status.success(),
            "exit failure:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        grab(&stdout, "dispersion D4")
    };
    let neutral = run("0");
    let cation = run("1");
    assert!(
        (neutral - cation).abs() > 1e-8,
        "D4 dispersion should respond to the net charge: {neutral} vs {cation}"
    );
}

#[test]
fn svwn_d4_errors_cleanly() {
    let xyz = water();
    let out = hartree(&[xyz.to_str().unwrap(), "--method", "svwn-d4"]);
    assert!(!out.status.success(), "svwn-d4 should fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no D4 parametrization"),
        "unexpected error:\n{stderr}"
    );
}

#[test]
fn post_hf_d4_errors_cleanly() {
    let xyz = water();
    for method in ["mp2-d4", "ccsd-d4", "ccsd(t)-d4"] {
        let out = hartree(&[xyz.to_str().unwrap(), "--method", method]);
        assert!(!out.status.success(), "{method} should fail");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("not supported for post-HF"),
            "{method} unexpected error:\n{stderr}"
        );
    }
}

#[test]
fn combined_d3_d4_suffixes_error_cleanly() {
    let xyz = water();
    for method in ["pbe-d3-d4", "pbe-d4-d3"] {
        let out = hartree(&[xyz.to_str().unwrap(), "--method", method]);
        assert!(!out.status.success(), "{method} should fail");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("unknown method") || stderr.contains("parametrization"),
            "{method} unexpected error:\n{stderr}"
        );
    }
}
