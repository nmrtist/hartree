use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn h2_xyz() -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("hartree_cli_mw_h2_{}_{n}.xyz", std::process::id()));
    std::fs::write(&path, "2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n").unwrap();
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
fn pbe_on_pople_basis_collects_dft_warnings() {
    let xyz = h2_xyz();
    let (ok, stdout, stderr) = run(&[xyz.to_str().unwrap(), "--basis", "6-31g", "--method", "pbe"]);
    assert!(ok, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(
        stdout.contains("method-quality assessment"),
        "no assessment section:\n{stdout}"
    );
    assert!(
        stdout.contains("warning: Pople basis 6-31g with DFT"),
        "no Pople/DFT warning:\n{stdout}"
    );
    assert!(
        stdout.contains("minimal/unpolarized"),
        "no small-basis warning:\n{stdout}"
    );
    assert!(
        stdout.contains("systematically underestimate"),
        "no pure-GGA barrier note:\n{stdout}"
    );
    assert!(
        stdout.contains("without a dispersion correction")
            && stdout.contains("D4 (parameter set \"pbe\")"),
        "no metadata-driven dispersion warning:\n{stdout}"
    );
}

#[test]
fn cc_basis_warns_with_dft_not_with_mp2() {
    let xyz = h2_xyz();
    let (ok, stdout, _) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "cc-pvdz",
        "--method",
        "pbe0",
    ]);
    assert!(ok);
    assert!(
        stdout.contains("correlation-consistent (cc-pVnZ) basis cc-pvdz with DFT"),
        "no cc/DFT warning:\n{stdout}"
    );

    let (ok, stdout, _) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "cc-pvdz",
        "--method",
        "mp2",
    ]);
    assert!(ok);
    assert!(
        !stdout.contains("with DFT"),
        "cc basis warned under MP2:\n{stdout}"
    );
}

#[test]
fn fixed_conditions_clear_their_warnings() {
    let xyz = h2_xyz();
    let (ok, stdout, _) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "def2-svp",
        "--method",
        "pbe-d4",
    ]);
    assert!(ok, "{stdout}");
    assert!(
        !stdout.contains("without a dispersion correction"),
        "dispersion warning despite -d4:\n{stdout}"
    );
    assert!(
        !stdout.contains("with DFT") && !stdout.contains("minimal/unpolarized"),
        "basis warnings despite def2-svp:\n{stdout}"
    );
    assert!(
        stdout.contains("systematically underestimate"),
        "missing GGA note:\n{stdout}"
    );
}

#[test]
fn no_method_warnings_flag_suppresses_section() {
    let xyz = h2_xyz();
    let (ok, stdout, stderr) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "6-31g",
        "--method",
        "pbe",
        "--no-method-warnings",
    ]);
    assert!(ok);
    assert!(
        !stdout.contains("method-quality assessment"),
        "section despite --no-method-warnings:\n{stdout}"
    );
    assert!(
        !stderr.contains("--recommend general"),
        "pointer despite --no-method-warnings:\n{stderr}"
    );
}

#[test]
fn hf_sto3g_notes_correlation_and_small_basis() {
    let xyz = h2_xyz();
    let (ok, stdout, _) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "rhf",
    ]);
    assert!(ok);
    assert!(
        stdout.contains("note: HF neglects electron correlation"),
        "no HF note:\n{stdout}"
    );
    assert!(
        stdout.contains("warning: basis sto-3g is minimal/unpolarized"),
        "no small-basis warning:\n{stdout}"
    );
}

#[test]
fn grid_sensitive_functional_on_lowered_grid_warns() {
    let xyz = h2_xyz();
    let (ok, stdout, _) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "def2-svp",
        "--method",
        "m06-2x",
        "--grid",
        "2",
    ]);
    assert!(ok, "{stdout}");
    assert!(
        stdout.contains("m06-2x is grid-sensitive and grid level 2 is below"),
        "no coarse-grid warning:\n{stdout}"
    );

    let (ok, stdout, _) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "def2-svp",
        "--method",
        "m06-2x",
    ]);
    assert!(ok, "{stdout}");
    assert!(
        !stdout.contains("is grid-sensitive and grid level"),
        "grid warning at the recommended default:\n{stdout}"
    );
}

#[test]
fn missing_method_prints_recommendation_pointer() {
    let xyz = h2_xyz();
    let (ok, _, stderr) = run(&[xyz.to_str().unwrap(), "--basis", "sto-3g"]);
    assert!(ok);
    assert!(
        stderr.contains("defaulting to RHF") && stderr.contains("hartree --recommend general"),
        "no default-method pointer:\n{stderr}"
    );
    let (ok, _, stderr) = run(&[
        xyz.to_str().unwrap(),
        "--basis",
        "sto-3g",
        "--method",
        "rhf",
    ]);
    assert!(ok);
    assert!(
        !stderr.contains("defaulting to RHF"),
        "pointer despite explicit --method:\n{stderr}"
    );
}

#[test]
fn recommend_outputs_match_snapshots() {
    let (ok, stdout, _) = run(&["--recommend", "general"]);
    assert!(ok);
    assert!(stdout.contains("recommended level of theory: general"));
    assert!(stdout.contains("level:     r2scan-3c (geometry optimization + frequencies)"));
    assert!(stdout.contains("rationale:"));
    assert!(stdout.contains("hartree molecule.xyz --method r2scan-3c --opt"));

    for task in ["barriers", "nci"] {
        let (ok, stdout, _) = run(&["--recommend", task]);
        assert!(ok, "{task}");
        assert!(
            stdout.contains("wb97m-v/def2-TZVPP single point on a r2scan-3c geometry"),
            "{task}:\n{stdout}"
        );
        assert!(
            stdout.contains("hartree molecule.xyz --method \"wb97m-v/def2-tzvpp // r2scan-3c\""),
            "{task}:\n{stdout}"
        );
        assert!(
            stdout.contains("analytic gradients are unavailable for RS hybrids"),
            "{task}:\n{stdout}"
        );
    }

    let (ok, stdout, _) = run(&["--recommend", "thermochemistry"]);
    assert!(ok);
    assert!(
        stdout.contains("hartree molecule.xyz --method \"wb97m-v/def2-tzvpp // r2scan-3c\" --freq")
    );
    assert!(stdout.contains("G = E_high + (G_low - E_low)"));

    let (ok, stdout, _) = run(&["--recommend", "kinetics"]);
    assert!(ok);
    assert!(stdout.contains("recommended level of theory: barriers"));

    let (ok, _, stderr) = run(&["--recommend", "everything"]);
    assert!(!ok);
    assert!(stderr.contains("unknown --recommend task"), "{stderr}");
    // Assert each available task is listed rather than a brittle contiguous
    // substring, so adding a recommendation task doesn't break this test.
    for task in hartree::guardrails::recommendation_tasks() {
        assert!(stderr.contains(task), "missing `{task}` in: {stderr}");
    }
}
