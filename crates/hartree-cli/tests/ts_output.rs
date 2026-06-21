//! Integration coverage for the `--ts-output` machine-readable transition-state
//! JSON: a small HCN→HNC saddle search at RHF/sto-3g must write a JSON file that
//! parses and carries the expected status/energy/geometry/verification fields.

use std::process::Command;

use serde_json::Value;

/// A near-saddle HCN→HNC guess (the hydrogen migrating across the C–N bond),
/// the same geometry the bench harness uses for this reaction.
const HCN_TS_GUESS: &str = "3\n\
hcn_hnc ts guess\n\
C        0.0000000000       0.0000000000       0.0000000000\n\
N        1.2210000000       0.0000000000       0.0000000000\n\
H        0.3550000000       1.1480000000       0.0000000000\n";

#[test]
fn ts_output_writes_parseable_json() {
    let dir = std::env::temp_dir();
    let xyz = dir.join("hartree_cli_ts_output_guess.xyz");
    let out = dir.join("hartree_cli_ts_output_result.json");
    // A stale file from a previous run must not be mistaken for this run's output.
    let _ = std::fs::remove_file(&out);
    std::fs::write(&xyz, HCN_TS_GUESS).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hartree"))
        .arg(&xyz)
        .args(["--ts", "--ts-algo", "prfo"])
        .args(["--method", "rhf", "--basis", "sto-3g"])
        .arg("--ts-output")
        .arg(&out)
        .output()
        .expect("run hartree --ts");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit failure.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        out.exists(),
        "--ts-output file was not written.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    let text = std::fs::read_to_string(&out).expect("read ts-output json");
    assert!(
        text.ends_with('\n'),
        "json should end with a trailing newline"
    );
    let json: Value = serde_json::from_str(&text).expect("ts-output is valid JSON");

    // Core fields are always present.
    assert!(
        json.get("status").and_then(Value::as_str).is_some(),
        "status"
    );
    assert!(
        json.get("converged").and_then(Value::as_bool).is_some(),
        "converged"
    );
    assert!(
        json.get("energy_eh").and_then(Value::as_f64).is_some(),
        "energy_eh"
    );
    assert!(
        json.get("n_steps").and_then(Value::as_u64).is_some(),
        "n_steps"
    );
    assert_eq!(json.get("method").and_then(Value::as_str), Some("rhf"));
    assert_eq!(json.get("basis").and_then(Value::as_str), Some("sto-3g"));

    // Geometry: one entry per atom, each with element + both unit systems.
    let geom = json
        .get("geometry")
        .and_then(Value::as_array)
        .expect("geometry array");
    assert_eq!(geom.len(), 3, "three atoms in the geometry");
    for atom in geom {
        assert!(atom.get("element").and_then(Value::as_str).is_some());
        assert_eq!(
            atom.get("xyz_angstrom")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(3)
        );
        assert_eq!(
            atom.get("xyz_bohr").and_then(Value::as_array).map(Vec::len),
            Some(3)
        );
    }

    // This guess is expected to converge to a verified first-order saddle, so the
    // harmonic verification block (imaginary frequency + mode count) is present.
    assert_eq!(json.get("converged").and_then(Value::as_bool), Some(true));
    assert_eq!(
        json.get("status").and_then(Value::as_str),
        Some("converged")
    );
    let verification = json
        .get("verification")
        .and_then(Value::as_object)
        .expect("verification block on a converged saddle");
    assert_eq!(
        verification
            .get("n_imaginary_modes")
            .and_then(Value::as_u64),
        Some(1),
        "a first-order saddle has exactly one imaginary mode"
    );
    assert!(
        verification
            .get("imaginary_frequency_cm1")
            .and_then(Value::as_f64)
            .is_some_and(|f| f < 0.0),
        "imaginary frequency is reported (negative by convention)"
    );

    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&xyz);
}
