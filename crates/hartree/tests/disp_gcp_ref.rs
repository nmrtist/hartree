use hartree::core::{Atom, Element, Molecule};
use hartree::disp::{gcp_r2scan3c_energy, gcp_r2scan3c_energy_gradient};
use serde_json::Value;

fn molecules() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/ref/geometries.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn build(geom: &Value) -> Molecule {
    let atoms: Vec<Atom> = geom["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                Element::from_symbol(a[0].as_str().unwrap()).unwrap(),
                [
                    a[1].as_f64().unwrap(),
                    a[2].as_f64().unwrap(),
                    a[3].as_f64().unwrap(),
                ],
            )
        })
        .collect();
    Molecule::new(atoms, 0, 1)
}

#[test]
fn gcp_matches_mctc_gcp_reference() {
    let refs: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/ref/gcp.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let geoms = molecules();
    let mut checked = 0;
    for entry in refs["entries"].as_array().unwrap() {
        let name = entry["molecule"].as_str().unwrap();
        let mol = build(&geoms["molecules"][name]);
        let e_ref = entry["energy"].as_f64().unwrap();
        let (e, g) = gcp_r2scan3c_energy_gradient(&mol);
        assert!(
            (e - e_ref).abs() < 1e-9,
            "{name}: E_gCP {e:.12} vs ref {e_ref:.12} (|d| = {:.3e})",
            (e - e_ref).abs()
        );
        assert_eq!(
            gcp_r2scan3c_energy(&mol),
            e,
            "{name}: energy-only path drifted"
        );
        for (i, row) in entry["gradient"].as_array().unwrap().iter().enumerate() {
            for k in 0..3 {
                let r = row[k].as_f64().unwrap();
                assert!(
                    (g[i][k] - r).abs() < 1e-9,
                    "{name}: grad[{i}][{k}] {:.12e} vs ref {r:.12e}",
                    g[i][k]
                );
            }
        }
        checked += 1;
    }
    assert!(checked >= 6, "expected >= 6 oracle entries, got {checked}");
}
