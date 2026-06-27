use hartree::core::{Atom, Element, Molecule};
use hartree::dft::FunctionalSpec;
use hartree::disp::{D4Params, Dispersion, GcpParams};
use hartree::{Job, JobOptions, Method};
use serde_json::Value;

fn load(path: &str) -> Value {
    let full = format!("{}/../../{path}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(full).unwrap()).unwrap()
}

fn molecule(name: &str) -> Molecule {
    let geoms = load("tests/ref/geometries.json");
    let rec = &geoms["molecules"][name];
    let atoms = rec["atoms"]
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
    Molecule::new(
        atoms,
        rec["charge"].as_i64().unwrap() as i32,
        rec["multiplicity"].as_u64().unwrap() as u32,
    )
}

fn composite_options() -> JobOptions {
    JobOptions {
        // The composite ships grid level 3 in production; this oracle deliberately pins the
        // denser reference-quality level 4 so it validates the functional + composite
        // machinery against the external references at fixed, maximal integration accuracy.
        grid_level: 4,
        dispersion: Some(Dispersion::D4(D4Params::R2SCAN_3C)),
        gcp: Some(GcpParams::R2SCAN_3C),
        ..JobOptions::default()
    }
}

#[test]
fn composite_d4_matches_dftd4_oracle() {
    let refs = load("tests/ref/r2scan3c.json");
    let disp = Dispersion::D4(D4Params::R2SCAN_3C);
    for entry in refs["entries"].as_array().unwrap() {
        let name = entry["molecule"].as_str().unwrap();
        let mol = molecule(name);
        let (e, g) = disp.energy_gradient(&mol);
        let e_ref = entry["e_d4"].as_f64().unwrap();
        assert!(
            (e - e_ref).abs() < 1e-9,
            "{name}: E_D4 {e:.12} vs dftd4 {e_ref:.12}"
        );
        for (i, row) in entry["d4_gradient"].as_array().unwrap().iter().enumerate() {
            for k in 0..3 {
                let r = row[k].as_f64().unwrap();
                assert!(
                    (g[i][k] - r).abs() < 1e-9,
                    "{name}: D4 grad[{i}][{k}] {:.3e} vs {r:.3e}",
                    g[i][k]
                );
            }
        }
    }
}

#[test]
fn plain_r2scan_scf_is_bit_identical_under_composite_options() {
    let mol = molecule("water");
    let method = Method::Dft(FunctionalSpec::parse("r2scan").unwrap());
    let plain = Job {
        molecule: mol.clone(),
        basis: "sto-3g".into(),
        method: method.clone(),
        options: JobOptions {
            grid_level: 0,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    let composite = Job {
        molecule: mol,
        basis: "sto-3g".into(),
        method,
        options: JobOptions {
            grid_level: 0,
            dispersion: Some(Dispersion::D4(D4Params::R2SCAN_3C)),
            gcp: Some(GcpParams::R2SCAN_3C),
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    assert_eq!(
        plain.scf.energy, composite.scf.energy,
        "the corrections must never enter the Fock matrix / SCF energy"
    );
    assert!(composite.dispersion_energy.is_some() && composite.gcp_energy.is_some());
    assert_eq!(plain.dispersion_energy, None);
    assert_eq!(plain.gcp_energy, None);
}

#[test]
#[ignore = "TZ-class composite oracle (def2-mTZVPP, grid 4); run with --release -- --ignored"]
fn composite_total_matches_composed_oracle() {
    const SCF_GATE: f64 = 1.3e-5;
    let refs = load("tests/ref/r2scan3c.json");
    for entry in refs["entries"].as_array().unwrap() {
        let name = entry["molecule"].as_str().unwrap();
        let result = Job {
            molecule: molecule(name),
            basis: "def2-mtzvpp".into(),
            method: Method::Dft(FunctionalSpec::parse("r2scan").unwrap()),
            options: composite_options(),
        }
        .run()
        .unwrap();
        assert!(result.scf.converged, "{name}: SCF did not converge");
        let (e_scf, e_d4, e_gcp) = (
            entry["e_scf"].as_f64().unwrap(),
            entry["e_d4"].as_f64().unwrap(),
            entry["e_gcp"].as_f64().unwrap(),
        );
        let d_scf = result.scf.energy - e_scf;
        let d_d4 = result.dispersion_energy.unwrap() - e_d4;
        let d_gcp = result.gcp_energy.unwrap() - e_gcp;
        let d_tot = result.best_energy() - entry["e_total"].as_f64().unwrap();
        eprintln!(
            "r2scan-3c {name}: dSCF {d_scf:+.2e}  dD4 {d_d4:+.2e}  dgCP {d_gcp:+.2e}  dtot {d_tot:+.2e}"
        );
        assert!(d_scf.abs() < SCF_GATE, "{name}: SCF off by {d_scf:.2e}");
        assert!(d_d4.abs() < 1e-9, "{name}: D4 off by {d_d4:.2e}");
        assert!(d_gcp.abs() < 1e-9, "{name}: gCP off by {d_gcp:.2e}");
        assert!(
            d_tot.abs() < SCF_GATE + 2e-9,
            "{name}: composite total off by {d_tot:.2e}"
        );
    }
}

#[test]
#[ignore = "TZ-class composite optimization (def2-mTZVPP, grid 4); run with --ignored"]
fn composite_optimizes_hf_to_fixture() {
    const BOHR_PER_ANGSTROM: f64 = 1.889_726_124_626_18;
    let r_fixture = 0.92346832 * BOHR_PER_ANGSTROM;
    let hf = Molecule::from_xyz("2\nstretched hydrogen fluoride\nF 0.0 0.0 0.0\nH 0.0 0.0 0.95\n")
        .unwrap();
    let result = Job {
        molecule: hf,
        basis: "def2-mtzvpp".into(),
        method: Method::Dft(FunctionalSpec::parse("r2scan").unwrap()),
        options: JobOptions {
            optimize_geometry: true,
            ..composite_options()
        },
    }
    .run()
    .unwrap();
    let opt = result.optimized_geometry.as_ref().unwrap();
    assert!(opt.converged, "r2scan-3c HF optimization must converge");
    let d = [
        opt.positions[1][0] - opt.positions[0][0],
        opt.positions[1][1] - opt.positions[0][1],
        opt.positions[1][2] - opt.positions[0][2],
    ];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    eprintln!(
        "r2scan-3c HF opt: r {r:.6} bohr (fixture {r_fixture:.6})  E {:.10}",
        result.best_energy()
    );
    assert!(
        (r - r_fixture).abs() < 2e-3,
        "bond length {r:.6} vs fixture {r_fixture:.6} bohr"
    );
    assert!((result.best_energy() - -100.4438572991).abs() < 1e-6);
}
