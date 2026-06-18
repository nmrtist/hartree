use hartree::composite::composite;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::FunctionalSpec;
use hartree::disp::{D3Params, Dispersion, SrbParams, d3bj_energy, srb_energy};
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
    let c = composite("b97-3c").unwrap();
    JobOptions {
        grid_level: c.grid_level,
        dispersion: Some(c.dispersion),
        gcp: c.gcp,
        srb: c.srb,
        ..JobOptions::default()
    }
}

#[test]
fn correction_terms_match_fixture() {
    let refs = load("tests/ref/b973c.json");
    let c = composite("b97-3c").unwrap();
    assert!(c.gcp.is_none(), "B97-3c uses SRB instead of gCP (paper)");
    for entry in refs["entries"].as_array().unwrap() {
        let name = entry["molecule"].as_str().unwrap();
        let mol = molecule(name);
        let e_disp = c.dispersion.energy(&mol);
        let e_srb = srb_energy(&mol, &c.srb.unwrap());
        let d_disp = e_disp - entry["e_disp"].as_f64().unwrap();
        let d_srb = e_srb - entry["e_srb"].as_f64().unwrap();
        assert!(d_disp.abs() < 1e-12, "{name}: E_disp off by {d_disp:.2e}");
        assert!(d_srb.abs() < 1e-12, "{name}: E_SRB off by {d_srb:.2e}");
        assert!(e_srb < 0.0, "{name}: SRB must be attractive");
    }
}

#[test]
fn plain_functional_scf_is_bit_identical_under_composite_options() {
    let mol = molecule("water");
    let method = Method::Dft(FunctionalSpec::parse("gga_xc_b97_3c").unwrap());
    let bare = Job {
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
    let dressed = Job {
        molecule: mol,
        basis: "sto-3g".into(),
        method,
        options: JobOptions {
            grid_level: 0,
            ..composite_options()
        },
    }
    .run()
    .unwrap();
    assert_eq!(
        bare.scf.energy, dressed.scf.energy,
        "the corrections must never enter the Fock matrix / SCF energy"
    );
    assert!(dressed.dispersion_energy.is_some() && dressed.srb_energy.is_some());
    assert!(dressed.gcp_energy.is_none(), "B97-3c has no gCP term");
}

#[test]
fn composite_equals_manually_selected_parts() {
    let mol = molecule("water");
    let run = |options: JobOptions| {
        Job {
            molecule: mol.clone(),
            basis: "sto-3g".into(),
            method: Method::Dft(FunctionalSpec::parse("gga_xc_b97_3c").unwrap()),
            options,
        }
        .run()
        .unwrap()
    };
    let composite_run = run(JobOptions {
        grid_level: 0,
        ..composite_options()
    });
    let plain = run(JobOptions {
        grid_level: 0,
        ..JobOptions::default()
    });
    let e_manual = plain.scf.energy
        + d3bj_energy(&mol, &D3Params::B97_3C)
        + srb_energy(&mol, &SrbParams::B97_3C);
    let d = composite_run.best_energy() - e_manual;
    assert!(
        d.abs() < 1e-12,
        "composite vs manual parts: {d:.2e} ({} vs {e_manual})",
        composite_run.best_energy()
    );
    assert!(matches!(
        composite("b97-3c").unwrap().dispersion,
        Dispersion::D3(p) if p == D3Params::B97_3C
    ));
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn bare_functional_matches_pin() {
    let refs = load("tests/ref/b973c.json");
    let pin = &refs["bare_functional"];
    let result = Job {
        molecule: molecule(pin["molecule"].as_str().unwrap()),
        basis: "def2-mtzvp".into(),
        method: Method::Dft(FunctionalSpec::parse("gga_xc_b97_3c").unwrap()),
        options: JobOptions {
            grid_level: 3,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    assert!(result.scf.converged);
    let d = result.scf.energy - pin["e_scf"].as_f64().unwrap();
    assert!(d.abs() < 5e-9, "bare functional off by {d:.2e}");
    assert_eq!(result.dft.as_ref().unwrap().exx_fraction, 0.0);
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn composite_total_matches_fixture() {
    let refs = load("tests/ref/b973c.json");
    for entry in refs["entries"].as_array().unwrap() {
        let name = entry["molecule"].as_str().unwrap();
        let result = Job {
            molecule: molecule(name),
            basis: "def2-mtzvp".into(),
            method: Method::Dft(FunctionalSpec::parse("gga_xc_b97_3c").unwrap()),
            options: composite_options(),
        }
        .run()
        .unwrap();
        assert!(result.scf.converged, "{name}: SCF did not converge");
        let d_scf = result.scf.energy - entry["e_scf"].as_f64().unwrap();
        let d_disp = result.dispersion_energy.unwrap() - entry["e_disp"].as_f64().unwrap();
        let d_srb = result.srb_energy.unwrap() - entry["e_srb"].as_f64().unwrap();
        let d_tot = result.best_energy() - entry["e_total"].as_f64().unwrap();
        eprintln!(
            "b97-3c {name}: dSCF {d_scf:+.2e}  dD3 {d_disp:+.2e}  dSRB {d_srb:+.2e}  dtot {d_tot:+.2e}"
        );
        assert!(d_scf.abs() < 5e-9, "{name}: SCF off by {d_scf:.2e}");
        assert!(d_disp.abs() < 1e-12, "{name}: D3 off by {d_disp:.2e}");
        assert!(d_srb.abs() < 1e-12, "{name}: SRB off by {d_srb:.2e}");
        assert!(d_tot.abs() < 6e-9, "{name}: total off by {d_tot:.2e}");
    }
}

#[test]
fn surface_analytic_gradient_includes_corrections() {
    use hartree::HfSurface;
    use hartree::opt::Surface;
    use hartree::scf::Reference;

    let mol = molecule("water");
    let c = composite("b97-3c").unwrap();
    let spec = FunctionalSpec::parse(c.functional).unwrap();
    let mut surface = HfSurface::new_dft(&mol, "sto-3g", Reference::Rhf, spec, 0).unwrap();
    surface.set_dispersion(c.dispersion);
    surface.set_srb(c.srb.unwrap());

    let x0: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    let ga = surface
        .analytic_gradient(&x0)
        .expect("pure GGA takes the analytic path")
        .unwrap();
    let h = 1e-4;
    let mut worst = 0.0f64;
    for i in 0..x0.len() {
        for k in 0..3 {
            let mut xp = x0.clone();
            xp[i][k] += h;
            let mut xm = x0.clone();
            xm[i][k] -= h;
            let fd = (surface.energy(&xp).unwrap() - surface.energy(&xm).unwrap()) / (2.0 * h);
            worst = worst.max((ga[i][k] - fd).abs());
        }
    }
    assert!(worst < 5e-6, "surface FD arbiter: worst = {worst:.3e}");
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn composite_optimizes_water() {
    let start = Molecule::from_xyz(
        "3\nnear-minimum water (B97-3c)\nO 0 0 0.12145002\nH 0 0.76760467 -0.48222501\nH 0 -0.74760467 -0.46222501\n",
    )
    .unwrap();
    let result = Job {
        molecule: start,
        basis: "def2-mtzvp".into(),
        method: Method::Dft(FunctionalSpec::parse("gga_xc_b97_3c").unwrap()),
        options: JobOptions {
            optimize_geometry: true,
            ..composite_options()
        },
    }
    .run()
    .unwrap();
    let opt = result.optimized_geometry.as_ref().unwrap();
    assert!(opt.converged, "B97-3c H2O optimization must converge");
    assert!(result.srb_energy.is_some() && result.dispersion_energy.is_some());
    assert!(result.best_energy() < -76.37, "sane optimized total");
}
