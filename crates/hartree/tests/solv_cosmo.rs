use hartree::solv::{
    CosmoAtom, CosmoExport, CosmoSegment, CpcmSolver, build_surface, parse_cosmo, write_cosmo,
};

fn born_export(q: f64, r: f64, ng: usize) -> CosmoExport {
    let surface = build_surface(&[[0.0, 0.0, 0.0]], &[r], ng).unwrap();
    let solver = CpcmSolver::new(&surface, f64::INFINITY).unwrap();
    let v: Vec<f64> = surface
        .zeta
        .iter()
        .map(|&z| q * libm::erf(z * r) / r)
        .collect();
    let charges = solver.charges(&v);
    let diel_energy = 0.5 * charges.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
    const BOHR_TO_AA: f64 = 0.529_177_210_903;
    let segments: Vec<CosmoSegment> = (0..surface.points.len())
        .map(|k| CosmoSegment {
            atom: 1,
            position: surface.points[k],
            charge: charges[k],
            area: surface.area[k] * BOHR_TO_AA * BOHR_TO_AA,
            potential: v[k],
        })
        .collect();
    CosmoExport {
        epsilon: f64::INFINITY,
        total_energy: -42.0,
        dielectric_energy: diel_energy,
        atoms: vec![CosmoAtom {
            symbol: "Na".to_string(),
            position: [0.0, 0.0, 0.0],
            radius: r * BOHR_TO_AA,
        }],
        segments,
    }
}

#[test]
fn cosmo_roundtrip_and_gauss_law() {
    let q = 1.0;
    let export = born_export(q, 3.0, 302);
    let text = write_cosmo(&export);

    for header in [
        "$info",
        "$cosmo",
        "$cosmo_data",
        "$coord_rad",
        "$screening_charge",
        "$cosmo_energy",
        "$segment_information",
    ] {
        assert!(text.contains(header), "missing block {header}");
    }
    assert!(text.contains("epsilon=infinity"));

    let parsed = parse_cosmo(&text);
    assert!(parsed.epsilon_infinite);
    assert!((parsed.fepsi - 0.5).abs() < 1e-9);
    assert_eq!(parsed.n_atoms, 1);
    assert_eq!(parsed.segment_charges.len(), export.segments.len());

    let area_sum: f64 = parsed.segment_areas.iter().sum();
    assert!(
        (area_sum - export.total_area()).abs() < 1e-3,
        "area sum {area_sum} vs {}",
        export.total_area()
    );

    assert!((parsed.dielectric_energy - export.dielectric_energy).abs() < 1e-9);
    assert!((parsed.total_energy - (-42.0)).abs() < 1e-9);

    let charge_sum: f64 = parsed.segment_charges.iter().sum();
    assert!(
        (charge_sum + q).abs() < 1e-2,
        "screening-charge sum {charge_sum} should be ≈ {}",
        -q
    );
    assert!((parsed.screening_charge_total - charge_sum).abs() < 1e-5);
}

#[test]
fn cosmo_charge_scales_with_solute_charge() {
    for q in [1.0, 2.0] {
        let export = born_export(q, 3.0, 302);
        let sum: f64 = export.segments.iter().map(|s| s.charge).sum();
        assert!((sum + q).abs() < 1e-2 * q, "Q={q}: sum {sum}");
    }
}
