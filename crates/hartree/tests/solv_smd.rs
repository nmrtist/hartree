use hartree::solv::smd::{
    DEFAULT_SASA_GRID, SASA_PROBE_ANGSTROM, cds_energy, sasa, smd_coulomb_radius,
};
use hartree::solv::{SMD_SOLVENTS, smd_solvent};

const BOHR: f64 = 0.529_177_210_92;

#[test]
fn sasa_single_sphere_is_analytic() {
    let r = (1.70 + SASA_PROBE_ANGSTROM) / BOHR;
    let areas = sasa(&[[0.0, 0.0, 0.0]], &[r], DEFAULT_SASA_GRID).unwrap();
    let exact = 4.0 * std::f64::consts::PI * r * r;
    assert!(
        (areas[0] - exact).abs() / exact < 1e-10,
        "single-sphere SASA {} vs analytic {exact}",
        areas[0]
    );
}

#[test]
fn sasa_separated_atoms_sum() {
    let r = (1.70 + SASA_PROBE_ANGSTROM) / BOHR;
    let far = 50.0; // bohr; far beyond contact
    let areas = sasa(
        &[[0.0, 0.0, 0.0], [far, 0.0, 0.0]],
        &[r, r],
        DEFAULT_SASA_GRID,
    )
    .unwrap();
    let exact = 4.0 * std::f64::consts::PI * r * r;
    assert!((areas[0] - exact).abs() / exact < 1e-10);
    assert!((areas[1] - exact).abs() / exact < 1e-10);
}

#[test]
fn sasa_buried_atom_is_zero() {
    let big = 5.0 / BOHR;
    let small = 1.0 / BOHR;
    let areas = sasa(
        &[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        &[big, small],
        DEFAULT_SASA_GRID,
    )
    .unwrap();
    assert!(
        areas[1] < 1e-6 * 4.0 * std::f64::consts::PI * small * small,
        "buried-atom exposed area {} not ~0",
        areas[1]
    );
}

#[test]
fn smd_oxygen_radius_acidity_rule() {
    assert!((smd_coulomb_radius(8, 0.82).unwrap() - 1.52).abs() < 1e-12);
    assert!((smd_coulomb_radius(8, 0.0).unwrap() - 2.294).abs() < 1e-12);
    assert!((smd_coulomb_radius(1, 0.0).unwrap() - 1.20).abs() < 1e-12);
    assert!((smd_coulomb_radius(6, 0.0).unwrap() - 1.85).abs() < 1e-12);
    assert!((smd_coulomb_radius(7, 0.0).unwrap() - 1.89).abs() < 1e-12);
}

fn methane() -> (Vec<usize>, Vec<[f64; 3]>) {
    let d = 1.09 / BOHR; // C–H ≈ 1.09 Å
    let a = d / 3.0_f64.sqrt();
    let zs = vec![6, 1, 1, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a],
    ];
    (zs, coords)
}

#[test]
fn cds_size_consistent() {
    let water = smd_solvent("water").unwrap();
    let (zs, coords) = methane();
    let e1 = cds_energy(&zs, &coords, water, DEFAULT_SASA_GRID).unwrap();

    let mut zs2 = zs.clone();
    zs2.extend_from_slice(&zs);
    let shift = 200.0;
    let mut coords2 = coords.clone();
    coords2.extend(coords.iter().map(|c| [c[0] + shift, c[1], c[2]]));
    let e2 = cds_energy(&zs2, &coords2, water, DEFAULT_SASA_GRID).unwrap();
    assert!(
        (e2 - 2.0 * e1).abs() < 1e-8,
        "size consistency: {e2} vs 2·{e1}"
    );
}

#[test]
fn cds_smooth_under_displacement() {
    let water = smd_solvent("water").unwrap();
    let (zs, mut coords) = methane();
    let e0 = cds_energy(&zs, &coords, water, DEFAULT_SASA_GRID).unwrap();
    coords[1][0] += 1e-4;
    let e1 = cds_energy(&zs, &coords, water, DEFAULT_SASA_GRID).unwrap();
    assert!((e1 - e0).abs() < 1e-6, "Δ {} too large", e1 - e0);
}

#[test]
fn cds_solvent_dependence() {
    let (zs, coords) = methane();
    let water = cds_energy(
        &zs,
        &coords,
        smd_solvent("water").unwrap(),
        DEFAULT_SASA_GRID,
    )
    .unwrap();
    let hexane = cds_energy(
        &zs,
        &coords,
        smd_solvent("n-hexane").unwrap(),
        DEFAULT_SASA_GRID,
    )
    .unwrap();
    let toluene = cds_energy(
        &zs,
        &coords,
        smd_solvent("toluene").unwrap(),
        DEFAULT_SASA_GRID,
    )
    .unwrap();
    assert!((water - hexane).abs() > 1e-6);
    assert!((water - toluene).abs() > 1e-6);
    assert!((hexane - toluene).abs() > 1e-6);
}

#[test]
fn solvent_library_resolves() {
    for s in &SMD_SOLVENTS {
        assert_eq!(s.name, s.name.to_ascii_lowercase());
        assert!(smd_solvent(s.name).is_some());
        assert!(s.epsilon > 1.0);
    }
    assert!(smd_solvent("WATER").is_some()); // case-insensitive
    assert!(smd_solvent("nonexistent-solvent").is_none());
}
