use hartree::solv::{DEFAULT_GBSA_GRID, alpb_solvent, gbsa_energy, gbsa_solvent};

fn water() -> (Vec<usize>, Vec<[f64; 3]>, Vec<f64>) {
    let zs = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.221_664_874_418_60],
        [0.0, 1.430_900_621_566_63, -0.886_659_497_674_41],
        [0.0, -1.430_900_621_566_63, -0.886_659_497_674_41],
    ];
    let qat = vec![-0.66, 0.33, 0.33];
    (zs, coords, qat)
}

fn approx(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() < tol,
        "{what}: {a:.10} vs {b:.10} (tol {tol:.0e})"
    );
}

#[test]
fn alpb_water_breakdown() {
    let (zs, coords, qat) = water();
    let p = alpb_solvent("water").expect("ALPB water");
    let bd = gbsa_energy(p, &zs, &coords, &qat, DEFAULT_GBSA_GRID).unwrap();
    approx(bd.g_born, -0.006_375_596_3, 1e-7, "ALPB water g_born");
    approx(bd.g_hb, -0.012_394_618_5, 1e-7, "ALPB water g_hb");
    approx(bd.g_sasa, 0.001_947_671_4, 1e-7, "ALPB water g_sasa");
    approx(bd.g_shift, 0.001_080_759_7, 1e-7, "ALPB water g_shift");
    approx(
        bd.g_solv,
        bd.g_born + bd.g_hb + bd.g_sasa + bd.g_shift,
        1e-12,
        "ALPB water sum",
    );
    assert!(bd.g_born < 0.0 && bd.g_hb < 0.0 && bd.g_sasa > 0.0);
}

#[test]
fn gbsa_water_breakdown() {
    let (zs, coords, qat) = water();
    let p = gbsa_solvent("water").expect("GBSA water");
    let bd = gbsa_energy(p, &zs, &coords, &qat, DEFAULT_GBSA_GRID).unwrap();
    approx(bd.g_born, -0.004_243_876_1, 1e-7, "GBSA water g_born");
    approx(bd.g_hb, -0.009_246_924_9, 1e-7, "GBSA water g_hb");
    approx(bd.g_sasa, 0.000_223_529_9, 1e-7, "GBSA water g_sasa");
    approx(bd.g_shift, 0.001_857_443_1, 1e-7, "GBSA water g_shift");
    let alpb = gbsa_energy(
        alpb_solvent("water").unwrap(),
        &zs,
        &coords,
        &qat,
        DEFAULT_GBSA_GRID,
    )
    .unwrap();
    assert!((bd.g_born - alpb.g_born).abs() > 1e-4);
}

#[test]
fn alpb_methanol_organic() {
    let (zs, coords, qat) = water();
    let p = alpb_solvent("methanol").expect("ALPB methanol");
    let bd = gbsa_energy(p, &zs, &coords, &qat, DEFAULT_GBSA_GRID).unwrap();
    approx(bd.g_born, -0.004_249_957_1, 1e-7, "ALPB methanol g_born");
    approx(bd.g_hb, -0.007_689_856_1, 1e-7, "ALPB methanol g_hb");
    approx(bd.g_sasa, -0.002_300_617_3, 1e-7, "ALPB methanol g_sasa");
    approx(bd.g_shift, 0.003_944_919_8, 1e-7, "ALPB methanol g_shift");
    let water = gbsa_energy(
        alpb_solvent("water").unwrap(),
        &zs,
        &coords,
        &qat,
        DEFAULT_GBSA_GRID,
    )
    .unwrap();
    assert!(bd.g_born > water.g_born);
}

#[test]
fn unknown_solvent_is_none() {
    assert!(alpb_solvent("unobtainium").is_none());
    assert!(gbsa_solvent("unobtainium").is_none());
    assert!(alpb_solvent("aniline").is_some());
    assert!(gbsa_solvent("aniline").is_none());
}
