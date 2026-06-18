use hartree::periodic::{
    Basis, Cell, GridXc, RealSpaceGrid, Shell, build_local_ks, kinetic_energy, local_energy,
};
use std::f64::consts::PI;

fn s_basis(alpha: f64, c: [f64; 3]) -> Basis {
    Basis::new(vec![Shell::new(0, c, vec![alpha], vec![1.0]).unwrap()])
}

#[test]
fn slater_exchange_of_gaussian_matches_analytic() {
    let alpha = 0.5;
    let l = 20.0_f64;
    let n = (l / 0.2).round() as usize; // 100, 7-smooth
    let center = [l / 2.0; 3];
    let basis = s_basis(alpha, center);
    let grid = RealSpaceGrid::new(Cell::cubic(l).unwrap(), [n, n, n]);

    let n_r = hartree::periodic::collocate_density(&basis, &[1.0], &grid);
    let q: f64 = n_r.iter().sum::<f64>() * grid.dv();
    assert!((q - 1.0).abs() < 1e-4, "∫n should be 1, got {q}");

    let (e_x_grid, _v) = GridXc::slater_exchange()
        .unwrap()
        .energy_potential(&n_r, grid.dv())
        .unwrap();

    let beta = 2.0 * alpha;
    let a_norm = (2.0 * alpha / PI).powf(1.5);
    let cx = 0.75 * (3.0 / PI).powf(1.0 / 3.0);
    let e_x_analytic = -cx * a_norm.powf(4.0 / 3.0) * (3.0 * PI / (4.0 * beta)).powf(1.5);

    assert!(
        (e_x_grid - e_x_analytic).abs() < 2e-3,
        "grid Slater exchange {e_x_grid} vs analytic {e_x_analytic}"
    );
}

#[test]
fn kinetic_energy_matches_analytic() {
    let alpha = 0.7;
    let basis = s_basis(alpha, [5.0, 5.0, 5.0]);
    let e_kin = kinetic_energy(&basis, &[1.0]);
    let t = basis.kinetic();
    assert!((e_kin - t[0]).abs() < 1e-14);
    assert!(
        (e_kin - 1.5 * alpha).abs() < 1e-12,
        "T = {e_kin}, want 3α/2 = {}",
        1.5 * alpha
    );
}

#[test]
fn local_ks_build_is_consistent() {
    let l = 18.0;
    let n = 72;
    let grid = RealSpaceGrid::new(Cell::cubic(l).unwrap(), [n, n, n]);
    let basis = Basis::new(vec![
        Shell::new(0, [8.0, 9.0, 9.0], vec![0.6], vec![1.0]).unwrap(),
        Shell::new(0, [10.0, 9.0, 9.0], vec![0.9], vec![1.0]).unwrap(),
    ]);
    let nao = basis.nao();
    let p = vec![1.0, 0.2, 0.2, 1.0];
    let xc = GridXc::lda().unwrap();

    let comps = local_energy(&basis, &p, &grid, &xc).unwrap();
    assert!(comps.n_electrons > 0.0 && comps.e_hartree > 0.0);
    assert!(
        comps.e_xc < 0.0,
        "XC energy should be negative, got {}",
        comps.e_xc
    );

    let ks = build_local_ks(&basis, &p, &grid, &xc).unwrap();
    assert_eq!(ks.v_matrix.len(), nao * nao);
    assert!((ks.e_hartree - comps.e_hartree).abs() < 1e-12);
    assert!((ks.e_xc - comps.e_xc).abs() < 1e-12);
    let v_dot_p: f64 = ks.v_matrix.iter().zip(&p).map(|(&v, &pp)| v * pp).sum();
    assert!(v_dot_p.is_finite());
    assert!(v_dot_p > 0.0, "∫(V_H+V_xc) n = {v_dot_p}");
}
