use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::{ConventionalProvider, DirectProvider, IntegralProvider};
use hartree::linalg::{mat_from_row_major, mat_to_row_major};

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(
                Element::from_symbol("O").unwrap(),
                [0.0, -0.143225816552, 0.0],
            ),
            Atom::new(
                Element::from_symbol("H").unwrap(),
                [1.638036840407, 1.136548822547, 0.0],
            ),
            Atom::new(
                Element::from_symbol("H").unwrap(),
                [-1.638036840407, 1.136548822547, 0.0],
            ),
        ],
        0,
        1,
    )
}

fn setup(basis: &str) -> (ConventionalProvider, usize) {
    let mol = water();
    let ao = BasisSet::load(basis).unwrap().build(&mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let n = ao.n_ao();
    (ConventionalProvider::new(ao.into_integral(), charges), n)
}

fn test_density(p: &ConventionalProvider) -> Vec<f64> {
    mat_to_row_major(&p.overlap())
}

#[test]
fn large_omega_limit_recovers_coulomb_k() {
    let (p, n) = setup("6-31g");
    let d = test_density(&p);
    let dm = mat_from_row_major(n, &d);

    let k_coulomb = mat_to_row_major(&p.build_jk(std::slice::from_ref(&dm)).exchange[0]);
    let k_lr = mat_to_row_major(&p.build_k_erf(std::slice::from_ref(&dm), 1.0e4).unwrap()[0]);

    let scale = k_coulomb.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let max_diff = k_coulomb
        .iter()
        .zip(&k_lr)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_diff / scale < 1e-6,
        "K_LR(ω=1e4) differs from K by rel {:.2e}",
        max_diff / scale
    );
}

#[test]
fn attenuation_is_monotone_in_omega() {
    let (p, n) = setup("sto-3g");
    let d = test_density(&p);
    let dm = mat_from_row_major(n, &d);

    let trace = |k: &[f64]| -> f64 { d.iter().zip(k).map(|(a, b)| a * b).sum() };
    let k_full = trace(&mat_to_row_major(
        &p.build_jk(std::slice::from_ref(&dm)).exchange[0],
    ));
    let tr_at = |omega: f64| -> f64 {
        let (p, _) = setup("sto-3g");
        trace(&mat_to_row_major(
            &p.build_k_erf(std::slice::from_ref(&dm), omega).unwrap()[0],
        ))
    };
    let t_03 = tr_at(0.3);
    let t_10 = tr_at(1.0);
    assert!(t_03 > 0.0, "Tr(D·K_LR) should be positive, got {t_03}");
    assert!(
        t_03 < t_10 && t_10 < k_full,
        "expected Tr(D·K_LR(0.3)) < Tr(D·K_LR(1.0)) < Tr(D·K): {t_03}, {t_10}, {k_full}"
    );
}

#[test]
fn grid_coulomb_erf_matches_layout_and_limits() {
    let (p, n) = setup("6-31g");
    let points = [[0.1, -0.2, 0.4], [1.0, 0.5, -0.3]];
    let a_c = p.grid_coulomb(&points).unwrap();
    let a_lr = p.grid_coulomb_erf(&points, 0.3).unwrap();
    assert_eq!(a_c.len(), points.len() * n * n);
    assert_eq!(a_lr.len(), a_c.len());
    for g in 0..points.len() {
        for mu in 0..n {
            let idx = g * n * n + mu * n + mu;
            assert!(a_lr[idx] > 0.0 && a_lr[idx] <= a_c[idx] + 1e-14);
        }
    }
    let a_big = p.grid_coulomb_erf(&points, 1.0e4).unwrap();
    let scale = a_c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let max_diff = a_c
        .iter()
        .zip(&a_big)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_diff / scale < 1e-6,
        "A_LR(ω=1e4) differs from A_C by rel {:.2e}",
        max_diff / scale
    );
}

#[test]
fn direct_backend_declines() {
    let mol = water();
    let ao = BasisSet::load("sto-3g").unwrap().build(&mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let n = ao.n_ao();
    let p = DirectProvider::new(ao.into_integral(), charges);
    let d = mat_from_row_major(n, &vec![0.0; n * n]);
    assert!(p.build_k_erf(&[d], 0.3).is_none());
}
