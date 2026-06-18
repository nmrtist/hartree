use hartree::integrals::{
    ConventionalProvider, DirectProvider, IntegralProvider, JkResult, integral,
};
use hartree::linalg::Mat;

fn small_basis() -> integral::Basis {
    let shells = vec![
        integral::Shell::new(0, [0.0, 0.0, 0.0], vec![1.3, 0.4], vec![0.6, 0.5]).unwrap(),
        integral::Shell::new(1, [0.0, 0.0, 1.2], vec![0.9], vec![1.0]).unwrap(),
    ];
    integral::Basis::new(shells)
}

#[test]
fn grid_coulomb_matches_negative_unit_nuclear() {
    let basis = small_basis();
    let reference = small_basis();
    let nao = basis.nao();
    let provider = ConventionalProvider::new(basis, vec![([0.0; 3], 1.0)]);

    let points: Vec<[f64; 3]> = (0..200)
        .map(|i| {
            let t = i as f64 * 0.037;
            [0.3 * t.sin(), 0.2 * t.cos(), 0.5 + 0.01 * t]
        })
        .collect();
    let a = provider.grid_coulomb(&points).expect("in-core supplies it");
    assert_eq!(a.len(), points.len() * nao * nao);

    for (g, p) in points.iter().enumerate() {
        let v = reference.nuclear(&[(*p, 1.0)]);
        for (idx, &vn) in v.iter().enumerate() {
            let got = a[g * nao * nao + idx];
            assert!(
                (got + vn).abs() <= 1e-12 * vn.abs().max(1.0),
                "point {g} element {idx}: A = {got}, -V_nuc = {}",
                -vn
            );
        }
    }
}

#[test]
fn direct_provider_supplies_grid_coulomb() {
    let conventional = ConventionalProvider::new(small_basis(), vec![([0.0; 3], 1.0)]);
    let direct = DirectProvider::new(small_basis(), vec![([0.0; 3], 1.0)]);
    let points = [[0.1, -0.2, 0.4], [1.0, 0.5, -0.3]];
    let a = conventional.grid_coulomb(&points).unwrap();
    let b = direct.grid_coulomb(&points).unwrap();
    assert_eq!(a, b);
}

#[test]
fn trait_default_declines() {
    struct Decliner;
    impl IntegralProvider for Decliner {
        fn n_basis(&self) -> usize {
            0
        }
        fn overlap(&self) -> Mat {
            unimplemented!()
        }
        fn kinetic(&self) -> Mat {
            unimplemented!()
        }
        fn nuclear(&self) -> Mat {
            unimplemented!()
        }
        fn build_jk(&self, _: &[Mat]) -> JkResult {
            unimplemented!()
        }
        fn dipole_integrals(&self, _: [f64; 3]) -> [Vec<f64>; 3] {
            unimplemented!()
        }
        fn ao_atom_indices(&self) -> Vec<usize> {
            unimplemented!()
        }
        fn charge_potential_3c(&self, _: &[([f64; 3], f64)]) -> Vec<f64> {
            unimplemented!()
        }
    }
    assert!(Decliner.grid_coulomb(&[[0.0; 3]]).is_none());
}
