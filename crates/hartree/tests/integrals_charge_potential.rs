use hartree::integrals::{
    ConventionalProvider, DfProvider, DirectProvider, IntegralProvider, integral,
};
use hartree::linalg::mat_to_row_major;

fn shells() -> Vec<integral::Shell> {
    vec![
        integral::Shell::new(0, [0.0, 0.0, 0.0], vec![0.8], vec![1.0]).unwrap(),
        integral::Shell::new(0, [0.0, 0.0, 1.4], vec![1.1], vec![1.0]).unwrap(),
    ]
}

fn charges() -> Vec<([f64; 3], f64)> {
    vec![([0.0, 0.0, 0.0], 2.0), ([0.0, 0.0, 1.4], 1.0)]
}

#[test]
fn sharp_gaussian_limit_recovers_point_charge_nuclear() {
    let provider = ConventionalProvider::new(integral::Basis::new(shells()), charges());
    let n = provider.n_basis();

    let site = [0.3, -0.2, 0.9];
    let v_point = integral::Basis::new(shells()).nuclear(&[(site, 1.0)]);

    let t = provider.charge_potential_3c(&[(site, 1e6)]);
    assert_eq!(t.len(), n * n);

    for i in 0..n * n {
        let diff = (t[i] + v_point[i]).abs();
        assert!(
            diff < 1e-6,
            "element {i}: gaussian {} vs point {}",
            t[i],
            -v_point[i]
        );
    }
}

#[test]
fn layout_is_charge_fastest_and_backends_agree() {
    let pts = vec![([0.3, -0.2, 0.9], 2.5), ([-1.0, 0.4, 0.2], 4.0)];
    let conv = ConventionalProvider::new(integral::Basis::new(shells()), charges());
    let direct = DirectProvider::new(integral::Basis::new(shells()), charges());
    let n = conv.n_basis();

    let t = conv.charge_potential_3c(&pts);
    assert_eq!(t.len(), n * n * pts.len());
    assert_eq!(t, direct.charge_potential_3c(&pts));

    for (k, &pt) in pts.iter().enumerate() {
        let single = conv.charge_potential_3c(&[pt]);
        for i in 0..n * n {
            assert_eq!(single[i], t[i * pts.len() + k], "k={k}, i={i}");
        }
    }

    for k in 0..pts.len() {
        for mu in 0..n {
            for nu in 0..n {
                let a = t[(mu * n + nu) * pts.len() + k];
                let b = t[(nu * n + mu) * pts.len() + k];
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    assert!(t.iter().all(|&x| x > 0.0));

    let aux = integral::Basis::new(vec![
        integral::Shell::new(0, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap(),
    ]);
    let df = DfProvider::new(integral::Basis::new(shells()), &aux, charges()).unwrap();
    assert_eq!(t, df.charge_potential_3c(&pts));
    let _ = mat_to_row_major(&conv.overlap());
}
