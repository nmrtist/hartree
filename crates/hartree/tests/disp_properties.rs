use hartree::core::{Atom, Element, Molecule};
use hartree::disp::{D3Params, d3bj_energy, d3bj_energy_gradient};

fn cluster() -> Molecule {
    let atoms = vec![
        Atom::new(Element::from_symbol("O").unwrap(), [0.1, -0.2, 0.3]),
        Atom::new(Element::from_symbol("H").unwrap(), [1.7, 0.9, -0.1]),
        Atom::new(Element::from_symbol("C").unwrap(), [-2.3, 1.1, 0.7]),
        Atom::new(Element::from_symbol("H").unwrap(), [-3.1, 2.6, -0.4]),
        Atom::new(Element::from_symbol("N").unwrap(), [3.9, -1.8, 2.2]),
        Atom::new(Element::from_symbol("Cl").unwrap(), [-1.0, -3.5, -2.0]),
    ];
    Molecule::new(atoms, 0, 1)
}

fn with_positions(mol: &Molecule, xyz: &[[f64; 3]]) -> Molecule {
    let atoms = mol
        .atoms
        .iter()
        .zip(xyz)
        .map(|(a, p)| Atom::new(a.element, *p))
        .collect();
    Molecule::new(atoms, mol.charge, mol.multiplicity)
}

#[test]
fn analytic_gradient_matches_finite_differences() {
    let mol = cluster();
    for method in ["pbe", "b3lyp", "hf"] {
        let params = D3Params::for_method(method).unwrap();
        let (_, grad) = d3bj_energy_gradient(&mol, &params);
        let h = 1e-5;
        for iat in 0..mol.atoms.len() {
            for k in 0..3 {
                let mut xp: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
                let mut xm = xp.clone();
                xp[iat][k] += h;
                xm[iat][k] -= h;
                let ep = d3bj_energy(&with_positions(&mol, &xp), &params);
                let em = d3bj_energy(&with_positions(&mol, &xm), &params);
                let fd = (ep - em) / (2.0 * h);
                assert!(
                    (grad[iat][k] - fd).abs() <= 1e-8,
                    "{method} atom {iat} comp {k}: analytic {:.3e} vs FD {:.3e}",
                    grad[iat][k],
                    fd
                );
            }
        }
    }
}

#[test]
fn energy_is_translation_and_rotation_invariant() {
    let mol = cluster();
    let params = D3Params::for_method("pbe").unwrap();
    let e0 = d3bj_energy(&mol, &params);

    let shifted: Vec<[f64; 3]> = mol
        .atoms
        .iter()
        .map(|a| {
            [
                a.position[0] + 3.7,
                a.position[1] - 1.2,
                a.position[2] + 0.5,
            ]
        })
        .collect();
    let e_shift = d3bj_energy(&with_positions(&mol, &shifted), &params);
    assert!(
        (e0 - e_shift).abs() <= 1e-13,
        "translation: {e0} vs {e_shift}"
    );

    let (c1, s1) = (0.83f64.cos(), 0.83f64.sin());
    let (c2, s2) = (0.41f64.cos(), 0.41f64.sin());
    let rotated: Vec<[f64; 3]> = mol
        .atoms
        .iter()
        .map(|a| {
            let [x, y, z] = a.position;
            let (x1, y1) = (c1 * x - s1 * y, s1 * x + c1 * y);
            [x1, c2 * y1 - s2 * z, s2 * y1 + c2 * z]
        })
        .collect();
    let e_rot = d3bj_energy(&with_positions(&mol, &rotated), &params);
    assert!((e0 - e_rot).abs() <= 1e-13, "rotation: {e0} vs {e_rot}");
}

#[test]
fn gradient_sums_to_zero() {
    let mol = cluster();
    for method in ["pbe", "blyp", "b3lyp", "pbe0", "hf"] {
        let params = D3Params::for_method(method).unwrap();
        let (_, grad) = d3bj_energy_gradient(&mol, &params);
        for k in 0..3 {
            let total: f64 = grad.iter().map(|g| g[k]).sum();
            assert!(
                total.abs() <= 1e-12,
                "{method}: net force component {k} = {total:.3e}"
            );
        }
    }
}
