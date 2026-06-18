use hartree::core::{Atom, Element, Molecule};
use hartree::disp::{D4Params, d4_energy, d4_energy_gradient};

fn cluster(charge: i32) -> Molecule {
    let atoms = vec![
        Atom::new(Element::from_symbol("O").unwrap(), [0.1, -0.2, 0.3]),
        Atom::new(Element::from_symbol("H").unwrap(), [1.7, 0.9, -0.1]),
        Atom::new(Element::from_symbol("C").unwrap(), [-2.3, 1.1, 0.7]),
        Atom::new(Element::from_symbol("H").unwrap(), [-3.1, 2.6, -0.4]),
        Atom::new(Element::from_symbol("N").unwrap(), [3.9, -1.8, 2.2]),
        Atom::new(Element::from_symbol("Cl").unwrap(), [-1.0, -3.5, -2.0]),
    ];
    Molecule::new(atoms, charge, 1)
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

fn fd_check(mol: &Molecule, params: &D4Params, label: &str) {
    let (_, grad) = d4_energy_gradient(mol, params);
    let h = 1e-4;
    for iat in 0..mol.atoms.len() {
        for k in 0..3 {
            let mut xp: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
            let mut xm = xp.clone();
            xp[iat][k] += h;
            xm[iat][k] -= h;
            let ep = d4_energy(&with_positions(mol, &xp), params);
            let em = d4_energy(&with_positions(mol, &xm), params);
            let fd = (ep - em) / (2.0 * h);
            assert!(
                (grad[iat][k] - fd).abs() <= 1e-9,
                "{label} atom {iat} comp {k}: analytic {:.6e} vs FD {:.6e}",
                grad[iat][k],
                fd
            );
        }
    }
}

#[test]
fn analytic_gradient_matches_finite_differences() {
    for (charge, methods) in [
        (0, vec!["pbe", "b3lyp", "r2scan", "hf"]),
        (1, vec!["pbe"]),
        (-1, vec!["b3lyp"]),
    ] {
        let mol = cluster(charge);
        for method in methods {
            let params = D4Params::for_method(method).unwrap();
            fd_check(&mol, &params, &format!("{method} charge {charge}"));
        }
    }
}

#[test]
fn analytic_gradient_matches_finite_differences_without_atm() {
    for charge in [0, 1] {
        let mol = cluster(charge);
        let mut params = D4Params::for_method("pbe").unwrap();
        params.s9 = 0.0;
        fd_check(&mol, &params, &format!("pbe s9=0 charge {charge}"));
    }
}

#[test]
fn energy_is_translation_and_rotation_invariant() {
    let mol = cluster(0);
    let params = D4Params::for_method("pbe").unwrap();
    let e0 = d4_energy(&mol, &params);

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
    let e_shift = d4_energy(&with_positions(&mol, &shifted), &params);
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
    let e_rot = d4_energy(&with_positions(&mol, &rotated), &params);
    assert!((e0 - e_rot).abs() <= 1e-13, "rotation: {e0} vs {e_rot}");
}

#[test]
fn gradient_has_zero_net_force_and_torque() {
    for charge in [0, -1] {
        let mol = cluster(charge);
        for method in ["pbe", "blyp", "b3lyp", "pbe0", "tpss", "r2scan", "hf"] {
            let params = D4Params::for_method(method).unwrap();
            let (_, grad) = d4_energy_gradient(&mol, &params);
            for k in 0..3 {
                let total: f64 = grad.iter().map(|g| g[k]).sum();
                assert!(
                    total.abs() <= 1e-12,
                    "{method} charge {charge}: net force component {k} = {total:.3e}"
                );
            }
            let mut torque = [0.0; 3];
            for (a, g) in mol.atoms.iter().zip(&grad) {
                let r = a.position;
                torque[0] += r[1] * g[2] - r[2] * g[1];
                torque[1] += r[2] * g[0] - r[0] * g[2];
                torque[2] += r[0] * g[1] - r[1] * g[0];
            }
            for (k, t) in torque.iter().enumerate() {
                assert!(
                    t.abs() <= 1e-12,
                    "{method} charge {charge}: net torque component {k} = {t:.3e}"
                );
            }
        }
    }
}

#[test]
fn atm_vanishes_for_diatomics() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_symbol("Cl").unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_symbol("Cl").unwrap(), [0.0, 0.0, 3.8]),
        ],
        0,
        1,
    );
    let with_atm = D4Params::for_method("pbe").unwrap();
    let mut without_atm = with_atm;
    without_atm.s9 = 0.0;
    let e1 = d4_energy(&mol, &with_atm);
    let e2 = d4_energy(&mol, &without_atm);
    assert_eq!(e1, e2, "ATM must be exactly zero for a diatomic");
}

#[test]
fn two_body_term_is_attractive() {
    for r in [3.0, 5.0, 8.0, 15.0] {
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_symbol("Ar").unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_symbol("Ar").unwrap(), [0.0, 0.0, r]),
            ],
            0,
            1,
        );
        let mut params = D4Params::for_method("pbe").unwrap();
        params.s9 = 0.0;
        let e = d4_energy(&mol, &params);
        assert!(
            e < 0.0,
            "two-body dispersion at r = {r} should be attractive: {e}"
        );
    }
}

#[test]
fn charge_dependence_is_sane() {
    let energy_at = |charge: i32| {
        let mol = cluster(charge);
        d4_energy(&mol, &D4Params::for_method("pbe").unwrap())
    };
    let (anion, neutral, cation) = (energy_at(-1), energy_at(0), energy_at(1));
    assert!(
        anion < neutral && neutral < cation && cation < 0.0,
        "expected E(anion) < E(neutral) < E(cation) < 0, got {anion:.6e}, {neutral:.6e}, {cation:.6e}"
    );
}
