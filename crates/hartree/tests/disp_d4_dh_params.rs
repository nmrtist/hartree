use hartree::core::{Atom, Element, Molecule};
use hartree::disp::{D4Params, Dispersion, d4_energy};

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

#[test]
fn dh_param_sets_published_values() {
    let b2plyp = D4Params::for_method("b2plyp").expect("b2plyp D4 set");
    assert_eq!(
        (b2plyp.s6, b2plyp.s8, b2plyp.a1, b2plyp.a2, b2plyp.s9),
        (0.64, 1.16888646, 0.44154604, 4.73114642, 1.0)
    );

    let revdsd = D4Params::for_method("revdsdpbep86").expect("revdsdpbep86 D4 set");
    assert_eq!(
        (revdsd.s6, revdsd.s8, revdsd.a1, revdsd.a2, revdsd.s9),
        (0.5132, 0.0, 0.44, 3.60, 1.0)
    );

    let pwpb95 = D4Params::for_method("pwpb95").expect("pwpb95 D4 set");
    assert_eq!(
        (pwpb95.s6, pwpb95.s8, pwpb95.a1, pwpb95.a2, pwpb95.s9),
        (0.82, -0.34639127, 0.41080636, 3.83878274, 1.0)
    );

    assert!(D4Params::for_method("B2PLYP").is_some());
    assert!(matches!(
        Dispersion::for_method(true, "pwpb95"),
        Some(Dispersion::D4(_))
    ));
    assert!(Dispersion::for_method(false, "revdsdpbep86").is_none());
}

#[test]
fn s6_is_applied_linearly() {
    let mol = cluster();
    for key in ["b2plyp", "revdsdpbep86", "pwpb95"] {
        let p = D4Params::for_method(key).unwrap();
        let e = d4_energy(&mol, &p);
        let e0 = d4_energy(&mol, &D4Params { s6: 0.0, ..p });
        let e1 = d4_energy(&mol, &D4Params { s6: 1.0, ..p });
        assert!(
            (e - (e0 + p.s6 * (e1 - e0))).abs() < 1e-14,
            "{key}: E(s6) is not linear in s6"
        );
        assert!(
            (e - e1).abs() > 1e-8,
            "{key}: s6 = {} had no effect (E = {e}, E(s6=1) = {e1})",
            p.s6
        );
        assert!(((e - e0) / (e1 - e0) - p.s6).abs() < 1e-12, "{key}: ratio");
    }
}
