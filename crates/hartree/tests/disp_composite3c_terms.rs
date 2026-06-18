use hartree::core::Molecule;
use hartree::disp::{
    D3Params, GcpParams, d3bj_energy, d3bj_energy_gradient, gcp_energy, gcp_energy_gradient,
};

fn crooked() -> Molecule {
    Molecule::from_xyz(
        "8\n\n\
         C  0.0123  0.0456 -0.0789\n\
         H  1.0900  0.0200  0.1100\n\
         O  -0.6100  1.1300  0.2400\n\
         H  -1.5500  1.0200  0.0300\n\
         N  -0.5800 -1.2400  0.1900\n\
         H  -0.1200 -2.0700 -0.1700\n\
         S  1.2000 -1.5000  1.9000\n\
         Cl -1.9000 -0.4000 -1.8000\n",
    )
    .unwrap()
}

fn water() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n")
        .unwrap()
}

#[test]
fn gcp_svp_single_atom_is_zero() {
    for sym in ["H", "C", "O", "Cl"] {
        let mol = Molecule::from_xyz(&format!("1\n\n{sym} 0 0 0\n")).unwrap();
        assert_eq!(gcp_energy(&mol, &GcpParams::B3LYP_3C), 0.0, "{sym}");
    }
}

#[test]
fn gcp_svp_is_size_consistent_over_noninteracting_fragments() {
    let mono = water();
    let e1 = gcp_energy(&mono, &GcpParams::B3LYP_3C);
    assert!(e1 > 0.0, "gCP is a repulsive BSSE correction: {e1}");
    let dimer = Molecule::from_xyz(
        "6\n\n\
         O 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n\
         O 500.0 0.0 0.117\nH 500.0 0.757 -0.470\nH 500.0 -0.757 -0.470\n",
    )
    .unwrap();
    let e2 = gcp_energy(&dimer, &GcpParams::B3LYP_3C);
    assert!(
        (e2 - 2.0 * e1).abs() < 1e-14,
        "size consistency: {e2} vs 2 x {e1}"
    );
}

#[test]
fn gcp_svp_decays_to_zero_at_long_range() {
    let pair = |r: f64| {
        let mol = Molecule::from_xyz(&format!("2\n\nH 0 0 0\nH 0 0 {r}\n")).unwrap();
        gcp_energy(&mol, &GcpParams::B3LYP_3C)
    };
    let mut prev = pair(1.0);
    assert!(prev > 0.0);
    for r in [2.0, 4.0, 8.0, 16.0] {
        let e = pair(r);
        assert!(e < prev, "not decaying at r = {r}: {e} vs {prev}");
        prev = e;
    }
    assert!(pair(50.0).abs() < 1e-30, "not zero at 50 Angstrom");
}

#[test]
fn gcp_svp_gradient_matches_central_differences() {
    let mol = crooked();
    let (_, ga) = gcp_energy_gradient(&mol, &GcpParams::B3LYP_3C);
    let h = 1e-5;
    let mut worst = 0.0f64;
    for (i, gai) in ga.iter().enumerate() {
        for (k, gaik) in gai.iter().enumerate() {
            let mut mp = mol.clone();
            mp.atoms[i].position[k] += h;
            let mut mm = mol.clone();
            mm.atoms[i].position[k] -= h;
            let fd = (gcp_energy(&mp, &GcpParams::B3LYP_3C)
                - gcp_energy(&mm, &GcpParams::B3LYP_3C))
                / (2.0 * h);
            worst = worst.max((gaik - fd).abs());
        }
    }
    assert!(worst < 1e-9, "gCP(SV(P)) FD arbiter: worst = {worst:.3e}");
}

#[test]
fn gcp_svp_energy_paths_agree_and_gradient_sums_to_zero() {
    let mol = crooked();
    let (e, g) = gcp_energy_gradient(&mol, &GcpParams::B3LYP_3C);
    assert_eq!(e, gcp_energy(&mol, &GcpParams::B3LYP_3C));
    for k in 0..3 {
        let s: f64 = g.iter().map(|gi| gi[k]).sum();
        assert!(s.abs() < 1e-14, "gradient sum component {k}: {s:.3e}");
    }
}

#[test]
fn atm_off_reproduces_two_body_d3() {
    let mol = crooked();
    let two_body = D3Params::for_method("b3lyp").unwrap();
    assert_eq!(two_body.s9, 0.0);
    let with_field = D3Params {
        s9: 0.0,
        ..D3Params::B3LYP_3C
    };
    let (e0, g0) = d3bj_energy_gradient(&mol, &two_body);
    let (e1, g1) = d3bj_energy_gradient(&mol, &with_field);
    assert_eq!(e0, e1);
    assert_eq!(g0, g1);
}

#[test]
fn atm_equilateral_triangle_sign_and_scaling() {
    let tri = |l: f64| -> f64 {
        let h = l * (3.0f64).sqrt() / 2.0;
        let mol = Molecule::from_xyz(&format!(
            "3\n\nAr 0 0 0\nAr {l} 0 0\nAr {} {h} 0\n",
            l / 2.0
        ))
        .unwrap();
        let on = d3bj_energy(&mol, &D3Params::B3LYP_3C);
        let off = d3bj_energy(
            &mol,
            &D3Params {
                s9: 0.0,
                ..D3Params::B3LYP_3C
            },
        );
        on - off
    };
    let e = tri(3.8);
    assert!(e > 0.0, "equilateral ATM must be repulsive: {e}");
    let (e1, e2) = (tri(20.0), tri(40.0));
    let ratio = e2 / e1;
    let expect = 2.0f64.powi(-9);
    assert!(
        (ratio / expect - 1.0).abs() < 1e-3,
        "ATM r^-9 scaling: ratio {ratio:.6e} vs {expect:.6e}"
    );
}

#[test]
fn atm_is_size_consistent() {
    let frag = |shift: f64| {
        format!(
            "3\n\nAr {shift} 0 0\nAr {} 0 0\nAr {} 3.3 0\n",
            shift + 3.8,
            shift + 1.9
        )
    };
    let m1 = Molecule::from_xyz(&frag(0.0)).unwrap();
    let lines: Vec<String> = frag(0.0).lines().skip(2).map(String::from).collect();
    let lines2: Vec<String> = frag(500.0).lines().skip(2).map(String::from).collect();
    let dimer_xyz = format!("6\n\n{}\n{}\n", lines.join("\n"), lines2.join("\n"));
    let m2 = Molecule::from_xyz(&dimer_xyz).unwrap();
    let e1 = d3bj_energy(&m1, &D3Params::B3LYP_3C);
    let e2 = d3bj_energy(&m2, &D3Params::B3LYP_3C);
    assert!(
        (e2 - 2.0 * e1).abs() < 1e-14,
        "ATM size consistency: {e2} vs 2 x {e1}"
    );
}

#[test]
fn atm_gradient_matches_central_differences() {
    let mol = crooked();
    let (_, ga) = d3bj_energy_gradient(&mol, &D3Params::B3LYP_3C);
    let h = 1e-5;
    let mut worst = 0.0f64;
    for (i, gai) in ga.iter().enumerate() {
        for (k, gaik) in gai.iter().enumerate() {
            let mut mp = mol.clone();
            mp.atoms[i].position[k] += h;
            let mut mm = mol.clone();
            mm.atoms[i].position[k] -= h;
            let fd = (d3bj_energy(&mp, &D3Params::B3LYP_3C)
                - d3bj_energy(&mm, &D3Params::B3LYP_3C))
                / (2.0 * h);
            worst = worst.max((gaik - fd).abs());
        }
    }
    assert!(worst < 1e-9, "D3-ATM FD arbiter: worst = {worst:.3e}");
}

#[test]
fn atm_energy_invariances_and_gradient_sum() {
    let mol = crooked();
    let (e, g) = d3bj_energy_gradient(&mol, &D3Params::B3LYP_3C);
    for k in 0..3 {
        let s: f64 = g.iter().map(|gi| gi[k]).sum();
        assert!(s.abs() < 1e-12, "gradient sum component {k}: {s:.3e}");
    }
    let mut shifted = mol.clone();
    for a in &mut shifted.atoms {
        a.position[0] += 7.3;
        a.position[1] -= 2.1;
        a.position[2] += 0.4;
    }
    assert!((d3bj_energy(&shifted, &D3Params::B3LYP_3C) - e).abs() < 1e-14);
    let mut rotated = mol.clone();
    for a in &mut rotated.atoms {
        let [x, y, z] = a.position;
        a.position = [-y, x, z];
    }
    assert!((d3bj_energy(&rotated, &D3Params::B3LYP_3C) - e).abs() < 1e-13);
}

#[test]
fn gcp_svp_matches_mctc_gcp_test_reference() {
    use hartree::core::{Atom, Element};
    let sym = [
        "Si", "H", "O", "B", "H", "B", "H", "Si", "Mg", "H", "B", "Mg", "H", "H", "Al", "Li",
    ];
    let xyz: [[f64; 3]; 16] = [
        [2.35657681818464, 1.56413352120650, 0.15633191455554],
        [3.73917059667152, 4.62925085487901, -2.78650603123275],
        [3.14560604851984, 2.89719360409943, -2.78417514965298],
        [-0.77007531623341, 2.61211360015145, 2.33311615406392],
        [-0.66443402867749, 4.05997385957096, 4.06971059462416],
        [-1.35794144684175, -3.75758532634922, -0.38278945412140],
        [-2.23158951954857, 3.52161941806966, 0.71501793840421],
        [-1.21528967345105, -1.22312093897026, 3.34968237398899],
        [-2.91125704119979, 0.39569409719795, -1.08361131016514],
        [-1.52127986148665, -5.83354834109118, 0.65117354687277],
        [0.96426075166446, -1.73395662042913, -0.29756953994186],
        [1.76477233718293, -5.12961800838251, 2.54274889905822],
        [-2.66597958259665, 1.28123105561800, -4.29750060549502],
        [-3.13008195341572, -3.60405914945209, -1.88201177072058],
        [3.70343741730607, -0.47466561516456, 4.15636571641078],
        [0.79410445392162, 0.79534398904606, -4.45998327664881],
    ];
    let atoms = sym
        .iter()
        .zip(&xyz)
        .map(|(s, p)| Atom::new(Element::from_symbol(s).unwrap(), *p))
        .collect();
    let mol = Molecule::new(atoms, 0, 2);
    let e = gcp_energy(&mol, &GcpParams::B3LYP_3C);
    let e_ref = 3.6732928803962006e-2;
    assert!(
        (e - e_ref).abs() < 1e-9,
        "gCP(DFT/SV(P)) vs mctc-gcp: {e:.15} vs {e_ref:.15} (|d| = {:.2e})",
        (e - e_ref).abs()
    );
}
