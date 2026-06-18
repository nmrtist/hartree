use hartree::core::Molecule;
use hartree::disp::{gcp_r2scan3c_energy, gcp_r2scan3c_energy_gradient};

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

fn fd_gradient(mol: &Molecule, h: f64) -> Vec<[f64; 3]> {
    let mut g = vec![[0.0; 3]; mol.atoms.len()];
    for (i, gi) in g.iter_mut().enumerate() {
        for (k, gik) in gi.iter_mut().enumerate() {
            let mut mp = mol.clone();
            mp.atoms[i].position[k] += h;
            let mut mm = mol.clone();
            mm.atoms[i].position[k] -= h;
            *gik = (gcp_r2scan3c_energy(&mp) - gcp_r2scan3c_energy(&mm)) / (2.0 * h);
        }
    }
    g
}

#[test]
fn analytic_gradient_matches_central_differences() {
    let mol = crooked();
    let (_, ga) = gcp_r2scan3c_energy_gradient(&mol);
    let gf = fd_gradient(&mol, 1e-5);
    let mut worst = 0.0f64;
    for (a, f) in ga.iter().zip(&gf) {
        for k in 0..3 {
            worst = worst.max((a[k] - f[k]).abs());
        }
    }
    assert!(
        worst < 1e-9,
        "gCP FD arbiter: worst |analytic - FD| = {worst:.3e}"
    );
}

#[test]
fn energy_is_translation_invariant() {
    let mol = crooked();
    let e0 = gcp_r2scan3c_energy(&mol);
    let mut shifted = mol.clone();
    for a in &mut shifted.atoms {
        a.position[0] += 7.3;
        a.position[1] -= 2.1;
        a.position[2] += 11.7;
    }
    let e1 = gcp_r2scan3c_energy(&shifted);
    assert!(
        (e0 - e1).abs() < 1e-13,
        "translation invariance violated: {e0} vs {e1}"
    );
}

#[test]
fn energy_is_rotation_invariant() {
    let mol = crooked();
    let e0 = gcp_r2scan3c_energy(&mol);
    let (c1, s1) = (0.7f64.cos(), 0.7f64.sin());
    let (c2, s2) = (0.4f64.cos(), 0.4f64.sin());
    let mut rotated = mol.clone();
    for a in &mut rotated.atoms {
        let [x, y, z] = a.position;
        let (x1, y1) = (c1 * x - s1 * y, s1 * x + c1 * y);
        let (y2, z2) = (c2 * y1 - s2 * z, s2 * y1 + c2 * z);
        a.position = [x1, y2, z2];
    }
    let e1 = gcp_r2scan3c_energy(&rotated);
    assert!(
        (e0 - e1).abs() < 1e-13,
        "rotation invariance violated: {e0} vs {e1}"
    );
}

#[test]
fn gradient_sums_to_zero() {
    let (_, g) = gcp_r2scan3c_energy_gradient(&crooked());
    for k in 0..3 {
        let s: f64 = g.iter().map(|gi| gi[k]).sum();
        assert!(s.abs() < 1e-12, "net force component {k} = {s:.3e}");
    }
}

#[test]
fn pbeh3c_set_gradient_matches_central_differences_and_laws() {
    use hartree::disp::{GcpParams, gcp_energy, gcp_energy_gradient};
    let mol = crooked();
    let p = GcpParams::PBEH_3C;
    let e = gcp_energy(&mol, &p);
    assert!(e > 0.0, "gCP must be repulsive, got {e}");
    let (_, ga) = gcp_energy_gradient(&mol, &p);
    let h = 1e-5;
    let mut worst = 0.0f64;
    for (i, gai) in ga.iter().enumerate() {
        for (k, gaik) in gai.iter().enumerate() {
            let mut mp = mol.clone();
            mp.atoms[i].position[k] += h;
            let mut mm = mol.clone();
            mm.atoms[i].position[k] -= h;
            let fd = (gcp_energy(&mp, &p) - gcp_energy(&mm, &p)) / (2.0 * h);
            worst = worst.max((gaik - fd).abs());
        }
    }
    assert!(
        worst < 1e-9,
        "PBEh-3c gCP FD arbiter: worst |analytic - FD| = {worst:.3e}"
    );
    let far = Molecule::from_xyz("2\n\nO 0 0 0\nC 0 0 40.0\n").unwrap(); // 75.6 bohr
    assert_eq!(gcp_energy(&far, &p), 0.0);
}
