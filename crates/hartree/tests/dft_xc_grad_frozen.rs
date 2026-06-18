use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc};

const LEVEL: usize = 3;

fn water_nosym() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(Element::from_z(8).unwrap(), [0.05, -0.10, 0.02]),
            Atom::new(Element::from_z(1).unwrap(), [1.70, 0.20, -0.30]),
            Atom::new(Element::from_z(1).unwrap(), [-0.40, 1.75, 0.25]),
        ],
        0,
        1,
    )
}

fn gridxc(mol: &Molecule, basis: &str, functional: &str) -> GridXc {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let spec = FunctionalSpec::parse(functional).unwrap();
    GridXc::new(mol, &ao, &spec, LEVEL).unwrap()
}

fn toy_density(n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let v = 0.4 / (1.0 + (i as f64 - j as f64).powi(2));
            d[i * n + j] = v;
        }
        d[i * n + i] += 0.6 + 0.05 * i as f64;
    }
    d
}

fn check(functional: &str) {
    let mol = water_nosym();
    let xc = gridxc(&mol, "sto-3g", functional);
    let n = xc.nao();
    let d_half: Vec<f64> = toy_density(n).iter().map(|x| 0.5 * x).collect();

    let g = xc.xc_gradient(&d_half, &d_half, true).unwrap();

    let h = 1e-4;
    let mut worst = 0.0_f64;
    #[allow(clippy::needless_range_loop)] // atom/axis index both g and the displaced geometry
    for atom in 0..mol.atoms.len() {
        for axis in 0..3 {
            let mut plus = mol.clone();
            plus.atoms[atom].position[axis] += h;
            let mut minus = mol.clone();
            minus.atoms[atom].position[axis] -= h;
            let ep = gridxc(&plus, "sto-3g", functional)
                .energy(&d_half, &d_half, true)
                .0;
            let em = gridxc(&minus, "sto-3g", functional)
                .energy(&d_half, &d_half, true)
                .0;
            let fd = (ep - em) / (2.0 * h);
            let err = (fd - g[atom][axis]).abs();
            worst = worst.max(err);
            eprintln!(
                "{functional} atom {atom} axis {axis}: an {:+.10}  fd {fd:+.10}  err {err:.2e}",
                g[atom][axis]
            );
        }
    }
    assert!(
        worst < 1e-7,
        "{functional}: frozen-density XC FD err {worst:.2e}"
    );
}

#[test]
fn svwn_frozen_density() {
    check("svwn");
}

#[test]
fn pbe_frozen_density() {
    check("pbe");
}

#[test]
fn b3lyp_frozen_density() {
    check("b3lyp");
}

#[test]
fn tpss_frozen_density() {
    check("tpss");
}

#[test]
fn r2scan_frozen_density() {
    check("r2scan");
}

fn check_polarized(functional: &str) {
    let mol = water_nosym();
    let xc = gridxc(&mol, "sto-3g", functional);
    let n = xc.nao();
    let d_a: Vec<f64> = toy_density(n).iter().map(|x| 0.6 * x).collect();
    let d_b: Vec<f64> = toy_density(n).iter().map(|x| 0.4 * x).collect();

    let g = xc.xc_gradient(&d_a, &d_b, false).unwrap();

    let h = 1e-4;
    let mut worst = 0.0_f64;
    #[allow(clippy::needless_range_loop)]
    for atom in 0..mol.atoms.len() {
        for axis in 0..3 {
            let mut plus = mol.clone();
            plus.atoms[atom].position[axis] += h;
            let mut minus = mol.clone();
            minus.atoms[atom].position[axis] -= h;
            let ep = gridxc(&plus, "sto-3g", functional)
                .energy(&d_a, &d_b, false)
                .0;
            let em = gridxc(&minus, "sto-3g", functional)
                .energy(&d_a, &d_b, false)
                .0;
            let fd = (ep - em) / (2.0 * h);
            worst = worst.max((fd - g[atom][axis]).abs());
        }
    }
    assert!(
        worst < 1e-7,
        "{functional}: polarized frozen-density XC FD err {worst:.2e}"
    );
}

#[test]
fn tpss_frozen_density_polarized() {
    check_polarized("tpss");
}

#[test]
fn r2scan_frozen_density_polarized() {
    check_polarized("r2scan");
}
