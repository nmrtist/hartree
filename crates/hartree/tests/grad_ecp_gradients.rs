use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::grad::hf_gradient;
use hartree::integrals::ConventionalProvider;
use hartree::scf::{Reference, ScfOptions, run_scf};

fn tight_options() -> ScfOptions {
    ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        // The lanthanide YbH2 4f14 case is a near-degenerate SCF that can take a few
        // hundred iterations at displaced geometries (a static level shift only makes it
        // worse), so keep a comfortable budget; AgH converges in tens.
        max_iter: 1024,
        ..ScfOptions::default()
    }
}

fn agh() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(Element::from_z(47).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [0.25, -0.35, 2.9]),
        ],
        0,
        1,
    )
}

fn ybh2() -> Molecule {
    // Yb (Z=70, ECP28 -> 42 valence) + 2 H = 44 e -> closed-shell RHF (na=nb=22).
    // Asymmetric bent geometry (~1.9 A bonds) so no gradient component is trivially zero.
    Molecule::new(
        vec![
            Atom::new(Element::from_z(70).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [0.3, -0.4, 3.6]),
            Atom::new(Element::from_z(1).unwrap(), [-0.3, 0.5, -3.5]),
        ],
        0,
        1,
    )
}

fn setup(mol: &Molecule, basis: &str) -> (ConventionalProvider, f64) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .zip(ao.ecp_core())
        .map(|(a, &c)| (a.position, a.element.z() as f64 - c as f64))
        .collect();
    let zeff: Vec<f64> = charges.iter().map(|&(_, q)| q).collect();
    let vnn = mol.nuclear_repulsion_with(&zeff);
    let ecps = ao.ecps().to_vec();
    assert!(!ecps.is_empty(), "AgH must carry the Ag ECP");
    let provider = ConventionalProvider::new(ao.into_integral(), charges).with_ecps(ecps);
    (provider, vnn)
}

fn energy(mol: &Molecule, basis: &str, na: usize, nb: usize) -> f64 {
    let (provider, vnn) = setup(mol, basis);
    let r = run_scf(&provider, na, nb, Reference::Rhf, vnn, &tight_options()).unwrap();
    assert!(r.converged, "SCF did not converge");
    r.energy
}

fn analytic(mol: &Molecule, basis: &str, na: usize, nb: usize) -> Vec<[f64; 3]> {
    let (provider, vnn) = setup(mol, basis);
    let r = run_scf(&provider, na, nb, Reference::Rhf, vnn, &tight_options()).unwrap();
    assert!(r.converged, "SCF did not converge");
    hf_gradient(&provider, mol, &r.density_alpha, &r.density_beta).unwrap()
}

fn finite_difference(mol: &Molecule, basis: &str, na: usize, nb: usize, h: f64) -> Vec<[f64; 3]> {
    use rayon::prelude::*;
    let natom = mol.len();
    let comps: Vec<f64> = (0..natom * 3)
        .into_par_iter()
        .map(|dof| {
            let (atom, axis) = (dof / 3, dof % 3);
            let mut plus = mol.clone();
            plus.atoms[atom].position[axis] += h;
            let mut minus = mol.clone();
            minus.atoms[atom].position[axis] -= h;
            (energy(&plus, basis, na, nb) - energy(&minus, basis, na, nb)) / (2.0 * h)
        })
        .collect();
    comps.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn max_component_error(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let mut worst = 0.0_f64;
    for (ga, gb) in a.iter().zip(b) {
        for k in 0..3 {
            worst = worst.max((ga[k] - gb[k]).abs());
        }
    }
    worst
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn agh_gradient_matches_finite_difference() {
    let mol = agh();
    let g = analytic(&mol, "def2-svp", 10, 10);
    let scale = g
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0_f64, |m, &x| m.max(x.abs()));
    assert!(scale > 1e-3, "gradient suspiciously small: {scale:.3e}");

    let fd = finite_difference(&mol, "def2-svp", 10, 10, 1e-3);
    let err = max_component_error(&g, &fd);
    eprintln!("AgH/def2-SVP analytic vs FD(h=1e-3): worst component {err:.3e} Eh/bohr");
    assert!(
        err < 1e-6,
        "analytic vs FD(h=1e-3) worst component {err:.3e} (bar 1e-6)"
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn agh_gradient_invariances() {
    let mol = agh();
    let g = analytic(&mol, "def2-svp", 10, 10);

    let mut t = [0.0; 3];
    for v in &g {
        for k in 0..3 {
            t[k] += v[k];
        }
    }
    let t_res = t.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    eprintln!("AgH/def2-SVP translational residual {t_res:.3e} Eh/bohr");
    assert!(t_res < 1e-9, "translational residual {t_res:.3e}");

    let mut r = [0.0; 3];
    for (atom, v) in g.iter().enumerate() {
        let p = mol.atoms[atom].position;
        r[0] += p[1] * v[2] - p[2] * v[1];
        r[1] += p[2] * v[0] - p[0] * v[2];
        r[2] += p[0] * v[1] - p[1] * v[0];
    }
    let r_res = r.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    assert!(r_res < 1e-8, "rotational residual {r_res:.3e}");
}

/// Lanthanide ECP gradient: Yb's def2-ECP has the h (l=5) local channel and s/p/d/f/g
/// projectors -- the high-l gradient regime integral 0.4.0 enabled (MAX_ECP_GRAD_L = 5,
/// with MAX_ECP_GRAD_PROJ = 5 exactly saturated). The AgH test above only exercises the
/// f-local (l=3) path. Closed-shell YbH2 (44 e) at an asymmetric geometry; analytic vs
/// finite difference confirms the h-local + g-projector gradient contributions are correct
/// (observed worst component ~6e-8 Eh/bohr).
#[test]
#[ignore = "very slow (~5 min): YbH2 lanthanide ECP gradient by finite difference"]
fn lanthanide_gradient_matches_finite_difference() {
    let mol = ybh2();
    let g = analytic(&mol, "def2-svp", 22, 22);
    let scale = g
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0_f64, |m, &x| m.max(x.abs()));
    assert!(scale > 1e-3, "gradient suspiciously small: {scale:.3e}");

    let mut t = [0.0; 3];
    for v in &g {
        for k in 0..3 {
            t[k] += v[k];
        }
    }
    let t_res = t.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    assert!(t_res < 1e-8, "translational residual {t_res:.3e}");

    let fd = finite_difference(&mol, "def2-svp", 22, 22, 1e-3);
    let err = max_component_error(&g, &fd);
    eprintln!(
        "YbH2/def2-SVP (h-local ECP) analytic vs FD(h=1e-3): worst component {err:.3e} Eh/bohr"
    );
    assert!(
        err < 1e-6,
        "analytic vs FD(h=1e-3) worst component {err:.3e} (bar 1e-6)"
    );
}
