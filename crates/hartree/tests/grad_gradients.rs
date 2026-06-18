use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::grad::hf_gradient;
use hartree::integrals::ConventionalProvider;
use hartree::scf::{Reference, ScfOptions, run_scf};

fn tight_options() -> ScfOptions {
    ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        max_iter: 256,
        ..ScfOptions::default()
    }
}

fn provider_for(mol: &Molecule, basis: &str) -> ConventionalProvider {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    ConventionalProvider::new(ao.into_integral(), charges)
}

fn energy(mol: &Molecule, basis: &str, na: usize, nb: usize, reference: Reference) -> f64 {
    let provider = provider_for(mol, basis);
    let r = run_scf(
        &provider,
        na,
        nb,
        reference,
        mol.nuclear_repulsion(),
        &tight_options(),
    )
    .unwrap();
    assert!(r.converged, "SCF did not converge");
    r.energy
}

fn analytic(
    mol: &Molecule,
    basis: &str,
    na: usize,
    nb: usize,
    reference: Reference,
) -> Vec<[f64; 3]> {
    let provider = provider_for(mol, basis);
    let r = run_scf(
        &provider,
        na,
        nb,
        reference,
        mol.nuclear_repulsion(),
        &tight_options(),
    )
    .unwrap();
    assert!(r.converged, "SCF did not converge");
    hf_gradient(&provider, mol, &r.density_alpha, &r.density_beta).unwrap()
}

fn finite_difference(
    mol: &Molecule,
    basis: &str,
    na: usize,
    nb: usize,
    reference: Reference,
    h: f64,
) -> Vec<[f64; 3]> {
    let natom = mol.len();
    let mut g = vec![[0.0; 3]; natom];
    for (atom, g_atom) in g.iter_mut().enumerate() {
        for (axis, slot) in g_atom.iter_mut().enumerate() {
            let mut plus = mol.clone();
            plus.atoms[atom].position[axis] += h;
            let mut minus = mol.clone();
            minus.atoms[atom].position[axis] -= h;
            let e_plus = energy(&plus, basis, na, nb, reference);
            let e_minus = energy(&minus, basis, na, nb, reference);
            *slot = (e_plus - e_minus) / (2.0 * h);
        }
    }
    g
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

fn max_abs_component(g: &[[f64; 3]]) -> f64 {
    g.iter()
        .flat_map(|v| v.iter())
        .fold(0.0_f64, |m, &x| m.max(x.abs()))
}

fn translational_residual(g: &[[f64; 3]]) -> f64 {
    let mut s = [0.0; 3];
    for v in g {
        for k in 0..3 {
            s[k] += v[k];
        }
    }
    s.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

fn rotational_residual(mol: &Molecule, g: &[[f64; 3]]) -> f64 {
    let mut t = [0.0; 3];
    for (atom, gv) in g.iter().enumerate() {
        let r = mol.atoms[atom].position;
        t[0] += r[1] * gv[2] - r[2] * gv[1];
        t[1] += r[2] * gv[0] - r[0] * gv[2];
        t[2] += r[0] * gv[1] - r[1] * gv[0];
    }
    t.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

fn validate_fd(
    name: &str,
    mol: &Molecule,
    basis: &str,
    na: usize,
    nb: usize,
    reference: Reference,
) {
    let g = analytic(mol, basis, na, nb, reference);
    assert!(
        max_abs_component(&g) > 0.02,
        "{name}: geometry is too close to a stationary point (max |g| = {:.3e}); \
         displace it so the test actually exercises the gradient",
        max_abs_component(&g)
    );

    let err_coarse =
        max_component_error(&g, &finite_difference(mol, basis, na, nb, reference, 1e-2));
    let err_mid = max_component_error(&g, &finite_difference(mol, basis, na, nb, reference, 1e-3));
    let err_fine = max_component_error(&g, &finite_difference(mol, basis, na, nb, reference, 1e-4));
    let best = err_coarse.min(err_mid).min(err_fine);

    eprintln!(
        "{name}: FD err  h=1e-2 {err_coarse:.2e}  h=1e-3 {err_mid:.2e}  h=1e-4 {err_fine:.2e}  \
         (max|g| {:.3e})",
        max_abs_component(&g)
    );

    assert!(
        err_coarse / err_mid > 30.0,
        "{name}: no h² convergence (err 1e-2 {err_coarse:.2e} / err 1e-3 {err_mid:.2e})"
    );
    assert!(
        best < 1e-7,
        "{name}: best FD–analytic mismatch {best:.2e} exceeds 1e-7"
    );

    let tr = translational_residual(&g);
    let rot = rotational_residual(mol, &g);
    assert!(tr < 1e-9, "{name}: Σg = {tr:.2e} (translation)");
    assert!(rot < 1e-9, "{name}: Σ R×g = {rot:.2e} (rotation)");
}

#[test]
fn h2_rhf_gradient() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 1.20]),
        ],
        0,
        1,
    );
    validate_fd("H2/6-31g", &mol, "6-31g", 1, 1, Reference::Rhf);
}

#[test]
fn water_displaced_rhf_gradient() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [1.70, 0.20, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [-0.40, 1.75, 0.0]),
        ],
        0,
        1,
    );
    validate_fd("H2O/sto-3g", &mol, "sto-3g", 5, 5, Reference::Rhf);
}

#[test]
fn ch3_uhf_gradient() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_z(6).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [2.20, 0.0, 0.15]),
            Atom::new(Element::from_z(1).unwrap(), [-1.05, 1.80, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [-1.05, -1.80, -0.15]),
        ],
        0,
        2,
    );
    validate_fd("CH3/sto-3g", &mol, "sto-3g", 5, 4, Reference::Uhf);
}

#[test]
#[ignore = "slow (~2 min): toy-size cc-pVDZ eri_grad; run with --ignored"]
fn water_ccpvdz_invariances() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [1.70, 0.20, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [-0.40, 1.75, 0.0]),
        ],
        0,
        1,
    );
    let g = analytic(&mol, "cc-pvdz", 5, 5, Reference::Rhf);
    assert!(
        max_abs_component(&g) > 0.02,
        "geometry too close to stationary"
    );
    let tr = translational_residual(&g);
    let rot = rotational_residual(&mol, &g);
    eprintln!("H2O/cc-pVDZ: Σg = {tr:.2e}, Σ R×g = {rot:.2e}");
    assert!(tr < 1e-9, "Σg = {tr:.2e}");
    assert!(rot < 1e-9, "Σ R×g = {rot:.2e}");

    let (atom, axis, h) = (1usize, 0usize, 1e-3);
    let mut plus = mol.clone();
    plus.atoms[atom].position[axis] += h;
    let mut minus = mol.clone();
    minus.atoms[atom].position[axis] -= h;
    let fd = (energy(&plus, "cc-pvdz", 5, 5, Reference::Rhf)
        - energy(&minus, "cc-pvdz", 5, 5, Reference::Rhf))
        / (2.0 * h);
    let err = (g[atom][axis] - fd).abs();
    eprintln!(
        "H2O/cc-pVDZ: analytic {:.8} vs FD(1e-3) {fd:.8}  err {err:.2e}",
        g[atom][axis]
    );
    assert!(err < 1e-5, "H2O/cc-pVDZ FD spot check err {err:.2e}");
}
