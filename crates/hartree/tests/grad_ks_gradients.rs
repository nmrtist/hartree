use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc, XcContributor};
use hartree::grad::ks_gradient;
use hartree::integrals::ConventionalProvider;
use hartree::scf::{Reference, ScfOptions, ScfResult, run_scf_with_xc};

const GRID_LEVEL: usize = 3;

fn tight_options() -> ScfOptions {
    ScfOptions {
        energy_tol: 1e-10,
        error_tol: 1e-6,
        max_iter: 512,
        incremental_fock: false,
        level_shift: 0.3,
        ..ScfOptions::default()
    }
}

fn provider_for(mol: &Molecule, basis: &str) -> (ConventionalProvider, hartree::basis::AoBasis) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let ao2 = BasisSet::load(basis).unwrap().build(mol).unwrap();
    (ConventionalProvider::new(ao.into_integral(), charges), ao2)
}

fn run_ks(
    mol: &Molecule,
    basis: &str,
    functional: &str,
    na: usize,
    nb: usize,
    reference: Reference,
) -> (ConventionalProvider, GridXc, ScfResult) {
    let (provider, ao) = provider_for(mol, basis);
    let spec = FunctionalSpec::parse(functional).unwrap();
    let xc = GridXc::new(mol, &ao, &spec, GRID_LEVEL).unwrap();
    let r = run_scf_with_xc(
        &provider,
        na,
        nb,
        reference,
        mol.nuclear_repulsion(),
        &tight_options(),
        Some(&xc as &dyn XcContributor),
    )
    .unwrap();
    assert!(r.converged, "KS SCF did not converge");
    (provider, xc, r)
}

fn ks_energy(
    mol: &Molecule,
    basis: &str,
    functional: &str,
    na: usize,
    nb: usize,
    reference: Reference,
) -> f64 {
    run_ks(mol, basis, functional, na, nb, reference).2.energy
}

fn analytic(
    mol: &Molecule,
    basis: &str,
    functional: &str,
    na: usize,
    nb: usize,
    reference: Reference,
) -> Vec<[f64; 3]> {
    let (provider, xc, r) = run_ks(mol, basis, functional, na, nb, reference);
    ks_gradient(
        &provider,
        mol,
        &xc as &dyn XcContributor,
        &r.density_alpha,
        &r.density_beta,
        reference == Reference::Rhf,
    )
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn finite_difference(
    mol: &Molecule,
    basis: &str,
    functional: &str,
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
            let e_plus = ks_energy(&plus, basis, functional, na, nb, reference);
            let e_minus = ks_energy(&minus, basis, functional, na, nb, reference);
            *slot = (e_plus - e_minus) / (2.0 * h);
        }
    }
    g
}

fn max_component_error(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0_f64, f64::max)
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

fn validate(
    name: &str,
    mol: &Molecule,
    basis: &str,
    functional: &str,
    na: usize,
    nb: usize,
    reference: Reference,
) {
    let g = analytic(mol, basis, functional, na, nb, reference);
    assert!(
        max_abs_component(&g) > 0.02,
        "{name}: geometry too close to a stationary point (max |g| = {:.3e})",
        max_abs_component(&g)
    );

    let err = max_component_error(
        &g,
        &finite_difference(mol, basis, functional, na, nb, reference, 1e-3),
    );
    let tr = translational_residual(&g);
    let rot = rotational_residual(mol, &g);
    eprintln!(
        "{name}: FD(1e-3) err {err:.2e}  sum(g) {tr:.2e}  torque {rot:.2e}  (max|g| {:.3e})",
        max_abs_component(&g)
    );
    assert!(err < 1e-6, "{name}: FD-analytic mismatch {err:.2e} > 1e-6");
    assert!(
        tr < 1e-10,
        "{name}: sum(g) = {tr:.2e} (weight-derivative term?)"
    );
    assert!(rot < 1e-5, "{name}: torque = {rot:.2e}");
}

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

fn oh_radical() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [0.45, 0.30, 2.05]),
        ],
        0,
        2,
    )
}

#[test]
fn rks_svwn_water() {
    validate(
        "H2O/svwn/sto-3g",
        &water_nosym(),
        "sto-3g",
        "svwn",
        5,
        5,
        Reference::Rhf,
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn rks_pbe_water() {
    validate(
        "H2O/pbe/sto-3g",
        &water_nosym(),
        "sto-3g",
        "pbe",
        5,
        5,
        Reference::Rhf,
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn rks_b3lyp_water() {
    validate(
        "H2O/b3lyp/sto-3g",
        &water_nosym(),
        "sto-3g",
        "b3lyp",
        5,
        5,
        Reference::Rhf,
    );
}

#[test]
fn uks_svwn_oh() {
    validate(
        "OH/svwn/6-31g",
        &oh_radical(),
        "6-31g",
        "svwn",
        5,
        4,
        Reference::Uhf,
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn uks_pbe_oh() {
    validate(
        "OH/pbe/6-31g",
        &oh_radical(),
        "6-31g",
        "pbe",
        5,
        4,
        Reference::Uhf,
    );
}

#[test]
fn uks_b3lyp_oh() {
    validate(
        "OH/b3lyp/6-31g",
        &oh_radical(),
        "6-31g",
        "b3lyp",
        5,
        4,
        Reference::Uhf,
    );
}

#[test]
fn rks_tpss_water() {
    validate(
        "H2O/tpss/sto-3g",
        &water_nosym(),
        "sto-3g",
        "tpss",
        5,
        5,
        Reference::Rhf,
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn rks_r2scan_water() {
    validate(
        "H2O/r2scan/sto-3g",
        &water_nosym(),
        "sto-3g",
        "r2scan",
        5,
        5,
        Reference::Rhf,
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn uks_tpss_oh() {
    validate(
        "OH/tpss/6-31g",
        &oh_radical(),
        "6-31g",
        "tpss",
        5,
        4,
        Reference::Uhf,
    );
}

#[test]
fn rks_wb97xv_water() {
    validate(
        "H2O/wb97x-v/sto-3g",
        &water_nosym(),
        "sto-3g",
        "wb97x-v",
        5,
        5,
        Reference::Rhf,
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn uks_wb97xv_oh() {
    validate(
        "OH/wb97x-v/6-31g",
        &oh_radical(),
        "6-31g",
        "wb97x-v",
        5,
        4,
        Reference::Uhf,
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn rks_wb97mv_water() {
    validate(
        "H2O/wb97m-v/sto-3g",
        &water_nosym(),
        "sto-3g",
        "wb97m-v",
        5,
        5,
        Reference::Rhf,
    );
}

#[test]
#[ignore = "cc-pVDZ KS FD sweep is minutes-class; run with --ignored"]
fn rks_b3lyp_water_ccpvdz() {
    validate(
        "H2O/b3lyp/cc-pvdz",
        &water_nosym(),
        "cc-pvdz",
        "b3lyp",
        5,
        5,
        Reference::Rhf,
    );
}
