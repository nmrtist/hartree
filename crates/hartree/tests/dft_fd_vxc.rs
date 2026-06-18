use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc, XcContributor};
use hartree::integrals::ConventionalProvider;
use hartree::scf::{Reference, ScfOptions, run_rhf, run_scf};
use rayon::prelude::*;

fn atom(sym: &str, pos: [f64; 3]) -> Atom {
    Atom::new(Element::from_symbol(sym).unwrap(), pos)
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            atom("O", [0.0, -0.143225816552, 0.0]),
            atom("H", [1.638036840407, 1.136548822547, 0.0]),
            atom("H", [-1.638036840407, 1.136548822547, 0.0]),
        ],
        0,
        1,
    )
}

fn oh() -> Molecule {
    Molecule::new(
        vec![atom("O", [0.0, 0.0, 0.0]), atom("H", [0.0, 0.0, 1.8344])],
        0,
        2,
    )
}

fn ne() -> Molecule {
    Molecule::new(vec![atom("Ne", [0.0, 0.0, 0.0])], 0, 1)
}

fn f_atom() -> Molecule {
    Molecule::new(vec![atom("F", [0.0, 0.0, 0.0])], 0, 2)
}

fn provider_for(ao: &hartree::basis::AoBasis, mol: &Molecule) -> ConventionalProvider {
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    ConventionalProvider::new(ao.clone().into_integral(), charges)
}

fn scf_opts() -> ScfOptions {
    ScfOptions {
        energy_tol: 1e-11,
        error_tol: 1e-9,
        ..ScfOptions::default()
    }
}

fn rhf_alpha_density(mol: &Molecule, basis: &str) -> (Vec<f64>, usize) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let n = ao.n_ao();
    let provider = provider_for(&ao, mol);
    let scf = run_rhf(
        &provider,
        mol.n_electrons() as usize,
        mol.nuclear_repulsion(),
        &scf_opts(),
    )
    .unwrap();
    assert!(scf.converged);
    (scf.density_alpha, n)
}

fn uhf_densities(mol: &Molecule, basis: &str) -> (Vec<f64>, Vec<f64>, usize) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let n = ao.n_ao();
    let provider = provider_for(&ao, mol);
    let n_elec = mol.n_electrons() as usize;
    let mult = 2; // doublet
    let n_alpha = (n_elec + (mult - 1)) / 2;
    let n_beta = n_elec - n_alpha;
    let scf = run_scf(
        &provider,
        n_alpha,
        n_beta,
        Reference::Uhf,
        mol.nuclear_repulsion(),
        &scf_opts(),
    )
    .unwrap();
    assert!(scf.converged);
    (scf.density_alpha, scf.density_beta, n)
}

fn perturb(d: &[f64], n: usize, mu: usize, nu: usize, eps: f64) -> Vec<f64> {
    let mut dp = d.to_vec();
    dp[mu * n + nu] += eps;
    dp[nu * n + mu] += eps;
    dp
}

const EPS: f64 = 1e-5;

fn fd_worst(
    xc: &GridXc,
    d_a: &[f64],
    d_b: &[f64],
    n: usize,
    restricted: bool,
    perturb_alpha: bool,
) -> f64 {
    let contrib = xc.eval(d_a, d_b, n, restricted);
    let v = if perturb_alpha {
        &contrib.vxc_alpha
    } else {
        &contrib.vxc_beta
    };

    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|mu| (mu..n).map(move |nu| (mu, nu)))
        .collect();
    pairs
        .par_iter()
        .map(|&(mu, nu)| {
            let (e_plus, e_minus) = if perturb_alpha {
                (
                    xc.energy(&perturb(d_a, n, mu, nu, EPS), d_b, restricted).0,
                    xc.energy(&perturb(d_a, n, mu, nu, -EPS), d_b, restricted).0,
                )
            } else {
                (
                    xc.energy(d_a, &perturb(d_b, n, mu, nu, EPS), restricted).0,
                    xc.energy(d_a, &perturb(d_b, n, mu, nu, -EPS), restricted).0,
                )
            };
            let fd = (e_plus - e_minus) / (2.0 * EPS);
            let expected = v[mu * n + nu] + v[nu * n + mu];
            (fd - expected).abs()
        })
        .reduce(|| 0.0_f64, f64::max)
}

const FUNCTIONALS: &[&str] = &["svwn", "pbe", "blyp", "b3lyp", "tpss", "r2scan"];

fn fd_tol(functional: &str) -> f64 {
    if functional == "r2scan" { 1e-5 } else { 1e-7 }
}

#[test]
fn fd_restricted_sto3g() {
    let (d_a, n) = rhf_alpha_density(&water(), "sto-3g");
    let ao = BasisSet::load("sto-3g").unwrap().build(&water()).unwrap();
    for &name in FUNCTIONALS {
        let spec = FunctionalSpec::parse(name).unwrap();
        let xc = GridXc::new(&water(), &ao, &spec, 1).unwrap();
        let worst = fd_worst(&xc, &d_a, &d_a, n, true, true);
        println!("FD restricted sto-3g {name}: worst |Δ| = {worst:e}");
        assert!(
            worst < fd_tol(name),
            "{name}: restricted FD worst {worst:e} exceeds {:e}",
            fd_tol(name)
        );
    }
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn fd_polarized_sto3g() {
    let (d_a, d_b, n) = uhf_densities(&oh(), "sto-3g");
    let ao = BasisSet::load("sto-3g").unwrap().build(&oh()).unwrap();
    for &name in FUNCTIONALS {
        let spec = FunctionalSpec::parse(name).unwrap();
        let xc = GridXc::new(&oh(), &ao, &spec, 1).unwrap();
        let wa = fd_worst(&xc, &d_a, &d_b, n, false, true);
        let wb = fd_worst(&xc, &d_a, &d_b, n, false, false);
        println!("FD polarized sto-3g {name}: worst α = {wa:e}, β = {wb:e}");
        assert!(
            wa.max(wb) < fd_tol(name),
            "{name}: polarized FD worst {:e} exceeds {:e}",
            wa.max(wb),
            fd_tol(name)
        );
    }
}

#[test]
#[ignore = "slow: cc-pvdz FD sweep; run with --ignored"]
fn fd_restricted_ccpvdz() {
    let (d_a, n) = rhf_alpha_density(&ne(), "cc-pvdz");
    let ao = BasisSet::load("cc-pvdz").unwrap().build(&ne()).unwrap();
    for &name in FUNCTIONALS {
        let spec = FunctionalSpec::parse(name).unwrap();
        let xc = GridXc::new(&ne(), &ao, &spec, 1).unwrap();
        let worst = fd_worst(&xc, &d_a, &d_a, n, true, true);
        println!("FD restricted cc-pvdz {name}: worst |Δ| = {worst:e}");
        assert!(
            worst < fd_tol(name),
            "{name}: cc-pvdz restricted FD worst {worst:e}"
        );
    }
}

#[test]
#[ignore = "slow: cc-pvdz FD sweep; run with --ignored"]
fn fd_polarized_ccpvdz() {
    let (d_a, d_b, n) = uhf_densities(&f_atom(), "cc-pvdz");
    let ao = BasisSet::load("cc-pvdz").unwrap().build(&f_atom()).unwrap();
    for &name in FUNCTIONALS {
        let spec = FunctionalSpec::parse(name).unwrap();
        let xc = GridXc::new(&f_atom(), &ao, &spec, 1).unwrap();
        let wa = fd_worst(&xc, &d_a, &d_b, n, false, true);
        let wb = fd_worst(&xc, &d_a, &d_b, n, false, false);
        println!("FD polarized cc-pvdz {name}: worst α = {wa:e}, β = {wb:e}");
        assert!(
            wa.max(wb) < fd_tol(name),
            "{name}: cc-pvdz polarized FD worst {:e} (see fd_tol — r2scan clamp note)",
            wa.max(wb)
        );
    }
}

#[test]
fn restricted_equals_polarized_closed_shell() {
    let (d_a, n) = rhf_alpha_density(&water(), "sto-3g");
    let ao = BasisSet::load("sto-3g").unwrap().build(&water()).unwrap();
    for &name in FUNCTIONALS {
        let spec = FunctionalSpec::parse(name).unwrap();
        let xc = GridXc::new(&water(), &ao, &spec, 3).unwrap();
        let r = xc.eval(&d_a, &d_a, n, true);
        let u = xc.eval(&d_a, &d_a, n, false);
        assert!(
            (r.exc - u.exc).abs() < 1e-12,
            "{name}: E_xc R={} U={}",
            r.exc,
            u.exc
        );
        for i in 0..n * n {
            assert!(
                (r.vxc_alpha[i] - u.vxc_alpha[i]).abs() < 1e-10,
                "{name}: V_α mismatch at {i}"
            );
            assert!(
                (u.vxc_alpha[i] - u.vxc_beta[i]).abs() < 1e-10,
                "{name}: polarized V_α != V_β for closed shell at {i}"
            );
        }
    }
}

#[test]
fn xc_contribution_sanity() {
    let (d_a, n) = rhf_alpha_density(&water(), "sto-3g");
    let ao = BasisSet::load("sto-3g").unwrap().build(&water()).unwrap();
    for &name in FUNCTIONALS {
        let spec = FunctionalSpec::parse(name).unwrap();
        let xc = GridXc::new(&water(), &ao, &spec, 3).unwrap();
        let c = xc.eval(&d_a, &d_a, n, true);
        assert!(
            c.exc < 0.0,
            "{name}: E_xc should be negative, got {}",
            c.exc
        );
        assert!(
            (c.n_elec_grid - 10.0).abs() < 1e-4,
            "{name}: ∫ρ = {} != 10",
            c.n_elec_grid
        );
        assert_eq!(c.vxc_alpha, c.vxc_beta, "{name}: restricted V_α != V_β");
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c.vxc_alpha[i * n + j] - c.vxc_alpha[j * n + i]).abs() < 1e-12,
                    "{name}: V not symmetric at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn lda_path_has_no_sigma() {
    let spec = FunctionalSpec::parse("svwn").unwrap();
    assert!(!spec.needs_sigma());
    let f = spec.build(hartree::dft::Spin::Unpolarized).unwrap();
    let rho = vec![0.3, 0.5, 0.7];
    let out = f.eval(3, &hartree::dft::XcInput::lda(&rho)).unwrap();
    assert!(out.vsigma.is_empty(), "svwn produced a non-empty vsigma");
}
