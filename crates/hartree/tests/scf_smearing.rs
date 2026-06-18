mod scf_common;

use hartree::core::Molecule;
use hartree::integrals::IntegralProvider;
use hartree::linalg::mat_to_row_major;
use hartree::scf::{Reference, ScfError, ScfOptions, Smearing, run_rhf, run_scf};

use scf_common::{geometries, provider_for, trace_ds};

fn smear_opts(temperature_k: f64) -> ScfOptions {
    ScfOptions {
        smearing: Some(Smearing::Fermi { temperature_k }),
        ..ScfOptions::default()
    }
}

fn assert_electron_count(
    result: &hartree::scf::ScfResult,
    provider: &impl IntegralProvider,
    n_alpha: usize,
    n_beta: usize,
) {
    let (fa, fb) = result.occupations.as_ref().expect("smeared occupations");
    let (sum_a, sum_b): (f64, f64) = (fa.iter().sum(), fb.iter().sum());
    assert!(
        (sum_a - n_alpha as f64).abs() < 1e-10,
        "Σf_α = {sum_a}, expected {n_alpha}"
    );
    assert!(
        (sum_b - n_beta as f64).abs() < 1e-10,
        "Σf_β = {sum_b}, expected {n_beta}"
    );
    assert!(fa.iter().chain(fb).all(|&f| (0.0..=1.0).contains(&f)));
    let s = mat_to_row_major(&provider.overlap());
    let n_check = trace_ds(&result.density, &s, result.n_basis);
    let n_elec = (n_alpha + n_beta) as f64;
    assert!(
        (n_check - n_elec).abs() < 1e-10,
        "Tr(DS) = {n_check}, expected {n_elec}"
    );
}

#[test]
fn t_to_zero_limit_matches_integer_scf() {
    let mol = geometries().molecules["water"].molecule();
    let provider = provider_for(&mol, "sto-3g");
    let n_elec = mol.n_electrons() as usize;

    let plain = run_rhf(
        &provider,
        n_elec,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
    )
    .unwrap();
    let smeared = run_rhf(&provider, n_elec, mol.nuclear_repulsion(), &smear_opts(1.0)).unwrap();
    assert!(plain.converged && smeared.converged);

    assert!(
        (smeared.energy - plain.energy).abs() < 1e-9,
        "T=1 K energy {} vs integer {}",
        smeared.energy,
        plain.energy
    );
    assert_electron_count(&smeared, &provider, n_elec / 2, n_elec / 2);
    let ts = smeared.electronic_entropy.unwrap();
    assert!(ts.abs() < 1e-12, "T·S_el = {ts}, expected ~0 at 1 K");
    assert!((smeared.free_energy.unwrap() - smeared.energy).abs() < 1e-12);
    assert!(plain.occupations.is_none());
    assert!(plain.electronic_entropy.is_none());
    assert!(plain.free_energy.is_none());
}

#[test]
fn stretched_h2_smears_fractionally() {
    let mol = Molecule::from_xyz("2\nstretched H2\nH 0 0 0\nH 0 0 1.5\n").unwrap();
    let provider = provider_for(&mol, "sto-3g");
    let result = run_rhf(&provider, 2, mol.nuclear_repulsion(), &smear_opts(5000.0)).unwrap();
    assert!(result.converged);

    let (fa, _) = result.occupations.as_ref().unwrap();
    let n_frac = fa.iter().filter(|&&f| f > 1e-10 && f < 1.0 - 1e-10).count();
    assert!(
        n_frac >= 2,
        "expected fractional frontier occupations: {fa:?}"
    );
    let ts = result.electronic_entropy.unwrap();
    assert!(ts > 0.0, "T·S_el = {ts}, expected > 0");
    let free = result.free_energy.unwrap();
    assert!(
        free < result.energy,
        "F = {free} must be below E = {}",
        result.energy
    );
    assert!((free - (result.energy - ts)).abs() < 1e-14);
    assert_electron_count(&result, &provider, 1, 1);
}

#[test]
fn dissociating_h2_has_large_fractions() {
    let mol = Molecule::from_xyz("2\ndissociating H2\nH 0 0 0\nH 0 0 3.0\n").unwrap();
    let provider = provider_for(&mol, "sto-3g");
    let result = run_rhf(&provider, 2, mol.nuclear_repulsion(), &smear_opts(5000.0)).unwrap();
    assert!(result.converged);
    let (fa, _) = result.occupations.as_ref().unwrap();
    assert!(
        fa[0] < 1.0 - 1e-3 && fa[1] > 1e-3,
        "expected strongly fractional occupations: {fa:?}"
    );
    assert!(result.electronic_entropy.unwrap() > 1e-4);
    assert!(result.free_energy.unwrap() < result.energy);
    assert_electron_count(&result, &provider, 1, 1);
}

#[test]
fn uhf_smearing_conserves_per_spin_counts() {
    let mol = geometries().molecules["oh"].molecule();
    let provider = provider_for(&mol, "sto-3g");
    let result = run_scf(
        &provider,
        5,
        4,
        Reference::Uhf,
        mol.nuclear_repulsion(),
        &smear_opts(2000.0),
    )
    .unwrap();
    assert!(result.converged);
    assert_electron_count(&result, &provider, 5, 4);
    assert!(result.electronic_entropy.unwrap() >= 0.0);
    assert!(result.free_energy.unwrap() <= result.energy);
}

#[test]
fn smearing_guards() {
    let mol = geometries().molecules["oh"].molecule();
    let provider = provider_for(&mol, "sto-3g");
    let rohf = run_scf(
        &provider,
        5,
        4,
        Reference::Rohf,
        mol.nuclear_repulsion(),
        &smear_opts(1000.0),
    );
    assert!(matches!(rohf, Err(ScfError::RohfSmearing)));

    let cold = run_scf(
        &provider,
        5,
        4,
        Reference::Uhf,
        mol.nuclear_repulsion(),
        &smear_opts(0.0),
    );
    assert!(matches!(cold, Err(ScfError::NonPositiveTemperature { .. })));
}
