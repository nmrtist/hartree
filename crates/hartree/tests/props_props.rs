use hartree::basis::BasisSet;
use hartree::core::units::AU_DIPOLE_TO_DEBYE;
use hartree::core::{Atom, Element, Molecule};
use hartree::grad::hf_gradient;
use hartree::integrals::ConventionalProvider;
use hartree::props::dipole::{center_of_mass, dipole_moment};
use hartree::props::frequencies::harmonic_frequencies;
use hartree::props::hessian::numerical_hessian;
use hartree::props::population::population_analysis;
use hartree::props::thermo::rrho_thermochemistry;
use hartree::scf::{ScfOptions, run_rhf};
use serde::Deserialize;

use std::collections::HashMap;

const GEOM_JSON: &str = include_str!("../../../tests/ref/geometries.json");

#[derive(Deserialize)]
struct Geometries {
    molecules: HashMap<String, GeomEntry>,
}
#[derive(Deserialize)]
struct GeomEntry {
    charge: i32,
    multiplicity: u32,
    atoms: Vec<(String, f64, f64, f64)>,
}

fn mol_from_entry(g: &GeomEntry) -> Molecule {
    let atoms = g
        .atoms
        .iter()
        .map(|(s, x, y, z)| Atom::new(Element::from_symbol(s).unwrap(), [*x, *y, *z]))
        .collect();
    Molecule::new(atoms, g.charge, g.multiplicity)
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

fn tight_opts() -> ScfOptions {
    ScfOptions {
        energy_tol: 1e-10,
        error_tol: 1e-10,
        ..ScfOptions::default()
    }
}

fn water_opt_sto3g() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(Element::from_z(8).unwrap(), [0.0, -0.091037, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [1.432565, 1.110455, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [-1.432565, 1.110455, 0.0]),
        ],
        0,
        1,
    )
}

#[test]
fn h2_dipole_is_zero() {
    let geoms: Geometries = serde_json::from_str(GEOM_JSON).unwrap();
    let mol = mol_from_entry(&geoms.molecules["h2"]);
    let provider = provider_for(&mol, "sto-3g");
    let scf = run_rhf(&provider, 2, mol.nuclear_repulsion(), &tight_opts()).unwrap();

    let com = center_of_mass(&mol);
    let mu = dipole_moment(&provider, &mol, &scf.density, com);
    for (k, component) in mu.iter().enumerate() {
        assert!(
            component.abs() < 1e-12,
            "H₂ dipole component {k} = {}, expected 0",
            component
        );
    }
}

#[test]
fn water_dipole_symmetry() {
    let mol = water_opt_sto3g();
    let provider = provider_for(&mol, "sto-3g");
    let scf = run_rhf(&provider, 10, mol.nuclear_repulsion(), &tight_opts()).unwrap();
    let com = center_of_mass(&mol);
    let mu = dipole_moment(&provider, &mol, &scf.density, com);

    assert!(mu[0].abs() < 1e-11, "water μ_x = {}", mu[0]);
    assert!(mu[2].abs() < 1e-11, "water μ_z = {}", mu[2]);
    assert!(
        mu[1].abs() > 0.5,
        "water μ_y should be substantial, got {}",
        mu[1]
    );
}

#[test]
fn mulliken_charge_sum() {
    let geoms: Geometries = serde_json::from_str(GEOM_JSON).unwrap();
    let mol = mol_from_entry(&geoms.molecules["water"]);
    let provider = provider_for(&mol, "6-31g");
    let scf = run_rhf(&provider, 10, mol.nuclear_repulsion(), &tight_opts()).unwrap();
    let pop = population_analysis(&provider, &mol, &scf.density_alpha, &scf.density_beta);

    let sum: f64 = pop.mulliken_charges.iter().sum();
    assert!(
        (sum - mol.charge as f64).abs() < 1e-10,
        "Mulliken charge sum = {sum}, expected {}",
        mol.charge
    );
}

#[test]
fn lowdin_charge_sum() {
    let mol = water_opt_sto3g();
    let provider = provider_for(&mol, "sto-3g");
    let scf = run_rhf(&provider, 10, mol.nuclear_repulsion(), &tight_opts()).unwrap();
    let pop = population_analysis(&provider, &mol, &scf.density_alpha, &scf.density_beta);

    let sum: f64 = pop.lowdin_charges.iter().sum();
    assert!(
        (sum - mol.charge as f64).abs() < 1e-9,
        "Löwdin charge sum = {sum}, expected {}",
        mol.charge
    );
}

#[test]
fn mayer_symmetric() {
    let mol = water_opt_sto3g();
    let provider = provider_for(&mol, "sto-3g");
    let scf = run_rhf(&provider, 10, mol.nuclear_repulsion(), &tight_opts()).unwrap();
    let pop = population_analysis(&provider, &mol, &scf.density_alpha, &scf.density_beta);

    let n = mol.len();
    for i in 0..n {
        for j in 0..n {
            let diff = (pop.mayer_bond_orders[i][j] - pop.mayer_bond_orders[j][i]).abs();
            assert!(diff < 1e-10, "Mayer[{i},{j}] != Mayer[{j},{i}]: {diff}");
        }
    }
}

#[test]
fn h2_mayer_bond_order_is_one() {
    let geoms: Geometries = serde_json::from_str(GEOM_JSON).unwrap();
    let mol = mol_from_entry(&geoms.molecules["h2"]);
    let provider = provider_for(&mol, "sto-3g");
    let scf = run_rhf(&provider, 2, mol.nuclear_repulsion(), &tight_opts()).unwrap();
    let pop = population_analysis(&provider, &mol, &scf.density_alpha, &scf.density_beta);

    let b_hh = pop.mayer_bond_orders[0][1];
    assert!(
        (b_hh - 1.0).abs() < 0.05,
        "H₂ Mayer bond order = {b_hh}, expected ~1"
    );
}

#[test]
#[ignore = "slow; run with --release -- --ignored"]
fn h2_frequencies_sto3g() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.027040]),
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 1.372960]),
        ],
        0,
        1,
    );

    let hess = build_hessian(&mol, "sto-3g", 0.005);
    let freq = harmonic_frequencies(&mol, &hess);

    let near_zero: Vec<f64> = freq.frequencies_cm1[..5].iter().map(|f| f.abs()).collect();
    for (k, &f) in near_zero.iter().enumerate() {
        assert!(f < 50.0, "trans/rot mode {k} = {f} cm⁻¹, expected ~0");
    }

    let vib = freq.frequencies_cm1[5];
    assert!(vib > 3000.0, "H₂ stretch = {vib} cm⁻¹, expected > 3000");
    assert_eq!(freq.n_imaginary, 0);
}

#[test]
#[ignore = "slow; run with --release -- --ignored"]
fn water_frequencies_sto3g_algebraic() {
    let mol = water_opt_sto3g();

    let hess = build_hessian(&mol, "sto-3g", 0.005);
    let freq = harmonic_frequencies(&mol, &hess);

    for (k, &f) in freq.frequencies_cm1[..6].iter().enumerate() {
        assert!(f.abs() < 50.0, "mode {k} = {f} cm⁻¹, expected trans/rot ~0");
    }
    for (k, &f) in freq.frequencies_cm1[6..].iter().enumerate() {
        assert!(f > 1000.0, "water vib mode {k} = {f} cm⁻¹, expected > 1000");
    }
    assert_eq!(freq.n_imaginary, 0, "should be a minimum");
}

#[test]
#[ignore = "slow; run with --release -- --ignored"]
fn water_thermo_monotonic() {
    let mol = water_opt_sto3g();
    let provider = provider_for(&mol, "sto-3g");
    let scf = run_rhf(&provider, 10, mol.nuclear_repulsion(), &tight_opts()).unwrap();
    let e_elec = scf.energy;

    let hess = build_hessian(&mol, "sto-3g", 0.005);
    let freq = harmonic_frequencies(&mol, &hess);
    let thermo = rrho_thermochemistry(&mol, &freq, e_elec, 298.15, 2, 1);

    assert!(thermo.zpe > 0.0, "ZPE must be positive");
    assert!(
        thermo.enthalpy > e_elec,
        "H(298.15 K) = {} must exceed E_elec = {}",
        thermo.enthalpy,
        e_elec
    );
    assert!(
        thermo.gibbs < thermo.enthalpy,
        "G = {} must be less than H = {} (S > 0)",
        thermo.gibbs,
        thermo.enthalpy
    );
}

fn build_hessian(mol: &Molecule, basis: &str, step: f64) -> Vec<f64> {
    let n_elec = mol.n_electrons() as usize;
    let basis = basis.to_string();
    numerical_hessian(mol, step, move |m| {
        let p = provider_for(m, &basis);
        let scf = run_rhf(&p, n_elec, m.nuclear_repulsion(), &tight_opts()).unwrap();
        let grad = hf_gradient(&p, m, &scf.density_alpha, &scf.density_beta).unwrap();
        grad.iter().flat_map(|g| g.iter().copied()).collect()
    })
}

const OPT_JSON: &str = include_str!("../../../tests/ref/opt_references.json");
const PROPS_JSON: &str = include_str!("../../../tests/ref/props_references.json");
const FREQ_JSON: &str = include_str!("../../../tests/ref/freq_references.json");

#[derive(Deserialize)]
struct OptRefs {
    entries: Vec<OptEntry>,
}
#[derive(Deserialize)]
struct OptEntry {
    molecule: String,
    basis: String,
    method: String,
    charge: i32,
    multiplicity: u32,
    geometry_bohr: Vec<(String, f64, f64, f64)>,
}

fn mol_from_opt(o: &OptEntry) -> Molecule {
    let atoms = o
        .geometry_bohr
        .iter()
        .map(|(s, x, y, z)| Atom::new(Element::from_symbol(s).unwrap(), [*x, *y, *z]))
        .collect();
    Molecule::new(atoms, o.charge, o.multiplicity)
}

fn opt_geometry(molecule: &str, basis: &str) -> Molecule {
    let opt: OptRefs = serde_json::from_str(OPT_JSON).unwrap();
    let o = opt
        .entries
        .into_iter()
        .find(|o| o.molecule == molecule && o.basis == basis && o.method == "rhf")
        .unwrap_or_else(|| panic!("no opt geometry for {molecule}/{basis}"));
    mol_from_opt(&o)
}

#[derive(Deserialize)]
struct PropsRefs {
    entries: Vec<PropsEntry>,
}
#[derive(Deserialize)]
struct PropsEntry {
    molecule: String,
    basis: String,
    dipole_magnitude_au: f64,
    mulliken_charges: Vec<f64>,
    lowdin_charges: Vec<f64>,
    mayer_bond_orders: HashMap<String, f64>,
}

#[test]
#[ignore = "ORCA oracle; run with --release -- --ignored"]
fn props_vs_orca() {
    let props: PropsRefs = serde_json::from_str(PROPS_JSON).unwrap();
    assert!(
        !props.entries.is_empty(),
        "props_references.json has no entries"
    );

    for e in &props.entries {
        let mol = opt_geometry(&e.molecule, &e.basis);
        let provider = provider_for(&mol, &e.basis);
        let scf = run_rhf(
            &provider,
            mol.n_electrons() as usize,
            mol.nuclear_repulsion(),
            &tight_opts(),
        )
        .unwrap();

        let com = center_of_mass(&mol);
        let mu = dipole_moment(&provider, &mol, &scf.density, com);
        let mag_au = (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]).sqrt();
        let ref_au = e.dipole_magnitude_au;
        eprintln!(
            "{}/{} |μ| = {:.8} a.u. ({:.6} D) hartree vs {:.8} a.u. ORCA  Δ = {:.2e}",
            e.molecule,
            e.basis,
            mag_au,
            mag_au * AU_DIPOLE_TO_DEBYE,
            ref_au,
            (mag_au - ref_au).abs()
        );
        assert!(
            (mag_au - ref_au).abs() < 1e-5,
            "{}/{} dipole: hartree {:.8} a.u. vs ORCA {:.8} a.u.",
            e.molecule,
            e.basis,
            mag_au,
            ref_au
        );

        let pop = population_analysis(&provider, &mol, &scf.density_alpha, &scf.density_beta);
        for (i, (&c, &r)) in pop
            .mulliken_charges
            .iter()
            .zip(&e.mulliken_charges)
            .enumerate()
        {
            eprintln!("  Mulliken[{i}] hartree {c:.6} vs ORCA {r:.6}");
            assert!(
                (c - r).abs() < 1e-4,
                "{}/{} Mulliken[{i}]: {c:.6} vs {r:.6}",
                e.molecule,
                e.basis
            );
        }
        for (i, (&c, &r)) in pop.lowdin_charges.iter().zip(&e.lowdin_charges).enumerate() {
            eprintln!("  Löwdin[{i}]   hartree {c:.6} vs ORCA {r:.6}");
            assert!(
                (c - r).abs() < 1e-4,
                "{}/{} Löwdin[{i}]: {c:.6} vs {r:.6}",
                e.molecule,
                e.basis
            );
        }
        for (k, &r) in &e.mayer_bond_orders {
            let idx: Vec<usize> = k.split(',').map(|s| s.parse().unwrap()).collect();
            let b = pop.mayer_bond_orders[idx[0]][idx[1]];
            eprintln!("  Mayer[{k}]    hartree {b:.4} vs ORCA {r:.4}");
            assert!(
                (b - r).abs() < 5e-3,
                "{}/{} Mayer[{k}]: {b:.4} vs {r:.4}",
                e.molecule,
                e.basis
            );
        }
    }
}

#[derive(Deserialize)]
struct FreqRefs {
    entries: Vec<FreqEntry>,
}
#[derive(Deserialize)]
struct FreqEntry {
    molecule: String,
    basis: String,
    multiplicity: u32,
    symmetry_number: u32,
    temperature: f64,
    frequencies_cm1: Vec<f64>,
    zpe_hartree: f64,
    enthalpy_hartree: f64,
    entropy_hartree_per_k: f64,
    gibbs_hartree: f64,
}

#[test]
#[ignore = "ORCA oracle; run with --release -- --ignored"]
fn freq_thermo_vs_orca() {
    let freqs: FreqRefs = serde_json::from_str(FREQ_JSON).unwrap();
    assert!(
        !freqs.entries.is_empty(),
        "freq_references.json has no entries"
    );

    for e in &freqs.entries {
        if e.basis != "sto-3g" {
            continue;
        }
        let mol = opt_geometry(&e.molecule, &e.basis);
        let hess = build_hessian(&mol, &e.basis, 0.005);
        let freq = harmonic_frequencies(&mol, &hess);

        for k in 0..6 {
            assert!(
                freq.frequencies_cm1[k].abs() < 10.0,
                "{}/{} trans/rot mode {k} = {:.3} cm⁻¹, expected ~0 (bad projection)",
                e.molecule,
                e.basis,
                freq.frequencies_cm1[k]
            );
        }
        assert_eq!(
            freq.n_imaginary, 0,
            "{}/{} should be a minimum",
            e.molecule, e.basis
        );

        let n = freq.frequencies_cm1.len();
        let hartree_real = &freq.frequencies_cm1[n - 3..];
        let orca_real = &e.frequencies_cm1[e.frequencies_cm1.len() - 3..];
        for (i, (&c, &r)) in hartree_real.iter().zip(orca_real).enumerate() {
            eprintln!(
                "{}/{} freq[{i}] hartree {c:.2} vs ORCA {r:.2} cm⁻¹",
                e.molecule, e.basis
            );
            assert!(
                (c - r).abs() < 2.0,
                "{}/{} freq mode {i}: {c:.2} vs {r:.2} cm⁻¹",
                e.molecule,
                e.basis
            );
        }

        let provider = provider_for(&mol, &e.basis);
        let scf = run_rhf(
            &provider,
            mol.n_electrons() as usize,
            mol.nuclear_repulsion(),
            &tight_opts(),
        )
        .unwrap();
        let thermo = rrho_thermochemistry(
            &mol,
            &freq,
            scf.energy,
            e.temperature,
            e.symmetry_number,
            e.multiplicity,
        );

        eprintln!(
            "{}/{} ZPE hartree {:.6} vs ORCA {:.6}; H {:.6} vs {:.6}; S {:.3e} vs {:.3e}; G {:.6} vs {:.6}",
            e.molecule,
            e.basis,
            thermo.zpe,
            e.zpe_hartree,
            thermo.enthalpy,
            e.enthalpy_hartree,
            thermo.entropy,
            e.entropy_hartree_per_k,
            thermo.gibbs,
            e.gibbs_hartree
        );
        assert!(
            (thermo.zpe - e.zpe_hartree).abs() < 5e-5,
            "{}/{} ZPE: {:.6} vs {:.6}",
            e.molecule,
            e.basis,
            thermo.zpe,
            e.zpe_hartree
        );
        assert!(
            (thermo.entropy - e.entropy_hartree_per_k).abs() < 5e-7,
            "{}/{} entropy: {:.3e} vs {:.3e}",
            e.molecule,
            e.basis,
            thermo.entropy,
            e.entropy_hartree_per_k
        );
        assert!(
            (thermo.enthalpy - e.enthalpy_hartree).abs() < 1e-4,
            "{}/{} enthalpy: {:.6} vs {:.6}",
            e.molecule,
            e.basis,
            thermo.enthalpy,
            e.enthalpy_hartree
        );
        assert!(
            (thermo.gibbs - e.gibbs_hartree).abs() < 1e-4,
            "{}/{} Gibbs: {:.6} vs {:.6}",
            e.molecule,
            e.basis,
            thermo.gibbs,
            e.gibbs_hartree
        );
    }
}
