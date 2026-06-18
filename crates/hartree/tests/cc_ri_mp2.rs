use hartree::basis::BasisSet;
use hartree::cc::{
    column_block, frozen_core_orbitals, rhf_mp2, rhf_ri_mp2, rhf_ri_mp2_b, transform_block,
    uhf_mp2, uhf_ri_mp2,
};
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::{ConventionalProvider, InCoreEri};
use hartree::scf::{Reference, ScfOptions, ScfResult, run_rhf, run_scf};
use serde::Deserialize;

use std::collections::HashMap;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");

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

fn molecule(name: &str) -> Molecule {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let g = &geoms.molecules[name];
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

fn rhf_for(mol: &Molecule, provider: &ConventionalProvider) -> ScfResult {
    let opts = ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        ..ScfOptions::default()
    };
    let scf = run_rhf(
        provider,
        mol.n_electrons() as usize,
        mol.nuclear_repulsion(),
        &opts,
    )
    .unwrap();
    assert!(scf.converged, "SCF must converge");
    scf
}

fn aux_for(mol: &Molecule, basis: &str) -> hartree::integrals::integral::Basis {
    BasisSet::load_aux(&format!("{basis}/c"))
        .unwrap()
        .build(mol)
        .unwrap()
        .into_integral()
}

fn compare_ri_vs_conventional(mol_name: &str, basis: &str, tol: f64) {
    let mol = molecule(mol_name);
    let provider = provider_for(&mol, basis);
    let scf = rhf_for(&mol, &provider);
    let n_frozen = frozen_core_orbitals(&mol);

    let conv = rhf_mp2(&provider, &scf, n_frozen);
    let ao = BasisSet::load(basis)
        .unwrap()
        .build(&mol)
        .unwrap()
        .into_integral();
    let ri = rhf_ri_mp2(&ao, &aux_for(&mol, basis), &scf, n_frozen).unwrap();

    let d_corr = ri.correlation_energy - conv.correlation_energy;
    let d_os = ri.opposite_spin - conv.opposite_spin;
    let d_ss = ri.same_spin - conv.same_spin;
    eprintln!(
        "RI-MP2 {mol_name}/{basis} (fc {n_frozen}, naux {}): Ecorr {:.10} vs {:.10} \
         (ΔE {:.2e}, ΔOS {:.2e}, ΔSS {:.2e})",
        ri.naux, ri.correlation_energy, conv.correlation_energy, d_corr, d_os, d_ss
    );
    assert!(
        d_corr.abs() <= tol,
        "{mol_name}/{basis} ΔEcorr = {d_corr:.2e}"
    );
    assert!(d_os.abs() <= tol, "{mol_name}/{basis} ΔE_OS = {d_os:.2e}");
    assert!(d_ss.abs() <= tol, "{mol_name}/{basis} ΔE_SS = {d_ss:.2e}");
    assert_eq!(ri.n_frozen, conv.n_frozen, "frozen-core counts must match");
    assert!(
        (ri.total_energy - (scf.energy + ri.correlation_energy)).abs() < 1e-12,
        "total = SCF + corr"
    );
}

#[test]
fn water_def2svp_matches_conventional_mp2() {
    compare_ri_vs_conventional("water", "def2-svp", 2e-4);
}

#[test]
fn nh4_plus_def2svp_matches_conventional_mp2() {
    compare_ri_vs_conventional("nh4_plus", "def2-svp", 2e-4);
}

#[test]
fn water_def2tzvp_matches_conventional_mp2() {
    compare_ri_vs_conventional("water", "def2-tzvp", 2e-4);
}

#[test]
fn fitted_iajb_matches_conventional_elements() {
    let mol = molecule("water");
    let basis = "def2-svp";
    let provider = provider_for(&mol, basis);
    let scf = rhf_for(&mol, &provider);
    let n_frozen = frozen_core_orbitals(&mol);

    let n = scf.n_basis;
    let m = scf.n_orbitals;
    let n_occ = scf.n_alpha;
    let n_act = n_occ - n_frozen;
    let n_virt = m - n_occ;
    let c_occ = column_block(&scf.mo_coeff_alpha, n, m, n_frozen, n_act);
    let c_virt = column_block(&scf.mo_coeff_alpha, n, m, n_occ, n_virt);
    let ovov = transform_block(provider.ao_eri(), n, [&c_occ, &c_virt, &c_occ, &c_virt]);
    let g = ovov.data();

    let ao = BasisSet::load(basis)
        .unwrap()
        .build(&mol)
        .unwrap()
        .into_integral();
    let bt = rhf_ri_mp2_b(&ao, &aux_for(&mol, basis), &scf, n_frozen).unwrap();
    assert_eq!(bt.n_act, n_act);
    assert_eq!(bt.n_virt, n_virt);

    let mut worst = 0.0_f64;
    for i in 0..n_act {
        for a in 0..n_virt {
            for j in 0..n_act {
                for b in 0..n_virt {
                    let exact = g[((i * n_virt + a) * n_act + j) * n_virt + b];
                    let fitted = bt.iajb(i, a, j, b);
                    worst = worst.max((fitted - exact).abs());
                }
            }
        }
    }
    eprintln!("RI-MP2 water/def2-svp: max |(ia|jb)_RI − (ia|jb)_conv| = {worst:.2e}");
    assert!(worst < 1e-3, "fitted (ia|jb) off by {worst:.2e}");
}

#[test]
fn frozen_core_convention_matches_conventional() {
    let mol = molecule("water");
    assert_eq!(frozen_core_orbitals(&mol), 1); // O 1s
    let provider = provider_for(&mol, "def2-svp");
    let scf = rhf_for(&mol, &provider);
    let ao = BasisSet::load("def2-svp")
        .unwrap()
        .build(&mol)
        .unwrap()
        .into_integral();
    let aux = aux_for(&mol, "def2-svp");
    let fc = rhf_ri_mp2(&ao, &aux, &scf, 1).unwrap();
    let ae = rhf_ri_mp2(&ao, &aux, &scf, 0).unwrap();
    assert_eq!(fc.n_frozen, 1);
    assert_eq!(ae.n_frozen, 0);
    assert!(ae.correlation_energy < fc.correlation_energy - 1e-4);
}

fn uhf_for(mol: &Molecule, provider: &ConventionalProvider) -> ScfResult {
    let n_elec = mol.n_electrons() as usize;
    let two_s = (mol.multiplicity - 1) as usize;
    let opts = ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        ..ScfOptions::default()
    };
    let scf = run_scf(
        provider,
        (n_elec + two_s) / 2,
        (n_elec - two_s) / 2,
        Reference::Uhf,
        mol.nuclear_repulsion(),
        &opts,
    )
    .unwrap();
    assert!(scf.converged, "UHF must converge");
    scf
}

#[test]
fn reference_mismatches_are_rejected() {
    let ao = |mol: &Molecule| {
        BasisSet::load("def2-svp")
            .unwrap()
            .build(mol)
            .unwrap()
            .into_integral()
    };

    let mol = molecule("oh"); // OH radical, doublet
    let provider = provider_for(&mol, "def2-svp");
    let scf = uhf_for(&mol, &provider);
    let err = rhf_ri_mp2(&ao(&mol), &aux_for(&mol, "def2-svp"), &scf, 1).unwrap_err();
    assert!(err.to_string().contains("closed-shell RHF reference"));

    let mol = molecule("water");
    let provider = provider_for(&mol, "def2-svp");
    let scf = rhf_for(&mol, &provider);
    let err = uhf_ri_mp2(&ao(&mol), &aux_for(&mol, "def2-svp"), &scf, 1).unwrap_err();
    assert!(err.to_string().contains("UHF reference"));
}

#[test]
fn conventional_uhf_mp2_closed_shell_matches_rhf_mp2() {
    let mol = molecule("water");
    let provider = provider_for(&mol, "def2-svp");
    let n_frozen = frozen_core_orbitals(&mol);
    let rhf = rhf_mp2(&provider, &rhf_for(&mol, &provider), n_frozen);
    let uhf = uhf_mp2(&provider, &uhf_for(&mol, &provider), n_frozen);
    eprintln!(
        "UHF-MP2 water/def2-svp closed-shell identity: ΔEcorr {:.2e}, ΔOS {:.2e}, ΔSS {:.2e}",
        uhf.correlation_energy - rhf.correlation_energy,
        uhf.opposite_spin - rhf.opposite_spin,
        uhf.same_spin - rhf.same_spin,
    );
    assert!((uhf.opposite_spin - rhf.opposite_spin).abs() < 1e-8);
    assert!((uhf.same_spin - rhf.same_spin).abs() < 1e-8);
    assert!((uhf.total_energy - rhf.total_energy).abs() < 1e-8);
}

#[test]
fn uhf_ri_mp2_closed_shell_matches_rhf_ri_mp2() {
    let mol = molecule("water");
    let basis = "def2-svp";
    let provider = provider_for(&mol, basis);
    let n_frozen = frozen_core_orbitals(&mol);
    let ao = BasisSet::load(basis)
        .unwrap()
        .build(&mol)
        .unwrap()
        .into_integral();
    let aux = aux_for(&mol, basis);
    let rhf = rhf_ri_mp2(&ao, &aux, &rhf_for(&mol, &provider), n_frozen).unwrap();
    let uhf = uhf_ri_mp2(&ao, &aux, &uhf_for(&mol, &provider), n_frozen).unwrap();
    eprintln!(
        "UHF RI-MP2 water/{basis} closed-shell identity: ΔEcorr {:.2e}, ΔOS {:.2e}, ΔSS {:.2e}",
        uhf.correlation_energy - rhf.correlation_energy,
        uhf.opposite_spin - rhf.opposite_spin,
        uhf.same_spin - rhf.same_spin,
    );
    assert_eq!(uhf.naux, rhf.naux);
    assert!((uhf.opposite_spin - rhf.opposite_spin).abs() < 1e-8);
    assert!((uhf.same_spin - rhf.same_spin).abs() < 1e-8);
    assert!((uhf.total_energy - rhf.total_energy).abs() < 1e-8);
}

fn compare_uhf_ri_vs_conventional(mol_name: &str, basis: &str, tol: f64) {
    let mol = molecule(mol_name);
    let provider = provider_for(&mol, basis);
    let scf = uhf_for(&mol, &provider);
    let n_frozen = frozen_core_orbitals(&mol);

    let conv = uhf_mp2(&provider, &scf, n_frozen);
    let ao = BasisSet::load(basis)
        .unwrap()
        .build(&mol)
        .unwrap()
        .into_integral();
    let ri = uhf_ri_mp2(&ao, &aux_for(&mol, basis), &scf, n_frozen).unwrap();

    let d_corr = ri.correlation_energy - conv.correlation_energy;
    let d_os = ri.opposite_spin - conv.opposite_spin;
    let d_ss = ri.same_spin - conv.same_spin;
    eprintln!(
        "UHF RI-MP2 {mol_name}/{basis} (fc {n_frozen}, naux {}): Ecorr {:.10} vs {:.10} \
         (ΔE {:.2e}, ΔOS {:.2e}, ΔSS {:.2e})",
        ri.naux, ri.correlation_energy, conv.correlation_energy, d_corr, d_os, d_ss
    );
    assert!(
        d_corr.abs() <= tol,
        "{mol_name}/{basis} ΔEcorr = {d_corr:.2e}"
    );
    assert!(d_os.abs() <= tol, "{mol_name}/{basis} ΔE_OS = {d_os:.2e}");
    assert!(d_ss.abs() <= tol, "{mol_name}/{basis} ΔE_SS = {d_ss:.2e}");
    assert_eq!(ri.n_frozen, conv.n_frozen, "frozen-core counts must match");
    assert!(
        (ri.total_energy - (scf.energy + ri.correlation_energy)).abs() < 1e-12,
        "total = SCF + corr"
    );
}

#[test]
fn oh_radical_def2svp_matches_conventional_uhf_mp2() {
    compare_uhf_ri_vs_conventional("oh", "def2-svp", 2e-4);
}

#[test]
fn ch3_radical_def2svp_matches_conventional_uhf_mp2() {
    compare_uhf_ri_vs_conventional("ch3", "def2-svp", 2e-4);
}

#[test]
fn uhf_frozen_core_convention_matches_conventional() {
    let mol = molecule("oh");
    assert_eq!(frozen_core_orbitals(&mol), 1); // O 1s
    let provider = provider_for(&mol, "def2-svp");
    let scf = uhf_for(&mol, &provider);
    let ao = BasisSet::load("def2-svp")
        .unwrap()
        .build(&mol)
        .unwrap()
        .into_integral();
    let aux = aux_for(&mol, "def2-svp");
    let fc = uhf_ri_mp2(&ao, &aux, &scf, 1).unwrap();
    let ae = uhf_ri_mp2(&ao, &aux, &scf, 0).unwrap();
    assert_eq!(fc.n_frozen, 1);
    assert_eq!(ae.n_frozen, 0);
    assert!(ae.correlation_energy < fc.correlation_energy - 1e-4);
    let conv_fc = uhf_mp2(&provider, &scf, 1);
    let conv_ae = uhf_mp2(&provider, &scf, 0);
    let core_ri = ae.correlation_energy - fc.correlation_energy;
    let core_conv = conv_ae.correlation_energy - conv_fc.correlation_energy;
    eprintln!("UHF core contribution: RI {core_ri:.10} vs conventional {core_conv:.10}");
    assert!((core_ri - core_conv).abs() <= 4e-4);
}
