//! Above-Kr integration-grid validation for the def2-ECP heavy-element range.
//!
//! The molecular integration grid is built from the FULL nuclear charge Z
//! (`MolecularGrid` keys off `element.z()`), and above Kr the Treutler-Ahlrichs
//! radial scaling parameter ξ is the element-independent fallback 1.0 — the
//! per-period growth of the radial-point count is what carries the resolution.
//! An ECP atom's density, however, holds only the *valence* electrons (Z − n_core).
//! These tests check that the full-Z grid + ξ=1.0 fallback still (a) integrates the
//! valence-only density to the right electron count and (b) yields KS energies that
//! are converged at the default grid level, across the 4d / 5p / 5d / 4f regimes.

use hartree::basis::BasisSet;
use hartree::core::Molecule;
use hartree::dft::FunctionalSpec;
use hartree::dft::MolecularGrid;
use hartree::dft::ao::par_blocks_fold;
use hartree::dft::density::batch_density;
use hartree::integrals::ConventionalProvider;
use hartree::scf::{ScfOptions, run_rhf};
use hartree::{Job, JobOptions, Method};

fn ecp_atom(sym: &str) -> Molecule {
    Molecule::from_xyz(&format!("1\n{sym}\n{sym} 0 0 0\n")).unwrap()
}

/// ∫ρ over the level-`level` grid for a closed-shell ECP atom, plus its valence count.
fn ecp_int_rho(sym: &str, level: usize) -> (f64, i64) {
    let mol = ecp_atom(sym);
    let set = BasisSet::load("def2-svp").unwrap();
    let ao = set.build(&mol).unwrap();
    let nao = ao.n_ao();
    let n_val = mol.n_electrons() - set.ecp_core_electrons(&mol) as i64;

    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .zip(ao.ecp_core())
        .map(|(a, &c)| (a.position, a.element.z() as f64 - c as f64))
        .collect();
    let zeff: Vec<f64> = charges.iter().map(|&(_, q)| q).collect();
    let vnn = mol.nuclear_repulsion_with(&zeff);
    let ecps = ao.ecps().to_vec();
    let provider = ConventionalProvider::new(ao.clone().into_integral(), charges).with_ecps(ecps);

    let opts = ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        ..ScfOptions::default()
    };
    let scf = run_rhf(&provider, n_val as usize, vnn, &opts).unwrap();
    assert!(scf.converged, "{sym}: ECP RHF did not converge");

    let grid = MolecularGrid::build(&mol, level).unwrap();
    let weights = &grid.weights;
    let int_rho = par_blocks_fold(
        ao.shells(),
        nao,
        &grid.points,
        false,
        || 0.0_f64,
        |acc, batch, start| {
            let bd = batch_density(batch, &scf.density, false);
            acc + bd
                .rho
                .iter()
                .enumerate()
                .map(|(p, r)| weights[start + p] * r)
                .sum::<f64>()
        },
        |a, b| a + b,
    )
    .unwrap();
    (int_rho, n_val)
}

fn ks_energy(sym: &str, xc: &str, level: usize) -> f64 {
    let r = Job {
        molecule: ecp_atom(sym),
        basis: "def2-svp".into(),
        method: Method::Dft(FunctionalSpec::parse(xc).unwrap()),
        options: JobOptions {
            grid_level: level,
            ..JobOptions::default()
        },
    }
    .run()
    .unwrap();
    assert!(r.scf.converged, "{sym}/{xc}: KS did not converge");
    r.scf.energy
}

#[test]
fn heavy_ecp_grid_integrates_valence_density() {
    for (sym, n_val_expect) in [("Pd", 18), ("Xe", 26), ("Hg", 20), ("Yb", 42)] {
        let (int_rho, n_val) = ecp_int_rho(sym, 3);
        assert_eq!(n_val, n_val_expect, "{sym} valence count");
        eprintln!(
            "{sym}: int_rho = {int_rho:.8}  N_val = {n_val}  d = {:.2e}",
            int_rho - n_val as f64
        );
        // Observed deviations are ≤1e-8 (Yb 4f is the loosest); 1e-6 matches the
        // light-element ∫ρ gate in dft_s_gate.rs and still catches any gross grid failure.
        assert!(
            (int_rho - n_val as f64).abs() < 1e-6,
            "{sym}: int_rho = {int_rho:.8} deviates from N_val = {n_val}"
        );
    }
}

#[test]
fn heavy_ecp_dft_grid_converged_l3_vs_l4() {
    // Xe (5p) and Yb (4f) are the steepest-valence cases above Kr, where the ξ=1.0
    // fallback is most exposed. Observed L3↔L4 drift is ≤1.1e-6 Eh; 5e-6 leaves headroom
    // for FP/threading variation while still asserting the default grid is converged.
    for sym in ["Xe", "Yb"] {
        let e3 = ks_energy(sym, "pbe", 3);
        let e4 = ks_energy(sym, "pbe", 4);
        eprintln!("{sym}/pbe: E(L3)={e3:.10} E(L4)={e4:.10} d={:.2e}", e3 - e4);
        assert!(
            (e3 - e4).abs() < 5e-6,
            "{sym}: grid not converged L3 vs L4: d={:.2e}",
            e3 - e4
        );
    }
}
