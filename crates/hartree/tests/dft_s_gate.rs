use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::ao::{eval_ao_batch, n_ao, par_blocks_fold};
use hartree::dft::density::batch_density;
use hartree::dft::{MolecularGrid, ShellData};
use hartree::integrals::{ConventionalProvider, IntegralProvider};
use hartree::linalg::mat_to_row_major;
use hartree::scf::{ScfOptions, run_rhf};

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

fn h2s() -> Molecule {
    Molecule::new(
        vec![
            atom("S", [0.0, 0.0, 0.0]),
            atom("H", [1.808, 0.0, 1.739]),
            atom("H", [-1.808, 0.0, 1.739]),
        ],
        0,
        1,
    )
}

fn provider_for(ao: &hartree::basis::AoBasis, mol: &Molecule) -> ConventionalProvider {
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    ConventionalProvider::new(ao.clone().into_integral(), charges)
}

fn quadrature_overlap(shells: &[ShellData], nao: usize, grid: &MolecularGrid) -> Vec<f64> {
    let weights = &grid.weights;
    par_blocks_fold(
        shells,
        nao,
        &grid.points,
        false,
        || vec![0.0_f64; nao * nao],
        |mut acc, batch, start| {
            for p in 0..batch.npts {
                let w = weights[start + p];
                let row = &batch.phi[p * nao..p * nao + nao];
                for mu in 0..nao {
                    let a = w * row[mu];
                    if a == 0.0 {
                        continue;
                    }
                    let dst = &mut acc[mu * nao..mu * nao + nao];
                    for nu in 0..nao {
                        dst[nu] += a * row[nu];
                    }
                }
            }
            acc
        },
        |mut a, b| {
            for (x, y) in a.iter_mut().zip(&b) {
                *x += y;
            }
            a
        },
    )
    .unwrap()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn gate_one(mol: &Molecule, basis: &str, level: usize) -> f64 {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let nao = ao.n_ao();
    assert_eq!(nao, n_ao(ao.shells()), "evaluator/AoBasis nao disagree");

    let s_ref = ao.integral().overlap();

    let grid = MolecularGrid::build(mol, level).unwrap();
    let s_quad = quadrature_overlap(ao.shells(), nao, &grid);

    max_abs_diff(&s_quad, &s_ref)
}

const S_GATE_L3: f64 = 7.0e-6;
const S_GATE_L4: f64 = 9.0e-7;

fn gate_for(level: usize) -> f64 {
    if level >= 4 { S_GATE_L4 } else { S_GATE_L3 }
}

const S_GATE_F_L3: f64 = 3.0e-5;
const S_GATE_F_L4: f64 = 8.0e-6;
// Diffuse (aug / *pd) bases on the production point-efficient level-3 grid (five-zone
// angular pruning + a moderated Treutler-Ahlrichs radial set): the worst diffuse
// overlap-quadrature residual is aug-cc-pVTZ on H2O at ~1.31e-4 (H2S ~1.13e-4). A
// grid-resolution artifact of the most diffuse functions, not an accuracy limit —
// the level-3 ORCA oracle (worst clean Δ ~1.9e-6) and the electron-count integral
// (∫ρ matches N_e to well within its pinned 1e-6) both hold. Reference-quality
// level 4 stays at 4e-5.
const S_GATE_AUG_L3: f64 = 1.5e-4;
const S_GATE_AUG_L4: f64 = 4.0e-5;

const S_GATE_G_L3: f64 = 2.6e-4;
const S_GATE_G_L4: f64 = 6.5e-5;

fn f_gate_for(basis: &str, level: usize) -> f64 {
    let aug = basis.starts_with("aug-") || basis.ends_with("pd");
    match (aug, level >= 4) {
        (false, false) => S_GATE_F_L3,
        (false, true) => S_GATE_F_L4,
        (true, false) => S_GATE_AUG_L3,
        (true, true) => S_GATE_AUG_L4,
    }
}

#[test]
fn s_gate_fast_subset() {
    let mol = water();
    for basis in ["sto-3g", "6-31g"] {
        let d = gate_one(&mol, basis, 3);
        println!("S-gate water/{basis}/L3: max|ΔS| = {d:e}");
        assert!(
            d < S_GATE_L3,
            "water/{basis}/L3 max|ΔS|={d:e} exceeds {S_GATE_L3:e}"
        );
    }
}

#[test]
#[ignore = "slow: cc-pvdz/def2-svp at level 4; run with --ignored"]
fn s_gate_all_bases() {
    let mols = [("h2o", water()), ("h2s", h2s())];
    let mut results: Vec<(String, f64)> = Vec::new();
    for (mname, mol) in &mols {
        for basis in ["sto-3g", "6-31g", "cc-pvdz", "def2-svp"] {
            for level in [3usize, 4] {
                let d = gate_one(mol, basis, level);
                println!("S-gate {mname}/{basis}/L{level}: max|ΔS| = {d:e}");
                results.push((format!("{mname}/{basis}/L{level}"), d));
            }
        }
    }
    let worst3 = results
        .iter()
        .filter(|(k, _)| k.ends_with("L3"))
        .map(|(_, d)| *d)
        .fold(0.0_f64, f64::max);
    let worst4 = results
        .iter()
        .filter(|(k, _)| k.ends_with("L4"))
        .map(|(_, d)| *d)
        .fold(0.0_f64, f64::max);
    println!("S-gate worst: L3 = {worst3:e}, L4 = {worst4:e}");
    for (label, d) in &results {
        let level = if label.ends_with("L4") { 4 } else { 3 };
        let gate = gate_for(level);
        assert!(*d < gate, "{label} max|ΔS|={d:e} exceeds gate {gate:e}");
    }
}

#[test]
fn s_gate_spherical_f_fast() {
    let d = gate_one(&water(), "cc-pvtz", 3);
    println!("S-gate water/cc-pvtz/L3: max|ΔS| = {d:e}");
    assert!(
        d < S_GATE_F_L3,
        "water/cc-pvtz/L3 max|ΔS|={d:e} exceeds {S_GATE_F_L3:e}"
    );
}

#[test]
fn s_gate_mtzvpp_fast() {
    let d = gate_one(&water(), "def2-mtzvpp", 3);
    println!("S-gate water/def2-mtzvpp/L3: max|ΔS| = {d:e}");
    assert!(
        d < S_GATE_F_L3,
        "water/def2-mtzvpp/L3 max|ΔS|={d:e} exceeds {S_GATE_F_L3:e}"
    );
}

#[test]
#[ignore = "slow: cc-pVTZ/def2-TZVP/aug-cc-pVTZ at level 4; run with --ignored"]
fn s_gate_spherical_f() {
    let mols = [("h2o", water()), ("h2s", h2s())];
    let mut results: Vec<(String, f64)> = Vec::new();
    for (mname, mol) in &mols {
        for basis in [
            "cc-pvtz",
            "def2-tzvp",
            "def2-tzvpp",
            "aug-cc-pvtz",
            "def2-tzvpd",
            "def2-svpd",
            "def2-mtzvpp",
        ] {
            for level in [3usize, 4] {
                let d = gate_one(mol, basis, level);
                println!("S-gate {mname}/{basis}/L{level}: max|ΔS| = {d:e}");
                results.push((format!("{mname}/{basis}/L{level}"), d));
            }
        }
    }
    let diffuse = |k: &str| {
        let basis = k.split('/').nth(1).unwrap();
        basis.starts_with("aug-") || basis.ends_with("pd")
    };
    let worst_compact = results
        .iter()
        .filter(|(k, _)| !diffuse(k))
        .map(|(_, d)| *d)
        .fold(0.0_f64, f64::max);
    let worst_aug = results
        .iter()
        .filter(|(k, _)| diffuse(k))
        .map(|(_, d)| *d)
        .fold(0.0_f64, f64::max);
    println!("S-gate spherical-f worst: compact = {worst_compact:e}, aug = {worst_aug:e}");
    for (label, d) in &results {
        let basis = label.split('/').nth(1).unwrap();
        let level = if label.ends_with("L4") { 4 } else { 3 };
        let gate = f_gate_for(basis, level);
        assert!(*d < gate, "{label} max|ΔS|={d:e} exceeds gate {gate:e}");
    }
}

#[test]
#[ignore = "slow: cc-pVQZ/def2-QZVP (QZ, up to ~130 bf) at level 4; run with --ignored"]
fn s_gate_spherical_g() {
    let mols = [("h2o", water()), ("h2s", h2s())];
    let mut results: Vec<(String, f64)> = Vec::new();
    for (mname, mol) in &mols {
        for basis in ["cc-pvqz", "def2-qzvp", "def2-qzvpp"] {
            for level in [3usize, 4] {
                let d = gate_one(mol, basis, level);
                println!("S-gate {mname}/{basis}/L{level}: max|ΔS| = {d:e}");
                results.push((format!("{mname}/{basis}/L{level}"), d));
            }
        }
    }
    let worst3 = results
        .iter()
        .filter(|(k, _)| k.ends_with("L3"))
        .map(|(_, d)| *d)
        .fold(0.0_f64, f64::max);
    let worst4 = results
        .iter()
        .filter(|(k, _)| k.ends_with("L4"))
        .map(|(_, d)| *d)
        .fold(0.0_f64, f64::max);
    println!("S-gate spherical-g worst: L3 = {worst3:e}, L4 = {worst4:e}");
    for (label, d) in &results {
        let level = if label.ends_with("L4") { 4 } else { 3 };
        let gate = if level >= 4 { S_GATE_G_L4 } else { S_GATE_G_L3 };
        assert!(*d < gate, "{label} max|ΔS|={d:e} exceeds gate {gate:e}");
    }
}

#[test]
fn integrated_density_matches_electron_count() {
    check_ne(&water(), "sto-3g", 3, 10.0);
}

#[test]
#[ignore = "slow: cc-pvdz density on a level-4 grid; run with --ignored"]
fn integrated_density_matches_electron_count_ccpvdz() {
    check_ne(&water(), "cc-pvdz", 4, 10.0);
}

fn check_ne(mol: &Molecule, basis: &str, level: usize, n_elec: f64) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let nao = ao.n_ao();
    let provider = provider_for(&ao, mol);

    let opts = ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        ..ScfOptions::default()
    };
    let scf = run_rhf(
        &provider,
        mol.n_electrons() as usize,
        mol.nuclear_repulsion(),
        &opts,
    )
    .unwrap();
    assert!(scf.converged);

    let s_ref = mat_to_row_major(&provider.overlap());
    let trace_ds: f64 = scf.density.iter().zip(&s_ref).map(|(d, s)| d * s).sum();

    let grid = MolecularGrid::build(mol, level).unwrap();
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

    println!("{basis}/L{level}: Tr(DS)={trace_ds:.10}  ∫ρ={int_rho:.10}  N_e={n_elec}");
    assert!((trace_ds - n_elec).abs() < 1e-9, "Tr(DS) != N_e");
    assert!(
        (int_rho - n_elec).abs() < 1e-6,
        "∫ρ={int_rho} deviates from N_e={n_elec}"
    );
}

#[test]
fn cartesian_d_overlap_matches_integral() {
    use hartree::integrals::integral::{Basis, Shell};

    let ca = [0.0, 0.0, 0.0];
    let cb = [0.0, 0.0, 2.4];
    let exps = vec![1.2, 0.4];
    let coeffs = vec![0.55, 0.45];

    let sa = Shell::new(2, ca, exps.clone(), coeffs.clone()).unwrap();
    let sb = Shell::new(2, cb, exps.clone(), coeffs.clone()).unwrap();
    let basis = Basis::new(vec![sa, sb]);
    let s_ref = basis.overlap(); // 12×12 row-major (6 cart components each)
    let nao = 12;

    let shells = vec![
        ShellData {
            l: 2,
            center: ca,
            exponents: exps.clone(),
            coefficients: coeffs.clone(),
            spherical: false,
        },
        ShellData {
            l: 2,
            center: cb,
            exponents: exps,
            coefficients: coeffs,
            spherical: false,
        },
    ];

    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_symbol("C").unwrap(), ca),
            Atom::new(Element::from_symbol("C").unwrap(), cb),
        ],
        0,
        1,
    );
    let grid = MolecularGrid::build(&mol, 4).unwrap();
    let s_quad = quadrature_overlap(&shells, nao, &grid);

    let d = max_abs_diff(&s_quad, &s_ref);
    println!("Cartesian-d overlap vs integral: max|ΔS| = {d:e}");
    assert!(d < 1e-6, "Cartesian-d max|ΔS|={d:e}");

    let _ = eval_ao_batch(&shells, nao, &grid.points[..1], false);
    assert!(s_quad.iter().any(|&x| x.abs() > 0.1));
}
