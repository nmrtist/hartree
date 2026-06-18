use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::density::batch_density_tau;
use hartree::dft::grid::MolecularGrid;
use hartree::dft::par_blocks_fold;
use hartree::integrals::{ConventionalProvider, IntegralProvider};
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

fn rhf_setup(mol: &Molecule, basis: &str) -> (Vec<f64>, Vec<f64>, hartree::basis::AoBasis) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let provider = ConventionalProvider::new(ao.clone().into_integral(), charges);
    let t = hartree::linalg::mat_to_row_major(&provider.kinetic());
    let scf = run_rhf(
        &provider,
        mol.n_electrons() as usize,
        mol.nuclear_repulsion(),
        &ScfOptions::default(),
    )
    .unwrap();
    assert!(scf.converged);
    let n = ao.n_ao();
    let d_tot: Vec<f64> = (0..n * n)
        .map(|i| scf.density_alpha[i] + scf.density_beta[i])
        .collect();
    (d_tot, t, ao)
}

#[derive(Clone, Copy, Default)]
struct TauAcc {
    integral: f64,
    worst_vw: f64,
}

fn tau_sweep(mol: &Molecule, d_tot: &[f64], ao: &hartree::basis::AoBasis, level: usize) -> TauAcc {
    let grid = MolecularGrid::build(mol, level).unwrap();
    let shells = ao.shells().to_vec();
    let nao = ao.n_ao();
    let weights = grid.weights.clone();
    par_blocks_fold(
        &shells,
        nao,
        &grid.points,
        true,
        TauAcc::default,
        |mut acc, batch, start| {
            let w = &weights[start..start + batch.npts];
            let bd = batch_density_tau(batch, d_tot, true, true);
            for (((&wp, &tau), &rho), g) in w.iter().zip(&bd.tau).zip(&bd.rho).zip(&bd.grad) {
                acc.integral += wp * tau;
                if rho > 1e-8 {
                    let vw = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) / (8.0 * rho);
                    acc.worst_vw = acc.worst_vw.max(vw - tau);
                }
            }
            acc
        },
        |a, b| TauAcc {
            integral: a.integral + b.integral,
            worst_vw: a.worst_vw.max(b.worst_vw),
        },
    )
    .unwrap()
}

#[test]
fn tau_integrates_to_kinetic_energy_water_631g() {
    let mol = water();
    let (d_tot, t, ao) = rhf_setup(&mol, "6-31g");
    let n = ao.n_ao();
    let t_analytic: f64 = (0..n * n).map(|i| d_tot[i] * t[i]).sum();
    let acc = tau_sweep(&mol, &d_tot, &ao, 3);
    println!(
        "∫τ = {:.10}  Tr(D·T) = {:.10}  Δ = {:+.2e}  worst vW violation = {:+.2e}",
        acc.integral,
        t_analytic,
        acc.integral - t_analytic,
        acc.worst_vw
    );
    assert!(
        (acc.integral - t_analytic).abs() < 1e-6,
        "∫τ = {} vs Tr(D·T) = {}",
        acc.integral,
        t_analytic
    );
    assert!(
        acc.worst_vw < 1e-10,
        "von Weizsäcker bound violated by {:e}",
        acc.worst_vw
    );
}

#[test]
#[ignore = "cc-pVDZ tier; run with --release -- --ignored"]
fn tau_integrates_to_kinetic_energy_water_ccpvdz() {
    let mol = water();
    let (d_tot, t, ao) = rhf_setup(&mol, "cc-pvdz");
    let n = ao.n_ao();
    let t_analytic: f64 = (0..n * n).map(|i| d_tot[i] * t[i]).sum();
    let acc = tau_sweep(&mol, &d_tot, &ao, 4);
    assert!(
        (acc.integral - t_analytic).abs() < 1e-6,
        "∫τ = {} vs Tr(D·T) = {}",
        acc.integral,
        t_analytic
    );
    assert!(acc.worst_vw < 1e-10, "vW violated by {:e}", acc.worst_vw);
}
