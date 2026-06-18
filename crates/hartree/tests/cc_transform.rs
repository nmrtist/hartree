use hartree::basis::BasisSet;
use hartree::cc::{column_block, core_hamiltonian_mo, transform_block};
use hartree::core::Molecule;
use hartree::integrals::{ConventionalProvider, InCoreEri, IntegralProvider};
use hartree::linalg::mat_to_row_major;
use hartree::scf::{ScfOptions, run_rhf};

fn provider_for(mol: &Molecule, basis: &str) -> ConventionalProvider {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    ConventionalProvider::new(ao.into_integral(), charges)
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            hartree::core::Atom::new(
                hartree::core::Element::from_symbol("O").unwrap(),
                [0.0, -0.143225816552, 0.0],
            ),
            hartree::core::Atom::new(
                hartree::core::Element::from_symbol("H").unwrap(),
                [1.638036840407, 1.136548822547, 0.0],
            ),
            hartree::core::Atom::new(
                hartree::core::Element::from_symbol("H").unwrap(),
                [-1.638036840407, 1.136548822547, 0.0],
            ),
        ],
        0,
        1,
    )
}

fn check_rhf_rebuild(basis: &str) {
    let mol = water();
    let provider = provider_for(&mol, basis);
    let opts = ScfOptions {
        energy_tol: 1e-12,
        error_tol: 1e-10,
        ..ScfOptions::default()
    };
    let scf = run_rhf(&provider, 10, mol.nuclear_repulsion(), &opts).unwrap();
    assert!(scf.converged);

    let n = scf.n_basis;
    let m = scf.n_orbitals;
    let no = scf.n_alpha; // doubly occupied count
    let c_occ = column_block(&scf.mo_coeff_alpha, n, m, 0, no);

    let h_ao = mat_to_row_major(&provider.core_hamiltonian());
    let h_oo = core_hamiltonian_mo(&h_ao, n, &c_occ, &c_occ); // [no, no]
    let oooo = transform_block(provider.ao_eri(), n, [&c_occ, &c_occ, &c_occ, &c_occ]);
    let g = oooo.data();
    let idx = |p: usize, q: usize, r: usize, s: usize| ((p * no + q) * no + r) * no + s;

    let mut e_elec = 0.0;
    for i in 0..no {
        e_elec += 2.0 * h_oo[i * no + i];
    }
    for i in 0..no {
        for j in 0..no {
            e_elec += 2.0 * g[idx(i, i, j, j)] - g[idx(i, j, i, j)];
        }
    }
    let e_total = e_elec + mol.nuclear_repulsion();

    let delta = e_total - scf.energy;
    eprintln!(
        "RHF-from-MO {basis}: {e_total:.12} vs SCF {:.12} (Δ = {delta:.2e})",
        scf.energy
    );
    assert!(
        delta.abs() < 1e-11,
        "{basis}: RHF rebuilt from MO integrals {e_total:.12} vs SCF {:.12} (Δ = {delta:.2e})",
        scf.energy
    );
}

#[test]
fn rhf_rebuild_sto3g() {
    check_rhf_rebuild("sto-3g");
}

#[test]
#[ignore = "cc-pVDZ AO→MO transform (slow tier per the tiering policy); run with --release -- --ignored"]
fn rhf_rebuild_ccpvdz() {
    check_rhf_rebuild("cc-pvdz");
}
