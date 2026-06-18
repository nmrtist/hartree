use hartree::HfSurface;
use hartree::core::Molecule;
use hartree::opt::Surface;
use hartree::scf::Reference;

fn water() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n").unwrap()
}

#[test]
fn solvated_surface_declines_analytic_gradient() {
    let mol = water();
    let positions: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();

    let mut gas = HfSurface::new(&mol, "sto-3g", Reference::Rhf).unwrap();
    assert!(
        gas.analytic_gradient(&positions).is_some(),
        "gas-phase RHF must offer the analytic gradient"
    );

    let mut solvated = HfSurface::new(&mol, "sto-3g", Reference::Rhf).unwrap();
    solvated.set_solvent(78.3553);
    assert!(
        solvated.analytic_gradient(&positions).is_none(),
        "solvated surface must decline the analytic gradient (FD fallback)"
    );
    let e_solvated = solvated.energy(&positions).unwrap();
    let e_gas = gas.energy(&positions).unwrap();
    assert!(e_solvated < e_gas);
}
