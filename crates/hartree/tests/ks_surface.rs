use hartree::HfSurface;
use hartree::core::Molecule;
use hartree::dft::FunctionalSpec;
use hartree::opt::{OptError, OptOptions, Surface, optimize};
use hartree::scf::Reference;

fn water() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n").unwrap()
}

fn dft_surface(functional: &str) -> HfSurface {
    let spec = FunctionalSpec::parse(functional).unwrap();
    HfSurface::new_dft(&water(), "sto-3g", Reference::Rhf, spec, 3).unwrap()
}

struct FdOnly(HfSurface);

impl Surface for FdOnly {
    fn energy(&mut self, positions: &[[f64; 3]]) -> Result<f64, OptError> {
        self.0.energy(positions)
    }
    fn analytic_gradient(
        &mut self,
        _positions: &[[f64; 3]],
    ) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        None
    }
}

#[test]
fn ks_surface_offers_analytic_gradient() {
    let positions: Vec<[f64; 3]> = water().atoms.iter().map(|a| a.position).collect();
    for functional in ["svwn", "pbe", "blyp", "b3lyp", "pbe0"] {
        let mut s = dft_surface(functional);
        assert!(
            s.analytic_gradient(&positions).is_some(),
            "{functional}: LDA/GGA/hybrid KS must offer the analytic gradient"
        );
    }
}

#[test]
fn meta_gga_surface_offers_analytic_gradient() {
    let positions: Vec<[f64; 3]> = water().atoms.iter().map(|a| a.position).collect();
    for functional in ["tpss", "r2scan"] {
        let mut s = dft_surface(functional);
        assert!(
            s.analytic_gradient(&positions).is_some(),
            "{functional}: tau-meta-GGA must offer the analytic gradient (vtau term)"
        );
    }
}

#[test]
fn solvated_ks_surface_declines_analytic_gradient() {
    let positions: Vec<[f64; 3]> = water().atoms.iter().map(|a| a.position).collect();
    let mut s = dft_surface("pbe");
    s.set_solvent(78.3553);
    assert!(
        s.analytic_gradient(&positions).is_none(),
        "solvated KS must stay on the FD fallback"
    );
}

#[test]
#[ignore = "slow; run with --include-ignored"]
fn ks_opt_analytic_matches_fd_minimum() {
    let mol = Molecule::from_xyz(
        "3\nnear-minimum water\nO 0 0 0.18776140\nH 0 0.78196482 -0.51538070\nH 0 -0.76196482 -0.50038070\n",
    )
    .unwrap();
    let opts = OptOptions::default();

    let mut analytic_surface = HfSurface::new_dft(
        &mol,
        "sto-3g",
        Reference::Rhf,
        FunctionalSpec::parse("pbe").unwrap(),
        3,
    )
    .unwrap();
    let positions: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    assert!(analytic_surface.analytic_gradient(&positions).is_some());
    let opt_an = optimize(&mol, &mut analytic_surface, &opts).unwrap();
    assert!(opt_an.converged, "analytic-path KS optimization converged");

    let mut fd_surface = FdOnly(
        HfSurface::new_dft(
            &mol,
            "sto-3g",
            Reference::Rhf,
            FunctionalSpec::parse("pbe").unwrap(),
            3,
        )
        .unwrap(),
    );
    let opt_fd = optimize(&mol, &mut fd_surface, &opts).unwrap();
    assert!(opt_fd.converged, "FD-path KS optimization converged");

    let de = (opt_an.energy - opt_fd.energy).abs();
    eprintln!(
        "PBE/sto-3g opt: analytic {:.10} ({} steps)  fd {:.10} ({} steps)  dE {de:.2e}",
        opt_an.energy, opt_an.iterations, opt_fd.energy, opt_fd.iterations
    );
    assert!(
        de < 1e-8,
        "analytic vs FD minimum energy differ by {de:.2e}"
    );
}
