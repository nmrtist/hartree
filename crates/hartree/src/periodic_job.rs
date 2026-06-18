use crate::basis::{GthBasisSet, GthSet};
use crate::core::Molecule;
use crate::periodic::{
    Cell, GridXc, PeriodicScfOptions, PeriodicScfResult, PeriodicSystem, periodic_forces,
    periodic_stress, run_scf_periodic,
};
use latx::KPoint;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicFunctional {
    Pade,
    Lda,
}

impl PeriodicFunctional {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "pade" | "pz" | "gth-pade" => Ok(Self::Pade),
            "lda" | "pw92" | "svwn" => Ok(Self::Lda),
            other => Err(format!(
                "unknown periodic functional {other:?}; supported: pade (GTH-PADE LDA, default), lda (Slater+PW92)"
            )),
        }
    }

    fn grid_xc(self) -> Result<GridXc, String> {
        match self {
            Self::Pade => Ok(GridXc::pade()),
            Self::Lda => GridXc::lda().map_err(|e| e.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PeriodicJob {
    pub molecule: Molecule,
    pub cell: Cell,
    pub kpoints: Vec<KPoint>,
    pub basis_name: String,
    pub basis_set: GthBasisSet,
    pub pseudo: GthSet,
    pub functional: PeriodicFunctional,
    pub options: PeriodicScfOptions,
    pub forces: bool,
    pub stress: bool,
}

impl PeriodicJob {
    pub fn gth_pade(
        molecule: Molecule,
        cell: Cell,
        kpoints: Vec<KPoint>,
        basis_name: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            molecule,
            cell,
            kpoints,
            basis_name: basis_name.to_string(),
            basis_set: GthBasisSet::load_pade().map_err(|e| e.to_string())?,
            pseudo: GthSet::load_pade().map_err(|e| e.to_string())?,
            functional: PeriodicFunctional::Pade,
            options: PeriodicScfOptions::default(),
            forces: false,
            stress: false,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PeriodicJobResult {
    pub scf: PeriodicScfResult,
    pub forces: Option<Vec<[f64; 3]>>,
    pub stress: Option<[[f64; 3]; 3]>,
}

pub fn run_periodic(job: &PeriodicJob) -> Result<PeriodicJobResult, String> {
    let sys = PeriodicSystem::build(&job.molecule, &job.basis_name, &job.basis_set, &job.pseudo)
        .map_err(|e| e.to_string())?;
    let xc = job.functional.grid_xc()?;

    let scf = run_scf_periodic(
        &sys.basis,
        &job.cell,
        &job.kpoints,
        sys.n_elec,
        &sys.atoms,
        &xc,
        &job.options,
    )
    .map_err(|e| e.to_string())?;

    let forces = if job.forces {
        Some(
            periodic_forces(
                &sys.basis,
                &job.cell,
                &job.kpoints,
                sys.n_elec,
                &sys.atoms,
                &xc,
                &job.options,
            )
            .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };

    let stress = if job.stress {
        Some(
            periodic_stress(
                &sys.basis,
                &job.cell,
                &job.kpoints,
                sys.n_elec,
                &sys.atoms,
                &xc,
                &job.options,
            )
            .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };

    Ok(PeriodicJobResult {
        scf,
        forces,
        stress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si2_primitive() -> (Molecule, Cell) {
        let a = 10.263;
        let cell = Cell::from_vectors(
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        )
        .unwrap();
        let si = crate::core::Element::from_symbol("Si").unwrap();
        let r0 = cell.frac_to_cart([0.0, 0.0, 0.0]);
        let r1 = cell.frac_to_cart([0.25, 0.25, 0.25]);
        let mol = Molecule::new(
            vec![
                crate::core::Atom::new(si, r0),
                crate::core::Atom::new(si, r1),
            ],
            0,
            1,
        );
        (mol, cell)
    }

    #[test]
    fn run_periodic_energy() {
        let (mol, cell) = si2_primitive();
        let mut job = PeriodicJob::gth_pade(mol, cell, vec![KPoint::gamma()], "SZV-GTH").unwrap();
        job.options.e_cut = 100.0;
        job.options.max_iter = 80;
        let r = run_periodic(&job).unwrap();
        assert!(r.scf.converged, "SCF did not converge");
        assert!(
            (r.scf.n_elec_grid - 8.0).abs() < 1e-2,
            "N = {}",
            r.scf.n_elec_grid
        );
        assert!(r.scf.energy.is_finite() && r.scf.energy < 0.0);
        assert!(r.forces.is_none() && r.stress.is_none());
    }

    #[test]
    fn run_periodic_with_forces() {
        let (mol, cell) = si2_primitive();
        let mut job = PeriodicJob::gth_pade(mol, cell, vec![KPoint::gamma()], "SZV-GTH").unwrap();
        job.options.e_cut = 80.0;
        job.options.max_iter = 100;
        job.forces = true;
        let r = run_periodic(&job).unwrap();
        let f = r.forces.expect("forces requested");
        assert_eq!(f.len(), 2);
        for (ax, (&f0, &f1)) in f[0].iter().zip(f[1].iter()).enumerate() {
            let net = f0 + f1;
            assert!(net.abs() < 2e-3, "net force axis {ax} = {net}");
            assert!(f0.is_finite() && f1.is_finite());
        }
    }

    #[test]
    fn functional_from_name() {
        assert_eq!(
            PeriodicFunctional::from_name("PADE").unwrap(),
            PeriodicFunctional::Pade
        );
        assert_eq!(
            PeriodicFunctional::from_name("lda").unwrap(),
            PeriodicFunctional::Lda
        );
        assert!(PeriodicFunctional::from_name("b3lyp").is_err());
    }
}
