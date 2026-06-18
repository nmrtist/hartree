use crate::basis::{GthBasisSet, GthPotential, GthSet};
use crate::core::Molecule;
use crate::integrals::integral::Basis;

use crate::periodic::PeriodicError;
use crate::periodic::scf::{NonlocalChannel, PeriodicAtom};

fn periodic_atom(pot: &GthPotential, center: [f64; 3]) -> PeriodicAtom {
    let channels = pot
        .nonlocal
        .iter()
        .filter(|nl| !nl.h.is_empty())
        .map(|nl| NonlocalChannel {
            l: nl.l,
            r_l: nl.r,
            h: nl.h.clone(),
        })
        .collect();
    PeriodicAtom {
        center,
        z_ion: pot.z_ion,
        r_loc: pot.local.r_loc,
        c: pot.local.c.clone(),
        channels,
    }
}

#[derive(Debug, Clone)]
pub struct PeriodicSystem {
    pub basis: Basis,
    pub atoms: Vec<PeriodicAtom>,
    pub n_elec: usize,
}

impl PeriodicSystem {
    pub fn build(
        molecule: &Molecule,
        basis_name: &str,
        basis_set: &GthBasisSet,
        pseudo: &GthSet,
    ) -> Result<Self, PeriodicError> {
        if molecule.atoms.is_empty() {
            return Err(PeriodicError::Config("the cell contains no atoms".into()));
        }
        let mut shells = Vec::new();
        let mut atoms = Vec::with_capacity(molecule.atoms.len());
        let mut z_sum = 0.0;
        for atom in &molecule.atoms {
            if atom.ghost {
                return Err(PeriodicError::Config(
                    "ghost atoms are not supported in periodic GPW (v1)".into(),
                ));
            }
            let z = atom.element.z();
            let sym = atom.element.symbol();
            let center = atom.position;
            let sh = basis_set.shells(basis_name, z, center).map_err(|_| {
                PeriodicError::Config(format!(
                    "no {basis_name} basis for {sym} (Z={z}); available: {:?}",
                    basis_set.basis_names()
                ))
            })?;
            shells.extend(sh);
            let pot = pseudo.get(z).ok_or_else(|| {
                PeriodicError::Config(format!(
                    "no {} pseudopotential for {sym} (Z={z})",
                    pseudo.name
                ))
            })?;
            atoms.push(periodic_atom(pot, center));
            z_sum += pot.z_ion;
        }
        let n_elec = z_sum.round() as usize;
        if !n_elec.is_multiple_of(2) {
            return Err(PeriodicError::Config(format!(
                "odd valence-electron count {n_elec}; v1 is spin-restricted (closed-shell)"
            )));
        }
        Ok(Self {
            basis: Basis::new(shells),
            atoms,
            n_elec,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element, Molecule};
    use crate::periodic::scf::{PeriodicScfOptions, run_scf_periodic};
    use crate::periodic::xc::GridXc;
    use latx::{Cell, KPoint};

    fn si2_molecule(cell: &Cell) -> Molecule {
        let si = Element::from_symbol("Si").unwrap();
        let r0 = cell.frac_to_cart([0.0, 0.0, 0.0]);
        let r1 = cell.frac_to_cart([0.25, 0.25, 0.25]);
        Molecule::new(vec![Atom::new(si, r0), Atom::new(si, r1)], 0, 1)
    }

    #[test]
    fn builds_si_szv_system() {
        let cell = Cell::cubic(10.263).unwrap();
        let mol = si2_molecule(&cell);
        let basis_set = GthBasisSet::load_pade().unwrap();
        let pseudo = GthSet::load_pade().unwrap();
        let sys = PeriodicSystem::build(&mol, "SZV-GTH", &basis_set, &pseudo).unwrap();
        assert_eq!(sys.basis.nao(), 8);
        assert_eq!(sys.atoms.len(), 2);
        assert_eq!(sys.n_elec, 8);
        assert_eq!(sys.basis.atoms().len(), 2);
    }

    #[test]
    fn missing_pseudo_or_basis_errors() {
        let he = Element::from_symbol("He").unwrap();
        let mol = Molecule::new(vec![Atom::new(he, [0.0, 0.0, 0.0])], 0, 1);
        let basis_set = GthBasisSet::load_pade().unwrap();
        let pseudo = GthSet::load_pade().unwrap();
        let err = PeriodicSystem::build(&mol, "SZV-GTH", &basis_set, &pseudo).unwrap_err();
        assert!(matches!(err, PeriodicError::Config(_)), "got {err:?}");
    }

    #[test]
    fn system_scf_converges() {
        let cell = Cell::from_vectors(
            [0.0, 10.263 / 2.0, 10.263 / 2.0],
            [10.263 / 2.0, 0.0, 10.263 / 2.0],
            [10.263 / 2.0, 10.263 / 2.0, 0.0],
        )
        .unwrap();
        let mol = si2_molecule(&cell);
        let basis_set = GthBasisSet::load_pade().unwrap();
        let pseudo = GthSet::load_pade().unwrap();
        let sys = PeriodicSystem::build(&mol, "SZV-GTH", &basis_set, &pseudo).unwrap();

        let xc = GridXc::pade();
        let options = PeriodicScfOptions {
            e_cut: 100.0,
            max_iter: 80,
            ..Default::default()
        };
        let r = run_scf_periodic(
            &sys.basis,
            &cell,
            &[KPoint::gamma()],
            sys.n_elec,
            &sys.atoms,
            &xc,
            &options,
        )
        .unwrap();
        assert!(
            r.converged,
            "SCF did not converge in {} iters",
            r.iterations
        );
        assert!((r.n_elec_grid - 8.0).abs() < 1e-2, "N = {}", r.n_elec_grid);
        assert!(
            r.energy.is_finite() && r.energy < 0.0,
            "energy = {}",
            r.energy
        );
    }
}
