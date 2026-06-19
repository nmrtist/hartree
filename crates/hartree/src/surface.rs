use crate::basis::BasisSet;
use crate::core::{Atom, Element, Molecule};
use crate::dft::{FunctionalSpec, GridXc};
use crate::disp::Dispersion;
use crate::grad::{hf_gradient, ks_gradient};
use crate::integrals::ConventionalProvider;
use crate::opt::{OptError, Surface};
use crate::scf::{Reference, ScfOptions, ScfResult, SolventModel, XcContributor, run_scf_with_env};
use crate::solv::Cpcm;

pub struct HfSurface {
    elements: Vec<Element>,
    charge: i32,
    multiplicity: u32,
    basis: String,
    reference: Reference,
    n_alpha: usize,
    n_beta: usize,
    scf_options: ScfOptions,
    functional: Option<(FunctionalSpec, usize)>,
    dispersion: Option<Dispersion>,
    solvent_eps: Option<f64>,
    smd: Option<crate::solv::SmdSolvent>,
    gcp: Option<crate::disp::GcpParams>,
    srb: Option<crate::disp::SrbParams>,
    cache: Option<CachedPoint>,
}

struct CachedPoint {
    positions: Vec<[f64; 3]>,
    provider: ConventionalProvider,
    molecule: Molecule,
    scf: ScfResult,
    xc: Option<GridXc>,
}

impl HfSurface {
    pub fn new(molecule: &Molecule, basis: &str, reference: Reference) -> Result<Self, String> {
        let ecp_core = BasisSet::load(basis)
            .map_err(|e| e.to_string())?
            .ecp_core_electrons(molecule) as i64;
        let n_elec = molecule.n_electrons() - ecp_core;
        if n_elec < 0 {
            return Err("charge exceeds the nuclear charge (negative electron count)".into());
        }
        let n_elec = n_elec as usize;
        let two_s = (molecule.multiplicity.saturating_sub(1)) as usize;
        if two_s > n_elec {
            return Err("multiplicity is too high for the electron count".into());
        }
        let n_alpha = (n_elec + two_s) / 2;
        let n_beta = (n_elec - two_s) / 2;
        if reference == Reference::Rhf && n_alpha != n_beta {
            return Err(
                "RHF requires a closed shell; use Reference::Uhf or Reference::Rohf".into(),
            );
        }
        Ok(Self {
            elements: molecule.atoms.iter().map(|a| a.element).collect(),
            charge: molecule.charge,
            multiplicity: molecule.multiplicity,
            basis: basis.to_string(),
            reference,
            n_alpha,
            n_beta,
            scf_options: ScfOptions {
                energy_tol: 1e-11,
                error_tol: 1e-9,
                ..ScfOptions::default()
            },
            functional: None,
            dispersion: None,
            solvent_eps: None,
            smd: None,
            gcp: None,
            srb: None,
            cache: None,
        })
    }

    pub fn new_dft(
        molecule: &Molecule,
        basis: &str,
        reference: Reference,
        functional: FunctionalSpec,
        grid_level: usize,
    ) -> Result<Self, String> {
        let mut surface = Self::new(molecule, basis, reference)?;
        surface.set_scf_convergence(1e-9, 1e-6);
        surface.functional = Some((functional, grid_level));
        Ok(surface)
    }

    pub fn set_dispersion(&mut self, dispersion: Dispersion) {
        self.dispersion = Some(dispersion);
    }

    pub fn set_gcp(&mut self, params: crate::disp::GcpParams) {
        self.gcp = Some(params);
    }

    pub fn set_srb(&mut self, params: crate::disp::SrbParams) {
        self.srb = Some(params);
    }

    pub fn set_solvent(&mut self, eps: f64) {
        self.solvent_eps = Some(eps);
    }

    pub fn set_smd(&mut self, solvent: crate::solv::SmdSolvent) {
        self.smd = Some(solvent);
    }

    /// Raise the SCF iteration limit (default 128).
    pub fn set_scf_max_iter(&mut self, max_iter: usize) {
        self.scf_options.max_iter = max_iter;
    }

    /// Virtual-orbital level shift (a.u.) to damp SCF oscillation at small-gap
    /// geometries; does not change the converged density.
    pub fn set_scf_level_shift(&mut self, shift: f64) {
        self.scf_options.level_shift = shift;
    }

    /// SCF convergence thresholds (energy change, DIIS error norm); a small-gap TS
    /// may need a looser pair to converge off the error floor.
    pub fn set_scf_convergence(&mut self, energy_tol: f64, error_tol: f64) {
        (self.scf_options.energy_tol, self.scf_options.error_tol) = (energy_tol, error_tol);
    }

    pub fn last_scf(&self) -> Option<&ScfResult> {
        self.cache.as_ref().map(|c| &c.scf)
    }

    fn molecule_at(&self, positions: &[[f64; 3]]) -> Molecule {
        let atoms = self
            .elements
            .iter()
            .zip(positions)
            .map(|(e, p)| Atom::new(*e, *p))
            .collect();
        Molecule::new(atoms, self.charge, self.multiplicity)
    }

    fn eval(&mut self, positions: &[[f64; 3]]) -> Result<&CachedPoint, OptError> {
        let hit = self
            .cache
            .as_ref()
            .is_some_and(|c| c.positions.as_slice() == positions);
        if !hit {
            let molecule = self.molecule_at(positions);
            let ao = BasisSet::load(&self.basis)
                .map_err(|e| OptError::Evaluation(e.to_string()))?
                .build(&molecule)
                .map_err(|e| OptError::Evaluation(e.to_string()))?;
            let setup = crate::job::ecp_setup(&molecule, &ao);
            let xc = match &self.functional {
                Some((spec, level)) => Some(
                    GridXc::new(&molecule, &ao, spec, *level)
                        .map_err(|e| OptError::Evaluation(e.to_string()))?,
                ),
                None => None,
            };
            let provider =
                ConventionalProvider::new(ao.into_integral(), setup.charges).with_ecps(setup.ecps);
            let cpcm = if let Some(s) = &self.smd {
                let zs: Vec<usize> = molecule
                    .atoms
                    .iter()
                    .map(|a| a.element.z() as usize)
                    .collect();
                let radii = crate::solv::smd_coulomb_radii(&zs, s.alpha)
                    .map_err(|e| OptError::Evaluation(e.to_string()))?;
                Some(
                    Cpcm::with_radii(
                        &provider,
                        &molecule,
                        s.epsilon,
                        crate::solv::DEFAULT_GRID,
                        &radii,
                    )
                    .map_err(|e| OptError::Evaluation(e.to_string()))?,
                )
            } else {
                self.solvent_eps
                    .map(|eps| Cpcm::new(&provider, &molecule, eps, crate::solv::DEFAULT_GRID))
                    .transpose()
                    .map_err(|e| OptError::Evaluation(e.to_string()))?
            };
            let scf = run_scf_with_env(
                &provider,
                self.n_alpha,
                self.n_beta,
                self.reference,
                setup.nuclear_repulsion,
                &self.scf_options,
                xc.as_ref().map(|x| x as &dyn XcContributor),
                cpcm.as_ref().map(|c| c as &dyn SolventModel),
            )
            .map_err(|e| OptError::Evaluation(e.to_string()))?;
            drop(cpcm);
            if !scf.converged {
                return Err(OptError::ScfNotConverged {
                    iterations: scf.iterations,
                });
            }
            self.cache = Some(CachedPoint {
                positions: positions.to_vec(),
                provider,
                molecule,
                scf,
                xc,
            });
        }
        Ok(self.cache.as_ref().unwrap())
    }

    /// Cartesian finite-difference Hessian with the `2·ndof` displaced-gradient
    /// evaluations (an independent SCF + gradient each) run in parallel.
    fn fd_hessian_parallel(
        &mut self,
        positions: &[[f64; 3]],
        fd_step: f64,
    ) -> Result<Vec<f64>, OptError> {
        use rayon::prelude::*;

        let natom = positions.len();
        let ndof = 3 * natom;

        // Owned snapshot so the parallel closure captures `Sync` data, not `&self`.
        let elements = self.elements.clone();
        let charge = self.charge;
        let multiplicity = self.multiplicity;
        let basis = BasisSet::load(&self.basis).map_err(|e| OptError::Evaluation(e.to_string()))?;
        let reference = self.reference;
        let n_alpha = self.n_alpha;
        let n_beta = self.n_beta;
        let scf_options = self.scf_options.clone();
        let functional = self.functional.clone();
        let dispersion = self.dispersion;
        let gcp = self.gcp;
        let srb = self.srb;

        let evals: Vec<(usize, f64)> = (0..ndof)
            .flat_map(|dof| [(dof, fd_step), (dof, -fd_step)])
            .collect();

        let grads: Result<Vec<Vec<[f64; 3]>>, OptError> = evals
            .par_iter()
            .map(|&(dof, delta)| {
                let mut x = positions.to_vec();
                x[dof / 3][dof % 3] += delta;
                gradient_at(
                    &elements,
                    charge,
                    multiplicity,
                    &basis,
                    reference,
                    n_alpha,
                    n_beta,
                    &scf_options,
                    &functional,
                    dispersion,
                    gcp,
                    srb,
                    &x,
                )
            })
            .collect();
        let grads = grads?;

        let mut h = vec![0.0f64; ndof * ndof];
        for dof in 0..ndof {
            let gp = &grads[2 * dof];
            let gm = &grads[2 * dof + 1];
            for j in 0..ndof {
                let gpj = gp[j / 3][j % 3];
                let gmj = gm[j / 3][j % 3];
                h[dof * ndof + j] = (gpj - gmj) / (2.0 * fd_step);
            }
        }
        // Symmetrize away the finite-difference asymmetry.
        for i in 0..ndof {
            for j in (i + 1)..ndof {
                let avg = 0.5 * (h[i * ndof + j] + h[j * ndof + i]);
                h[i * ndof + j] = avg;
                h[j * ndof + i] = avg;
            }
        }
        Ok(h)
    }
}

/// Stateless SCF + analytic gradient at one geometry (no `&mut` cache), so
/// [`HfSurface::fd_hessian_parallel`] can evaluate displaced geometries concurrently.
#[allow(clippy::too_many_arguments)]
fn gradient_at(
    elements: &[Element],
    charge: i32,
    multiplicity: u32,
    basis: &BasisSet,
    reference: Reference,
    n_alpha: usize,
    n_beta: usize,
    scf_options: &ScfOptions,
    functional: &Option<(FunctionalSpec, usize)>,
    dispersion: Option<Dispersion>,
    gcp: Option<crate::disp::GcpParams>,
    srb: Option<crate::disp::SrbParams>,
    positions: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, OptError> {
    let atoms = elements
        .iter()
        .zip(positions)
        .map(|(e, p)| Atom::new(*e, *p))
        .collect();
    let molecule = Molecule::new(atoms, charge, multiplicity);

    let ao = basis
        .build(&molecule)
        .map_err(|e| OptError::Evaluation(e.to_string()))?;
    let setup = crate::job::ecp_setup(&molecule, &ao);
    let xc = match functional {
        Some((spec, level)) => Some(
            GridXc::new(&molecule, &ao, spec, *level)
                .map_err(|e| OptError::Evaluation(e.to_string()))?,
        ),
        None => None,
    };
    let provider =
        ConventionalProvider::new(ao.into_integral(), setup.charges).with_ecps(setup.ecps);
    let scf = run_scf_with_env(
        &provider,
        n_alpha,
        n_beta,
        reference,
        setup.nuclear_repulsion,
        scf_options,
        xc.as_ref().map(|x| x as &dyn XcContributor),
        None,
    )
    .map_err(|e| OptError::Evaluation(e.to_string()))?;
    if !scf.converged {
        return Err(OptError::ScfNotConverged {
            iterations: scf.iterations,
        });
    }

    let restricted = reference == Reference::Rhf;
    let mut grad = match &xc {
        None => hf_gradient(&provider, &molecule, &scf.density_alpha, &scf.density_beta)
            .map_err(|e| OptError::Evaluation(e.to_string()))?,
        Some(xc) => ks_gradient(
            &provider,
            &molecule,
            xc as &dyn XcContributor,
            &scf.density_alpha,
            &scf.density_beta,
            restricted,
        )
        .map_err(|e| OptError::Evaluation(e.to_string()))?,
    };
    if let Some(disp) = dispersion {
        let (_, disp_grad) = disp.energy_gradient(&molecule);
        for (g, d) in grad.iter_mut().zip(&disp_grad) {
            for k in 0..3 {
                g[k] += d[k];
            }
        }
    }
    if let Some(p) = gcp {
        let (_, gcp_grad) = crate::disp::gcp_energy_gradient(&molecule, &p);
        for (g, d) in grad.iter_mut().zip(&gcp_grad) {
            for k in 0..3 {
                g[k] += d[k];
            }
        }
    }
    if let Some(p) = srb {
        let (_, srb_grad) = crate::disp::srb_energy_gradient(&molecule, &p);
        for (g, d) in grad.iter_mut().zip(&srb_grad) {
            for k in 0..3 {
                g[k] += d[k];
            }
        }
    }
    Ok(grad)
}

impl Surface for HfSurface {
    fn energy(&mut self, positions: &[[f64; 3]]) -> Result<f64, OptError> {
        let dispersion = self.dispersion;
        let gcp = self.gcp;
        let srb = self.srb;
        let smd = self.smd;
        let point = self.eval(positions)?;
        let e_disp = dispersion.map_or(0.0, |d| d.energy(&point.molecule));
        let e_gcp = gcp.map_or(0.0, |p| crate::disp::gcp_energy(&point.molecule, &p));
        let e_srb = srb.map_or(0.0, |p| crate::disp::srb_energy(&point.molecule, &p));
        let e_cds = match &smd {
            Some(s) => {
                let zs: Vec<usize> = point
                    .molecule
                    .atoms
                    .iter()
                    .map(|a| a.element.z() as usize)
                    .collect();
                let coords: Vec<[f64; 3]> =
                    point.molecule.atoms.iter().map(|a| a.position).collect();
                crate::solv::cds_energy(&zs, &coords, s, crate::solv::smd::DEFAULT_SASA_GRID)
                    .map_err(|e| OptError::Evaluation(e.to_string()))?
            }
            None => 0.0,
        };
        Ok(point.scf.energy + e_disp + e_gcp + e_srb + e_cds)
    }

    fn fd_hessian(
        &mut self,
        positions: &[[f64; 3]],
        fd_step: f64,
    ) -> Option<Result<Vec<f64>, OptError>> {
        // Solvent / ROHF expose no analytic gradient — let the driver fall back
        // to its serial finite difference.
        if self.reference == Reference::Rohf || self.solvent_eps.is_some() || self.smd.is_some() {
            return None;
        }
        Some(self.fd_hessian_parallel(positions, fd_step))
    }

    fn analytic_gradient(
        &mut self,
        positions: &[[f64; 3]],
    ) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        if self.reference == Reference::Rohf || self.solvent_eps.is_some() || self.smd.is_some() {
            return None;
        }
        let restricted = self.reference == Reference::Rhf;
        let dispersion = self.dispersion;
        let gcp = self.gcp;
        let srb = self.srb;
        // `eval` returns either `OptError::ScfNotConverged` (a capped SCF) or its
        // own `OptError::Evaluation` (basis/grid/solvent build); both flow straight
        // through `and_then` unchanged. Only the gradient build below adds further
        // `OptError::Evaluation` failures.
        let result = self.eval(positions).and_then(|point| {
            let mut grad = match &point.xc {
                None => hf_gradient(
                    &point.provider,
                    &point.molecule,
                    &point.scf.density_alpha,
                    &point.scf.density_beta,
                )
                .map_err(|e| OptError::Evaluation(e.to_string()))?,
                Some(xc) => ks_gradient(
                    &point.provider,
                    &point.molecule,
                    xc as &dyn XcContributor,
                    &point.scf.density_alpha,
                    &point.scf.density_beta,
                    restricted,
                )
                .map_err(|e| OptError::Evaluation(e.to_string()))?,
            };
            if let Some(disp) = dispersion {
                let (_, disp_grad) = disp.energy_gradient(&point.molecule);
                for (g, d) in grad.iter_mut().zip(&disp_grad) {
                    for k in 0..3 {
                        g[k] += d[k];
                    }
                }
            }
            if let Some(p) = gcp {
                let (_, gcp_grad) = crate::disp::gcp_energy_gradient(&point.molecule, &p);
                for (g, d) in grad.iter_mut().zip(&gcp_grad) {
                    for k in 0..3 {
                        g[k] += d[k];
                    }
                }
            }
            if let Some(p) = srb {
                let (_, srb_grad) = crate::disp::srb_energy_gradient(&point.molecule, &p);
                for (g, d) in grad.iter_mut().zip(&srb_grad) {
                    for k in 0..3 {
                        g[k] += d[k];
                    }
                }
            }
            Ok(grad)
        });
        Some(result)
    }
}

#[cfg(test)]
#[path = "surface_tests.rs"]
mod tests;
