//! Implicit solvation: C-PCM, SMD, ALPB/GBSA, and `.cosmo` export.

use std::sync::OnceLock;

use crate::core::Molecule;
use crate::integrals::IntegralProvider;
use crate::linalg::{Mat, cholesky_lower, cholesky_solve_in_place, gemm, mat_from_row_major};
use crate::scf::{SolventContribution, SolventModel};
use libm::erf;
use thiserror::Error;

pub mod cosmo;
pub mod gbsa;
pub mod smd;
mod surface;

pub use cosmo::{CosmoAtom, CosmoExport, CosmoSegment, parse_cosmo, write_cosmo};
pub use gbsa::{
    DEFAULT_GBSA_GRID, GBSA_PARAMS, GbsaBreakdown, GbsaParams, alpb_solvent, alpb_solvent_names,
    gbsa_energy, gbsa_solvent, gbsa_solvent_names,
};
pub use smd::{SMD_SOLVENTS, SmdSolvent, cds_energy, smd_coulomb_radii, smd_solvent};
pub use surface::{CavitySurface, build_surface, cavity_radius};

#[derive(Debug, Error)]
pub enum SolvError {
    #[error("Lebedev grid with {0} points is not available for the cavity surface")]
    BadGridSize(usize),
    #[error("no C-PCM cavity radius for element Z = {0} (H-Ar only)")]
    NoRadius(usize),
    #[error("dielectric constant must be > 1, got {0}")]
    BadEpsilon(f64),
    #[error("C-PCM surface matrix S is not positive definite")]
    SurfaceNotPositiveDefinite,
    #[error("cavity surface has no exposed points")]
    EmptySurface,
}

pub const SOLVENTS: [(&str, f64); 6] = [
    ("water", 78.3553),
    ("acetonitrile", 35.688),
    ("methanol", 32.613),
    ("dmso", 46.826),
    ("chloroform", 4.7113),
    ("toluene", 2.3741),
];

pub fn solvent_epsilon(name: &str) -> Option<f64> {
    let lower = name.to_ascii_lowercase();
    SOLVENTS
        .iter()
        .find(|(n, _)| *n == lower)
        .map(|&(_, eps)| eps)
}

pub fn f_epsilon(eps: f64) -> f64 {
    if eps.is_infinite() {
        1.0
    } else {
        (eps - 1.0) / eps
    }
}

pub const DEFAULT_GRID: usize = 302;

pub struct CpcmSolver {
    l: Mat,
    f_eps: f64,
}

impl CpcmSolver {
    pub fn new(surface: &CavitySurface, eps: f64) -> Result<Self, SolvError> {
        if eps.partial_cmp(&1.0) != Some(std::cmp::Ordering::Greater) {
            return Err(SolvError::BadEpsilon(eps));
        }
        let k = surface.points.len();
        if k == 0 {
            return Err(SolvError::EmptySurface);
        }
        let mut s = vec![0.0; k * k];
        for i in 0..k {
            let zi = surface.zeta[i];
            s[i * k + i] = zi * (2.0 / std::f64::consts::PI).sqrt() / surface.switch_f[i];
            for j in 0..i {
                let zj = surface.zeta[j];
                let zij = zi * zj / (zi * zi + zj * zj).sqrt();
                let p = surface.points[i];
                let q = surface.points[j];
                let r =
                    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                let v = erf(zij * r) / r;
                s[i * k + j] = v;
                s[j * k + i] = v;
            }
        }
        let l = cholesky_lower(&mat_from_row_major(k, &s))
            .ok_or(SolvError::SurfaceNotPositiveDefinite)?;
        Ok(Self {
            l,
            f_eps: f_epsilon(eps),
        })
    }

    pub fn charges(&self, v: &[f64]) -> Vec<f64> {
        let mut q = v.to_vec();
        cholesky_solve_in_place(&self.l, &mut q, 1);
        for x in &mut q {
            *x *= -self.f_eps;
        }
        q
    }

    pub fn f_eps(&self) -> f64 {
        self.f_eps
    }
}

pub fn interaction_energy(q: &[f64], v: &[f64]) -> f64 {
    q.iter().zip(v).map(|(a, b)| a * b).sum::<f64>() * 0.5
}

const POINT_BLOCK: usize = 512;

const CACHE_MAX_DOUBLES: usize = 32_000_000;

pub struct Cpcm<'a, P: IntegralProvider> {
    provider: &'a P,
    surface: CavitySurface,
    solver: CpcmSolver,
    v_nuc: Vec<f64>,
    n: usize,
    cache: OnceLock<Option<Vec<f64>>>,
}

impl<'a, P: IntegralProvider> Cpcm<'a, P> {
    pub fn new(
        provider: &'a P,
        molecule: &Molecule,
        eps: f64,
        ng: usize,
    ) -> Result<Self, SolvError> {
        let radii = molecule
            .atoms
            .iter()
            .map(|a| surface::cavity_radius(a.element.z() as usize))
            .collect::<Result<Vec<f64>, _>>()?;
        Self::with_radii(provider, molecule, eps, ng, &radii)
    }

    pub fn with_radii(
        provider: &'a P,
        molecule: &Molecule,
        eps: f64,
        ng: usize,
        radii: &[f64],
    ) -> Result<Self, SolvError> {
        let centers: Vec<[f64; 3]> = molecule.atoms.iter().map(|a| a.position).collect();
        let surface = build_surface(&centers, radii, ng)?;
        let solver = CpcmSolver::new(&surface, eps)?;

        let v_nuc = surface
            .points
            .iter()
            .zip(&surface.zeta)
            .map(|(p, &zeta)| {
                let mut v = 0.0;
                for atom in &molecule.atoms {
                    let c = atom.position;
                    let r = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2))
                        .sqrt();
                    v += f64::from(atom.element.z()) * erf(zeta * r) / r;
                }
                v
            })
            .collect();

        Ok(Self {
            provider,
            surface,
            solver,
            v_nuc,
            n: provider.n_basis(),
            cache: OnceLock::new(),
        })
    }

    pub fn n_points(&self) -> usize {
        self.surface.points.len()
    }

    fn block_charges(&self, lo: usize, hi: usize) -> Vec<([f64; 3], f64)> {
        (lo..hi)
            .map(|k| (self.surface.points[k], self.surface.zeta[k]))
            .collect()
    }

    fn cached_tensor(&self) -> Option<&Vec<f64>> {
        self.cache
            .get_or_init(|| {
                let k = self.surface.points.len();
                (self.n * self.n * k <= CACHE_MAX_DOUBLES)
                    .then(|| self.provider.charge_potential_3c(&self.block_charges(0, k)))
            })
            .as_ref()
    }
}

impl<P: IntegralProvider> SolventModel for Cpcm<'_, P> {
    fn eval(&self, d_total: &[f64], n: usize) -> SolventContribution {
        debug_assert_eq!(n, self.n);
        let npts = self.surface.points.len();

        let mut v_el = vec![0.0; npts];
        let mut v_solv = vec![0.0; n * n];

        if let Some(t) = self.cached_tensor() {
            v_el = gemm(d_total, 1, n * n, t, npts);
            let v = self.total_potential(&v_el);
            let q = self.solver.charges(&v);
            let e_solv = interaction_energy(&q, &v);
            let f = gemm(t, n * n, npts, &q, 1);
            for i in 0..n * n {
                v_solv[i] = -f[i];
            }
            return SolventContribution { e_solv, v_solv };
        }

        let mut lo = 0;
        while lo < npts {
            let hi = (lo + POINT_BLOCK).min(npts);
            let t = self
                .provider
                .charge_potential_3c(&self.block_charges(lo, hi));
            let ve = gemm(d_total, 1, n * n, &t, hi - lo);
            v_el[lo..hi].copy_from_slice(&ve);
            lo = hi;
        }
        let v = self.total_potential(&v_el);
        let q = self.solver.charges(&v);
        let e_solv = interaction_energy(&q, &v);

        let mut lo = 0;
        while lo < npts {
            let hi = (lo + POINT_BLOCK).min(npts);
            let t = self
                .provider
                .charge_potential_3c(&self.block_charges(lo, hi));
            let f = gemm(&t, n * n, hi - lo, &q[lo..hi], 1);
            for i in 0..n * n {
                v_solv[i] -= f[i];
            }
            lo = hi;
        }
        SolventContribution { e_solv, v_solv }
    }
}

impl<P: IntegralProvider> Cpcm<'_, P> {
    fn total_potential(&self, v_el: &[f64]) -> Vec<f64> {
        self.v_nuc
            .iter()
            .zip(v_el)
            .map(|(vn, ve)| vn - ve)
            .collect()
    }

    fn electronic_potential(&self, d_total: &[f64], n: usize) -> Vec<f64> {
        let npts = self.surface.points.len();
        if let Some(t) = self.cached_tensor() {
            return gemm(d_total, 1, n * n, t, npts);
        }
        let mut v_el = vec![0.0; npts];
        let mut lo = 0;
        while lo < npts {
            let hi = (lo + POINT_BLOCK).min(npts);
            let t = self
                .provider
                .charge_potential_3c(&self.block_charges(lo, hi));
            let ve = gemm(d_total, 1, n * n, &t, hi - lo);
            v_el[lo..hi].copy_from_slice(&ve);
            lo = hi;
        }
        v_el
    }

    pub fn cosmo_segments(
        &self,
        d_total: &[f64],
        n: usize,
    ) -> (Vec<crate::solv::cosmo::CosmoSegment>, f64) {
        let v_el = self.electronic_potential(d_total, n);
        let v = self.total_potential(&v_el);
        let q = self.solver.charges(&v);
        let diel_energy = interaction_energy(&q, &v);
        let segments = (0..self.surface.points.len())
            .map(|k| crate::solv::cosmo::CosmoSegment {
                atom: self.surface.atom[k] + 1,
                position: self.surface.points[k],
                charge: q[k],
                area: crate::solv::cosmo::bohr2_to_aa2(self.surface.area[k]),
                potential: v[k],
            })
            .collect();
        (segments, diel_energy)
    }
}
