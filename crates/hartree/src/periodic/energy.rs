use crate::integrals::integral::Basis;
use crate::integrals::integral::periodic::{
    RealSpaceGrid, collocate_density, hartree, integrate_potential,
};
use crate::periodic::PeriodicError;
use crate::periodic::xc::GridXc;

#[must_use]
pub fn kinetic_energy(basis: &Basis, p: &[f64]) -> f64 {
    let t = basis.kinetic();
    p.iter().zip(&t).map(|(&pij, &tij)| pij * tij).sum()
}

pub struct LocalKsBuild {
    pub v_matrix: Vec<f64>,
    pub e_hartree: f64,
    pub e_xc: f64,
}

pub fn build_local_ks(
    basis: &Basis,
    p: &[f64],
    grid: &RealSpaceGrid,
    xc: &GridXc,
) -> Result<LocalKsBuild, PeriodicError> {
    let n_r = collocate_density(basis, p, grid);
    let (v_h, e_hartree) = hartree(&n_r, grid);
    let (e_xc, v_xc) = xc.energy_potential(&n_r, grid.dv())?;
    let v_tot: Vec<f64> = v_h.iter().zip(&v_xc).map(|(&a, &b)| a + b).collect();
    let v_matrix = integrate_potential(basis, &v_tot, grid);
    Ok(LocalKsBuild {
        v_matrix,
        e_hartree,
        e_xc,
    })
}

pub struct GpwLocalEnergy {
    pub e_kin: f64,
    pub e_hartree: f64,
    pub e_xc: f64,
    pub n_electrons: f64,
}

pub fn local_energy(
    basis: &Basis,
    p: &[f64],
    grid: &RealSpaceGrid,
    xc: &GridXc,
) -> Result<GpwLocalEnergy, PeriodicError> {
    let n_r = collocate_density(basis, p, grid);
    let n_electrons = n_r.iter().sum::<f64>() * grid.dv();
    let (_v_h, e_hartree) = hartree(&n_r, grid);
    let (e_xc, _v_xc) = xc.energy_potential(&n_r, grid.dv())?;
    let e_kin = kinetic_energy(basis, p);
    Ok(GpwLocalEnergy {
        e_kin,
        e_hartree,
        e_xc,
        n_electrons,
    })
}
