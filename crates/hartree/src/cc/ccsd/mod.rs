mod diis;
pub mod spin_adapted;
pub mod spin_orbital;
pub mod triples;

pub use spin_adapted::rccsd_spin_adapted;
pub use spin_orbital::rccsd_spin_orbital;
pub use triples::rccsd_t_spin_adapted;

#[derive(Debug, Clone, Copy)]
pub struct CcsdOptions {
    pub max_iter: usize,
    pub energy_tol: f64,
    pub amplitude_tol: f64,
    pub diis_dim: usize,
}

impl Default for CcsdOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            energy_tol: 1e-11,
            amplitude_tol: 1e-9,
            diis_dim: 8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CcsdTResult {
    pub ccsd: CcsdResult,
    pub triples_energy: f64,
    pub total_energy: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct CcsdResult {
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub scf_energy: f64,
    pub mp2_correlation: f64,
    pub converged: bool,
    pub iterations: usize,
    pub n_frozen: usize,
    pub t1_diagnostic: f64,
}

pub(crate) fn t1_diagnostic_from(t1: &[f64], n_occ_rows: usize) -> f64 {
    if n_occ_rows == 0 {
        return 0.0;
    }
    let norm_sq: f64 = t1.iter().map(|x| x * x).sum();
    (norm_sq / (2.0 * n_occ_rows as f64)).sqrt()
}
