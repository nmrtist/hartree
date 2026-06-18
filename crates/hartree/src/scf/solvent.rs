#[derive(Debug, Clone)]
pub struct SolventContribution {
    pub e_solv: f64,
    pub v_solv: Vec<f64>,
}

pub trait SolventModel {
    fn eval(&self, d_total: &[f64], n: usize) -> SolventContribution;
}
