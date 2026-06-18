#[derive(Debug, Clone)]
pub struct XcContribution {
    pub exc: f64,
    pub vxc_alpha: Vec<f64>,
    pub vxc_beta: Vec<f64>,
    pub n_elec_grid: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeSeparation {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
}

pub trait XcContributor {
    fn exx_fraction(&self) -> f64;

    fn range_separation(&self) -> Option<RangeSeparation> {
        None
    }

    fn eval(&self, d_alpha: &[f64], d_beta: &[f64], n: usize, restricted: bool) -> XcContribution;

    fn gradient(
        &self,
        _d_alpha: &[f64],
        _d_beta: &[f64],
        _restricted: bool,
    ) -> Option<Vec<[f64; 3]>> {
        None
    }
}
