use crate::periodic::PeriodicError;
use xcx::{Functional, FunctionalId, Spin, XcInput};

const PADE_A0: f64 = 0.458_165_293_283_142_9;
const PADE_A1: f64 = 2.217_058_676_663_745;
const PADE_A2: f64 = 0.740_555_173_535_705_3;
const PADE_A3: f64 = 0.019_682_278_786_179_98;
const PADE_B1: f64 = 1.0;
const PADE_B2: f64 = 4.504_130_959_426_697;
const PADE_B3: f64 = 1.110_667_363_742_916;
const PADE_B4: f64 = 0.023_592_917_514_275_06;
const PADE_RSFAC: f64 = 0.620_350_490_899_4;
const PADE_RHO_FLOOR: f64 = 1e-20;

fn pade_eps_vxc(rho: f64) -> (f64, f64) {
    if rho <= PADE_RHO_FLOOR {
        return (0.0, 0.0);
    }
    let rs = PADE_RSFAC * rho.powf(-1.0 / 3.0);
    let p = PADE_A0 + (PADE_A1 + (PADE_A2 + PADE_A3 * rs) * rs) * rs;
    let q = (PADE_B1 + (PADE_B2 + (PADE_B3 + PADE_B4 * rs) * rs) * rs) * rs;
    let eps = -p / q;
    let dp = PADE_A1 + (2.0 * PADE_A2 + 3.0 * PADE_A3 * rs) * rs;
    let dq = PADE_B1 + (2.0 * PADE_B2 + (3.0 * PADE_B3 + 4.0 * PADE_B4 * rs) * rs) * rs;
    let depade = (1.0 / 3.0) * rs * (dp * q - p * dq) / (q * q);
    (eps, eps + depade)
}

pub struct GridXc {
    kind: XcKind,
}

enum XcKind {
    Xcx(Vec<Functional>),
    Pade,
}

impl GridXc {
    pub fn new(ids: &[FunctionalId]) -> Result<Self, PeriodicError> {
        let funcs = ids
            .iter()
            .map(|&id| Functional::new(id, Spin::Unpolarized))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            kind: XcKind::Xcx(funcs),
        })
    }

    pub fn slater_exchange() -> Result<Self, PeriodicError> {
        Self::new(&[FunctionalId::LdaX])
    }

    pub fn lda() -> Result<Self, PeriodicError> {
        Self::new(&[FunctionalId::LdaX, FunctionalId::LdaCPw])
    }

    #[must_use]
    pub fn pade() -> Self {
        Self { kind: XcKind::Pade }
    }

    pub fn energy_potential(&self, n_r: &[f64], dv: f64) -> Result<(f64, Vec<f64>), PeriodicError> {
        let np = n_r.len();
        match &self.kind {
            XcKind::Xcx(funcs) => {
                let mut eps_sum = vec![0.0; np]; // Σ ε_xc per particle
                let mut vxc = vec![0.0; np]; // Σ vrho = V_xc(r)
                for f in funcs {
                    let res = f.eval(np, &XcInput::lda(n_r))?;
                    for (es, &e) in eps_sum.iter_mut().zip(&res.exc) {
                        *es += e;
                    }
                    for (vs, &v) in vxc.iter_mut().zip(&res.vrho) {
                        *vs += v;
                    }
                }
                let e_xc = dv * n_r.iter().zip(&eps_sum).map(|(&n, &e)| n * e).sum::<f64>();
                Ok((e_xc, vxc))
            }
            XcKind::Pade => {
                let mut vxc = vec![0.0; np];
                let mut e_xc = 0.0;
                for (g, &n) in n_r.iter().enumerate() {
                    let (eps, v) = pade_eps_vxc(n);
                    e_xc += n * eps;
                    vxc[g] = v;
                }
                Ok((dv * e_xc, vxc))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pade_energy_density_sanity_values() {
        for (rs, eps_ref, v_ref) in [
            (1.0, -0.517_514_153_310_863_1, -0.677_964_586_410_288_1),
            (2.0, -0.273_638_647_292_287_7, -0.356_560_280_530_856_1),
            (3.0, -0.189_640_449_859_685_7, -0.246_549_099_561_781_6),
        ] {
            let n = (PADE_RSFAC / rs).powi(3);
            let (eps, v) = pade_eps_vxc(n);
            assert!(
                (eps - eps_ref).abs() < 1e-12,
                "rs={rs}: ε {eps} vs {eps_ref}"
            );
            assert!((v - v_ref).abs() < 1e-12, "rs={rs}: v {v} vs {v_ref}");
        }
    }
}
