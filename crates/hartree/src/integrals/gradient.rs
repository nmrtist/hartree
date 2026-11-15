//! Integral-derivative terms shared by the in-core and integral-direct providers.

use super::{
    ConventionalProvider, DEFAULT_SCREENING_TAU, DirectProvider, GradientError, GradientProvider,
    MatrixGradient,
};
use rayon::prelude::*;
use std::sync::OnceLock;

/// Built on first use: most providers never compute a gradient.
pub(super) type GradContractorCell = OnceLock<Result<integral::DirectGradContractor, String>>;

fn backend_error(e: integral::IntegralError) -> GradientError {
    GradientError::Backend(e.to_string())
}

fn matrix_gradient(
    g: Result<integral::Gradient1e, integral::IntegralError>,
) -> Result<MatrixGradient, GradientError> {
    let g = g.map_err(backend_error)?;
    let natom = g.natom();
    let n = g.nao();
    let nn = n * n;
    let mut data = vec![0.0; natom * 3 * nn];
    for atom in 0..natom {
        for axis in 0..3 {
            let off = (atom * 3 + axis) * nn;
            data[off..off + nn].copy_from_slice(g.block(atom, axis));
        }
    }
    Ok(MatrixGradient { natom, n, data })
}

/// The exchange-type part of the two-particle density of a pair of spin densities,
/// `−½·exchange·(Dα[μλ]Dα[νσ] + Dβ[μλ]Dβ[νσ])`, plus, when `coulomb` is set, the
/// Coulomb-type part `½·P[μν]P[λσ]` with `P = Dα + Dβ`.
#[derive(Clone, Copy)]
struct SpinGamma {
    coulomb: bool,
    exchange: f64,
}

fn contract_eri_gradient(
    cell: &GradContractorCell,
    basis: &integral::Basis,
    density_alpha: &[f64],
    density_beta: &[f64],
    gamma: SpinGamma,
    kernel: integral::EriKernel,
    tau: f64,
) -> Result<Vec<[f64; 3]>, GradientError> {
    let contractor = cell
        .get_or_init(|| basis.direct_grad_contractor().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| GradientError::Backend(e.clone()))?;
    let nn = contractor.nao() * contractor.nao();
    assert_eq!(density_alpha.len(), nn, "density_alpha must be nao²");
    assert_eq!(density_beta.len(), nn, "density_beta must be nao²");

    let total: Vec<f64> = density_alpha
        .iter()
        .zip(density_beta)
        .map(|(a, b)| a + b)
        .collect();
    let coulomb = [integral::GradDensityTerm {
        factor: 0.5,
        left: &total,
        right: &total,
    }];
    let exchange = [density_alpha, density_beta].map(|d| integral::GradDensityTerm {
        factor: -0.5 * gamma.exchange,
        left: d,
        right: d,
    });
    let density = contractor.density(
        if gamma.coulomb { &coulomb } else { &[] },
        if gamma.exchange != 0.0 {
            &exchange
        } else {
            &[]
        },
    );

    let zero = || {
        (
            integral::DirectGradWorkspace::new(),
            vec![[0.0; 3]; contractor.natom()],
        )
    };
    let (_, grad) = contractor
        .bra_pairs()
        .par_iter()
        .fold(zero, |(mut workspace, mut grad), &bra| {
            contractor.accumulate_bra_pair(&mut workspace, bra, &density, kernel, tau, &mut grad);
            (workspace, grad)
        })
        .reduce(zero, |(workspace, mut a), (_, b)| {
            for (ga, gb) in a.iter_mut().zip(&b) {
                for axis in 0..3 {
                    ga[axis] += gb[axis];
                }
            }
            (workspace, a)
        });
    Ok(grad)
}

macro_rules! impl_gradient_provider {
    ($provider:ty, $tau:expr) => {
        impl GradientProvider for $provider {
            fn overlap_gradient(&self) -> Result<MatrixGradient, GradientError> {
                matrix_gradient(self.basis.overlap_grad())
            }

            fn kinetic_gradient(&self) -> Result<MatrixGradient, GradientError> {
                matrix_gradient(self.basis.kinetic_grad())
            }

            fn nuclear_gradient(&self) -> Result<MatrixGradient, GradientError> {
                matrix_gradient(self.basis.nuclear_grad(&self.charges))
            }

            fn eri_gradient(
                &self,
                density_alpha: &[f64],
                density_beta: &[f64],
                c_x: f64,
            ) -> Result<Vec<[f64; 3]>, GradientError> {
                let tau: fn(&$provider) -> f64 = $tau;
                contract_eri_gradient(
                    &self.grad_contractor,
                    &self.basis,
                    density_alpha,
                    density_beta,
                    SpinGamma {
                        coulomb: true,
                        exchange: c_x,
                    },
                    integral::EriKernel::Coulomb,
                    tau(self),
                )
            }

            fn eri_gradient_erf(
                &self,
                density_alpha: &[f64],
                density_beta: &[f64],
                c_x: f64,
                omega: f64,
            ) -> Result<Option<Vec<[f64; 3]>>, GradientError> {
                if !(omega.is_finite() && omega > 0.0) {
                    return Err(GradientError::Backend(format!(
                        "range-separation parameter ω must be finite and positive, got {omega}"
                    )));
                }
                let tau: fn(&$provider) -> f64 = $tau;
                contract_eri_gradient(
                    &self.grad_contractor,
                    &self.basis,
                    density_alpha,
                    density_beta,
                    SpinGamma {
                        coulomb: false,
                        exchange: c_x,
                    },
                    integral::EriKernel::Erf { omega },
                    tau(self),
                )
                .map(Some)
            }

            fn ecp_gradient_contract(
                &self,
                density: &[f64],
            ) -> Result<Option<Vec<[f64; 3]>>, GradientError> {
                if self.ecps.is_empty() {
                    return Ok(None);
                }
                self.basis
                    .ecp_grad_contract(&self.ecps, density)
                    .map(Some)
                    .map_err(backend_error)
            }
        }
    };
}

impl_gradient_provider!(ConventionalProvider, |_| DEFAULT_SCREENING_TAU);
impl_gradient_provider!(DirectProvider, |p| p.tau);
