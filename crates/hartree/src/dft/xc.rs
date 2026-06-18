use crate::basis::{AoBasis, ShellData};
use crate::core::Molecule;
use crate::scf::{RangeSeparation, XcContribution, XcContributor};
use xcx::{CamParams, Functional, Spin, Vv10Params, XcInput, XcResult};

use crate::dft::ao::{self, AoBatch};
use crate::dft::density::{
    batch_density_tau, pack_rho_polarized, pack_sigma_polarized, pack_tau_polarized, sigma_dot,
};
use crate::dft::error::Result;
use crate::dft::functional::FunctionalSpec;
use crate::dft::grid::MolecularGrid;

pub struct GridXc {
    grid: MolecularGrid,
    shells: Vec<ShellData>,
    nao: usize,
    pub(crate) partition: crate::dft::grid::BeckePartition,
    pub(crate) ao_atom: Vec<usize>,
    pub(crate) natom: usize,
    level: usize,
    name: &'static str,
    func_unpol: Functional,
    func_pol: Functional,
    needs_sigma: bool,
    needs_tau: bool,
    exx_fraction: f64,
    cam: Option<CamParams>,
    vv10: Option<Vv10Params>,
}

#[derive(Clone)]
struct Acc {
    exc: f64,
    nelec: f64,
    v_a: Vec<f64>,
    v_b: Vec<f64>,
}

impl GridXc {
    pub fn new(mol: &Molecule, ao: &AoBasis, spec: &FunctionalSpec, level: usize) -> Result<Self> {
        let grid = MolecularGrid::build(mol, level)?;
        let shells = ao.shells().to_vec();
        ao::ensure_supported(&shells)?;
        let ao_atom = crate::dft::gradient::ao_atom_map(&shells, mol)?;
        Ok(Self {
            grid,
            shells,
            nao: ao.n_ao(),
            partition: crate::dft::grid::BeckePartition::new(mol),
            ao_atom,
            natom: mol.atoms.len(),
            level,
            name: spec.name(),
            func_unpol: spec.build(Spin::Unpolarized)?,
            func_pol: spec.build(Spin::Polarized)?,
            needs_sigma: spec.needs_sigma(),
            needs_tau: spec.needs_tau(),
            exx_fraction: spec.exx_fraction(),
            cam: spec.cam(),
            vv10: spec.vv10(),
        })
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn n_points(&self) -> usize {
        self.grid.len()
    }

    pub fn grid(&self) -> &MolecularGrid {
        &self.grid
    }

    pub fn needs_sigma(&self) -> bool {
        self.needs_sigma
    }

    pub fn needs_tau(&self) -> bool {
        self.needs_tau
    }

    pub fn nao(&self) -> usize {
        self.nao
    }

    pub(crate) fn shells(&self) -> &[ShellData] {
        &self.shells
    }

    pub(crate) fn func_pol(&self) -> &Functional {
        &self.func_pol
    }

    pub fn vv10_params(&self) -> Option<Vv10Params> {
        self.vv10
    }

    pub fn cam_params(&self) -> Option<CamParams> {
        self.cam
    }

    pub fn vv10_energy(
        &self,
        mol: &Molecule,
        d_tot: &[f64],
    ) -> Option<crate::dft::error::Result<f64>> {
        let p = self.vv10?;
        Some(crate::dft::vv10::vv10_energy(
            mol,
            &self.shells,
            self.nao,
            d_tot,
            p.b,
            p.c,
        ))
    }

    pub fn energy(&self, d_alpha: &[f64], d_beta: &[f64], restricted: bool) -> (f64, f64) {
        let acc = if restricted {
            let d_tot = add(d_alpha, d_beta);
            self.fold_restricted(&d_tot, false)
        } else {
            self.fold_polarized(d_alpha, d_beta, false)
        };
        (acc.exc, acc.nelec)
    }

    fn fold_restricted(&self, d_tot: &[f64], want_potential: bool) -> Acc {
        let nao = self.nao;
        let weights = &self.grid.weights;
        ao::par_blocks_fold(
            &self.shells,
            nao,
            &self.grid.points,
            self.needs_sigma || self.needs_tau,
            || Acc::new(nao, want_potential, false),
            |mut acc, batch, start| {
                let np = batch.npts;
                let w = &weights[start..start + np];
                let bd = batch_density_tau(batch, d_tot, self.needs_sigma, self.needs_tau);

                let res = self.eval_xc_unpol(np, &bd.rho, bd.grad.as_slice(), &bd.tau);

                for ((&wp, &n_tot), &ex) in w.iter().zip(&bd.rho).zip(&res.exc) {
                    acc.exc += wp * n_tot * ex;
                    acc.nelec += wp * n_tot;
                }

                if want_potential {
                    let a = self.build_a_restricted(batch, w, &bd.grad, &res);
                    accumulate_v(&mut acc.v_a, &batch.phi, &a, np, nao);
                    if self.needs_tau {
                        accumulate_v_tau(&mut acc.v_a, batch, w, &res.vtau, 1, 0, nao);
                    }
                }
                acc
            },
            Acc::reduce,
        )
        .expect("shells validated in GridXc::new")
    }

    fn fold_polarized(&self, d_a: &[f64], d_b: &[f64], want_potential: bool) -> Acc {
        let nao = self.nao;
        let weights = &self.grid.weights;
        ao::par_blocks_fold(
            &self.shells,
            nao,
            &self.grid.points,
            self.needs_sigma || self.needs_tau,
            || Acc::new(nao, want_potential, true),
            |mut acc, batch, start| {
                let np = batch.npts;
                let w = &weights[start..start + np];
                let bd_a = batch_density_tau(batch, d_a, self.needs_sigma, self.needs_tau);
                let bd_b = batch_density_tau(batch, d_b, self.needs_sigma, self.needs_tau);

                let rho = pack_rho_polarized(&bd_a.rho, &bd_b.rho);
                let res = if self.needs_sigma {
                    let sigma = pack_sigma_polarized(&bd_a.grad, &bd_b.grad);
                    let input = XcInput::gga(&rho, &sigma);
                    if self.needs_tau {
                        let tau = pack_tau_polarized(&bd_a.tau, &bd_b.tau);
                        self.func_pol
                            .eval(np, &input.with_tau(&tau))
                            .expect("xcx polarized meta-GGA eval")
                    } else {
                        self.func_pol
                            .eval(np, &input)
                            .expect("xcx polarized GGA eval")
                    }
                } else {
                    self.func_pol
                        .eval(np, &XcInput::lda(&rho))
                        .expect("xcx polarized LDA eval")
                };

                for ((&wp, (&ra, &rb)), &ex) in
                    w.iter().zip(bd_a.rho.iter().zip(&bd_b.rho)).zip(&res.exc)
                {
                    let n_tot = ra + rb;
                    acc.exc += wp * n_tot * ex;
                    acc.nelec += wp * n_tot;
                }

                if want_potential {
                    let a_a =
                        self.build_a_polarized(batch, w, &bd_a.grad, &bd_b.grad, &res, Channel::A);
                    let a_b =
                        self.build_a_polarized(batch, w, &bd_a.grad, &bd_b.grad, &res, Channel::B);
                    accumulate_v(&mut acc.v_a, &batch.phi, &a_a, np, nao);
                    accumulate_v(&mut acc.v_b, &batch.phi, &a_b, np, nao);
                    if self.needs_tau {
                        accumulate_v_tau(&mut acc.v_a, batch, w, &res.vtau, 2, 0, nao);
                        accumulate_v_tau(&mut acc.v_b, batch, w, &res.vtau, 2, 1, nao);
                    }
                }
                acc
            },
            Acc::reduce,
        )
        .expect("shells validated in GridXc::new")
    }

    pub(crate) fn eval_xc_unpol(
        &self,
        np: usize,
        rho: &[f64],
        grad: &[[f64; 3]],
        tau: &[f64],
    ) -> XcResult {
        if self.needs_sigma {
            let sigma = sigma_dot(grad, grad);
            let input = XcInput::gga(rho, &sigma);
            if self.needs_tau {
                self.func_unpol
                    .eval(np, &input.with_tau(tau))
                    .expect("xcx unpolarized meta-GGA eval")
            } else {
                self.func_unpol
                    .eval(np, &input)
                    .expect("xcx unpolarized GGA eval")
            }
        } else {
            self.func_unpol
                .eval(np, &XcInput::lda(rho))
                .expect("xcx unpolarized LDA eval")
        }
    }

    fn build_a_restricted(
        &self,
        batch: &AoBatch,
        w: &[f64],
        grad: &[[f64; 3]],
        res: &XcResult,
    ) -> Vec<f64> {
        let nao = self.nao;
        let np = batch.npts;
        let mut a = vec![0.0; np * nao];
        for p in 0..np {
            let half_vrho = 0.5 * w[p] * res.vrho[p];
            let phi = &batch.phi[p * nao..p * nao + nao];
            let arow = &mut a[p * nao..p * nao + nao];
            if self.needs_sigma {
                let coeff = 2.0 * w[p] * res.vsigma[p];
                let g = grad[p];
                let (dx, dy, dz) = batch.dphi_rows(p);
                for mu in 0..nao {
                    let grad_phi = g[0] * dx[mu] + g[1] * dy[mu] + g[2] * dz[mu];
                    arow[mu] = half_vrho * phi[mu] + coeff * grad_phi;
                }
            } else {
                for mu in 0..nao {
                    arow[mu] = half_vrho * phi[mu];
                }
            }
        }
        a
    }

    fn build_a_polarized(
        &self,
        batch: &AoBatch,
        w: &[f64],
        grad_a: &[[f64; 3]],
        grad_b: &[[f64; 3]],
        res: &XcResult,
        chan: Channel,
    ) -> Vec<f64> {
        let nao = self.nao;
        let np = batch.npts;
        let mut a = vec![0.0; np * nao];
        for p in 0..np {
            let vrho = match chan {
                Channel::A => res.vrho[2 * p],
                Channel::B => res.vrho[2 * p + 1],
            };
            let half_vrho = 0.5 * w[p] * vrho;
            let phi = &batch.phi[p * nao..p * nao + nao];
            let arow = &mut a[p * nao..p * nao + nao];

            if self.needs_sigma {
                let (s_same, s_cross) = match chan {
                    Channel::A => (res.vsigma[3 * p], res.vsigma[3 * p + 1]),
                    Channel::B => (res.vsigma[3 * p + 2], res.vsigma[3 * p + 1]),
                };
                let (g_same, g_other) = match chan {
                    Channel::A => (grad_a[p], grad_b[p]),
                    Channel::B => (grad_b[p], grad_a[p]),
                };
                let gvec = [
                    2.0 * s_same * g_same[0] + s_cross * g_other[0],
                    2.0 * s_same * g_same[1] + s_cross * g_other[1],
                    2.0 * s_same * g_same[2] + s_cross * g_other[2],
                ];
                let (dx, dy, dz) = batch.dphi_rows(p);
                for mu in 0..nao {
                    let grad_phi = gvec[0] * dx[mu] + gvec[1] * dy[mu] + gvec[2] * dz[mu];
                    arow[mu] = half_vrho * phi[mu] + w[p] * grad_phi;
                }
            } else {
                for mu in 0..nao {
                    arow[mu] = half_vrho * phi[mu];
                }
            }
        }
        a
    }
}

#[derive(Clone, Copy)]
enum Channel {
    A,
    B,
}

impl Acc {
    fn new(nao: usize, want_potential: bool, polarized: bool) -> Self {
        let v_a = if want_potential {
            vec![0.0; nao * nao]
        } else {
            Vec::new()
        };
        let v_b = if want_potential && polarized {
            vec![0.0; nao * nao]
        } else {
            Vec::new()
        };
        Acc {
            exc: 0.0,
            nelec: 0.0,
            v_a,
            v_b,
        }
    }

    fn reduce(mut a: Acc, b: Acc) -> Acc {
        a.exc += b.exc;
        a.nelec += b.nelec;
        for (x, y) in a.v_a.iter_mut().zip(&b.v_a) {
            *x += y;
        }
        for (x, y) in a.v_b.iter_mut().zip(&b.v_b) {
            *x += y;
        }
        a
    }
}

impl AoBatch {
    fn dphi_rows(&self, p: usize) -> (&[f64], &[f64], &[f64]) {
        let lo = p * self.nao;
        let hi = lo + self.nao;
        (
            &self.dphi[0][lo..hi],
            &self.dphi[1][lo..hi],
            &self.dphi[2][lo..hi],
        )
    }
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn transpose_rowmajor(src: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = src[i * cols + j];
        }
    }
    out
}

fn accumulate_v(v: &mut [f64], phi: &[f64], a: &[f64], np: usize, nao: usize) {
    let phit = transpose_rowmajor(phi, np, nao);
    let g = crate::linalg::gemm(&phit, nao, np, a, nao);
    for mu in 0..nao {
        for nu in 0..nao {
            v[mu * nao + nu] += g[mu * nao + nu] + g[nu * nao + mu];
        }
    }
}

fn accumulate_v_tau(
    v: &mut [f64],
    batch: &AoBatch,
    w: &[f64],
    vtau: &[f64],
    stride: usize,
    offset: usize,
    nao: usize,
) {
    let np = batch.npts;
    let mut a = vec![0.0; np * nao];
    for dk in &batch.dphi {
        for p in 0..np {
            let c = 0.25 * w[p] * vtau[stride * p + offset];
            let drow = &dk[p * nao..p * nao + nao];
            let arow = &mut a[p * nao..p * nao + nao];
            for mu in 0..nao {
                arow[mu] = c * drow[mu];
            }
        }
        accumulate_v(v, dk, &a, np, nao);
    }
}

impl XcContributor for GridXc {
    fn exx_fraction(&self) -> f64 {
        self.exx_fraction
    }

    fn range_separation(&self) -> Option<RangeSeparation> {
        self.cam.map(|c| RangeSeparation {
            omega: c.omega,
            alpha: c.alpha,
            beta: c.beta,
        })
    }

    fn eval(&self, d_alpha: &[f64], d_beta: &[f64], n: usize, restricted: bool) -> XcContribution {
        debug_assert_eq!(n, self.nao, "density dimension must match the AO basis");
        if restricted {
            let d_tot = add(d_alpha, d_beta);
            let acc = self.fold_restricted(&d_tot, true);
            let v = acc.v_a;
            XcContribution {
                exc: acc.exc,
                vxc_beta: v.clone(),
                vxc_alpha: v,
                n_elec_grid: acc.nelec,
            }
        } else {
            let acc = self.fold_polarized(d_alpha, d_beta, true);
            XcContribution {
                exc: acc.exc,
                vxc_alpha: acc.v_a,
                vxc_beta: acc.v_b,
                n_elec_grid: acc.nelec,
            }
        }
    }

    fn gradient(&self, d_alpha: &[f64], d_beta: &[f64], restricted: bool) -> Option<Vec<[f64; 3]>> {
        self.xc_gradient(d_alpha, d_beta, restricted).ok()
    }
}
