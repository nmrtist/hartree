//! The integral-provider seam over the integral engine, for molecular and periodic integrals.

pub use integral;

use crate::linalg::{
    Mat, cholesky_lower, gemm, mat_from_row_major, mat_to_row_major,
    solve_lower_triangular_cols_in_place,
};
use rayon::prelude::*;
use thiserror::Error;

mod gradient;

pub struct JkResult {
    pub coulomb: Vec<Mat>,
    pub exchange: Vec<Mat>,
}

pub trait IntegralProvider {
    fn n_basis(&self) -> usize;

    fn overlap(&self) -> Mat;

    fn kinetic(&self) -> Mat;

    fn nuclear(&self) -> Mat;

    fn core_hamiltonian(&self) -> Mat {
        self.kinetic() + self.nuclear()
    }

    fn build_jk(&self, densities: &[Mat]) -> JkResult;

    /// Coulomb-only build. Pure (semi-)local functionals carry no exact exchange, so the
    /// SCF never needs K — only J enters their Fock matrix. The default falls back to the
    /// full J/K build and discards K; backends whose K build is the dominant cost (the
    /// density-fitted and integral-direct ones) override this to skip K entirely.
    fn build_j(&self, densities: &[Mat]) -> Vec<Mat> {
        self.build_jk(densities).coulomb
    }

    /// Screening-aware Coulomb-only build, the J counterpart of [`build_jk_screened`]
    /// used by the incremental-Fock path. Same contract as [`build_j`].
    fn build_j_screened(&self, densities: &[Mat]) -> Vec<Mat> {
        self.build_jk_screened(densities).coulomb
    }

    fn dipole_integrals(&self, origin: [f64; 3]) -> [Vec<f64>; 3];

    fn ao_atom_indices(&self) -> Vec<usize>;

    fn effective_nuclear_charges(&self) -> Option<Vec<f64>> {
        None
    }

    fn build_jk_screened(&self, densities: &[Mat]) -> JkResult {
        self.build_jk(densities)
    }

    fn build_k_erf(&self, _densities: &[Mat], _omega: f64) -> Option<Vec<Mat>> {
        None
    }

    fn grid_coulomb(&self, _points: &[[f64; 3]]) -> Option<Vec<f64>> {
        None
    }

    fn grid_coulomb_erf(&self, _points: &[[f64; 3]], _omega: f64) -> Option<Vec<f64>> {
        None
    }

    fn charge_potential_3c(&self, charges: &[([f64; 3], f64)]) -> Vec<f64>;
}

pub trait InCoreEri {
    fn ao_eri(&self) -> &[f64];
}

impl InCoreEri for ConventionalProvider {
    fn ao_eri(&self) -> &[f64] {
        self.eri_tensor()
    }
}

#[derive(Debug, Error)]
pub enum GradientError {
    #[error("integral-derivative backend error: {0}")]
    Backend(String),
}

#[derive(Debug, Clone)]
pub struct MatrixGradient {
    natom: usize,
    n: usize,
    data: Vec<f64>,
}

impl MatrixGradient {
    pub fn natom(&self) -> usize {
        self.natom
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn block(&self, atom: usize, axis: usize) -> &[f64] {
        let nn = self.n * self.n;
        let off = (atom * 3 + axis) * nn;
        &self.data[off..off + nn]
    }
}

pub trait GradientProvider {
    fn overlap_gradient(&self) -> Result<MatrixGradient, GradientError>;

    fn kinetic_gradient(&self) -> Result<MatrixGradient, GradientError>;

    fn nuclear_gradient(&self) -> Result<MatrixGradient, GradientError>;

    /// `Σ Γ[μνλσ]·∂(μν|λσ)/∂R` per atom for the two-particle density of the spin
    /// densities `Dα`, `Dβ` (row-major `nao²`, symmetric):
    /// `Γ = ½·P[μν]P[λσ] − ½·c_x·(Dα[μλ]Dα[νσ] + Dβ[μλ]Dβ[νσ])`, `P = Dα + Dβ`.
    fn eri_gradient(
        &self,
        density_alpha: &[f64],
        density_beta: &[f64],
        c_x: f64,
    ) -> Result<Vec<[f64; 3]>, GradientError>;

    /// The exchange part of [`Self::eri_gradient`] alone, over the `erf(ω·r₁₂)/r₁₂`
    /// kernel. `None` when the backend has no erf-attenuated ERI derivative.
    fn eri_gradient_erf(
        &self,
        _density_alpha: &[f64],
        _density_beta: &[f64],
        _c_x: f64,
        _omega: f64,
    ) -> Result<Option<Vec<[f64; 3]>>, GradientError> {
        Ok(None)
    }

    fn ecp_gradient_contract(
        &self,
        _density: &[f64],
    ) -> Result<Option<Vec<[f64; 3]>>, GradientError> {
        Ok(None)
    }
}

pub struct ConventionalProvider {
    basis: integral::Basis,
    charges: Vec<([f64; 3], f64)>,
    ecps: Vec<integral::Ecp>,
    n_basis: usize,
    eri: std::sync::OnceLock<Vec<f64>>,
    eri_lr: std::sync::OnceLock<(f64, Vec<f64>)>,
    grad_contractor: gradient::GradContractorCell,
}

impl ConventionalProvider {
    pub fn new(basis: integral::Basis, charges: Vec<([f64; 3], f64)>) -> Self {
        let n_basis = basis.nao();
        Self {
            basis,
            charges,
            ecps: Vec::new(),
            n_basis,
            eri: std::sync::OnceLock::new(),
            eri_lr: std::sync::OnceLock::new(),
            grad_contractor: std::sync::OnceLock::new(),
        }
    }

    #[must_use]
    pub fn with_ecps(mut self, ecps: Vec<integral::Ecp>) -> Self {
        self.ecps = ecps;
        self
    }

    fn eri_tensor(&self) -> &[f64] {
        self.eri.get_or_init(|| {
            if rayon::current_num_threads() == 1 {
                self.basis.eri()
            } else {
                build_eri_parallel(&self.basis)
            }
        })
    }

    fn eri_lr_tensor(&self, omega: f64) -> &[f64] {
        let (cached_omega, tensor) = self.eri_lr.get_or_init(|| {
            (
                omega,
                self.basis.eri_kernel(integral::EriKernel::Erf { omega }),
            )
        });
        assert_eq!(
            *cached_omega, omega,
            "ConventionalProvider caches one range-separation ω per lifetime \
             (cached {cached_omega}, requested {omega})"
        );
        tensor
    }

    fn k_from_tensor(eri: &[f64], d: &[f64], n: usize) -> Vec<f64> {
        let mut k = vec![0.0; n * n];
        k.par_chunks_mut(n).enumerate().for_each(|(mu, k_row)| {
            for (nu, k_slot) in k_row.iter_mut().enumerate() {
                let mut k_sum = 0.0;
                for lambda in 0..n {
                    let exchange_base = ((mu * n + lambda) * n + nu) * n;
                    let d_row = lambda * n;
                    for sigma in 0..n {
                        k_sum += eri[exchange_base + sigma] * d[d_row + sigma];
                    }
                }
                *k_slot = k_sum;
            }
        });
        k
    }
}

fn build_eri_parallel(basis: &integral::Basis) -> Vec<f64> {
    let builder = basis.eri_builder();
    let mut out = vec![0.0; builder.output_len()];
    {
        let mut tasks = builder.partition(&mut out);

        let shells = basis.shells();
        tasks.sort_unstable_by(|a, b| {
            bra_pair_cost(shells, b.bra()).cmp(&bra_pair_cost(shells, a.bra()))
        });

        tasks.par_iter_mut().for_each(|task| builder.fill(task));
    } // tasks (and their &mut borrows of `out`) drop here, releasing `out` for return.
    out
}

fn bra_pair_cost(shells: &[integral::Shell], (i, j): (usize, usize)) -> u64 {
    fn weight(s: &integral::Shell) -> u64 {
        let l = s.l() as u64;
        let n_cart = (l + 1) * (l + 2) / 2;
        s.n_prim() as u64 * n_cart * (l + 1)
    }
    weight(&shells[i]) * weight(&shells[j])
}

fn charge_potential_3c_impl(basis: &integral::Basis, charges: &[([f64; 3], f64)]) -> Vec<f64> {
    let aux_shells: Vec<integral::Shell> = charges
        .iter()
        .map(|&(center, zeta)| {
            let a = zeta * zeta;
            let pi = std::f64::consts::PI;
            let coeff = (a / pi).powf(1.5) / (2.0 * a / pi).powf(0.75);
            integral::Shell::new(0, center, vec![a], vec![coeff])
                .expect("unit-charge s shell is always valid")
        })
        .collect();
    let aux = integral::Basis::new(aux_shells);

    let builder = basis.eri_3c_builder(&aux);
    let mut out = vec![0.0; builder.output_len()];
    {
        let mut tasks = builder.partition(&mut out);
        let shells = basis.shells();
        tasks.sort_unstable_by(|a, b| {
            bra_pair_cost(shells, b.bra()).cmp(&bra_pair_cost(shells, a.bra()))
        });
        tasks.par_iter_mut().for_each(|task| builder.fill(task));
    }
    out
}

const GRID_COULOMB_CHUNK: usize = 128;

fn grid_coulomb_impl(basis: &integral::Basis, points: &[[f64; 3]]) -> Vec<f64> {
    let mm = basis.nao() * basis.nao();
    let mut out = vec![0.0; points.len() * mm];
    if mm == 0 || points.is_empty() {
        return out;
    }
    out.par_chunks_mut(GRID_COULOMB_CHUNK * mm)
        .zip(points.par_chunks(GRID_COULOMB_CHUNK))
        .for_each(|(chunk_out, chunk_pts)| basis.grid_coulomb_into(chunk_pts, chunk_out));
    out
}

fn grid_coulomb_erf_impl(basis: &integral::Basis, points: &[[f64; 3]], omega: f64) -> Vec<f64> {
    let mm = basis.nao() * basis.nao();
    let mut out = vec![0.0; points.len() * mm];
    if mm == 0 || points.is_empty() {
        return out;
    }
    out.par_chunks_mut(GRID_COULOMB_CHUNK * mm)
        .zip(points.par_chunks(GRID_COULOMB_CHUNK))
        .for_each(|(chunk_out, chunk_pts)| {
            basis.grid_coulomb_kernel_into(chunk_pts, integral::EriKernel::Erf { omega }, chunk_out)
        });
    out
}

fn nuclear_with_ecp(
    basis: &integral::Basis,
    charges: &[([f64; 3], f64)],
    ecps: &[integral::Ecp],
) -> Vec<f64> {
    let mut v = basis.nuclear(charges);
    if !ecps.is_empty() {
        for (vi, wi) in v.iter_mut().zip(basis.ecp(ecps)) {
            *vi += wi;
        }
    }
    v
}

fn ao_atom_map(basis: &integral::Basis, charges: &[([f64; 3], f64)]) -> Vec<usize> {
    let mut out = Vec::new();
    for shell in basis.shells() {
        let center = shell.center();
        let idx = charges
            .iter()
            .position(|(c, _)| *c == center)
            .expect("shell center not found in charges — internal invariant violated");
        for _ in 0..shell.n_func() {
            out.push(idx);
        }
    }
    out
}

impl IntegralProvider for ConventionalProvider {
    fn n_basis(&self) -> usize {
        self.n_basis
    }

    fn overlap(&self) -> Mat {
        mat_from_row_major(self.n_basis, &self.basis.overlap())
    }

    fn kinetic(&self) -> Mat {
        mat_from_row_major(self.n_basis, &self.basis.kinetic())
    }

    fn nuclear(&self) -> Mat {
        mat_from_row_major(
            self.n_basis,
            &nuclear_with_ecp(&self.basis, &self.charges, &self.ecps),
        )
    }

    fn dipole_integrals(&self, origin: [f64; 3]) -> [Vec<f64>; 3] {
        self.basis.dipole(origin)
    }

    fn ao_atom_indices(&self) -> Vec<usize> {
        ao_atom_map(&self.basis, &self.charges)
    }

    fn effective_nuclear_charges(&self) -> Option<Vec<f64>> {
        Some(self.charges.iter().map(|&(_, q)| q).collect())
    }

    fn charge_potential_3c(&self, charges: &[([f64; 3], f64)]) -> Vec<f64> {
        charge_potential_3c_impl(&self.basis, charges)
    }

    fn build_jk(&self, densities: &[Mat]) -> JkResult {
        let n = self.n_basis;
        let eri = self.eri_tensor();
        let mut coulomb = Vec::with_capacity(densities.len());
        let mut exchange = Vec::with_capacity(densities.len());

        for density in densities {
            let d = mat_to_row_major(density);
            let mut j = vec![0.0; n * n];
            let mut k = vec![0.0; n * n];
            j.par_chunks_mut(n)
                .zip(k.par_chunks_mut(n))
                .enumerate()
                .for_each(|(mu, (j_row, k_row))| {
                    for (nu, (j_slot, k_slot)) in j_row.iter_mut().zip(k_row.iter_mut()).enumerate()
                    {
                        let mut j_sum = 0.0;
                        let mut k_sum = 0.0;
                        for lambda in 0..n {
                            let coulomb_base = ((mu * n + nu) * n + lambda) * n;
                            let exchange_base = ((mu * n + lambda) * n + nu) * n;
                            let d_row = lambda * n;
                            for sigma in 0..n {
                                let d_ls = d[d_row + sigma];
                                j_sum += eri[coulomb_base + sigma] * d_ls;
                                k_sum += eri[exchange_base + sigma] * d_ls;
                            }
                        }
                        *j_slot = j_sum;
                        *k_slot = k_sum;
                    }
                });
            coulomb.push(mat_from_row_major(n, &j));
            exchange.push(mat_from_row_major(n, &k));
        }

        JkResult { coulomb, exchange }
    }

    fn build_j(&self, densities: &[Mat]) -> Vec<Mat> {
        let n = self.n_basis;
        let eri = self.eri_tensor();
        densities
            .iter()
            .map(|density| {
                let d = mat_to_row_major(density);
                let mut j = vec![0.0; n * n];
                j.par_chunks_mut(n).enumerate().for_each(|(mu, j_row)| {
                    for (nu, j_slot) in j_row.iter_mut().enumerate() {
                        let mut j_sum = 0.0;
                        for lambda in 0..n {
                            let coulomb_base = ((mu * n + nu) * n + lambda) * n;
                            let d_row = lambda * n;
                            for sigma in 0..n {
                                j_sum += eri[coulomb_base + sigma] * d[d_row + sigma];
                            }
                        }
                        *j_slot = j_sum;
                    }
                });
                mat_from_row_major(n, &j)
            })
            .collect()
    }

    fn grid_coulomb(&self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        Some(grid_coulomb_impl(&self.basis, points))
    }

    fn grid_coulomb_erf(&self, points: &[[f64; 3]], omega: f64) -> Option<Vec<f64>> {
        Some(grid_coulomb_erf_impl(&self.basis, points, omega))
    }

    fn build_k_erf(&self, densities: &[Mat], omega: f64) -> Option<Vec<Mat>> {
        let n = self.n_basis;
        let eri = self.eri_lr_tensor(omega);
        Some(
            densities
                .iter()
                .map(|density| {
                    let d = mat_to_row_major(density);
                    mat_from_row_major(n, &Self::k_from_tensor(eri, &d, n))
                })
                .collect(),
        )
    }
}

pub const DEFAULT_SCREENING_TAU: f64 = 1e-12;

pub struct DirectProvider {
    basis: integral::Basis,
    charges: Vec<([f64; 3], f64)>,
    ecps: Vec<integral::Ecp>,
    n_basis: usize,
    contractor: integral::DirectContractor,
    grad_contractor: gradient::GradContractorCell,
    tau: f64,
}

impl DirectProvider {
    pub fn new(basis: integral::Basis, charges: Vec<([f64; 3], f64)>) -> Self {
        Self::with_screening(basis, charges, DEFAULT_SCREENING_TAU)
    }

    pub fn with_screening(basis: integral::Basis, charges: Vec<([f64; 3], f64)>, tau: f64) -> Self {
        let n_basis = basis.nao();
        let contractor = basis.direct_contractor();
        Self {
            basis,
            charges,
            ecps: Vec::new(),
            n_basis,
            contractor,
            grad_contractor: std::sync::OnceLock::new(),
            tau,
        }
    }

    #[must_use]
    pub fn with_ecps(mut self, ecps: Vec<integral::Ecp>) -> Self {
        self.ecps = ecps;
        self
    }

    pub fn screening_tau(&self) -> f64 {
        self.tau
    }

    fn contract_densities(
        &self,
        densities: &[Vec<f64>],
        want_k: bool,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.n_basis;
        let drefs: Vec<&[f64]> = densities.iter().map(Vec::as_slice).collect();
        let bounds = self.contractor.density_shell_bounds(&drefs);
        let brefs: Vec<&[f64]> = bounds.iter().map(Vec::as_slice).collect();
        let nd = densities.len();
        let zero = || {
            (
                integral::DirectWorkspace::new(),
                vec![vec![0.0; n * n]; nd],
                vec![vec![0.0; n * n]; nd],
            )
        };
        let (_, j, k) = self
            .contractor
            .bra_pairs()
            .par_iter()
            .fold(zero, |(mut workspace, mut j, mut k), &bra| {
                let mut outputs: Vec<integral::DirectBuffers<'_>> = j
                    .iter_mut()
                    .zip(k.iter_mut())
                    .map(|(coulomb, exchange)| integral::DirectBuffers {
                        coulomb,
                        exchange: want_k.then_some(exchange.as_mut_slice()),
                    })
                    .collect();
                self.contractor.accumulate_bra_pair(
                    &mut workspace,
                    bra,
                    &drefs,
                    &brefs,
                    self.tau,
                    want_k,
                    &mut outputs,
                );
                (workspace, j, k)
            })
            .reduce(zero, |(workspace, mut ja, mut ka), (_, jb, kb)| {
                for density in 0..nd {
                    for i in 0..ja[density].len() {
                        ja[density][i] += jb[density][i];
                        if want_k {
                            ka[density][i] += kb[density][i];
                        }
                    }
                }
                (workspace, ja, ka)
            });
        (j, k)
    }
}

#[derive(Debug, Error)]
pub enum DfError {
    #[error("density-fitting metric (P|Q) is not positive definite (near-singular auxiliary fit)")]
    MetricNotPositiveDefinite,
}

pub struct DfProvider {
    basis: integral::Basis,
    charges: Vec<([f64; 3], f64)>,
    ecps: Vec<integral::Ecp>,
    n_basis: usize,
    naux: usize,
    b: Vec<f64>,
}

#[inline]
fn pair_index(mu: usize, nu: usize) -> usize {
    debug_assert!(mu >= nu);
    mu * (mu + 1) / 2 + nu
}

impl DfProvider {
    pub fn new(
        basis: integral::Basis,
        aux: &integral::Basis,
        charges: Vec<([f64; 3], f64)>,
    ) -> Result<Self, DfError> {
        Self::with_screening(basis, aux, charges, DEFAULT_SCREENING_TAU)
    }

    pub fn with_screening(
        basis: integral::Basis,
        aux: &integral::Basis,
        charges: Vec<([f64; 3], f64)>,
        tau: f64,
    ) -> Result<Self, DfError> {
        let n_basis = basis.nao();
        let naux = aux.nao();

        let metric = mat_from_row_major(naux, &aux.eri_2c());
        let l = cholesky_lower(&metric).ok_or(DfError::MetricNotPositiveDefinite)?;

        let schwarz = basis.schwarz_bounds();
        let aux_max = aux.schwarz_aux_bounds().into_iter().fold(0.0_f64, f64::max);
        let nsh = basis.shells().len();
        let builder = basis.eri_3c_builder(aux);
        let mut out = vec![0.0; builder.output_len()];
        {
            let mut tasks = builder.partition(&mut out);
            tasks.retain(|t| {
                let (i, j) = t.bra();
                schwarz[i * nsh + j] * aux_max >= tau
            });
            let shells = basis.shells();
            tasks.sort_unstable_by(|a, b| {
                bra_pair_cost(shells, b.bra()).cmp(&bra_pair_cost(shells, a.bra()))
            });
            tasks.par_iter_mut().for_each(|task| builder.fill(task));
        }

        let n = n_basis;
        for mu in 0..n {
            for nu in 0..=mu {
                let src = (mu * n + nu) * naux;
                let dst = pair_index(mu, nu) * naux;
                out.copy_within(src..src + naux, dst);
            }
        }
        out.truncate(n * (n + 1) / 2 * naux);
        out.shrink_to_fit();

        let npair = n * (n + 1) / 2;
        solve_lower_triangular_cols_in_place(&l, &mut out, npair);

        Ok(Self {
            basis,
            charges,
            ecps: Vec::new(),
            n_basis,
            naux,
            b: out,
        })
    }

    #[must_use]
    pub fn with_ecps(mut self, ecps: Vec<integral::Ecp>) -> Self {
        self.ecps = ecps;
        self
    }

    pub fn naux(&self) -> usize {
        self.naux
    }

    /// Density-fitted Coulomb matrix J = B (Bᵀ d), the cheap O(naux·npair) half of the
    /// fitted build. Factored out of [`Self::jk_for_density`] so the exchange-free SCF
    /// path can skip the expensive O(naux·n³) K contraction entirely.
    fn j_for_density(&self, d: &[f64], j_out: &mut [f64]) {
        let n = self.n_basis;
        let naux = self.naux;
        let npair = n * (n + 1) / 2;

        let mut dw = vec![0.0; npair];
        for mu in 0..n {
            for nu in 0..=mu {
                let w = if mu == nu { 1.0 } else { 2.0 };
                dw[pair_index(mu, nu)] = w * d[mu * n + nu];
            }
        }
        let gamma = gemm(&dw, 1, npair, &self.b, naux);
        let j_packed = gemm(&self.b, npair, naux, &gamma, 1);
        for mu in 0..n {
            for nu in 0..=mu {
                let v = j_packed[pair_index(mu, nu)];
                j_out[mu * n + nu] = v;
                j_out[nu * n + mu] = v;
            }
        }
    }

    fn jk_for_density(&self, d: &[f64], j_out: &mut [f64], k_out: &mut [f64]) {
        let n = self.n_basis;
        let naux = self.naux;
        self.j_for_density(d, j_out);

        let qs: Vec<usize> = (0..naux).collect();
        let partials: Vec<Vec<f64>> = qs
            .par_chunks(16)
            .map(|chunk| {
                let mut k = vec![0.0; n * n];
                for &q in chunk {
                    let bq = self.unpack_bq(q);
                    let t = gemm(&bq, n, n, d, n);
                    let kq = gemm(&t, n, n, &bq, n);
                    for (acc, v) in k.iter_mut().zip(&kq) {
                        *acc += v;
                    }
                }
                k
            })
            .collect();
        k_out.fill(0.0);
        for partial in &partials {
            for (acc, v) in k_out.iter_mut().zip(partial) {
                *acc += v;
            }
        }
    }

    fn unpack_bq(&self, q: usize) -> Vec<f64> {
        let n = self.n_basis;
        let naux = self.naux;
        let mut bq = vec![0.0; n * n];
        for mu in 0..n {
            for nu in 0..=mu {
                let v = self.b[pair_index(mu, nu) * naux + q];
                bq[mu * n + nu] = v;
                bq[nu * n + mu] = v;
            }
        }
        bq
    }
}

impl IntegralProvider for DfProvider {
    fn n_basis(&self) -> usize {
        self.n_basis
    }

    fn overlap(&self) -> Mat {
        mat_from_row_major(self.n_basis, &self.basis.overlap())
    }

    fn kinetic(&self) -> Mat {
        mat_from_row_major(self.n_basis, &self.basis.kinetic())
    }

    fn nuclear(&self) -> Mat {
        mat_from_row_major(
            self.n_basis,
            &nuclear_with_ecp(&self.basis, &self.charges, &self.ecps),
        )
    }

    fn dipole_integrals(&self, origin: [f64; 3]) -> [Vec<f64>; 3] {
        self.basis.dipole(origin)
    }

    fn ao_atom_indices(&self) -> Vec<usize> {
        ao_atom_map(&self.basis, &self.charges)
    }

    fn effective_nuclear_charges(&self) -> Option<Vec<f64>> {
        Some(self.charges.iter().map(|&(_, q)| q).collect())
    }

    fn charge_potential_3c(&self, charges: &[([f64; 3], f64)]) -> Vec<f64> {
        charge_potential_3c_impl(&self.basis, charges)
    }

    fn grid_coulomb(&self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        Some(grid_coulomb_impl(&self.basis, points))
    }

    fn grid_coulomb_erf(&self, points: &[[f64; 3]], omega: f64) -> Option<Vec<f64>> {
        Some(grid_coulomb_erf_impl(&self.basis, points, omega))
    }

    fn build_jk(&self, densities: &[Mat]) -> JkResult {
        let n = self.n_basis;
        let mut coulomb = Vec::with_capacity(densities.len());
        let mut exchange = Vec::with_capacity(densities.len());
        for density in densities {
            let d = mat_to_row_major(density);
            let mut j = vec![0.0; n * n];
            let mut k = vec![0.0; n * n];
            self.jk_for_density(&d, &mut j, &mut k);
            coulomb.push(mat_from_row_major(n, &j));
            exchange.push(mat_from_row_major(n, &k));
        }
        JkResult { coulomb, exchange }
    }

    /// Coulomb-only fit: the O(naux·npair) J build without the O(naux·n³) K contraction,
    /// which is ~99% of [`Self::jk_for_density`]'s cost. The exchange-free SCF path takes
    /// this so a pure functional never pays for an exchange matrix it multiplies by zero.
    fn build_j(&self, densities: &[Mat]) -> Vec<Mat> {
        let n = self.n_basis;
        densities
            .iter()
            .map(|density| {
                let d = mat_to_row_major(density);
                let mut j = vec![0.0; n * n];
                self.j_for_density(&d, &mut j);
                mat_from_row_major(n, &j)
            })
            .collect()
    }
}

impl IntegralProvider for DirectProvider {
    fn n_basis(&self) -> usize {
        self.n_basis
    }

    fn charge_potential_3c(&self, charges: &[([f64; 3], f64)]) -> Vec<f64> {
        charge_potential_3c_impl(&self.basis, charges)
    }

    fn overlap(&self) -> Mat {
        mat_from_row_major(self.n_basis, &self.basis.overlap())
    }

    fn kinetic(&self) -> Mat {
        mat_from_row_major(self.n_basis, &self.basis.kinetic())
    }

    fn nuclear(&self) -> Mat {
        mat_from_row_major(
            self.n_basis,
            &nuclear_with_ecp(&self.basis, &self.charges, &self.ecps),
        )
    }

    fn dipole_integrals(&self, origin: [f64; 3]) -> [Vec<f64>; 3] {
        self.basis.dipole(origin)
    }

    fn ao_atom_indices(&self) -> Vec<usize> {
        ao_atom_map(&self.basis, &self.charges)
    }

    fn effective_nuclear_charges(&self) -> Option<Vec<f64>> {
        Some(self.charges.iter().map(|&(_, q)| q).collect())
    }

    fn grid_coulomb(&self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        Some(grid_coulomb_impl(&self.basis, points))
    }

    fn grid_coulomb_erf(&self, points: &[[f64; 3]], omega: f64) -> Option<Vec<f64>> {
        Some(grid_coulomb_erf_impl(&self.basis, points, omega))
    }

    fn build_jk(&self, densities: &[Mat]) -> JkResult {
        let n = self.n_basis;
        let densities: Vec<Vec<f64>> = densities.iter().map(mat_to_row_major).collect();
        let (j, k) = self.contract_densities(&densities, true);
        let coulomb = j.iter().map(|m| mat_from_row_major(n, m)).collect();
        let exchange = k.iter().map(|m| mat_from_row_major(n, m)).collect();
        JkResult { coulomb, exchange }
    }

    fn build_jk_screened(&self, densities: &[Mat]) -> JkResult {
        let n = self.n_basis;
        let densities: Vec<Vec<f64>> = densities.iter().map(mat_to_row_major).collect();
        let (j, k) = self.contract_densities(&densities, true);
        let coulomb = j.iter().map(|m| mat_from_row_major(n, m)).collect();
        let exchange = k.iter().map(|m| mat_from_row_major(n, m)).collect();
        JkResult { coulomb, exchange }
    }

    fn build_j(&self, densities: &[Mat]) -> Vec<Mat> {
        let n = self.n_basis;
        let densities: Vec<Vec<f64>> = densities.iter().map(mat_to_row_major).collect();
        self.contract_densities(&densities, false)
            .0
            .iter()
            .map(|j| mat_from_row_major(n, j))
            .collect()
    }

    fn build_j_screened(&self, densities: &[Mat]) -> Vec<Mat> {
        let n = self.n_basis;
        let densities: Vec<Vec<f64>> = densities.iter().map(mat_to_row_major).collect();
        self.contract_densities(&densities, false)
            .0
            .iter()
            .map(|j| mat_from_row_major(n, j))
            .collect()
    }
}
