use crate::basis::ShellData;
use crate::core::Molecule;
use crate::integrals::{IntegralProvider, JkResult};
use crate::linalg::{Mat, gemm, mat_from_row_major, mat_to_row_major, symmetric_eigh};

use crate::dft::ao;
use crate::dft::error::{DftError, Result};
use crate::dft::grid::MolecularGrid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CosxGrid {
    Small,
    Medium,
}

impl CosxGrid {
    pub fn level(self) -> usize {
        match self {
            CosxGrid::Small => 0,
            CosxGrid::Medium => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CosxGrid::Small => "small",
            CosxGrid::Medium => "medium",
        }
    }
}

pub const COSX_DEFAULT_GRID: CosxGrid = CosxGrid::Medium;

pub const FIT_EIG_CUTOFF: f64 = 1e-10;

const BATCH_TARGET_BYTES: usize = 32 << 20; // 32 MiB

pub struct CosxExchange {
    shells: Vec<ShellData>,
    nao: usize,
    points: Vec<[f64; 3]>,
    weights: Vec<f64>,
    fit: Option<Vec<f64>>,
    description: String,
    batch: usize,
}

impl CosxExchange {
    pub fn new(
        mol: &Molecule,
        shells: &[ShellData],
        nao: usize,
        overlap: &[f64],
        grid: CosxGrid,
    ) -> Result<Self> {
        Self::with_grid_level(mol, shells, nao, Some(overlap), grid.level(), grid.name())
    }

    pub fn with_grid_level(
        mol: &Molecule,
        shells: &[ShellData],
        nao: usize,
        overlap: Option<&[f64]>,
        level: usize,
        grid_name: &str,
    ) -> Result<Self> {
        ao::ensure_supported(shells)?;
        let grid = MolecularGrid::build(mol, level)?;
        let mut this = Self {
            shells: shells.to_vec(),
            nao,
            points: grid.points,
            weights: grid.weights,
            fit: None,
            description: format!("{grid_name} (level {level})"),
            batch: (BATCH_TARGET_BYTES / (8 * nao * nao).max(1)).clamp(16, 1024),
        };
        if let Some(s) = overlap {
            assert_eq!(s.len(), nao * nao, "overlap must be nao × nao");
            this.fit = Some(this.fit_factor(s));
        }
        Ok(this)
    }

    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    pub fn fitted(&self) -> bool {
        self.fit.is_some()
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    fn fit_factor(&self, s: &[f64]) -> Vec<f64> {
        let nao = self.nao;
        let weights = &self.weights;
        let m = ao::par_blocks_fold(
            &self.shells,
            nao,
            &self.points,
            false,
            || vec![0.0; nao * nao],
            |mut acc, batch, start| {
                let np = batch.npts;
                let wx = scale_rows(&batch.phi, &weights[start..start + np], np, nao);
                let xt = transpose_rowmajor(&batch.phi, np, nao);
                let g = gemm(&xt, nao, np, &wx, nao);
                for (a, v) in acc.iter_mut().zip(&g) {
                    *a += v;
                }
                acc
            },
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            },
        )
        .expect("shells validated in with_grid_level");

        let eigh = symmetric_eigh(&mat_from_row_major(nao, &m));
        let lambda_max = eigh.values.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let cutoff = FIT_EIG_CUTOFF * lambda_max;
        let v = mat_to_row_major(&eigh.vectors); // columns = eigenvectors
        let mut v_scaled = v.clone();
        for (j, &lam) in eigh.values.iter().enumerate() {
            let inv = if lam > cutoff { 1.0 / lam } else { 0.0 };
            for i in 0..nao {
                v_scaled[i * nao + j] *= inv;
            }
        }
        let vt = transpose_rowmajor(&v, nao, nao);
        let m_pinv = gemm(&v_scaled, nao, nao, &vt, nao);
        gemm(&m_pinv, nao, nao, s, nao)
    }

    pub fn build_k<P: IntegralProvider + ?Sized>(
        &self,
        provider: &P,
        densities: &[Mat],
    ) -> Option<Vec<Mat>> {
        self.build_k_kernels(provider, densities, &[CosxKernel::Coulomb])
            .map(|mut ks| ks.remove(0))
    }

    pub fn build_k_erf<P: IntegralProvider + ?Sized>(
        &self,
        provider: &P,
        densities: &[Mat],
        omega: f64,
    ) -> Option<Vec<Mat>> {
        self.build_k_kernels(provider, densities, &[CosxKernel::Erf(omega)])
            .map(|mut ks| ks.remove(0))
    }

    pub fn build_k_rs<P: IntegralProvider + ?Sized>(
        &self,
        provider: &P,
        densities: &[Mat],
        omega: f64,
    ) -> Option<(Vec<Mat>, Vec<Mat>)> {
        let mut ks = self.build_k_kernels(
            provider,
            densities,
            &[CosxKernel::Coulomb, CosxKernel::Erf(omega)],
        )?;
        let klr = ks.remove(1);
        let kc = ks.remove(0);
        Some((kc, klr))
    }

    fn build_k_kernels<P: IntegralProvider + ?Sized>(
        &self,
        provider: &P,
        densities: &[Mat],
        kernels: &[CosxKernel],
    ) -> Option<Vec<Vec<Mat>>> {
        let nao = self.nao;
        let mm = nao * nao;
        let ds: Vec<Vec<f64>> = densities.iter().map(mat_to_row_major).collect();
        let mut ks = vec![vec![vec![0.0; mm]; densities.len()]; kernels.len()];

        for (chunk_idx, chunk) in self.points.chunks(self.batch).enumerate() {
            let np = chunk.len();
            let w = &self.weights[chunk_idx * self.batch..chunk_idx * self.batch + np];
            let batch = ao::eval_ao_batch(&self.shells, nao, chunk, false);
            let x = &batch.phi; // np × nao
            let wx = scale_rows(x, w, np, nao);
            let bra = match &self.fit {
                Some(fit) => gemm(&wx, np, nao, fit, nao),
                None => wx,
            };
            let bra_t = transpose_rowmajor(&bra, np, nao);
            let fs: Vec<Vec<f64>> = ds.iter().map(|d| gemm(x, np, nao, d, nao)).collect();

            for (kernel, ks_kernel) in kernels.iter().zip(ks.iter_mut()) {
                let a = match kernel {
                    CosxKernel::Coulomb => provider.grid_coulomb(chunk)?,
                    CosxKernel::Erf(omega) => provider.grid_coulomb_erf(chunk, *omega)?,
                };
                for (f, k) in fs.iter().zip(ks_kernel.iter_mut()) {
                    let mut g = vec![0.0; np * nao];
                    for p in 0..np {
                        let ap = &a[p * mm..(p + 1) * mm];
                        let fp = &f[p * nao..(p + 1) * nao];
                        let gp = &mut g[p * nao..(p + 1) * nao];
                        for nu in 0..nao {
                            let row = &ap[nu * nao..(nu + 1) * nao];
                            let mut acc = 0.0;
                            for (av, fv) in row.iter().zip(fp) {
                                acc += av * fv;
                            }
                            gp[nu] = acc;
                        }
                    }
                    let kc = gemm(&bra_t, nao, np, &g, nao);
                    for (kk, v) in k.iter_mut().zip(&kc) {
                        *kk += v;
                    }
                }
            }
        }

        Some(
            ks.into_iter()
                .map(|ks_kernel| {
                    ks_kernel
                        .into_iter()
                        .map(|mut k| {
                            for mu in 0..nao {
                                for nu in 0..mu {
                                    let avg = 0.5 * (k[mu * nao + nu] + k[nu * nao + mu]);
                                    k[mu * nao + nu] = avg;
                                    k[nu * nao + mu] = avg;
                                }
                            }
                            mat_from_row_major(nao, &k)
                        })
                        .collect()
                })
                .collect(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CosxKernel {
    Coulomb,
    Erf(f64),
}

fn scale_rows(x: &[f64], w: &[f64], np: usize, nao: usize) -> Vec<f64> {
    let mut out = vec![0.0; np * nao];
    for p in 0..np {
        let wp = w[p];
        for mu in 0..nao {
            out[p * nao + mu] = wp * x[p * nao + mu];
        }
    }
    out
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

pub struct CosxProvider<'a, P: IntegralProvider> {
    inner: &'a P,
    cosx: CosxExchange,
    omega: Option<f64>,
    klr_cache: std::sync::Mutex<Option<KlrCacheEntry>>,
}

type KlrCacheEntry = (Vec<Vec<f64>>, Vec<Mat>);

impl<'a, P: IntegralProvider> CosxProvider<'a, P> {
    pub fn new(inner: &'a P, cosx: CosxExchange) -> Result<Self> {
        if inner.grid_coulomb(&[]).is_none() {
            return Err(DftError::CosxUnsupportedBackend);
        }
        Ok(Self {
            inner,
            cosx,
            omega: None,
            klr_cache: std::sync::Mutex::new(None),
        })
    }

    pub fn with_range_separation(inner: &'a P, cosx: CosxExchange, omega: f64) -> Result<Self> {
        if inner.grid_coulomb(&[]).is_none() || inner.grid_coulomb_erf(&[], omega).is_none() {
            return Err(DftError::CosxUnsupportedBackend);
        }
        Ok(Self {
            inner,
            cosx,
            omega: Some(omega),
            klr_cache: std::sync::Mutex::new(None),
        })
    }

    pub fn cosx(&self) -> &CosxExchange {
        &self.cosx
    }

    pub fn range_separation_omega(&self) -> Option<f64> {
        self.omega
    }
}

impl<P: IntegralProvider> IntegralProvider for CosxProvider<'_, P> {
    fn n_basis(&self) -> usize {
        self.inner.n_basis()
    }

    fn overlap(&self) -> Mat {
        self.inner.overlap()
    }

    fn kinetic(&self) -> Mat {
        self.inner.kinetic()
    }

    fn nuclear(&self) -> Mat {
        self.inner.nuclear()
    }

    fn dipole_integrals(&self, origin: [f64; 3]) -> [Vec<f64>; 3] {
        self.inner.dipole_integrals(origin)
    }

    fn ao_atom_indices(&self) -> Vec<usize> {
        self.inner.ao_atom_indices()
    }

    fn charge_potential_3c(&self, charges: &[([f64; 3], f64)]) -> Vec<f64> {
        self.inner.charge_potential_3c(charges)
    }

    fn grid_coulomb(&self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.inner.grid_coulomb(points)
    }

    fn grid_coulomb_erf(&self, points: &[[f64; 3]], omega: f64) -> Option<Vec<f64>> {
        self.inner.grid_coulomb_erf(points, omega)
    }

    fn build_jk(&self, densities: &[Mat]) -> JkResult {
        let coulomb = self.inner.build_j(densities);
        let exchange = match self.omega {
            Some(omega) => {
                let (kc, klr) = self
                    .cosx
                    .build_k_rs(self.inner, densities, omega)
                    .expect("grid kernel support probed at construction");
                let key: Vec<Vec<f64>> = densities.iter().map(mat_to_row_major).collect();
                *self.klr_cache.lock().unwrap() = Some((key, klr));
                kc
            }
            None => self
                .cosx
                .build_k(self.inner, densities)
                .expect("grid_coulomb support probed at construction"),
        };
        JkResult { coulomb, exchange }
    }

    fn build_k_erf(&self, densities: &[Mat], omega: f64) -> Option<Vec<Mat>> {
        let rs_omega = self.omega?;
        if rs_omega == omega {
            let cached = self.klr_cache.lock().unwrap().take();
            if let Some((key, klr)) = cached {
                let matches = key.len() == densities.len()
                    && key
                        .iter()
                        .zip(densities)
                        .all(|(k, d)| *k == mat_to_row_major(d));
                if matches {
                    return Some(klr);
                }
            }
        }
        self.cosx.build_k_erf(self.inner, densities, omega)
    }
}
