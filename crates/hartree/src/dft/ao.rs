mod harmonics;

use crate::basis::ShellData;
use rayon::prelude::*;

use crate::dft::error::{DftError, Result};

pub use harmonics::{cart_components, cart_norm, n_cart, n_func, shell_transform};

pub const BLOCK_SIZE: usize = 512;

pub const MAX_L: usize = 6;

pub const SCREEN_TOL: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct AoBatch {
    pub npts: usize,
    pub nao: usize,
    pub phi: Vec<f64>,
    pub dphi: [Vec<f64>; 3],
    pub hess: [Vec<f64>; 6],
    pub with_grad: bool,
    pub with_hess: bool,
}

pub fn hess_index(j: usize, k: usize) -> usize {
    const MAP: [[usize; 3]; 3] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];
    MAP[j][k]
}

fn shell_offsets(shells: &[ShellData]) -> Vec<usize> {
    let mut offs = Vec::with_capacity(shells.len());
    let mut acc = 0;
    for s in shells {
        offs.push(acc);
        acc += n_func(s.l as usize, s.spherical);
    }
    offs
}

pub fn n_ao(shells: &[ShellData]) -> usize {
    shells
        .iter()
        .map(|s| n_func(s.l as usize, s.spherical))
        .sum()
}

pub fn ensure_supported(shells: &[ShellData]) -> Result<()> {
    for s in shells {
        if s.l as usize > MAX_L {
            return Err(DftError::UnsupportedAngularMomentum(s.l));
        }
    }
    Ok(())
}

pub fn eval_ao_batch(
    shells: &[ShellData],
    nao: usize,
    points: &[[f64; 3]],
    with_grad: bool,
) -> AoBatch {
    eval_ao_batch_full(shells, nao, points, with_grad, false)
}

pub fn eval_ao_batch_full(
    shells: &[ShellData],
    nao: usize,
    points: &[[f64; 3]],
    with_grad: bool,
    with_hess: bool,
) -> AoBatch {
    let with_grad = with_grad || with_hess;
    let npts = points.len();
    let mut phi = vec![0.0; npts * nao];
    let mut dphi = if with_grad {
        [
            vec![0.0; npts * nao],
            vec![0.0; npts * nao],
            vec![0.0; npts * nao],
        ]
    } else {
        [Vec::new(), Vec::new(), Vec::new()]
    };
    let mut hess = if with_hess {
        std::array::from_fn(|_| vec![0.0; npts * nao])
    } else {
        std::array::from_fn(|_| Vec::new())
    };

    let offsets = shell_offsets(shells);

    for (si, s) in shells.iter().enumerate() {
        if let Some(cn) = shell_significant(s, points) {
            eval_shell_block(
                s,
                &cn,
                offsets[si],
                nao,
                points,
                with_grad,
                with_hess,
                &mut phi,
                &mut dphi,
                &mut hess,
            );
        }
    }

    AoBatch {
        npts,
        nao,
        phi,
        dphi,
        hess,
        with_grad,
        with_hess,
    }
}

/// A block of AO values compacted to the shells that are non-negligible on the block's
/// points: `ao` is an [`AoBatch`] whose width (`nao`) is the number of *locally significant*
/// AO functions `m`, and `loc2glob[local] = global` maps each local column back to its
/// global AO index. Built by [`eval_ao_block`]. Because a shell is included iff it passes
/// the exact same screen the dense [`eval_ao_batch_full`] uses to skip it (its columns
/// would otherwise be exact zeros), contractions run at width `m` reproduce the dense
/// result to round-off while costing O(np·m²) instead of O(np·nao²) — the standard
/// block-sparse XC quadrature (asymptotically linear-scaling for large systems).
pub struct AoBlock {
    pub ao: AoBatch,
    pub loc2glob: Vec<usize>,
}

/// Evaluate AOs on `points`, keeping only the shells significant on this block. See
/// [`AoBlock`]. An empty block (no significant shell) has `ao.nao == 0` and is skipped by
/// the caller.
pub fn eval_ao_block(
    shells: &[ShellData],
    points: &[[f64; 3]],
    with_grad: bool,
    with_hess: bool,
) -> AoBlock {
    let with_grad = with_grad || with_hess;
    let npts = points.len();
    let offsets = shell_offsets(shells);

    // Pass 1: pick out the significant shells (same screen as the dense path), assign each a
    // contiguous local column range, and build the local→global AO map.
    let mut sig: Vec<(usize, Vec<f64>, usize)> = Vec::new(); // (shell index, cn, local offset)
    let mut loc2glob: Vec<usize> = Vec::new();
    let mut m = 0usize;
    for (si, s) in shells.iter().enumerate() {
        if let Some(cn) = shell_significant(s, points) {
            let nf = n_func(s.l as usize, s.spherical);
            sig.push((si, cn, m));
            for f in 0..nf {
                loc2glob.push(offsets[si] + f);
            }
            m += nf;
        }
    }

    let mut phi = vec![0.0; npts * m];
    let mut dphi = if with_grad {
        [
            vec![0.0; npts * m],
            vec![0.0; npts * m],
            vec![0.0; npts * m],
        ]
    } else {
        [Vec::new(), Vec::new(), Vec::new()]
    };
    let mut hess = if with_hess {
        std::array::from_fn(|_| vec![0.0; npts * m])
    } else {
        std::array::from_fn(|_| Vec::new())
    };

    // Pass 2: evaluate the significant shells into the compacted (np × m) layout.
    for (si, cn, loff) in &sig {
        eval_shell_block(
            &shells[*si],
            cn,
            *loff,
            m,
            points,
            with_grad,
            with_hess,
            &mut phi,
            &mut dphi,
            &mut hess,
        );
    }

    AoBlock {
        ao: AoBatch {
            npts,
            nao: m,
            phi,
            dphi,
            hess,
            with_grad,
            with_hess,
        },
        loc2glob,
    }
}

/// The per-block significance screen: returns the normalized contracted coefficients `cn`
/// when shell `s` is non-negligible on `points`, or `None` when it can be dropped. The
/// dense and block-sparse evaluators share this single predicate so they include exactly
/// the same shells — the guarantee behind their numerical agreement.
fn shell_significant(s: &ShellData, points: &[[f64; 3]]) -> Option<Vec<f64>> {
    let l = s.l as usize;
    debug_assert!(l <= MAX_L);
    let [cx, cy, cz] = s.center;
    let cn: Vec<f64> = s
        .exponents
        .iter()
        .zip(&s.coefficients)
        .map(|(&a, &c)| c * harmonics::shell_norm(a, l))
        .collect();

    let (mut dmin2, mut dmax2) = (f64::INFINITY, 0.0_f64);
    for p in points {
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        let dz = p[2] - cz;
        let r2 = dx * dx + dy * dy + dz * dz;
        dmin2 = dmin2.min(r2);
        dmax2 = dmax2.max(r2);
    }
    let radial_cap: f64 = cn
        .iter()
        .zip(&s.exponents)
        .map(|(&c, &a)| c.abs() * (-a * dmin2).exp())
        .sum();
    let mono_cap = dmax2.powf(l as f64 / 2.0); // (dmax)^l
    if radial_cap * mono_cap < SCREEN_TOL && !*NO_SCREEN {
        None
    } else {
        Some(cn)
    }
}

/// Escape hatch: when set, the block-sparse path keeps every shell (no screening), so its
/// compacted width equals `nao` and `loc2glob` is the identity. Used to verify the
/// gather/scatter machinery reproduces the dense path bit-for-bit, isolating it from the
/// (expected, harmless) GEMM-reassociation differences that genuine compaction introduces.
static NO_SCREEN: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("HARTREE_XC_NO_SCREEN").is_some());

/// Evaluate one (already-screened) shell `s` with normalized coefficients `cn` into the
/// AO arrays, writing its `n_func` functions at column offset `off` of a row stride
/// `stride` (the batch width). The dense path passes `off = global offset, stride = nao`;
/// the block-sparse path passes the shell's local offset and the compacted width `m`. All
/// numerics are identical between the two — only the output addressing differs.
#[allow(clippy::too_many_arguments)]
fn eval_shell_block(
    s: &ShellData,
    cn: &[f64],
    off: usize,
    stride: usize,
    points: &[[f64; 3]],
    with_grad: bool,
    with_hess: bool,
    phi: &mut [f64],
    dphi: &mut [Vec<f64>; 3],
    hess: &mut [Vec<f64>; 6],
) {
    let l = s.l as usize;
    let [cx, cy, cz] = s.center;
    let comps = harmonics::cart_components(l);
    let ncart = comps.len();
    let transform = if s.spherical {
        Some(harmonics::shell_transform(l)) // (2l+1) × ncart, row-major
    } else {
        None
    };
    let nf = n_func(l, s.spherical);

    let mut vcart = vec![0.0; ncart];
    let mut gxcart = vec![0.0; ncart];
    let mut gycart = vec![0.0; ncart];
    let mut gzcart = vec![0.0; ncart];
    let mut hcart: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; ncart]);
    let mut px = vec![0.0; l + 1];
    let mut py = vec![0.0; l + 1];
    let mut pz = vec![0.0; l + 1];

    for (pi, p) in points.iter().enumerate() {
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        let dz = p[2] - cz;
        let r2 = dx * dx + dy * dy + dz * dz;

        let mut radial = 0.0;
        let mut radial_g = 0.0;
        let mut radial_h = 0.0;
        for (&c, &a) in cn.iter().zip(&s.exponents) {
            let e = c * (-a * r2).exp();
            radial += e;
            if with_grad {
                radial_g += -2.0 * a * e;
            }
            if with_hess {
                radial_h += 4.0 * a * a * e;
            }
        }

        px[0] = 1.0;
        py[0] = 1.0;
        pz[0] = 1.0;
        for k in 1..=l {
            px[k] = px[k - 1] * dx;
            py[k] = py[k - 1] * dy;
            pz[k] = pz[k - 1] * dz;
        }

        for (c, &[lx, ly, lz]) in comps.iter().enumerate() {
            let mono = px[lx] * py[ly] * pz[lz];
            vcart[c] = radial * mono;
            if with_grad {
                let dmono_x = if lx > 0 {
                    lx as f64 * px[lx - 1] * py[ly] * pz[lz]
                } else {
                    0.0
                };
                let dmono_y = if ly > 0 {
                    ly as f64 * px[lx] * py[ly - 1] * pz[lz]
                } else {
                    0.0
                };
                let dmono_z = if lz > 0 {
                    lz as f64 * px[lx] * py[ly] * pz[lz - 1]
                } else {
                    0.0
                };
                gxcart[c] = radial_g * dx * mono + radial * dmono_x;
                gycart[c] = radial_g * dy * mono + radial * dmono_y;
                gzcart[c] = radial_g * dz * mono + radial * dmono_z;

                if with_hess {
                    let d = [dx, dy, dz];
                    let dmono = [dmono_x, dmono_y, dmono_z];
                    let m2 = [
                        if lx >= 2 {
                            (lx * (lx - 1)) as f64 * px[lx - 2] * py[ly] * pz[lz]
                        } else {
                            0.0
                        },
                        if lx >= 1 && ly >= 1 {
                            (lx * ly) as f64 * px[lx - 1] * py[ly - 1] * pz[lz]
                        } else {
                            0.0
                        },
                        if lx >= 1 && lz >= 1 {
                            (lx * lz) as f64 * px[lx - 1] * py[ly] * pz[lz - 1]
                        } else {
                            0.0
                        },
                        if ly >= 2 {
                            (ly * (ly - 1)) as f64 * px[lx] * py[ly - 2] * pz[lz]
                        } else {
                            0.0
                        },
                        if ly >= 1 && lz >= 1 {
                            (ly * lz) as f64 * px[lx] * py[ly - 1] * pz[lz - 1]
                        } else {
                            0.0
                        },
                        if lz >= 2 {
                            (lz * (lz - 1)) as f64 * px[lx] * py[ly] * pz[lz - 2]
                        } else {
                            0.0
                        },
                    ];
                    for j in 0..3 {
                        for k in j..3 {
                            let h = hess_index(j, k);
                            let delta = if j == k { mono } else { 0.0 };
                            hcart[h][c] = radial_h * d[j] * d[k] * mono
                                + radial_g * (delta + d[j] * dmono[k] + d[k] * dmono[j])
                                + radial * m2[h];
                        }
                    }
                }
            }
        }

        let base = pi * stride + off;
        match &transform {
            None => {
                phi[base..base + nf].copy_from_slice(&vcart[..nf]);
                if with_grad {
                    dphi[0][base..base + nf].copy_from_slice(&gxcart[..nf]);
                    dphi[1][base..base + nf].copy_from_slice(&gycart[..nf]);
                    dphi[2][base..base + nf].copy_from_slice(&gzcart[..nf]);
                }
                if with_hess {
                    for (dst, src) in hess.iter_mut().zip(&hcart) {
                        dst[base..base + nf].copy_from_slice(&src[..nf]);
                    }
                }
            }
            Some(mt) => {
                for q in 0..nf {
                    let row = &mt[q * ncart..q * ncart + ncart];
                    let mut sv = 0.0;
                    let (mut sgx, mut sgy, mut sgz) = (0.0, 0.0, 0.0);
                    for j in 0..ncart {
                        let mij = row[j];
                        sv += mij * vcart[j];
                        if with_grad {
                            sgx += mij * gxcart[j];
                            sgy += mij * gycart[j];
                            sgz += mij * gzcart[j];
                        }
                    }
                    phi[base + q] = sv;
                    if with_grad {
                        dphi[0][base + q] = sgx;
                        dphi[1][base + q] = sgy;
                        dphi[2][base + q] = sgz;
                    }
                    if with_hess {
                        for (dst, src) in hess.iter_mut().zip(&hcart) {
                            let mut sh = 0.0;
                            for j in 0..ncart {
                                sh += row[j] * src[j];
                            }
                            dst[base + q] = sh;
                        }
                    }
                }
            }
        }
    }
}

pub fn par_blocks_fold<T, Id, Fold, Red>(
    shells: &[ShellData],
    nao: usize,
    points: &[[f64; 3]],
    with_grad: bool,
    identity: Id,
    fold: Fold,
    reduce: Red,
) -> Result<T>
where
    T: Send,
    Id: Fn() -> T + Sync + Send,
    Fold: Fn(T, &AoBatch, usize) -> T + Sync + Send,
    Red: Fn(T, T) -> T + Sync + Send,
{
    par_blocks_fold_full(
        shells, nao, points, with_grad, false, identity, fold, reduce,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn par_blocks_fold_full<T, Id, Fold, Red>(
    shells: &[ShellData],
    nao: usize,
    points: &[[f64; 3]],
    with_grad: bool,
    with_hess: bool,
    identity: Id,
    fold: Fold,
    reduce: Red,
) -> Result<T>
where
    T: Send,
    Id: Fn() -> T + Sync + Send,
    Fold: Fn(T, &AoBatch, usize) -> T + Sync + Send,
    Red: Fn(T, T) -> T + Sync + Send,
{
    ensure_supported(shells)?;
    let n = points.len();
    let acc = (0..n)
        .into_par_iter()
        .step_by(BLOCK_SIZE)
        .fold(&identity, |acc, start| {
            let end = (start + BLOCK_SIZE).min(n);
            let batch = eval_ao_batch_full(shells, nao, &points[start..end], with_grad, with_hess);
            fold(acc, &batch, start)
        })
        .reduce(&identity, &reduce);
    Ok(acc)
}

/// Like [`par_blocks_fold`] but evaluates each block with [`eval_ao_block`], handing the
/// fold a block-sparse [`AoBlock`] (width = locally significant AO count) instead of the
/// dense `nao`-wide [`AoBatch`]. The fold body gathers the density and scatters the
/// potential through `block.loc2glob`; everything in between runs at the compacted width.
/// Empty blocks (no significant shell) are skipped. The `HARTREE_XC_NO_SCREEN` escape hatch
/// forces every shell to be kept (width = `nao`), recovering the dense layout for
/// bit-for-bit cross-checks of the compaction machinery.
pub fn par_blocks_fold_local<T, Id, Fold, Red>(
    shells: &[ShellData],
    points: &[[f64; 3]],
    with_grad: bool,
    identity: Id,
    fold: Fold,
    reduce: Red,
) -> Result<T>
where
    T: Send,
    Id: Fn() -> T + Sync + Send,
    Fold: Fn(T, &AoBlock, usize) -> T + Sync + Send,
    Red: Fn(T, T) -> T + Sync + Send,
{
    ensure_supported(shells)?;
    let n = points.len();
    let acc = (0..n)
        .into_par_iter()
        .step_by(BLOCK_SIZE)
        .fold(&identity, |acc, start| {
            let end = (start + BLOCK_SIZE).min(n);
            let block = eval_ao_block(shells, &points[start..end], with_grad, false);
            if block.ao.nao == 0 {
                return acc;
            }
            fold(acc, &block, start)
        })
        .reduce(&identity, &reduce);
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::ShellData;

    fn s_shell(center: [f64; 3]) -> ShellData {
        ShellData {
            l: 0,
            center,
            exponents: vec![0.8],
            coefficients: vec![1.0],
            spherical: false,
        }
    }

    #[test]
    fn offsets_and_counts() {
        let shells = vec![
            ShellData {
                l: 0,
                center: [0.0; 3],
                exponents: vec![1.0],
                coefficients: vec![1.0],
                spherical: false,
            },
            ShellData {
                l: 1,
                center: [0.0; 3],
                exponents: vec![1.0],
                coefficients: vec![1.0],
                spherical: false,
            },
            ShellData {
                l: 2,
                center: [0.0; 3],
                exponents: vec![1.0],
                coefficients: vec![1.0],
                spherical: true,
            },
        ];
        assert_eq!(n_ao(&shells), 9);
        assert_eq!(shell_offsets(&shells), vec![0, 1, 4]);
    }

    #[test]
    fn unsupported_l_errors() {
        let shells = vec![ShellData {
            l: 7,
            center: [0.0; 3],
            exponents: vec![1.0],
            coefficients: vec![1.0],
            spherical: true,
        }];
        assert!(ensure_supported(&shells).is_err());
    }

    #[test]
    fn s_value_and_gradient() {
        let shells = vec![s_shell([0.1, -0.2, 0.3])];
        let pts = vec![[0.5, 0.4, -0.1], [0.0, 0.0, 0.0]];
        let batch = eval_ao_batch(&shells, 1, &pts, true);

        let a = 0.8;
        let n = cart_norm(a, 0, 0, 0);
        for (pi, p) in pts.iter().enumerate() {
            let dx = p[0] - 0.1;
            let dy = p[1] + 0.2;
            let dz = p[2] - 0.3;
            let r2 = dx * dx + dy * dy + dz * dz;
            let expect = n * (-a * r2).exp();
            assert!((batch.phi[pi] - expect).abs() < 1e-12);
            assert!((batch.dphi[0][pi] - (-2.0 * a * dx * expect)).abs() < 1e-12);
            assert!((batch.dphi[1][pi] - (-2.0 * a * dy * expect)).abs() < 1e-12);
            assert!((batch.dphi[2][pi] - (-2.0 * a * dz * expect)).abs() < 1e-12);
        }
    }

    #[test]
    fn spherical_d_gradient_matches_fd() {
        let shells = vec![ShellData {
            l: 2,
            center: [0.0, 0.0, 0.0],
            exponents: vec![1.3, 0.4],
            coefficients: vec![0.5, 0.6],
            spherical: true,
        }];
        let p = [0.37, -0.21, 0.44];
        let eps = 1e-6;
        let nao = 5;
        let base = eval_ao_batch(&shells, nao, &[p], true);
        for axis in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[axis] += eps;
            pm[axis] -= eps;
            let bp = eval_ao_batch(&shells, nao, &[pp], false);
            let bm = eval_ao_batch(&shells, nao, &[pm], false);
            for k in 0..nao {
                let fd = (bp.phi[k] - bm.phi[k]) / (2.0 * eps);
                let an = base.dphi[axis][k];
                assert!(
                    (fd - an).abs() < 1e-6,
                    "axis {axis} comp {k}: fd={fd} an={an}"
                );
            }
        }
    }

    #[test]
    fn second_derivatives_match_fd_of_gradient() {
        let shells = vec![
            ShellData {
                l: 2,
                center: [0.0, 0.0, 0.0],
                exponents: vec![1.3, 0.4],
                coefficients: vec![0.5, 0.6],
                spherical: true,
            },
            ShellData {
                l: 1,
                center: [0.3, -0.1, 0.2],
                exponents: vec![0.9],
                coefficients: vec![1.0],
                spherical: false,
            },
        ];
        let nao = 8; // 5 (spherical d) + 3 (p)
        let p = [0.37, -0.21, 0.44];
        let eps = 1e-6;
        let base = eval_ao_batch_full(&shells, nao, &[p], true, true);
        for j in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[j] += eps;
            pm[j] -= eps;
            let bp = eval_ao_batch(&shells, nao, &[pp], true);
            let bm = eval_ao_batch(&shells, nao, &[pm], true);
            for k in 0..3 {
                for mu in 0..nao {
                    let fd = (bp.dphi[k][mu] - bm.dphi[k][mu]) / (2.0 * eps);
                    let an = base.hess[hess_index(j, k)][mu];
                    assert!((fd - an).abs() < 5e-6, "∂{j}∂{k} φ_{mu}: fd={fd} an={an}");
                }
            }
        }
    }

    #[test]
    fn screening_zeros_distant_shell() {
        let shells = vec![s_shell([0.0, 0.0, 0.0])];
        let pts = vec![[50.0, 0.0, 0.0]];
        let batch = eval_ao_batch(&shells, 1, &pts, false);
        assert_eq!(batch.phi[0], 0.0);
    }
}
