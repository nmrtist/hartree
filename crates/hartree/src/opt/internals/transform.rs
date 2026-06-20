//! Linear-algebra transforms between Cartesian and redundant internal coordinates:
//! the internal gradient, the internal Hessian, the completeness rank, and the
//! iterative back-transformation of an internal step. They build on the Wilson
//! B-matrix and coordinate values defined in the parent module; split out to keep
//! each file under the line cap.

use super::{Internal, values, wilson_b, wrap_to_pi};
use crate::linalg::{mat_from_row_major, mat_to_row_major, symmetric_eigh};

const PINV_REL_TOL: f64 = 1e-8;

pub fn internal_gradient(b: &[f64], gx: &[f64], nq: usize, ndof: usize) -> Vec<f64> {
    let mut bg = vec![0.0; nq];
    for (i, slot) in bg.iter_mut().enumerate() {
        let row = i * ndof;
        let mut s = 0.0;
        for j in 0..ndof {
            s += b[row + j] * gx[j];
        }
        *slot = s;
    }
    let g = gram(b, nq, ndof);
    let ginv = pseudo_inverse(&g, nq);
    matvec(&ginv, &bg, nq)
}

/// Transform a Cartesian Hessian into redundant internal coordinates,
/// `H_q = G⁻ B Hₓ Bᵀ G⁻` with `G = B Bᵀ` and `G⁻` its Moore–Penrose pseudo-inverse.
/// The Wilson-B-matrix-derivative term `Σ_k g_k ∂²q_k/∂x²` is neglected — the standard
/// approximation when the internal Hessian only *seeds* an eigenvector-following step
/// that a quasi-Newton (Bofill) update then refines, and the post-convergence saddle
/// verification re-derives the spectrum in the exact mass-weighted Cartesian frame
/// independently. The redundant null space maps to exactly-zero eigenvalues (`G⁻`
/// annihilates it) and the gradient has no component there, so the caller's non-null
/// filter drops it without contaminating the step. `hess_cart` is row-major
/// `ndof×ndof`; the result is row-major `nq×nq` (symmetrized against round-off).
pub fn internal_hessian(defs: &[Internal], x: &[[f64; 3]], hess_cart: &[f64]) -> Vec<f64> {
    let ndof = 3 * x.len();
    let nq = defs.len();
    if nq == 0 {
        return Vec::new();
    }
    let b = wilson_b(defs, x);
    let ginv = pseudo_inverse(&gram(&b, nq, ndof), nq);

    // M = G⁻ B  (nq × ndof): the same map that takes the Cartesian gradient to the
    // internal one, so `H_q = M Hₓ Mᵀ`.
    let mut m = vec![0.0; nq * ndof];
    for i in 0..nq {
        for j in 0..ndof {
            let mut s = 0.0;
            for k in 0..nq {
                s += ginv[i * nq + k] * b[k * ndof + j];
            }
            m[i * ndof + j] = s;
        }
    }

    // T = Hₓ Mᵀ  (ndof × nq), then H_q = M T  (nq × nq).
    let mut t = vec![0.0; ndof * nq];
    for a in 0..ndof {
        for i in 0..nq {
            let mut s = 0.0;
            for b_idx in 0..ndof {
                s += hess_cart[a * ndof + b_idx] * m[i * ndof + b_idx];
            }
            t[a * nq + i] = s;
        }
    }
    let mut hq = vec![0.0; nq * nq];
    for i in 0..nq {
        for j in 0..nq {
            let mut s = 0.0;
            for a in 0..ndof {
                s += m[i * ndof + a] * t[a * nq + j];
            }
            hq[i * nq + j] = s;
        }
    }
    for i in 0..nq {
        for j in (i + 1)..nq {
            let avg = 0.5 * (hq[i * nq + j] + hq[j * nq + i]);
            hq[i * nq + j] = avg;
            hq[j * nq + i] = avg;
        }
    }
    hq
}

/// The number of independent internal coordinates the set spans at `x` — the rank of
/// `G = B Bᵀ`. A *complete* set has rank ≥ `3N − 6` (`3N − 5` for a linear molecule);
/// a smaller rank means the redundant internals cannot represent every internal
/// displacement, so an optimizer working in them could not move along the missing
/// direction. The caller uses this to fall back to Cartesian coordinates rather than
/// run a crippled search.
pub fn internal_rank(defs: &[Internal], x: &[[f64; 3]]) -> usize {
    let ndof = 3 * x.len();
    let nq = defs.len();
    if nq == 0 {
        return 0;
    }
    let g = gram(&wilson_b(defs, x), nq, ndof);
    let eig = symmetric_eigh(&mat_from_row_major(nq, &g));
    let lam_max = eig.values.iter().cloned().fold(0.0_f64, f64::max);
    let thresh = PINV_REL_TOL * lam_max.max(1e-300);
    eig.values.iter().filter(|&&l| l > thresh).count()
}

/// Map an internal-coordinate step `dq` taken from `x0` back to Cartesian positions by
/// iterating `Δx = Bᵀ G⁻ Δq_remaining` until the move stalls. Dihedral residuals are
/// wrapped into `(−π, π]` so a torsion target past the ±π branch is reached along the
/// short arc rather than chased the long way around.
pub fn back_transform(defs: &[Internal], x0: &[[f64; 3]], dq: &[f64]) -> Vec<[f64; 3]> {
    let natom = x0.len();
    let ndof = 3 * natom;
    let nq = defs.len();

    let q0 = values(defs, x0);
    let q_target: Vec<f64> = q0.iter().zip(dq).map(|(a, d)| a + d).collect();

    let mut x = x0.to_vec();
    for _ in 0..50 {
        let b = wilson_b(defs, &x);
        let g = gram(&b, nq, ndof);
        let ginv = pseudo_inverse(&g, nq);
        let qcur = values(defs, &x);
        let dq_rem: Vec<f64> = defs
            .iter()
            .zip(&q_target)
            .zip(&qcur)
            .map(|((d, &t), &c)| match d {
                Internal::Dihedral(..) => wrap_to_pi(t - c),
                _ => t - c,
            })
            .collect();
        let gd = matvec(&ginv, &dq_rem, nq); // nq
        let mut max_dx = 0.0_f64;
        for (a, xa) in x.iter_mut().enumerate() {
            for c in 0..3 {
                let mut s = 0.0;
                for i in 0..nq {
                    s += b[i * ndof + (3 * a + c)] * gd[i];
                }
                xa[c] += s;
                max_dx = max_dx.max(s.abs());
            }
        }
        if max_dx < 1e-11 {
            break;
        }
    }
    x
}

fn gram(b: &[f64], nq: usize, ndof: usize) -> Vec<f64> {
    let mut g = vec![0.0; nq * nq];
    for i in 0..nq {
        for j in i..nq {
            let mut s = 0.0;
            for d in 0..ndof {
                s += b[i * ndof + d] * b[j * ndof + d];
            }
            g[i * nq + j] = s;
            g[j * nq + i] = s;
        }
    }
    g
}

fn pseudo_inverse(g: &[f64], nq: usize) -> Vec<f64> {
    if nq == 0 {
        return Vec::new();
    }
    let eig = symmetric_eigh(&mat_from_row_major(nq, g));
    let vectors = mat_to_row_major(&eig.vectors); // column k = eigenvector k
    let lam_max = eig.values.iter().cloned().fold(0.0_f64, f64::max);
    let thresh = PINV_REL_TOL * lam_max.max(1e-300);

    let mut out = vec![0.0; nq * nq];
    for k in 0..nq {
        let lam = eig.values[k];
        if lam <= thresh {
            continue;
        }
        let inv = 1.0 / lam;
        for i in 0..nq {
            let vik = vectors[i * nq + k];
            if vik == 0.0 {
                continue;
            }
            for j in 0..nq {
                out[i * nq + j] += inv * vik * vectors[j * nq + k];
            }
        }
    }
    out
}

fn matvec(m: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for (i, slot) in y.iter_mut().enumerate() {
        let row = i * n;
        let mut s = 0.0;
        for j in 0..n {
            s += m[row + j] * x[j];
        }
        *slot = s;
    }
    y
}
