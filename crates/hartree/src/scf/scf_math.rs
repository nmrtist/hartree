use crate::linalg::{mat_from_row_major, mat_to_row_major, symmetric_eigh};

pub(crate) fn mul(a: &[f64], b: &[f64], p: usize, q: usize, r: usize) -> Vec<f64> {
    let mut c = vec![0.0; p * r];
    for i in 0..p {
        for k in 0..q {
            let a_ik = a[i * q + k];
            if a_ik == 0.0 {
                continue;
            }
            let b_row = k * r;
            let c_row = i * r;
            for j in 0..r {
                c[c_row + j] += a_ik * b[b_row + j];
            }
        }
    }
    c
}

pub(crate) fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}

pub(crate) fn eigh(a: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let e = symmetric_eigh(&mat_from_row_major(m, a));
    (e.values, mat_to_row_major(&e.vectors))
}

pub(crate) fn xtax(a: &[f64], x: &[f64], n: usize, m: usize) -> Vec<f64> {
    let xt = transpose(x, n, m);
    let ax = mul(a, x, n, n, m);
    mul(&xt, &ax, m, n, m)
}

pub(crate) fn vtav(a: &[f64], v: &[f64], m: usize) -> Vec<f64> {
    let vt = transpose(v, m, m);
    let av = mul(a, v, m, m, m);
    mul(&vt, &av, m, m, m)
}

pub(crate) fn commutator(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let ab = mul(a, b, m, m, m);
    let ba = mul(b, a, m, m, m);
    let mut c = vec![0.0; m * m];
    for i in 0..m * m {
        c[i] = ab[i] - ba[i];
    }
    c
}

pub(crate) fn orth_occ_density(v: &[f64], m: usize, n_occ: usize) -> Vec<f64> {
    let mut d = vec![0.0; m * m];
    for a in 0..m {
        for b in 0..m {
            let mut s = 0.0;
            for i in 0..n_occ {
                s += v[a * m + i] * v[b * m + i];
            }
            d[a * m + b] = s;
        }
    }
    d
}

pub(crate) fn orth_frac_density(v: &[f64], m: usize, occ: &[f64]) -> Vec<f64> {
    let mut d = vec![0.0; m * m];
    for (i, &f) in occ.iter().enumerate() {
        if f <= crate::scf::smearing::OCC_CUTOFF {
            continue;
        }
        for a in 0..m {
            let fva = f * v[a * m + i];
            if fva == 0.0 {
                continue;
            }
            for b in 0..m {
                d[a * m + b] += fva * v[b * m + i];
            }
        }
    }
    d
}

pub(crate) fn ao_from_orth(x: &[f64], d_orth: &[f64], n: usize, m: usize) -> Vec<f64> {
    let xd = mul(x, d_orth, n, m, m); // n×m
    let xt = transpose(x, n, m); // m×n
    mul(&xd, &xt, n, m, n) // n×n
}

/// Project an AO-basis density `d_ao` (n×n, row-major) into the orthonormal working basis
/// defined by the canonical orthogonalizer `x` (n×m): returns Xᵀ S D S X (m×m). This is
/// the inverse of [`ao_from_orth`]: for a full-rank basis the round trip
/// `ao_from_orth(x, orth_from_ao(d, s, x))` reproduces `d`; when `x` has dropped linearly
/// dependent combinations it yields the component of `d` representable in the kept subspace.
pub(crate) fn orth_from_ao(d_ao: &[f64], s: &[f64], x: &[f64], n: usize, m: usize) -> Vec<f64> {
    let sd = mul(s, d_ao, n, n, n); // S D
    let sds = mul(&sd, s, n, n, n); // S D S
    xtax(&sds, x, n, m) // Xᵀ (S D S) X
}

pub(crate) fn canonical_orthogonalizer(s: &[f64], n: usize, thresh: f64) -> (Vec<f64>, usize) {
    let (values, u) = eigh(s, n); // u: n×n, eigenvector `col` is u[·*n + col]
    let kept: Vec<usize> = (0..n).filter(|&i| values[i] > thresh).collect();
    let m = kept.len();

    let mut x = vec![0.0; n * m];
    for (k, &i) in kept.iter().enumerate() {
        let inv_sqrt = values[i].sqrt().recip();
        for mu in 0..n {
            x[mu * m + k] = u[mu * n + i] * inv_sqrt;
        }
    }
    (x, m)
}

pub(crate) fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        let diag = a[col * n + col];
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    Some((0..n).map(|i| b[i] / a[i * n + i]).collect())
}
