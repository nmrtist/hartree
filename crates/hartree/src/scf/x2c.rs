use crate::scf::scf_math::{canonical_orthogonalizer, eigh, mul, transpose, xtax};
use thiserror::Error;

pub const SPEED_OF_LIGHT_AU: f64 = 137.035999084;

const T_EIG_FLOOR: f64 = 1e-10;

const R_EIG_FLOOR: f64 = 1e-10;

#[derive(Debug, Error)]
pub enum X2cError {
    #[error("X2C: basis has no linearly independent functions")]
    EmptyBasis,

    #[error("X2C: kinetic matrix is not positive definite (eigenvalue {eigenvalue:e})")]
    KineticNotPositiveDefinite { eigenvalue: f64 },

    #[error("X2C: large-component block of the electronic solutions is singular")]
    LargeComponentSingular,

    #[error("X2C: renormalization metric is not positive definite (eigenvalue {eigenvalue:e})")]
    RenormalizationNotPositiveDefinite { eigenvalue: f64 },
}

#[derive(Debug, Clone)]
pub struct X2cTransform {
    pub h: Vec<f64>,
    pub electronic_eigenvalues: Vec<f64>,
    pub n_orbitals: usize,
}

pub fn x2c1e_hcore(
    s: &[f64],
    t: &[f64],
    v: &[f64],
    w: &[f64],
    n: usize,
    c: f64,
    lindep: f64,
) -> Result<X2cTransform, X2cError> {
    assert_eq!(s.len(), n * n, "S must be n×n");
    assert_eq!(t.len(), n * n, "T must be n×n");
    assert_eq!(v.len(), n * n, "V must be n×n");
    assert_eq!(w.len(), n * n, "W must be n×n");
    let c2 = c * c;

    let (xs, m) = canonical_orthogonalizer(s, n, lindep);
    if m == 0 {
        return Err(X2cError::EmptyBasis);
    }
    let tt = xtax(t, &xs, n, m);
    let vv = xtax(v, &xs, n, m);
    let ww = xtax(w, &xs, n, m);

    let t_inv_sqrt = sym_inv_sqrt(&tt, m, T_EIG_FLOOR)
        .map_err(|eigenvalue| X2cError::KineticNotPositiveDefinite { eigenvalue })?;
    let k: Vec<f64> = t_inv_sqrt.iter().map(|x| x * (2.0 * c2).sqrt()).collect();

    let mut lr = vec![0.0; m * m];
    for i in 0..m * m {
        lr[i] = ww[i] / (4.0 * c2) - tt[i];
    }

    let tk = mul(&tt, &k, m, m, m);
    let kt = transpose(&tk, m, m); // K·T̃ = (T̃·K)ᵀ (both factors symmetric)
    let klrk = {
        let a = mul(&k, &lr, m, m, m);
        mul(&a, &k, m, m, m)
    };
    let dim = 2 * m;
    let mut hp = vec![0.0; dim * dim];
    for i in 0..m {
        for j in 0..m {
            hp[i * dim + j] = vv[i * m + j];
            hp[i * dim + (m + j)] = tk[i * m + j];
            hp[(m + i) * dim + j] = kt[i * m + j];
            hp[(m + i) * dim + (m + j)] = klrk[i * m + j];
        }
    }
    symmetrize(&mut hp, dim);

    let (e, u) = eigh(&hp, dim);
    let electronic_eigenvalues = e[m..].to_vec();

    let mut a_l = vec![0.0; m * m];
    let mut b_p = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            a_l[i * m + j] = u[i * dim + (m + j)];
            b_p[i * m + j] = u[(m + i) * dim + (m + j)];
        }
    }
    let b_l = mul(&k, &b_p, m, m, m);

    let xt = solve_multi(transpose(&a_l, m, m), transpose(&b_l, m, m), m, m)
        .ok_or(X2cError::LargeComponentSingular)?;
    let x = transpose(&xt, m, m);

    let xtx = xtax(&tt, &x, m, m);
    let mut st = vec![0.0; m * m];
    for i in 0..m * m {
        st[i] = xtx[i] / (2.0 * c2);
    }
    for i in 0..m {
        st[i * m + i] += 1.0;
    }
    let r = sym_inv_sqrt(&st, m, R_EIG_FLOOR)
        .map_err(|eigenvalue| X2cError::RenormalizationNotPositiveDefinite { eigenvalue })?;

    let tx = mul(&tt, &x, m, m, m);
    let txt = transpose(&tx, m, m);
    let xlx = xtax(&lr, &x, m, m);
    let mut l = vec![0.0; m * m];
    for i in 0..m * m {
        l[i] = vv[i] + tx[i] + txt[i] + xlx[i];
    }
    let mut h_orth = {
        let rl = mul(&r, &l, m, m, m); // Rᵀ = R (symmetric)
        mul(&rl, &r, m, m, m)
    };
    symmetrize(&mut h_orth, m);

    let p = mul(s, &xs, n, n, m);
    let ph = mul(&p, &h_orth, n, m, m);
    let mut h = mul(&ph, &transpose(&p, n, m), n, m, n);
    symmetrize(&mut h, n);

    Ok(X2cTransform {
        h,
        electronic_eigenvalues,
        n_orbitals: m,
    })
}

fn symmetrize(a: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            let avg = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = avg;
            a[j * n + i] = avg;
        }
    }
}

fn sym_inv_sqrt(a: &[f64], m: usize, floor: f64) -> Result<Vec<f64>, f64> {
    let (e, u) = eigh(a, m);
    if e[0] <= floor {
        return Err(e[0]);
    }
    let mut ud = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            ud[i * m + j] = u[i * m + j] / e[j].sqrt();
        }
    }
    Ok(mul(&ud, &transpose(&u, m, m), m, m, m))
}

fn solve_multi(mut a: Vec<f64>, mut b: Vec<f64>, m: usize, k: usize) -> Option<Vec<f64>> {
    for col in 0..m {
        let mut pivot = col;
        let mut best = a[col * m + col].abs();
        for row in (col + 1)..m {
            let v = a[row * m + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != col {
            for j in 0..m {
                a.swap(col * m + j, pivot * m + j);
            }
            for j in 0..k {
                b.swap(col * k + j, pivot * k + j);
            }
        }
        let diag = a[col * m + col];
        for row in 0..m {
            if row == col {
                continue;
            }
            let factor = a[row * m + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for j in col..m {
                a[row * m + j] -= factor * a[col * m + j];
            }
            for j in 0..k {
                b[row * k + j] -= factor * b[col * k + j];
            }
        }
    }
    let mut x = vec![0.0; m * k];
    for i in 0..m {
        let diag = a[i * m + i];
        for j in 0..k {
            x[i * k + j] = b[i * k + j] / diag;
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_function_nonrelativistic_limit() {
        let (s, t, v, w) = ([1.0], [0.8], [-2.0], [3.0]);
        let out = x2c1e_hcore(&s, &t, &v, &w, 1, 1e8, 1e-9).unwrap();
        assert!((out.h[0] - (t[0] + v[0])).abs() < 1e-10, "h = {}", out.h[0]);
    }

    #[test]
    fn one_function_relativistic_lowering() {
        let (s, t, v, w) = ([1.0], [0.8], [-2.0], [-3.0]);
        let out = x2c1e_hcore(&s, &t, &v, &w, 1, SPEED_OF_LIGHT_AU, 1e-9).unwrap();
        assert!(out.h[0] < t[0] + v[0]);
        assert!((out.h[0] - out.electronic_eigenvalues[0]).abs() < 1e-12);
    }

    #[test]
    fn solve_multi_identity() {
        let a = vec![2.0, 1.0, 0.0, 4.0];
        let b = a.clone();
        let x = solve_multi(a, b, 2, 2).unwrap();
        for (i, want) in [1.0, 0.0, 0.0, 1.0].iter().enumerate() {
            assert!((x[i] - want).abs() < 1e-14);
        }
    }
}
