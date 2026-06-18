//! Dense linear algebra over `faer`: eigensolvers and matrix helpers.

pub type Mat = faer::Mat<f64>;

pub struct Eigh {
    pub values: Vec<f64>,
    pub vectors: Mat,
}

pub fn symmetric_eigh(a: &Mat) -> Eigh {
    let decomposition = a
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("symmetric eigendecomposition failed to converge");

    let eigenvalues = decomposition.S();
    let values: Vec<f64> = (0..a.nrows()).map(|i| eigenvalues[i]).collect();
    let vectors = decomposition.U().to_owned();

    Eigh { values, vectors }
}

pub fn mat_from_row_major(n: usize, data: &[f64]) -> Mat {
    assert_eq!(data.len(), n * n, "expected {n}×{n} = {} elements", n * n);
    Mat::from_fn(n, n, |i, j| data[i * n + j])
}

pub fn mat_to_row_major(m: &Mat) -> Vec<f64> {
    let (rows, cols) = (m.nrows(), m.ncols());
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = m[(i, j)];
        }
    }
    out
}

pub fn gemm(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * k, "lhs must be {m}×{k}");
    assert_eq!(b.len(), k * n, "rhs must be {k}×{n}");
    let mut out = vec![0.0; m * n];
    let am = faer::MatRef::from_row_major_slice(a, m, k);
    let bm = faer::MatRef::from_row_major_slice(b, k, n);
    let cm = faer::MatMut::from_row_major_slice_mut(&mut out, m, n);
    faer::linalg::matmul::matmul(cm, faer::Accum::Replace, am, bm, 1.0, faer::Par::Seq);
    out
}

pub fn cholesky_lower(a: &Mat) -> Option<Mat> {
    a.llt(faer::Side::Lower).ok().map(|f| f.L().to_owned())
}

pub fn solve_lower_triangular_cols_in_place(l: &Mat, rhs: &mut [f64], ncols: usize) {
    let n = l.nrows();
    assert_eq!(l.ncols(), n, "L must be square");
    assert_eq!(rhs.len(), n * ncols, "rhs must be {n}×{ncols} column-major");
    let view = faer::MatMut::from_column_major_slice_mut(rhs, n, ncols);
    l.solve_lower_triangular_in_place(view);
}

pub fn cholesky_solve_in_place(l: &Mat, rhs: &mut [f64], ncols: usize) {
    let n = l.nrows();
    assert_eq!(l.ncols(), n, "L must be square");
    assert_eq!(rhs.len(), n * ncols, "rhs must be {n}×{ncols} column-major");
    let view = faer::MatMut::from_column_major_slice_mut(rhs, n, ncols);
    l.solve_lower_triangular_in_place(view);
    let view = faer::MatMut::from_column_major_slice_mut(rhs, n, ncols);
    l.transpose().solve_upper_triangular_in_place(view);
}

pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
    assert_eq!(
        k,
        b.nrows(),
        "inner dimensions {k} and {} disagree",
        b.nrows()
    );
    Mat::from_fn(m, n, |i, j| {
        let mut sum = 0.0;
        for p in 0..k {
            sum += a[(i, p)] * b[(p, j)];
        }
        sum
    })
}

pub fn transpose(a: &Mat) -> Mat {
    Mat::from_fn(a.ncols(), a.nrows(), |i, j| a[(j, i)])
}

pub type C64 = faer::c64;

pub type CMat = faer::Mat<C64>;

pub struct HermitianEigh {
    pub values: Vec<f64>,
    pub vectors: CMat,
}

pub fn hermitian_eigh(a: &CMat) -> HermitianEigh {
    let decomposition = a
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("hermitian eigendecomposition failed to converge");
    let eigenvalues = decomposition.S();
    let values: Vec<f64> = (0..a.nrows()).map(|i| eigenvalues[i].re).collect();
    let vectors = decomposition.U().to_owned();
    HermitianEigh { values, vectors }
}

pub fn hermitian_geneig(h: &CMat, s: &CMat) -> HermitianEigh {
    let n = h.nrows();
    assert_eq!(h.ncols(), n, "H must be square");
    assert_eq!((s.nrows(), s.ncols()), (n, n), "S must match H");

    let l = s
        .llt(faer::Side::Lower)
        .expect("overlap S(k) is not positive definite")
        .L()
        .to_owned();

    let mut y = h.to_owned();
    l.solve_lower_triangular_in_place(y.as_mut());
    let mut w = y.adjoint().to_owned();
    l.solve_lower_triangular_in_place(w.as_mut());
    let hp = w.adjoint().to_owned();

    let dec = hermitian_eigh(&hp);
    let mut c = dec.vectors;
    l.adjoint().solve_upper_triangular_in_place(c.as_mut());
    HermitianEigh {
        values: dec.values,
        vectors: c,
    }
}

pub fn cmat_from_row_major(n: usize, data: &[C64]) -> CMat {
    assert_eq!(data.len(), n * n, "expected {n}×{n} = {} elements", n * n);
    CMat::from_fn(n, n, |i, j| data[i * n + j])
}

pub fn cmat_to_row_major(m: &CMat) -> Vec<C64> {
    let (rows, cols) = (m.nrows(), m.ncols());
    let mut out = vec![C64::new(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = m[(i, j)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn row_major_round_trip_and_matmul() {
        let a = mat_from_row_major(2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mat_to_row_major(&a), vec![1.0, 2.0, 3.0, 4.0]);
        let id = mat_from_row_major(2, &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(mat_to_row_major(&matmul(&a, &id)), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mat_to_row_major(&transpose(&a)), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn gemm_rectangular() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        assert_eq!(gemm(&a, 2, 3, &b, 2), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cholesky_and_forward_solve() {
        let a = mat![[4.0, 2.0], [2.0, 10.0]];
        let l = cholesky_lower(&a).unwrap();
        assert!((l[(0, 0)] - 2.0).abs() < 1e-14);
        assert!((l[(1, 0)] - 1.0).abs() < 1e-14);
        assert!((l[(1, 1)] - 3.0).abs() < 1e-14);
        assert_eq!(l[(0, 1)], 0.0);

        let mut rhs = [2.0, 4.0, 4.0, 11.0];
        solve_lower_triangular_cols_in_place(&l, &mut rhs, 2);
        for (got, want) in rhs.iter().zip([1.0, 1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-14, "got {got}, want {want}");
        }

        assert!(cholesky_lower(&mat![[1.0, 2.0], [2.0, 1.0]]).is_none());
    }

    #[test]
    fn cholesky_full_solve() {
        let a = mat![[4.0, 2.0], [2.0, 10.0]];
        let l = cholesky_lower(&a).unwrap();
        let mut rhs = [8.0, 22.0];
        cholesky_solve_in_place(&l, &mut rhs, 1);
        assert!((rhs[0] - 1.0).abs() < 1e-14);
        assert!((rhs[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn diagonal_eigenvalues() {
        let a = mat![[2.0, 0.0], [0.0, 3.0]];
        let eigh = symmetric_eigh(&a);
        assert!((eigh.values[0] - 2.0).abs() < 1e-12);
        assert!((eigh.values[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn off_diagonal_eigenvalues() {
        let a = mat![[0.0, 1.0], [1.0, 0.0]];
        let eigh = symmetric_eigh(&a);
        assert!((eigh.values[0] + 1.0).abs() < 1e-12);
        assert!((eigh.values[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hermitian_eigenvalues() {
        let a = cmat_from_row_major(
            2,
            &[
                C64::new(1.0, 0.0),
                C64::new(0.0, 1.0),
                C64::new(0.0, -1.0),
                C64::new(1.0, 0.0),
            ],
        );
        let eigh = hermitian_eigh(&a);
        assert!(
            (eigh.values[0] - 0.0).abs() < 1e-12,
            "λ₀ = {}",
            eigh.values[0]
        );
        assert!(
            (eigh.values[1] - 2.0).abs() < 1e-12,
            "λ₁ = {}",
            eigh.values[1]
        );
    }

    #[test]
    fn generalized_hermitian_eigenproblem() {
        let n = 2;
        let h = cmat_from_row_major(
            n,
            &[
                C64::new(2.0, 0.0),
                C64::new(1.0, 1.0),
                C64::new(1.0, -1.0),
                C64::new(3.0, 0.0),
            ],
        );
        let s = cmat_from_row_major(
            n,
            &[
                C64::new(2.0, 0.0),
                C64::new(0.0, 0.5),
                C64::new(0.0, -0.5),
                C64::new(1.0, 0.0),
            ],
        );
        let eigh = hermitian_geneig(&h, &s);
        assert!(eigh.values[0] <= eigh.values[1]);
        let c = &eigh.vectors;
        let hc = &h * c;
        let sc = &s * c;
        for k in 0..n {
            for i in 0..n {
                let resid = hc[(i, k)] - sc[(i, k)] * eigh.values[k];
                assert!(resid.norm() < 1e-10, "residual ({i},{k}) = {resid}");
            }
        }
        let csc = c.adjoint() * &s * c;
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((csc[(i, j)].re - want).abs() < 1e-10 && csc[(i, j)].im.abs() < 1e-10);
            }
        }
    }
}
