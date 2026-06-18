//! Dense tensors and reshape-to-GEMM contractions for the post-HF kernels.

use std::borrow::Cow;

use crate::linalg::gemm;
use rayon::prelude::*;

const PARALLEL_PERMUTE_THRESHOLD: usize = 1 << 18;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "data length {} does not match shape {shape:?} (= {n})",
            data.len()
        );
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; n],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn into_data(self) -> Vec<f64> {
        self.data
    }

    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.data.len(), "reshape changes element count");
        self.shape = shape;
        self
    }

    pub fn permute(&self, perm: &[usize]) -> Tensor {
        let nd = self.shape.len();
        assert_eq!(perm.len(), nd, "permutation rank mismatch");
        let new_shape: Vec<usize> = perm.iter().map(|&p| self.shape[p]).collect();
        let old_strides = row_major_strides(&self.shape);
        let total = self.data.len();
        let mut out = vec![0.0; total];

        if total >= PARALLEL_PERMUTE_THRESHOLD {
            let new_strides = row_major_strides(&new_shape);
            out.par_iter_mut().enumerate().for_each(|(lin, slot)| {
                let mut rem = lin;
                let mut old_lin = 0;
                for i in 0..nd {
                    let coord = rem / new_strides[i];
                    rem -= coord * new_strides[i];
                    old_lin += coord * old_strides[perm[i]];
                }
                *slot = self.data[old_lin];
            });
        } else {
            let mut idx = vec![0usize; nd];
            for slot in out.iter_mut() {
                let mut old_lin = 0;
                for (i, &p) in perm.iter().enumerate() {
                    old_lin += idx[i] * old_strides[p];
                }
                *slot = self.data[old_lin];
                for ax in (0..nd).rev() {
                    idx[ax] += 1;
                    if idx[ax] < new_shape[ax] {
                        break;
                    }
                    idx[ax] = 0;
                }
            }
        }
        Tensor {
            shape: new_shape,
            data: out,
        }
    }

    pub fn rotate_left(&self) -> Tensor {
        let nd = self.shape.len();
        let perm: Vec<usize> = (1..nd).chain(std::iter::once(0)).collect();
        self.permute(&perm)
    }
}

fn permuted_data<'a>(t: &'a Tensor, perm: &[usize]) -> Cow<'a, [f64]> {
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        Cow::Borrowed(&t.data)
    } else {
        Cow::Owned(t.permute(perm).into_data())
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let nd = shape.len();
    let mut strides = vec![1usize; nd];
    for i in (0..nd.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn tensordot(a: &Tensor, a_axes: &[usize], b: &Tensor, b_axes: &[usize]) -> Tensor {
    assert_eq!(
        a_axes.len(),
        b_axes.len(),
        "contracted-axis lists must match in length"
    );
    for (&aa, &ba) in a_axes.iter().zip(b_axes) {
        assert_eq!(
            a.shape[aa], b.shape[ba],
            "contracted dimensions disagree: a[{aa}]={} vs b[{ba}]={}",
            a.shape[aa], b.shape[ba]
        );
    }

    let a_free: Vec<usize> = (0..a.shape.len()).filter(|i| !a_axes.contains(i)).collect();
    let b_free: Vec<usize> = (0..b.shape.len()).filter(|i| !b_axes.contains(i)).collect();

    let rows: usize = a_free.iter().map(|&i| a.shape[i]).product();
    let cols: usize = b_free.iter().map(|&i| b.shape[i]).product();
    let k: usize = a_axes.iter().map(|&i| a.shape[i]).product();

    let a_perm: Vec<usize> = a_free.iter().chain(a_axes).copied().collect();
    let b_perm: Vec<usize> = b_axes.iter().chain(&b_free).copied().collect();
    let am = permuted_data(a, &a_perm);
    let bm = permuted_data(b, &b_perm);

    let c = gemm(&am, rows, k, &bm, cols);

    let mut shape: Vec<usize> = a_free.iter().map(|&i| a.shape[i]).collect();
    shape.extend(b_free.iter().map(|&i| b.shape[i]));
    Tensor { shape, data: c }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_single_axis(a: &Tensor, aa: usize, b: &Tensor, ba: usize) -> Tensor {
        let a_free: Vec<usize> = (0..a.ndim()).filter(|&i| i != aa).collect();
        let b_free: Vec<usize> = (0..b.ndim()).filter(|&i| i != ba).collect();
        let a_str = row_major_strides(a.shape());
        let b_str = row_major_strides(b.shape());
        let kdim = a.shape()[aa];

        let mut out_shape: Vec<usize> = a_free.iter().map(|&i| a.shape()[i]).collect();
        out_shape.extend(b_free.iter().map(|&i| b.shape()[i]));
        let total: usize = out_shape.iter().product();
        let out_str = row_major_strides(&out_shape);

        let mut out = vec![0.0; total];
        let mut a_idx = vec![0usize; a_free.len()];
        let a_total: usize = a_free
            .iter()
            .map(|&i| a.shape()[i])
            .product::<usize>()
            .max(1);
        for _ in 0..a_total {
            let mut b_idx = vec![0usize; b_free.len()];
            let b_total: usize = b_free
                .iter()
                .map(|&i| b.shape()[i])
                .product::<usize>()
                .max(1);
            for _ in 0..b_total {
                let mut s = 0.0;
                for k in 0..kdim {
                    let mut al = k * a_str[aa];
                    for (p, &ax) in a_free.iter().enumerate() {
                        al += a_idx[p] * a_str[ax];
                    }
                    let mut bl = k * b_str[ba];
                    for (p, &bx) in b_free.iter().enumerate() {
                        bl += b_idx[p] * b_str[bx];
                    }
                    s += a.data()[al] * b.data()[bl];
                }
                let mut ol = 0;
                for (p, &v) in a_idx.iter().enumerate() {
                    ol += v * out_str[p];
                }
                for (p, &v) in b_idx.iter().enumerate() {
                    ol += v * out_str[a_free.len() + p];
                }
                out[ol] = s;
                increment(
                    &mut b_idx,
                    &b_free.iter().map(|&i| b.shape()[i]).collect::<Vec<_>>(),
                );
            }
            increment(
                &mut a_idx,
                &a_free.iter().map(|&i| a.shape()[i]).collect::<Vec<_>>(),
            );
        }
        Tensor::new(out_shape, out)
    }

    fn increment(idx: &mut [usize], shape: &[usize]) {
        for ax in (0..idx.len()).rev() {
            idx[ax] += 1;
            if idx[ax] < shape[ax] {
                return;
            }
            idx[ax] = 0;
        }
    }

    fn ramp(shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::new(shape, (0..n).map(|i| (i as f64 * 0.37).sin()).collect())
    }

    #[test]
    fn permute_matches_manual() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tt = t.permute(&[1, 0]);
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn tensordot_is_matmul() {
        let a = ramp(vec![2, 3]);
        let b = ramp(vec![3, 4]);
        let c = tensordot(&a, &[1], &b, &[0]);
        assert_eq!(c.shape(), &[2, 4]);
        let expect = gemm(a.data(), 2, 3, b.data(), 4);
        for (x, y) in c.data().iter().zip(&expect) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn tensordot_4index_vs_naive() {
        let a = ramp(vec![3, 4, 5]);
        let b = ramp(vec![5, 2, 3]);
        let c = tensordot(&a, &[2], &b, &[0]); // shape [3,4,2,3]
        let reference = naive_single_axis(&a, 2, &b, 0);
        assert_eq!(c.shape(), reference.shape());
        for (x, y) in c.data().iter().zip(reference.data()) {
            assert!((x - y).abs() < 1e-12, "tensordot vs naive mismatch");
        }
    }

    #[test]
    fn rotate_left_cycles_axes() {
        let t = ramp(vec![2, 3, 4, 5]);
        let r = t.rotate_left();
        assert_eq!(r.shape(), &[3, 4, 5, 2]);
        let back = t.rotate_left().rotate_left().rotate_left().rotate_left();
        assert_eq!(back, t);
    }
}
