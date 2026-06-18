use crate::tensor::{Tensor, tensordot};

pub fn column_block(c: &[f64], n: usize, m: usize, start: usize, count: usize) -> Tensor {
    assert!(start + count <= m, "column block out of range");
    let mut data = vec![0.0; n * count];
    for row in 0..n {
        for k in 0..count {
            data[row * count + k] = c[row * m + (start + k)];
        }
    }
    Tensor::new(vec![n, count], data)
}

pub fn transform_block(ao_eri: &[f64], n: usize, c: [&Tensor; 4]) -> Tensor {
    let mut t = Tensor::new(vec![n, n, n, n], ao_eri.to_vec());
    for ci in c {
        debug_assert_eq!(ci.shape()[0], n, "coefficient block has wrong AO dimension");
        t = tensordot(ci, &[0], &t, &[0]);
        t = t.rotate_left();
    }
    t
}

pub fn core_hamiltonian_mo(h_ao: &[f64], n: usize, cp: &Tensor, cq: &Tensor) -> Vec<f64> {
    let h = Tensor::new(vec![n, n], h_ao.to_vec());
    let h_cq = tensordot(&h, &[1], cq, &[0]); // [n, m_q]
    let h_mo = tensordot(cp, &[0], &h_cq, &[0]); // [m_p, m_q]
    h_mo.into_data()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_block_extracts_columns() {
        let c = vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        let b = column_block(&c, 3, 3, 1, 2);
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data(), &[2.0, 3.0, 5.0, 6.0, 8.0, 9.0]);
    }

    #[test]
    fn identity_transform_is_noop() {
        let n = 2;
        let ao: Vec<f64> = (0..n * n * n * n)
            .map(|i| (i as f64 * 0.13).cos())
            .collect();
        let id = Tensor::new(vec![n, n], vec![1.0, 0.0, 0.0, 1.0]);
        let out = transform_block(&ao, n, [&id, &id, &id, &id]);
        assert_eq!(out.shape(), &[n, n, n, n]);
        for (x, y) in out.data().iter().zip(&ao) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn transform_vs_naive() {
        let n = 3usize;
        let ao: Vec<f64> = (0..n.pow(4)).map(|i| (i as f64 * 0.21).sin()).collect();
        let cp = Tensor::new(vec![n, 1], vec![0.5, -0.2, 0.9]);
        let cq = Tensor::new(vec![n, 1], vec![0.1, 0.7, -0.3]);
        let cr = Tensor::new(vec![n, 1], vec![-0.6, 0.4, 0.8]);
        let cs = Tensor::new(vec![n, 1], vec![0.2, -0.5, 0.33]);
        let out = transform_block(&ao, n, [&cp, &cq, &cr, &cs]);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);

        let mut expect = 0.0;
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        let v = ao[((p * n + q) * n + r) * n + s];
                        expect += cp.data()[p] * cq.data()[q] * cr.data()[r] * cs.data()[s] * v;
                    }
                }
            }
        }
        assert!((out.data()[0] - expect).abs() < 1e-12);
    }
}
