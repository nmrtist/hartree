use crate::scf::scf_math::{dot, solve_linear};

pub(crate) struct Diis {
    capacity: usize,
    focks: Vec<Vec<f64>>,
    errors: Vec<Vec<f64>>,
}

impl Diis {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            focks: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, fock: Vec<f64>, error: Vec<f64>) {
        self.focks.push(fock);
        self.errors.push(error);
        while self.focks.len() > self.capacity {
            self.focks.remove(0);
            self.errors.remove(0);
        }
    }

    pub(crate) fn extrapolate(&self) -> Vec<f64> {
        let k = self.focks.len();
        let latest = self.focks[k - 1].clone();
        if k < 2 {
            return latest;
        }

        let dim = k + 1;
        let mut b = vec![0.0; dim * dim];
        for i in 0..k {
            for j in 0..k {
                b[i * dim + j] = dot(&self.errors[i], &self.errors[j]);
            }
            b[i * dim + k] = -1.0;
            b[k * dim + i] = -1.0;
        }
        let mut rhs = vec![0.0; dim];
        rhs[k] = -1.0;

        let coeffs = match solve_linear(b, rhs, dim) {
            Some(c) => c,
            None => return latest,
        };

        let mut fock = vec![0.0; latest.len()];
        for (i, &ci) in coeffs.iter().take(k).enumerate() {
            for (dst, &src) in fock.iter_mut().zip(&self.focks[i]) {
                *dst += ci * src;
            }
        }
        fock
    }
}
