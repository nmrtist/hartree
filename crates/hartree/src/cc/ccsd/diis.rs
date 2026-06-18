pub(crate) struct AmplitudeDiis {
    capacity: usize,
    amplitudes: Vec<Vec<f64>>,
    errors: Vec<Vec<f64>>,
}

impl AmplitudeDiis {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            amplitudes: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, amplitude: Vec<f64>, error: Vec<f64>) {
        self.amplitudes.push(amplitude);
        self.errors.push(error);
        while self.amplitudes.len() > self.capacity {
            self.amplitudes.remove(0);
            self.errors.remove(0);
        }
    }

    pub(crate) fn extrapolate(&self) -> Vec<f64> {
        let k = self.amplitudes.len();
        let latest = self.amplitudes[k - 1].clone();
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

        let mut out = vec![0.0; latest.len()];
        for (i, &ci) in coeffs.iter().take(k).enumerate() {
            for (dst, &src) in out.iter_mut().zip(&self.amplitudes[i]) {
                *dst += ci * src;
            }
        }
        out
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
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
