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

        // Near convergence the error overlaps B_ij = <e_i, e_j> shrink quadratically,
        // so the raw augmented system becomes badly ill-conditioned exactly in the tail
        // and the linear solve can fail or lose accuracy. Solving over a window of the
        // most-recent vectors, dropping the oldest whenever the solve is singular, keeps
        // DIIS extrapolating to tight tolerances instead of falling back to plain,
        // un-accelerated SCF steps (which converge only linearly on near-degenerate
        // systems such as heavy/lanthanide ECP atoms).
        for start in 0..(k - 1) {
            if let Some(fock) = self.solve_window(start) {
                return fock;
            }
        }
        latest
    }

    /// Solve the C-DIIS coefficients over the window `errors[start..]` (the most-recent
    /// `k - start` vectors) and return the extrapolated Fock, or `None` if the system is
    /// singular so the caller can retry over a smaller window.
    fn solve_window(&self, start: usize) -> Option<Vec<f64>> {
        let m = self.errors.len() - start;
        let dim = m + 1;

        // Error-overlap block, scaled by its largest magnitude. Scaling the objective
        // B -> B / s_max leaves the minimiser (the coefficients c_i, constrained to sum
        // to 1) unchanged while greatly improving the conditioning of the solve.
        let mut block = vec![0.0; m * m];
        let mut s_max = 0.0_f64;
        for a in 0..m {
            for c in 0..m {
                let v = dot(&self.errors[start + a], &self.errors[start + c]);
                block[a * m + c] = v;
                s_max = s_max.max(v.abs());
            }
        }
        let scale = if s_max > 0.0 { 1.0 / s_max } else { 1.0 };

        let mut b = vec![0.0; dim * dim];
        for a in 0..m {
            for c in 0..m {
                b[a * dim + c] = block[a * m + c] * scale;
            }
            b[a * dim + m] = -1.0;
            b[m * dim + a] = -1.0;
        }
        let mut rhs = vec![0.0; dim];
        rhs[m] = -1.0;

        let coeffs = solve_linear(b, rhs, dim)?;
        if coeffs.iter().take(m).any(|c| !c.is_finite()) {
            return None;
        }

        let mut fock = vec![0.0; self.focks[start].len()];
        for (a, &ca) in coeffs.iter().take(m).enumerate() {
            for (dst, &src) in fock.iter_mut().zip(&self.focks[start + a]) {
                *dst += ca * src;
            }
        }
        Some(fock)
    }
}
