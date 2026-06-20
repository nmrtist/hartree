//! Shared analytic-surface fixtures for the transition-state tests.
//!
//! The submodules group the tests by theme; everything they share — the H3
//! geometries, the internal-coordinate basis builder, and the analytic
//! [`Surface`] implementations (a quadratic saddle and an anharmonic double well)
//! — lives here. Children see these private items through the module hierarchy.

use super::numerics::{gram_schmidt, trans_rot_vectors};
use crate::core::{Atom, Element, Molecule};
use crate::opt::{OptError, Surface};

mod convergence;
mod dimer;
mod irc;
mod linear;
mod robustness;
mod saddle;

/// A bent, non-collinear arrangement of three identical atoms (Bohr). Equal
/// masses make mass-weighting a uniform scale, so the mass-weighted Hessian
/// eigenvectors coincide with the Cartesian ones — which keeps the analytic
/// expectations below exact.
fn h3_positions() -> Vec<[f64; 3]> {
    vec![[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 1.8, 0.0]]
}

fn h3_molecule(x: &[[f64; 3]]) -> Molecule {
    let atoms = x
        .iter()
        .map(|&p| Atom::new(Element::from_z(1).unwrap(), p))
        .collect();
    Molecule::new(atoms, 0, 2)
}

/// An orthonormal basis of the internal subspace (orthogonal to the
/// translation/rotation modes). Returns `ndof - n_tr` vectors of length `ndof`,
/// where `n_tr` is the number of surviving trans/rot modes (6 bent, 5 linear).
fn internal_basis(x: &[[f64; 3]]) -> Vec<Vec<f64>> {
    let unit = vec![1.0; x.len()];
    let tr = gram_schmidt(&trans_rot_vectors(x, &unit));
    let ndof = 3 * x.len();
    let n_internal = ndof - tr.len();
    let mut internal: Vec<Vec<f64>> = Vec::new();
    for i in 0..ndof {
        let mut e = vec![0.0f64; ndof];
        e[i] = 1.0;
        for v in tr.iter().chain(internal.iter()) {
            let p: f64 = e.iter().zip(v).map(|(a, b)| a * b).sum();
            for (ek, &vk) in e.iter_mut().zip(v) {
                *ek -= p * vk;
            }
        }
        let n: f64 = e.iter().map(|a| a * a).sum::<f64>().sqrt();
        if n > 1e-6 {
            for ek in &mut e {
                *ek /= n;
            }
            internal.push(e);
        }
        if internal.len() == n_internal {
            break;
        }
    }
    internal
}

/// A constant Cartesian Hessian H = sum_k curv_k w_k w_k^T over internal
/// directions: eigenvalues curv_k on the internal space, 0 on trans/rot.
fn hessian_from(internal: &[Vec<f64>], curv: &[f64]) -> Vec<f64> {
    let ndof = internal[0].len();
    let mut h = vec![0.0f64; ndof * ndof];
    for (w, &c) in internal.iter().zip(curv) {
        for i in 0..ndof {
            for j in 0..ndof {
                h[i * ndof + j] += c * w[i] * w[j];
            }
        }
    }
    h
}

/// Exact quadratic surface E = 1/2 (x-x0)^T H (x-x0) with analytic gradient. The
/// saddle is at `x0`.
struct Quadratic {
    x0: Vec<[f64; 3]>,
    h: Vec<f64>,
}
impl Quadratic {
    fn dx(&self, x: &[[f64; 3]]) -> Vec<f64> {
        let mut d = Vec::with_capacity(3 * x.len());
        for (a, b) in x.iter().zip(&self.x0) {
            d.push(a[0] - b[0]);
            d.push(a[1] - b[1]);
            d.push(a[2] - b[2]);
        }
        d
    }
}
impl Surface for Quadratic {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        let dx = self.dx(x);
        let n = dx.len();
        let mut e = 0.0;
        for i in 0..n {
            for j in 0..n {
                e += 0.5 * dx[i] * self.h[i * n + j] * dx[j];
            }
        }
        Ok(e)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        let dx = self.dx(x);
        let n = dx.len();
        let g: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| self.h[i * n + j] * dx[j]).sum())
            .collect();
        Some(Ok((0..n / 3)
            .map(|a| [g[3 * a], g[3 * a + 1], g[3 * a + 2]])
            .collect()))
    }
}

/// An anharmonic saddle: a quartic double well along internal direction w1
/// (a maximum at the origin) and harmonic minima along w2, w3. The saddle is at
/// `x_ref`.
struct Anharmonic {
    x_ref: Vec<[f64; 3]>,
    w: Vec<Vec<f64>>,
    a: f64,
    b: f64,
    k2: f64,
    k3: f64,
}
impl Anharmonic {
    fn q(&self, x: &[[f64; 3]], k: usize) -> f64 {
        let mut s = 0.0;
        for a in 0..x.len() {
            for c in 0..3 {
                s += self.w[k][3 * a + c] * (x[a][c] - self.x_ref[a][c]);
            }
        }
        s
    }
}
impl Surface for Anharmonic {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        let q1 = self.q(x, 0);
        let q2 = self.q(x, 1);
        let q3 = self.q(x, 2);
        Ok(-0.5 * self.a * q1 * q1
            + 0.25 * self.b * q1.powi(4)
            + 0.5 * self.k2 * q2 * q2
            + 0.5 * self.k3 * q3 * q3)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        let q1 = self.q(x, 0);
        let q2 = self.q(x, 1);
        let q3 = self.q(x, 2);
        let dq1 = -self.a * q1 + self.b * q1.powi(3);
        let dq2 = self.k2 * q2;
        let dq3 = self.k3 * q3;
        let n = 3 * x.len();
        let g: Vec<f64> = (0..n)
            .map(|i| dq1 * self.w[0][i] + dq2 * self.w[1][i] + dq3 * self.w[2][i])
            .collect();
        Some(Ok((0..x.len())
            .map(|a| [g[3 * a], g[3 * a + 1], g[3 * a + 2]])
            .collect()))
    }
}

fn mode_overlap(reaction: &[[f64; 3]], w: &[f64]) -> f64 {
    let mut flat = Vec::new();
    for a in reaction {
        flat.extend_from_slice(a);
    }
    flat.iter().zip(w).map(|(a, b)| a * b).sum::<f64>().abs()
}
