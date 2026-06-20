//! Tests for the redundant-internal-coordinate minimizer ([`super::optimize`]),
//! split out of `mod.rs` to keep that file under the line cap. The harmonic
//! analytic surfaces and the real-`HfSurface` SCF-failure check live here.

use super::*;
use crate::core::{Atom, Element};

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn ang(i: [f64; 3], k: [f64; 3], j: [f64; 3]) -> f64 {
    let u = [i[0] - k[0], i[1] - k[1], i[2] - k[2]];
    let v = [j[0] - k[0], j[1] - k[1], j[2] - k[2]];
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv))
        .clamp(-1.0, 1.0)
        .acos()
}

struct HarmonicDiatomic {
    k: f64,
    r0: f64,
}
impl Surface for HarmonicDiatomic {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        let r = dist(x[0], x[1]);
        Ok(0.5 * self.k * (r - self.r0).powi(2))
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        let r = dist(x[0], x[1]);
        let e = [
            (x[0][0] - x[1][0]) / r,
            (x[0][1] - x[1][1]) / r,
            (x[0][2] - x[1][2]) / r,
        ];
        let f = self.k * (r - self.r0);
        Some(Ok(vec![
            [f * e[0], f * e[1], f * e[2]],
            [-f * e[0], -f * e[1], -f * e[2]],
        ]))
    }
}

struct HarmonicWater {
    kb: f64,
    b0: f64,
    ka: f64,
    a0: f64,
}
impl Surface for HarmonicWater {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        let r01 = dist(x[0], x[1]);
        let r02 = dist(x[0], x[2]);
        let th = ang(x[1], x[0], x[2]);
        Ok(
            0.5 * self.kb * ((r01 - self.b0).powi(2) + (r02 - self.b0).powi(2))
                + 0.5 * self.ka * (th - self.a0).powi(2),
        )
    }
    fn analytic_gradient(&mut self, _x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        None
    }
}

#[test]
fn diatomic_harmonic_analytic() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 1.10]),
        ],
        0,
        1,
    );
    let mut surf = HarmonicDiatomic { k: 0.5, r0: 1.40 };
    let result = optimize(&mol, &mut surf, &OptOptions::default()).unwrap();
    assert!(result.converged, "diatomic did not converge");
    let r = dist(result.positions[0], result.positions[1]);
    assert!((r - 1.40).abs() < 1e-5, "optimized r = {r}, want 1.40");
}

#[test]
fn triatomic_harmonic_fd() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [1.70, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [-0.45, 1.70, 0.0]),
        ],
        0,
        1,
    );
    let mut surf = HarmonicWater {
        kb: 0.5,
        b0: 1.81,
        ka: 0.2,
        a0: 1.823, // ~104.5°
    };
    let result = optimize(&mol, &mut surf, &OptOptions::default()).unwrap();
    assert!(
        result.converged,
        "triatomic did not converge in {} steps",
        result.iterations
    );
    let r01 = dist(result.positions[0], result.positions[1]);
    let r02 = dist(result.positions[0], result.positions[2]);
    let th = ang(
        result.positions[1],
        result.positions[0],
        result.positions[2],
    );
    assert!((r01 - 1.81).abs() < 1e-4, "r01 = {r01}");
    assert!((r02 - 1.81).abs() < 1e-4, "r02 = {r02}");
    assert!((th - 1.823).abs() < 1e-4, "theta = {th}");
}

/// A SCF non-convergence on the real `HfSurface` propagates out of `optimize`
/// as the typed `OptError::ScfNotConverged` (via the first `surface.energy`
/// call), not a prose `Evaluation` string. Water/sto-3g cannot converge in one
/// SCF iteration, so `set_scf_max_iter(1)` forces the failure.
#[test]
fn scf_non_convergence_propagates_through_optimize() {
    use crate::scf::Reference;
    use crate::surface::HfSurface;

    let mol =
        Molecule::from_xyz("3\nwater\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n").unwrap();
    let mut surface = HfSurface::new(&mol, "sto-3g", Reference::Rhf).unwrap();
    surface.set_scf_max_iter(1);

    let err = optimize(&mol, &mut surface, &OptOptions::default()).unwrap_err();
    assert!(
        matches!(err, OptError::ScfNotConverged { iterations: 1 }),
        "expected ScfNotConverged {{ iterations: 1 }}, got {err:?}"
    );
}
