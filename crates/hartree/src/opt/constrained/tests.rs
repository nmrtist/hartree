//! Tests for the constrained internal-coordinate minimizer ([`super::optimize_constrained`]):
//! analytic surfaces cheap enough to run in debug, checking that a held coordinate stays at
//! its target to strict tolerance while the free coordinates relax to their unconstrained
//! minima.

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

/// A bent triatomic (central atom 0 bonded to atoms 1 and 2) with independent harmonic
/// stretches on the two O–H bonds and a harmonic bend about the central atom, each toward
/// its own equilibrium. The free minimum sits at (`r0`, `r0`, `theta0`); constraining one
/// stretch (or the angle) pins it elsewhere while the rest relax.
struct Triatomic {
    k_bond: f64,
    r0: f64,
    k_ang: f64,
    theta0: f64,
}
impl Surface for Triatomic {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        let r01 = dist(x[0], x[1]);
        let r02 = dist(x[0], x[2]);
        let t = ang(x[1], x[0], x[2]);
        Ok(0.5 * self.k_bond * (r01 - self.r0).powi(2)
            + 0.5 * self.k_bond * (r02 - self.r0).powi(2)
            + 0.5 * self.k_ang * (t - self.theta0).powi(2))
    }
    fn analytic_gradient(&mut self, _x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        None
    }
}

fn bent_triatomic() -> Molecule {
    // O at the centre bonded to two H's, deliberately off equilibrium so the optimizer
    // has to move both bonds and the angle.
    Molecule::new(
        vec![
            Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [1.7, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [-0.45, 1.70, 0.0]),
        ],
        0,
        1,
    )
}

#[test]
fn held_bond_stays_at_target_while_others_relax() {
    let mut surface = Triatomic {
        k_bond: 0.6,
        r0: 1.9,
        k_ang: 0.25,
        theta0: 1.9, // ~109°
    };
    let mol = bent_triatomic();
    // Hold the 0–1 bond stretched well off its 1.9 equilibrium.
    let target = 2.6;
    let constraints = [Constraint {
        coordinate: Internal::Bond(0, 1),
        target,
    }];
    let opts = OptOptions::default();
    let res = optimize_constrained(&mol, &mut surface, &constraints, &opts).expect("optimize");
    assert!(res.converged, "did not converge");

    let p = &res.positions;
    // Strict constraint satisfaction.
    let held = dist(p[0], p[1]);
    assert!(
        (held - target).abs() < 1e-6,
        "held bond {held} != target {target}"
    );
    // The free bond and the free angle relax to their unconstrained equilibria.
    let free_bond = dist(p[0], p[2]);
    assert!(
        (free_bond - 1.9).abs() < 1e-4,
        "free bond {free_bond} != 1.9"
    );
    let free_ang = ang(p[1], p[0], p[2]);
    assert!(
        (free_ang - 1.9).abs() < 1e-3,
        "free angle {free_ang} != 1.9"
    );
}

#[test]
fn held_angle_stays_at_target_while_bonds_relax() {
    let mut surface = Triatomic {
        k_bond: 0.6,
        r0: 1.9,
        k_ang: 0.25,
        theta0: 1.9,
    };
    let mol = bent_triatomic();
    // Drive the central H–O–H angle to a value away from its 1.9 rad equilibrium.
    let target = 1.4;
    let constraints = [Constraint {
        coordinate: Internal::Angle(1, 0, 2),
        target,
    }];
    let opts = OptOptions::default();
    let res = optimize_constrained(&mol, &mut surface, &constraints, &opts).expect("optimize");
    assert!(res.converged, "did not converge");

    let p = &res.positions;
    let held = ang(p[1], p[0], p[2]);
    assert!(
        (held - target).abs() < 1e-6,
        "held angle {held} != target {target}"
    );
    // Both bonds relax to equilibrium since they do not couple to the angle here.
    assert!(
        (dist(p[0], p[1]) - 1.9).abs() < 1e-4,
        "0-1 bond not relaxed"
    );
    assert!(
        (dist(p[0], p[2]) - 1.9).abs() < 1e-4,
        "0-2 bond not relaxed"
    );
}

#[test]
fn no_constraints_reaches_the_free_minimum() {
    // With an empty constraint set the result is the ordinary minimum (a sanity check
    // that the free-only path degenerates to the unconstrained minimizer's answer).
    let mut surface = Triatomic {
        k_bond: 0.6,
        r0: 1.9,
        k_ang: 0.25,
        theta0: 1.9,
    };
    let mol = bent_triatomic();
    let opts = OptOptions::default();
    let res = optimize_constrained(&mol, &mut surface, &[], &opts).expect("optimize");
    assert!(res.converged);
    let p = &res.positions;
    assert!((dist(p[0], p[1]) - 1.9).abs() < 1e-4);
    assert!((dist(p[0], p[2]) - 1.9).abs() < 1e-4);
    assert!((ang(p[1], p[0], p[2]) - 1.9).abs() < 1e-3);
}
