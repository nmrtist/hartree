//! Wrong-negative-eigenvalue recovery: when a climb settles on a higher-order
//! saddle, re-seeding along the reaction coordinate and re-climbing reaches the
//! first-order saddle. Recovery is gated on a reaction-coordinate seed and the
//! `max_recover` budget.

use super::*;
use crate::opt::ts::{TsOptions, TsStatus, find_transition_state};

/// Two double wells — a reaction coordinate `w0` and a *spurious* coordinate `w1` —
/// plus a stiff harmonic `w2`. The origin is a SECOND-order saddle (`w0` and `w1`
/// both negative there); the first-order saddles sit at the `w1` well minima
/// (`q(w1) = ±√(a2/b2)`), where `w1` has become a minimum direction.
struct DoubleDoubleWell {
    x_ref: Vec<[f64; 3]>,
    w: Vec<Vec<f64>>,
    a1: f64,
    b1: f64,
    a2: f64,
    b2: f64,
    k3: f64,
}
impl DoubleDoubleWell {
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
impl Surface for DoubleDoubleWell {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        let q1 = self.q(x, 0);
        let q2 = self.q(x, 1);
        let q3 = self.q(x, 2);
        Ok(
            -0.5 * self.a1 * q1 * q1 + 0.25 * self.b1 * q1.powi(4) - 0.5 * self.a2 * q2 * q2
                + 0.25 * self.b2 * q2.powi(4)
                + 0.5 * self.k3 * q3 * q3,
        )
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        let q1 = self.q(x, 0);
        let q2 = self.q(x, 1);
        let q3 = self.q(x, 2);
        let dq1 = -self.a1 * q1 + self.b1 * q1.powi(3);
        let dq2 = -self.a2 * q2 + self.b2 * q2.powi(3);
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

/// Start displaced along `w0` only, with `q(w1) = 0` exactly: the first climb
/// minimizes `w1` with no gradient there, so it settles on the origin second-order
/// saddle. Without recovery that is the (wrong) answer; with recovery the search
/// re-seeds off it — descending the spurious `w1` while climbing `w0` — and reaches
/// a genuine first-order saddle.
#[test]
fn recovery_reaches_first_order_saddle_from_second_order() {
    let x_ref = h3_positions();
    let basis = internal_basis(&x_ref);
    let mut start = x_ref.clone();
    for a in 0..3 {
        for c in 0..3 {
            start[a][c] += 0.15 * basis[0][3 * a + c];
        }
    }
    let seed: Vec<[f64; 3]> = (0..x_ref.len())
        .map(|a| [basis[0][3 * a], basis[0][3 * a + 1], basis[0][3 * a + 2]])
        .collect();
    let make = || DoubleDoubleWell {
        x_ref: x_ref.clone(),
        w: basis.clone(),
        a1: 0.5,
        b1: 1.0,
        a2: 0.4,
        b2: 1.0,
        k3: 0.9,
    };

    // No recovery budget: the seeded climb lands on the origin second-order saddle.
    let mut no_recover = TsOptions::default();
    no_recover.recalc_hessian = 5;
    no_recover.reaction_mode_seed = Some(seed.clone());
    no_recover.max_recover = 0;
    let mut surf = make();
    let r0 = find_transition_state(&h3_molecule(&start), &mut surf, &no_recover, None).unwrap();
    assert_eq!(r0.status, TsStatus::WrongImaginaryModeCount);
    assert_eq!(r0.verification.unwrap().negative_eigenvalues.len(), 2);

    // With recovery (default budget): re-seeding off the second-order saddle and
    // re-climbing reaches a first-order saddle.
    let mut recover = no_recover.clone();
    recover.max_recover = 2;
    let mut surf = make();
    let r = find_transition_state(&h3_molecule(&start), &mut surf, &recover, None).unwrap();
    assert_eq!(
        r.status,
        TsStatus::Converged,
        "recovery did not reach a first-order saddle ({:?} after {} iters)",
        r.status,
        r.iterations
    );
    assert_eq!(r.verification.unwrap().negative_eigenvalues.len(), 1);
}

/// Recovery needs a reaction-coordinate seed: with a budget but no seed it is inert
/// (it must not fall back to the softest mode), so the search still reports the
/// wrong mode count.
#[test]
fn recovery_without_seed_is_inert() {
    let x_ref = h3_positions();
    let basis = internal_basis(&x_ref);
    let mut start = x_ref.clone();
    for a in 0..3 {
        for c in 0..3 {
            start[a][c] += 0.15 * basis[0][3 * a + c];
        }
    }
    let mut opts = TsOptions::default();
    opts.recalc_hessian = 5;
    opts.max_recover = 2; // budget present, but no seed
    let mut surf = DoubleDoubleWell {
        x_ref: x_ref.clone(),
        w: basis,
        a1: 0.5,
        b1: 1.0,
        a2: 0.4,
        b2: 1.0,
        k3: 0.9,
    };
    let r = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(r.status, TsStatus::WrongImaginaryModeCount);
}

/// Backward compatibility: a `TsOptions` serialized before `max_recover` existed
/// still deserializes, defaulting to `2`.
#[test]
fn options_round_trip_defaults_max_recover() {
    let opts = TsOptions::default();
    assert_eq!(opts.max_recover, 2);
    let json = serde_json::to_string(&opts).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value.as_object_mut().unwrap().remove("max_recover");
    let legacy: TsOptions = serde_json::from_value(value).unwrap();
    assert_eq!(legacy.max_recover, 2);
}
