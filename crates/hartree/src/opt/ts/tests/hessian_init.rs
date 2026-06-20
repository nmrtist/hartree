//! The initial-Hessian seam ([`Surface::seed_hessian`] + `TsOptions::hessian_init`):
//! under [`Auto`](crate::opt::ts::HessianInit::Auto) the climb starts from a
//! surface-provided model Hessian, skipping the initial finite-difference build;
//! under [`Fd`](crate::opt::ts::HessianInit::Fd) it always finite-differences.

use super::*;
use crate::opt::ts::{HessianInit, TsError, TsOptions, TsStatus, find_transition_state};
use std::cell::Cell;

/// Which seed the test surface offers through [`Surface::seed_hessian`].
enum SeedKind {
    /// Offer nothing (the default surface behaviour).
    None,
    /// Offer the exact Hessian — equal to the finite-difference one for this quadratic
    /// surface, so seeding changes only the gradient count, not the path.
    Exact,
    /// Offer a malformed (wrong-length) Hessian.
    WrongSize,
}

/// A quadratic saddle that can hand the optimizer a seed Hessian and counts gradient
/// evaluations (each finite-difference Hessian column is one such call).
struct Seeded {
    inner: Quadratic,
    seed: Option<Vec<f64>>,
    grad_calls: Cell<usize>,
}
impl Surface for Seeded {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        self.inner.energy(x)
    }
    fn analytic_gradient(&mut self, x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        self.grad_calls.set(self.grad_calls.get() + 1);
        self.inner.analytic_gradient(x)
    }
    fn seed_hessian(&mut self, _x: &[[f64; 3]]) -> Option<Result<Vec<f64>, OptError>> {
        self.seed.clone().map(Ok)
    }
}

/// Run the same quadratic-saddle search under a given initial-Hessian policy and seed,
/// returning the status and the gradient-evaluation count.
fn run(hessian_init: HessianInit, seed_kind: SeedKind) -> Result<(TsStatus, usize), TsError> {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.06 * basis[0][i] + 0.04 * basis[1][i];
        }
    }
    let seed = match seed_kind {
        SeedKind::None => None,
        SeedKind::Exact => Some(h.clone()),
        SeedKind::WrongSize => Some(vec![0.0; 4]),
    };
    let mut surf = Seeded {
        inner: Quadratic { x0: x0.clone(), h },
        seed,
        grad_calls: Cell::new(0),
    };
    let mut opts = TsOptions::default();
    opts.hessian_init = hessian_init;
    let r = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None)?;
    Ok((r.status, surf.grad_calls.get()))
}

/// Auto uses the surface's seed Hessian, skipping the initial finite-difference build
/// (2·ndof = 18 gradient calls for three atoms), and still reaches the saddle.
#[test]
fn auto_uses_the_seed_hessian_and_skips_the_initial_fd_build() {
    let (unseeded_status, unseeded_calls) = run(HessianInit::Auto, SeedKind::None).unwrap();
    let (seeded_status, seeded_calls) = run(HessianInit::Auto, SeedKind::Exact).unwrap();
    assert_eq!(unseeded_status, TsStatus::Converged);
    assert_eq!(seeded_status, TsStatus::Converged);
    assert!(
        seeded_calls < unseeded_calls,
        "seeded {seeded_calls} should be < unseeded {unseeded_calls}"
    );
    assert_eq!(unseeded_calls - seeded_calls, 18);
}

/// `Fd` ignores the seed and finite-differences the initial Hessian anyway.
#[test]
fn fd_mode_ignores_the_seed() {
    let (auto_status, auto_calls) = run(HessianInit::Auto, SeedKind::Exact).unwrap();
    let (fd_status, fd_calls) = run(HessianInit::Fd, SeedKind::Exact).unwrap();
    assert_eq!(auto_status, TsStatus::Converged);
    assert_eq!(fd_status, TsStatus::Converged);
    assert_eq!(fd_calls - auto_calls, 18);
}

/// A wrong-length seed is a surface-contract violation, surfaced as a numerical error
/// rather than silently ignored.
#[test]
fn wrong_size_seed_is_numerical_error() {
    let err = run(HessianInit::Auto, SeedKind::WrongSize).unwrap_err();
    assert!(matches!(err, TsError::Numerical(_)), "got {err:?}");
}

/// Backward compatibility: a `TsOptions` serialized before `hessian_init` existed
/// deserializes with the default `Auto`.
#[test]
fn options_round_trip_defaults_hessian_init() {
    let opts = TsOptions::default();
    assert_eq!(opts.hessian_init, HessianInit::Auto);
    let json = serde_json::to_string(&opts).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value.as_object_mut().unwrap().remove("hessian_init");
    let legacy: TsOptions = serde_json::from_value(value).unwrap();
    assert_eq!(legacy.hessian_init, HessianInit::Auto);
}
