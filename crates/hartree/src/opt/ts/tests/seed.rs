//! The reaction-coordinate seed (`TsOptions::reaction_mode_seed`): on the first
//! step P-RFO follows the Hessian mode of maximum overlap with the seed rather
//! than the softest mode, so it climbs the right coordinate even when a softer
//! spectator mode would otherwise capture `follow_mode = 0`.

use super::*;
use crate::ext::kabsch::kabsch_rmsd;
use crate::opt::ts::{TsError, TsOptions, TsStatus, find_transition_state};

/// The spectator-vs-reaction anharmonic surface: a double well along `w0` (the
/// reaction coordinate, curvature `-a` at the saddle) with two harmonic modes; the
/// `w1` spectator is deliberately *soft* (`k2 = 0.2`).
fn anharmonic(x_ref: &[[f64; 3]], basis: &[Vec<f64>]) -> Anharmonic {
    Anharmonic {
        x_ref: x_ref.to_vec(),
        w: basis.to_vec(),
        a: 0.5,
        b: 1.0,
        k2: 0.2,
        k3: 0.9,
    }
}

/// With no seed, `follow_mode = 0` chases the soft spectator bend and never reaches
/// the saddle; supplying the reaction-coordinate seed (the `w0` direction) anchors
/// the climb and converges to the correct single-imaginary saddle. The two runs
/// differ *only* in the seed.
#[test]
fn seed_recovers_saddle_a_spectator_softest_guess_misses() {
    // Start on the inner wall of the double well at q0 = 0.55, where w0's curvature
    // has gone convex (-0.5 + 3·0.55² ≈ 0.41 > k2 = 0.2), so the *softest* mode at
    // the guess is the spectator w1, not the reaction coordinate.
    let x_ref = h3_positions();
    let basis = internal_basis(&x_ref);
    let mut start = x_ref.clone();
    for a in 0..3 {
        for c in 0..3 {
            let i = 3 * a + c;
            start[a][c] += 0.55 * basis[0][i] + 0.10 * basis[1][i];
        }
    }
    let mol = h3_molecule(&start);

    let mut base = TsOptions::default();
    base.recalc_hessian = 5;

    // No seed: the softest mode at the guess is the spectator w1, so the search
    // follows it and fails to reach the saddle.
    let mut surf = anharmonic(&x_ref, &basis);
    let no_seed = find_transition_state(&mol, &mut surf, &base, None).unwrap();
    assert_ne!(
        no_seed.status,
        TsStatus::Converged,
        "follow_mode=0 unexpectedly reached the saddle after {} iters",
        no_seed.iterations
    );

    // Seed along the reaction coordinate (w0, already a Cartesian direction).
    let seed: Vec<[f64; 3]> = (0..mol.len())
        .map(|a| [basis[0][3 * a], basis[0][3 * a + 1], basis[0][3 * a + 2]])
        .collect();
    let mut seeded = base.clone();
    seeded.reaction_mode_seed = Some(seed);

    let mut surf = anharmonic(&x_ref, &basis);
    let with_seed = find_transition_state(&mol, &mut surf, &seeded, None).unwrap();
    assert_eq!(
        with_seed.status,
        TsStatus::Converged,
        "seeded search status {:?} after {} iters",
        with_seed.status,
        with_seed.iterations
    );
    let rmsd = kabsch_rmsd(&with_seed.positions, &x_ref).unwrap();
    assert!(rmsd < 1e-3, "RMSD to saddle = {rmsd:e}");
    assert_eq!(
        with_seed.verification.unwrap().negative_eigenvalues.len(),
        1
    );
}

/// A seed whose length does not match the molecule cannot be a reaction
/// coordinate; the driver rejects it up front as a bad initial guess (before any
/// surface evaluation), rather than silently ignoring it.
#[test]
fn wrong_length_seed_is_bad_initial_guess() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut surf = Quadratic { x0: x0.clone(), h };
    let mut opts = TsOptions::default();
    // One atom direction, but the molecule has three atoms.
    opts.reaction_mode_seed = Some(vec![[1.0, 0.0, 0.0]]);
    let err = find_transition_state(&h3_molecule(&x0), &mut surf, &opts, None).unwrap_err();
    assert!(matches!(err, TsError::BadInitialGuess(_)), "got {err:?}");
}

/// Backward compatibility: a `TsOptions` serialized before `reaction_mode_seed`
/// existed (no such key) still deserializes, defaulting the field to `None`; and a
/// seed that is set round-trips intact.
#[test]
fn options_round_trip_defaults_reaction_mode_seed() {
    let opts = TsOptions::default();
    assert!(opts.reaction_mode_seed.is_none());
    let json = serde_json::to_string(&opts).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value.as_object_mut().unwrap().remove("reaction_mode_seed");
    let legacy: TsOptions = serde_json::from_value(value).unwrap();
    assert!(legacy.reaction_mode_seed.is_none());

    let mut set = TsOptions::default();
    set.reaction_mode_seed = Some(vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let back: TsOptions = serde_json::from_str(&serde_json::to_string(&set).unwrap()).unwrap();
    assert_eq!(back.reaction_mode_seed, set.reaction_mode_seed);
}
