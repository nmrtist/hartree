//! Crate-internal tests for the transition-state (`--ts`) gating table in
//! [`Job::run`] and for the `ts_ok` contribution to [`JobResult::converged`].
//!
//! Relocated here via `#[path]` from `job.rs`, so this module is still a child
//! of `job` and can see private items and construct the crate-local
//! `#[non_exhaustive]` [`TsResult`] / [`JobResult`] by struct literal.

use super::*;
use crate::core::Molecule;
use crate::dft::FunctionalSpec;
use crate::opt::ts::{TsResult, TsStatus};
use crate::scf::Smearing;

/// Minimal closed-shell H2 so any incidental single point is trivial.
fn h2() -> Molecule {
    Molecule::from_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n").unwrap()
}

/// H2 with a ghost H center appended, for the ghost-atom gate. Mirrors the
/// `Gh(<symbol>)` XYZ syntax used in `tests/ghost.rs`.
fn h2_with_ghost() -> Molecule {
    Molecule::from_xyz("3\nh2+ghost\nH 0 0 0\nH 0 0 0.74\nGh(H) 0 0 3.0\n").unwrap()
}

/// Build a transition-state H2 job, apply `mutate`, and run it. Uses the
/// struct-update form (clean under `clippy::field_reassign_with_default`).
fn ts_job(method: Method, mutate: impl FnOnce(&mut JobOptions)) -> Result<JobResult, String> {
    let mut options = JobOptions {
        transition_state: true,
        ..Default::default()
    };
    mutate(&mut options);
    Job {
        molecule: h2(),
        basis: "sto-3g".into(),
        method,
        options,
    }
    .run()
}

/// Assert the result is `Err` and that the message contains `needle`, so an
/// unrelated incidental error cannot make the case pass spuriously.
fn assert_rejects(result: Result<JobResult, String>, needle: &str, what: &str) {
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("{what}: expected Err, got Ok"),
    };
    assert!(
        err.contains(needle),
        "{what}: message {err:?} does not contain {needle:?}"
    );
}

// ---------------------------------------------------------------------------
// `--ts` rejection cases: each asserts a distinctive substring of the message
// that the intended gate (and only that gate) produces.
// ---------------------------------------------------------------------------

#[test]
fn ts_rejects_post_hf() {
    // TS-branch gate (no analytic CC gradient). Mp2 with TS on.
    assert_rejects(
        ts_job(Method::Mp2, |_| {}),
        "transition-state search is not supported for post-HF",
        "post-HF",
    );
}

#[test]
fn ts_rejects_ri() {
    assert_rejects(
        ts_job(Method::Rhf, |o| o.ri = true),
        "the RI-JK backend does not support transition-state search",
        "--ri",
    );
}

#[test]
fn ts_rejects_direct() {
    assert_rejects(
        ts_job(Method::Rhf, |o| o.direct = true),
        "integral-direct backend does not support transition-state search",
        "--direct",
    );
}

#[test]
fn ts_rejects_cosx() {
    assert_rejects(
        ts_job(Method::Rhf, |o| o.cosx = true),
        "COSX is energy-only: transition-state search is not supported",
        "--cosx",
    );
}

#[test]
fn ts_rejects_x2c() {
    assert_rejects(
        ts_job(Method::Rhf, |o| o.x2c = true),
        "X2C is energy-only: transition-state search is not supported",
        "--x2c",
    );
}

#[test]
fn ts_rejects_smearing() {
    assert_rejects(
        ts_job(Method::Rhf, |o| {
            o.smearing = Some(Smearing::Fermi {
                temperature_k: 5000.0,
            })
        }),
        "Fermi smearing is energy-only: transition-state search is not",
        "--smear",
    );
}

#[test]
fn ts_rejects_implicit_solvent() {
    assert_rejects(
        ts_job(Method::Rhf, |o| o.solvent_eps = Some(78.4)),
        "transition-state search in implicit solvent is not supported",
        "--eps",
    );
}

#[test]
fn ts_rejects_ghost() {
    let options = JobOptions {
        transition_state: true,
        ..Default::default()
    };
    let result = Job {
        molecule: h2_with_ghost(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options,
    }
    .run();
    assert_rejects(
        result,
        "transition-state search with ghost atoms is not supported",
        "ghost",
    );
}

#[test]
fn ts_rejects_vv10() {
    // wB97X-V carries the VV10 nonlocal correlation kernel.
    let method = Method::Dft(FunctionalSpec::parse("wb97x-v").unwrap());
    assert_rejects(
        ts_job(method, |_| {}),
        "is VV10-carrying: transition-state search is not supported",
        "VV10",
    );
}

#[test]
fn ts_rejects_double_hybrid() {
    // B2PLYP is a double hybrid (PT2 step, no analytic gradient).
    let method = Method::Dft(FunctionalSpec::parse("b2plyp").unwrap());
    assert_rejects(
        ts_job(method, |_| {}),
        "is a double hybrid: transition-state search is not supported",
        "double hybrid",
    );
}

#[test]
fn ts_rejects_opt_ts_mutual_exclusion() {
    assert_rejects(
        ts_job(Method::Rhf, |o| o.optimize_geometry = true),
        "geometry optimization and transition-state search are mutually exclusive",
        "--opt/--ts",
    );
}

#[test]
fn ts_rejects_cosmo_file() {
    assert_rejects(
        ts_job(Method::Rhf, |o| o.cosmo_file = Some("dummy.cosmo".into())),
        "COSMO file export cannot be combined with geometry optimization or",
        "--cosmo-file",
    );
}

/// `ts_options` must reach `find_transition_state` unchanged. Capping
/// `ts_options.max_iter` at 1 forces the search to stop with
/// [`TsStatus::NotConverged`] (the library default of 300 would converge on this
/// trivial H2), so observing that status proves the field was threaded through
/// rather than overridden by a hard-coded `TsOptions::default()`. Cheap: a
/// single saddle iteration plus the per-iteration gradient.
#[test]
fn ts_options_algorithm_is_threaded() {
    let mut options = JobOptions {
        transition_state: true,
        ..Default::default()
    };
    options.ts_options.max_iter = 1;
    let result = Job {
        molecule: h2(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options,
    }
    .run()
    .unwrap();
    let ts = result
        .transition_state
        .expect("transition-state result attached");
    assert_eq!(
        ts.status,
        TsStatus::NotConverged,
        "max_iter=1 must force NotConverged, proving ts_options is threaded"
    );
}

// ---------------------------------------------------------------------------
// `JobResult::converged()` reflects the `ts_ok` term.
// ---------------------------------------------------------------------------

/// Build a `TsResult` with the given status and otherwise-empty fields.
/// Crate-internal, so `#[non_exhaustive]` does not block the struct literal.
fn ts_result(status: TsStatus) -> TsResult {
    TsResult {
        positions: Vec::new(),
        energy: 0.0,
        status,
        iterations: 0,
        history: Vec::new(),
        verification: None,
        irc: None,
        diagnostic: None,
    }
}

#[test]
fn converged_reflects_ts_ok() {
    let base = Job {
        molecule: h2(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions::default(),
    }
    .run()
    .unwrap();
    // No TS attached: `ts_ok` is vacuously true.
    assert!(base.scf.converged && base.transition_state.is_none() && base.converged());

    // A converged saddle keeps `converged()` true...
    let mut converged_case = base.clone();
    converged_case.transition_state = Some(ts_result(TsStatus::Converged));
    assert!(converged_case.converged());

    // ...while a non-converged saddle drives it false purely via `ts_ok`.
    let mut nonconverged_case = base.clone();
    nonconverged_case.transition_state = Some(ts_result(TsStatus::NotConverged));
    assert!(!nonconverged_case.converged());

    // A wrong-imaginary-mode-count result is likewise not "converged".
    let mut wrong_modes_case = base;
    wrong_modes_case.transition_state = Some(ts_result(TsStatus::WrongImaginaryModeCount));
    assert!(!wrong_modes_case.converged());
}

/// The CLI-facing flatteners replace a bare SCF non-convergence with an actionable
/// recovery hint, while leaving every other error's own message intact.
#[test]
fn scf_non_convergence_yields_recovery_hint() {
    let opt = OptError::ScfNotConverged { iterations: 1 };
    assert_eq!(opt_error_message(&opt), SCF_RECOVERY_HINT);
    // Other `OptError` cases keep their own prose.
    let other = OptError::Evaluation("basis load failed".into());
    assert_eq!(opt_error_message(&other), other.to_string());

    // The TS flattener unwraps `SurfaceEvaluation(ScfNotConverged)` to the same hint.
    let ts = TsError::SurfaceEvaluation(OptError::ScfNotConverged { iterations: 1 });
    assert_eq!(ts_error_message(&ts), SCF_RECOVERY_HINT);
    // A non-SCF TsError keeps its own message.
    let ts_other = TsError::BadInitialGuess("too few atoms".into());
    assert_eq!(ts_error_message(&ts_other), ts_other.to_string());
}

/// The NEB-TS flattener unwraps an SCF non-convergence from *either* stage to the
/// recovery hint, while leaving other errors' own messages intact.
#[test]
fn neb_ts_flattener_maps_scf_failure_to_hint() {
    use crate::opt::ts::{NebError, NebTsError};
    let scf = || OptError::ScfNotConverged { iterations: 1 };
    let from_ts = NebTsError::Ts(TsError::SurfaceEvaluation(scf()));
    assert_eq!(neb_ts_error_message(&from_ts), SCF_RECOVERY_HINT);
    let from_neb = NebTsError::Neb(NebError::SurfaceEvaluation(scf()));
    assert_eq!(neb_ts_error_message(&from_neb), SCF_RECOVERY_HINT);
    // A non-SCF stage error keeps its own message.
    let other = NebTsError::Neb(NebError::BadEndpoints("ordering".into()));
    assert_eq!(neb_ts_error_message(&other), other.to_string());
}

/// A two-endpoint product without a transition-state search is rejected (the guess
/// would have nowhere to go), with a message naming the missing `--ts`.
#[test]
fn ts_guess_without_transition_state_is_rejected() {
    let options = JobOptions {
        transition_state: false,
        ts_guess: Some(TsGuessInput::new(h2())),
        ..Default::default()
    };
    let result = Job {
        molecule: h2(),
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options,
    }
    .run();
    assert_rejects(
        result,
        "without requesting a transition-state search",
        "ts_guess gate",
    );
}
