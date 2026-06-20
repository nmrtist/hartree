//! Transition-state (saddle-point) search on a [`Surface`].
//!
//! Where [`optimize`](crate::opt::optimize) walks downhill to a minimum, the
//! drivers here climb to a *first-order saddle point*: a geometry that is a
//! maximum along exactly one mode (the reaction coordinate) and a minimum along
//! every other. They consume the same [`Surface`] trait as `optimize` — energy
//! plus analytic gradient, with the finite-difference fallback in
//! [`crate::opt::fd`] — so any energy method that exposes a gradient path can
//! drive a TS search. The surface is built exactly as `Job::run_inner` builds an
//! `HfSurface` for the minimizer (see `crate::surface`); only the optimizer on
//! top differs.
//!
//! Unlike the minimizer, which works in redundant internal coordinates, the
//! saddle search is specified in **mass-weighted Cartesian coordinates**: the
//! "exactly one negative mode" criterion and the reported reaction mode are only
//! well defined after mass-weighting and projecting out the 6 (or 5)
//! translational/rotational modes, the same frame [`crate::props::frequencies`]
//! uses to count imaginary frequencies. Cartesian operation also avoids the
//! minimizer's bonds-and-angles internal set, which (built from a covalent-radius
//! cutoff) may not contain a stretched forming/breaking bond or a torsional
//! reaction coordinate.
//!
//! Convergence is reported through [`TsStatus`], mirroring how
//! [`optimize`](crate::opt::optimize) returns `Ok` with `converged = false`
//! rather than erroring: a run that fails to converge, or that converges to a
//! point with the wrong number of negative modes, still returns an `Ok`
//! [`TsResult`] carrying the best-so-far geometry and the full eigenvalue
//! spectrum (for restart / visualization). [`TsError`] is reserved for genuine
//! compute faults that yield no usable geometry.
//!
//! Like the rest of `opt`, this is pure synchronous numerics: no async runtime,
//! no threads, no global state. Concurrency, cross-thread progress, and
//! cancellation belong to the caller (e.g. silicolab's `JobManager`); the one
//! cross-cutting seam exposed here is the borrowed [`Progress`] observer.
//!
//! The P-RFO driver (in [`prfo`]), the Hessian-free dimer driver (in [`dimer`]),
//! and the shared [`verify_saddle`] check are implemented. The [`guess`] submodule builds the
//! single near-saddle geometry these drivers consume from a reactant + product
//! pair (IDPP interpolation with subgraph atom mapping). This file is the public
//! contract — types, the two entry points, and their docs; the numerics live in
//! [`numerics`], [`prfo`], and the IRC path tracer in [`irc`].
//!
//! For the *double-ended* case — two known minima, no good single guess — the
//! separate climbing-image NEB driver
//! [`find_minimum_energy_path`](self::find_minimum_energy_path) (in `neb`) relaxes a
//! whole band of images onto the minimum-energy path and rides a climbing image up
//! to the saddle, yielding a transition-state guess plus its reaction-coordinate
//! tangent to hand back to [`find_transition_state`] for a tight finish.

pub mod guess;

mod climb;
mod dimer;
mod dimer_rotate;
mod frame;
mod internal_frame;
mod irc;
mod neb;
mod numerics;
mod options;
mod prfo;
mod step;

pub use irc::{IrcEndpoints, IrcMethod};
pub use neb::{
    NebError, NebOptions, NebResult, NebStatus, NebTsError, NebTsResult, find_minimum_energy_path,
    find_transition_state_from_endpoints,
};
pub use numerics::SaddleVerification;
pub use options::{Coordinates, HessianInit, TsAlgorithm, TsOptions, VerifyHessian};
// The analytic-surface tests favour explicit index loops over the atom/Cartesian
// grid and `TsOptions::default()` + field mutation (the documented way to build
// the `#[non_exhaustive]` options); both read clearer here than the lint's
// rewrites, so silence them module-wide.
#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::field_reassign_with_default)]
mod tests;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::Molecule;
use crate::opt::{OptError, OptStep, Surface};

/// Outcome classification for a [`TsResult`].
///
/// A saddle search has three "soft" non-success outcomes that still produce a
/// usable geometry (so they ride on `Ok(TsResult)`, like `optimize`'s
/// `converged: false`, rather than a [`TsError`]): it can run out of iterations,
/// it can be stopped early by an observer, or it can converge geometrically to a
/// point whose Hessian has the wrong number of negative modes (the
/// [`verify_saddle`] check can return 0 — a minimum — or ≥2 — a higher-order
/// saddle). `#[non_exhaustive]`: handle unknown future variants as "not a
/// verified saddle".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TsStatus {
    /// Force/step thresholds met **and** [`verify_saddle`] found exactly one
    /// negative mode: a verified first-order saddle point.
    Converged,
    /// Hit `max_iter` without meeting the force/step thresholds.
    /// [`TsResult::positions`] is the best geometry found, for restart.
    NotConverged,
    /// Geometry converged, but the post-convergence [`verify_saddle`] check found
    /// a number of negative modes other than one (see
    /// [`TsResult::verification`] for the spectrum: empty = minimum, ≥2 =
    /// higher-order saddle).
    WrongImaginaryModeCount,
    /// A [`Progress`] observer returned [`Flow::Stop`] before convergence.
    StoppedEarly,
}

/// Structured outcome of a transition-state search.
///
/// Mirrors [`OptResult`](crate::opt::OptResult) (final geometry, energy,
/// iteration count, and the per-iteration `history`) and adds the TS-specific
/// verification fields. It is a machine-readable record for the downstream agent,
/// not a log string. `#[non_exhaustive]` so further data can be added without
/// breaking downstream consumers.
///
/// The [`status`](TsResult::status) field classifies the outcome; non-success
/// outcomes that still produced a usable geometry are reported here (not as a
/// [`TsError`]) so the geometry and spectrum survive for restart / visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TsResult {
    /// Best geometry at termination (Cartesian, atomic units, atom order matching
    /// the input molecule). For [`TsStatus::Converged`] this is the saddle point;
    /// otherwise it is the best point reached.
    pub positions: Vec<[f64; 3]>,
    /// Energy at `positions`.
    pub energy: f64,
    /// Outcome classification. See [`TsStatus`]; [`TsResult::converged`] is the
    /// convenience predicate for the `Converged` case.
    pub status: TsStatus,
    /// Number of optimization iterations taken.
    pub iterations: usize,
    /// Per-iteration convergence trace, reusing the minimizer's
    /// [`OptStep`] record so the same plotting/early-stop machinery serves both.
    pub history: Vec<OptStep>,
    /// Result of the shared post-convergence [`verify_saddle`] step (negative
    /// modes, reaction-mode eigenvector, imaginary frequency). `Some` whenever
    /// the check ran — i.e. for [`TsStatus::Converged`] (one negative mode) and
    /// [`TsStatus::WrongImaginaryModeCount`] (0 or ≥2). `None` when the run never
    /// reached verification ([`TsStatus::NotConverged`] / [`TsStatus::StoppedEarly`]).
    pub verification: Option<SaddleVerification>,
    /// Optional IRC-endpoint confirmation that the saddle connects two minima;
    /// `None` unless [`TsOptions::confirm_irc`] was set and the trace ran. See
    /// [`IrcEndpoints`].
    pub irc: Option<IrcEndpoints>,
    /// A short, human-readable note on *why* the search stopped without a verified
    /// saddle — e.g. that it exhausted `max_iter`, that the climbing step shrank to
    /// the trust floor, or that the Hessian spectrum was near-degenerate at the
    /// stopping point. `None` for a [`Converged`](TsStatus::Converged) run and
    /// whenever no distinguishing cause is available; the field is purely
    /// diagnostic and never affects [`status`](TsResult::status) or
    /// [`converged`](TsResult::converged). `#[serde(default)]` so records written
    /// before this field existed still deserialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
}

impl TsResult {
    /// `true` iff the search reached a verified first-order saddle
    /// ([`TsStatus::Converged`]).
    pub fn converged(&self) -> bool {
        matches!(self.status, TsStatus::Converged)
    }
}

/// Per-iteration observation hook for a running search.
///
/// [`find_transition_state`] invokes [`Progress::step`] once per accepted
/// iteration with the same [`OptStep`] it appends to [`TsResult::history`]. A GUI
/// (silicolab) can draw a live convergence curve from these; an agent can decide
/// to abort early.
///
/// The hook is a *borrowed trait object* (`Option<&dyn Progress>` on the driver)
/// so the search stays single-threaded and imposes no allocation or ownership on
/// the observer. `step` takes `&self`; an observer that needs to accumulate state
/// uses interior mutability (`Cell`/`RefCell`/atomics/a channel) on its own side
/// — hartree itself spawns no threads. Returning [`Flow::Stop`] asks the driver
/// to halt at the next iteration boundary and return its best result so far with
/// [`TsStatus::StoppedEarly`].
///
/// This is the one cross-cutting parameter committed to now: `find_*` is a
/// published signature, and adding a parameter later would be a breaking change.
pub trait Progress {
    /// Observe one accepted iteration and signal whether to continue.
    fn step(&self, step: &OptStep) -> Flow;
}

/// Control-flow signal returned by a [`Progress`] observer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Keep iterating.
    Continue,
    /// Stop at the next iteration boundary; the driver returns its best result so
    /// far with [`TsStatus::StoppedEarly`].
    Stop,
}

/// Genuine compute faults of a transition-state search — failures that yield no
/// usable geometry. Non-convergence and a wrong negative-mode count are *not*
/// here: they are soft outcomes carried on `Ok(TsResult)` via [`TsStatus`] so the
/// best-so-far geometry survives.
///
/// Unlike the rest of hartree (which surfaces `Result<_, String>`), this is a
/// *typed* enum so the downstream agent can branch on the variant without parsing
/// prose. Surface energy/gradient failures wrap the existing
/// [`OptError`] via `#[from]`. `#[non_exhaustive]` matching the crate's other
/// public error enum (`PeriodicError`), so new cases can be added non-breakingly.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TsError {
    /// A [`Surface`] energy or gradient evaluation failed (transparently wraps the
    /// underlying [`OptError`]). An SCF non-convergence — which TS geometries, with
    /// their small HOMO-LUMO gaps, provoke more readily than minima — now surfaces
    /// as `TsError::SurfaceEvaluation(OptError::ScfNotConverged { .. })`, so a caller
    /// can branch on it (and recover: tighten the SCF, change the guess, raise the
    /// level shift) without parsing prose.
    #[error(transparent)]
    SurfaceEvaluation(#[from] OptError),
    /// A numerical-linear-algebra failure that leaves no usable geometry: the
    /// mass-weighted Hessian's eigendecomposition did not converge, or a
    /// finite-difference Hessian carried a non-finite entry that the driver's one
    /// self-healing recompute could not clear. Carries a human-readable reason.
    #[error("transition-state numerics failed: {0}")]
    Numerical(String),
    /// The starting geometry is unusable as a TS guess (e.g. too few atoms for a
    /// reaction coordinate, no negative curvature to follow, or degenerate
    /// coordinates). Carries a human-readable reason.
    #[error("bad initial guess for transition-state search: {0}")]
    BadInitialGuess(String),
    /// The requested [`TsAlgorithm`] has no implementation yet. P-RFO and the
    /// dimer method are implemented; this guards future variants (e.g. a
    /// single-geometry NEB flavour). Returned (rather than panicking) so a caller
    /// that selects an unimplemented algorithm gets a branchable, recoverable error.
    #[error("transition-state algorithm {0:?} is not yet implemented")]
    AlgorithmNotImplemented(TsAlgorithm),
}

/// Verify that `positions` is a first-order saddle — the post-convergence step
/// shared by every [`TsAlgorithm`].
///
/// Diagonalizes the mass-weighted, translation/rotation-projected Cartesian
/// Hessian at `positions` (built by finite-differencing `surface`'s gradient with
/// `options.fd_step`, reusing the [`crate::props::frequencies`] frame) and reports
/// the negative-mode spectrum as a [`SaddleVerification`]. P-RFO already carries a
/// working Hessian into convergence; the dimer method, which never forms a full
/// Hessian, pays one extra Hessian evaluation here. [`find_transition_state`] maps
/// the result onto [`TsStatus`]: exactly one negative mode ⇒
/// [`Converged`](TsStatus::Converged); zero or ≥2 ⇒
/// [`WrongImaginaryModeCount`](TsStatus::WrongImaginaryModeCount) — both `Ok`
/// outcomes. Exposed so a caller (or agent) can independently verify an
/// externally supplied TS guess.
///
/// # Errors
/// [`TsError::SurfaceEvaluation`] if a finite-difference energy/gradient
/// evaluation fails while building the Hessian, or [`TsError::Numerical`] if the
/// mass-weighted Hessian carries a non-finite entry or its eigendecomposition
/// fails to converge.
pub fn verify_saddle<S: Surface>(
    molecule: &Molecule,
    surface: &mut S,
    positions: &[[f64; 3]],
    options: &TsOptions,
) -> Result<SaddleVerification, TsError> {
    Ok(verify_with_hessian(molecule, surface, positions, options)?.0)
}

/// Like [`verify_saddle`], but also returns the finite-difference Cartesian Hessian
/// it built. The drivers reuse that Hessian for a Hessian-corrector IRC trace at the
/// same geometry instead of recomputing it (the spectrum in the returned
/// [`SaddleVerification`] already came from it), turning two saddle Hessians into one.
pub(crate) fn verify_with_hessian<S: Surface>(
    molecule: &Molecule,
    surface: &mut S,
    positions: &[[f64; 3]],
    options: &TsOptions,
) -> Result<(SaddleVerification, Vec<f64>), TsError> {
    let hessian = numerics::fd_hessian(surface, positions, options.fd_step)?;
    let mol = numerics::with_positions(molecule, positions);
    let verification = numerics::saddle_from_hessian(&mol, &hessian, options.negative_mode_tol)
        .map_err(TsError::Numerical)?;
    Ok((verification, hessian))
}

/// The P-RFO post-convergence verification, honoring
/// [`TsOptions::verify_hessian`](TsOptions::verify_hessian): under
/// [`Strict`](VerifyHessian::Strict) it finite-differences a fresh Hessian (this is
/// [`verify_with_hessian`]); under [`Maintained`](VerifyHessian::Maintained) it
/// classifies from the `maintained` quasi-Newton Hessian P-RFO carried into
/// convergence; under [`Auto`](VerifyHessian::Auto) it classifies from `maintained`
/// but falls back to a fresh Hessian when the spectrum is ambiguous near the
/// negative-mode threshold. Returns the verification and the Hessian it was drawn
/// from (which the driver reuses for a Hessian-corrector IRC and for recovery).
pub(super) fn verify_classified<S: Surface>(
    molecule: &Molecule,
    surface: &mut S,
    positions: &[[f64; 3]],
    maintained: &[f64],
    options: &TsOptions,
) -> Result<(SaddleVerification, Vec<f64>), TsError> {
    let from_maintained = || -> Result<SaddleVerification, TsError> {
        let mol = numerics::with_positions(molecule, positions);
        numerics::saddle_from_hessian(&mol, maintained, options.negative_mode_tol)
            .map_err(TsError::Numerical)
    };
    match options.verify_hessian {
        VerifyHessian::Strict => verify_with_hessian(molecule, surface, positions, options),
        VerifyHessian::Maintained => Ok((from_maintained()?, maintained.to_vec())),
        VerifyHessian::Auto => {
            let verification = from_maintained()?;
            if numerics::spectrum_ambiguous(&verification.eigenvalues, options.negative_mode_tol) {
                verify_with_hessian(molecule, surface, positions, options)
            } else {
                Ok((verification, maintained.to_vec()))
            }
        }
    }
}

/// Search for a first-order saddle point on `surface`, starting from `molecule`.
///
/// The TS analogue of [`optimize`](crate::opt::optimize): it consumes the same
/// [`Surface`] (energy + analytic gradient, finite-difference fallback) and
/// returns a structured [`TsResult`] whose [`TsStatus`] classifies the outcome.
/// `options.algorithm` selects the method. The optional `progress` observer is
/// called once per accepted iteration and may request an early stop (see
/// [`Progress`]). After geometric convergence the driver runs the shared
/// [`verify_saddle`] check to confirm exactly one negative mode.
///
/// Pure and synchronous — no threads, no async, no global state. Concurrency and
/// cancellation are the caller's concern.
///
/// # Errors
/// Returns [`TsError`] only for genuine compute faults:
/// [`SurfaceEvaluation`](TsError::SurfaceEvaluation) wrapping an underlying
/// [`OptError`], [`BadInitialGuess`](TsError::BadInitialGuess), or
/// [`AlgorithmNotImplemented`](TsError::AlgorithmNotImplemented) for a future,
/// not-yet-built algorithm variant. Failure to converge and a wrong negative-mode
/// count are *not* errors — they are reported via [`TsStatus`] on an `Ok` result
/// that retains the best-so-far geometry.
pub fn find_transition_state<S: Surface>(
    molecule: &Molecule,
    surface: &mut S,
    options: &TsOptions,
    progress: Option<&dyn Progress>,
) -> Result<TsResult, TsError> {
    match options.algorithm {
        TsAlgorithm::Prfo => prfo::run_prfo(molecule, surface, options, progress),
        TsAlgorithm::Dimer => dimer::run_dimer(molecule, surface, options, progress),
        // `TsAlgorithm` is `#[non_exhaustive]`: keep the catch-all so a future
        // variant (e.g. a single-geometry NEB flavour) compiles to a clean
        // `AlgorithmNotImplemented` rather than a non-exhaustive-match error.
        #[allow(unreachable_patterns)]
        other => Err(TsError::AlgorithmNotImplemented(other)),
    }
}
