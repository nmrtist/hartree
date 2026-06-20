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

pub mod guess;

mod dimer;
mod irc;
mod numerics;
mod prfo;
mod step;

pub use irc::{IrcEndpoints, IrcMethod};
pub use numerics::SaddleVerification;
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

/// Saddle-point search algorithm.
///
/// [`Prfo`](TsAlgorithm::Prfo) is the local Newton-type method: it follows a
/// Hessian eigenvector uphill and needs a guess already inside the saddle's
/// quadratic basin. [`Dimer`](TsAlgorithm::Dimer) is a Hessian-free alternative
/// that estimates the lowest-curvature mode from two nearby gradient
/// evaluations — cheaper per step and more forgiving of the initial guess.
///
/// `#[non_exhaustive]` because a nudged-elastic-band (NEB) chain-of-states method
/// is planned (the `// Neb` slot below). NEB optimizes an entire path between two
/// *minima* rather than refining a single guess, so it does not fit the
/// single-geometry [`find_transition_state`] entry point and is expected to
/// arrive as its own driver with its own option/result types; the marker keeps
/// the door open to add a single-geometry NEB-flavoured variant non-breakingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TsAlgorithm {
    /// Partitioned rational-function optimization with eigenvector following:
    /// maximize along the chosen Hessian mode, minimize along the rest.
    #[default]
    Prfo,
    /// Dimer method: rotate a pair of nearby images to locate the softest mode,
    /// then translate with the force component along that mode inverted.
    Dimer,
    // Neb — nudged elastic band (chain-of-states). See the type-level note: a
    // path optimizer needs two endpoints, so it is expected to land as its own
    // driver rather than a variant consumed by `find_transition_state`.
}

/// Options for a transition-state search.
///
/// The shared knobs (`max_iter`, the trust radii, the force/displacement
/// thresholds, and `fd_step`) mirror [`OptOptions`](crate::opt::OptOptions) in
/// name and units so a TS search reads like the minimizer it is built on (the
/// `max_iter` default is raised — saddle searches need more steps). The remaining
/// fields are TS-specific; each notes which algorithm it applies to.
/// `#[non_exhaustive]` so per-algorithm knobs (e.g. dimer rotation tolerances,
/// IRC step controls) can be added without a breaking change; construct via
/// [`TsOptions::default`] and update the fields you need.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TsOptions {
    /// Which saddle-point algorithm to run.
    pub algorithm: TsAlgorithm,

    /// Maximum number of iterations before the search gives up and returns
    /// [`TsStatus::NotConverged`].
    pub max_iter: usize,
    /// Initial trust radius for the step-restricted update (as in `OptOptions`).
    pub trust_radius: f64,
    /// Upper bound on the trust radius.
    pub max_trust: f64,
    /// Lower bound on the trust radius.
    pub min_trust: f64,
    /// Finite-difference step for gradients (when the surface exposes none) and
    /// for the numerical Hessian / curvature estimates the algorithms need.
    pub fd_step: f64,

    /// Convergence threshold on the largest force component (atomic units). The
    /// force is measured after projecting out the rigid-body translation/rotation
    /// modes, so a residual net force/torque does not set a noise floor on
    /// convergence.
    pub max_force: f64,
    /// Convergence threshold on the RMS force (after the same trans/rot projection
    /// as [`max_force`](Self::max_force)).
    pub rms_force: f64,
    /// Convergence threshold on the largest step component.
    pub max_disp: f64,
    /// Convergence threshold on the RMS step.
    pub rms_disp: f64,

    /// P-RFO only: index of the (ascending) Hessian mode to follow uphill,
    /// `0` = softest mode. Ignored by the dimer method, which discovers the mode
    /// it follows.
    pub follow_mode: usize,
    /// P-RFO only: recompute the (finite-difference) Hessian every
    /// `recalc_hessian` accepted steps. `0` computes it once at the start and
    /// then maintains it by an **indefinite-preserving** quasi-Newton update
    /// (SR1 / Bofill) — *not* the minimizer's positive-definite BFGS, which would
    /// erase the negative reaction mode P-RFO must keep. A nonzero value trades
    /// surface evaluations (each recompute is ~6N central-difference gradients) for a fresh exact
    /// Hessian and is the more robust choice on rugged surfaces.
    pub recalc_hessian: usize,
    /// Dimer only: half-separation between the two dimer images (atomic units)
    /// used to finite-difference the curvature along the dimer axis.
    pub dimer_delta: f64,

    /// A mode counts as negative (the reaction coordinate) when its eigenvalue
    /// `λ < -negative_mode_tol`, where `λ` is an eigenvalue of the
    /// **mass-weighted, translation/rotation-projected** Cartesian Hessian
    /// (atomic units) — the same spectrum [`crate::props::frequencies`] builds,
    /// but with a deliberately *coarser* cut than that module's 1 cm⁻¹ imaginary
    /// threshold. The default `1e-4` a.u. is ≈ 51 cm⁻¹ (via `√λ·FREQ_CONV_CM1`);
    /// the looser cut absorbs finite-difference-Hessian noise so a stiff saddle
    /// is not spuriously demoted to a higher-order one. The trade-off: an
    /// *ultrasoft* saddle (imaginary mode below ≈51 cm⁻¹) is reported as
    /// [`TsStatus::WrongImaginaryModeCount`] even though a frequency job would
    /// call it imaginary — lower `negative_mode_tol` to chase a floppy TS (at the
    /// cost of noise sensitivity). Modes within `±tol` of zero are the soft
    /// trans/rot residue and are not counted, so a clean first-order saddle has
    /// exactly one mode past this threshold. Drives the [`verify_saddle`] check
    /// and hence [`TsStatus::WrongImaginaryModeCount`].
    pub negative_mode_tol: f64,
    /// If set, after convergence trace the intrinsic reaction coordinate a short
    /// way downhill in both senses of the reaction mode to confirm the saddle
    /// connects two distinct basins; the endpoints land in [`TsResult::irc`].
    /// `false` skips the (extra surface evaluations) check.
    pub confirm_irc: bool,

    /// Maximum times a single step is shrunk (to a quarter of the trust radius) and
    /// retried from the same geometry before the search gives up on it. A trial step
    /// is retried when its surface evaluation fails to converge
    /// ([`OptError::ScfNotConverged`]) or returns a non-finite energy, and — for
    /// P-RFO, which carries a quadratic model — when the step grossly overshoots the
    /// model. Whatever the retry budget, an unrecovered trial-step SCF failure ends
    /// the search *softly* ([`TsStatus::NotConverged`] with best-so-far) rather than
    /// surfacing a [`TsError`]; only a failure at an already-accepted point (the
    /// initial geometry, or an accepted step's gradient) is a hard error. `0`
    /// disables backtracking: a converged step is accepted unconditionally and the
    /// first unrecovered SCF failure soft-stops. Retries do not consume
    /// [`max_iter`](Self::max_iter) iterations.
    #[serde(default = "default_max_step_retries")]
    pub max_step_retries: usize,

    /// IRC only ([`confirm_irc`](Self::confirm_irc)): which intrinsic-reaction-
    /// coordinate integrator traces the path off the saddle. See [`IrcMethod`];
    /// the default [`Dvv`](IrcMethod::Dvv) is Hessian-free.
    #[serde(default)]
    pub irc_method: IrcMethod,
    /// IRC only: arc-length step of the integrator, in mass-weighted coordinates
    /// (`√amu·bohr`). Also the size of the initial displacement off the saddle ridge.
    #[serde(default = "default_irc_step")]
    pub irc_step: f64,
    /// IRC only: maximum integration steps **per endpoint** before the trace stops
    /// and reports the endpoint as not converged.
    #[serde(default = "default_irc_max_steps")]
    pub irc_max_steps: usize,
    /// IRC only: convergence threshold on the trans/rot-projected RMS force (atomic
    /// units) — the trace has reached a minimum once it falls below this.
    #[serde(default = "default_irc_gtol")]
    pub irc_gtol: f64,
}

/// Default step-retry budget (see [`TsOptions::max_step_retries`]); also the serde
/// default so options serialized before the field round-trip unchanged.
fn default_max_step_retries() -> usize {
    6
}

/// Serde/`Default` values for the IRC controls, so options serialized before these
/// fields existed round-trip unchanged (see [`TsOptions::irc_step`] etc.).
fn default_irc_step() -> f64 {
    0.1
}
fn default_irc_max_steps() -> usize {
    150
}
fn default_irc_gtol() -> f64 {
    1e-3
}

impl Default for TsOptions {
    fn default() -> Self {
        Self {
            algorithm: TsAlgorithm::Prfo,
            // Shared knobs mirror `OptOptions::default`, except `max_iter` and the
            // convergence thresholds. `max_iter` is raised because a
            // finite-differenced saddle search (with periodic Hessian recomputes)
            // consumes more steps than a downhill relaxation. The force/step
            // thresholds are *loosened* relative to the minimizer's 3e-6: a P-RFO
            // step is driven by a finite-difference Hessian (and, between
            // recomputes, a quasi-Newton update), for which the minimizer's target
            // is impractically tight. These are a realistic "converged saddle"
            // tolerance, comparable to a normal geometry optimization; tighten
            // per-job via `TsOptions` when a specific saddle warrants it.
            max_iter: 300,
            // Conservative trust region: a climbing step easily overshoots into a
            // non-convergent SCF region, so cap well below the minimizer's.
            trust_radius: 0.2,
            max_trust: 0.3,
            min_trust: 1e-4,
            fd_step: 5e-3,
            max_force: 1.0e-4,
            rms_force: 5.0e-5,
            max_disp: 1.0e-3,
            rms_disp: 5.0e-4,
            // TS-specific.
            follow_mode: 0,
            recalc_hessian: 0,
            dimer_delta: 1e-2,
            negative_mode_tol: 1e-4,
            confirm_irc: false,
            max_step_retries: default_max_step_retries(),
            irc_method: IrcMethod::Dvv,
            irc_step: default_irc_step(),
            irc_max_steps: default_irc_max_steps(),
            irc_gtol: default_irc_gtol(),
        }
    }
}

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
    let hessian = numerics::fd_hessian(surface, positions, options.fd_step)?;
    let molecule = numerics::with_positions(molecule, positions);
    numerics::saddle_from_hessian(&molecule, &hessian, options.negative_mode_tol)
        .map_err(TsError::Numerical)
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
