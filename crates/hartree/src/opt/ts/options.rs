//! The transition-state search option types: [`TsAlgorithm`] (which driver) and
//! [`TsOptions`] (every knob), split out of `mod.rs` so the option set can grow
//! per-algorithm without crowding the public-contract file. Both are re-exported
//! from the parent, so callers still use `crate::opt::ts::{TsAlgorithm, TsOptions}`.

use serde::{Deserialize, Serialize};

use crate::opt::ts::IrcMethod;

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
/// single-geometry [`find_transition_state`](crate::opt::ts::find_transition_state)
/// entry point and is expected to arrive as its own driver with its own
/// option/result types; the marker keeps the door open to add a single-geometry
/// NEB-flavoured variant non-breakingly.
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

/// Which Hessian the post-convergence verification uses to count negative modes —
/// **P-RFO only** (the dimer carries no Hessian and always finite-differences one).
///
/// [`Strict`](VerifyHessian::Strict) (the default) always finite-differences a fresh
/// Hessian at the converged geometry — the most accurate, at ≈6N extra gradients.
/// [`Maintained`](VerifyHessian::Maintained) reuses the quasi-Newton (Bofill) Hessian
/// P-RFO already carries into convergence, spending no extra gradients but trusting an
/// approximate Hessian (so the reported spectrum/frequencies are approximate too).
/// [`Auto`](VerifyHessian::Auto) reuses the maintained Hessian when its spectrum is
/// unambiguous and falls back to a fresh one only when a mode sits near the
/// negative-mode threshold — the cheap choice that stays safe. `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VerifyHessian {
    /// Always finite-difference a fresh verification Hessian (byte-identical to the
    /// historical behaviour).
    #[default]
    Strict,
    /// Always reuse the maintained (Bofill) Hessian; never finite-difference.
    Maintained,
    /// Reuse the maintained Hessian when its spectrum is clearly classified; fall back
    /// to a fresh finite-difference Hessian only when a mode is near the threshold.
    Auto,
}

/// Which coordinate frame the P-RFO climb takes its steps in — **P-RFO only** (the
/// dimer method discovers its own translation direction and ignores this).
///
/// [`MassWeighted`](Coordinates::MassWeighted) (the default) climbs in mass-weighted
/// Cartesian coordinates with the rigid-body modes projected out — the same frame the
/// imaginary-frequency criterion lives in, and the historical behaviour.
/// [`Internal`](Coordinates::Internal) climbs in redundant internal coordinates
/// (bonds and valence angles, with disconnected fragments bridged so a
/// forming/breaking bond is represented). Internal coordinates condition a soft
/// reaction coordinate — a long symmetric stretch, a floppy angle — that a Cartesian
/// step sizes poorly, at the cost of an iterative back-transformation per step. The
/// maintained Hessian, its quasi-Newton update, the convergence test, and the
/// post-convergence saddle verification are identical either way; only the step
/// direction and length differ. When the generated internal set cannot span the
/// molecule's internal space, the search transparently falls back to the mass-weighted
/// Cartesian frame. `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Coordinates {
    /// Mass-weighted Cartesian coordinates, rigid-body modes projected out (the
    /// historical frame, byte-for-byte unchanged).
    #[default]
    MassWeighted,
    /// Redundant internal coordinates (bonds + valence angles), with a Cartesian
    /// fallback when the set is incomplete.
    Internal,
}

/// How the saddle search builds the **initial** climbing Hessian — **P-RFO only**
/// (the dimer is Hessian-free).
///
/// [`Auto`](HessianInit::Auto) (the default) uses the surface's
/// [`seed_hessian`](crate::opt::Surface::seed_hessian) when it provides one — a cheap
/// model or learned Hessian to start from — and otherwise finite-differences it.
/// [`Fd`](HessianInit::Fd) always finite-differences, ignoring any seed. Either way
/// the Hessian is only a starting point: P-RFO refines it by quasi-Newton updates and
/// the post-convergence verification is independent (see [`VerifyHessian`]).
/// `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HessianInit {
    /// Use the surface's seed Hessian if it offers one, else finite-difference.
    #[default]
    Auto,
    /// Always finite-difference the initial Hessian, ignoring any seed.
    Fd,
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
    /// [`TsStatus::NotConverged`](crate::opt::ts::TsStatus::NotConverged).
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

    /// P-RFO only: refresh the maintained (Bofill) Hessian from finite differences
    /// once the trans/rot-projected force has failed to improve for this many
    /// consecutive accepted steps — a recovery aid for **soft, floppy surfaces**.
    ///
    /// On a rugged low-curvature surface (an intramolecular hydrogen transfer, a
    /// large-amplitude floppy mode) the quasi-Newton update can settle into spurious
    /// curvatures inherited from a far-from-saddle initial Hessian and never shed
    /// them, so the climb plateaus far from convergence and exhausts
    /// [`max_iter`](Self::max_iter). A single fresh finite-difference Hessian sheds the
    /// spurious modes and restores descent; this triggers one when the force stalls,
    /// and re-arms only after the next non-improving streak (so it cannot storm).
    /// Unlike [`recalc_hessian`](Self::recalc_hessian)'s fixed cadence, this spends the
    /// extra ≈6N-gradient Hessians *only* when the climb is actually stuck, leaving a
    /// well-behaved search untouched.
    ///
    /// `0` (the default) disables it: the climb is byte-for-byte the historical
    /// Bofill-maintained search, so the shipped default path is unchanged. A typical
    /// enabled value is `5`. The dimer method (Hessian-free) ignores it.
    #[serde(default = "default_stall_refresh")]
    pub stall_refresh: usize,

    /// A mode counts as negative (the reaction coordinate) when its eigenvalue
    /// `λ < -negative_mode_tol`, where `λ` is an eigenvalue of the
    /// **mass-weighted, translation/rotation-projected** Cartesian Hessian
    /// (atomic units) — the same spectrum [`crate::props::frequencies`] builds,
    /// but with a deliberately *coarser* cut than that module's 1 cm⁻¹ imaginary
    /// threshold. The default `1e-5` a.u. is ≈ 16 cm⁻¹ (via `√λ·FREQ_CONV_CM1`);
    /// the looser cut absorbs finite-difference-Hessian noise so a stiff saddle
    /// is not spuriously demoted to a higher-order one, while still recognizing a
    /// genuinely soft imaginary mode. The trade-off: an *ultrasoft* saddle
    /// (imaginary mode below ≈16 cm⁻¹) is reported as wrong-mode-count even though a
    /// frequency job would call it imaginary — lower `negative_mode_tol` to chase a
    /// floppy TS (at the cost of noise sensitivity).
    /// Modes within `±tol` of zero are the soft trans/rot residue and are not
    /// counted, so a clean first-order saddle has exactly one mode past this
    /// threshold. Drives the
    /// [`verify_saddle`](crate::opt::ts::verify_saddle) check.
    pub negative_mode_tol: f64,
    /// If set, after convergence trace the intrinsic reaction coordinate a short
    /// way downhill in both senses of the reaction mode to confirm the saddle
    /// connects two distinct basins; the endpoints land in
    /// [`TsResult::irc`](crate::opt::ts::TsResult::irc). `false` skips the (extra
    /// surface evaluations) check.
    pub confirm_irc: bool,

    /// Maximum times a single step is shrunk (to a quarter of the trust radius) and
    /// retried from the same geometry before the search gives up on it. A trial step
    /// is retried when its surface evaluation fails to converge
    /// ([`OptError::ScfNotConverged`](crate::opt::OptError::ScfNotConverged)) or
    /// returns a non-finite energy, and — for P-RFO, which carries a quadratic
    /// model — when the step grossly overshoots the model. Whatever the retry
    /// budget, an unrecovered trial-step SCF failure ends the search *softly*
    /// ([`TsStatus::NotConverged`](crate::opt::ts::TsStatus::NotConverged) with
    /// best-so-far) rather than surfacing a [`TsError`](crate::opt::ts::TsError);
    /// only a failure at an already-accepted point (the initial geometry, or an
    /// accepted step's gradient) is a hard error. `0` disables backtracking: a
    /// converged step is accepted unconditionally and the first unrecovered SCF
    /// failure soft-stops. Retries do not consume [`max_iter`](Self::max_iter)
    /// iterations.
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

    /// P-RFO only: a reaction-coordinate seed (one Cartesian direction per atom,
    /// input atom order) — typically the normalized forming/breaking-bond vector
    /// the guess builder reports. When set, the **first** step follows the Hessian
    /// mode of maximum overlap with it (after the same mass-weighting the spectrum
    /// uses) rather than the [`follow_mode`](Self::follow_mode)-th softest mode,
    /// making the choice robust to an avoided-crossing reordering of the soft modes
    /// at the guess. Later steps track the followed mode by overlap as before.
    /// `None` (the default) falls back to `follow_mode`; the dimer method, which
    /// discovers its own mode, ignores it.
    #[serde(default)]
    pub reaction_mode_seed: Option<Vec<[f64; 3]>>,

    /// P-RFO only: maximum number of times the search, after converging to a point
    /// with the wrong number of negative modes
    /// ([`TsStatus::WrongImaginaryModeCount`](crate::opt::ts::TsStatus::WrongImaginaryModeCount)),
    /// will displace off that point and re-climb before giving up. Recovery needs a
    /// [`reaction_mode_seed`](Self::reaction_mode_seed): it descends the *spurious*
    /// negative mode(s) — those **not** aligned with the seed — while re-climbing the
    /// seeded reaction coordinate, so a search that settled on a higher-order saddle
    /// (or a minimum) can still reach the first-order saddle. Each re-climb gets a
    /// fresh [`max_iter`](Self::max_iter) budget. `0` disables recovery (today's
    /// behaviour); the default is `2`. With no seed it has no effect.
    #[serde(default = "default_max_recover")]
    pub max_recover: usize,

    /// P-RFO only: which Hessian the post-convergence verification uses (see
    /// [`VerifyHessian`]). The default [`Strict`](VerifyHessian::Strict) reproduces the
    /// historical fresh finite-difference verification; [`Auto`](VerifyHessian::Auto)
    /// skips that ≈6N-gradient Hessian on a cleanly-classified success. The dimer
    /// method ignores it (it never forms a maintained Hessian).
    #[serde(default)]
    pub verify_hessian: VerifyHessian,

    /// P-RFO only: how the initial climbing Hessian is built (see [`HessianInit`]).
    /// The default [`Auto`](HessianInit::Auto) finite-differences it unless the
    /// surface offers a [`seed_hessian`](crate::opt::Surface::seed_hessian); existing
    /// surfaces (which offer none) are unaffected. The dimer ignores it.
    #[serde(default)]
    pub hessian_init: HessianInit,

    /// P-RFO only: which coordinate frame the climb steps in (see [`Coordinates`]).
    /// The default [`MassWeighted`](Coordinates::MassWeighted) reproduces the
    /// historical mass-weighted Cartesian search exactly;
    /// [`Internal`](Coordinates::Internal) steps in redundant internal coordinates for
    /// better conditioning of soft reaction coordinates. The dimer ignores it.
    #[serde(default)]
    pub coordinates: Coordinates,
}

/// Default step-retry budget (see [`TsOptions::max_step_retries`]); also the serde
/// default so options serialized before the field round-trip unchanged.
fn default_max_step_retries() -> usize {
    6
}

/// Default stalled-Hessian refresh window (see [`TsOptions::stall_refresh`]); `0`
/// disables the aid, reproducing the historical climb. Also the serde default so
/// options serialized before the field existed round-trip unchanged.
fn default_stall_refresh() -> usize {
    0
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

/// Default reaction-coordinate recovery budget (see [`TsOptions::max_recover`]);
/// also the serde default so options serialized before the field round-trip.
fn default_max_recover() -> usize {
    2
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
            stall_refresh: default_stall_refresh(),
            negative_mode_tol: 1e-5,
            confirm_irc: false,
            max_step_retries: default_max_step_retries(),
            irc_method: IrcMethod::Dvv,
            irc_step: default_irc_step(),
            irc_max_steps: default_irc_max_steps(),
            irc_gtol: default_irc_gtol(),
            reaction_mode_seed: None,
            max_recover: default_max_recover(),
            verify_hessian: VerifyHessian::Strict,
            hessian_init: HessianInit::Auto,
            coordinates: Coordinates::default(),
        }
    }
}
