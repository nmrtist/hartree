//! Double-ended minimum-energy-path search: a climbing-image nudged elastic band
//! (CI-NEB).
//!
//! Where [`find_transition_state`](super::find_transition_state) refines a *single*
//! near-saddle guess, [`find_minimum_energy_path`] takes the two *minima* a reaction
//! connects (reactant and product) and relaxes a whole band of intermediate images
//! onto the minimum-energy path between them. A climbing image then rides the highest
//! point of the band toward the saddle, giving an approximate transition state plus
//! its reaction-coordinate tangent — exactly the geometry and seed a local refiner
//! ([`find_transition_state`] with a
//! [`reaction_mode_seed`](super::TsOptions::reaction_mode_seed)) needs to converge a
//! tight saddle. This is the production "NEB-TS" workflow ORCA and Q-Chem use to get
//! into the saddle basin when no good single guess exists.
//!
//! It is a *separate driver*, not a [`TsAlgorithm`](super::TsAlgorithm) variant: a
//! path optimizer needs two endpoints and returns a band, not a single geometry, so
//! it cannot share [`find_transition_state`]'s single-geometry contract. Like the
//! rest of `opt`, it is pure synchronous numerics — the per-image gradients are
//! evaluated one at a time on a single [`Surface`]; image-level parallelism is the
//! caller's concern.
//!
//! Unlike the P-RFO / dimer drivers (which work in mass-weighted Cartesians for the
//! imaginary-mode bookkeeping), the band is relaxed in **plain Cartesians**, the
//! standard NEB frame (Henkelman, Uberuaga & Jónsson, J. Chem. Phys. 113, 9901
//! (2000)). Mass-weighting re-enters only when the peak geometry is handed to a
//! mass-weighted refiner.

mod band;
mod optimizer;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::guess::{self, GuessOptions};
use super::{Progress, TsError, TsOptions, TsResult, find_transition_state};
use crate::core::Molecule;
use crate::opt::{OptError, OptStep, Surface};

/// Outcome classification for a [`NebResult`], mirroring
/// [`TsStatus`](super::TsStatus) for the path driver. `#[non_exhaustive]`: treat an
/// unknown future variant as "not a converged path".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NebStatus {
    /// The maximum NEB force on the band fell below
    /// [`NebOptions::gtol`] (with the climbing image active, when one was requested):
    /// the band lies on the minimum-energy path and the climbing image sits at the
    /// barrier top.
    Converged,
    /// Hit [`NebOptions::max_iter`] without meeting the force threshold.
    /// [`NebResult`] carries the last band reached, for restart / inspection.
    NotConverged,
    /// A [`Progress`] observer returned [`Flow::Stop`](super::Flow::Stop) before
    /// convergence; the last band is returned.
    StoppedEarly,
}

/// Genuine compute faults of a path search — failures that yield no usable band.
/// Non-convergence is *not* here: it rides on `Ok(NebResult)` via [`NebStatus`].
/// `#[non_exhaustive]`, matching the crate's other public error enums.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NebError {
    /// A [`Surface`] energy or gradient evaluation failed (transparently wraps the
    /// underlying [`OptError`], including an SCF non-convergence at an image).
    #[error(transparent)]
    SurfaceEvaluation(#[from] OptError),
    /// The two endpoints are unusable as a band's ends: different atom counts, a
    /// different atom ordering (without [`NebOptions::map_atoms`], which permits it),
    /// too few atoms, or `n_images == 0`. Carries a human-readable reason.
    #[error("bad NEB endpoints: {0}")]
    BadEndpoints(String),
    /// A numerical failure that leaves no usable band. Carries a reason.
    #[error("NEB numerics failed: {0}")]
    Numerical(String),
}

/// Options for a climbing-image NEB path search; construct via
/// [`NebOptions::default`] and update the fields you need. Every field is
/// `#[serde(default)]` (through the container attribute), so options serialized
/// before a field existed round-trip unchanged. `#[non_exhaustive]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct NebOptions {
    /// Number of interior images on the band (the two endpoints are fixed and not
    /// counted). More images resolve a longer or more curved path at proportionally
    /// more gradient evaluations per iteration.
    pub n_images: usize,
    /// Spring constant `k` (Hartree/Bohr²) coupling adjacent images along the band
    /// tangent; it controls image spacing and does not bias the path off the MEP
    /// (only the *parallel* component of the spring force is kept — the nudge).
    pub spring_k: f64,
    /// Enable the climbing image: once the band is partly relaxed, the highest-energy
    /// interior image drops its springs and inverts its parallel true force, climbing
    /// to the barrier top. With it off the band is a plain NEB and the reported peak
    /// is only the highest band image.
    pub climbing: bool,
    /// Activate the climbing image once the band's maximum force has fallen to a
    /// modest level, or unconditionally after this many iterations (whichever comes
    /// first) — climbing from a cold, unrelaxed band is unstable. Ignored when
    /// [`climbing`](Self::climbing) is `false`.
    pub climb_after: usize,
    /// Maximum relaxation iterations before the search reports
    /// [`NebStatus::NotConverged`].
    pub max_iter: usize,
    /// Convergence threshold on the largest NEB-force component over all interior
    /// images (atomic units). When climbing, this includes the climbing image's full
    /// force, so convergence means the peak sits at the saddle.
    pub gtol: f64,
    /// Finite-difference step (Bohr) for the gradient when the surface exposes no
    /// analytic one. Unused for a surface with analytic gradients (the usual case).
    pub fd_step: f64,
    /// Rigidly Kabsch-align the product endpoint onto the reactant before building the
    /// band, removing an arbitrary relative orientation/translation between two
    /// separately optimized minima. Leave `false` for a surface whose energy depends
    /// on absolute coordinates (e.g. an analytic test surface), where re-orienting an
    /// endpoint would change its energy; leave `false` too when the endpoints already
    /// share a frame.
    pub align: bool,
    /// Reorder the product's atoms onto the reactant's by atom mapping before building
    /// the band, lifting the requirement that the two endpoints already list their atoms
    /// in the same order. With it `false` (the default) the driver keeps the strict
    /// identical-ordering check; with it `true` the product is permuted to match the
    /// reactant (via the same connectivity-plus-geometry mapping the guess builder uses,
    /// using [`bond_factor`](Self::bond_factor) for the connectivity). The endpoints must
    /// still share an element multiset.
    pub map_atoms: bool,
    /// Covalent-radius multiplier for the bond cutoff used when
    /// [`map_atoms`](Self::map_atoms) builds endpoint connectivity. Unused otherwise.
    pub bond_factor: f64,
    /// FIRE initial time step (Bitzek *et al.*, Phys. Rev. Lett. 97, 170201 (2006)).
    pub fire_dt: f64,
    /// FIRE maximum time step.
    pub fire_dt_max: f64,
    /// FIRE: number of consecutive downhill (power > 0) steps before the time step is
    /// allowed to grow.
    pub fire_n_min: usize,
    /// FIRE time-step growth factor (applied while descending).
    pub fire_f_inc: f64,
    /// FIRE time-step shrink factor (applied when the power goes negative).
    pub fire_f_dec: f64,
    /// FIRE initial velocity-mixing coefficient.
    pub fire_alpha_start: f64,
    /// FIRE mixing-coefficient decay (applied while descending).
    pub fire_f_alpha: f64,
    /// Cap on the largest per-coordinate displacement of a single FIRE step (Bohr);
    /// keeps an early, large-force step from launching an image into a non-convergent
    /// region of the surface. The step direction is preserved (uniform down-scaling).
    pub fire_max_step: f64,
}

impl Default for NebOptions {
    fn default() -> Self {
        Self {
            n_images: 8,
            spring_k: 0.1,
            climbing: true,
            climb_after: 5,
            max_iter: 300,
            gtol: 1.0e-3,
            fd_step: 5.0e-3,
            align: false,
            map_atoms: false,
            bond_factor: 1.3,
            // FIRE defaults follow the original paper / ASE, in atomic units.
            fire_dt: 0.1,
            fire_dt_max: 1.0,
            fire_n_min: 5,
            fire_f_inc: 1.1,
            fire_f_dec: 0.5,
            fire_alpha_start: 0.1,
            fire_f_alpha: 0.99,
            fire_max_step: 0.2,
        }
    }
}

/// Structured outcome of a [`find_minimum_energy_path`] search. `#[non_exhaustive]`
/// so further data can be added without breaking consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NebResult {
    /// The relaxed band, **including** the two fixed endpoints: `images[0]` is the
    /// reactant, `images[n_images + 1]` the product, and `images[1..=n_images]` the
    /// interior images that were optimized. Cartesian, atomic units, shared atom
    /// order.
    pub images: Vec<Vec<[f64; 3]>>,
    /// Energy at each entry of [`images`](Self::images) (same length, same order).
    pub energies: Vec<f64>,
    /// Index into [`images`](Self::images) of the highest-energy interior image — the
    /// climbing image, the best on-band approximation to the transition state.
    pub climbing_image: usize,
    /// Forward barrier: the climbing-image energy minus the reactant-endpoint energy
    /// (`energies[climbing_image] - energies[0]`), atomic units.
    pub barrier: f64,
    /// The climbing-image geometry (`images[climbing_image]`), surfaced directly as
    /// the transition-state guess to feed a local refiner.
    pub peak_geometry: Vec<[f64; 3]>,
    /// The (unit) reaction-coordinate tangent at the climbing image, one Cartesian
    /// vector per atom — the direction to hand a P-RFO refiner as its
    /// [`reaction_mode_seed`](super::TsOptions::reaction_mode_seed). Plain Cartesian
    /// (the refiner mass-weights it internally).
    pub peak_tangent: Vec<[f64; 3]>,
    /// Outcome classification; see [`NebStatus`].
    pub status: NebStatus,
    /// Number of relaxation iterations taken.
    pub iterations: usize,
    /// Per-iteration convergence trace (reusing the minimizer's [`OptStep`]); the
    /// `energy` field carries the climbing/peak-image energy and the force fields the
    /// band's NEB-force norms.
    pub history: Vec<OptStep>,
}

impl NebResult {
    /// `true` iff the band reached [`NebStatus::Converged`].
    pub fn converged(&self) -> bool {
        matches!(self.status, NebStatus::Converged)
    }
}

/// Find the minimum-energy path between two minima with a climbing-image NEB.
///
/// `reactant` and `product` are the two endpoint minima. By default they must hold the
/// same atoms in the same order; set [`NebOptions::map_atoms`] to permute the product
/// onto the reactant first (then they need only share an element multiset). `surface`
/// evaluates energies and gradients for that shared composition (built exactly as for
/// [`find_transition_state`]); it is queried at one image at a time. The optional
/// `progress` observer is called once per iteration and may request an early stop.
///
/// On success the returned [`NebResult`] carries the relaxed band, the climbing-image
/// geometry and tangent (the transition-state guess + seed), and the forward barrier.
///
/// # Errors
/// Returns [`NebError`] only for genuine compute faults:
/// [`SurfaceEvaluation`](NebError::SurfaceEvaluation) wrapping an [`OptError`],
/// [`BadEndpoints`](NebError::BadEndpoints) for malformed endpoints, or
/// [`Numerical`](NebError::Numerical). Failure to converge is reported via
/// [`NebStatus`] on an `Ok` result that retains the last band.
pub fn find_minimum_energy_path<S: Surface>(
    reactant: &Molecule,
    product: &Molecule,
    surface: &mut S,
    options: &NebOptions,
    progress: Option<&dyn Progress>,
) -> Result<NebResult, NebError> {
    let n = reactant.len();
    if n < 2 {
        return Err(NebError::BadEndpoints(format!(
            "need at least two atoms, got {n}"
        )));
    }
    if product.len() != n {
        return Err(NebError::BadEndpoints(format!(
            "reactant has {n} atoms but product has {}",
            product.len()
        )));
    }
    // With `map_atoms`, permute the product onto the reactant's atom order first; the
    // identical-ordering check below then passes by construction. Without it, the product
    // is used as given and the check enforces the ordering.
    let mapped_product;
    let product: &Molecule = if options.map_atoms {
        mapped_product =
            guess::reorder_product_onto_reactant(reactant, product, options.bond_factor)
                .map_err(|e| NebError::BadEndpoints(e.to_string()))?;
        &mapped_product
    } else {
        product
    };
    for i in 0..n {
        let (zr, zp) = (reactant.atoms[i].element.z(), product.atoms[i].element.z());
        if zr != zp {
            return Err(NebError::BadEndpoints(format!(
                "atom {i} differs between endpoints (reactant Z={zr}, product Z={zp}); \
                 this driver requires identical atom ordering (or set map_atoms)"
            )));
        }
    }
    if options.n_images == 0 {
        return Err(NebError::BadEndpoints("n_images must be ≥ 1".to_string()));
    }
    if options.max_iter == 0 {
        return Err(NebError::BadEndpoints("max_iter must be ≥ 1".to_string()));
    }

    let react_pos: Vec<[f64; 3]> = reactant.atoms.iter().map(|a| a.position).collect();
    let mut prod_pos: Vec<[f64; 3]> = product.atoms.iter().map(|a| a.position).collect();
    if options.align {
        prod_pos = band::kabsch_align(&prod_pos, &react_pos);
    }

    // Build the interior band by IDPP interpolation, then bracket it with the two
    // fixed endpoints into the full chain the optimizer relaxes.
    let interior = guess::band::interpolate_band(
        &react_pos,
        &prod_pos,
        options.n_images,
        &GuessOptions::default(),
    );
    let mut images = Vec::with_capacity(options.n_images + 2);
    images.push(react_pos);
    images.extend(interior);
    images.push(prod_pos);

    let relaxed = optimizer::relax(surface, images, options, progress)?;

    let climbing_image = relaxed.climbing_image;
    let peak_geometry = relaxed.images[climbing_image].clone();
    let barrier = relaxed.energies[climbing_image] - relaxed.energies[0];
    Ok(NebResult {
        images: relaxed.images,
        energies: relaxed.energies,
        climbing_image,
        barrier,
        peak_geometry,
        peak_tangent: relaxed.peak_tangent,
        status: relaxed.status,
        iterations: relaxed.iterations,
        history: relaxed.history,
    })
}

/// Combined outcome of the double-ended → local-refiner pipeline
/// ([`find_transition_state_from_endpoints`]): the relaxed band and the tight saddle
/// the refiner converged from its climbing image. `#[non_exhaustive]` so more data
/// can be added without breaking consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NebTsResult {
    /// The climbing-image NEB band that produced the transition-state guess (the
    /// geometry and the reaction-coordinate tangent). Its `status` and `barrier`
    /// describe the path itself; see [`NebResult`].
    pub neb: NebResult,
    /// The saddle the local refiner converged from the NEB climbing image, seeded with
    /// the band's reaction-coordinate tangent. Its `status` and `verification` describe
    /// the saddle; see [`TsResult`].
    pub transition_state: TsResult,
}

/// Genuine compute faults of the NEB-TS pipeline — either stage's hard error.
/// Soft non-convergence still rides on the inner [`NebResult`]/[`TsResult`] status
/// fields. `#[non_exhaustive]`, matching the crate's other public error enums.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NebTsError {
    /// The double-ended path search failed (see [`NebError`]).
    #[error("path search failed: {0}")]
    Neb(#[from] NebError),
    /// The local saddle refinement failed (see [`TsError`]).
    #[error("saddle refinement failed: {0}")]
    Ts(#[from] TsError),
}

/// Run the production "NEB-TS" workflow end to end: relax a climbing-image band
/// between two minima, then hand its climbing image to the local P-RFO refiner for a
/// tight saddle.
///
/// This is the thin convenience wrapper over the two steps
/// [`tests/neb_reference.rs`](../../../../tests/neb_reference.rs) demonstrates: it calls
/// [`find_minimum_energy_path`] to get into the saddle basin, builds the refiner's
/// starting geometry from the climbing image (in `reactant`'s atom order and
/// composition), installs the band's reaction-coordinate tangent
/// ([`NebResult::peak_tangent`]) as the saddle search's
/// [`reaction_mode_seed`](super::TsOptions::reaction_mode_seed) — **replacing any seed
/// already in `ts_options`**, since the relaxed band's tangent is the reaction
/// coordinate by construction — and calls [`find_transition_state`]. The single
/// `surface` drives both stages (the two endpoints, the band, and the saddle all share
/// one composition).
///
/// The band's status is *not* a hard gate: a band that only reached
/// [`NebStatus::NotConverged`] still positions its climbing image at the highest point
/// reached, which is often a serviceable refiner seed — inspect
/// [`NebTsResult::neb`]'s status to judge the path, and
/// [`NebTsResult::transition_state`]'s status to judge the saddle.
///
/// # Errors
/// [`NebTsError::Neb`] if the path search hits a genuine compute fault (see
/// [`NebError`]); [`NebTsError::Ts`] if the refinement does (see [`TsError`]).
/// Non-convergence of either stage is *not* an error — it is reported via the inner
/// result's status field.
pub fn find_transition_state_from_endpoints<S: Surface>(
    reactant: &Molecule,
    product: &Molecule,
    surface: &mut S,
    neb_options: &NebOptions,
    ts_options: &TsOptions,
    progress: Option<&dyn Progress>,
) -> Result<NebTsResult, NebTsError> {
    let neb = find_minimum_energy_path(reactant, product, surface, neb_options, progress)?;

    // The refiner starts from the climbing image, carried back into the reactant's
    // atom order/composition (NEB relaxes positions only).
    let mut guess = reactant.clone();
    for (atom, &p) in guess.atoms.iter_mut().zip(&neb.peak_geometry) {
        atom.position = p;
    }

    // Seed the saddle search with the relaxed band's reaction-coordinate tangent (the
    // refiner mass-weights it internally), preserving every other ts_options knob.
    let mut ts_options = ts_options.clone();
    ts_options.reaction_mode_seed = Some(neb.peak_tangent.clone());

    let transition_state = find_transition_state(&guess, surface, &ts_options, progress)?;
    Ok(NebTsResult {
        neb,
        transition_state,
    })
}
