use crate::basis::BasisError;
use crate::basis::BasisSet;
use crate::cc::{
    CcsdOptions, CcsdResult, CcsdTResult, Mp2Result, RiMp2Result, frozen_core_orbitals,
    rccsd_spin_adapted, rccsd_t_spin_adapted, rhf_mp2, rhf_ri_mp2, uhf_mp2, uhf_ri_mp2,
};
use crate::core::Molecule;
use crate::dft::{
    COSX_DEFAULT_GRID, CosxExchange, CosxProvider, CubeParams, FodResult, FunctionalSpec, GridXc,
};
use crate::disp::Dispersion;
use crate::integrals::{ConventionalProvider, DfProvider, DirectProvider, IntegralProvider};
use crate::linalg::{mat_from_row_major, mat_to_row_major};
use crate::opt::internals::Internal;
use crate::opt::ts::guess::{
    CoordScanOptions, GuessOptions, MappingConfidence, ScanOptions, build_ts_guess,
    build_ts_guess_scanned, coord_scan_peak,
};
use crate::opt::ts::{
    NebOptions, NebTsError, Progress, TsError, TsOptions, TsResult, find_transition_state,
    find_transition_state_from_endpoints,
};
use crate::opt::{OptError, OptOptions, OptResult, Surface, optimize};
use crate::props::dipole::{center_of_mass, dipole_moment};
use crate::props::frequencies::{FrequencyResult, harmonic_frequencies};
use crate::props::hessian::numerical_hessian;
use crate::props::population::{PopulationAnalysis, population_analysis};
use crate::props::thermo::{ThermoResult, rrho_thermochemistry_w0};
use crate::scf::{
    Reference, ScfOptions, ScfResult, Smearing, SolventModel, XcContributor, run_scf_with_env,
};
use crate::solv::Cpcm;

use crate::surface::HfSurface;

pub(crate) struct EcpAwareSetup {
    pub charges: Vec<([f64; 3], f64)>,
    pub ecps: Vec<crate::integrals::integral::Ecp>,
    pub nuclear_repulsion: f64,
}

pub(crate) fn ecp_setup(mol: &Molecule, ao: &crate::basis::AoBasis) -> EcpAwareSetup {
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .zip(ao.ecp_core())
        .map(|(a, &c)| (a.position, a.z_eff() as f64 - c as f64))
        .collect();
    let zeff: Vec<f64> = charges.iter().map(|&(_, q)| q).collect();
    EcpAwareSetup {
        ecps: ao.ecps().to_vec(),
        nuclear_repulsion: mol.nuclear_repulsion_with(&zeff),
        charges,
    }
}

pub(crate) fn x2c_hcore_override(
    ao: &crate::basis::AoBasis,
    charges: &[([f64; 3], f64)],
    lindep: f64,
) -> Result<Vec<f64>, String> {
    let b = ao.integral();
    let s = b.overlap();
    let t = b.kinetic();
    let v = b.nuclear(charges);
    let w = b.pvp_charges(charges);
    crate::scf::x2c::x2c1e_hcore(
        &s,
        &t,
        &v,
        &w,
        b.nao(),
        crate::scf::x2c::SPEED_OF_LIGHT_AU,
        lindep,
    )
    .map(|out| out.h)
    .map_err(|e| e.to_string())
}

pub fn ecp_summary(mol: &Molecule, set: &BasisSet) -> Vec<(String, u32, u32)> {
    let mut seen = Vec::new();
    for atom in &mol.atoms {
        let z = atom.element.z();
        if let Some(e) = set.ecp_for(z)
            && !seen.iter().any(|&(_, sz, _)| sz == z)
        {
            seen.push((atom.element.symbol().to_string(), z, e.n_core));
        }
    }
    seen
}

#[derive(Debug, Clone)]
pub enum Method {
    Rhf,
    Uhf,
    Rohf,
    Mp2,
    Ccsd,
    CcsdT,
    Dft(FunctionalSpec),
}

/// Two-endpoint input for a transition-state search: a reactant (the job's
/// [`molecule`](Job::molecule)) plus the `product` it reacts to. When this is set on a
/// `transition_state` job, the search no longer starts from a single near-saddle guess
/// — instead the job constructs one between the two minima and refines it.
///
/// Two construction routes:
/// - the default builds a single image-dependent-pair-potential (IDPP) guess between
///   the endpoints (cheap, no extra surface evaluations) and seeds the saddle search
///   with the forming/breaking-bond reaction coordinate it reports;
/// - [`use_neb`](Self::use_neb) instead relaxes a whole climbing-image NEB band onto
///   the minimum-energy path and refines its climbing image (robust when no good
///   single guess exists, at the cost of the band relaxation).
///
/// Either way the guess shares the reactant's atom count and composition, so the job's
/// memory estimate and resource guardrails are unchanged. `#[non_exhaustive]`;
/// construct via [`TsGuessInput::new`] and set the fields you need.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TsGuessInput {
    /// The product minimum. Must hold the same atom multiset as the reactant (the IDPP
    /// route maps atoms by connectivity; the NEB route requires identical atom
    /// ordering for now).
    pub product: Molecule,
    /// Relax a climbing-image NEB band between the endpoints instead of building a
    /// single IDPP guess. More robust for floppy or bimolecular reactions; more
    /// expensive (one gradient per interior image per band iteration).
    pub use_neb: bool,
    /// NEB band controls, consulted only when [`use_neb`](Self::use_neb) is set.
    pub neb_options: NebOptions,
    /// IDPP guess-builder controls, consulted on the default (non-NEB) route.
    pub guess_options: GuessOptions,
    /// On the IDPP route (`use_neb = false`), place the guess at the *energy* maximum of
    /// the interpolated path instead of a fixed interpolation fraction: `Some(n)` scans
    /// `n` path points (must be ≥ 3), evaluating the SCF surface, and parabola-fits the
    /// peak — a better single-point guess at the cost of `n` extra single-point energies.
    /// `None` (the default) uses the single geometric IDPP image. Ignored when
    /// [`use_neb`](Self::use_neb) is set (the band already finds the peak).
    pub scan_points: Option<usize>,
}

impl TsGuessInput {
    /// A two-endpoint TS input for `product`, defaulting to the single-IDPP-guess route
    /// ([`use_neb`](Self::use_neb) `= false`) with default band/guess controls.
    pub fn new(product: Molecule) -> Self {
        Self {
            product,
            use_neb: false,
            neb_options: NebOptions::default(),
            guess_options: GuessOptions::default(),
            scan_points: None,
        }
    }
}

/// A distinguished-coordinate (relaxed) scan for a transition-state search: drive one
/// internal coordinate of the job's [`molecule`](Job::molecule) across a value range,
/// relaxing every other coordinate at each grid point, and refine the saddle from the
/// energy-peaked relaxed geometry. Single-ended — it needs no product, unlike the IDPP /
/// NEB routes in [`TsGuessInput`]. `#[non_exhaustive]`; construct via [`CoordScanSpec::new`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CoordScanSpec {
    /// The internal coordinate to drive (a bond, valence angle, or torsion), keyed by its
    /// atom indices into [`molecule`](Job::molecule).
    pub coordinate: Internal,
    /// Start of the driven coordinate's value range (Bohr for a bond, radians for an
    /// angle/dihedral).
    pub start: f64,
    /// End of the driven coordinate's value range (same units as [`start`](Self::start)).
    pub end: f64,
    /// Number of grid points across `[start, end]` inclusive (must be ≥ 3).
    pub n_points: usize,
}

impl CoordScanSpec {
    /// A scan of `coordinate` over `[start, end]` with `n_points` grid points.
    pub fn new(coordinate: Internal, start: f64, end: f64, n_points: usize) -> Self {
        Self {
            coordinate,
            start,
            end,
            n_points,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobOptions {
    pub all_electron: bool,
    pub direct: bool,
    pub ri: bool,
    pub compute_properties: bool,
    pub compute_frequencies: bool,
    pub single_point_hessian: bool,
    pub optimize_geometry: bool,
    /// Request a transition-state (saddle-point) search instead of a relaxation;
    /// gated to the same gradient-capable method/backend combinations as
    /// `optimize_geometry`. Mutually exclusive with it.
    pub transition_state: bool,
    pub symmetry_number: u32,
    pub qrrho_w0_cm1: f64,
    pub grid_level: usize,
    pub dispersion: Option<Dispersion>,
    pub solvent_eps: Option<f64>,
    pub smd: Option<String>,
    pub alpb: Option<String>,
    pub gbsa: Option<String>,
    pub cosmo_file: Option<std::path::PathBuf>,
    pub gcp: Option<crate::disp::GcpParams>,
    pub srb: Option<crate::disp::SrbParams>,
    pub smearing: Option<Smearing>,
    pub fod: bool,
    pub fod_cube: Option<std::path::PathBuf>,
    pub ri_mp2: bool,
    pub cosx: bool,
    pub x2c: bool,
    /// Knobs for the transition-state search (algorithm, trust radii, IRC, ...);
    /// only consulted when `transition_state` is set. See [`TsOptions`].
    pub ts_options: TsOptions,
    /// Optional two-endpoint input for the transition-state search: when `Some` (and
    /// `transition_state` is set), `molecule` is the reactant and this carries the
    /// product. The job builds a near-saddle guess between the two minima — a single
    /// IDPP guess, or a climbing-image NEB band — seeds the reaction coordinate, and
    /// refines the saddle, instead of starting from `molecule` as a single guess. The
    /// guess shares the reactant's composition, so the memory estimate is unchanged.
    /// See [`TsGuessInput`].
    pub ts_guess: Option<TsGuessInput>,
    /// Optional single-ended distinguished-coordinate scan for the transition-state
    /// search: when `Some` (and `transition_state` is set), one internal coordinate of
    /// `molecule` is driven across a range, the rest relaxed at each grid point, and the
    /// saddle refined from the energy peak. Mutually exclusive with [`ts_guess`](Self::ts_guess)
    /// (which is the two-endpoint route). See [`CoordScanSpec`].
    pub ts_coord_scan: Option<CoordScanSpec>,
    /// Cap the rayon worker count for this job. When `Some(k)` with `k >= 1`,
    /// [`Job::run`] runs the whole job (the pre-flight memory estimate and the
    /// solve) inside a scoped `k`-thread pool, so the library's internal data
    /// parallelism is bounded without touching the process-global pool — letting
    /// a host run several jobs with independent core budgets. `None` (and
    /// `Some(0)`) inherit rayon's default global pool. The count is passed
    /// straight to rayon: a very large `k` will attempt to spawn that many OS
    /// threads, so callers are responsible for choosing a sensible value.
    pub n_threads: Option<usize>,
    /// Soft peak-memory budget in bytes. When `Some(limit)`, [`Job::run`] first
    /// computes [`crate::estimate_memory`]; if the selected backend's estimate
    /// exceeds `limit` it either auto-switches to the integral-direct backend
    /// (when that is a valid, lower-memory substitute) or refuses with an
    /// actionable error — *before* the SCF allocates. `None` disables the check.
    /// An auto-switch is reported, not silent: it is recorded in
    /// [`JobResult::backend_downgrade`] and added as a method warning.
    pub mem_budget_bytes: Option<u64>,
}

impl Default for JobOptions {
    fn default() -> Self {
        Self {
            all_electron: false,
            direct: false,
            ri: false,
            compute_properties: false,
            compute_frequencies: false,
            single_point_hessian: false,
            optimize_geometry: false,
            transition_state: false,
            symmetry_number: 1,
            qrrho_w0_cm1: crate::props::thermo::QRRHO_W0_DEFAULT_CM1,
            grid_level: 3,
            dispersion: None,
            solvent_eps: None,
            smd: None,
            alpb: None,
            gbsa: None,
            cosmo_file: None,
            gcp: None,
            srb: None,
            smearing: None,
            fod: false,
            fod_cube: None,
            ri_mp2: false,
            cosx: false,
            x2c: false,
            ts_options: TsOptions::default(),
            ts_guess: None,
            ts_coord_scan: None,
            n_threads: None,
            mem_budget_bytes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PostHfResult {
    Mp2 {
        result: Mp2Result,
        n_frozen: usize,
    },
    RiMp2 {
        result: RiMp2Result,
        n_frozen: usize,
        aux_basis: String,
    },
    Ccsd {
        result: CcsdResult,
        n_frozen: usize,
    },
    CcsdT {
        result: CcsdTResult,
        n_frozen: usize,
    },
}

impl PostHfResult {
    pub fn total_energy(&self) -> f64 {
        match self {
            Self::Mp2 { result, .. } => result.total_energy,
            Self::RiMp2 { result, .. } => result.total_energy,
            Self::Ccsd { result, .. } => result.total_energy,
            Self::CcsdT { result, .. } => result.total_energy,
        }
    }

    pub fn converged(&self) -> bool {
        match self {
            Self::Mp2 { .. } | Self::RiMp2 { .. } => true,
            Self::Ccsd { result, .. } => result.converged,
            Self::CcsdT { result, .. } => result.ccsd.converged,
        }
    }

    pub fn n_frozen(&self) -> usize {
        match self {
            Self::Mp2 { n_frozen, .. }
            | Self::RiMp2 { n_frozen, .. }
            | Self::Ccsd { n_frozen, .. }
            | Self::CcsdT { n_frozen, .. } => *n_frozen,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DoubleHybridData {
    pub functional_name: String,
    pub scf_functional_name: String,
    pub e_scf: f64,
    pub e_os: f64,
    pub e_ss: f64,
    pub c_os: f64,
    pub c_ss: f64,
    pub n_frozen: usize,
    pub vv10_scale: f64,
    pub pt2_aux_basis: Option<String>,
}

impl DoubleHybridData {
    pub fn pt2_energy(&self) -> f64 {
        self.pt2_energy_with(self.c_os, self.c_ss)
    }

    pub fn pt2_energy_with(&self, c_os: f64, c_ss: f64) -> f64 {
        c_os * self.e_os + c_ss * self.e_ss
    }
}

#[derive(Debug, Clone)]
pub struct SmdData {
    pub solvent: String,
    pub epsilon: f64,
    pub e_gas: f64,
    pub e_solution: f64,
    pub g_ep: f64,
    pub g_cds: f64,
    pub dg_solv: f64,
}

#[derive(Debug, Clone)]
pub struct GbsaData {
    pub model: &'static str,
    pub solvent: String,
    pub epsilon: f64,
    pub g_born: f64,
    pub g_hb: f64,
    pub g_sasa: f64,
    pub g_shift: f64,
    pub g_solv: f64,
}

#[derive(Debug, Clone)]
pub struct PropertiesResult {
    pub dipole_au: [f64; 3],
    pub population: PopulationAnalysis,
}

#[derive(Debug, Clone)]
pub struct FrequencyData {
    pub frequencies: FrequencyResult,
    pub thermochemistry: ThermoResult,
    pub is_sph: bool,
}

#[derive(Debug, Clone)]
pub struct DftDiagnostics {
    pub functional_name: String,
    pub grid_level: usize,
    pub n_grid_points: usize,
    pub exx_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct CosxDiagnostics {
    pub grid: String,
    pub n_points: usize,
    pub overlap_fitted: bool,
    pub rs_omega: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RiDiagnostics {
    pub aux_basis: String,
    pub naux: usize,
}

/// A report that [`Job::run`] automatically switched the integral backend to fit
/// `mem_budget_bytes`. The downgraded job still ran to completion — this records
/// what changed so a caller can surface it; it never blocks the run.
#[derive(Debug, Clone)]
pub struct BackendDowngrade {
    /// The backend the job's options would have selected.
    pub from: crate::EstimateBackend,
    /// The lower-memory backend actually used to fit the budget.
    pub to: crate::EstimateBackend,
    /// Estimated peak bytes for `from` — the figure that exceeded the budget.
    pub estimated_bytes: u64,
    /// The configured `mem_budget_bytes` the estimate exceeded.
    pub budget_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct JobResult {
    pub scf: ScfResult,
    pub optimized_geometry: Option<OptResult>,
    pub transition_state: Option<TsResult>,
    /// Reactant→product atom-mapping diagnostic for a two-endpoint transition-state
    /// search, when one ran (otherwise `None`). A low confidence flags symmetric or
    /// equivalent atoms the mapping could not uniquely resolve, which a caller can warn on.
    pub mapping_confidence: Option<MappingConfidence>,
    pub post_hf: Option<PostHfResult>,
    pub properties: Option<PropertiesResult>,
    pub frequencies: Option<FrequencyData>,
    pub dft: Option<DftDiagnostics>,
    pub ri: Option<RiDiagnostics>,
    pub cosx: Option<CosxDiagnostics>,
    pub dispersion_energy: Option<f64>,
    pub gcp_energy: Option<f64>,
    pub srb_energy: Option<f64>,
    pub fod: Option<FodResult>,
    pub vv10_energy: Option<f64>,
    pub double_hybrid: Option<DoubleHybridData>,
    pub smd: Option<SmdData>,
    pub gbsa: Option<GbsaData>,
    pub method_warnings: Vec<String>,
    /// Set when [`Job::run`] automatically changed the integral backend to fit
    /// `mem_budget_bytes`. The job still ran (a report, not a block); `None`
    /// means no budget-driven downgrade happened. See [`BackendDowngrade`].
    pub backend_downgrade: Option<BackendDowngrade>,
}

impl JobResult {
    pub fn best_energy(&self) -> f64 {
        let base = match (&self.post_hf, &self.double_hybrid) {
            (Some(p), _) => p.total_energy(),
            (None, Some(dh)) => dh.e_scf + dh.pt2_energy(),
            (None, None) => self.scf.energy,
        };
        base + self.dispersion_energy.unwrap_or(0.0)
            + self.gcp_energy.unwrap_or(0.0)
            + self.srb_energy.unwrap_or(0.0)
            + self.vv10_energy.unwrap_or(0.0)
            + self.smd.as_ref().map_or(0.0, |s| s.g_cds)
            + self.gbsa.as_ref().map_or(0.0, |g| g.g_solv)
    }

    pub fn converged(&self) -> bool {
        let scf_ok = self.scf.converged;
        let post_ok = self.post_hf.as_ref().is_none_or(|p| p.converged());
        let opt_ok = self.optimized_geometry.as_ref().is_none_or(|o| o.converged);
        let ts_ok = self.transition_state.as_ref().is_none_or(|t| t.converged());
        scf_ok && post_ok && opt_ok && ts_ok
    }
}

#[derive(Clone)]
pub struct Job {
    pub molecule: Molecule,
    pub basis: String,
    pub method: Method,
    pub options: JobOptions,
}

impl Job {
    /// Run the job to completion, honoring the optional resource controls in
    /// [`JobOptions`]. A thread cap (`n_threads`) installs a scoped rayon pool
    /// around the whole job; inside it a memory budget (`mem_budget_bytes`) is
    /// resolved — it may transparently switch to the integral-direct backend or
    /// refuse the job. With neither set this is a thin pass-through to the solver.
    pub fn run(&self) -> Result<JobResult, String> {
        // Install the optional scoped pool around BOTH the pre-flight estimate
        // and the solve, so the thread cap bounds all of this job's parallelism.
        match self.options.n_threads {
            Some(threads) if threads >= 1 => rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| format!("could not build a {threads}-thread pool: {e}"))?
                .install(|| self.run_planned()),
            _ => self.run_planned(),
        }
    }

    /// Resolve the memory budget (possibly downgrading the backend or refusing),
    /// then run the solver core. Runs inside the scoped pool set up by [`Self::run`].
    fn run_planned(&self) -> Result<JobResult, String> {
        let (planned, downgrade) = self.plan_for_budget()?;
        let mut result = planned.run_core()?;
        if let Some(report) = downgrade {
            // Report the automatic backend change two ways without blocking the
            // run (it already completed): a human-readable method warning that
            // summaries render, and a structured field a caller can branch on.
            result.method_warnings.push(format!(
                "estimated {} for the {} backend exceeded the {} mem_budget_bytes; \
                 automatically switched to the {} backend to fit the budget",
                crate::estimate::human_bytes(report.estimated_bytes),
                report.from,
                crate::estimate::human_bytes(report.budget_bytes),
                report.to,
            ));
            result.backend_downgrade = Some(report);
        }
        Ok(result)
    }

    /// Resolve a memory-budget-aware execution plan: the job unchanged when no
    /// budget is set or the estimate fits; an owned integral-direct clone when
    /// downgrading is a valid way to fit; or an error when it cannot be made to
    /// fit. Runs before any heavy allocation so an over-budget job is refused
    /// rather than left to OOM.
    fn plan_for_budget(
        &self,
    ) -> Result<(std::borrow::Cow<'_, Self>, Option<BackendDowngrade>), String> {
        let Some(budget) = self.options.mem_budget_bytes else {
            return Ok((std::borrow::Cow::Borrowed(self), None));
        };
        let estimate = crate::estimate_memory(self)?;
        if estimate.peak_bytes <= budget {
            return Ok((std::borrow::Cow::Borrowed(self), None));
        }
        if self.direct_downgrade_eligible() {
            let mut downgraded = self.clone();
            downgraded.options.direct = true;
            // Clear the budget on the clone so its own run does not re-plan.
            downgraded.options.mem_budget_bytes = None;
            if let Ok(direct) = crate::estimate_memory(&downgraded)
                && direct.peak_bytes <= budget
            {
                let report = BackendDowngrade {
                    from: estimate.backend,
                    to: direct.backend,
                    estimated_bytes: estimate.peak_bytes,
                    budget_bytes: budget,
                };
                return Ok((std::borrow::Cow::Owned(downgraded), Some(report)));
            }
        }
        Err(format!(
            "estimated peak memory {} exceeds the {} budget for the {} backend; reduce the \
             basis set, or select a lower-memory backend (--direct for SCF-level energies, \
             --ri for density-fitted Coulomb)",
            crate::estimate::human_bytes(estimate.peak_bytes),
            crate::estimate::human_bytes(budget),
            estimate.backend,
        ))
    }

    /// Whether a job currently bound for the conventional in-core backend could
    /// instead run on the integral-direct backend with an equivalent result.
    /// Mirrors the integral-direct capability gates in `run_inner`; when in
    /// doubt it returns `false`, so the budget path refuses rather than silently
    /// running an unsupported combination.
    fn direct_downgrade_eligible(&self) -> bool {
        let o = &self.options;
        // Only the conventional path downgrades; --ri/--direct already chose.
        if o.ri || o.direct {
            return false;
        }
        // The integral-direct backend supports SCF-level single points only.
        if o.optimize_geometry
            || o.transition_state
            || o.compute_properties
            || o.compute_frequencies
            || o.fod
            || o.cosx
            || o.cosmo_file.is_some()
        {
            return false;
        }
        if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
            return false;
        }
        // Double hybrids, range-separated (CAM), and VV10 functionals all need a
        // backend that stores its integrals.
        if let Method::Dft(spec) = &self.method
            && (spec.double_hybrid().is_some() || spec.cam().is_some() || spec.vv10().is_some())
        {
            return false;
        }
        true
    }

    fn run_core(&self) -> Result<JobResult, String> {
        let mut result = self.run_inner()?;
        result.method_warnings = crate::guardrails::assess_job(self);
        if let Some(name) = &self.options.smd
            && result.converged()
        {
            let solvent = resolve_smd_solvent(name)?;
            let mol = match &result.optimized_geometry {
                Some(opt) => {
                    let atoms = self
                        .molecule
                        .atoms
                        .iter()
                        .zip(&opt.positions)
                        .map(|(a, p)| crate::core::Atom::new(a.element, *p))
                        .collect();
                    Molecule::new(atoms, self.molecule.charge, self.molecule.multiplicity)
                }
                None => self.molecule.clone(),
            };
            let zs: Vec<usize> = mol.atoms.iter().map(|a| a.element.z() as usize).collect();
            let coords: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
            let g_cds =
                crate::solv::cds_energy(&zs, &coords, solvent, crate::solv::smd::DEFAULT_SASA_GRID)
                    .map_err(|e| e.to_string())?;
            let gas = Job {
                molecule: mol,
                basis: self.basis.clone(),
                method: self.method.clone(),
                options: JobOptions {
                    smd: None,
                    solvent_eps: None,
                    optimize_geometry: false,
                    compute_properties: false,
                    compute_frequencies: false,
                    dispersion: None,
                    gcp: None,
                    srb: None,
                    fod: false,
                    fod_cube: None,
                    ..self.options.clone()
                },
            }
            .run_inner()?;
            if !gas.scf.converged {
                return Err("SMD gas-phase reference SCF did not converge".into());
            }
            let g_ep = result.scf.energy - gas.scf.energy;
            result.smd = Some(SmdData {
                solvent: solvent.name.to_string(),
                epsilon: solvent.epsilon,
                e_gas: gas.scf.energy,
                e_solution: result.scf.energy,
                g_ep,
                g_cds,
                dg_solv: g_ep + g_cds,
            });
        }

        if (self.options.alpb.is_some() || self.options.gbsa.is_some()) && result.scf.converged {
            let (params, model_label) = if let Some(name) = &self.options.alpb {
                (resolve_alpb_solvent(name)?, "ALPB")
            } else {
                (
                    resolve_gbsa_solvent(self.options.gbsa.as_ref().unwrap())?,
                    "GBSA",
                )
            };
            let mol = &self.molecule;
            let ao = BasisSet::load(&self.basis)
                .map_err(|e| e.to_string())?
                .build(mol)
                .map_err(|e| e.to_string())?;
            let setup = ecp_setup(mol, &ao);
            let provider =
                ConventionalProvider::new(ao.into_integral(), setup.charges).with_ecps(setup.ecps);
            let pop = population_analysis(
                &provider,
                mol,
                &result.scf.density_alpha,
                &result.scf.density_beta,
            );
            let zs: Vec<usize> = mol.atoms.iter().map(|a| a.element.z() as usize).collect();
            let coords: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
            let bd = crate::solv::gbsa_energy(
                params,
                &zs,
                &coords,
                &pop.mulliken_charges,
                crate::solv::DEFAULT_GBSA_GRID,
            )
            .map_err(|e| e.to_string())?;
            result.gbsa = Some(GbsaData {
                model: model_label,
                solvent: params.name.to_string(),
                epsilon: params.epsv,
                g_born: bd.g_born,
                g_hb: bd.g_hb,
                g_sasa: bd.g_sasa,
                g_shift: bd.g_shift,
                g_solv: bd.g_solv,
            });
        }
        Ok(result)
    }

    fn run_cosmo_export(
        &self,
        mol: &Molecule,
        n_alpha: usize,
        n_beta: usize,
        reference: Reference,
    ) -> Result<JobResult, String> {
        const BOHR_TO_AA: f64 = 0.529_177_210_903;
        let opts = &self.options;
        let path = opts.cosmo_file.as_ref().expect("cosmo_file path");
        let ao = BasisSet::load(&self.basis)
            .map_err(|e| e.to_string())?
            .build(mol)
            .map_err(|e| e.to_string())?;
        let setup = ecp_setup(mol, &ao);
        let grid_xc = if let Method::Dft(spec) = &self.method {
            Some(GridXc::new(mol, &ao, spec, opts.grid_level).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let dft_diag = grid_xc.as_ref().map(|g| DftDiagnostics {
            functional_name: g.name().to_string(),
            grid_level: g.level(),
            n_grid_points: g.n_points(),
            exx_fraction: g.exx_fraction(),
        });
        let scf_opts = if grid_xc.is_some() {
            ScfOptions {
                energy_tol: 1e-9,
                error_tol: 1e-6,
                ..ScfOptions::default()
            }
        } else {
            ScfOptions::default()
        };
        let xc_ref = grid_xc.as_ref().map(|g| g as &dyn XcContributor);
        let provider =
            ConventionalProvider::new(ao.into_integral(), setup.charges).with_ecps(setup.ecps);
        let eps = f64::INFINITY;
        let cpcm =
            Cpcm::new(&provider, mol, eps, crate::solv::DEFAULT_GRID).map_err(|e| e.to_string())?;
        let scf = run_scf_with_env(
            &provider,
            n_alpha,
            n_beta,
            reference,
            setup.nuclear_repulsion,
            &scf_opts,
            xc_ref,
            Some(&cpcm as &dyn SolventModel),
        )
        .map_err(|e| e.to_string())?;

        if scf.converged {
            let (segments, dielectric_energy) = cpcm.cosmo_segments(&scf.density, scf.n_basis);
            let atoms = mol
                .atoms
                .iter()
                .map(|a| {
                    let r = crate::solv::cavity_radius(a.element.z() as usize)
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>(crate::solv::CosmoAtom {
                        symbol: a.element.symbol().to_string(),
                        position: a.position,
                        radius: r * BOHR_TO_AA,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let export = crate::solv::CosmoExport {
                epsilon: eps,
                total_energy: scf.energy,
                dielectric_energy,
                atoms,
                segments,
            };
            std::fs::write(path, crate::solv::write_cosmo(&export))
                .map_err(|e| format!("writing COSMO file {}: {e}", path.display()))?;
        }

        Ok(JobResult {
            method_warnings: Vec::new(),
            backend_downgrade: None,
            scf,
            optimized_geometry: None,
            transition_state: None,
            mapping_confidence: None,
            post_hf: None,
            properties: None,
            frequencies: None,
            dft: dft_diag,
            ri: None,
            cosx: None,
            dispersion_energy: None,
            gcp_energy: None,
            srb_energy: None,
            fod: None,
            vv10_energy: None,
            double_hybrid: None,
            smd: None,
            gbsa: None,
        })
    }

    fn run_inner(&self) -> Result<JobResult, String> {
        let mol = &self.molecule;
        let opts = &self.options;
        let n_solv = [
            opts.solvent_eps.is_some(),
            opts.smd.is_some(),
            opts.alpb.is_some(),
            opts.gbsa.is_some(),
            opts.cosmo_file.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count();
        if n_solv > 1 {
            return Err(
                "the solvation models are mutually exclusive: choose at most one of \
                 --solvent/--eps (C-PCM), --smd, --alpb, --gbsa, or --cosmo-file"
                    .into(),
            );
        }
        if let Some(name) = &opts.smd {
            resolve_smd_solvent(name)?;
        }
        if let Some(name) = &opts.alpb {
            resolve_alpb_solvent(name)?;
        }
        if let Some(name) = &opts.gbsa {
            resolve_gbsa_solvent(name)?;
        }
        if opts.alpb.is_some() || opts.gbsa.is_some() {
            let model = if opts.alpb.is_some() { "ALPB" } else { "GBSA" };
            if opts.optimize_geometry {
                return Err(format!(
                    "{model} is a post-SCF single-point correction: geometry \
                     optimization is not supported (no {model} gradient on ab-initio charges)"
                ));
            }
            if opts.compute_frequencies {
                return Err(format!(
                    "{model} is a post-SCF single-point correction: frequencies are \
                     not supported"
                ));
            }
        }
        let solvated = opts.solvent_eps.is_some()
            || opts.smd.is_some()
            || opts.alpb.is_some()
            || opts.gbsa.is_some()
            || opts.cosmo_file.is_some();

        if mol.has_ghosts() {
            if mol.n_real_atoms() == 0 {
                return Err(
                    "ghost-only molecule: every atom is a ghost (basis functions only); \
                     at least one real atom is required"
                        .into(),
                );
            }
            if opts.optimize_geometry {
                return Err("geometry optimization with ghost atoms is not supported \
                     (the internal-coordinate model assumes real nuclei); run single points"
                    .into());
            }
            if opts.compute_frequencies {
                return Err(
                    "frequencies with ghost atoms are not supported (RRHO mass-weighting \
                     assumes real nuclei on every center)"
                        .into(),
                );
            }
            if opts.compute_properties {
                return Err("properties with ghost atoms are not supported (nuclear \
                     dipole and Mulliken charges need the ghost-aware Z convention)"
                    .into());
            }
            if solvated {
                return Err(
                    "implicit solvation (C-PCM/SMD) with ghost atoms is not supported this \
                     round (the cavity construction assumes real atoms)"
                        .into(),
                );
            }
        }

        if let Method::Dft(spec) = &self.method
            && spec.double_hybrid().is_some()
        {
            let name = spec.name();
            if mol.multiplicity > 1 {
                return Err(format!(
                    "{name} is a double hybrid: only closed-shell (RKS) references are \
                     supported (got multiplicity {}); open-shell double hybrids \
                     are out of scope",
                    mol.multiplicity
                ));
            }
            if opts.optimize_geometry {
                return Err(format!(
                    "{name} is a double hybrid: geometry optimization is not supported \
                     (no PT2 gradient)"
                ));
            }
            if opts.transition_state {
                return Err(format!(
                    "{name} is a double hybrid: transition-state search is not supported \
                     (no PT2 gradient)"
                ));
            }
            if opts.compute_frequencies {
                return Err(format!(
                    "{name} is a double hybrid: frequencies are not supported (no PT2 gradient)"
                ));
            }
            if opts.direct {
                return Err(format!(
                    "{name} is a double hybrid: --direct is not supported (the PT2 step \
                     needs the conventional in-core ERI tensor)"
                ));
            }
            if opts.ri {
                return Err(format!(
                    "{name} is a double hybrid: --ri is not supported (the PT2 step needs \
                     the conventional in-core ERI tensor)"
                ));
            }
            if opts.cosx {
                return Err(format!(
                    "{name} is a double hybrid: --cosx is not supported (the PT2 step needs \
                     orbitals from exact exchange)"
                ));
            }
            if opts.smearing.is_some() {
                return Err(format!(
                    "{name} is a double hybrid: Fermi smearing is not supported (fractional \
                     occupations have no single-determinant PT2 treatment)"
                ));
            }
            if opts.fod {
                return Err(format!(
                    "{name} is a double hybrid: FOD analysis is not supported (the smeared \
                     diagnostic has no PT2 counterpart)"
                ));
            }
            if solvated {
                return Err(format!(
                    "{name} is a double hybrid: implicit solvation (C-PCM/SMD) is not \
                     supported (the PT2 step on solvated KS orbitals is unvalidated)"
                ));
            }
        }

        if let Method::Dft(spec) = &self.method {
            let rs = spec.cam().is_some();
            let vv10 = spec.vv10().is_some();
            if rs || vv10 {
                let what = match (rs, vv10) {
                    (true, true) => "range-separated (CAM) and VV10-carrying",
                    (true, false) => "range-separated (CAM)",
                    _ => "VV10-carrying",
                };
                let name = spec.name();
                if opts.direct {
                    return Err(format!(
                        "{name} is {what}: --direct is not supported (erf-attenuated exchange \
                         and the VV10 evaluation need a backend that stores its integrals)"
                    ));
                }
                if rs && opts.ri && !opts.cosx {
                    return Err(format!(
                        "{name} is range-separated (CAM): --ri alone is not supported (the \
                         RI-JK backend has no erf-attenuated long-range exchange); add --cosx \
                         to serve K semi-numerically over the RI-J Coulomb, or use the \
                         conventional in-core backend"
                    ));
                }
                if vv10 && opts.optimize_geometry {
                    return Err(format!(
                        "{name} is VV10-carrying: geometry optimization is not supported (the \
                         nonlocal VV10 energy E_nl has no gradient, so the optimizer would \
                         converge on a surface missing E_nl)"
                    ));
                }
                if vv10 && opts.compute_frequencies {
                    return Err(format!(
                        "{name} is VV10-carrying: frequencies are not supported (the nonlocal \
                         VV10 energy E_nl has no gradient, so the Hessian would differentiate \
                         a surface missing E_nl)"
                    ));
                }
            }
        }
        if opts.x2c {
            if opts.optimize_geometry {
                return Err(
                    "X2C is energy-only: geometry optimization is not supported \
                     (no X2C analytic gradient; the picture-change gradient terms are \
                     unimplemented)"
                        .into(),
                );
            }
            if opts.compute_frequencies {
                return Err("X2C is energy-only: frequencies are not supported (no X2C \
                     analytic gradient)"
                    .into());
            }
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err("X2C with post-HF methods is not supported (the correlated \
                     treatment of X2C orbitals is unvalidated); use an SCF-level method"
                    .into());
            }
            if let Method::Dft(spec) = &self.method
                && spec.double_hybrid().is_some()
            {
                return Err(format!(
                    "{} is a double hybrid: X2C is not supported (the PT2 step \
                     on X2C orbitals is unvalidated)",
                    spec.name()
                ));
            }
        }

        if opts.cosx {
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err(
                    "COSX applies to SCF-level methods only (HF and DFT functionals), not \
                     post-HF (the correlated steps need exact-exchange orbitals and in-core ERIs)"
                        .into(),
                );
            }
            if opts.direct {
                return Err(
                    "--cosx with --direct is not wired; use COSX with the default \
                     in-core backend or with --ri"
                        .into(),
                );
            }
            if opts.optimize_geometry {
                return Err(
                    "COSX is energy-only: geometry optimization is not supported (no COSX \
                     gradient)"
                        .into(),
                );
            }
            if opts.compute_frequencies {
                return Err(
                    "COSX is energy-only: frequencies are not supported (no COSX gradient)".into(),
                );
            }
            if opts.fod {
                return Err(
                    "FOD analysis with COSX is not supported (the diagnostic is calibrated on \
                     exact exchange); run --fod without --cosx"
                        .into(),
                );
            }
        }
        if opts.ri_mp2 {
            let dh = matches!(&self.method, Method::Dft(spec) if spec.double_hybrid().is_some());
            if !matches!(self.method, Method::Mp2) && !dh {
                return Err(
                    "ri_mp2 (RI-MP2) applies to the MP2 method only (--method mp2), or to the \
                     PT2 step of a double-hybrid functional"
                        .into(),
                );
            }
            if opts.direct {
                return Err(
                    "ri_mp2 with --direct is not supported; use the default in-core \
                     backend or --ri for the SCF step"
                        .into(),
                );
            }
        }
        if opts.ri {
            if opts.direct {
                return Err(
                    "--ri and --direct are contradictory: density fitting builds and stores \
                     the fitted B tensor, the direct backend stores nothing"
                        .into(),
                );
            }
            if opts.optimize_geometry {
                return Err("the RI-JK backend does not support geometry optimization".into());
            }
            let ri_mp2_exception = matches!(self.method, Method::Mp2) && opts.ri_mp2;
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT)
                && !ri_mp2_exception
            {
                return Err(
                    "the RI-JK backend does not support post-HF methods (needs in-core ERI); \
                     RI-MP2 (ri_mp2 / --ri-mp2) is the density-fitted exception"
                        .into(),
                );
            }
            if opts.compute_properties || opts.compute_frequencies {
                return Err("the RI-JK backend does not support properties or frequencies".into());
            }
        }
        if opts.direct {
            if opts.optimize_geometry {
                return Err(
                    "integral-direct backend does not support geometry optimization".into(),
                );
            }
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err(
                    "integral-direct backend does not support post-HF methods (needs in-core ERI)"
                        .into(),
                );
            }
            if opts.compute_properties || opts.compute_frequencies {
                return Err(
                    "integral-direct backend does not support properties or frequencies".into(),
                );
            }
        }
        if opts.compute_frequencies
            && !matches!(self.method, Method::Rhf | Method::Uhf | Method::Dft(_))
        {
            return Err(
                "vibrational frequencies require a method with a gradient path (RHF, UHF, or a \
                 DFT functional); post-HF and ROHF have no analytic gradient"
                    .into(),
            );
        }
        if opts.optimize_geometry && opts.transition_state {
            return Err(
                "geometry optimization and transition-state search are mutually exclusive: \
                 request at most one"
                    .into(),
            );
        }
        if opts.ts_guess.is_some() && !opts.transition_state {
            return Err(
                "a transition-state product endpoint was given without requesting a \
                 transition-state search: set transition_state (the CLI --ts flag)"
                    .into(),
            );
        }
        if opts.ts_coord_scan.is_some() && !opts.transition_state {
            return Err(
                "a distinguished-coordinate scan was given without requesting a \
                 transition-state search: set transition_state (the CLI --ts flag)"
                    .into(),
            );
        }
        if opts.ts_coord_scan.is_some() && opts.ts_guess.is_some() {
            return Err(
                "a distinguished-coordinate scan and a product endpoint are mutually \
                 exclusive: choose the single-ended scan or the two-endpoint route"
                    .into(),
            );
        }
        if opts.cosmo_file.is_some() && (opts.optimize_geometry || opts.transition_state) {
            return Err(
                "COSMO file export cannot be combined with geometry optimization or \
                 transition-state search: request one at a time"
                    .into(),
            );
        }
        if opts.optimize_geometry
            && matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT)
        {
            return Err(
                "geometry optimization is not supported for post-HF methods (no analytic CC gradient)".into(),
            );
        }
        if opts.dispersion.is_some()
            && matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT)
        {
            return Err(
                "dispersion corrections apply to SCF-level methods only (HF and DFT functionals), not post-HF".into(),
            );
        }
        if solvated {
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err(
                    "implicit solvation (C-PCM/SMD) applies to SCF-level methods only (HF and DFT functionals), not post-HF".into(),
                );
            }
            if opts.compute_frequencies {
                return Err(
                    "frequencies in solvent are not supported (the numerical Hessian of the \
                     finite-difference-effective solvated surface is noise-prone); run --freq \
                     in gas phase"
                        .into(),
                );
            }
        }

        if opts.fod_cube.is_some() && !opts.fod {
            return Err("fod_cube requires the FOD analysis itself (set fod = true)".into());
        }
        if opts.fod {
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err(
                    "FOD analysis applies to SCF-level methods only (HF and DFT functionals), \
                     not post-HF (the diagnostic is defined on the smeared mean-field \
                     occupations)"
                        .into(),
                );
            }
            if opts.direct || opts.ri {
                return Err(
                    "FOD analysis requires the conventional in-core backend (not --direct/--ri)"
                        .into(),
                );
            }
        }
        let fod_temperature = opts.fod.then(|| match opts.smearing {
            Some(Smearing::Fermi { temperature_k }) => temperature_k,
            None => {
                let a_x = match &self.method {
                    Method::Dft(spec) => spec.exx_fraction(),
                    _ => 1.0, // Hartree–Fock: full exact exchange
                };
                crate::dft::fod_default_temperature(a_x)
            }
        });
        let smearing = opts
            .smearing
            .or(fod_temperature.map(|temperature_k| Smearing::Fermi { temperature_k }));

        if smearing.is_some() {
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err(
                    "Fermi smearing applies to SCF-level methods only (HF and DFT functionals), \
                     not post-HF (fractional occupations have no single-determinant correlation \
                     treatment)"
                        .into(),
                );
            }
            if opts.optimize_geometry {
                return Err(
                    "Fermi smearing is energy-only: geometry optimization with smearing is not \
                     supported (no smeared gradient)"
                        .into(),
                );
            }
            if opts.compute_frequencies {
                return Err(
                    "Fermi smearing is energy-only: frequencies with smearing are not supported \
                     (no smeared gradient)"
                        .into(),
                );
            }
            if matches!(self.method, Method::Rohf) {
                return Err(
                    "Fermi smearing requires the RHF or UHF reference (RKS/UKS for DFT); ROHF is \
                     not supported"
                        .into(),
                );
            }
        }

        let basis_set = BasisSet::load(&self.basis).map_err(|e| e.to_string())?;
        let ecp_core = basis_set.ecp_core_electrons(mol) as i64;
        if ecp_core > 0 {
            let what = ecp_summary(mol, &basis_set)
                .iter()
                .map(|(sym, z, nc)| format!("{sym} (Z={z}, ECP-{nc})"))
                .collect::<Vec<_>>()
                .join(", ");
            if opts.x2c {
                return Err(format!(
                    "X2C with ECP atoms ({what}) double-counts relativity: the ECP already \
                     folds scalar-relativistic core effects into the potential; use an \
                     all-electron basis with --x2c, or drop --x2c and keep the ECP"
                ));
            }
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err(format!(
                    "post-HF methods with ECP atoms ({what}) are not supported \
                     (the frozen-core convention for ECP cores is unvalidated)"
                ));
            }
            if let Method::Dft(spec) = &self.method
                && spec.double_hybrid().is_some()
            {
                return Err(format!(
                    "double hybrids with ECP atoms ({what}) are not supported \
                     (the PT2 step's frozen-core convention for ECP cores is unvalidated)"
                ));
            }
            if opts.compute_properties {
                return Err(format!(
                    "properties with ECP atoms ({what}) are not supported \
                     (nuclear dipole and Mulliken charges need the effective-Z convention)"
                ));
            }
            if solvated {
                return Err(format!(
                    "implicit solvation (C-PCM/SMD) with ECP atoms ({what}) is not supported \
                     (the nuclear surface potential needs the effective-Z convention)"
                ));
            }
            if opts.fod {
                return Err(format!(
                    "FOD analysis with ECP atoms ({what}) is not supported \
                     (the diagnostic is calibrated on all-electron references)"
                ));
            }
            if opts.ri {
                return Err(format!(
                    "RI-JK with ECP atoms ({what}) is not supported: the                      def2-universal-jkfit auxiliary set is vendored for H-Kr only"
                ));
            }
        }

        let (n_alpha, n_beta) = alpha_beta_electrons(mol, ecp_core)?;

        let reference = method_reference(&self.method, mol.multiplicity);
        if reference == Reference::Rhf && n_alpha != n_beta {
            return Err("RHF requires a closed shell; use Method::Uhf or Method::Rohf".into());
        }

        if opts.cosmo_file.is_some() {
            if opts.direct || opts.ri {
                return Err(
                    "--cosmo-file uses the conventional in-core backend (not --direct/--ri)".into(),
                );
            }
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err(
                    "--cosmo-file applies to SCF-level methods only (HF and DFT functionals)"
                        .into(),
                );
            }
            return self.run_cosmo_export(mol, n_alpha, n_beta, reference);
        }

        if let Method::Dft(spec) = &self.method
            && spec.double_hybrid().is_some()
        {
            if n_alpha != n_beta {
                return Err("double hybrids require a closed shell (RKS reference)".into());
            }
            return self.run_double_hybrid(mol, n_alpha, n_beta);
        }

        if opts.optimize_geometry {
            let opt_opts = OptOptions::default();
            let mut surface = if let Method::Dft(spec) = &self.method {
                HfSurface::new_dft(mol, &self.basis, reference, spec.clone(), opts.grid_level)?
            } else {
                HfSurface::new(mol, &self.basis, reference)?
            };
            if let Some(disp) = opts.dispersion {
                surface.set_dispersion(disp);
            }
            if let Some(gcp) = opts.gcp {
                surface.set_gcp(gcp);
            }
            if let Some(srb) = opts.srb {
                surface.set_srb(srb);
            }
            if let Some(eps) = opts.solvent_eps {
                surface.set_solvent(eps);
            }
            if let Some(name) = &opts.smd {
                surface.set_smd(*resolve_smd_solvent(name)?);
            }
            let opt = optimize(mol, &mut surface, &opt_opts).map_err(|e| opt_error_message(&e))?;
            let scf = surface
                .last_scf()
                .cloned()
                .ok_or("surface has no cached SCF after optimization")?;
            let final_mol = (opts.dispersion.is_some() || opts.gcp.is_some() || opts.srb.is_some())
                .then(|| {
                    let atoms = mol
                        .atoms
                        .iter()
                        .zip(&opt.positions)
                        .map(|(a, p)| crate::core::Atom::new(a.element, *p))
                        .collect();
                    Molecule::new(atoms, mol.charge, mol.multiplicity)
                });
            let dispersion_energy = opts
                .dispersion
                .map(|disp| disp.energy(final_mol.as_ref().unwrap()));
            let gcp_energy = opts
                .gcp
                .map(|p| crate::disp::gcp_energy(final_mol.as_ref().unwrap(), &p));
            let srb_energy = opts
                .srb
                .map(|p| crate::disp::srb_energy(final_mol.as_ref().unwrap(), &p));
            return Ok(JobResult {
                method_warnings: Vec::new(),
                backend_downgrade: None,
                scf,
                optimized_geometry: Some(opt),
                transition_state: None,
                mapping_confidence: None,
                post_hf: None,
                properties: None,
                frequencies: None,
                dft: None,
                ri: None,
                cosx: None,
                dispersion_energy,
                gcp_energy,
                srb_energy,
                fod: None,
                vv10_energy: None,
                double_hybrid: None,
                smd: None,
                gbsa: None,
            });
        }

        if opts.transition_state {
            // A transition-state search drives the same energy+gradient `Surface`
            // as `optimize`, so it is gated to a SUBSET of the gradient-capable
            // combinations `optimize_geometry` allows: TS additionally rejects
            // implicit solvent, because the saddle search relies on the analytic
            // gradient path that the solvated surface does not expose (see
            // `surface.rs` `analytic_gradient`, which returns `None` for solvent),
            // whereas relaxation tolerates the finite-difference fallback. The
            // mutual-exclusion and double-hybrid gates live in the validation
            // prologue above, alongside their `optimize_geometry` counterparts.
            if matches!(self.method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err(
                    "transition-state search is not supported for post-HF methods \
                     (no analytic CC gradient)"
                        .into(),
                );
            }
            if opts.ri {
                return Err("the RI-JK backend does not support transition-state search".into());
            }
            if opts.direct {
                return Err(
                    "integral-direct backend does not support transition-state search".into(),
                );
            }
            if opts.cosx {
                return Err(
                    "COSX is energy-only: transition-state search is not supported \
                     (no COSX gradient)"
                        .into(),
                );
            }
            if opts.x2c {
                return Err(
                    "X2C is energy-only: transition-state search is not supported \
                     (no X2C analytic gradient)"
                        .into(),
                );
            }
            if opts.smearing.is_some() {
                return Err(
                    "Fermi smearing is energy-only: transition-state search is not \
                     supported (no smeared gradient)"
                        .into(),
                );
            }
            if solvated {
                return Err(
                    "transition-state search in implicit solvent is not supported \
                     (no analytic solvation gradient on this path)"
                        .into(),
                );
            }
            if mol.has_ghosts() {
                return Err(
                    "transition-state search with ghost atoms is not supported (ghost \
                     centers carry no nuclei, so the mass-weighted reaction coordinate is \
                     undefined)"
                        .into(),
                );
            }
            // (The double-hybrid "no PT2 gradient" rejection is handled in the
            // consolidated double-hybrid block above, before that method dispatches
            // to `run_double_hybrid`. Only the VV10 gate must be inline: the
            // consolidated VV10 block gates `optimize_geometry`, not TS.)
            if let Method::Dft(spec) = &self.method
                && spec.vv10().is_some()
            {
                return Err(format!(
                    "{} is VV10-carrying: transition-state search is not supported (the \
                     nonlocal VV10 energy E_nl has no gradient)",
                    spec.name()
                ));
            }

            // Build the surface exactly as the relaxation branch does, then drive
            // the saddle search. The algorithm and the per-job knobs (trust radii,
            // IRC confirmation, ...) come from the job's `ts_options`, which the
            // CLI populates from `--ts-*` flags; an empty job keeps the defaults.
            let ts_opts = self.options.ts_options.clone();
            let mut surface = if let Method::Dft(spec) = &self.method {
                HfSurface::new_dft(mol, &self.basis, reference, spec.clone(), opts.grid_level)?
            } else {
                HfSurface::new(mol, &self.basis, reference)?
            };
            prepare_ts_surface(&mut surface);
            if let Some(disp) = opts.dispersion {
                surface.set_dispersion(disp);
            }
            if let Some(gcp) = opts.gcp {
                surface.set_gcp(gcp);
            }
            if let Some(srb) = opts.srb {
                surface.set_srb(srb);
            }
            // Single-guess (start from `mol`), two-endpoint (build a guess between
            // `mol` and the product, then refine), or a single-ended distinguished-
            // coordinate scan. Each guess shares the reactant's composition, so the same
            // `surface` drives it.
            let (ts, mapping_confidence) = match (&opts.ts_coord_scan, &opts.ts_guess) {
                (Some(spec), _) => {
                    // Reject a driven coordinate that names an atom outside the molecule
                    // before it reaches the Wilson-B builder, which indexes the geometry
                    // directly (a raw index would otherwise panic the process).
                    let n_atoms = mol.len();
                    let in_range = match spec.coordinate {
                        Internal::Bond(i, j) => i < n_atoms && j < n_atoms,
                        Internal::Angle(i, k, j) => i.max(k).max(j) < n_atoms,
                        Internal::Dihedral(i, j, k, l) => i.max(j).max(k).max(l) < n_atoms,
                        Internal::LinearBend(i, k, j, _) => i.max(k).max(j) < n_atoms,
                    };
                    if !in_range {
                        return Err(format!(
                            "--ts-scan-coord references an atom index outside the \
                             {n_atoms}-atom molecule"
                        ));
                    }
                    // Drive one internal coordinate across its range, relaxing the rest at
                    // each grid value, and refine the saddle from the energy peak. The
                    // peak's reaction-coordinate tangent seeds the climb's reaction mode.
                    let peak = coord_scan_peak(
                        mol,
                        &CoordScanOptions::new(
                            spec.coordinate,
                            spec.start,
                            spec.end,
                            spec.n_points,
                        ),
                        &mut surface,
                    )
                    .map_err(|e| e.to_string())?;
                    let peak_mol = {
                        let atoms = mol
                            .atoms
                            .iter()
                            .zip(&peak.geometry)
                            .map(|(a, p)| crate::core::Atom::new(a.element, *p))
                            .collect();
                        Molecule::new(atoms, mol.charge, mol.multiplicity)
                    };
                    let mut seeded = ts_opts.clone();
                    if seeded.reaction_mode_seed.is_none() {
                        seeded.reaction_mode_seed = Some(peak.tangent);
                    }
                    (
                        find_transition_state(&peak_mol, &mut surface, &seeded, None)
                            .map_err(|e| ts_error_message(&e))?,
                        None,
                    )
                }
                (None, None) => (
                    find_transition_state(mol, &mut surface, &ts_opts, None)
                        .map_err(|e| ts_error_message(&e))?,
                    None,
                ),
                (None, Some(g)) if g.use_neb => {
                    let neb_ts = find_transition_state_from_endpoints(
                        mol,
                        &g.product,
                        &mut surface,
                        &g.neb_options,
                        &ts_opts,
                        None,
                    )
                    .map_err(|e| neb_ts_error_message(&e))?;
                    let confidence = neb_ts.neb.mapping_confidence.clone();
                    (neb_ts.transition_state, confidence)
                }
                (None, Some(g)) => {
                    // Build a guess between the endpoints and seed the saddle search with
                    // the reaction coordinate (unless the caller already supplied a seed in
                    // `ts_options`). The reactant is one fragment (already a combined
                    // geometry); the IDPP builder maps it onto the product. `scan_points`
                    // selects between a single geometric image and the energy-peaked scan
                    // (which evaluates the surface, so it shares `surface`).
                    let guess = match g.scan_points {
                        Some(n_points) => build_ts_guess_scanned(
                            std::slice::from_ref(mol),
                            &g.product,
                            &mut surface,
                            &ScanOptions {
                                guess: g.guess_options.clone(),
                                n_points,
                            },
                        )
                        .map_err(|e| e.to_string())?,
                        None => {
                            build_ts_guess(std::slice::from_ref(mol), &g.product, &g.guess_options)
                                .map_err(|e| e.to_string())?
                        }
                    };
                    let mut seeded = ts_opts.clone();
                    if seeded.reaction_mode_seed.is_none() {
                        seeded.reaction_mode_seed = guess.reaction_mode_seed();
                    }
                    let confidence = Some(guess.mapping_confidence.clone());
                    let ts = find_transition_state(&guess.molecule, &mut surface, &seeded, None)
                        .map_err(|e| ts_error_message(&e))?;
                    (ts, confidence)
                }
            };
            let scf = surface
                .last_scf()
                .cloned()
                .ok_or("surface has no cached SCF after transition-state search")?;
            let final_mol = (opts.dispersion.is_some() || opts.gcp.is_some() || opts.srb.is_some())
                .then(|| {
                    let atoms = mol
                        .atoms
                        .iter()
                        .zip(&ts.positions)
                        .map(|(a, p)| crate::core::Atom::new(a.element, *p))
                        .collect();
                    Molecule::new(atoms, mol.charge, mol.multiplicity)
                });
            let dispersion_energy = opts
                .dispersion
                .map(|disp| disp.energy(final_mol.as_ref().unwrap()));
            let gcp_energy = opts
                .gcp
                .map(|p| crate::disp::gcp_energy(final_mol.as_ref().unwrap(), &p));
            let srb_energy = opts
                .srb
                .map(|p| crate::disp::srb_energy(final_mol.as_ref().unwrap(), &p));
            return Ok(JobResult {
                method_warnings: Vec::new(),
                backend_downgrade: None,
                scf,
                optimized_geometry: None,
                transition_state: Some(ts),
                mapping_confidence,
                post_hf: None,
                properties: None,
                frequencies: None,
                dft: None,
                ri: None,
                cosx: None,
                dispersion_energy,
                gcp_energy,
                srb_energy,
                fod: None,
                vv10_energy: None,
                double_hybrid: None,
                smd: None,
                gbsa: None,
            });
        }

        let dispersion_energy = opts.dispersion.map(|disp| disp.energy(mol));
        let gcp_energy = opts.gcp.map(|p| crate::disp::gcp_energy(mol, &p));
        let srb_energy = opts.srb.map(|p| crate::disp::srb_energy(mol, &p));

        if opts.ri {
            let (scf, ri_diag, dft_diag, cosx_diag, vv10_energy) =
                self.run_ri(mol, n_alpha, n_beta, reference, smearing)?;
            let post_hf = if opts.ri_mp2 && scf.converged {
                Some(self.run_ri_mp2_step(mol, &scf)?)
            } else {
                None
            };
            return Ok(JobResult {
                method_warnings: Vec::new(),
                backend_downgrade: None,
                scf,
                optimized_geometry: None,
                transition_state: None,
                mapping_confidence: None,
                post_hf,
                properties: None,
                frequencies: None,
                dft: dft_diag,
                ri: Some(ri_diag),
                cosx: cosx_diag,
                dispersion_energy,
                gcp_energy,
                srb_energy,
                fod: None,
                vv10_energy,
                double_hybrid: None,
                smd: None,
                gbsa: None,
            });
        }

        if opts.direct {
            let scf = self.run_direct(mol, n_alpha, n_beta, reference, smearing)?;
            return Ok(JobResult {
                method_warnings: Vec::new(),
                backend_downgrade: None,
                scf,
                optimized_geometry: None,
                transition_state: None,
                mapping_confidence: None,
                post_hf: None,
                properties: None,
                frequencies: None,
                dft: None,
                ri: None,
                cosx: None,
                dispersion_energy,
                gcp_energy,
                srb_energy,
                fod: None,
                vv10_energy: None,
                double_hybrid: None,
                smd: None,
                gbsa: None,
            });
        }

        let (scf, provider, dft_diag, vv10_energy, cosx_diag) =
            self.run_conventional(mol, n_alpha, n_beta, reference, smearing)?;

        if !scf.converged {
            return Ok(JobResult {
                method_warnings: Vec::new(),
                backend_downgrade: None,
                scf,
                optimized_geometry: None,
                transition_state: None,
                mapping_confidence: None,
                post_hf: None,
                properties: None,
                frequencies: None,
                dft: dft_diag,
                ri: None,
                cosx: cosx_diag,
                dispersion_energy,
                gcp_energy,
                srb_energy,
                fod: None,
                vv10_energy: None,
                double_hybrid: None,
                smd: None,
                gbsa: None,
            });
        }

        let fod = if opts.fod {
            let temperature_k = fod_temperature.expect("fod implies a FOD temperature");
            let fod_result =
                crate::dft::fod_analysis(&scf, temperature_k).map_err(|e| e.to_string())?;
            if let Some(path) = &opts.fod_cube {
                let ao = BasisSet::load(&self.basis)
                    .map_err(|e| e.to_string())?
                    .build(mol)
                    .map_err(|e| e.to_string())?;
                let (da, db) = crate::dft::fod_density_matrices(&scf).map_err(|e| e.to_string())?;
                let d_tot: Vec<f64> = da.iter().zip(&db).map(|(a, b)| a + b).collect();
                crate::dft::write_fod_cube(
                    path,
                    mol,
                    ao.shells(),
                    ao.n_ao(),
                    &d_tot,
                    &CubeParams::default(),
                )
                .map_err(|e| format!("writing FOD cube {}: {e}", path.display()))?;
            }
            Some(fod_result)
        } else {
            None
        };

        let n_frozen = if opts.all_electron {
            0
        } else {
            frozen_core_orbitals(mol)
        };
        let post_hf = match &self.method {
            Method::Mp2 if opts.ri_mp2 => Some(self.run_ri_mp2_step(mol, &scf)?),
            Method::Mp2 => Some(PostHfResult::Mp2 {
                result: match scf.reference {
                    Reference::Uhf => uhf_mp2(&provider, &scf, n_frozen),
                    _ => rhf_mp2(&provider, &scf, n_frozen),
                },
                n_frozen,
            }),
            Method::Ccsd => Some(PostHfResult::Ccsd {
                result: rccsd_spin_adapted(&provider, &scf, n_frozen, &CcsdOptions::default()),
                n_frozen,
            }),
            Method::CcsdT => Some(PostHfResult::CcsdT {
                result: rccsd_t_spin_adapted(&provider, &scf, n_frozen, &CcsdOptions::default()),
                n_frozen,
            }),
            _ => None,
        };

        let properties = if opts.compute_properties {
            let com = center_of_mass(mol);
            let dipole_au = dipole_moment(&provider, mol, &scf.density, com);
            let population =
                population_analysis(&provider, mol, &scf.density_alpha, &scf.density_beta);
            Some(PropertiesResult {
                dipole_au,
                population,
            })
        } else {
            None
        };

        let frequencies = if opts.compute_frequencies {
            let fd_step = OptOptions::default().fd_step;
            let method = &self.method;
            let basis = &self.basis;
            let make_surface = |m: &Molecule| -> Result<HfSurface, String> {
                let mut surface = if let Method::Dft(spec) = method {
                    HfSurface::new_dft(m, basis, reference, spec.clone(), opts.grid_level)
                } else {
                    HfSurface::new(m, basis, reference)
                }
                .map_err(|e| e.to_string())?;
                if let Some(disp) = opts.dispersion {
                    surface.set_dispersion(disp);
                }
                if let Some(gcp) = opts.gcp {
                    surface.set_gcp(gcp);
                }
                if let Some(srb) = opts.srb {
                    surface.set_srb(srb);
                }
                Ok(surface)
            };
            let grad_err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
            let hess = numerical_hessian(mol, 0.005, |m: &Molecule| {
                let pos: Vec<[f64; 3]> = m.atoms.iter().map(|a| a.position).collect();
                let mut surface = match make_surface(m) {
                    Ok(s) => s,
                    Err(e) => {
                        grad_err.lock().unwrap().get_or_insert(e);
                        return vec![0.0; 3 * m.len()];
                    }
                };
                let grad = match surface.analytic_gradient(&pos) {
                    Some(r) => r,
                    None => crate::opt::fd::central_difference(&mut surface, &pos, fd_step),
                };
                match grad {
                    Ok(g) => g.iter().flat_map(|x| x.iter().copied()).collect(),
                    Err(e) => {
                        grad_err
                            .lock()
                            .unwrap()
                            .get_or_insert_with(|| e.to_string());
                        vec![0.0; 3 * m.len()]
                    }
                }
            });
            if let Some(e) = grad_err.into_inner().unwrap() {
                return Err(format!("frequency gradient evaluation failed: {e}"));
            }
            let freq = if opts.single_point_hessian {
                let central: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
                let mut surface = make_surface(mol)?;
                let grad = surface
                    .analytic_gradient(&central)
                    .unwrap_or_else(|| {
                        crate::opt::fd::central_difference(&mut surface, &central, fd_step)
                    })
                    .map_err(|e| format!("SPH central-geometry gradient failed: {e}"))?;
                let g_flat: Vec<f64> = grad.iter().flat_map(|x| x.iter().copied()).collect();
                crate::props::sph::sph_frequencies(mol, &hess, &g_flat)
            } else {
                harmonic_frequencies(mol, &hess)
            };
            let thermo = rrho_thermochemistry_w0(
                mol,
                &freq,
                scf.energy
                    + dispersion_energy.unwrap_or(0.0)
                    + gcp_energy.unwrap_or(0.0)
                    + srb_energy.unwrap_or(0.0),
                298.15,
                opts.symmetry_number,
                mol.multiplicity,
                opts.qrrho_w0_cm1,
            );
            Some(FrequencyData {
                frequencies: freq,
                thermochemistry: thermo,
                is_sph: opts.single_point_hessian,
            })
        } else {
            None
        };

        Ok(JobResult {
            method_warnings: Vec::new(),
            backend_downgrade: None,
            scf,
            optimized_geometry: None,
            transition_state: None,
            mapping_confidence: None,
            post_hf,
            properties,
            frequencies,
            dft: dft_diag,
            ri: None,
            cosx: cosx_diag,
            dispersion_energy,
            gcp_energy,
            srb_energy,
            fod,
            vv10_energy,
            double_hybrid: None,
            smd: None,
            gbsa: None,
        })
    }

    fn build_cpcm<'a, P: IntegralProvider>(
        &self,
        provider: &'a P,
        mol: &Molecule,
    ) -> Result<Option<Cpcm<'a, P>>, String> {
        if let Some(name) = &self.options.smd {
            let s = resolve_smd_solvent(name)?;
            let zs: Vec<usize> = mol.atoms.iter().map(|a| a.element.z() as usize).collect();
            let radii = crate::solv::smd_coulomb_radii(&zs, s.alpha).map_err(|e| e.to_string())?;
            return Cpcm::with_radii(provider, mol, s.epsilon, crate::solv::DEFAULT_GRID, &radii)
                .map(Some)
                .map_err(|e| e.to_string());
        }
        self.options
            .solvent_eps
            .map(|eps| Cpcm::new(provider, mol, eps, crate::solv::DEFAULT_GRID))
            .transpose()
            .map_err(|e| e.to_string())
    }

    fn run_ri_mp2_step(&self, mol: &Molecule, scf: &ScfResult) -> Result<PostHfResult, String> {
        let n_frozen = if self.options.all_electron {
            0
        } else {
            frozen_core_orbitals(mol)
        };
        let (aux_name, result) = self.run_ri_mp2_correlation(mol, scf, n_frozen)?;
        Ok(PostHfResult::RiMp2 {
            result,
            n_frozen,
            aux_basis: aux_name,
        })
    }

    fn run_ri_mp2_correlation(
        &self,
        mol: &Molecule,
        scf: &ScfResult,
        n_frozen: usize,
    ) -> Result<(String, RiMp2Result), String> {
        let aux_name = format!("{}/c", self.basis.to_ascii_lowercase());
        let aux_set = BasisSet::load_aux(&aux_name).map_err(|e| match e {
            BasisError::UnknownAuxSet(_) => format!(
                "RI-MP2 needs the MP2-fit auxiliary basis {aux_name:?}, but no /C partner is \
                 bundled for orbital basis {:?} (available: def2-svp/c, def2-tzvp/c); there is \
                 no silent fallback to the JK fitting set",
                self.basis
            ),
            other => other.to_string(),
        })?;
        let aux = aux_set
            .build(mol)
            .map_err(|e| e.to_string())?
            .into_integral();
        let ao = BasisSet::load(&self.basis)
            .map_err(|e| e.to_string())?
            .build(mol)
            .map_err(|e| e.to_string())?
            .into_integral();
        let result = match scf.reference {
            Reference::Uhf => uhf_ri_mp2(&ao, &aux, scf, n_frozen),
            _ => rhf_ri_mp2(&ao, &aux, scf, n_frozen),
        }
        .map_err(|e| e.to_string())?;
        Ok((aux_name, result))
    }

    fn run_double_hybrid(
        &self,
        mol: &Molecule,
        n_alpha: usize,
        n_beta: usize,
    ) -> Result<JobResult, String> {
        let opts = &self.options;
        let Method::Dft(spec) = &self.method else {
            return Err("run_double_hybrid requires a DFT method".into());
        };
        let dh = spec
            .double_hybrid()
            .ok_or("run_double_hybrid requires double-hybrid metadata")?;

        let xdh = spec.name() == "wb97m(2)";
        let scf_spec = if xdh {
            FunctionalSpec::parse("wb97m-v").map_err(|e| e.to_string())?
        } else {
            spec.clone()
        };

        let ao = BasisSet::load(&self.basis)
            .map_err(|e| e.to_string())?
            .build(mol)
            .map_err(|e| e.to_string())?;
        let setup = ecp_setup(mol, &ao);
        let grid_xc =
            GridXc::new(mol, &ao, &scf_spec, opts.grid_level).map_err(|e| e.to_string())?;
        let grid_xc_dh = if xdh {
            Some(GridXc::new(mol, &ao, spec, opts.grid_level).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let dft_diag = Some(DftDiagnostics {
            functional_name: spec.name().to_string(),
            grid_level: grid_xc.level(),
            n_grid_points: grid_xc.n_points(),
            exx_fraction: spec.exx_fraction(),
        });
        let provider =
            ConventionalProvider::new(ao.into_integral(), setup.charges).with_ecps(setup.ecps);
        let scf_opts = ScfOptions {
            energy_tol: 1e-9,
            error_tol: 1e-6,
            ..ScfOptions::default()
        };
        let scf = run_scf_with_env(
            &provider,
            n_alpha,
            n_beta,
            Reference::Rhf,
            setup.nuclear_repulsion,
            &scf_opts,
            Some(&grid_xc as &dyn XcContributor),
            None,
        )
        .map_err(|e| e.to_string())?;

        let dispersion_energy = opts.dispersion.map(|disp| disp.energy(mol));

        if !scf.converged {
            return Ok(JobResult {
                method_warnings: Vec::new(),
                backend_downgrade: None,
                scf,
                optimized_geometry: None,
                transition_state: None,
                mapping_confidence: None,
                post_hf: None,
                properties: None,
                frequencies: None,
                dft: dft_diag,
                ri: None,
                cosx: None,
                dispersion_energy,
                gcp_energy: None,
                srb_energy: None,
                fod: None,
                vv10_energy: None,
                double_hybrid: None,
                smd: None,
                gbsa: None,
            });
        }

        let n = scf.n_basis;
        let (e_scf, vv10_energy, vv10_scale) = if xdh {
            let gx_dh = grid_xc_dh.as_ref().expect("xdh builds the target GridXc");
            let (exc_mv, _) = grid_xc.energy(&scf.density_alpha, &scf.density_beta, true);
            let (exc_m2, _) = gx_dh.energy(&scf.density_alpha, &scf.density_beta, true);
            let cam_mv = scf_spec.cam().ok_or("ωB97M-V must carry CAM parameters")?;
            let cam_m2 = spec.cam().ok_or("ωB97M(2) must carry CAM parameters")?;
            let da = mat_from_row_major(n, &scf.density_alpha);
            let jk = provider.build_jk(std::slice::from_ref(&da));
            let k = mat_to_row_major(&jk.exchange[0]);
            let exx_for = |omega: f64, alpha: f64, beta: f64| -> Result<f64, String> {
                let klr = provider
                    .build_k_erf(std::slice::from_ref(&da), omega)
                    .ok_or("double hybrids need the in-core erf-attenuated exchange")?;
                let klr = mat_to_row_major(&klr[0]);
                let mut e = 0.0;
                for i in 0..n * n {
                    e += scf.density_alpha[i] * (alpha * k[i] + beta * klr[i]);
                }
                Ok(-e)
            };
            let exx_mv = exx_for(cam_mv.omega, cam_mv.alpha, cam_mv.beta)?;
            let exx_m2 = exx_for(cam_m2.omega, cam_m2.alpha, cam_m2.beta)?;
            let e_scf = scf.energy - exc_mv - exx_mv + exc_m2 + exx_m2;
            let scale = 1.0 - dh.c_os;
            let e_nl = gx_dh
                .vv10_energy(mol, &scf.density)
                .transpose()
                .map_err(|e| e.to_string())?;
            (e_scf, e_nl.map(|e| scale * e), scale)
        } else {
            (scf.energy, None, 1.0)
        };

        let n_frozen = if opts.all_electron {
            0
        } else {
            frozen_core_orbitals(mol)
        };
        let (e_os, e_ss, pt2_aux_basis) = if opts.ri_mp2 {
            let (aux_name, ri) = self.run_ri_mp2_correlation(mol, &scf, n_frozen)?;
            (ri.opposite_spin, ri.same_spin, Some(aux_name))
        } else {
            let mp2 = rhf_mp2(&provider, &scf, n_frozen);
            (mp2.opposite_spin, mp2.same_spin, None)
        };
        let double_hybrid = Some(DoubleHybridData {
            functional_name: spec.name().to_string(),
            scf_functional_name: scf_spec.name().to_string(),
            e_scf,
            e_os,
            e_ss,
            c_os: dh.c_os,
            c_ss: dh.c_ss,
            n_frozen,
            vv10_scale,
            pt2_aux_basis,
        });

        let properties = if opts.compute_properties {
            let com = center_of_mass(mol);
            let dipole_au = dipole_moment(&provider, mol, &scf.density, com);
            let population =
                population_analysis(&provider, mol, &scf.density_alpha, &scf.density_beta);
            Some(PropertiesResult {
                dipole_au,
                population,
            })
        } else {
            None
        };

        Ok(JobResult {
            method_warnings: Vec::new(),
            backend_downgrade: None,
            scf,
            optimized_geometry: None,
            transition_state: None,
            mapping_confidence: None,
            post_hf: None,
            properties,
            frequencies: None,
            dft: dft_diag,
            ri: None,
            cosx: None,
            dispersion_energy,
            gcp_energy: None,
            srb_energy: None,
            fod: None,
            vv10_energy,
            double_hybrid,
            smd: None,
            gbsa: None,
        })
    }

    fn run_ri(
        &self,
        mol: &Molecule,
        n_alpha: usize,
        n_beta: usize,
        reference: Reference,
        smearing: Option<Smearing>,
    ) -> Result<RiRun, String> {
        const AUX_BASIS: &str = "def2-universal-jkfit";
        let opts = &self.options;
        let ao = BasisSet::load(&self.basis)
            .map_err(|e| e.to_string())?
            .build(mol)
            .map_err(|e| e.to_string())?;
        let aux = BasisSet::load_aux(AUX_BASIS)
            .map_err(|e| e.to_string())?
            .build(mol)
            .map_err(|e| e.to_string())?
            .into_integral();
        let setup = ecp_setup(mol, &ao);
        let grid_xc = if let Method::Dft(spec) = &self.method {
            Some(GridXc::new(mol, &ao, spec, opts.grid_level).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let dft_diag = grid_xc.as_ref().map(|g| DftDiagnostics {
            functional_name: g.name().to_string(),
            grid_level: g.level(),
            n_grid_points: g.n_points(),
            exx_fraction: g.exx_fraction(),
        });
        let base_opts = if grid_xc.is_some() {
            ScfOptions {
                energy_tol: 1e-9,
                error_tol: 1e-6,
                ..ScfOptions::default()
            }
        } else {
            ScfOptions::default()
        };
        let base_opts = ScfOptions {
            smearing,
            ..base_opts
        };
        let hcore_override = opts
            .x2c
            .then(|| x2c_hcore_override(&ao, &setup.charges, base_opts.lindep_thresh))
            .transpose()?;
        let base_opts = ScfOptions {
            hcore_override,
            ..base_opts
        };
        let xc_ref = grid_xc.as_ref().map(|g| g as &dyn XcContributor);
        let cosx_setup = opts.cosx.then(|| (ao.shells().to_vec(), ao.n_ao()));
        let provider = DfProvider::new(ao.into_integral(), &aux, setup.charges)
            .map_err(|e| e.to_string())?
            .with_ecps(setup.ecps);
        let ri_diag = RiDiagnostics {
            aux_basis: AUX_BASIS.to_string(),
            naux: provider.naux(),
        };
        let cpcm = self.build_cpcm(&provider, mol)?;
        let solv_ref = cpcm.as_ref().map(|c| c as &dyn SolventModel);
        let (scf, cosx_diag) = if let Some((shells, nao)) = cosx_setup {
            let cam = match &self.method {
                Method::Dft(spec) => spec.cam(),
                _ => None,
            };
            let s = mat_to_row_major(&provider.overlap());
            let cosx = CosxExchange::new(mol, &shells, nao, &s, COSX_DEFAULT_GRID)
                .map_err(|e| e.to_string())?;
            let diag = CosxDiagnostics {
                grid: cosx.description().to_string(),
                n_points: cosx.n_points(),
                overlap_fitted: cosx.fitted(),
                rs_omega: cam.map(|c| c.omega),
            };
            let wrapped = match cam {
                Some(c) => CosxProvider::with_range_separation(&provider, cosx, c.omega),
                None => CosxProvider::new(&provider, cosx),
            }
            .map_err(|e| e.to_string())?;
            let scf = run_scf_with_env(
                &wrapped,
                n_alpha,
                n_beta,
                reference,
                setup.nuclear_repulsion,
                &base_opts,
                xc_ref,
                solv_ref,
            )
            .map_err(|e| e.to_string())?;
            (scf, Some(diag))
        } else {
            let scf = run_scf_with_env(
                &provider,
                n_alpha,
                n_beta,
                reference,
                setup.nuclear_repulsion,
                &base_opts,
                xc_ref,
                solv_ref,
            )
            .map_err(|e| e.to_string())?;
            (scf, None)
        };
        let vv10_energy = match (&grid_xc, scf.converged) {
            (Some(g), true) => g
                .vv10_energy(mol, &scf.density)
                .transpose()
                .map_err(|e| e.to_string())?,
            _ => None,
        };
        Ok((scf, ri_diag, dft_diag, cosx_diag, vv10_energy))
    }

    fn run_direct(
        &self,
        mol: &Molecule,
        n_alpha: usize,
        n_beta: usize,
        reference: Reference,
        smearing: Option<Smearing>,
    ) -> Result<ScfResult, String> {
        let opts = &self.options;
        let ao = BasisSet::load(&self.basis)
            .map_err(|e| e.to_string())?
            .build(mol)
            .map_err(|e| e.to_string())?;
        let setup = ecp_setup(mol, &ao);
        let grid_xc = if let Method::Dft(spec) = &self.method {
            Some(GridXc::new(mol, &ao, spec, opts.grid_level).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let base_opts = if grid_xc.is_some() {
            ScfOptions {
                energy_tol: 1e-9,
                error_tol: 1e-6,
                ..ScfOptions::default()
            }
        } else {
            ScfOptions::default()
        };
        let base_opts = ScfOptions {
            smearing,
            ..base_opts
        };
        let hcore_override = opts
            .x2c
            .then(|| x2c_hcore_override(&ao, &setup.charges, base_opts.lindep_thresh))
            .transpose()?;
        let base_opts = ScfOptions {
            hcore_override,
            ..base_opts
        };
        let xc_ref = grid_xc.as_ref().map(|g| g as &dyn XcContributor);
        let provider = DirectProvider::new(ao.into_integral(), setup.charges).with_ecps(setup.ecps);
        let cpcm = self.build_cpcm(&provider, mol)?;
        let solv_ref = cpcm.as_ref().map(|c| c as &dyn SolventModel);
        run_scf_with_env(
            &provider,
            n_alpha,
            n_beta,
            reference,
            setup.nuclear_repulsion,
            &ScfOptions {
                incremental_fock: true,
                ..base_opts
            },
            xc_ref,
            solv_ref,
        )
        .map_err(|e| e.to_string())
    }

    fn run_conventional(
        &self,
        mol: &Molecule,
        n_alpha: usize,
        n_beta: usize,
        reference: Reference,
        smearing: Option<Smearing>,
    ) -> Result<ConventionalRun, String> {
        let opts = &self.options;
        let ao = BasisSet::load(&self.basis)
            .map_err(|e| e.to_string())?
            .build(mol)
            .map_err(|e| e.to_string())?;
        let setup = ecp_setup(mol, &ao);
        let grid_xc = if let Method::Dft(spec) = &self.method {
            Some(GridXc::new(mol, &ao, spec, opts.grid_level).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let dft_diag = grid_xc.as_ref().map(|g| DftDiagnostics {
            functional_name: g.name().to_string(),
            grid_level: g.level(),
            n_grid_points: g.n_points(),
            exx_fraction: g.exx_fraction(),
        });
        let base_opts = if grid_xc.is_some() {
            ScfOptions {
                energy_tol: 1e-9,
                error_tol: 1e-6,
                ..ScfOptions::default()
            }
        } else {
            ScfOptions::default()
        };
        let base_opts = ScfOptions {
            smearing,
            ..base_opts
        };
        let hcore_override = opts
            .x2c
            .then(|| x2c_hcore_override(&ao, &setup.charges, base_opts.lindep_thresh))
            .transpose()?;
        let base_opts = ScfOptions {
            hcore_override,
            ..base_opts
        };
        let xc_ref = grid_xc.as_ref().map(|g| g as &dyn XcContributor);
        let cosx_setup = opts.cosx.then(|| (ao.shells().to_vec(), ao.n_ao()));
        let provider =
            ConventionalProvider::new(ao.into_integral(), setup.charges).with_ecps(setup.ecps);
        let cpcm = self.build_cpcm(&provider, mol)?;
        let solv_ref = cpcm.as_ref().map(|c| c as &dyn SolventModel);
        let (scf, cosx_diag) = if let Some((shells, nao)) = cosx_setup {
            let cam = match &self.method {
                Method::Dft(spec) => spec.cam(),
                _ => None,
            };
            let s = mat_to_row_major(&provider.overlap());
            let cosx = CosxExchange::new(mol, &shells, nao, &s, COSX_DEFAULT_GRID)
                .map_err(|e| e.to_string())?;
            let diag = CosxDiagnostics {
                grid: cosx.description().to_string(),
                n_points: cosx.n_points(),
                overlap_fitted: cosx.fitted(),
                rs_omega: cam.map(|c| c.omega),
            };
            let wrapped = match cam {
                Some(c) => CosxProvider::with_range_separation(&provider, cosx, c.omega),
                None => CosxProvider::new(&provider, cosx),
            }
            .map_err(|e| e.to_string())?;
            let scf = run_scf_with_env(
                &wrapped,
                n_alpha,
                n_beta,
                reference,
                setup.nuclear_repulsion,
                &base_opts,
                xc_ref,
                solv_ref,
            )
            .map_err(|e| e.to_string())?;
            (scf, Some(diag))
        } else {
            let scf = run_scf_with_env(
                &provider,
                n_alpha,
                n_beta,
                reference,
                setup.nuclear_repulsion,
                &base_opts,
                xc_ref,
                solv_ref,
            )
            .map_err(|e| e.to_string())?;
            (scf, None)
        };
        let vv10_energy = match (&grid_xc, scf.converged) {
            (Some(g), true) => g
                .vv10_energy(mol, &scf.density)
                .transpose()
                .map_err(|e| e.to_string())?,
            _ => None,
        };
        Ok((scf, provider, dft_diag, vv10_energy, cosx_diag))
    }
}

type RiRun = (
    ScfResult,
    RiDiagnostics,
    Option<DftDiagnostics>,
    Option<CosxDiagnostics>,
    Option<f64>,
);

type ConventionalRun = (
    ScfResult,
    ConventionalProvider,
    Option<DftDiagnostics>,
    Option<f64>,
    Option<CosxDiagnostics>,
);

/// Recovery hint shown when an SCF fails to converge during a geometry/TS run; the
/// `Job` path flattens typed errors to a single user-facing `String`.
const SCF_RECOVERY_HINT: &str = "SCF did not converge — try a better initial geometry, a larger SCF level shift, \
     or more SCF iterations";

/// Flatten an [`OptError`] for the CLI, replacing the bare SCF-non-convergence
/// message with an actionable recovery hint.
fn opt_error_message(e: &OptError) -> String {
    match e {
        OptError::ScfNotConverged { .. } => SCF_RECOVERY_HINT.to_string(),
        // `OptError` is `#[non_exhaustive]`; everything else keeps its own message.
        other => other.to_string(),
    }
}

/// Flatten a [`TsError`] for the CLI; an SCF non-convergence reaching the surface
/// (the common TS failure) gets the same recovery hint as the minimizer path.
fn ts_error_message(e: &TsError) -> String {
    match e {
        TsError::SurfaceEvaluation(OptError::ScfNotConverged { .. }) => {
            SCF_RECOVERY_HINT.to_string()
        }
        // `TsError` is `#[non_exhaustive]`; everything else keeps its own message.
        other => other.to_string(),
    }
}

/// Flatten a [`NebTsError`] (the two-endpoint NEB-TS pipeline) for the CLI. An SCF
/// non-convergence at a band image or during the refinement gets the same recovery
/// hint as the single-geometry path; everything else keeps its own message.
fn neb_ts_error_message(e: &NebTsError) -> String {
    use crate::opt::ts::NebError;
    match e {
        NebTsError::Ts(TsError::SurfaceEvaluation(OptError::ScfNotConverged { .. }))
        | NebTsError::Neb(NebError::SurfaceEvaluation(OptError::ScfNotConverged { .. })) => {
            SCF_RECOVERY_HINT.to_string()
        }
        other => other.to_string(),
    }
}

fn resolve_smd_solvent(name: &str) -> Result<&'static crate::solv::SmdSolvent, String> {
    crate::solv::smd_solvent(name).ok_or_else(|| {
        let names: Vec<&str> = crate::solv::SMD_SOLVENTS.iter().map(|s| s.name).collect();
        format!(
            "unknown SMD solvent {name:?} (available: {})",
            names.join(", ")
        )
    })
}

fn resolve_alpb_solvent(name: &str) -> Result<&'static crate::solv::GbsaParams, String> {
    crate::solv::alpb_solvent(name).ok_or_else(|| {
        format!(
            "unknown ALPB solvent {name:?} (available: {})",
            crate::solv::alpb_solvent_names().join(", ")
        )
    })
}

fn resolve_gbsa_solvent(name: &str) -> Result<&'static crate::solv::GbsaParams, String> {
    crate::solv::gbsa_solvent(name).ok_or_else(|| {
        format!(
            "unknown GBSA solvent {name:?} (available: {})",
            crate::solv::gbsa_solvent_names().join(", ")
        )
    })
}

/// Resolve the α/β electron counts from the molecule and ECP core size,
/// applying the same charge/multiplicity validation `run_inner` requires.
/// Shared with [`crate::estimate_memory`] so both derive occupancies the same
/// way.
pub(crate) fn alpha_beta_electrons(
    mol: &Molecule,
    ecp_core: i64,
) -> Result<(usize, usize), String> {
    let n_elec = mol.n_electrons() - ecp_core;
    if n_elec < 0 {
        return Err("charge exceeds nuclear charge (negative electron count)".into());
    }
    let n_elec = n_elec as usize;
    let two_s = (mol.multiplicity.saturating_sub(1)) as usize;
    if two_s > n_elec {
        return Err("multiplicity is too high for the electron count".into());
    }
    Ok(((n_elec + two_s) / 2, (n_elec - two_s) / 2))
}

fn method_reference(method: &Method, multiplicity: u32) -> Reference {
    match method {
        Method::Rhf | Method::Ccsd | Method::CcsdT => Reference::Rhf,
        Method::Uhf => Reference::Uhf,
        Method::Rohf => Reference::Rohf,
        Method::Mp2 | Method::Dft(_) => {
            if multiplicity > 1 {
                Reference::Uhf
            } else {
                Reference::Rhf
            }
        }
    }
}

pub fn optimize_geometry(
    molecule: &Molecule,
    basis: &str,
    reference: Reference,
    options: &OptOptions,
) -> Result<OptResult, String> {
    let mut surface = HfSurface::new(molecule, basis, reference)?;
    optimize(molecule, &mut surface, options).map_err(|e| e.to_string())
}

pub fn optimize_geometry_dft(
    molecule: &Molecule,
    basis: &str,
    reference: Reference,
    functional: FunctionalSpec,
    grid_level: usize,
    options: &OptOptions,
) -> Result<OptResult, String> {
    let mut surface = HfSurface::new_dft(molecule, basis, reference, functional, grid_level)?;
    optimize(molecule, &mut surface, options).map_err(|e| e.to_string())
}

/// Apply the SCF settings a saddle search needs: transition-state geometries have
/// small HOMO–LUMO gaps, so the SCF gets extra iterations and a level shift to
/// converge. The error threshold is held a little looser than the surface default
/// because a small-gap commutator can floor just short of the tightest tolerance;
/// the resulting gradients are still well within what the search needs.
fn prepare_ts_surface(surface: &mut HfSurface) {
    surface.set_scf_max_iter(400);
    surface.set_scf_level_shift(0.3);
    surface.set_scf_convergence(1e-10, 5e-8);
}

/// Locate a first-order saddle point on a Hartree–Fock (or wavefunction-SCF)
/// surface, returning the TYPED [`TsResult`]/[`TsError`] contract.
///
/// The transition-state analogue of [`optimize_geometry`], and the entry point a
/// programmatic agent should prefer over the `Job` flag: it preserves the
/// [`TsError`] variant, which the `Job` path necessarily flattens to `String`
/// (because `Job::run` returns `Result<_, String>`). Builds an
/// [`HfSurface`] with the same saddle-search SCF settings the `Job` path uses;
/// `progress` is the optional per-iteration
/// observer (see [`Progress`]). Surface-construction failures (e.g. an
/// inconsistent charge/multiplicity) are reported as
/// [`TsError::SurfaceEvaluation`].
pub fn transition_state(
    molecule: &Molecule,
    basis: &str,
    reference: Reference,
    options: &TsOptions,
    progress: Option<&dyn Progress>,
) -> Result<TsResult, TsError> {
    let mut surface = HfSurface::new(molecule, basis, reference)
        .map_err(|e| TsError::SurfaceEvaluation(OptError::Evaluation(e)))?;
    prepare_ts_surface(&mut surface);
    find_transition_state(molecule, &mut surface, options, progress)
}

/// Locate a first-order saddle point on a Kohn–Sham (DFT) surface; the DFT
/// analogue of [`optimize_geometry_dft`]. See [`transition_state`].
pub fn transition_state_dft(
    molecule: &Molecule,
    basis: &str,
    reference: Reference,
    functional: FunctionalSpec,
    grid_level: usize,
    options: &TsOptions,
    progress: Option<&dyn Progress>,
) -> Result<TsResult, TsError> {
    let mut surface = HfSurface::new_dft(molecule, basis, reference, functional, grid_level)
        .map_err(|e| TsError::SurfaceEvaluation(OptError::Evaluation(e)))?;
    prepare_ts_surface(&mut surface);
    find_transition_state(molecule, &mut surface, options, progress)
}

#[cfg(test)]
#[path = "job_tests.rs"]
mod tests;
