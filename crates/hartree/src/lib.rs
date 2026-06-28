//! hartree — a quantum chemistry and solid-state physics package implemented in Rust.
//!
//! The library is organized by layer: foundation (`core`, `linalg`, `tensor`),
//! basis/integrals, SCF and post-SCF methods, corrections/extensions, and the
//! higher-level drivers (`job`, `multilevel`, `w1`, …) that orchestrate them.

/// The crate version, taken from `CARGO_PKG_VERSION` at compile time so it always
/// matches the version declared in `Cargo.toml`.
///
/// ```
/// assert_eq!(hartree::VERSION, env!("CARGO_PKG_VERSION"));
/// assert!(!hartree::VERSION.is_empty());
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Foundation
pub mod core;
pub mod linalg;
pub mod tensor;

// Basis sets and the integral seam
pub mod basis;
pub mod integrals;

// SCF and post-SCF methods
pub mod cc;
pub mod dft;
pub mod grad;
pub mod props;
pub mod scf;

// Corrections and extensions
pub mod disp;
pub mod ext;
pub mod opt;
pub mod solv;

// Periodic (solid-state) GPW Kohn–Sham
pub mod periodic;

// High-level orchestration
pub mod composite;
pub mod cp;
mod estimate;
pub mod guardrails;
mod job;
pub mod multilevel;
mod periodic_job;
mod sad;
mod surface;
pub mod w1;

pub use cp::{CpFragments, CpResult, counterpoise};
pub use estimate::{EstimateBackend, MemoryEstimate, MemoryTerm, estimate_memory};
pub use job::{
    BackendDowngrade, CoordScanSpec, CosxDiagnostics, DftDiagnostics, FrequencyData, GbsaData, Job,
    JobOptions, JobResult, Method, PostHfResult, PropertiesResult, RiDiagnostics, SmdData,
    TsGuessInput, ecp_summary, optimize_geometry, optimize_geometry_dft, transition_state,
    transition_state_dft,
};
pub use periodic_job::{PeriodicFunctional, PeriodicJob, PeriodicJobResult, run_periodic};
pub use surface::HfSurface;

// Flagship types lifted to the crate root for convenience.
pub use basis::BasisSet;
pub use core::{Atom, Element, Molecule};
