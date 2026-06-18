//! Periodic GPW Kohn-Sham DFT: k-point SCF, Ewald, analytic forces and stress, bands, and DOS.

pub use crate::integrals::integral::periodic::{
    RealSpaceGrid, collocate_density, hartree, integrate_potential,
};
pub use crate::integrals::integral::{Basis, Shell};
pub use latx::{Cell, KPoint, MonkhorstPack};

pub use xcx::FunctionalId;

mod converged;
mod energy;
mod error;
mod ewald;
mod forces;
mod post;
mod pseudo;
mod scf;
mod stress;
mod system;
mod xc;

pub use energy::{GpwLocalEnergy, LocalKsBuild, build_local_ks, kinetic_energy, local_energy};
pub use error::PeriodicError;
pub use ewald::{ewald_energy, ewald_energy_eta};
pub use forces::{finite_difference_forces, periodic_forces};
pub use post::{BandStructure, Dos, band_structure, density_of_states};
pub use pseudo::{
    GthLocalAtom, core_charge_density, core_charge_forces, core_charge_stress,
    local_pp_short_range, local_sr_forces, local_sr_stress, overlap_energy, overlap_forces,
    overlap_stress, self_energy,
};
pub use scf::{
    EnergyComponents, NonlocalChannel, PeriodicAtom, PeriodicScfOptions, PeriodicScfResult,
    run_scf_periodic,
};
pub use stress::{apply_strain, finite_difference_strain_energy, periodic_stress};
pub use system::PeriodicSystem;
pub use xc::GridXc;
