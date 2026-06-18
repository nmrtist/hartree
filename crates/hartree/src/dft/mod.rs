//! Kohn-Sham DFT: molecular integration grid, AO/density evaluation, and the XC seam.

pub mod ao;
pub mod cosx;
pub mod density;
mod error;
pub mod fod;
mod functional;
mod gradient;
pub mod grid;
pub mod vv10;
pub mod xc;

pub use ao::{AoBatch, eval_ao_batch, eval_ao_batch_full, par_blocks_fold, par_blocks_fold_full};
pub use cosx::{COSX_DEFAULT_GRID, CosxExchange, CosxGrid, CosxProvider};
pub use density::BatchDensity;
pub use error::{DftError, Result};
pub use fod::{
    CubeParams, FodResult, fod_analysis, fod_default_temperature, fod_density_matrices,
    fod_grid_integral, fod_weights, write_fod_cube,
};
pub use functional::FunctionalSpec;
pub use grid::MolecularGrid;
pub use xc::GridXc;

pub use crate::basis::ShellData;
pub use crate::scf::{XcContribution, XcContributor};

pub use xcx::{CamParams, DoubleHybridParams, Functional, Spin, Vv10Params, XcInput, XcResult};
