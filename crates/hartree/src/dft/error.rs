use thiserror::Error;

pub type Result<T> = std::result::Result<T, DftError>;

#[derive(Debug, Error)]
pub enum DftError {
    #[error("unknown functional: {0:?}")]
    UnknownFunctional(String),

    #[error(
        "functional {0:?} requires kinetic-energy density (meta-GGA); \
         only tpss and r2scan are supported"
    )]
    NeedsTau(String),

    #[error("functional {0:?} requires the density Laplacian; unsupported")]
    NeedsLapl(String),

    #[error("functional {0:?} is range-separated (CAM); unsupported")]
    RangeSeparated(String),

    #[error("functional {0:?} uses VV10 nonlocal correlation; unsupported")]
    Vv10(String),

    #[error("grid level {0} out of range (expected 0..=4)")]
    InvalidGridLevel(usize),

    #[error("element Z={0} is not supported by the DFT grid (Z = 1–86, H–Rn)")]
    UnsupportedElement(u32),

    #[error("angular momentum l={0} exceeds the grid evaluator maximum")]
    UnsupportedAngularMomentum(u32),

    #[error("shell center matches no atom (bitwise-equal-centers rule violated)")]
    ShellAtomMismatch,

    #[error(
        "COSX semi-numerical exchange requires an integral backend that supplies \
         grid-point Coulomb matrices (grid_coulomb); this backend declines"
    )]
    CosxUnsupportedBackend,

    #[error(
        "FOD analysis needs a fractional-occupation (Fermi-smeared) SCF result; \
         run the SCF with smearing enabled"
    )]
    NoFractionalOccupations,

    #[error("xcx error: {0}")]
    Xc(#[from] xcx::XcError),
}
