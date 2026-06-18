use thiserror::Error;

pub type Result<T> = std::result::Result<T, BasisError>;

#[derive(Debug, Error)]
pub enum BasisError {
    #[error("unknown basis set: {0:?}")]
    UnknownSet(String),

    #[error(
        "{0:?} is an auxiliary fitting set, not an orbital basis; it cannot be used as --basis"
    )]
    AuxiliaryAsOrbital(String),

    #[error("unknown auxiliary basis set: {0:?}")]
    UnknownAuxSet(String),

    #[error("element Z={z} is not defined in basis set {set:?}")]
    ElementNotInSet { z: u32, set: String },

    #[error("malformed basis JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("basis schema error: {0}")]
    Schema(String),

    #[error("integral shell error: {0}")]
    Integral(#[from] integral::IntegralError),
}
