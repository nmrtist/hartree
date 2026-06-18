use thiserror::Error;

pub type Result<T> = std::result::Result<T, HartreeError>;

#[derive(Debug, Error)]
pub enum HartreeError {
    #[error("unknown element symbol: {0:?}")]
    UnknownElement(String),

    #[error("invalid atomic number: {0} (supported range is 1..=118)")]
    InvalidAtomicNumber(u32),

    #[error("malformed XYZ input: {0}")]
    MalformedXyz(String),

    #[error(
        "inconsistent spin state: {n_electrons} electrons cannot have multiplicity {multiplicity}"
    )]
    InconsistentSpin { n_electrons: i64, multiplicity: u32 },

    #[error("{0}")]
    Other(String),
}
