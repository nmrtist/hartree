use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PeriodicError {
    #[error("exchange–correlation evaluation failed: {0}")]
    Xc(#[from] xcx::XcError),

    #[error("dimension mismatch: {0}")]
    Dimension(String),

    #[error("invalid periodic configuration: {0}")]
    Config(String),
}
