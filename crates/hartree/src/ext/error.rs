use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ExtError {
    #[error(
        "{program} binary not found: not on PATH and {env_var} is not set. \
         Install it from {install_hint} (conda: `conda install -c conda-forge {conda_pkg}`), \
         or point {env_var} at the executable."
    )]
    BinaryNotFound {
        program: &'static str,
        env_var: &'static str,
        install_hint: &'static str,
        conda_pkg: &'static str,
    },

    #[error("{program} failed (exit status {status}): {stderr_tail}")]
    SubprocessFailed {
        program: &'static str,
        status: String,
        stderr_tail: String,
    },

    #[error("{program} did not produce the expected output file {path}")]
    MissingOutput {
        program: &'static str,
        path: PathBuf,
    },

    #[error("parsing {what}: {message}")]
    Parse { what: &'static str, message: String },

    #[error("I/O error ({context}): {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    #[error("conformer generation: {0}")]
    ConfGen(String),
}

impl ExtError {
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        ExtError::Io {
            context: context.into(),
            source,
        }
    }
}
