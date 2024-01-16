pub mod atomic_pattern;
pub mod config;
pub mod dag;
pub mod decomposition;
pub mod wop_atomic_pattern;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Err {
    NotComposable(String),
    NoParametersFound,
}

impl std::fmt::Display for Err {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::NotComposable(details) => write!(f, "Program can not be composed: {details}"),
            Self::NoParametersFound => write!(f, "No crypto parameters could be found"),
        }
    }
}

type Result<T> = std::result::Result<T, Err>;
