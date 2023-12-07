pub mod atomic_pattern;
pub mod config;
pub mod dag;
pub mod decomposition;
pub mod wop_atomic_pattern;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Err {
    NotComposable,
    NoParametersFound,
}

impl std::fmt::Display for Err {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::NotComposable => write!(f, "NotComposable"),
            Self::NoParametersFound => write!(f, "NoParametersFound"),
        }
    }
}

type Result<T> = std::result::Result<T, Err>;
