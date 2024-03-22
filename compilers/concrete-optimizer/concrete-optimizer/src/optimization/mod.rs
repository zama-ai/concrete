pub mod atomic_pattern;
pub mod config;
pub mod dag;
pub mod decomposition;
pub mod wop_atomic_pattern;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Err {
    NotComposable(String),
    NoParametersFound(Vec<usize>),
}

impl Err {
    fn error_nodes(&self) -> Vec<usize> {
        match self {
            Self::NotComposable(_details) => vec![],
            Self::NoParametersFound(nodes) => nodes.clone(),
        }
    }
}

impl std::fmt::Display for Err {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::NotComposable(details) => write!(f, "Program can not be composed: {details}"),
            Self::NoParametersFound(_) => write!(f, "No crypto parameters could be found"),
        }
    }
}

type Result<T> = std::result::Result<T, Err>;
