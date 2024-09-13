use dag::multi_parameters::variance_constraint::VarianceConstraint;

pub mod atomic_pattern;
pub mod config;
pub mod dag;
pub mod decomposition;
pub mod wop_atomic_pattern;

#[derive(Clone, Debug, PartialEq)]
pub enum Err {
    NotComposable(String),
    NoParametersFound,
    UnfeasibleVarianceConstraint(Box<VarianceConstraint>),
}

impl std::fmt::Display for Err {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::NotComposable(details) => write!(f, "Program can not be composed (see https://docs.zama.ai/concrete/compilation/common_errors#id-9.-non-composable-circuit): {details}"),
            Self::NoParametersFound => write!(f, "No crypto parameters could be found"),
            Self::UnfeasibleVarianceConstraint(constraint) => {
                write!(
                    f,
                    "Unfeasible noise constraint encountered (see https://docs.zama.ai/concrete/compilation/common_errors#id-8.-unfeasible-noise-constraint): At location {}:\n{}.",
                    constraint.location,
                    constraint
                )
            }
        }
    }
}

type Result<T> = std::result::Result<T, Err>;
