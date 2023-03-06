#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)] // u64 to f64
#![allow(clippy::cast_possible_truncation)] // u64 to usize
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::similar_names)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::too_many_arguments)]
#![warn(unused_results)]

pub mod computing_cost;

pub mod config;
pub mod dag;
pub mod global_parameters;
pub mod noise_estimator;
pub mod optimization;
pub mod parameters;
pub mod utils;
pub mod weight;

pub use concrete_security_curves::gaussian::security::supported_security_levels;
