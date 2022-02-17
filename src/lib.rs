#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![allow(clippy::cast_precision_loss)] // u64 to f64
#![allow(clippy::cast_possible_truncation)] // u64 to usize
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unreadable_literal)]
#![warn(unused_results)]

pub mod computing_cost;

pub mod global_parameters;
pub mod graph;
pub mod noise_estimator;
pub mod parameters;
pub mod weight;
