pub mod blind_rotate;
pub mod common;
pub mod cut;
pub mod keyswitch;

use std::sync::Arc;

pub use common::MacroParam;
pub use cut::cut_complexity_noise;

use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;

pub struct PersistDecompCache {
    pub ks: keyswitch::PersistDecompCache,
    pub br: blind_rotate::PersistDecompCache,
}

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Option<Arc<dyn ComplexityModel>>,
) -> PersistDecompCache {
    PersistDecompCache {
        ks: keyswitch::cache(security_level, processing_unit, complexity_model.clone()),
        br: blind_rotate::cache(security_level, processing_unit, complexity_model),
    }
}

trait ComplexityModelClone: ComplexityModel + Clone {}

impl<T: ComplexityModel + Clone> ComplexityModelClone for T {}
