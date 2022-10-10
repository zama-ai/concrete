pub mod blind_rotate;
pub mod circuit_bootstrap;
pub mod common;
pub mod keyswitch;
pub mod pp_switch;

pub use common::MacroParam;

use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;

use std::sync::Arc;

pub struct PersistDecompCache {
    pub ks: keyswitch::PersistDecompCache,
    pub br: blind_rotate::PersistDecompCache,
    pub pp: pp_switch::PersistDecompCache,
    pub cb: circuit_bootstrap::PersistDecompCache,
}

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Option<Arc<dyn ComplexityModel>>,
) -> PersistDecompCache {
    let complexity_model = complexity_model.unwrap_or_else(|| processing_unit.complexity_model());
    PersistDecompCache {
        ks: keyswitch::cache(security_level, processing_unit, complexity_model.clone()),
        br: blind_rotate::cache(security_level, processing_unit, complexity_model.clone()),
        pp: pp_switch::cache(security_level, processing_unit, complexity_model.clone()),
        cb: circuit_bootstrap::cache(security_level, processing_unit, complexity_model.clone()),
    }
}

trait ComplexityModelClone: ComplexityModel + Clone {}

impl<T: ComplexityModel + Clone> ComplexityModelClone for T {}
