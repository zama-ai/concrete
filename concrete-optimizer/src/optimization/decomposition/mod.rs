pub mod blind_rotate;
pub mod circuit_bootstrap;
pub mod common;
pub mod keyswitch;
pub mod pp_switch;

pub use common::MacroParam;

use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;

use std::sync::Arc;

pub struct PersistDecompCaches {
    pub ks: keyswitch::PersistDecompCache,
    pub br: blind_rotate::PersistDecompCache,
    pub pp: pp_switch::PersistDecompCache,
    pub cb: circuit_bootstrap::PersistDecompCache,
}

pub struct DecompCaches {
    pub blind_rotate: blind_rotate::Cache,
    pub keyswitch: keyswitch::Cache,
    pub pp_switch: pp_switch::Cache,
    pub cb_pbs: circuit_bootstrap::Cache,
}

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Option<Arc<dyn ComplexityModel>>,
) -> PersistDecompCaches {
    PersistDecompCaches::new(security_level, processing_unit, complexity_model)
}

impl PersistDecompCaches {
    pub fn new(
        security_level: u64,
        processing_unit: config::ProcessingUnit,
        complexity_model: Option<Arc<dyn ComplexityModel>>,
    ) -> Self {
        let complexity_model =
            complexity_model.unwrap_or_else(|| processing_unit.complexity_model());
        Self {
            ks: keyswitch::cache(security_level, processing_unit, complexity_model.clone()),
            br: blind_rotate::cache(security_level, processing_unit, complexity_model.clone()),
            pp: pp_switch::cache(security_level, processing_unit, complexity_model.clone()),
            cb: circuit_bootstrap::cache(security_level, processing_unit, complexity_model.clone()),
        }
    }

    pub fn backport(&self, cache: DecompCaches) {
        self.ks.backport(cache.keyswitch);
        self.br.backport(cache.blind_rotate);
        self.pp.backport(cache.pp_switch);
        self.cb.backport(cache.cb_pbs);
    }

    pub fn caches(&self) -> DecompCaches {
        DecompCaches {
            blind_rotate: self.br.cache(),
            keyswitch: self.ks.cache(),
            pp_switch: self.pp.cache(),
            cb_pbs: self.cb.cache(),
        }
    }
}
