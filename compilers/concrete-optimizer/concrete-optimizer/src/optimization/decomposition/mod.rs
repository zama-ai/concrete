pub mod circuit_bootstrap;
pub mod cmux;
pub mod common;
pub mod keyswitch;
pub mod pp_switch;

pub use common::MacroParam;

use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;

use std::sync::Arc;

pub struct PersistDecompCaches {
    pub ks: keyswitch::PersistDecompCache,
    pub cmux: cmux::PersistDecompCache,
    pub pp: pp_switch::PersistDecompCache,
    pub cb: circuit_bootstrap::PersistDecompCache,
    pub cache_on_disk: bool,
}

pub struct DecompCaches {
    pub cmux: cmux::Cache,
    pub keyswitch: keyswitch::Cache,
    pub pp_switch: pp_switch::Cache,
    pub cb_pbs: circuit_bootstrap::Cache,
}

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Option<Arc<dyn ComplexityModel>>,
    cache_on_disk: bool,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
) -> PersistDecompCaches {
    PersistDecompCaches::new(
        security_level,
        processing_unit,
        complexity_model,
        cache_on_disk,
        ciphertext_modulus_log,
        fft_precision,
    )
}

impl PersistDecompCaches {
    pub fn new(
        security_level: u64,
        processing_unit: config::ProcessingUnit,
        complexity_model: Option<Arc<dyn ComplexityModel>>,
        cache_on_disk: bool,
        ciphertext_modulus_log: u32,
        fft_precision: u32,
    ) -> Self {
        let complexity_model =
            complexity_model.unwrap_or_else(|| processing_unit.complexity_model());
        let res = Self {
            ks: keyswitch::cache(
                security_level,
                processing_unit,
                complexity_model.clone(),
                ciphertext_modulus_log,
            ),
            cmux: cmux::cache(
                security_level,
                processing_unit,
                complexity_model.clone(),
                ciphertext_modulus_log,
                fft_precision,
            ),
            pp: pp_switch::cache(
                security_level,
                processing_unit,
                complexity_model.clone(),
                ciphertext_modulus_log,
            ),
            cb: circuit_bootstrap::cache(
                security_level,
                processing_unit,
                complexity_model,
                ciphertext_modulus_log,
                fft_precision,
            ),
            cache_on_disk,
        };
        if cache_on_disk {
            res.ks.read();
            res.cmux.read();
            res.pp.read();
            res.cb.read();
        }
        res
    }

    pub fn backport(&self, cache: DecompCaches) {
        if !self.cache_on_disk {
            return;
        }
        self.ks.backport(cache.keyswitch);
        self.cmux.backport(cache.cmux);
        self.pp.backport(cache.pp_switch);
        self.cb.backport(cache.cb_pbs);
    }

    pub fn caches(&self) -> DecompCaches {
        DecompCaches {
            cmux: self.cmux.cache(),
            keyswitch: self.ks.cache(),
            pp_switch: self.pp.cache(),
            cb_pbs: self.cb.cache(),
        }
    }
}
