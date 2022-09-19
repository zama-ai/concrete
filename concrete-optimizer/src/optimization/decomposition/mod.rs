pub mod blind_rotate;
pub mod common;
pub mod cut;
pub mod keyswitch;

pub use common::MacroParam;
pub use cut::cut_complexity_noise;

pub struct PersistDecompCache {
    pub ks: keyswitch::PersistDecompCache,
    pub br: blind_rotate::PersistDecompCache,
}

pub fn cache(security_level: u64) -> PersistDecompCache {
    PersistDecompCache {
        ks: keyswitch::cache(security_level),
        br: blind_rotate::cache(security_level),
    }
}
