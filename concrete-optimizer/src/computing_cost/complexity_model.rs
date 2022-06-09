use super::complexity::Complexity;
use crate::parameters::{KeyswitchParameters, LweDimension, PbsParameters};

pub trait ComplexityModel: Send + Sync {
    fn pbs_complexity(&self, params: PbsParameters, ciphertext_modulus_log: u32) -> Complexity;
    fn ks_complexity(&self, params: KeyswitchParameters, ciphertext_modulus_log: u32)
        -> Complexity;
    fn levelled_complexity(
        &self,
        sum_size: u64,
        lwe_dimension: LweDimension,
        ciphertext_modulus_log: u32,
    ) -> Complexity;
}
