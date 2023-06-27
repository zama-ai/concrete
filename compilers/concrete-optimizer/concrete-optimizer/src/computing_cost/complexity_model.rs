use super::complexity::Complexity;
use crate::parameters::{
    CmuxParameters, KeyswitchParameters, LweDimension, PbsParameters, TensorProductGlweParameters,
    TracePackingParameters,
};

pub trait ComplexityModel: Send + Sync {
    fn pbs_complexity(&self, params: PbsParameters, ciphertext_modulus_log: u32) -> Complexity;
    fn cmux_complexity(&self, params: CmuxParameters, ciphertext_modulus_log: u32) -> Complexity;
    fn ks_complexity(&self, params: KeyswitchParameters, ciphertext_modulus_log: u32)
        -> Complexity;
    fn fft_complexity(&self, glwe_polynomial_size: f64, ciphertext_modulus_log: u32) -> Complexity;
    fn levelled_complexity(
        &self,
        sum_size: u64,
        lwe_dimension: LweDimension,
        ciphertext_modulus_log: u32,
    ) -> Complexity;
    fn multi_bit_pbs_complexity(
        &self,
        params: PbsParameters,
        ciphertext_modulus_log: u32,
        grouping_factor: u32,
        jit_fft: bool,
    ) -> Complexity;
    fn tensor_product_complexity(
        &self,
        params: TensorProductGlweParameters,
        ciphertext_modulus_log: u32,
    ) -> Complexity;
    fn trace_packing_complexity(
        &self,
        params: TracePackingParameters,
        ciphertext_modulus_log: u32,
        index_set: &[usize],
    ) -> Complexity;
}
