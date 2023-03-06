use super::complexity::Complexity;
use super::complexity_model::ComplexityModel;
use crate::parameters::AtomicPatternParameters;

#[allow(dead_code)]
pub fn atomic_pattern_complexity(
    complexity_model: &dyn ComplexityModel,
    sum_size: u64,
    params: AtomicPatternParameters,
    ciphertext_modulus_log: u32,
) -> Complexity {
    let multisum_complexity = complexity_model.levelled_complexity(
        sum_size,
        params.input_lwe_dimension,
        ciphertext_modulus_log,
    );
    let ks_complexity =
        complexity_model.ks_complexity(params.ks_parameters(), ciphertext_modulus_log);
    let pbs_complexity =
        complexity_model.pbs_complexity(params.pbs_parameters(), ciphertext_modulus_log);

    multisum_complexity + ks_complexity + pbs_complexity
}
