use crate::parameters::{BrDecompositionParameters, KsDecompositionParameters};

type Id = u64;
/* An Id is unique per key type. Starting from 0 for the first key ... */
type SecretLweKeyId = Id;
type BootstrapKeyId = Id;
type KeySwitchKeyId = Id;
type ConversionKeySwitchKeyId = Id;

#[derive(Debug, Clone)]
pub struct SecretLweKey {
    /* Big and small secret keys */
    pub identifier: SecretLweKeyId,
    pub polynomial_size: u64,
    pub glwe_dimension: u64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct BootstrapKey {
    /* Public TLU bootstrap keys */
    pub identifier: BootstrapKeyId,
    pub input_key: SecretLweKey,
    pub output_key: SecretLweKey,
    pub br_decomposition_parameter: BrDecompositionParameters,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct KeySwitchKey {
    /* Public TLU keyswitch keys */
    pub identifier: KeySwitchKeyId,
    pub input_key: SecretLweKey,
    pub output_key: SecretLweKey,
    pub ks_decomposition_parameter: KsDecompositionParameters,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ConversionKeySwitchKey {
    /* Public conversion to make cyphertext with incompatible keys compatible.
    It's currently only between two big secret keys. */
    pub identifier: ConversionKeySwitchKeyId,
    pub input_key: SecretLweKey,
    pub output_key: SecretLweKey,
    pub ks_decomposition_parameter: KsDecompositionParameters,
    pub fast_keyswitch: bool,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct CircuitKeys {
    /* All keys used in a circuit, sorted by Id for each key type */
    pub secret_keys: Vec<SecretLweKey>,
    pub keyswitch_keys: Vec<KeySwitchKey>,
    pub bootstrap_keys: Vec<BootstrapKey>,
    pub conversion_keyswitch_keys: Vec<ConversionKeySwitchKey>,
}

#[derive(Debug, Clone)]
pub struct InstructionKeys {
    /* Describe for each intructions what is the key of inputs/outputs.
       For tlus, it gives the internal keyswitch/pbs keys.
       It also express if the output need to be converted to other keys. */
    /* Note: Levelled instructions doesn't need to use any keys.*/
    pub input_key: SecretLweKeyId,
    pub tlu_keyswitch_key: KeySwitchKeyId,
    pub tlu_bootstrap_key: BootstrapKeyId,
    pub output_key: SecretLweKeyId,
    pub extra_conversion_keys: Vec<ConversionKeySwitchKeyId>,
}

#[derive(Debug, Clone)]
pub struct CircuitSolution {
    pub circuit_keys: CircuitKeys,
    /* instructions keys ordered by instructions index of the original dag original (i.e. in same order):
       dag = new()
       instr_0 = dag.input()           // instr_0.index == 0
       instr_1 = dag.add_lut(instr_0)  // instr_1.index == 1
       sol.instructions_keys[instr_0.index] gives instr_0_keys
    */
    pub instructions_keys: Vec<InstructionKeys>,
    /* complexity of the full circuit */
    pub complexity: f64,
    /* highest p_error attained in a TLU */
    pub p_error: f64,
    /* result error rate, assuming any error will propagate to the result */
    pub global_p_error: f64,
}
