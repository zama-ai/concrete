use crate::parameters::{BrDecompositionParameters, KsDecompositionParameters};

use crate::optimization::dag::multi_parameters::optimize::MacroParameters;

pub type Id = u64;
/* An Id is unique per key type. Starting from 0 for the first key ... */
pub type SecretLweKeyId = Id;
pub type BootstrapKeyId = Id;
pub type KeySwitchKeyId = Id;
pub type ConversionKeySwitchKeyId = Id;

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
    /* Public conversion to make compatible ciphertext with incompatible keys.
    It's currently only between two big secret keys. */
    pub identifier: ConversionKeySwitchKeyId,
    pub input_key: SecretLweKey,
    pub output_key: SecretLweKey,
    pub ks_decomposition_parameter: KsDecompositionParameters,
    pub fast_keyswitch: bool,
    pub description: String,
}

#[derive(Debug, Default, Clone)]
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
    pub is_feasible: bool,
    pub error_msg: String,
}

pub struct ExpandedCircuitKeys {
    pub big_secret_keys: Vec<SecretLweKey>,
    pub small_secret_keys: Vec<SecretLweKey>,
    pub keyswitch_keys: Vec<Vec<Option<KeySwitchKey>>>,
    pub bootstrap_keys: Vec<BootstrapKey>,
    pub conversion_keyswitch_keys: Vec<Vec<Option<ConversionKeySwitchKey>>>,
}

impl ExpandedCircuitKeys {
    pub fn of(params: &super::optimize::Parameters) -> Self {
        let nb_partitions = params.macro_params.len();
        let big_secret_keys: Vec<_> = params
            .macro_params
            .iter()
            .enumerate()
            .map(|(i, v): (usize, &Option<MacroParameters>)| {
                let glwe_params = v.unwrap().glwe_params;
                SecretLweKey {
                    identifier: i as Id,
                    polynomial_size: glwe_params.polynomial_size(),
                    glwe_dimension: glwe_params.glwe_dimension,
                    description: format!("big-secret[{i}]"),
                }
            })
            .collect();
        let small_secret_keys: Vec<_> = params
            .macro_params
            .iter()
            .enumerate()
            .map(|(i, v): (usize, &Option<MacroParameters>)| {
                let polynomial_size = v.unwrap().internal_dim;
                SecretLweKey {
                    identifier: (nb_partitions + i) as Id,
                    polynomial_size,
                    glwe_dimension: 1,
                    description: format!("small-secret[{i}]"),
                }
            })
            .collect();
        let bootstrap_keys: Vec<_> = params
            .micro_params
            .pbs
            .iter()
            .enumerate()
            .map(|(i, v): (usize, &Option<_>)| {
                let br_decomposition_parameter = v.unwrap().decomp;
                BootstrapKey {
                    identifier: i as Id,
                    input_key: small_secret_keys[i].clone(),
                    output_key: big_secret_keys[i].clone(),
                    br_decomposition_parameter,
                    description: format!("pbs[{i}]"),
                }
            })
            .collect();
        let mut keyswitch_keys = vec![vec![None; nb_partitions]; nb_partitions];
        let mut conversion_keyswitch_keys = vec![vec![None; nb_partitions]; nb_partitions];
        let mut identifier_ks = 0 as Id;
        let mut identifier_fks = 0 as Id;
        #[allow(clippy::needless_range_loop)]
        for src in 0..nb_partitions {
            for dst in 0..nb_partitions {
                let cross_key = |name: &str| {
                    if src == dst {
                        format!("{name}[{src}]")
                    } else {
                        format!("{name}[{src}->{dst}]")
                    }
                };
                if let Some(ks) = params.micro_params.ks[src][dst] {
                    let identifier = identifier_ks;
                    keyswitch_keys[src][dst] = Some(KeySwitchKey {
                        identifier,
                        input_key: big_secret_keys[src].clone(),
                        output_key: small_secret_keys[dst].clone(),
                        ks_decomposition_parameter: ks.decomp,
                        description: cross_key("ks"),
                    });
                    identifier_ks += 1;
                }
                if let Some(fks) = params.micro_params.fks[src][dst] {
                    let identifier = identifier_fks;
                    conversion_keyswitch_keys[src][dst] = Some(ConversionKeySwitchKey {
                        identifier,
                        input_key: big_secret_keys[src].clone(),
                        output_key: big_secret_keys[dst].clone(),
                        ks_decomposition_parameter: fks.decomp,
                        fast_keyswitch: true,
                        description: cross_key("fks"),
                    });
                    identifier_fks += 1;
                }
            }
        }
        Self {
            big_secret_keys,
            small_secret_keys,
            keyswitch_keys,
            bootstrap_keys,
            conversion_keyswitch_keys,
        }
    }

    pub fn compacted(self) -> CircuitKeys {
        CircuitKeys {
            secret_keys: [self.big_secret_keys, self.small_secret_keys].concat(),
            keyswitch_keys: self
                .keyswitch_keys
                .into_iter()
                .flatten()
                .flatten()
                .collect(),
            bootstrap_keys: self.bootstrap_keys,
            conversion_keyswitch_keys: self
                .conversion_keyswitch_keys
                .into_iter()
                .flatten()
                .flatten()
                .collect(),
        }
    }
}
