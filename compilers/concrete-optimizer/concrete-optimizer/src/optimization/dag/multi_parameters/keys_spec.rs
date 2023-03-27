use crate::optimization::wop_atomic_pattern;
use crate::parameters::{BrDecompositionParameters, KsDecompositionParameters};

use crate::optimization::dag::multi_parameters::optimize::MacroParameters;

pub type Id = u64;
/* An Id is unique per key type. Starting from 0 for the first key ... */
pub type SecretLweKeyId = Id;
pub type BootstrapKeyId = Id;
pub type KeySwitchKeyId = Id;
pub type ConversionKeySwitchKeyId = Id;
pub type CircuitBoostrapKeyId = Id;
pub type PrivateFunctionalPackingBoostrapKeyId = Id;
pub const NO_KEY_ID: Id = Id::MAX;

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

#[derive(Debug, Clone)]
pub struct CircuitBoostrapKey {
    pub identifier: ConversionKeySwitchKeyId,
    pub representation_key: SecretLweKey,
    pub br_decomposition_parameter: BrDecompositionParameters,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct PrivateFunctionalPackingBoostrapKey {
    pub identifier: PrivateFunctionalPackingBoostrapKeyId,
    pub representation_key: SecretLweKey,
    pub br_decomposition_parameter: BrDecompositionParameters,
    pub description: String,
}

#[derive(Debug, Default, Clone)]
pub struct CircuitKeys {
    /* All keys used in a circuit, sorted by Id for each key type */
    pub secret_keys: Vec<SecretLweKey>,
    pub keyswitch_keys: Vec<KeySwitchKey>,
    pub bootstrap_keys: Vec<BootstrapKey>,
    pub conversion_keyswitch_keys: Vec<ConversionKeySwitchKey>,
    pub circuit_bootstrap_keys: Vec<CircuitBoostrapKey>,
    pub private_functional_packing_keys: Vec<PrivateFunctionalPackingBoostrapKey>,
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
    pub tlu_circuit_bootstrap_key: CircuitBoostrapKeyId,
    pub tlu_private_functional_packing_key: PrivateFunctionalPackingBoostrapKeyId,
    pub output_key: SecretLweKeyId,
    pub extra_conversion_keys: Vec<ConversionKeySwitchKeyId>,
}

#[derive(Debug, Default, Clone)]
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
    pub crt_decomposition: Vec<u64>, // empty in native case
    pub is_feasible: bool,
    pub error_msg: String,
}

impl CircuitSolution {
    pub fn no_solution(error_msg: impl Into<String>) -> Self {
        Self {
            is_feasible: false,
            complexity: f64::INFINITY,
            p_error: 1.0,
            global_p_error: 1.0,
            error_msg: error_msg.into(),
            ..Self::default()
        }
    }

    pub fn from_wop_solution(sol: wop_atomic_pattern::Solution, nb_instr: usize) -> Self {
        let big_key = SecretLweKey {
            identifier: 0,
            polynomial_size: sol.glwe_polynomial_size,
            glwe_dimension: sol.glwe_dimension,
            description: "big-secret".into(),
        };
        let small_key = SecretLweKey {
            identifier: 1,
            polynomial_size: sol.internal_ks_output_lwe_dimension,
            glwe_dimension: 1,
            description: "small-secret".into(),
        };
        let keyswitch_key = KeySwitchKey {
            identifier: 0,
            input_key: big_key.clone(),
            output_key: small_key.clone(),
            ks_decomposition_parameter: KsDecompositionParameters {
                level: sol.ks_decomposition_level_count,
                log2_base: sol.ks_decomposition_base_log,
            },
            description: "tlu keyswitch".into(),
        };
        let bootstrap_key = BootstrapKey {
            identifier: 0,
            input_key: small_key.clone(),
            output_key: big_key.clone(),
            br_decomposition_parameter: BrDecompositionParameters {
                level: sol.br_decomposition_level_count,
                log2_base: sol.br_decomposition_base_log,
            },
            description: "tlu bootstrap".into(),
        };
        let circuit_bootstrap_key = CircuitBoostrapKey {
            identifier: 0,
            representation_key: big_key.clone(),
            br_decomposition_parameter: BrDecompositionParameters {
                level: sol.cb_decomposition_level_count,
                log2_base: sol.cb_decomposition_base_log,
            },
            description: "circuit bootstrap for woppbs".into(),
        };
        let private_functional_packing_key = PrivateFunctionalPackingBoostrapKey {
            identifier: 0,
            representation_key: big_key.clone(),
            br_decomposition_parameter: BrDecompositionParameters {
                level: sol.pp_decomposition_level_count,
                log2_base: sol.pp_decomposition_base_log,
            },
            description: "private functional packing for woppbs".into(),
        };
        let instruction_keys = InstructionKeys {
            input_key: big_key.identifier,
            tlu_keyswitch_key: keyswitch_key.identifier,
            tlu_bootstrap_key: bootstrap_key.identifier,
            output_key: big_key.identifier,
            extra_conversion_keys: vec![],
            tlu_circuit_bootstrap_key: circuit_bootstrap_key.identifier,
            tlu_private_functional_packing_key: private_functional_packing_key.identifier,
        };
        let instructions_keys = vec![instruction_keys; nb_instr];
        let circuit_keys = CircuitKeys {
            secret_keys: [big_key, small_key].into(),
            keyswitch_keys: [keyswitch_key].into(),
            bootstrap_keys: [bootstrap_key].into(),
            conversion_keyswitch_keys: [].into(),
            circuit_bootstrap_keys: [circuit_bootstrap_key].into(),
            private_functional_packing_keys: [private_functional_packing_key].into(),
        };
        let is_feasible = sol.p_error < 1.0;
        let error_msg = if is_feasible {
            ""
        } else {
            "No crypto-parameters for the given constraints"
        }
        .into();
        Self {
            circuit_keys,
            instructions_keys,
            complexity: sol.complexity,
            p_error: sol.p_error,
            global_p_error: sol.p_error,
            crt_decomposition: sol.crt_decomposition,
            is_feasible: true,
            error_msg,
        }
    }
}

pub struct ExpandedCircuitKeys {
    pub big_secret_keys: Vec<SecretLweKey>,
    pub small_secret_keys: Vec<SecretLweKey>,
    pub keyswitch_keys: Vec<Vec<Option<KeySwitchKey>>>,
    pub bootstrap_keys: Vec<BootstrapKey>,
    pub conversion_keyswitch_keys: Vec<Vec<Option<ConversionKeySwitchKey>>>,
    pub circuit_bootstrap_keys: Vec<CircuitBoostrapKey>,
    pub private_functional_packing_keys: Vec<PrivateFunctionalPackingBoostrapKey>,
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
            // for now woppbs never get by that path
            circuit_bootstrap_keys: vec![],
            private_functional_packing_keys: vec![],
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
            circuit_bootstrap_keys: self.circuit_bootstrap_keys,
            private_functional_packing_keys: self.private_functional_packing_keys,
        }
    }
}
