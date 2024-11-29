use std::collections::HashMap;

use crate::optimization::{atomic_pattern, wop_atomic_pattern};
use crate::parameters::{BrDecompositionParameters, KsDecompositionParameters};

use crate::optimization::dag::multi_parameters::optimize::{MacroParameters, REAL_FAST_KS};

pub type Id = u64;
/* An Id is unique per key type. Starting from 0 for the first key ... */
pub type SecretLweKeyId = Id;
pub type BootstrapKeyId = Id;
pub type KeySwitchKeyId = Id;
pub type ConversionKeySwitchKeyId = Id;
pub type CircuitBoostrapKeyId = Id;
pub type PrivateFunctionalPackingBoostrapKeyId = Id;
pub const NO_KEY_ID: Id = Id::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretLweKey {
    /* Big and small secret keys */
    pub identifier: SecretLweKeyId,
    pub polynomial_size: u64,
    pub glwe_dimension: u64,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapKey {
    /* Public TLU bootstrap keys */
    pub identifier: BootstrapKeyId,
    pub input_key: SecretLweKey,
    pub output_key: SecretLweKey,
    pub br_decomposition_parameter: BrDecompositionParameters,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeySwitchKey {
    /* Public TLU keyswitch keys */
    pub identifier: KeySwitchKeyId,
    pub input_key: SecretLweKey,
    pub output_key: SecretLweKey,
    pub ks_decomposition_parameter: KsDecompositionParameters,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    /* Describe for each instructions what is the key of inputs/outputs.
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

    pub fn from_native_solution(sol: atomic_pattern::Solution, nb_instr: usize) -> Self {
        let is_feasible = sol.p_error < 1.0;
        let error_msg = if is_feasible {
            ""
        } else {
            "No crypto-parameters for the given constraints"
        }
        .into();
        let big_key = SecretLweKey {
            identifier: 0,
            polynomial_size: sol.glwe_polynomial_size,
            glwe_dimension: sol.glwe_dimension,
            description: "big representation".into(),
        };
        if sol.internal_ks_output_lwe_dimension == 0 {
            let instruction_keys = InstructionKeys {
                input_key: big_key.identifier,
                tlu_keyswitch_key: NO_KEY_ID,
                tlu_bootstrap_key: NO_KEY_ID,
                tlu_circuit_bootstrap_key: NO_KEY_ID,
                tlu_private_functional_packing_key: NO_KEY_ID,
                output_key: big_key.identifier,
                extra_conversion_keys: vec![],
            };
            let circuit_keys = CircuitKeys {
                secret_keys: [big_key].into(),
                keyswitch_keys: [].into(),
                bootstrap_keys: [].into(),
                conversion_keyswitch_keys: [].into(),
                circuit_bootstrap_keys: [].into(),
                private_functional_packing_keys: [].into(),
            };
            return Self {
                circuit_keys,
                instructions_keys: vec![instruction_keys; nb_instr],
                crt_decomposition: vec![],
                complexity: sol.complexity,
                p_error: sol.p_error,
                global_p_error: sol.global_p_error,
                is_feasible,
                error_msg,
            };
        }
        let small_key = SecretLweKey {
            identifier: 1,
            polynomial_size: sol.internal_ks_output_lwe_dimension,
            glwe_dimension: 1,
            description: "small representation".into(),
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
        let instruction_keys = InstructionKeys {
            input_key: big_key.identifier,
            tlu_keyswitch_key: keyswitch_key.identifier,
            tlu_bootstrap_key: bootstrap_key.identifier,
            tlu_circuit_bootstrap_key: NO_KEY_ID,
            tlu_private_functional_packing_key: NO_KEY_ID,
            output_key: big_key.identifier,
            extra_conversion_keys: vec![],
        };
        let instructions_keys = vec![instruction_keys; nb_instr];
        let circuit_keys = CircuitKeys {
            secret_keys: [big_key, small_key].into(),
            keyswitch_keys: [keyswitch_key].into(),
            bootstrap_keys: [bootstrap_key].into(),
            conversion_keyswitch_keys: [].into(),
            circuit_bootstrap_keys: [].into(),
            private_functional_packing_keys: [].into(),
        };
        Self {
            circuit_keys,
            instructions_keys,
            crt_decomposition: vec![],
            complexity: sol.complexity,
            p_error: sol.p_error,
            global_p_error: sol.global_p_error,
            is_feasible,
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

pub struct KeySharing {
    secret_keys: HashMap<Id, SecretLweKey>,
    bootstrap_keys: HashMap<Id, BootstrapKey>,
    keyswitch_keys: HashMap<Id, KeySwitchKey>,
    conversion_keyswitch_keys: HashMap<Id, Option<ConversionKeySwitchKey>>,
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
                        fast_keyswitch: REAL_FAST_KS,
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

    fn shared_boostrap_keys(
        &self,
        final_keys: &HashMap<Id, SecretLweKey>,
        final_groups: &HashMap<Id, Vec<Id>>,
    ) -> (Vec<BootstrapKey>, HashMap<Id, BootstrapKey>) {
        let final_key_id = |k: &SecretLweKey| final_keys[&k.identifier].identifier;
        let mut canon_final_bootstraps: HashMap<(Id, Id, BrDecompositionParameters), BootstrapKey> =
            HashMap::new();
        let mut final_bootstraps: HashMap<Id, BootstrapKey> = HashMap::new();
        let mut bootstrap_keys = vec![];
        for key in &self.bootstrap_keys {
            let final_id_in = final_key_id(&key.input_key);
            let final_id_out = final_key_id(&key.output_key);
            let canon_key = (final_id_in, final_id_out, key.br_decomposition_parameter);
            #[allow(clippy::option_if_let_else)]
            let final_bootstrap =
                if let Some(final_bootstrap) = canon_final_bootstraps.get(&canon_key) {
                    final_bootstrap.clone()
                } else {
                    let mut final_bootstrap = key.clone();
                    final_bootstrap.identifier = canon_final_bootstraps.len() as u64;
                    final_bootstrap.input_key = final_keys[&key.input_key.identifier].clone();
                    final_bootstrap.output_key = final_keys[&key.output_key.identifier].clone();
                    final_bootstrap.description = format!(
                        "pbs[#{} : partitions {:?} -> {:?}]",
                        final_bootstrap.identifier,
                        final_groups[&final_id_in],
                        final_groups[&final_id_out]
                    );
                    _ = canon_final_bootstraps.insert(canon_key, final_bootstrap.clone());
                    bootstrap_keys.push(final_bootstrap.clone());
                    final_bootstrap
                };
            _ = final_bootstraps.insert(key.identifier, final_bootstrap.clone());
        }
        (bootstrap_keys, final_bootstraps)
    }

    fn shared_keyswitch_keys(
        &self,
        final_keys: &HashMap<Id, SecretLweKey>,
        final_groups: &HashMap<Id, Vec<Id>>,
    ) -> (Vec<Vec<Option<KeySwitchKey>>>, HashMap<Id, KeySwitchKey>) {
        let final_key_id = |k: &SecretLweKey| final_keys[&k.identifier].identifier;
        let mut canon_final_keyswitchs: HashMap<(Id, Id, KsDecompositionParameters), KeySwitchKey> =
            HashMap::new();
        let mut final_keyswitchs: HashMap<Id, KeySwitchKey> = HashMap::new();
        let mut keyswitch_keys = self.keyswitch_keys.clone();
        for (i, keys) in self.keyswitch_keys.iter().enumerate() {
            for (j, key) in keys.iter().enumerate() {
                if let Some(key) = key {
                    let final_id_in = final_key_id(&key.input_key);
                    let final_id_out = final_key_id(&key.output_key);
                    let canon_key = (final_id_in, final_id_out, key.ks_decomposition_parameter);
                    #[allow(clippy::option_if_let_else)]
                    let final_keyswitch = if let Some(final_keyswitch) =
                        canon_final_keyswitchs.get(&canon_key)
                    {
                        keyswitch_keys[i][j] = None;
                        final_keyswitch.clone()
                    } else {
                        let mut final_keyswitch = key.clone();
                        final_keyswitch.identifier = canon_final_keyswitchs.len() as u64;
                        final_keyswitch.input_key = final_keys[&key.input_key.identifier].clone();
                        final_keyswitch.output_key = final_keys[&key.output_key.identifier].clone();
                        final_keyswitch.description = format!(
                            "ks[#{} : partitions {:?} -> {:?}]",
                            final_keyswitch.identifier,
                            final_groups[&final_id_in],
                            final_groups[&final_id_out]
                        );
                        _ = canon_final_keyswitchs.insert(canon_key, final_keyswitch.clone());
                        keyswitch_keys[i][j] = Some(final_keyswitch.clone());
                        final_keyswitch
                    };
                    _ = final_keyswitchs.insert(key.identifier, final_keyswitch.clone());
                }
            }
        }
        (keyswitch_keys, final_keyswitchs)
    }

    #[allow(clippy::type_complexity)]
    fn shared_conversion_keyswitch_keys(
        &self,
        final_keys: &HashMap<Id, SecretLweKey>,
        final_groups: &HashMap<Id, Vec<Id>>,
    ) -> (
        Vec<Vec<Option<ConversionKeySwitchKey>>>,
        HashMap<Id, Option<ConversionKeySwitchKey>>,
    ) {
        let final_key_id = |k: &SecretLweKey| final_keys[&k.identifier].identifier;
        let mut canon_final_c_keyswitchs: HashMap<
            (Id, Id, KsDecompositionParameters),
            ConversionKeySwitchKey,
        > = HashMap::new();
        let mut final_c_keyswitchs: HashMap<Id, Option<ConversionKeySwitchKey>> = HashMap::new();
        let mut conversion_keyswitch_keys = self.conversion_keyswitch_keys.clone();
        for (i, keys) in self.conversion_keyswitch_keys.iter().enumerate() {
            for (j, key) in keys.iter().enumerate() {
                if let Some(key) = key {
                    let final_id_in = final_key_id(&key.input_key);
                    let final_id_out = final_key_id(&key.output_key);
                    if final_id_in == final_id_out {
                        conversion_keyswitch_keys[i][j] = None;
                        _ = final_c_keyswitchs.insert(key.identifier, None);
                        continue;
                    }
                    let canon_key = (final_id_in, final_id_out, key.ks_decomposition_parameter);
                    #[allow(clippy::option_if_let_else)]
                    let final_c_keyswitch = if let Some(final_c_keyswitch) =
                        canon_final_c_keyswitchs.get(&canon_key)
                    {
                        conversion_keyswitch_keys[i][j] = None;
                        final_c_keyswitch.clone()
                    } else {
                        let mut final_c_keyswitch = key.clone();
                        final_c_keyswitch.identifier = canon_final_c_keyswitchs.len() as u64;
                        final_c_keyswitch.input_key = final_keys[&key.input_key.identifier].clone();
                        final_c_keyswitch.output_key =
                            final_keys[&key.output_key.identifier].clone();
                        final_c_keyswitch.description = format!(
                            "fks[#{} : partitions {:?} -> {:?}]",
                            final_c_keyswitch.identifier,
                            final_groups[&final_id_in],
                            final_groups[&final_id_out]
                        );
                        _ = canon_final_c_keyswitchs.insert(canon_key, final_c_keyswitch.clone());
                        conversion_keyswitch_keys[i][j] = Some(final_c_keyswitch.clone());
                        final_c_keyswitch
                    };
                    _ = final_c_keyswitchs.insert(key.identifier, Some(final_c_keyswitch.clone()));
                }
            }
        }
        (conversion_keyswitch_keys, final_c_keyswitchs)
    }

    #[allow(clippy::too_many_lines)]
    pub fn shared_keys(self) -> (Self, KeySharing) {
        // initial key to common key
        let mut leader: HashMap<Id, &SecretLweKey> = HashMap::new();
        let mut groups: HashMap<Id, Vec<Id>> = HashMap::new();
        // initial key to final key (identifier change + description change)
        let mut final_keys: HashMap<Id, SecretLweKey> = HashMap::new();
        let mut final_groups: HashMap<Id, Vec<Id>> = HashMap::new();
        let mut new_secret_keys = [vec![], vec![]];

        for (case, &secret_keys) in [&self.big_secret_keys, &self.small_secret_keys]
            .iter()
            .enumerate()
        {
            for (i, key0) in secret_keys.iter().enumerate() {
                let already_shared = leader.contains_key(&key0.identifier);
                if already_shared {
                    continue;
                }
                _ = leader.insert(key0.identifier, key0);
                _ = groups.insert(key0.identifier, vec![key0.identifier]);
                // Finding all similar keys, making this a group
                for key1 in &secret_keys[i + 1..] {
                    let same_size = key0.polynomial_size == key1.polynomial_size
                        && key0.glwe_dimension == key1.glwe_dimension;
                    if same_size {
                        _ = leader.insert(key1.identifier, key0);
                        groups
                            .get_mut(&key0.identifier)
                            .unwrap()
                            .push(key1.identifier);
                    }
                }
            }
            // Create the unified key based on the groups
            for key in secret_keys {
                if !groups.contains_key(&key.identifier) {
                    continue;
                }
                let leader_key = key;
                let group = groups[&leader_key.identifier].clone();
                let mut final_key = leader_key.clone();
                final_key.identifier = (new_secret_keys[0].len() + new_secret_keys[1].len()) as u64;
                let case_str = if case == 0 { "big" } else { "small" };
                final_key.description = format!(
                    "{}-secret[#{} : partitions {:?}]",
                    case_str, final_key.identifier, group
                );
                _ = final_groups.insert(final_key.identifier, vec![]);
                new_secret_keys[case].push(final_key.clone());
                for key_id in group {
                    _ = final_keys.insert(key_id, final_key.clone());
                    final_groups
                        .get_mut(&final_key.identifier)
                        .unwrap()
                        .push(key_id);
                }
            }
        }
        let big_secret_keys = new_secret_keys[0].clone();
        let small_secret_keys = new_secret_keys[1].clone();

        let (bootstrap_keys, final_bootstraps) =
            self.shared_boostrap_keys(&final_keys, &final_groups);
        let (keyswitch_keys, final_keyswitchs) =
            self.shared_keyswitch_keys(&final_keys, &final_groups);
        let (conversion_keyswitch_keys, final_c_keyswitchs) =
            self.shared_conversion_keyswitch_keys(&final_keys, &final_groups);
        (
            Self {
                big_secret_keys,
                small_secret_keys,
                keyswitch_keys,
                bootstrap_keys,
                conversion_keyswitch_keys,
                ..self
            },
            KeySharing {
                secret_keys: final_keys,
                keyswitch_keys: final_keyswitchs,
                bootstrap_keys: final_bootstraps,
                conversion_keyswitch_keys: final_c_keyswitchs,
            },
        )
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

impl InstructionKeys {
    fn shared_keys_1(&self, sharing: &KeySharing) -> Self {
        let tlu_keyswitch_key = if self.tlu_keyswitch_key == NO_KEY_ID {
            NO_KEY_ID
        } else {
            sharing.keyswitch_keys[&self.tlu_keyswitch_key].identifier
        };
        let tlu_bootstrap_key = if self.tlu_bootstrap_key == NO_KEY_ID {
            NO_KEY_ID
        } else {
            sharing.bootstrap_keys[&self.tlu_bootstrap_key].identifier
        };
        Self {
            input_key: sharing.secret_keys[&self.input_key].identifier,
            tlu_bootstrap_key,
            tlu_keyswitch_key,
            output_key: sharing.secret_keys[&self.output_key].identifier,
            extra_conversion_keys: self
                .extra_conversion_keys
                .iter()
                .filter_map(|fks_id| {
                    sharing.conversion_keyswitch_keys[fks_id]
                        .as_ref()
                        .map(|k| k.identifier)
                })
                .collect(),
            ..self.clone()
        }
    }

    pub fn shared_keys(instructions_keys: &[Self], sharing: &KeySharing) -> Vec<Self> {
        instructions_keys
            .iter()
            .map(|i| i.shared_keys_1(sharing))
            .collect()
    }
}
