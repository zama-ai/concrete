use crate::{
    optimization::dag::multi_parameters::{
        partition_cut::PartitionCut, partitions::PartitionIndex,
    },
    parameters::{BrDecompositionParameters, GlweParameters, KsDecompositionParameters},
};
use serde::{Deserialize, Serialize};

use super::MacroParameters;

/// A trait to restrict search space in the optimization algorithm.
///
/// The trait methods are called at different level of the optimization algorithm to
/// perform cuts depending on whether the parameters are available.
pub trait SearchSpaceRestriction {
    /// Return whether the glwe parameters are available for the given partition.
    fn is_available_glwe(&self, partition: PartitionIndex, glwe_params: GlweParameters) -> bool;

    /// Return whether the macro parameters are available for the given partition.
    fn is_available_macro(
        &self,
        partition: PartitionIndex,
        macro_parameters: MacroParameters,
    ) -> bool;

    /// Return whether the pbs parameters are available for the given partition.
    fn is_available_micro_pbs(
        &self,
        partition: PartitionIndex,
        macro_parameters: MacroParameters,
        pbs_parameters: BrDecompositionParameters,
    ) -> bool;

    /// Return whether the ks parameters are available for the given partitions.
    fn is_available_micro_ks(
        &self,
        from_partition: PartitionIndex,
        from_macro: MacroParameters,
        to_partition: PartitionIndex,
        to_macro: MacroParameters,
        ks_parameters: KsDecompositionParameters,
    ) -> bool;

    /// Return whether the fks parameters are available for the given partitions.
    fn is_available_micro_fks(
        &self,
        from_partition: PartitionIndex,
        from_macro: MacroParameters,
        to_partition: PartitionIndex,
        to_macro: MacroParameters,
        ks_parameters: KsDecompositionParameters,
    ) -> bool;
}

// Allow references to restrictions to be used as restrictions.
impl<A: SearchSpaceRestriction> SearchSpaceRestriction for &A {
    fn is_available_glwe(&self, partition: PartitionIndex, glwe_params: GlweParameters) -> bool {
        (*self).is_available_glwe(partition, glwe_params)
    }

    fn is_available_macro(
        &self,
        partition: PartitionIndex,
        macro_parameters: MacroParameters,
    ) -> bool {
        (*self).is_available_macro(partition, macro_parameters)
    }

    fn is_available_micro_pbs(
        &self,
        partition: PartitionIndex,
        macro_parameters: MacroParameters,
        pbs_parameters: BrDecompositionParameters,
    ) -> bool {
        (*self).is_available_micro_pbs(partition, macro_parameters, pbs_parameters)
    }

    fn is_available_micro_ks(
        &self,
        from_partition: PartitionIndex,
        from_macro: MacroParameters,
        to_partition: PartitionIndex,
        to_macro: MacroParameters,
        ks_parameters: KsDecompositionParameters,
    ) -> bool {
        (*self).is_available_micro_ks(
            from_partition,
            from_macro,
            to_partition,
            to_macro,
            ks_parameters,
        )
    }

    fn is_available_micro_fks(
        &self,
        from_partition: PartitionIndex,
        from_macro: MacroParameters,
        to_partition: PartitionIndex,
        to_macro: MacroParameters,
        ks_parameters: KsDecompositionParameters,
    ) -> bool {
        (*self).is_available_micro_fks(
            from_partition,
            from_macro,
            to_partition,
            to_macro,
            ks_parameters,
        )
    }
}

// Allow tuples of restrictions to be used as restriction
macro_rules! impl_tuple {
    ($($gen_ty: ident),*) => {
        impl<$($gen_ty : SearchSpaceRestriction),*> SearchSpaceRestriction for ($($gen_ty),*){

            #[allow(non_snake_case)]
            fn is_available_glwe(&self, partition: PartitionIndex, glwe_params: GlweParameters) -> bool{
                let ($($gen_ty),*) = self;
                $($gen_ty.is_available_glwe(partition, glwe_params))&&*
            }

            #[allow(non_snake_case)]
            fn is_available_macro(
                &self,
                partition: PartitionIndex,
                macro_parameters: MacroParameters,
            ) -> bool {
                let ($($gen_ty),*) = self;
                $($gen_ty.is_available_macro(partition, macro_parameters))&&*
            }

            #[allow(non_snake_case)]
            fn is_available_micro_pbs(
                &self,
                partition: PartitionIndex,
                macro_parameters: MacroParameters,
                pbs_parameters: BrDecompositionParameters,
            ) -> bool {
                let ($($gen_ty),*) = self;
                $($gen_ty.is_available_micro_pbs(partition, macro_parameters, pbs_parameters))&&*
            }

            #[allow(non_snake_case)]
            fn is_available_micro_ks(
                &self,
                from_partition: PartitionIndex,
                from_macro: MacroParameters,
                to_partition: PartitionIndex,
                to_macro: MacroParameters,
                ks_parameters: KsDecompositionParameters,
            ) -> bool {
                let ($($gen_ty),*) = self;
                $($gen_ty.is_available_micro_ks(from_partition, from_macro, to_partition, to_macro, ks_parameters))&&*
            }

            #[allow(non_snake_case)]
            fn is_available_micro_fks(
                &self,
                from_partition: PartitionIndex,
                from_macro: MacroParameters,
                to_partition: PartitionIndex,
                to_macro: MacroParameters,
                ks_parameters: KsDecompositionParameters,
            ) -> bool {
                let ($($gen_ty),*) = self;
                $($gen_ty.is_available_micro_fks(from_partition, from_macro, to_partition, to_macro, ks_parameters))&&*
            }
        }
    };
}

impl_tuple! {A, B}
impl_tuple! {A, B, C}
impl_tuple! {A, B, C, D}
impl_tuple! {A, B, C, D, E}

/// A restriction performing no restriction at all.
pub struct NoSearchSpaceRestriction;

impl SearchSpaceRestriction for NoSearchSpaceRestriction {
    fn is_available_glwe(&self, _partition: PartitionIndex, _glwe_params: GlweParameters) -> bool {
        true
    }

    fn is_available_macro(
        &self,
        _partition: PartitionIndex,
        _macro_parameters: MacroParameters,
    ) -> bool {
        true
    }

    fn is_available_micro_pbs(
        &self,
        _partition: PartitionIndex,
        _macro_parameters: MacroParameters,
        _pbs_parameters: BrDecompositionParameters,
    ) -> bool {
        true
    }

    fn is_available_micro_ks(
        &self,
        _from_partition: PartitionIndex,
        _from_macro: MacroParameters,
        _to_partition: PartitionIndex,
        _to_macro: MacroParameters,
        _ks_parameters: KsDecompositionParameters,
    ) -> bool {
        true
    }

    fn is_available_micro_fks(
        &self,
        _from_partition: PartitionIndex,
        _from_macro: MacroParameters,
        _to_partition: PartitionIndex,
        _to_macro: MacroParameters,
        _ks_parameters: KsDecompositionParameters,
    ) -> bool {
        true
    }
}

/// An object restricting the search space based on smaller ranges.
#[derive(Serialize, Deserialize)]
pub struct RangeRestriction {
    pub glwe_log_polynomial_sizes: Vec<u64>,
    pub glwe_dimensions: Vec<u64>,
    pub internal_lwe_dimensions: Vec<u64>,
    pub pbs_level_count: Vec<u64>,
    pub pbs_base_log: Vec<u64>,
    pub ks_level_count: Vec<u64>,
    pub ks_base_log: Vec<u64>,
}

impl SearchSpaceRestriction for RangeRestriction {
    fn is_available_glwe(&self, _partition: PartitionIndex, glwe_params: GlweParameters) -> bool {
        (self.glwe_dimensions.is_empty()
            || self.glwe_dimensions.contains(&glwe_params.glwe_dimension))
            && (self.glwe_log_polynomial_sizes.is_empty()
                || self
                    .glwe_log_polynomial_sizes
                    .contains(&glwe_params.log2_polynomial_size))
    }

    fn is_available_macro(
        &self,
        _partition: PartitionIndex,
        macro_parameters: MacroParameters,
    ) -> bool {
        (self.glwe_dimensions.is_empty()
            || self
                .glwe_dimensions
                .contains(&macro_parameters.glwe_params.glwe_dimension))
            && (self.glwe_log_polynomial_sizes.is_empty()
                || self
                    .glwe_log_polynomial_sizes
                    .contains(&macro_parameters.glwe_params.log2_polynomial_size))
            && (self.internal_lwe_dimensions.is_empty()
                || self
                    .internal_lwe_dimensions
                    .contains(&macro_parameters.internal_dim))
    }

    fn is_available_micro_pbs(
        &self,
        _partition: PartitionIndex,
        macro_parameters: MacroParameters,
        pbs_parameters: BrDecompositionParameters,
    ) -> bool {
        (self.glwe_dimensions.is_empty()
            || self
                .glwe_dimensions
                .contains(&macro_parameters.glwe_params.glwe_dimension))
            && (self.glwe_log_polynomial_sizes.is_empty()
                || self
                    .glwe_log_polynomial_sizes
                    .contains(&macro_parameters.glwe_params.log2_polynomial_size))
            && (self.internal_lwe_dimensions.is_empty()
                || self
                    .internal_lwe_dimensions
                    .contains(&macro_parameters.internal_dim))
            && (self.pbs_base_log.is_empty()
                || self.pbs_base_log.contains(&pbs_parameters.log2_base))
            && (self.pbs_level_count.is_empty()
                || self.pbs_level_count.contains(&pbs_parameters.level))
    }

    fn is_available_micro_ks(
        &self,
        _from_partition: PartitionIndex,
        _from_macro: MacroParameters,
        _to_partition: PartitionIndex,
        _to_macro: MacroParameters,
        ks_parameters: KsDecompositionParameters,
    ) -> bool {
        (self.ks_base_log.is_empty() || self.ks_base_log.contains(&ks_parameters.log2_base))
            && (self.ks_level_count.is_empty()
                || self.ks_level_count.contains(&ks_parameters.level))
    }

    fn is_available_micro_fks(
        &self,
        _from_partition: PartitionIndex,
        _from_macro: MacroParameters,
        _to_partition: PartitionIndex,
        _to_macro: MacroParameters,
        ks_parameters: KsDecompositionParameters,
    ) -> bool {
        (self.ks_base_log.is_empty() || self.ks_base_log.contains(&ks_parameters.log2_base))
            && (self.ks_level_count.is_empty()
                || self.ks_level_count.contains(&ks_parameters.level))
    }
}

#[allow(unused)]
#[derive(Serialize, Deserialize)]
pub struct LweSecretKeyInfo {
    lwe_dimension: u64,
}

#[derive(Serialize, Deserialize)]
pub struct LweBootstrapKeyInfo {
    level_count: u64,
    base_log: u64,
    glwe_dimension: u64,
    polynomial_size: u64,
    input_lwe_dimension: u64,
}

#[derive(Serialize, Deserialize)]
pub struct LweKeyswitchKeyInfo {
    level_count: u64,
    base_log: u64,
    input_lwe_dimension: u64,
    output_lwe_dimension: u64,
}

#[allow(unused)]
#[derive(Serialize, Deserialize)]
pub struct KeysetInfo {
    lwe_secret_keys: Vec<LweSecretKeyInfo>,
    lwe_bootstrap_keys: Vec<LweBootstrapKeyInfo>,
    lwe_keyswitch_keys: Vec<LweKeyswitchKeyInfo>,
}

/// An object restricting the search space based on a keyset.
#[derive(Serialize, Deserialize)]
pub struct KeysetRestriction {
    info: KeysetInfo,
}

impl SearchSpaceRestriction for KeysetRestriction {
    fn is_available_glwe(&self, _partition: PartitionIndex, glwe_params: GlweParameters) -> bool {
        self.info.lwe_bootstrap_keys.iter().any(|k| {
            k.glwe_dimension == glwe_params.glwe_dimension
                && k.polynomial_size == 2_u64.pow(glwe_params.log2_polynomial_size as u32)
        })
    }

    fn is_available_macro(
        &self,
        _partition: PartitionIndex,
        macro_parameters: MacroParameters,
    ) -> bool {
        self.info.lwe_bootstrap_keys.iter().any(|k| {
            k.glwe_dimension == macro_parameters.glwe_params.glwe_dimension
                && k.polynomial_size
                    == 2_u64.pow(macro_parameters.glwe_params.log2_polynomial_size as u32)
                && k.input_lwe_dimension == macro_parameters.internal_dim
        })
    }

    fn is_available_micro_pbs(
        &self,
        _partition: PartitionIndex,
        macro_parameters: MacroParameters,
        pbs_parameters: BrDecompositionParameters,
    ) -> bool {
        self.info.lwe_bootstrap_keys.iter().any(|k| {
            k.glwe_dimension == macro_parameters.glwe_params.glwe_dimension
                && k.polynomial_size
                    == 2_u64.pow(macro_parameters.glwe_params.log2_polynomial_size as u32)
                && k.input_lwe_dimension == macro_parameters.internal_dim
                && k.level_count == pbs_parameters.level
                && k.base_log == pbs_parameters.log2_base
        })
    }

    fn is_available_micro_ks(
        &self,
        _from_partition: PartitionIndex,
        from_macro: MacroParameters,
        _to_partition: PartitionIndex,
        to_macro: MacroParameters,
        ks_parameters: KsDecompositionParameters,
    ) -> bool {
        self.info.lwe_keyswitch_keys.iter().any(|k| {
            k.input_lwe_dimension == from_macro.glwe_params.sample_extract_lwe_dimension()
                && k.output_lwe_dimension == to_macro.internal_dim
                && k.level_count == ks_parameters.level
                && k.base_log == ks_parameters.log2_base
        })
    }

    fn is_available_micro_fks(
        &self,
        _from_partition: PartitionIndex,
        from_macro: MacroParameters,
        _to_partition: PartitionIndex,
        to_macro: MacroParameters,
        ks_parameters: KsDecompositionParameters,
    ) -> bool {
        self.info.lwe_keyswitch_keys.iter().any(|k| {
            k.input_lwe_dimension == from_macro.glwe_params.sample_extract_lwe_dimension()
                && k.output_lwe_dimension == to_macro.glwe_params.sample_extract_lwe_dimension()
                && k.level_count == ks_parameters.level
                && k.base_log == ks_parameters.log2_base
        })
    }
}

/// An object restricting the search space for external partitions using partitioning informations.
pub struct ExternalPartitionRestriction(pub PartitionCut);

impl SearchSpaceRestriction for ExternalPartitionRestriction {
    fn is_available_glwe(&self, partition: PartitionIndex, glwe_params: GlweParameters) -> bool {
        !self.0.is_external_partition(&partition)
            || self.0.external_partitions[partition.0 - self.0.n_internal_partitions()]
                .macro_params
                .glwe_params
                == glwe_params
    }

    fn is_available_macro(
        &self,
        partition: PartitionIndex,
        macro_parameters: MacroParameters,
    ) -> bool {
        !self.0.is_external_partition(&partition)
            || self.0.external_partitions[partition.0 - self.0.n_internal_partitions()].macro_params
                == macro_parameters
    }

    fn is_available_micro_pbs(
        &self,
        partition: PartitionIndex,
        macro_parameters: MacroParameters,
        _pbs_parameters: BrDecompositionParameters,
    ) -> bool {
        !self.0.is_external_partition(&partition)
            || self.0.external_partitions[partition.0 - self.0.n_internal_partitions()].macro_params
                == macro_parameters
    }

    fn is_available_micro_ks(
        &self,
        from_partition: PartitionIndex,
        from_macro: MacroParameters,
        to_partition: PartitionIndex,
        to_macro: MacroParameters,
        _ks_parameters: KsDecompositionParameters,
    ) -> bool {
        (!self.0.is_external_partition(&from_partition)
            || self.0.external_partitions[from_partition.0 - self.0.n_internal_partitions()]
                .macro_params
                == from_macro)
            && (!self.0.is_external_partition(&to_partition)
                || self.0.external_partitions[to_partition.0 - self.0.n_internal_partitions()]
                    .macro_params
                    == to_macro)
    }

    fn is_available_micro_fks(
        &self,
        from_partition: PartitionIndex,
        from_macro: MacroParameters,
        to_partition: PartitionIndex,
        to_macro: MacroParameters,
        _ks_parameters: KsDecompositionParameters,
    ) -> bool {
        (!self.0.is_external_partition(&from_partition)
            || self.0.external_partitions[from_partition.0 - self.0.n_internal_partitions()]
                .macro_params
                == from_macro)
            && (!self.0.is_external_partition(&to_partition)
                || self.0.external_partitions[to_partition.0 - self.0.n_internal_partitions()]
                    .macro_params
                    == to_macro)
    }
}
