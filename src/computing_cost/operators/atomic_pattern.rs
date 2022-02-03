use super::super::complexity::Complexity;
use super::keyswitch_lwe::KeySwitchLWEComplexity;
use super::pbs::PbsComplexity;
use super::{keyswitch_lwe, pbs};

#[allow(clippy::too_many_arguments)]
pub trait AtomicPatternComplexity {
    fn complexity(
        &self,
        sum_size: u64,
        input_lwe_dimension: u64,              //n_big
        internal_ks_output_lwe_dimension: u64, //n_small
        ks_decomposition_level_count: u64,     //l(BS)
        ks_decomposition_base_log: u64,        //b(BS)
        glwe_polynomial_size: u64,             //N
        glwe_dimension: u64,                   //k
        br_decomposition_level_count: u64,     //l(KS)
        br_decomposition_base_log: u64,        //b(ks)
        ciphertext_modulus_log: u64,
    ) -> Complexity;
}

pub struct KsPbs<KS: KeySwitchLWEComplexity, PBS: PbsComplexity> {
    pub ks_lwe: KS,
    pub pbs: PBS,
}

impl<KS, PBS> AtomicPatternComplexity for KsPbs<KS, PBS>
where
    KS: KeySwitchLWEComplexity,
    PBS: PbsComplexity,
{
    fn complexity(
        &self,
        sum_size: u64,
        input_lwe_dimension: u64,              //n_big
        internal_ks_output_lwe_dimension: u64, //n_small
        ks_decomposition_level_count: u64,     //l(KS)
        ks_decomposition_base_log: u64,        //b(KS) // not used
        glwe_polynomial_size: u64,             //N
        glwe_dimension: u64,                   //k
        br_decomposition_level_count: u64,     //l(BR) // not used
        br_decomposition_base_log: u64,        //b(BR)
        ciphertext_modulus_log: u64,
    ) -> Complexity {
        let multisum_complexity = (sum_size * input_lwe_dimension) as f64;
        let ks_complexity = {
            self.ks_lwe.complexity(
                input_lwe_dimension,
                internal_ks_output_lwe_dimension,
                ks_decomposition_level_count,
                ks_decomposition_base_log,
                ciphertext_modulus_log,
            )
        };
        let pbs_complexity = self.pbs.complexity(
            internal_ks_output_lwe_dimension,
            glwe_polynomial_size,
            glwe_dimension,
            br_decomposition_level_count,
            br_decomposition_base_log,
            ciphertext_modulus_log,
        );
        multisum_complexity + ks_complexity + pbs_complexity
    }
}

pub type Default = KsPbs<keyswitch_lwe::Default, pbs::Default>;
pub const DEFAULT: Default = KsPbs {
    ks_lwe: keyswitch_lwe::DEFAULT,
    pbs: pbs::DEFAULT,
};
