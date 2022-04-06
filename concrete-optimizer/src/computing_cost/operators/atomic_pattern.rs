use super::super::complexity::Complexity;
use super::keyswitch_lwe::KeySwitchLWEComplexity;
use super::pbs::PbsComplexity;
use super::{keyswitch_lwe, pbs};
use crate::parameters::AtomicPatternParameters;

#[allow(clippy::too_many_arguments)]
pub trait AtomicPatternComplexity {
    fn complexity(
        &self,
        sum_size: u64,
        params: AtomicPatternParameters,
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
        params: AtomicPatternParameters,
        ciphertext_modulus_log: u64,
    ) -> Complexity {
        let multisum_complexity = (sum_size * params.input_lwe_dimension.0) as f64;
        let ks_complexity = self
            .ks_lwe
            .complexity(params.ks_parameters(), ciphertext_modulus_log);
        let pbs_complexity = self
            .pbs
            .complexity(params.pbs_parameters(), ciphertext_modulus_log);
        multisum_complexity + ks_complexity + pbs_complexity
    }
}

pub type Default = KsPbs<keyswitch_lwe::Default, pbs::Default>;
pub const DEFAULT: Default = KsPbs {
    ks_lwe: keyswitch_lwe::DEFAULT,
    pbs: pbs::DEFAULT,
};
