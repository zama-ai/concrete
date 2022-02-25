//! The cryptographic parameter set.
//!
//! This module provides the structure containing the cryptographic parameters required for the
//! homomorphic evaluation of Boolean circuit as well as a list of secure cryptographic parameter
//! sets.
//!
//! Two parameter sets are provided:
//!  * `concrete_boolean::parameters::DEFAULT_PARAMETERS`
//!  * `concrete_boolean::parameters::TFHE_LIB_PARAMETERS`
//!
//! They ensure the correctness of the Boolean circuit evaluation result (up to a certain
//! probability) along with 128-bits of security.
//!
//! The two parameter sets offer a trade-off in terms of execution time versus error probability.
//! The `DEFAULT_PARAMETERS` set offers better performances on homomorphic circuit evaluation
//! with an higher probability error in comparison with the `TFHE_LIB_PARAMETERS`.
//! Note that if you desire, you can also create your own set of parameters.
//! This is an unsafe operation as failing to properly fix the parameters will potentially result
//! with an incorrect and/or insecure computation.
// TODO: speak about the lattice estimator and give the explicit used commit for the parameters

use concrete_core::prelude::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    StandardDev,
};
use serde::{Deserialize, Serialize};

/// A set of cryptographic parameters for homomorphic Boolean circuit evaluation.
#[derive(Serialize, Clone, Deserialize, Debug, PartialEq)]
pub struct BooleanParameters {
    pub(crate) lwe_dimension: LweDimension,
    pub(crate) glwe_dimension: GlweDimension,
    pub(crate) polynomial_size: PolynomialSize,
    pub(crate) lwe_modular_std_dev: StandardDev,
    pub(crate) glwe_modular_std_dev: StandardDev,
    pub(crate) pbs_base_log: DecompositionBaseLog,
    pub(crate) pbs_level: DecompositionLevelCount,
    pub(crate) ks_base_log: DecompositionBaseLog,
    pub(crate) ks_level: DecompositionLevelCount,
}

impl BooleanParameters {
    /// Constructs a new set of parameters for boolean circuit evaluation.
    ///
    /// # Safety
    ///
    /// This function is unsafe, as failing to fix the parameters properly would yield incorrect
    /// and insecure computation. Unless you are a cryptographer who really knows the impact of each
    /// of those parameters, you __must__ stick with the provided parameters [`DEFAULT_PARAMETERS`]
    /// and [`TFHE_LIB_PARAMETERS`], which both offer correct results with 128 bits of security.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_insecure(
        lwe_dimension: LweDimension,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
        lwe_modular_std_dev: StandardDev,
        glwe_modular_std_dev: StandardDev,
        pbs_base_log: DecompositionBaseLog,
        pbs_level: DecompositionLevelCount,
        ks_base_log: DecompositionBaseLog,
        ks_level: DecompositionLevelCount,
    ) -> BooleanParameters {
        BooleanParameters {
            lwe_dimension,
            glwe_dimension,
            polynomial_size,
            lwe_modular_std_dev,
            glwe_modular_std_dev,
            pbs_base_log,
            pbs_level,
            ks_level,
            ks_base_log,
        }
    }
}

/// Default parameter set.
///
/// This parameter set ensures 128-bits of security, and a probability of error is upper-bounded by
/// $2^{-25}$. The secret keys generated with this parameter set are uniform binary.
/// This parameter set allows to evaluate faster Boolean circuits than the `TFHE_LIB_PARAMETERS`
/// one.
pub const DEFAULT_PARAMETERS: BooleanParameters = BooleanParameters {
    lwe_dimension: LweDimension(586),
    glwe_dimension: GlweDimension(2),
    polynomial_size: PolynomialSize(512),
    lwe_modular_std_dev: StandardDev(0.000_089_761_673_968_349_98), // 2^{-13.44...}
    glwe_modular_std_dev: StandardDev(0.000_000_029_890_407_929_674_34), // 2^{-24.9...}
    pbs_base_log: DecompositionBaseLog(8),
    pbs_level: DecompositionLevelCount(2),
    ks_base_log: DecompositionBaseLog(2),
    ks_level: DecompositionLevelCount(5),
};

/// Parameter set used in [TFHE library](https://tfhe.github.io/tfhe/) for 128-bits of security.
///
/// Details about this set are provided
/// [here](https://github.com/tfhe/tfhe/blob/master/src/libtfhe/tfhe_gate_bootstrapping.cpp).
/// The secret keys generated with this parameter set are uniform binary.
/// This parameter set ensures a probability of error is upper-bounded by $2^{-165}$.
pub const TFHE_LIB_PARAMETERS: BooleanParameters = BooleanParameters {
    lwe_dimension: LweDimension(630),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(1024),
    lwe_modular_std_dev: StandardDev(0.000_030_517_578_125), // 2^{-15}
    glwe_modular_std_dev: StandardDev(0.000_000_029_802_322_387_695_313), // 2^{-25}
    pbs_base_log: DecompositionBaseLog(7),
    pbs_level: DecompositionLevelCount(3),
    ks_base_log: DecompositionBaseLog(2),
    ks_level: DecompositionLevelCount(8),
};
