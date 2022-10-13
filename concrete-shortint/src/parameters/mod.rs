//! Module with the definition of parameters for short-integers.
//!
//! This module provides the structure containing the cryptographic parameters required for the
//! homomorphic evaluation of integer circuits as well as a list of secure cryptographic parameter
//! sets.

pub use concrete_core::prelude::{
    DecompositionBaseLog, DecompositionLevelCount, DispersionParameter, GlweDimension,
    LweDimension, PolynomialSize, StandardDev,
};
use serde::{Deserialize, Serialize};

pub mod parameters_wopbs;
pub mod parameters_wopbs_message_carry;
pub(crate) mod parameters_wopbs_prime_moduli;

/// The number of bits on which the message will be encoded.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub struct MessageModulus(pub usize);

/// The number of bits on which the carry will be encoded.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub struct CarryModulus(pub usize);

/// A structure defining the set of cryptographic parameters for homomorphic integer circuit
/// evaluation.
#[derive(Serialize, Copy, Clone, Deserialize, Debug, PartialEq)]
pub struct Parameters {
    pub lwe_dimension: LweDimension,
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub lwe_modular_std_dev: StandardDev,
    pub glwe_modular_std_dev: StandardDev,
    pub pbs_base_log: DecompositionBaseLog,
    pub pbs_level: DecompositionLevelCount,
    pub ks_base_log: DecompositionBaseLog,
    pub ks_level: DecompositionLevelCount,
    pub pfks_level: DecompositionLevelCount,
    pub pfks_base_log: DecompositionBaseLog,
    pub pfks_modular_std_dev: StandardDev,
    pub cbs_level: DecompositionLevelCount,
    pub cbs_base_log: DecompositionBaseLog,
    pub message_modulus: MessageModulus,
    pub carry_modulus: CarryModulus,
}

impl Parameters {
    /// Constructs a new set of parameters for integer circuit evaluation.
    ///
    /// # Safety
    ///
    /// This function is unsafe, as failing to fix the parameters properly would yield incorrect
    /// and unsecure computation. Unless you are a cryptographer who really knows the impact of each
    /// of those parameters, you __must__ stick with the provided parameters.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_unsecure(
        lwe_dimension: LweDimension,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
        lwe_modular_std_dev: StandardDev,
        glwe_modular_std_dev: StandardDev,
        pbs_base_log: DecompositionBaseLog,
        pbs_level: DecompositionLevelCount,
        ks_base_log: DecompositionBaseLog,
        ks_level: DecompositionLevelCount,
        pfks_level: DecompositionLevelCount,
        pfks_base_log: DecompositionBaseLog,
        pfks_modular_std_dev: StandardDev,
        cbs_level: DecompositionLevelCount,
        cbs_base_log: DecompositionBaseLog,
        message_modulus: MessageModulus,
        carry_modulus: CarryModulus,
    ) -> Parameters {
        Parameters {
            lwe_dimension,
            glwe_dimension,
            polynomial_size,
            lwe_modular_std_dev,
            glwe_modular_std_dev,
            pbs_base_log,
            pbs_level,
            ks_level,
            ks_base_log,
            pfks_level,
            pfks_base_log,
            pfks_modular_std_dev,
            cbs_level,
            cbs_base_log,
            message_modulus,
            carry_modulus,
        }
    }
}

impl Default for Parameters {
    fn default() -> Self {
        DEFAULT_PARAMETERS
    }
}

/// Vector containing all parameter sets
pub const ALL_PARAMETER_VEC: [Parameters; 28] = WITH_CARRY_PARAMETERS_VEC;

/// Vector containing all parameter sets where the carry space is strictly greater than one
pub const WITH_CARRY_PARAMETERS_VEC: [Parameters; 28] = [
    PARAM_MESSAGE_1_CARRY_1,
    PARAM_MESSAGE_1_CARRY_2,
    PARAM_MESSAGE_1_CARRY_3,
    PARAM_MESSAGE_1_CARRY_4,
    PARAM_MESSAGE_1_CARRY_5,
    PARAM_MESSAGE_1_CARRY_6,
    PARAM_MESSAGE_1_CARRY_7,
    PARAM_MESSAGE_2_CARRY_1,
    PARAM_MESSAGE_2_CARRY_2,
    PARAM_MESSAGE_2_CARRY_3,
    PARAM_MESSAGE_2_CARRY_4,
    PARAM_MESSAGE_2_CARRY_5,
    PARAM_MESSAGE_2_CARRY_6,
    PARAM_MESSAGE_3_CARRY_1,
    PARAM_MESSAGE_3_CARRY_2,
    PARAM_MESSAGE_3_CARRY_3,
    PARAM_MESSAGE_3_CARRY_4,
    PARAM_MESSAGE_3_CARRY_5,
    PARAM_MESSAGE_4_CARRY_1,
    PARAM_MESSAGE_4_CARRY_2,
    PARAM_MESSAGE_4_CARRY_3,
    PARAM_MESSAGE_4_CARRY_4,
    PARAM_MESSAGE_5_CARRY_1,
    PARAM_MESSAGE_5_CARRY_2,
    PARAM_MESSAGE_5_CARRY_3,
    PARAM_MESSAGE_6_CARRY_1,
    PARAM_MESSAGE_6_CARRY_2,
    PARAM_MESSAGE_7_CARRY_1,
];

/// Vector containing all parameter sets where the carry space is strictly greater than one
pub const BIVARIATE_PBS_COMPLIANT_PARAMETER_SET_VEC: [Parameters; 16] = [
    PARAM_MESSAGE_1_CARRY_1,
    PARAM_MESSAGE_1_CARRY_2,
    PARAM_MESSAGE_1_CARRY_3,
    PARAM_MESSAGE_1_CARRY_4,
    PARAM_MESSAGE_1_CARRY_5,
    PARAM_MESSAGE_1_CARRY_6,
    PARAM_MESSAGE_1_CARRY_7,
    PARAM_MESSAGE_2_CARRY_2,
    PARAM_MESSAGE_2_CARRY_3,
    PARAM_MESSAGE_2_CARRY_4,
    PARAM_MESSAGE_2_CARRY_5,
    PARAM_MESSAGE_2_CARRY_6,
    PARAM_MESSAGE_3_CARRY_3,
    PARAM_MESSAGE_3_CARRY_4,
    PARAM_MESSAGE_3_CARRY_5,
    PARAM_MESSAGE_4_CARRY_4,
];

/// Default parameter set
pub const DEFAULT_PARAMETERS: Parameters = PARAM_MESSAGE_2_CARRY_2;

/// Nomenclature: PARAM_MESSAGE_X_CARRY_Y: the message (respectively carry) modulus is
/// encoded over X (reps. Y) bits, i.e., message_modulus = 2^{X} (resp. carry_modulus = 2^{Y}).
pub const PARAM_MESSAGE_1_CARRY_1: Parameters = Parameters {
    lwe_dimension: LweDimension(585),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(1024),
    lwe_modular_std_dev: StandardDev(0.000091417760042025734377),
    glwe_modular_std_dev: StandardDev(0.0000000298904079296743355770570),
    pbs_base_log: DecompositionBaseLog(8),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(2),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(8),
    pfks_modular_std_dev: StandardDev(0.0000000298904079296743355770570),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(8),
    message_modulus: MessageModulus(2),
    carry_modulus: CarryModulus(2),
};

pub const PARAM_MESSAGE_1_CARRY_2: Parameters = Parameters {
    lwe_dimension: LweDimension(679),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_modular_std_dev: StandardDev(0.0000163947616287537526368),
    glwe_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(2),
    carry_modulus: CarryModulus(4),
};
pub const PARAM_MESSAGE_2_CARRY_1: Parameters = Parameters {
    lwe_dimension: LweDimension(650),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(1024),
    lwe_modular_std_dev: StandardDev(0.0000278582899041329652826),
    glwe_modular_std_dev: StandardDev(0.0000000298904079296743355770570),
    pbs_base_log: DecompositionBaseLog(6),
    pbs_level: DecompositionLevelCount(3),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(2),
    pfks_level: DecompositionLevelCount(3),
    pfks_base_log: DecompositionBaseLog(6),
    pfks_modular_std_dev: StandardDev(0.0000000298904079296743355770570),
    cbs_level: DecompositionLevelCount(3),
    cbs_base_log: DecompositionBaseLog(6),
    message_modulus: MessageModulus(4),
    carry_modulus: CarryModulus(2),
};
pub const PARAM_MESSAGE_1_CARRY_3: Parameters = Parameters {
    lwe_dimension: LweDimension(721),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_modular_std_dev: StandardDev(0.00000760747502572033357575),
    glwe_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(2),
    carry_modulus: CarryModulus(8),
};
pub const PARAM_MESSAGE_2_CARRY_2: Parameters = Parameters {
    lwe_dimension: LweDimension(720),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_modular_std_dev: StandardDev(0.00000774783151517677815848),
    glwe_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(0),
    pfks_base_log: DecompositionBaseLog(0),
    pfks_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    cbs_level: DecompositionLevelCount(0),
    cbs_base_log: DecompositionBaseLog(0),
    message_modulus: MessageModulus(4),
    carry_modulus: CarryModulus(4),
};
pub const PARAM_MESSAGE_3_CARRY_1: Parameters = Parameters {
    lwe_dimension: LweDimension(720),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_modular_std_dev: StandardDev(0.00000774783151517677815848),
    glwe_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(8),
    carry_modulus: CarryModulus(2),
};
pub const PARAM_MESSAGE_1_CARRY_4: Parameters = Parameters {
    lwe_dimension: LweDimension(806),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_modular_std_dev: StandardDev(0.0000016083173981113161172144),
    glwe_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(2),
    carry_modulus: CarryModulus(16),
};
pub const PARAM_MESSAGE_2_CARRY_3: Parameters = Parameters {
    lwe_dimension: LweDimension(776),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_modular_std_dev: StandardDev(0.00000278330461375383057442),
    glwe_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(4),
    carry_modulus: CarryModulus(8),
};
pub const PARAM_MESSAGE_3_CARRY_2: Parameters = Parameters {
    lwe_dimension: LweDimension(774),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_modular_std_dev: StandardDev(0.000002886954936071319246944),
    glwe_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(8),
    carry_modulus: CarryModulus(4),
};
pub const PARAM_MESSAGE_4_CARRY_1: Parameters = Parameters {
    lwe_dimension: LweDimension(774),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_modular_std_dev: StandardDev(0.000002886954936071319246944),
    glwe_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.00000000000000022148688116005568513645324585951),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(16),
    carry_modulus: CarryModulus(2),
};
pub const PARAM_MESSAGE_1_CARRY_5: Parameters = Parameters {
    lwe_dimension: LweDimension(833),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(4096),
    lwe_modular_std_dev: StandardDev(0.0000009817517371869340369524),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(2),
    carry_modulus: CarryModulus(32),
};
pub const PARAM_MESSAGE_2_CARRY_4: Parameters = Parameters {
    lwe_dimension: LweDimension(833),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(4096),
    lwe_modular_std_dev: StandardDev(0.0000009817517371869340369524),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(4),
    carry_modulus: CarryModulus(16),
};
pub const PARAM_MESSAGE_3_CARRY_3: Parameters = Parameters {
    lwe_dimension: LweDimension(829),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(4096),
    lwe_modular_std_dev: StandardDev(0.0000010562341599676662606703),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(11),
    ks_base_log: DecompositionBaseLog(2),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(8),
    carry_modulus: CarryModulus(8),
};
pub const PARAM_MESSAGE_4_CARRY_2: Parameters = Parameters {
    lwe_dimension: LweDimension(839),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(4096),
    lwe_modular_std_dev: StandardDev(0.0000008797593324565154619988),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(16),
    carry_modulus: CarryModulus(4),
};
pub const PARAM_MESSAGE_5_CARRY_1: Parameters = Parameters {
    lwe_dimension: LweDimension(834),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(4096),
    lwe_modular_std_dev: StandardDev(0.0000009639667315270401488172),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(32),
    carry_modulus: CarryModulus(2),
};
pub const PARAM_MESSAGE_1_CARRY_6: Parameters = Parameters {
    lwe_dimension: LweDimension(895),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(8192),
    lwe_modular_std_dev: StandardDev(0.00000031604174719374318760879),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(2),
    carry_modulus: CarryModulus(64),
};
pub const PARAM_MESSAGE_2_CARRY_5: Parameters = Parameters {
    lwe_dimension: LweDimension(892),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(8192),
    lwe_modular_std_dev: StandardDev(0.00000033385920279766705870777),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(4),
    carry_modulus: CarryModulus(32),
};
pub const PARAM_MESSAGE_3_CARRY_4: Parameters = Parameters {
    lwe_dimension: LweDimension(892),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(8192),
    lwe_modular_std_dev: StandardDev(0.00000033385920279766705870777),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(8),
    carry_modulus: CarryModulus(16),
};
pub const PARAM_MESSAGE_4_CARRY_3: Parameters = Parameters {
    lwe_dimension: LweDimension(892),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(8192),
    lwe_modular_std_dev: StandardDev(0.00000033385920279766705870777),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(16),
    carry_modulus: CarryModulus(8),
};
pub const PARAM_MESSAGE_5_CARRY_2: Parameters = Parameters {
    lwe_dimension: LweDimension(892),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(8192),
    lwe_modular_std_dev: StandardDev(0.00000033385920279766705870777),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(32),
    carry_modulus: CarryModulus(4),
};
pub const PARAM_MESSAGE_6_CARRY_1: Parameters = Parameters {
    lwe_dimension: LweDimension(950),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(8192),
    lwe_modular_std_dev: StandardDev(0.000000115628437836360299069429),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(1),
    pfks_base_log: DecompositionBaseLog(23),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(1),
    cbs_base_log: DecompositionBaseLog(23),
    message_modulus: MessageModulus(64),
    carry_modulus: CarryModulus(2),
};
pub const PARAM_MESSAGE_1_CARRY_7: Parameters = Parameters {
    lwe_dimension: LweDimension(927),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(16384),
    lwe_modular_std_dev: StandardDev(0.000000176066288458559164346451),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(12),
    pbs_level: DecompositionLevelCount(3),
    ks_level: DecompositionLevelCount(7),
    ks_base_log: DecompositionBaseLog(3),
    pfks_level: DecompositionLevelCount(3),
    pfks_base_log: DecompositionBaseLog(12),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(3),
    cbs_base_log: DecompositionBaseLog(12),
    message_modulus: MessageModulus(2),
    carry_modulus: CarryModulus(128),
};
pub const PARAM_MESSAGE_2_CARRY_6: Parameters = Parameters {
    lwe_dimension: LweDimension(985),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(16384),
    lwe_modular_std_dev: StandardDev(0.000000060978609304495400241767),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(7),
    ks_base_log: DecompositionBaseLog(3),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(4),
    carry_modulus: CarryModulus(64),
};
pub const PARAM_MESSAGE_3_CARRY_5: Parameters = Parameters {
    lwe_dimension: LweDimension(957),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(16384),
    lwe_modular_std_dev: StandardDev(0.000000101738944975511049379371),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(8),
    carry_modulus: CarryModulus(32),
};
pub const PARAM_MESSAGE_4_CARRY_4: Parameters = Parameters {
    lwe_dimension: LweDimension(953),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(16384),
    lwe_modular_std_dev: StandardDev(0.000000109457559392883651121995),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(16),
    carry_modulus: CarryModulus(16),
};
pub const PARAM_MESSAGE_5_CARRY_3: Parameters = Parameters {
    lwe_dimension: LweDimension(953),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(16384),
    lwe_modular_std_dev: StandardDev(0.000000109457559392883651121995),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(32),
    carry_modulus: CarryModulus(8),
};
pub const PARAM_MESSAGE_6_CARRY_2: Parameters = Parameters {
    lwe_dimension: LweDimension(952),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(16384),
    lwe_modular_std_dev: StandardDev(0.000000111477030863892616504447),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(64),
    carry_modulus: CarryModulus(4),
};
pub const PARAM_MESSAGE_7_CARRY_1: Parameters = Parameters {
    lwe_dimension: LweDimension(952),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(16384),
    lwe_modular_std_dev: StandardDev(0.000000111477030863892616504447),
    glwe_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    pbs_base_log: DecompositionBaseLog(15),
    pbs_level: DecompositionLevelCount(2),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(4),
    pfks_level: DecompositionLevelCount(2),
    pfks_base_log: DecompositionBaseLog(15),
    pfks_modular_std_dev: StandardDev(0.0000000000000000002168404344971008860665043427092021),
    cbs_level: DecompositionLevelCount(2),
    cbs_base_log: DecompositionBaseLog(15),
    message_modulus: MessageModulus(128),
    carry_modulus: CarryModulus(2),
};

/// Return a parameter set from a message and carry moduli.
///
/// # Example
///
/// ```rust
/// use concrete_shortint::parameters::{
///     get_parameters_from_message_and_carry, PARAM_MESSAGE_3_CARRY_1,
/// };
/// let message_space = 7;
/// let carry_space = 2;
/// let param = get_parameters_from_message_and_carry(message_space, carry_space);
/// assert_eq!(param, PARAM_MESSAGE_3_CARRY_1);
/// ```
pub fn get_parameters_from_message_and_carry(msg_space: usize, carry_space: usize) -> Parameters {
    let mut out = Parameters::default();
    let mut flag: bool = false;
    let mut rescaled_message_space = f64::ceil(f64::log2(msg_space as f64)) as usize;
    rescaled_message_space = 1 << rescaled_message_space;
    let mut rescaled_carry_space = f64::ceil(f64::log2(carry_space as f64)) as usize;
    rescaled_carry_space = 1 << rescaled_carry_space;

    for param in ALL_PARAMETER_VEC {
        if param.message_modulus.0 == rescaled_message_space
            && param.carry_modulus.0 == rescaled_carry_space
        {
            out = param;
            flag = true;
            break;
        }
    }
    if !flag {
        println!(
            "### WARNING: NO PARAMETERS FOUND for msg_space = {} and carry_space = {} ### ",
            rescaled_message_space, rescaled_carry_space
        );
    }
    out
}
