use concrete_shortint::parameters::{
    CarryModulus, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension,
    MessageModulus, Parameters, PolynomialSize, StandardDev,
};

use crate::errors::{UninitializedClientKey, UninitializedServerKey};
use crate::{ClientKey, GenericShortInt};

use super::{
    ShortIntegerClientKey, ShortIntegerParameter, ShortIntegerServerKey,
    StaticShortIntegerParameter,
};

use paste::paste;

/// Generic Parameter struct for short integers.
///
/// It allows to customize the same parameters as the ones
/// from the underlying `concrete_shortint` with the exception of
/// the number of bits of message as its embeded in the type.
#[derive(Copy, Clone, Debug)]
pub struct ShortIntegerParameterSet<const MESSAGE_BITS: u8> {
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
    pub carry_modulus: CarryModulus,
}

impl<const MESSAGE_BITS: u8> ShortIntegerParameterSet<MESSAGE_BITS> {
    const fn from_static(params: &'static Parameters) -> Self {
        if params.message_modulus.0 != 1 << MESSAGE_BITS as usize {
            panic!("Invalid bit number");
        }
        Self {
            lwe_dimension: params.lwe_dimension,
            glwe_dimension: params.glwe_dimension,
            polynomial_size: params.polynomial_size,
            lwe_modular_std_dev: params.lwe_modular_std_dev,
            glwe_modular_std_dev: params.glwe_modular_std_dev,
            pbs_base_log: params.pbs_base_log,
            pbs_level: params.pbs_level,
            ks_base_log: params.ks_base_log,
            ks_level: params.ks_level,
            pfks_level: params.pfks_level,
            pfks_base_log: params.pfks_base_log,
            pfks_modular_std_dev: params.pfks_modular_std_dev,
            cbs_level: params.cbs_level,
            cbs_base_log: params.cbs_base_log,
            carry_modulus: params.carry_modulus,
        }
    }
}

impl<const MESSAGE_BITS: u8> From<ShortIntegerParameterSet<MESSAGE_BITS>> for Parameters {
    fn from(params: ShortIntegerParameterSet<MESSAGE_BITS>) -> Self {
        Self {
            lwe_dimension: params.lwe_dimension,
            glwe_dimension: params.glwe_dimension,
            polynomial_size: params.polynomial_size,
            lwe_modular_std_dev: params.lwe_modular_std_dev,
            glwe_modular_std_dev: params.glwe_modular_std_dev,
            pbs_base_log: params.pbs_base_log,
            pbs_level: params.pbs_level,
            ks_base_log: params.ks_base_log,
            ks_level: params.ks_level,
            pfks_level: params.pfks_level,
            pfks_base_log: params.pfks_base_log,
            pfks_modular_std_dev: params.pfks_modular_std_dev,
            cbs_level: params.cbs_level,
            cbs_base_log: params.cbs_base_log,
            message_modulus: MessageModulus(1 << MESSAGE_BITS as usize),
            carry_modulus: params.carry_modulus,
        }
    }
}

/// The Id that is used to identify and retrieve the corresponding keys
#[derive(Copy, Clone, Default)]
pub struct ShorIntId<const MESSAGE_BITS: u8>;

impl<const MESSAGE_BITS: u8> ShortIntegerParameter for ShortIntegerParameterSet<MESSAGE_BITS> {
    type Id = ShorIntId<MESSAGE_BITS>;
}

impl<const MESSAGE_BITS: u8> StaticShortIntegerParameter
    for ShortIntegerParameterSet<MESSAGE_BITS>
{
    const MESSAGE_BITS: u8 = MESSAGE_BITS;
}

/// Defines a new static shortint type.
///
/// It needs as input the:
///     - name of the type
///     - the number of bits of message the type has
///     - the keychain member where ClientKey / Server Key is stored
///
/// It generates code:
///     - type alias for the client key, server key, parameter and shortint types
///     - the trait impl on the id type to access the keys
macro_rules! static_shortint_type {
    (
        $(#[$outer:meta])*
        $name:ident {
            num_bits: $num_bits:literal,
            keychain_member: $($member:ident).*,
        }
    ) => {
        paste! {

            #[doc = concat!("Parameters for the [", stringify!($name), "] data type.")]
            #[cfg_attr(doc, cfg(feature = "shortints"))]
            pub type [<$name Parameters>] = ShortIntegerParameterSet<$num_bits>;

            pub(in crate::shortints) type [<$name ClientKey>] = ShortIntegerClientKey<[<$name Parameters>]>;
            pub(in crate::shortints) type [<$name ServerKey>] = ShortIntegerServerKey<[<$name Parameters>]>;

            $(#[$outer])*
            #[doc=concat!("An unsigned integer type with ", stringify!($num_bits), " bits.")]
            #[cfg_attr(doc, cfg(feature = "shortints"))]
            pub type $name = GenericShortInt<[<$name Parameters>]>;

            impl_ref_key_from_keychain!(
                for <[<$name Parameters>] as ShortIntegerParameter>::Id {
                    key_type: [<$name ClientKey>],
                    keychain_member: $($member).*,
                    type_variant: crate::errors::Type::$name,
                }
            );

            impl_with_global_key!(
                for <[<$name Parameters>] as ShortIntegerParameter>::Id {
                    key_type: [<$name ServerKey>],
                    keychain_member: $($member).*,
                    type_variant: crate::errors::Type::$name,
                }
            );
        }
    };
}

static_shortint_type! {
    FheUint2 {
        num_bits: 2,
        keychain_member: shortint_key.uint2_key,
    }
}

static_shortint_type! {
    FheUint3 {
        num_bits: 3,
        keychain_member: shortint_key.uint3_key,
    }
}

static_shortint_type! {
    FheUint4 {
        num_bits: 4,
        keychain_member: shortint_key.uint4_key,
    }
}

impl FheUint2Parameters {
    // pub fn with_carry_0() -> Self {
    //     Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_0)
    // }

    pub fn with_carry_1() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_1)
    }

    pub fn with_carry_2() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2)
    }

    pub fn with_carry_3() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_3)
    }

    pub fn with_carry_4() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_4)
    }

    pub fn with_carry_5() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_5)
    }

    pub fn with_carry_6() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_6)
    }
}

impl Default for FheUint2Parameters {
    fn default() -> Self {
        Self::with_carry_2()
    }
}

impl FheUint3Parameters {
    // pub fn with_carry_0() -> Self {
    //     Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_0)
    // }

    pub fn with_carry_1() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_1)
    }

    pub fn with_carry_2() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_2)
    }

    pub fn with_carry_3() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_3)
    }

    pub fn with_carry_4() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_4)
    }

    pub fn with_carry_5() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_5)
    }
}

impl Default for FheUint3Parameters {
    fn default() -> Self {
        Self::with_carry_3()
    }
}

impl FheUint4Parameters {
    // pub fn with_carry_0() -> Self {
    //     Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_0)
    // }

    pub fn with_carry_1() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_1)
    }

    pub fn with_carry_2() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_2)
    }

    pub fn with_carry_3() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_3)
    }

    pub fn with_carry_4() -> Self {
        Self::from_static(&concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4)
    }
}

impl Default for FheUint4Parameters {
    fn default() -> Self {
        Self::with_carry_4()
    }
}
