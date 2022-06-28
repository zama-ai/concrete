use std::fmt::Debug;

use client_key::ShortIntegerClientKey;
use concrete_shortint::parameters::{
    CarryModulus, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension,
    MessageModulus, Parameters, PolynomialSize, StandardDev,
};
use server_key::ShortIntegerServerKey;
pub use types::GenericShortInt;

use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey};
use crate::ClientKey;

mod client_key;
mod server_key;
mod types;

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
    // Feels like this could/should be a const fn, but panics are not yet allowed
    // in const fn, and we need a check to make sure params has same message precision
    // as MESSAGE_BITS
    fn from_static(params: &'static Parameters) -> Self {
        assert_eq!(
            params.message_modulus.0,
            1 << MESSAGE_BITS as usize,
            "Incoherent bit size"
        );
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

pub trait ShortIntegerParameter: Copy + Into<Parameters> {
    const MESSAGE_BIT_SIZE: u8;
}

impl<const MESSAGE_BITS: u8> ShortIntegerParameter for ShortIntegerParameterSet<MESSAGE_BITS> {
    const MESSAGE_BIT_SIZE: u8 = MESSAGE_BITS;
}

/// Parameters for the [FheUint2] data type.
#[cfg_attr(doc, cfg(feature = "shortints"))]
pub type FheUint2Parameters = ShortIntegerParameterSet<2>;

/// Parameters for the [FheUint3] data type.
#[cfg_attr(doc, cfg(feature = "shortints"))]
pub type FheUint3Parameters = ShortIntegerParameterSet<3>;

/// Parameters for the [FheUint4] data type.
#[cfg_attr(doc, cfg(feature = "shortints"))]
pub type FheUint4Parameters = ShortIntegerParameterSet<4>;

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

macro_rules! static_shortint_type {
    (
        $(#[$outer:meta])*
        $name:ident {
            parameters: $parameters:ty,
            client_key_name: $client_key_name:ident,
            server_key_name: $server_key_name:ident,
            keychain_member: $($member:ident).*,
            type_variant: $enum_variant:expr,
        }
    ) => {
        pub(super) type $client_key_name = ShortIntegerClientKey<$parameters>;
        pub(super) type $server_key_name = ShortIntegerServerKey<$parameters>;

        $(#[$outer])*
        #[cfg_attr(doc, cfg(feature = "shortints"))]
        pub type $name = GenericShortInt<$parameters>;

        impl_ref_key_from_keychain!(
            for $client_key_name {
                keychain_member: $($member).*,
                type_variant: $enum_variant,
            }
        );

        impl_with_global_key!(
            for $server_key_name {
                keychain_member: $($member).*,
                type_variant: $enum_variant,
            }
        );
    };
}

static_shortint_type! {
    #[doc="An unsigned integer type with 2 bits."]
    FheUint2 {
        parameters: FheUint2Parameters,
        client_key_name: FheUint2ClientKey,
        server_key_name: FheUint2ServerKey,
        keychain_member: shortint_key.uint2_key,
        type_variant: Type::FheUint2,
    }
}

static_shortint_type! {
    #[doc="An unsigned integer type with 3 bits."]
    FheUint3 {
        parameters: FheUint3Parameters,
        client_key_name: FheUint3ClientKey,
        server_key_name: FheUint3ServerKey,
        keychain_member: shortint_key.uint3_key,
        type_variant: Type::FheUint3,
    }
}

static_shortint_type! {
    #[doc="An unsigned integer type with 4 bits."]
    FheUint4 {
        parameters: FheUint4Parameters,
        client_key_name: FheUint4ClientKey,
        server_key_name: FheUint4ServerKey,
        keychain_member: shortint_key.uint4_key,
        type_variant: Type::FheUint4,
    }
}
