#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::errors::{UninitializedClientKey, UninitializedServerKey};
use crate::integers::client_key::GenericIntegerClientKey;
use crate::integers::parameters::{
    IntegerParameter, IntegerParameterSet, RadixParameters, StaticIntegerParameter,
};
use crate::integers::server_key::GenericIntegerServerKey;
use crate::keys::RefKeyFromKeyChain;
use crate::traits::{FheDecrypt, FheEncrypt};
use crate::{ClientKey, FheUint2Parameters};

use super::base::GenericInteger;
use paste::paste;

macro_rules! define_static_integer_parameters {
    (
        num_bits: $num_bits:literal,
        block_parameters: $block_parameters:expr,
        num_block: $num_block:literal,
    ) => {
        paste! {

            #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
            #[derive(Copy, Clone, Debug, Default)]
            pub struct [<FheUint $num_bits Id>];

            #[derive(Copy, Clone, Debug)]
            pub struct [<FheUint $num_bits Parameters>](RadixParameters);

            impl From<[<FheUint $num_bits Parameters>]> for IntegerParameterSet {
                fn from(v: [<FheUint $num_bits Parameters>]) -> IntegerParameterSet {
                    Self::from(v.0)
                }
            }

            impl Default for [<FheUint $num_bits Parameters>] {
                fn default() -> Self {
                    Self(
                        RadixParameters {
                            block_parameters: $block_parameters,
                            num_block: $num_block,
                        },
                    )
                }
            }

            impl IntegerParameter for [<FheUint $num_bits Parameters>] {
                type Id = [<FheUint $num_bits Id>];
            }

            impl StaticIntegerParameter for [<FheUint $num_bits Parameters>] {
                const MESSAGE_BITS: usize = $num_bits;
            }
        }
    };
}

macro_rules! static_int_type {
    (
        $(#[$outer:meta])*
        $name:ident {
            num_bits: $num_bits:literal,
            keychain_member: $($member:ident).*,
            parameters: {
                block_parameters: $block_parameters:expr,
                num_block: $num_block:literal,
            },
        }
    ) => {

        define_static_integer_parameters!(
            num_bits: $num_bits,
            block_parameters: $block_parameters,
            num_block: $num_block,
        );


        paste! {
            pub(in crate::integers) type [<$name ClientKey>] = GenericIntegerClientKey<[<$name Parameters>]>;
            pub(in crate::integers) type [<$name ServerKey>] = GenericIntegerServerKey<[<$name Parameters>]>;

            $(#[$outer])*
            #[cfg_attr(doc, cfg(feature = "integers"))]
            pub type $name = GenericInteger<[<$name Parameters>]>;

            impl_ref_key_from_keychain!(
                for <[<$name Parameters>] as IntegerParameter>::Id {
                    key_type: [<$name ClientKey>],
                    keychain_member: $($member).*,
                    type_variant: crate::errors::Type::$name,
                }
            );

            impl_with_global_key!(
                for <[<$name Parameters>] as IntegerParameter>::Id {
                    key_type: [<$name ServerKey>],
                    keychain_member: $($member).*,
                    type_variant: crate::errors::Type::$name,
                }
            );
        }
    };
}

static_int_type! {
    #[doc="An unsigned integer type with 8 bits."]
    FheUint8 {
        num_bits: 8,
        keychain_member: integer_key.uint8_key,
        parameters: {
            block_parameters: FheUint2Parameters::with_carry_2().into(),
            num_block: 4,
        },
    }
}

static_int_type! {
    #[doc="An unsigned integer type with 12 bits."]
    FheUint12 {
        num_bits: 12,
        keychain_member: integer_key.uint12_key,
        parameters: {
            block_parameters: FheUint2Parameters::with_carry_2().into(),
            num_block: 6,
        },
    }
}

static_int_type! {
    #[doc="An unsigned integer type with 16 bits."]
    FheUint16 {
        num_bits: 16,
        keychain_member: integer_key.uint16_key,
        parameters: {
            block_parameters: FheUint2Parameters::with_carry_2().into(),
            num_block: 8,
        },
    }
}

impl FheEncrypt<u8> for GenericInteger<FheUint8Parameters> {
    #[track_caller]
    fn encrypt(value: u8, key: &ClientKey) -> Self {
        let id = <FheUint8Parameters as IntegerParameter>::Id::default();
        let key = id.unwrapped_ref_key(key);
        let ciphertext = key.key.encrypt(u64::from(value));
        Self::new(ciphertext, id)
    }
}

impl FheDecrypt<u8> for FheUint8 {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> u8 {
        let id = <FheUint8Parameters as IntegerParameter>::Id::default();
        let key = id.unwrapped_ref_key(key);
        key.key.decrypt(&*self.ciphertext.borrow()) as u8
    }
}

impl FheEncrypt<u16> for FheUint16 {
    #[track_caller]
    fn encrypt(value: u16, key: &ClientKey) -> Self {
        let id = <FheUint16Parameters as IntegerParameter>::Id::default();
        let key = id.unwrapped_ref_key(key);
        let ciphertext = key.key.encrypt(u64::from(value));
        Self::new(ciphertext, id)
    }
}

impl FheDecrypt<u16> for FheUint16 {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> u16 {
        let id = <FheUint16Parameters as IntegerParameter>::Id::default();
        let key = id.unwrapped_ref_key(key);
        key.key.decrypt(&*self.ciphertext.borrow()) as u16
    }
}
