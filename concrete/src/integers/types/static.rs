#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::errors::{UninitializedClientKey, UninitializedServerKey};
use crate::integers::client_key::GenericIntegerClientKey;
use crate::integers::parameters::{
    EvaluationIntegerKey, IntegerParameter, RadixParameters, RadixRepresentation,
    StaticIntegerParameter, StaticRadixParameter,
};
use crate::integers::server_key::GenericIntegerServerKey;
use crate::keys::RefKeyFromKeyChain;
use crate::traits::{FheDecrypt, FheEncrypt};
use crate::ClientKey;

use super::base::GenericInteger;
#[cfg(feature = "internal-keycache")]
use concrete_integer::keycache::{KEY_CACHE, KEY_CACHE_WOPBS};
use concrete_integer::wopbs::WopbsKey;
use paste::paste;

macro_rules! define_static_integer_parameters {
    (
        Radix {
            num_bits: $num_bits:literal,
            block_parameters: $block_parameters:expr,
            num_block: $num_block:literal,
        }
    ) => {
        paste! {
            #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
            #[derive(Copy, Clone, Debug, Default)]
            pub struct [<FheUint $num_bits Id>];

            #[derive(Copy, Clone, Debug)]
            pub struct [<FheUint $num_bits Parameters>](RadixParameters);

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
                type InnerCiphertext = concrete_integer::RadixCiphertext;
                type InnerClientKey = concrete_integer::RadixClientKey;
                type InnerServerKey = concrete_integer::ServerKey;
            }

            impl From<[<FheUint $num_bits Parameters>]> for RadixParameters {
                fn from(p: [<FheUint $num_bits Parameters>]) -> Self {
                    p.0
                }
            }

            impl StaticIntegerParameter for [<FheUint $num_bits Parameters>] {
                type Representation = RadixRepresentation;
                const MESSAGE_BITS: usize = $num_bits;
            }

            impl StaticRadixParameter for [<FheUint $num_bits Parameters>] {}
        }
    };
    (
        Crt {
            num_bits: $num_bits:literal,
            block_parameters: $block_parameters:expr,
            moduli: $moduli:expr,
        }
    ) => {
        paste! {
            #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
            #[derive(Copy, Clone, Debug, Default)]
            pub struct [<FheUint $num_bits Id>];

            #[derive(Copy, Clone, Debug)]
            pub struct [<FheUint $num_bits Parameters>](CrtParameters);

            impl Default for [<FheUint $num_bits Parameters>] {
                fn default() -> Self {
                    Self(
                        CrtParameters {
                            block_parameters: $block_parameters,
                            moduli: $moduli,
                        },
                    )
                }
            }

            impl IntegerParameter for [<FheUint $num_bits Parameters>] {
                type Id = [<FheUint $num_bits Id>];
                type InnerCiphertext = concrete_integer::CrtCiphertext;
                type InnerClientKey = concrete_integer::CrtClientKey;
                type InnerServerKey = concrete_integer::ServerKey;
            }

            impl From<[<FheUint $num_bits Parameters>]> for CrtCiphertext {
                fn from(p: [<FheUint $num_bits Parameters>]) -> Self {
                    p.0
                }
            }

            impl StaticIntegerParameter for [<FheUint $num_bits Parameters>] {
                type Representation = CrtRepresentation;
                const MESSAGE_BITS: usize = $num_bits;
            }

            impl StaticCrtParameter for [<FheUint $num_bits Parameters>] {}
        }
    };
}

macro_rules! static_int_type {
    // This rule generates the types specialization
    // as well as call the macros
    // that implement necessary traits for the ClientKey and ServerKey
    (
        @impl_types_and_key_traits,
        $(#[$outer:meta])*
        $name:ident {
            num_bits: $num_bits:literal,
            keychain_member: $($member:ident).*,
        }
    ) => {
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

    // For Radix parameters
    (
        $(#[$outer:meta])*
        $name:ident {
            num_bits: $num_bits:literal,
            keychain_member: $($member:ident).*,
            parameters: Radix {
                block_parameters: $block_parameters:expr,
                num_block: $num_block:literal,
            },
        }
    ) => {
        define_static_integer_parameters!(
            Radix {
                num_bits: $num_bits,
                block_parameters: $block_parameters,
                num_block: $num_block,
            }
        );

        static_int_type!(
            @impl_types_and_key_traits,
            $(#[$outer])*
            $name {
                num_bits: $num_bits,
                keychain_member: $($member).*,
            }
        );
    };
    // For Crt parameters
    (
        $(#[$outer:meta])*
        $name:ident {
            num_bits: $num_bits:literal,
            keychain_member: $($member:ident).*,
            parameters: Crt {
                block_parameters: $block_parameters:expr,
                moduli: $moduli:expr,
            },
        }
    ) => {
        define_static_integer_parameters!(
            Crt {
                num_bits: $num_bits,
                block_parameters: $block_parameters,
                moduli: $moduli,
            }
        );

        static_int_type!(
            @impl_types_and_key_traits,
            $(#[$outer])*
            $name {
                num_bits: $num_bits,
                keychain_member: $($member).*,
            }
        );
    };
}

impl<C> EvaluationIntegerKey<C> for concrete_integer::ServerKey
where
    C: AsRef<concrete_integer::ClientKey>,
{
    fn new(client_key: &C) -> Self {
        #[cfg(feature = "internal-keycache")]
        {
            KEY_CACHE
                .get_from_params(client_key.as_ref().parameters())
                .1
        }
        #[cfg(not(feature = "internal-keycache"))]
        {
            concrete_integer::ServerKey::new(client_key)
        }
    }

    fn new_wopbs_key(client_key: &C, server_key: &Self) -> WopbsKey {
        #[cfg(not(feature = "internal-keycache"))]
        {
            WopbsKey::new_wopbs_key(client_key.as_ref(), server_key)
        }
        #[cfg(feature = "internal-keycache")]
        {
            let _ = &server_key; // silence warning
            KEY_CACHE_WOPBS
                .get_from_params(client_key.as_ref().parameters())
                
        }
    }
}

static_int_type! {
    #[doc="An unsigned integer type with 8 bits."]
    FheUint8 {
        num_bits: 8,
        keychain_member: integer_key.uint8_key,
        parameters: Radix {
            block_parameters: concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2,
            num_block: 4,
        },
    }
}

static_int_type! {
    #[doc="An unsigned integer type with 12 bits."]
    FheUint12 {
        num_bits: 12,
        keychain_member: integer_key.uint12_key,
        parameters: Radix {
            block_parameters: concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2,
            num_block: 6,
        },
    }
}

static_int_type! {
    #[doc="An unsigned integer type with 16 bits."]
    FheUint16 {
        num_bits: 16,
        keychain_member: integer_key.uint16_key,
        parameters: Radix {
            block_parameters: concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2,
            num_block: 8,
        },
    }
}

impl FheEncrypt<u8> for GenericInteger<FheUint8Parameters> {
    #[track_caller]
    fn encrypt(value: u8, key: &ClientKey) -> Self {
        let id = <FheUint8Parameters as IntegerParameter>::Id::default();
        let key = id.unwrapped_ref_key(key);
        let ciphertext = key.inner.encrypt(u64::from(value));
        Self::new(ciphertext, id)
    }
}

impl FheDecrypt<u8> for FheUint8 {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> u8 {
        let id = <FheUint8Parameters as IntegerParameter>::Id::default();
        let key = id.unwrapped_ref_key(key);
        key.inner.decrypt(&self.ciphertext.borrow()) as u8
    }
}

impl FheEncrypt<u16> for FheUint16 {
    #[track_caller]
    fn encrypt(value: u16, key: &ClientKey) -> Self {
        let id = <FheUint16Parameters as IntegerParameter>::Id::default();
        let key = id.unwrapped_ref_key(key);
        let ciphertext = key.inner.encrypt(u64::from(value));
        Self::new(ciphertext, id)
    }
}

impl FheDecrypt<u16> for FheUint16 {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> u16 {
        let id = <FheUint16Parameters as IntegerParameter>::Id::default();
        let key = id.unwrapped_ref_key(key);
        key.inner.decrypt(&self.ciphertext.borrow()) as u16
    }
}
