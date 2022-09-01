use std::borrow::Borrow;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::FheBoolParameters;
use concrete_boolean::ciphertext::Ciphertext;
use concrete_boolean::client_key::ClientKey as BooleanClientKey;
use concrete_boolean::prelude::BinaryBooleanGates;
use concrete_boolean::server_key::ServerKey;

use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey};
use crate::global_state::WithGlobalKey;
use crate::keys::{ClientKey, RefKeyFromKeyChain};
use std::ops::{BitAnd, BitOr, BitXor};

// Has Overridable Operator:
// - and => BitAnd => &
// - not => Not => !
// - or => BitOr => |
// - xor => BitXor => ^
//
// Does Not have overridable operator:
// - mux -> But maybe by using a macro_rules with regular function we can have some sufficiently
//   nice syntax sugar
// - nand
// - nor
// - xnor should be Eq => ==,  But Eq requires to return a bool not a FHE bool So we cant do it
// - ||, && cannot be overloaded, maybe a well-crafted macro-rules that implements `if-else` could
//   bring this syntax sugar

use crate::traits::{FheDecrypt, FheEncrypt};

/// The FHE boolean data type.
///
/// To be able to use this type, the cargo feature `booleans` must be enabled,
/// and your config should also enable the type with either default parameters or custom ones.
///
/// # Example
/// ```rust
/// use concrete::prelude::*;
/// #[cfg(feature = "booleans")]
/// # {
/// use concrete::{generate_keys, set_server_key, ConfigBuilder, FheBool};
///
/// // Enable booleans in the config
/// let config = ConfigBuilder::all_disabled().enable_default_bool().build();
///
/// // With the booleans enabled in the config, the needed keys and details
/// // can be taken care of.
/// let (client_key, server_key) = generate_keys(config);
///
/// let ttrue = FheBool::encrypt(true, &client_key);
/// let ffalse = FheBool::encrypt(false, &client_key);
///
/// // Do not forget to set the server key before doing any computation
/// set_server_key(server_key);
///
/// let fhe_result = ttrue & ffalse;
///
/// let clear_result = fhe_result.decrypt(&client_key);
/// assert_eq!(clear_result, false);
/// # }
/// ```
#[cfg_attr(doc, cfg(feature = "booleans"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct FheBool {
    ciphertext: Ciphertext,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct FheBoolClientKey {
    key: BooleanClientKey,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub(super) struct FheBoolServerKey {
    key: ServerKey,
}

impl_with_global_key!(
    for FheBoolServerKey {
        keychain_member: bool_key.key,
        type_variant: Type::FheBool,
    }
);

impl FheBool {
    pub fn nand(&self, rhs: &Self) -> Self {
        FheBoolServerKey::with_unwrapped_global_mut(|key| key.nand(self, rhs))
    }

    pub fn eq(&self, other: &Self) -> Self {
        FheBoolServerKey::with_unwrapped_global_mut(|key| key.xnor(self, other))
    }

    pub fn neq(&self, other: &Self) -> Self {
        FheBoolServerKey::with_unwrapped_global_mut(|key| {
            let eq = key.xnor(self, other);
            key.not(&eq)
        })
    }
}

impl FheEncrypt<bool> for FheBool {
    #[track_caller]
    fn encrypt(value: bool, key: &ClientKey) -> Self {
        let key = FheBoolClientKey::unwrapped_ref_key(key);
        key.key.encrypt(value).into()
    }
}

#[cfg(feature = "__newer_booleans")]
impl crate::traits::FheTrivialEncrypt<bool> for FheBool {
    fn encrypt_trivial(value: bool) -> Self {
        Ciphertext::Trivial(value).into()
    }
}

impl FheDecrypt<bool> for FheBool {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> bool {
        let key = FheBoolClientKey::unwrapped_ref_key(key);
        key.key.decrypt(&self.ciphertext)
    }
}

impl From<Ciphertext> for FheBool {
    fn from(ciphertext: Ciphertext) -> Self {
        Self { ciphertext }
    }
}

impl FheBoolClientKey {
    pub(crate) fn new(parameters: FheBoolParameters) -> Self {
        Self {
            #[cfg(feature = "__newer_booleans")]
            key: BooleanClientKey::new(parameters.into()),
            #[cfg(not(feature = "__newer_booleans"))]
            key: BooleanClientKey::new(&parameters.0),
        }
    }
}

impl FheBoolServerKey {
    pub(crate) fn new(key: &FheBoolClientKey) -> Self {
        Self {
            key: ServerKey::new(&key.key),
        }
    }

    fn and(&mut self, lhs: &FheBool, rhs: &FheBool) -> FheBool {
        self.key.and(&lhs.ciphertext, &rhs.ciphertext).into()
    }

    fn or(&mut self, lhs: &FheBool, rhs: &FheBool) -> FheBool {
        self.key.or(&lhs.ciphertext, &rhs.ciphertext).into()
    }

    fn xor(&mut self, lhs: &FheBool, rhs: &FheBool) -> FheBool {
        self.key.xor(&lhs.ciphertext, &rhs.ciphertext).into()
    }

    fn xnor(&mut self, lhs: &FheBool, rhs: &FheBool) -> FheBool {
        self.key.xnor(&lhs.ciphertext, &rhs.ciphertext).into()
    }

    fn nand(&mut self, lhs: &FheBool, rhs: &FheBool) -> FheBool {
        self.key.nand(&lhs.ciphertext, &rhs.ciphertext).into()
    }

    fn not(&mut self, lhs: &FheBool) -> FheBool {
        self.key.not(&lhs.ciphertext).into()
    }

    fn mux(
        &mut self,
        condition: &FheBool,
        then_result: &FheBool,
        else_result: &FheBool,
    ) -> FheBool {
        self.key
            .mux(
                &condition.ciphertext,
                &then_result.ciphertext,
                &else_result.ciphertext,
            )
            .into()
    }
}

#[cfg_attr(doc, cfg(feature = "booleans"))]
pub fn if_then_else<B1, B2>(ct_condition: B1, ct_then: B2, ct_else: B2) -> FheBool
where
    B1: Borrow<FheBool>,
    B2: Borrow<FheBool>,
{
    FheBoolServerKey::with_unwrapped_global_mut(|key| {
        key.mux(ct_condition.borrow(), ct_then.borrow(), ct_else.borrow())
    })
}

impl RefKeyFromKeyChain for FheBoolClientKey {
    fn ref_key(keys: &ClientKey) -> Result<&Self, UninitializedClientKey> {
        keys.bool_key
            .key
            .as_ref()
            .ok_or(UninitializedClientKey(Type::FheBool))
    }
}

macro_rules! fhe_bool_impl_operation(
    ($trait_name:ident($trait_method:ident) => $key_method:ident) => {
        impl<B> $trait_name<B> for FheBool
        where B: Borrow<FheBool> {
            type Output = FheBool;

            fn $trait_method(self, rhs: B) -> Self::Output {
                <&Self as $trait_name<B>>::$trait_method(&self, rhs)
            }
        }

        impl<B> $trait_name<B> for &FheBool
        where B: Borrow<FheBool> {
            type Output = FheBool;

            fn $trait_method(self, rhs: B) -> Self::Output {
                FheBoolServerKey::with_unwrapped_global_mut(|key| {
                  key.$key_method(self, rhs.borrow())
                })
            }
        }
    };
);

fhe_bool_impl_operation!(BitAnd(bitand) => and);
fhe_bool_impl_operation!(BitOr(bitor) => or);
fhe_bool_impl_operation!(BitXor(bitxor) => xor);

impl ::std::ops::Not for FheBool {
    type Output = Self;

    fn not(self) -> Self::Output {
        FheBoolServerKey::with_unwrapped_global_mut(|key| key.not(&self))
    }
}

impl ::std::ops::Not for &FheBool {
    type Output = FheBool;

    fn not(self) -> Self::Output {
        FheBoolServerKey::with_unwrapped_global_mut(|key| key.not(self))
    }
}
