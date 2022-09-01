use concrete_boolean::ciphertext::Ciphertext;
use concrete_boolean::client_key::ClientKey as BooleanClientKey;
use concrete_boolean::prelude::*;
use concrete_boolean::server_key::ServerKey;

use std::borrow::Borrow;
use std::ops::{BitAnd, BitOr, BitXor, Not};

use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey, UnwrapResultExt};
use crate::{ClientKey, FheBoolParameters};

use crate::traits::{DynamicFheEncryptor, FheDecrypt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Acts as an index in the vec of dynamic bool keys
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct BooleanTypeId(pub(super) usize);

/// struct to create new values of a dynamically defined type of boolean
///
/// You get one of these by using `ConfigBuilder`'s [add_bool_type]
///
/// [add_bool_type]: crate::ConfigBuilder::add_bool_type
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "booleans"))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DynFheBoolEncryptor {
    type_id: BooleanTypeId,
}

impl From<BooleanTypeId> for DynFheBoolEncryptor {
    fn from(type_id: BooleanTypeId) -> Self {
        Self { type_id }
    }
}

#[cfg(feature = "__newer_booleans")]
impl crate::traits::DynamicFheTrivialEncryptor<bool> for DynFheBoolEncryptor {
    type FheType = DynFheBool;

    fn encrypt_trivial(&self, value: bool) -> Self::FheType {
        Self::FheType {
            ciphertext: Ciphertext::Trivial(value),
            type_id: self.type_id,
        }
    }
}

impl DynamicFheEncryptor<bool> for DynFheBoolEncryptor {
    type FheType = DynFheBool;

    #[track_caller]
    fn encrypt(&self, value: bool, key: &ClientKey) -> Self::FheType {
        let client_key = key
            .bool_key
            .dynamic_keys
            .get(self.type_id.0)
            .ok_or(UninitializedServerKey(Type::DynamicBool))
            .unwrap_display();
        let ciphertext = client_key.key.encrypt(value);
        DynFheBool {
            ciphertext,
            type_id: self.type_id,
        }
    }
}

/// An instance of a dynamically defined boolean type
///
/// To create a new value, you need to use its corresponding [DynFheBoolEncryptor]
#[cfg_attr(doc, cfg(feature = "booleans"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct DynFheBool {
    ciphertext: Ciphertext,
    type_id: BooleanTypeId,
}

impl FheDecrypt<bool> for DynFheBool {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> bool {
        let client_key = key
            .bool_key
            .dynamic_keys
            .get(self.type_id.0)
            .ok_or(UninitializedClientKey(Type::DynamicBool))
            .unwrap_display();
        client_key.key.decrypt(&self.ciphertext)
    }
}

impl DynFheBool {
    pub fn nand(&self, rhs: &Self) -> Self {
        self.with_unwrapped_global_key(|key| key.nand(self, rhs))
    }

    pub fn xnor(&self, rhs: &Self) -> Self {
        self.with_unwrapped_global_key(|key| key.xnor(self, rhs))
    }

    fn panic_if_not_same_type_id(&self, other: &Self) {
        if self.type_id != other.type_id {
            panic!("Cannot use two booleans that were not encrypted with the same encryptor");
        }
    }

    /// Helper function, to reduce boiler plate
    ///
    /// It tries to find the thread_local server key tied
    /// to the DynFheBool type_id and execute the
    /// function given if the key was found, otherwise it panics
    /// with a somewhat user friendly error message.
    ///
    /// # Panic
    ///
    /// This panics if the key corresponding to `self.type_id`
    /// was not found
    fn with_unwrapped_global_key<R, F>(&self, func: F) -> R
    where
        F: FnOnce(&mut DynFheBoolServerKey) -> R,
    {
        crate::global_state::with_internal_keys_mut(|key| {
            let server_key = key
                .bool_key
                .dynamic_keys
                .get_mut(self.type_id.0)
                .ok_or(UninitializedServerKey(Type::DynamicBool))
                .unwrap_display();
            func(server_key)
        })
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct DynFheBoolClientKey {
    key: BooleanClientKey,
}

impl DynFheBoolClientKey {
    pub(crate) fn new(parameters: FheBoolParameters) -> Self {
        Self {
            #[cfg(feature = "__newer_booleans")]
            key: BooleanClientKey::new(parameters.into()),
            #[cfg(not(feature = "__newer_booleans"))]
            key: BooleanClientKey::new(&parameters.0),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub(super) struct DynFheBoolServerKey {
    key: ServerKey,
}

impl DynFheBoolServerKey {
    pub(crate) fn new(key: &DynFheBoolClientKey) -> Self {
        Self {
            key: ServerKey::new(&key.key),
        }
    }

    fn and(&mut self, lhs: &DynFheBool, rhs: &DynFheBool) -> DynFheBool {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.and(&lhs.ciphertext, &rhs.ciphertext);
        DynFheBool {
            ciphertext,
            type_id: lhs.type_id,
        }
    }

    fn or(&mut self, lhs: &DynFheBool, rhs: &DynFheBool) -> DynFheBool {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.or(&lhs.ciphertext, &rhs.ciphertext);
        DynFheBool {
            ciphertext,
            type_id: lhs.type_id,
        }
    }

    fn xor(&mut self, lhs: &DynFheBool, rhs: &DynFheBool) -> DynFheBool {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.xor(&lhs.ciphertext, &rhs.ciphertext);
        DynFheBool {
            ciphertext,
            type_id: lhs.type_id,
        }
    }

    fn nand(&mut self, lhs: &DynFheBool, rhs: &DynFheBool) -> DynFheBool {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.nand(&lhs.ciphertext, &rhs.ciphertext);
        DynFheBool {
            ciphertext,
            type_id: lhs.type_id,
        }
    }

    fn xnor(&mut self, lhs: &DynFheBool, rhs: &DynFheBool) -> DynFheBool {
        let ciphertext = self.key.xnor(&lhs.ciphertext, &rhs.ciphertext);
        DynFheBool {
            ciphertext,
            type_id: lhs.type_id,
        }
    }

    fn not(&mut self, lhs: &DynFheBool) -> DynFheBool {
        let ciphertext = self.key.not(&lhs.ciphertext);
        DynFheBool {
            ciphertext,
            type_id: lhs.type_id,
        }
    }

    // fn mux(
    //     &mut self,
    //     condition: &DynFheBool,
    //     then_result: &DynFheBool,
    //     else_result: &DynFheBool,
    // ) -> DynFheBool {
    //     then_result.panic_if_not_same_type_id(else_result);
    //     let ciphertext = self.key.mux(
    //         &condition.ciphertext,
    //         &then_result.ciphertext,
    //         &else_result.ciphertext,
    //     );
    //     DynFheBool {
    //         ciphertext,
    //         type_id: condition.type_id,
    //     }
    // }
}

macro_rules! dyn_fhe_bool_impl_operation(
    ($trait_name:ident($trait_method:ident) => $key_method:ident) => {
        impl<B> $trait_name<B> for DynFheBool
        where B: Borrow<DynFheBool> {
            type Output = DynFheBool;

            fn $trait_method(self, rhs: B) -> Self::Output {
                <&Self as $trait_name<B>>::$trait_method(&self, rhs)
            }
        }

        impl<B> $trait_name<B> for &DynFheBool
        where B: Borrow<DynFheBool> {
            type Output = DynFheBool;

            fn $trait_method(self, rhs: B) -> Self::Output {
                self.with_unwrapped_global_key(|key| {
                    key.$key_method(self, rhs.borrow())
                })
            }
        }
    };
);

dyn_fhe_bool_impl_operation!(BitAnd(bitand) => and);
dyn_fhe_bool_impl_operation!(BitOr(bitor) => or);
dyn_fhe_bool_impl_operation!(BitXor(bitxor) => xor);

impl Not for DynFheBool {
    type Output = Self;

    fn not(self) -> Self::Output {
        self.with_unwrapped_global_key(|key| key.not(&self))
    }
}

impl Not for &DynFheBool {
    type Output = DynFheBool;

    fn not(self) -> Self::Output {
        self.with_unwrapped_global_key(|key| key.not(self))
    }
}
