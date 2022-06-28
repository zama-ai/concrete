use concrete_integer::ciphertext::Ciphertext;
use concrete_integer::client_key::ClientKey as IntegerClientKey;
use concrete_integer::server_key::ServerKey;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, MulAssign,
    Neg, Sub, SubAssign,
};

use crate::errors::{Type, UninitializedClientKey, UnwrapResultExt};
use crate::traits::{DynamicFheEncryptor, FheDecrypt};
use crate::{ClientKey, DynShortIntParameters};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) struct IntegerTypeId(pub(super) usize);

impl IntegerTypeId {
    fn with_unwrapped_global_key<R, F>(&self, func: F) -> R
    where
        F: FnOnce(&mut DynIntegerServerKey) -> R,
    {
        crate::global_state::with_internal_keys_mut(|key| {
            let key = key
                .integer_key
                .custom_keys
                .get_mut(self.0)
                .expect("Server key for this dynamic integer type is not set");
            func(key)
        })
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "integers"))]
#[derive(Copy, Clone, Debug)]
pub struct DynIntegerParameters {
    pub block_parameters: DynShortIntParameters,
    pub num_block: usize,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "integers"))]
#[derive(Clone)]
pub struct DynIntegerEncryptor {
    type_id: IntegerTypeId,
}

impl From<IntegerTypeId> for DynIntegerEncryptor {
    fn from(type_id: IntegerTypeId) -> Self {
        Self { type_id }
    }
}

impl DynamicFheEncryptor<u64> for DynIntegerEncryptor {
    type FheType = DynInteger;

    #[track_caller]
    fn encrypt(&self, value: u64, key: &ClientKey) -> Self::FheType {
        let key = key
            .integer_key
            .custom_keys
            .get(self.type_id.0)
            .ok_or(UninitializedClientKey(Type::DynamicInteger))
            .unwrap_display();

        let ciphertext = key.key.encrypt(value);
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: self.type_id,
        }
    }
}

/// An instance of a dynamically defined integer type
///
/// To create a new value, you need to use its corresponding [DynIntegerEncryptor]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "integers"))]
#[derive(Clone)]
pub struct DynInteger {
    ciphertext: RefCell<Ciphertext>,
    type_id: IntegerTypeId,
}

impl FheDecrypt<u64> for DynInteger {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> u64 {
        let key = key
            .integer_key
            .custom_keys
            .get(self.type_id.0)
            .ok_or(UninitializedClientKey(Type::DynamicInteger))
            .unwrap_display();
        key.key.decrypt(&self.ciphertext.borrow())
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct DynIntegerClientKey {
    key: IntegerClientKey,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub(super) struct DynIntegerServerKey {
    key: ServerKey,
}

impl From<DynIntegerParameters> for DynIntegerClientKey {
    fn from(parameters: DynIntegerParameters) -> Self {
        Self {
            key: IntegerClientKey::new(parameters.block_parameters, parameters.num_block),
        }
    }
}

impl DynIntegerServerKey {
    pub(super) fn new(client_key: &DynIntegerClientKey) -> Self {
        Self {
            key: ServerKey::new(&client_key.key),
        }
    }

    fn smart_neg(&mut self, lhs: &DynInteger) -> DynInteger {
        let ciphertext = self.key.smart_neg(&mut lhs.ciphertext.borrow_mut());
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_add(&mut self, lhs: &DynInteger, rhs: &DynInteger) -> DynInteger {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_add(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_sub(&mut self, lhs: &DynInteger, rhs: &DynInteger) -> DynInteger {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_sub(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_mul(&mut self, lhs: &DynInteger, rhs: &DynInteger) -> DynInteger {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_mul(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_bitand(&mut self, lhs: &DynInteger, rhs: &DynInteger) -> DynInteger {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_bitand(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_bitor(&mut self, lhs: &DynInteger, rhs: &DynInteger) -> DynInteger {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_bitor(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_bitxor(&mut self, lhs: &DynInteger, rhs: &DynInteger) -> DynInteger {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_bitxor(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_add_assign(&mut self, lhs: &mut DynInteger, rhs: &DynInteger) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_add_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_sub_assign(&mut self, lhs: &mut DynInteger, rhs: &DynInteger) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_sub_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_mul_assign(&mut self, lhs: &mut DynInteger, rhs: &DynInteger) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_mul_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_bitand_assign(&mut self, lhs: &mut DynInteger, rhs: &DynInteger) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_bitand_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_bitor_assign(&mut self, lhs: &mut DynInteger, rhs: &DynInteger) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_bitor_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_bitxor_assign(&mut self, lhs: &mut DynInteger, rhs: &DynInteger) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_bitxor_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_scalar_add(&mut self, lhs: &DynInteger, rhs: u64) -> DynInteger {
        let ciphertext = self
            .key
            .smart_scalar_add(&mut lhs.ciphertext.borrow_mut(), rhs);
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_scalar_sub(&mut self, lhs: &DynInteger, rhs: u64) -> DynInteger {
        let ciphertext = self
            .key
            .smart_scalar_sub(&mut lhs.ciphertext.borrow_mut(), rhs);
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_scalar_mul(&mut self, lhs: &DynInteger, rhs: u64) -> DynInteger {
        let ciphertext = self
            .key
            .smart_scalar_mul(&mut lhs.ciphertext.borrow_mut(), rhs);
        DynInteger {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_scalar_add_assign(&mut self, lhs: &mut DynInteger, rhs: u64) {
        self.key
            .smart_scalar_add_assign(&mut lhs.ciphertext.borrow_mut(), rhs);
    }

    fn smart_scalar_sub_assign(&mut self, lhs: &mut DynInteger, rhs: u64) {
        self.key
            .smart_scalar_sub_assign(&mut lhs.ciphertext.borrow_mut(), rhs);
    }

    fn smart_scalar_mul_assign(&mut self, lhs: &mut DynInteger, rhs: u64) {
        self.key
            .smart_scalar_mul_assign(&mut lhs.ciphertext.borrow_mut(), rhs);
    }
}

impl DynInteger {
    fn panic_if_not_same_type_id(&self, other: &Self) {
        if self.type_id != other.type_id {
            panic!("Cannot use two integers that were not encrypted with the same encryptor");
        }
    }

    fn with_unwrapped_global_key<R, F>(&self, func: F) -> R
    where
        F: FnOnce(&mut DynIntegerServerKey) -> R,
    {
        self.type_id.with_unwrapped_global_key(func)
    }
}

macro_rules! dyn_integer_impl_operation {
    ($trait_name:ident($trait_method:ident) => $key_method:ident) => {
        impl<B> $trait_name<B> for &DynInteger
        where
            B: Borrow<DynInteger>,
        {
            type Output = DynInteger;

            fn $trait_method(self, rhs: B) -> Self::Output {
                let rhs = rhs.borrow();
                if self.type_id != rhs.type_id {
                    panic!("Cannot add two types without same type id");
                }

                self.with_unwrapped_global_key(|server_key| server_key.$key_method(self, &rhs))
            }
        }

        impl<B> $trait_name<B> for DynInteger
        where
            B: Borrow<DynInteger>,
        {
            type Output = DynInteger;

            fn $trait_method(self, rhs: B) -> Self::Output {
                <&Self as $trait_name<B>>::$trait_method(&self, rhs)
            }
        }
    };
}

macro_rules! dyn_integer_impl_operation_assign {
    ($trait_name:ident($trait_method:ident) => $key_method:ident) => {
        impl<B> $trait_name<B> for DynInteger
        where
            B: Borrow<DynInteger>,
        {
            fn $trait_method(&mut self, rhs: B) {
                let type_id = self.type_id;
                type_id.with_unwrapped_global_key(|server_key| {
                    server_key.$key_method(self, rhs.borrow())
                })
            }
        }
    };
}

// Scalar Operations
macro_rules! dyn_integer_impl_scalar_operation {
    ($trait_name:ident($trait_method:ident) => $key_method:ident($($scalar_type:ty),*)) => {
        $(
            impl $trait_name<$scalar_type> for &DynInteger {
                type Output = DynInteger;

                fn $trait_method(self, rhs: $scalar_type) -> Self::Output {
                    self.with_unwrapped_global_key(|server_key| server_key.$key_method(self, u64::from(rhs)))
                }
            }

            impl $trait_name<$scalar_type> for DynInteger {
                type Output = DynInteger;

                fn $trait_method(self, rhs: $scalar_type) -> Self::Output {
                    <&Self as $trait_name<$scalar_type>>::$trait_method(&self, rhs)
                }
            }

            impl $trait_name<&DynInteger> for $scalar_type {
                type Output = DynInteger;

                fn $trait_method(self, rhs: &DynInteger) -> Self::Output {
                    <&DynInteger as $trait_name<$scalar_type>>::$trait_method(rhs, self)
                }
            }

            impl $trait_name<DynInteger> for $scalar_type {
                type Output = DynInteger;

                fn $trait_method(self, rhs: DynInteger) -> Self::Output {
                    <Self as $trait_name<&DynInteger>>::$trait_method(self, &rhs)
                }
            }
        )*
    };
}

macro_rules! dyn_integer_impl_scalar_operation_assign {
    ($trait_name:ident($trait_method:ident) => $key_method:ident($($scalar_type:ty),*)) => {
        $(
            impl $trait_name<$scalar_type> for DynInteger {
                fn $trait_method(&mut self, rhs: $scalar_type) {
                    let type_id = self.type_id;
                    type_id.with_unwrapped_global_key(|server_key| server_key.$key_method(self, u64::from(rhs)))
                }
            }
        )*
    };
}

impl Neg for DynInteger {
    type Output = DynInteger;

    fn neg(self) -> Self::Output {
        <&Self as Neg>::neg(&self)
    }
}

impl Neg for &DynInteger {
    type Output = DynInteger;

    fn neg(self) -> Self::Output {
        self.with_unwrapped_global_key(|key| key.smart_neg(self))
    }
}

dyn_integer_impl_operation!(Add(add) => smart_add);
dyn_integer_impl_operation!(Sub(sub) => smart_sub);
dyn_integer_impl_operation!(Mul(mul) => smart_mul);
dyn_integer_impl_operation!(BitAnd(bitand) => smart_bitand);
dyn_integer_impl_operation!(BitOr(bitor) => smart_bitor);
dyn_integer_impl_operation!(BitXor(bitxor) => smart_bitxor);

dyn_integer_impl_operation_assign!(AddAssign(add_assign) => smart_add_assign);
dyn_integer_impl_operation_assign!(SubAssign(sub_assign) => smart_sub_assign);
dyn_integer_impl_operation_assign!(MulAssign(mul_assign) => smart_mul_assign);
dyn_integer_impl_operation_assign!(BitAndAssign(bitand_assign) => smart_bitand_assign);
dyn_integer_impl_operation_assign!(BitOrAssign(bitor_assign) => smart_bitor_assign);
dyn_integer_impl_operation_assign!(BitXorAssign(bitxor_assign) => smart_bitxor_assign);

dyn_integer_impl_scalar_operation!(Mul(mul) => smart_scalar_mul(u8, u16, u32, u64));
dyn_integer_impl_scalar_operation!(Add(add) => smart_scalar_add(u8, u16, u32, u64));
dyn_integer_impl_scalar_operation!(Sub(sub) => smart_scalar_sub(u8, u16, u32, u64));

dyn_integer_impl_scalar_operation_assign!(MulAssign(mul_assign) => smart_scalar_mul_assign(u8, u16, u32, u64));
dyn_integer_impl_scalar_operation_assign!(AddAssign(add_assign) => smart_scalar_add_assign(u8, u16, u32, u64));
dyn_integer_impl_scalar_operation_assign!(SubAssign(sub_assign) => smart_scalar_sub_assign(u8, u16, u32, u64));
