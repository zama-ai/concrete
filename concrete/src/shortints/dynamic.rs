use concrete_shortint::ciphertext::Ciphertext;
use concrete_shortint::client_key::ClientKey as ShortintClientKey;
use concrete_shortint::server_key::ServerKey;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Rem, Shl, Shr, Sub, SubAssign,
};

use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey, UnwrapResultExt};
use crate::traits::{DynamicFheTryEncryptor, FheBootstrap, FheDecrypt};
use crate::{ClientKey, OutOfRangeError};
pub use concrete_shortint::parameters::Parameters as DynShortIntParameters;

#[cfg(feature = "internal-keycache")]
use concrete_shortint::keycache::KEY_CACHE;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) struct ShortIntTypeId(pub usize);

impl ShortIntTypeId {
    fn with_unwrapped_global_key<R, F>(&self, func: F) -> R
    where
        F: FnOnce(&mut DynShortIntServerKey) -> R,
    {
        crate::global_state::with_internal_keys_mut(|key| {
            let server_key = key
                .shortint_key
                .dynamic_keys
                .get_mut(self.0)
                .ok_or(UninitializedServerKey(Type::DynamicShortInt))
                .unwrap_display();
            func(server_key)
        })
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct DynShortIntClientKey {
    key: ShortintClientKey,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub(super) struct DynShortIntServerKey {
    key: ServerKey,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "shortints"))]
#[derive(Clone)]
pub struct DynShortInt {
    ciphertext: RefCell<Ciphertext>,
    type_id: ShortIntTypeId,
}

/// An instance of a dynamically defined shortint type
///
/// To create a new value, you need to use its corresponding [DynShortIntEncryptor]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "shortints"))]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DynShortIntEncryptor {
    type_id: ShortIntTypeId,
}

impl DynShortIntClientKey {
    pub(crate) fn new(parameters: DynShortIntParameters) -> Self {
        #[cfg(feature = "internal-keycache")]
        let key = KEY_CACHE.get_from_param(parameters).client_key().clone();
        #[cfg(not(feature = "internal-keycache"))]
        let key = ShortintClientKey::new(parameters);

        Self { key }
    }
}

impl DynShortInt {
    fn panic_if_not_same_type_id(&self, other: &Self) {
        if self.type_id != other.type_id {
            panic!("Cannot use two shortints that were not encrypted with the same encryptor");
        }
    }

    fn with_unwrapped_global_key<R, F>(&self, func: F) -> R
    where
        F: FnOnce(&mut DynShortIntServerKey) -> R,
    {
        crate::global_state::with_internal_keys_mut(|key| {
            let server_key = key
                .shortint_key
                .dynamic_keys
                .get_mut(self.type_id.0)
                .ok_or(UninitializedServerKey(Type::DynamicShortInt))
                .unwrap_display();
            func(server_key)
        })
    }
}

impl FheDecrypt<u8> for DynShortInt {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> u8 {
        let client_key = &key
            .shortint_key
            .dynamic_keys
            .get(self.type_id.0)
            .ok_or(UninitializedClientKey(Type::DynamicShortInt))
            .unwrap_display();
        client_key.key.decrypt(&self.ciphertext.borrow()) as u8
    }
}

impl<T> DynamicFheTryEncryptor<T> for DynShortIntEncryptor
where
    T: TryInto<u8>,
{
    type FheType = DynShortInt;
    type Error = OutOfRangeError;

    #[track_caller]
    fn try_encrypt(&self, value: T, key: &ClientKey) -> Result<Self::FheType, Self::Error> {
        let client_key = &key
            .shortint_key
            .dynamic_keys
            .get(self.type_id.0)
            .ok_or(UninitializedClientKey(Type::DynamicShortInt))
            .unwrap_display();

        let value = value.try_into().map_err(|_err| OutOfRangeError)?;
        if value as usize >= client_key.key.parameters.message_modulus.0 {
            Err(OutOfRangeError)
        } else {
            let message = u64::from(value);
            let encrypted_int = client_key.key.encrypt(message);
            Ok(DynShortInt {
                ciphertext: RefCell::new(encrypted_int),
                type_id: self.type_id,
            })
        }
    }
}

impl From<ShortIntTypeId> for DynShortIntEncryptor {
    fn from(type_id: ShortIntTypeId) -> Self {
        Self { type_id }
    }
}

impl DynShortIntServerKey {
    pub(crate) fn new(client_key: &DynShortIntClientKey) -> Self {
        #[cfg(feature = "internal-keycache")]
        let key = KEY_CACHE
            .get_from_param(client_key.key.parameters)
            .server_key()
            .clone();
        #[cfg(not(feature = "internal-keycache"))]
        let key = ServerKey::new(&client_key.key);
        Self { key }
    }
}

impl DynShortIntServerKey {
    fn add(&mut self, lhs: &DynShortInt, rhs: &DynShortInt) -> DynShortInt {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_add(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );

        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn sub(&mut self, lhs: &DynShortInt, rhs: &DynShortInt) -> DynShortInt {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_sub(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );

        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn mul(&mut self, lhs: &DynShortInt, rhs: &DynShortInt) -> DynShortInt {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_mul_lsb(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );

        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn div(&mut self, lhs: &DynShortInt, rhs: &DynShortInt) -> DynShortInt {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_div(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );

        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_bitand(&mut self, lhs: &DynShortInt, rhs: &DynShortInt) -> DynShortInt {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_bitand(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );

        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_bitor(&mut self, lhs: &DynShortInt, rhs: &DynShortInt) -> DynShortInt {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_bitor(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );

        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_bitxor(&mut self, lhs: &DynShortInt, rhs: &DynShortInt) -> DynShortInt {
        lhs.panic_if_not_same_type_id(rhs);
        let ciphertext = self.key.smart_bitxor(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );

        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn neg(&mut self, lhs: &DynShortInt) -> DynShortInt {
        let ciphertext = self.key.smart_neg(&mut lhs.ciphertext.borrow_mut());
        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_add_assign(&mut self, lhs: &mut DynShortInt, rhs: &DynShortInt) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_add_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_sub_assign(&mut self, lhs: &mut DynShortInt, rhs: &DynShortInt) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_sub_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_mul_assign(&mut self, lhs: &mut DynShortInt, rhs: &DynShortInt) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_mul_lsb_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_div_assign(&mut self, lhs: &mut DynShortInt, rhs: &DynShortInt) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_div_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_bitand_assign(&mut self, lhs: &mut DynShortInt, rhs: &DynShortInt) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_bitand_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_bitor_assign(&mut self, lhs: &mut DynShortInt, rhs: &DynShortInt) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_bitor_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_bitxor_assign(&mut self, lhs: &mut DynShortInt, rhs: &DynShortInt) {
        lhs.panic_if_not_same_type_id(rhs);
        self.key.smart_bitxor_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_scalar_left_shift(&mut self, lhs: &DynShortInt, rhs: u8) -> DynShortInt {
        let ciphertext = self
            .key
            .smart_scalar_left_shift(&mut lhs.ciphertext.borrow_mut(), rhs);
        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn unchecked_scalar_right_shift(&mut self, lhs: &DynShortInt, rhs: u8) -> DynShortInt {
        let ciphertext = self
            .key
            .unchecked_scalar_right_shift(&lhs.ciphertext.borrow(), rhs);
        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_scalar_add(&mut self, lhs: &DynShortInt, rhs: u8) -> DynShortInt {
        let ciphertext = self
            .key
            .smart_scalar_add(&mut lhs.ciphertext.borrow_mut(), rhs);
        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_scalar_sub(&mut self, lhs: &DynShortInt, rhs: u8) -> DynShortInt {
        let ciphertext = self
            .key
            .smart_scalar_sub(&mut lhs.ciphertext.borrow_mut(), rhs);
        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn smart_scalar_mul(&mut self, lhs: &DynShortInt, rhs: u8) -> DynShortInt {
        let ciphertext = self
            .key
            .smart_scalar_mul(&mut lhs.ciphertext.borrow_mut(), rhs);
        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn unchecked_scalar_div(&mut self, lhs: &DynShortInt, rhs: u8) -> DynShortInt {
        let ciphertext = self.key.unchecked_scalar_div(&lhs.ciphertext.borrow(), rhs);
        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn unchecked_scalar_mod(&mut self, lhs: &DynShortInt, rhs: u8) -> DynShortInt {
        let ciphertext = self.key.unchecked_scalar_mod(&lhs.ciphertext.borrow(), rhs);
        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn bootstrap_with<F>(&mut self, lhs: &DynShortInt, func: F) -> DynShortInt
    where
        F: Fn(u64) -> u64,
    {
        let accumulator = self.key.generate_accumulator(func);
        let ciphertext = self
            .key
            .keyswitch_programmable_bootstrap(&lhs.ciphertext.borrow(), &accumulator);
        DynShortInt {
            ciphertext: RefCell::new(ciphertext),
            type_id: lhs.type_id,
        }
    }

    fn bootstrap_inplace_with<F>(&mut self, lhs: &mut DynShortInt, func: F)
    where
        F: Fn(u64) -> u64,
    {
        let accumulator = self.key.generate_accumulator(func);
        self.key.keyswitch_programmable_bootstrap_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &accumulator,
        );
    }
}

impl FheBootstrap for DynShortInt {
    fn map<F: Fn(u64) -> u64>(&self, func: F) -> Self {
        self.with_unwrapped_global_key(|key| key.bootstrap_with(self, func))
    }

    fn apply<F: Fn(u64) -> u64>(&mut self, func: F) {
        // Cannot use self.with_unwrapped_global_key, as the closure
        // we would give, will make a second borrow of self
        // which will rightfully make the compiler unhappy
        crate::global_state::with_internal_keys_mut(|key| {
            let server_key = key
                .shortint_key
                .dynamic_keys
                .get_mut(self.type_id.0)
                .ok_or(UninitializedServerKey(Type::DynamicShortInt))
                .unwrap_display();
            server_key.bootstrap_inplace_with(self, func)
        });
    }
}

macro_rules! dyn_short_int_impl_operation {
    ($trait_name:ident($trait_method:ident) => $key_method:ident) => {
        impl<B> $trait_name<B> for &DynShortInt
        where
            B: Borrow<DynShortInt>,
        {
            type Output = DynShortInt;

            fn $trait_method(self, rhs: B) -> Self::Output {
                let rhs = rhs.borrow();
                if self.type_id != rhs.type_id {
                    panic!("Cannot add two types without same type id");
                }

                self.with_unwrapped_global_key(|server_key| server_key.$key_method(self, &rhs))
            }
        }

        impl<B> $trait_name<B> for DynShortInt
        where
            B: Borrow<DynShortInt>,
        {
            type Output = DynShortInt;

            fn $trait_method(self, rhs: B) -> Self::Output {
                <&Self as $trait_name<B>>::$trait_method(&self, rhs)
            }
        }
    };
}

macro_rules! dyn_short_int_impl_operation_assign (
    ($trait_name:ident($trait_method:ident, $op:tt) => $key_method:ident) => {
        impl<I> $trait_name<I> for DynShortInt
        where
            I: Borrow<Self>,
        {
            fn $trait_method(&mut self, rhs: I) {
                let rhs = rhs.borrow();
                // Simple trick to make borrow checker happy
                let type_id = self.type_id;
                type_id.with_unwrapped_global_key(|server_key| server_key.$key_method(self, rhs))
            }
        }
    }
);

// Scalar Operations
macro_rules! dyn_short_int_impl_scalar_operation {
    ($trait_name:ident($trait_method:ident) => $key_method:ident) => {
        impl $trait_name<u8> for &DynShortInt {
            type Output = DynShortInt;

            fn $trait_method(self, rhs: u8) -> Self::Output {
                self.with_unwrapped_global_key(|server_key| server_key.$key_method(self, rhs))
            }
        }

        impl $trait_name<u8> for DynShortInt {
            type Output = DynShortInt;

            fn $trait_method(self, rhs: u8) -> Self::Output {
                <&Self as $trait_name<u8>>::$trait_method(&self, rhs)
            }
        }

        impl $trait_name<&DynShortInt> for u8 {
            type Output = DynShortInt;

            fn $trait_method(self, rhs: &DynShortInt) -> Self::Output {
                <&DynShortInt as $trait_name<u8>>::$trait_method(rhs, self)
            }
        }

        impl $trait_name<DynShortInt> for u8 {
            type Output = DynShortInt;

            fn $trait_method(self, rhs: DynShortInt) -> Self::Output {
                <Self as $trait_name<&DynShortInt>>::$trait_method(self, &rhs)
            }
        }
    };
}

impl Neg for &DynShortInt {
    type Output = DynShortInt;

    fn neg(self) -> Self::Output {
        self.with_unwrapped_global_key(|key| key.neg(self))
    }
}

impl Neg for DynShortInt {
    type Output = DynShortInt;

    fn neg(self) -> Self::Output {
        <&Self as Neg>::neg(&self)
    }
}

dyn_short_int_impl_operation!(Add(add) => add);
dyn_short_int_impl_operation!(Sub(sub) => sub);
dyn_short_int_impl_operation!(Mul(mul) => mul);
dyn_short_int_impl_operation!(Div(div) => div);
dyn_short_int_impl_operation!(BitAnd(bitand) => smart_bitand);
dyn_short_int_impl_operation!(BitOr(bitor) => smart_bitor);
dyn_short_int_impl_operation!(BitXor(bitxor) => smart_bitxor);

dyn_short_int_impl_operation_assign!(AddAssign(add_assign,+=) => smart_add_assign);
dyn_short_int_impl_operation_assign!(SubAssign(sub_assign,-=) => smart_sub_assign);
dyn_short_int_impl_operation_assign!(MulAssign(mul_assign,*=) => smart_mul_assign);
dyn_short_int_impl_operation_assign!(DivAssign(div_assign,/=) => smart_div_assign);
dyn_short_int_impl_operation_assign!(BitAndAssign(bitand_assign,&=) => smart_bitand_assign);
dyn_short_int_impl_operation_assign!(BitOrAssign(bitor_assign,|=) => smart_bitor_assign);
dyn_short_int_impl_operation_assign!(BitXorAssign(bitxor_assign,^=) => smart_bitxor_assign);

dyn_short_int_impl_scalar_operation!(Add(add) => smart_scalar_add);
dyn_short_int_impl_scalar_operation!(Sub(sub) => smart_scalar_sub);
dyn_short_int_impl_scalar_operation!(Mul(mul) => smart_scalar_mul);
dyn_short_int_impl_scalar_operation!(Div(div) => unchecked_scalar_div);
dyn_short_int_impl_scalar_operation!(Rem(rem) => unchecked_scalar_mod);
dyn_short_int_impl_scalar_operation!(Shl(shl) => smart_scalar_left_shift);
dyn_short_int_impl_scalar_operation!(Shr(shr) => unchecked_scalar_right_shift);
