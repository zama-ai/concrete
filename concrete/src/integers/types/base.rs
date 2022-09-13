use std::borrow::Borrow;
use std::cell::RefCell;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, MulAssign,
    Neg, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use concrete_integer::ciphertext::Ciphertext;

use crate::global_state::WithGlobalKey;
use crate::integers::client_key::GenericIntegerClientKey;
use crate::integers::parameters::{IntegerParameter, StaticIntegerParameter};
use crate::integers::server_key::GenericIntegerServerKey;
use crate::keys::RefKeyFromKeyChain;
use crate::traits::{FheDecrypt, FheNumberConstant, FheTryEncrypt};
use crate::{ClientKey, OutOfRangeError};

/// A Generic FHE unsigned integer
///
/// Contrary to *shortints*, these integers can in theory by parametrized to
/// represent integers of any number of bits (eg: 16, 24, 32, 64).
///
/// However, in practice going above 16 bits may not be ideal as the
/// computations would not scale and become very expensive.
///
/// Integers works by combining together multiple shortints
/// with one of the available representation.
///
/// This struct is generic over some parameters, as its the parameters
/// that controls how many bit they represent.
/// You will need to use one of this type specialization (e.g., [FheUint8], [FheUint12],
/// [FheUint16]).
///
/// Its the type that overloads the operators (`+`, `-`, `*`),
/// since the `GenericInteger` type is not `Copy` the operators are also overloaded
/// to work with references.
///
///
/// To be able to use this type, the cargo feature `integers` must be enabled,
/// and your config should also enable the type with either default parameters or custom ones.
///
///
/// [FheUint8]: crate::FheUint8
/// [FheUint12]: crate::FheUint12
/// [FheUint16]: crate::FheUint16
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, doc(cfg(feature = "integers")))]
#[derive(Clone)]
pub struct GenericInteger<P: IntegerParameter> {
    pub(in crate::integers) ciphertext: RefCell<Ciphertext>,
    pub(in crate::integers) id: P::Id,
}

impl<P> GenericInteger<P>
where
    P: IntegerParameter,
{
    pub(in crate::integers) fn new(ciphertext: Ciphertext, id: P::Id) -> Self {
        Self {
            ciphertext: RefCell::new(ciphertext),
            id,
        }
    }
}

impl<P> GenericInteger<P>
where
    P: StaticIntegerParameter,
{
    pub const MIN: u64 = 0;
    pub const MAX: u64 = (1 << (P::MESSAGE_BITS as u64)) - 1;

    pub const MODULUS: u64 = (1 << (P::MESSAGE_BITS as u64));
}

impl<P> FheNumberConstant for GenericInteger<P>
where
    P: StaticIntegerParameter,
{
    const MIN: u64 = Self::MIN;
    const MAX: u64 = Self::MAX;
    const MODULUS: u64 = Self::MODULUS;
}

impl<P> FheDecrypt<u64> for GenericInteger<P>
where
    P: IntegerParameter,
    P::Id: RefKeyFromKeyChain<Key = GenericIntegerClientKey<P>>,
{
    fn decrypt(&self, key: &ClientKey) -> u64 {
        let key = self.id.unwrapped_ref_key(key);
        key.key.decrypt(&*self.ciphertext.borrow())
    }
}

impl<P, T> FheTryEncrypt<T> for GenericInteger<P>
where
    T: TryInto<u64>,
    P: StaticIntegerParameter,
    P::Id: RefKeyFromKeyChain<Key = GenericIntegerClientKey<P>> + Default,
{
    type Error = OutOfRangeError;

    fn try_encrypt(value: T, key: &ClientKey) -> Result<Self, Self::Error> {
        let value = value.try_into().map_err(|_err| OutOfRangeError)?;
        if value > Self::MAX {
            Err(OutOfRangeError)
        } else {
            let id = P::Id::default();
            let key = id.unwrapped_ref_key(key);
            let ciphertext = key.key.encrypt(value);
            Ok(Self::new(ciphertext, id))
        }
    }
}

macro_rules! generic_integer_impl_operation (
    ($trait_name:ident($trait_method:ident,$op:tt) => $key_method:ident) => {
        #[doc = concat!(" Allows using the `", stringify!($op), "` operator between a")]
        #[doc = " `GenericInteger` and a `GenericInteger` or a `&GenericInteger`"]
        #[doc = " "]
        #[doc = " # Examples "]
        #[doc = " "]
        #[doc = " ```"]
        #[doc = " # fn main() -> Result<(), concrete::OutOfRangeError> {"]
        #[doc = " use concrete::prelude::*;"]
        #[doc = " use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint8};"]
        #[doc = " use std::num::Wrapping;"]
        #[doc = " "]
        #[doc = " let config = ConfigBuilder::all_disabled()"]
        #[doc = "     .enable_default_uint8()"]
        #[doc = "     .build();"]
        #[doc = " let (keys, server_key) = generate_keys(config);"]
        #[doc = " "]
        #[doc = " let a = FheUint8::try_encrypt(142, &keys)?;"]
        #[doc = " let b = FheUint8::try_encrypt(83, &keys)?;"]
        #[doc = " "]
        #[doc = " set_server_key(server_key);"]
        #[doc = " "]
        #[doc = concat!(" let c = a ", stringify!($op), " b;")]
        #[doc = " let decrypted: u8 = c.decrypt(&keys);"]
        #[doc = concat!(" let expected = Wrapping(142u8) ", stringify!($op), " Wrapping(83u8);")]
        #[doc = " assert_eq!(decrypted, expected.0);"]
        #[doc = " # Ok(())"]
        #[doc = " # }"]
        #[doc = " ```"]
        #[doc = " "]
        #[doc = " "]
        #[doc = " ```"]
        #[doc = " # fn main() -> Result<(), concrete::OutOfRangeError> {"]
        #[doc = " use concrete::prelude::*;"]
        #[doc = " use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint8};"]
        #[doc = " use std::num::Wrapping;"]
        #[doc = " "]
        #[doc = " let config = ConfigBuilder::all_disabled()"]
        #[doc = "     .enable_default_uint8()"]
        #[doc = "     .build();"]
        #[doc = " let (keys, server_key) = generate_keys(config);"]
        #[doc = " "]
        #[doc = " let a = FheUint8::try_encrypt(208, &keys)?;"]
        #[doc = " let b = FheUint8::try_encrypt(29, &keys)?;"]
        #[doc = " "]
        #[doc = " set_server_key(server_key);"]
        #[doc = " "]
        #[doc = concat!(" let c = a ", stringify!($op), " &b;")]
        #[doc = " let decrypted: u8 = c.decrypt(&keys);"]
        #[doc = concat!(" let expected = Wrapping(208u8) ", stringify!($op), " Wrapping(29u8);")]
        #[doc = " assert_eq!(decrypted, expected.0);"]
        #[doc = " # Ok(())"]
        #[doc = " # }"]
        #[doc = " ```"]
        impl<P, B> $trait_name<B> for GenericInteger<P>
        where
            P: IntegerParameter,
            B: Borrow<Self>,
            P::Id: WithGlobalKey<Key=GenericIntegerServerKey<P>>,
        {
            type Output = Self;

            fn $trait_method(self, rhs: B) -> Self::Output {
                <&Self as $trait_name<B>>::$trait_method(&self, rhs)
            }
        }

        impl<P, B> $trait_name<B> for &GenericInteger<P>
        where
            P: IntegerParameter,
            B: Borrow<GenericInteger<P>>,
            P::Id: WithGlobalKey<Key=GenericIntegerServerKey<P>>,
        {
            type Output = GenericInteger<P>;

            fn $trait_method(self, rhs: B) -> Self::Output {
                self.id.with_unwrapped_global_mut(|key| {
                    key.$key_method(self, rhs.borrow())
                })
            }
        }
    }
);

macro_rules! generic_integer_impl_operation_assign (
    ($trait_name:ident($trait_method:ident, $op:tt) => $key_method:ident) => {
        impl<P, I> $trait_name<I> for GenericInteger<P>
        where
            P: IntegerParameter,
            P::Id: WithGlobalKey<Key=GenericIntegerServerKey<P>>,
            I: Borrow<Self>,
        {
            fn $trait_method(&mut self, rhs: I) {
                self.id.with_unwrapped_global_mut(|key| {
                    key.$key_method(&self, rhs.borrow())
                })
            }
        }
    }
);

macro_rules! generic_integer_impl_scalar_operation {
    ($trait_name:ident($trait_method:ident) => $key_method:ident($($scalar_type:ty),*)) => {
        $(
            impl<P> $trait_name<$scalar_type> for GenericInteger<P>
            where
                P: IntegerParameter,
                P::Id: WithGlobalKey<Key=GenericIntegerServerKey<P>>,
            {
                type Output = GenericInteger<P>;

                fn $trait_method(self, rhs: $scalar_type) -> Self::Output {
                    <&Self as $trait_name<$scalar_type>>::$trait_method(&self, rhs)
                }
            }

            impl<P> $trait_name<$scalar_type> for &GenericInteger<P>
            where
                P: IntegerParameter,
                P::Id: WithGlobalKey<Key=GenericIntegerServerKey<P>>,
            {
                type Output = GenericInteger<P>;

                fn $trait_method(self, rhs: $scalar_type) -> Self::Output {
                    self.id.with_unwrapped_global_mut(|key| {
                        key.$key_method(self, rhs.into())
                    })
                }
            }
        )*
    };
}

macro_rules! generic_integer_impl_scalar_operation_assign {
    ($trait_name:ident($trait_method:ident) => $key_method:ident($($scalar_type:ty),*)) => {
        $(
            impl<P> $trait_name<$scalar_type> for GenericInteger<P>
                where
                    P: IntegerParameter,
                    P::Id: WithGlobalKey<Key=GenericIntegerServerKey<P>>,
            {
                fn $trait_method(&mut self, rhs: $scalar_type) {
                    self.id.with_unwrapped_global_mut(|key| {
                        key.$key_method(self, u64::from(rhs))
                    });
                }
            }
        )*
    }
}

generic_integer_impl_operation!(Add(add,+) => smart_add);
generic_integer_impl_operation!(Sub(sub,-) => smart_sub);
generic_integer_impl_operation!(Mul(mul,*) => smart_mul);
generic_integer_impl_operation!(BitAnd(bitand,&) => smart_bitand);
generic_integer_impl_operation!(BitOr(bitor,^) => smart_bitor);
generic_integer_impl_operation!(BitXor(bitxor,|) => smart_bitxor);

generic_integer_impl_operation_assign!(AddAssign(add_assign,+=) => smart_add_assign);
generic_integer_impl_operation_assign!(SubAssign(sub_assign,-=) => smart_sub_assign);
generic_integer_impl_operation_assign!(MulAssign(mul_assign,*=) => smart_mul_assign);
generic_integer_impl_operation_assign!(BitAndAssign(bitand_assign,&=) => smart_bitand_assign);
generic_integer_impl_operation_assign!(BitOrAssign(bitor_assign,|=) => smart_bitor_assign);
generic_integer_impl_operation_assign!(BitXorAssign(bitxor_assign,^=) => smart_bitxor_assign);

generic_integer_impl_scalar_operation!(Mul(mul) => smart_scalar_mul(u8, u16, u32, u64));
generic_integer_impl_scalar_operation!(Add(add) => smart_scalar_add(u8, u16, u32, u64));
generic_integer_impl_scalar_operation!(Sub(sub) => smart_scalar_sub(u8, u16, u32, u64));
generic_integer_impl_scalar_operation!(Shl(shl) => unchecked_scalar_left_shift(u8, u16, u32, u64));
generic_integer_impl_scalar_operation!(Shr(shr) => unchecked_scalar_right_shift(u8, u16, u32, u64));

generic_integer_impl_scalar_operation_assign!(AddAssign(add_assign) => smart_scalar_add_assign(u8, u16, u32, u64));
generic_integer_impl_scalar_operation_assign!(SubAssign(sub_assign) => smart_scalar_sub_assign(u8, u16, u32, u64));
generic_integer_impl_scalar_operation_assign!(MulAssign(mul_assign) => smart_scalar_mul_assign(u8, u16, u32, u64));
generic_integer_impl_scalar_operation_assign!(ShlAssign(shl_assign) => unchecked_scalar_left_shift_assign(u8, u16, u32, u64));
generic_integer_impl_scalar_operation_assign!(ShrAssign(shr_assign) => unchecked_scalar_right_shift_assign(u8, u16, u32, u64));

impl<P> Neg for GenericInteger<P>
where
    P: IntegerParameter,
    P::Id: WithGlobalKey<Key = GenericIntegerServerKey<P>>,
{
    type Output = GenericInteger<P>;

    fn neg(self) -> Self::Output {
        <&Self as Neg>::neg(&self)
    }
}

impl<P> Neg for &GenericInteger<P>
where
    P: IntegerParameter,
    P::Id: WithGlobalKey<Key = GenericIntegerServerKey<P>>,
{
    type Output = GenericInteger<P>;

    fn neg(self) -> Self::Output {
        self.id.with_unwrapped_global_mut(|key| key.smart_neg(self))
    }
}
