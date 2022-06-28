use std::borrow::Borrow;
use std::cell::RefCell;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Rem, Shl, Shr, Sub, SubAssign,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use concrete_shortint::ciphertext::Ciphertext;

use crate::errors::OutOfRangeError;
use crate::global_state::WithGlobalKey;
use crate::keys::{ClientKey, RefKeyFromKeyChain};
use crate::traits::{FheBootstrap, FheDecrypt, FheNumberConstant, FheTryEncrypt};

use super::client_key::ShortIntegerClientKey;
use super::server_key::ShortIntegerServerKey;
use super::ShortIntegerParameter;

/// A Generic short FHE unsigned integer
///
/// Short means less than 7 bits.
///
/// It is generic over some parameters, as its the parameters
/// that controls how many bit they represent.
///
/// Its the type that overloads the operators (`+`, `-`, `*`).
/// Since the `GenericShortInt` type is not `Copy` the operators are also overloaded
/// to work with references.
///
/// You will need to use one of this type specialization (e.g., [FheUint2], [FheUint3]).
///
/// To be able to use this type, the cargo feature `shortints` must be enabled,
/// and your config should also enable the type with either default parameters or custom ones.
///
/// # Example
///
/// To use FheUint2
///
/// ```
/// # #[cfg(feature = "shortints")]
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use concrete::prelude::*;
/// use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint2};
///
/// // Enable the FheUint2 type in the config
/// let config = ConfigBuilder::all_disabled().enable_default_uint2().build();
///
/// // With the FheUint2 type enabled in the config, the needed keys and details
/// // can be taken care of.
/// let (client_key, server_key) = generate_keys(config);
///
/// let a = FheUint2::try_encrypt(0, &client_key)?;
/// let b = FheUint2::try_encrypt(1, &client_key)?;
///
/// // Do not forget to set the server key before doing any computation
/// set_server_key(server_key);
///
/// // Since FHE types are bigger than native rust type they are not `Copy`,
/// // meaning that to reuse the same value in a computation and avoid the cost
/// // of calling `clone`, you'll have to use references:
/// let c = a + &b;
/// // `a` was moved but not `b`, so `a` cannot be reused, but `b` can
/// let d = &c + b;
/// // `b` was moved but not `c`, so `b` cannot be reused, but `c` can
/// let fhe_result = d + c;
/// // both `d` and `c` were moved.
///
/// let expected: u8 = {
///     let a = 0;
///     let b = 1;
///
///     let c = a + b;
///     let d = c + b;
///     d + c
/// };
/// let clear_result = fhe_result.decrypt(&client_key);
/// assert_eq!(expected, 3);
/// assert_eq!(clear_result, expected);
///
/// # Ok(())
/// # }
/// ```
///
/// [FheUint2]: crate::FheUint2
/// [FheUint3]: crate::FheUint3
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "shortints"))]
#[derive(Clone)]
pub struct GenericShortInt<P: ShortIntegerParameter> {
    /// The actual ciphertext.
    /// 'Hidden' behind a RefCell because some methods
    /// of the corresponding `ServerKey` (in concrete-shortint)
    /// require the ciphertext to be a `&mut`, while we only have a `&` in rust operators.
    pub(super) ciphertext: RefCell<Ciphertext>,
    _marker: std::marker::PhantomData<P>,
}

impl<P> GenericShortInt<P>
where
    P: ShortIntegerParameter,
{
    /// Minimum value this type can hold, always 0.
    pub const MIN: u8 = 0;

    /// Maximum value this type can hold.
    pub const MAX: u8 = (1 << P::MESSAGE_BIT_SIZE) - 1;

    pub const MODULUS: u8 = (1 << P::MESSAGE_BIT_SIZE);
}

impl<P> FheNumberConstant for GenericShortInt<P>
where
    P: ShortIntegerParameter,
{
    const MIN: u64 = 0;

    const MAX: u64 = Self::MAX as u64;

    const MODULUS: u64 = Self::MODULUS as u64;
}

impl<P> GenericShortInt<P>
where
    P: ShortIntegerParameter,
    ShortIntegerServerKey<P>: WithGlobalKey,
{
    pub fn lt(&self, other: &Self) -> Self {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|server_key| {
            server_key.smart_less(self, other)
        })
    }

    pub fn le(&self, other: &Self) -> Self {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|server_key| {
            server_key.smart_less_or_equal(self, other)
        })
    }

    pub fn gt(&self, other: &Self) -> Self {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|server_key| {
            server_key.smart_greater(self, other)
        })
    }

    pub fn ge(&self, other: &Self) -> Self {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|server_key| {
            server_key.smart_greater_or_equal(self, other)
        })
    }

    pub fn eq(&self, other: &Self) -> Self {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|server_key| {
            server_key.smart_equal(self, other)
        })
    }

    pub fn bivariate_pbs<F>(&self, other: &Self, func: F) -> Self
    where
        F: Fn(u8, u8) -> u8,
    {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|server_key| {
            server_key.bivariate_bps(self, other, func)
        })
    }
}

impl<P> FheBootstrap for GenericShortInt<P>
where
    P: ShortIntegerParameter,
    ShortIntegerServerKey<P>: WithGlobalKey,
{
    fn map<F>(&self, func: F) -> Self
    where
        F: Fn(u64) -> u64,
    {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|key| key.bootstrap_with(self, func))
    }

    fn apply<F>(&mut self, func: F)
    where
        F: Fn(u64) -> u64,
    {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|key| {
            key.bootstrap_inplace_with(self, func);
        })
    }
}

impl<P: ShortIntegerParameter> From<Ciphertext> for GenericShortInt<P> {
    fn from(ciphertext: Ciphertext) -> Self {
        Self {
            ciphertext: RefCell::new(ciphertext),
            _marker: Default::default(),
        }
    }
}

impl<P, T> FheTryEncrypt<T> for GenericShortInt<P>
where
    T: TryInto<u8>,
    P: ShortIntegerParameter,
    ShortIntegerClientKey<P>: RefKeyFromKeyChain,
{
    type Error = OutOfRangeError;

    /// Try to create a new value.
    ///
    /// As Shortints exposed by this crate have between 2 and 7 bits,
    /// creating a value from a rust `u8` may not be possible.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "shortints")]
    /// # {
    /// # use concrete::{ConfigBuilder, FheUint3, generate_keys, set_server_key};
    /// # let config = ConfigBuilder::all_disabled().enable_default_uint3().build();
    /// # let (client_key, server_key) = generate_keys(config);
    /// # set_server_key(server_key);
    /// use concrete::prelude::*;
    /// use concrete::OutOfRangeError;
    ///
    /// // The maximum value that can be represented with 3 bits is 7.
    /// let a = FheUint3::try_encrypt(8, &client_key);
    /// assert_eq!(a.is_err(), true);
    ///
    /// let a = FheUint3::try_encrypt(7, &client_key);
    /// assert_eq!(a.is_ok(), true);
    /// # }
    /// ```
    #[track_caller]
    fn try_encrypt(value: T, key: &ClientKey) -> Result<Self, Self::Error> {
        let value = value.try_into().map_err(|_err| OutOfRangeError)?;
        if value > Self::MAX {
            Err(OutOfRangeError)
        } else {
            let key = ShortIntegerClientKey::<P>::unwrapped_ref_key(key);
            Ok(key.encrypt(value))
        }
    }
}

impl<P> FheDecrypt<u8> for GenericShortInt<P>
where
    P: ShortIntegerParameter,
    ShortIntegerClientKey<P>: RefKeyFromKeyChain,
{
    /// Decrypt the encrypted value to a u8
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "shortints")]
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use concrete::{ConfigBuilder, FheUint3, FheUint2, generate_keys, set_server_key};
    /// # let config = ConfigBuilder::all_disabled().enable_default_uint3().enable_default_uint2().build();
    /// # let (client_key, server_key) = generate_keys(config);
    /// # set_server_key(server_key);
    /// use concrete::OutOfRangeError;
    /// use concrete::prelude::*;
    ///
    /// let a = FheUint2::try_encrypt(2, &client_key)?;
    /// let a_clear = a.decrypt(&client_key);
    /// assert_eq!(a_clear, 2);
    ///
    /// let a = FheUint3::try_encrypt(7, &client_key)?;
    /// let a_clear = a.decrypt(&client_key);
    /// assert_eq!(a_clear, 7);
    /// # Ok(())
    /// # }
    /// ```
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> u8 {
        let key = ShortIntegerClientKey::<P>::unwrapped_ref_key(key);
        key.decrypt(self) as u8
    }
}

macro_rules! short_int_impl_operation(
    ($trait_name:ident($trait_method:ident, $op:tt) => $key_method:ident) => {
        #[doc = concat!(" Allows using the `", stringify!($op), "` operator between a")]
        #[doc = " `GenericFheUint` and a `GenericFheUint` or a `&GenericFheUint`"]
        #[doc = " "]
        #[doc = " # Examples "]
        #[doc = " "]
        #[doc = " ```"]
        #[doc = " # fn main() -> Result<(), concrete::OutOfRangeError> {"]
        #[doc = " use concrete::prelude::*;"]
        #[doc = " use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint2};"]
        #[doc = " "]
        #[doc = " let config = ConfigBuilder::all_disabled()"]
        #[doc = "     .enable_default_uint2()"]
        #[doc = "     .build();"]
        #[doc = " let (keys, server_key) = generate_keys(config);"]
        #[doc = " "]
        #[doc = " let a = FheUint2::try_encrypt(2, &keys)?;"]
        #[doc = " let b = FheUint2::try_encrypt(1, &keys)?;"]
        #[doc = " "]
        #[doc = " set_server_key(server_key);"]
        #[doc = " "]
        #[doc = concat!(" let c = a ", stringify!($op), " b;")]
        #[doc = " let decrypted = c.decrypt(&keys);"]
        #[doc = concat!(" let expected = 2 ", stringify!($op), " 1;")]
        #[doc = " assert_eq!(decrypted, expected);"]
        #[doc = " # Ok(())"]
        #[doc = " # }"]
        #[doc = " ```"]
        #[doc = " "]
        #[doc = " "]
        #[doc = " ```"]
        #[doc = " # fn main() -> Result<(), concrete::OutOfRangeError> {"]
        #[doc = " use concrete::prelude::*;"]
        #[doc = " use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint2};"]
        #[doc = " "]
        #[doc = " let config = ConfigBuilder::all_disabled()"]
        #[doc = "     .enable_default_uint2()"]
        #[doc = "     .build();"]
        #[doc = " let (keys, server_key) = generate_keys(config);"]
        #[doc = " "]
        #[doc = " let a = FheUint2::try_encrypt(2, &keys)?;"]
        #[doc = " let b = FheUint2::try_encrypt(1, &keys)?;"]
        #[doc = " "]
        #[doc = " set_server_key(server_key);"]
        #[doc = " "]
        #[doc = concat!(" let c = a ", stringify!($op), " &b;")]
        #[doc = " let decrypted = c.decrypt(&keys);"]
        #[doc = concat!(" let expected = 2 ", stringify!($op), " 1;")]
        #[doc = " assert_eq!(decrypted, expected);"]
        #[doc = " # Ok(())"]
        #[doc = " # }"]
        #[doc = " ```"]
        impl<P, I> $trait_name<I> for GenericShortInt<P>
        where
            P: ShortIntegerParameter,
            ShortIntegerServerKey<P>: WithGlobalKey,
            I: Borrow<Self>,
        {
            type Output = Self;

            fn $trait_method(self, rhs: I) -> Self::Output {
                ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|key| {
                    key.$key_method(&self, rhs.borrow())
                })
            }
        }

        #[doc = concat!(" Allows using the `", stringify!($op), "` operator between a")]
        #[doc = " `&GenericFheUint` and a `GenericFheUint` or a `&GenericFheUint`"]
        #[doc = " "]
        #[doc = " # Examples "]
        #[doc = " "]
        #[doc = " ```"]
        #[doc = " # fn main() -> Result<(), concrete::OutOfRangeError> {"]
        #[doc = " use concrete::prelude::*;"]
        #[doc = " use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint2};"]
        #[doc = " "]
        #[doc = " let config = ConfigBuilder::all_disabled()"]
        #[doc = "     .enable_default_uint2()"]
        #[doc = "     .build();"]
        #[doc = " let (keys, server_key) = generate_keys(config);"]
        #[doc = " "]
        #[doc = " let a = FheUint2::try_encrypt(2, &keys)?;"]
        #[doc = " let b = FheUint2::try_encrypt(1, &keys)?;"]
        #[doc = " "]
        #[doc = " set_server_key(server_key);"]
        #[doc = " "]
        #[doc = concat!(" let c = &a ", stringify!($op), " b;")]
        #[doc = " let decrypted = c.decrypt(&keys);"]
        #[doc = concat!(" let expected = 2 ", stringify!($op), " 1;")]
        #[doc = " assert_eq!(decrypted, expected);"]
        #[doc = " # Ok(())"]
        #[doc = " # }"]
        #[doc = " ```"]
        #[doc = " "]
        #[doc = " "]
        #[doc = " ```"]
        #[doc = " # fn main() -> Result<(), concrete::OutOfRangeError> {"]
        #[doc = " use concrete::prelude::*;"]
        #[doc = " use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint2};"]
        #[doc = " "]
        #[doc = " let config = ConfigBuilder::all_disabled()"]
        #[doc = "     .enable_default_uint2()"]
        #[doc = "     .build();"]
        #[doc = " let (keys, server_key) = generate_keys(config);"]
        #[doc = " "]
        #[doc = " let a = FheUint2::try_encrypt(2, &keys)?;"]
        #[doc = " let b = FheUint2::try_encrypt(1, &keys)?;"]
        #[doc = " "]
        #[doc = " set_server_key(server_key);"]
        #[doc = " "]
        #[doc = concat!(" let c = &a ", stringify!($op), " &b;")]
        #[doc = " let decrypted = c.decrypt(&keys);"]
        #[doc = concat!(" let expected = 2 ", stringify!($op), " 1;")]
        #[doc = " assert_eq!(decrypted, expected);"]
        #[doc = " # Ok(())"]
        #[doc = " # }"]
        #[doc = " ```"]
        impl<P, I> $trait_name<I> for &GenericShortInt<P>
        where
            P: ShortIntegerParameter,
            ShortIntegerServerKey<P>: WithGlobalKey,
            I: Borrow<GenericShortInt<P>>,
        {
            type Output = GenericShortInt<P>;

            fn $trait_method(self, rhs: I) -> Self::Output {
                ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|key| {
                    key.$key_method(self, rhs.borrow())
                })
            }
        }
    };
);

macro_rules! short_int_impl_operation_assign (
    ($trait_name:ident($trait_method:ident, $op:tt) => $key_method:ident) => {
        #[doc = concat!(" Allows using the `", stringify!($op), "` operator between a")]
        #[doc = " `GenericFheUint` and a `GenericFheUint` or a `&GenericFheUint`"]
        #[doc = " "]
        #[doc = " # Examples "]
        #[doc = " "]
        #[doc = " ```"]
        #[doc = " # fn main() -> Result<(), concrete::OutOfRangeError> {"]
        #[doc = " use concrete::prelude::*;"]
        #[doc = " use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint2};"]
        #[doc = " "]
        #[doc = " let config = ConfigBuilder::all_disabled()"]
        #[doc = "     .enable_default_uint2()"]
        #[doc = "     .build();"]
        #[doc = " let (keys, server_key) = generate_keys(config);"]
        #[doc = " "]
        #[doc = " let mut a = FheUint2::try_encrypt(2, &keys)?;"]
        #[doc = " let b = FheUint2::try_encrypt(1, &keys)?;"]
        #[doc = " "]
        #[doc = " set_server_key(server_key);"]
        #[doc = " "]
        #[doc = concat!(" a ", stringify!($op), " b;")]
        #[doc = " let decrypted = a.decrypt(&keys);"]
        #[doc = " let mut expected = 2;"]
        #[doc = concat!(" expected ", stringify!($op), " 1;")]
        #[doc = " assert_eq!(decrypted, expected);"]
        #[doc = " # Ok(())"]
        #[doc = " # }"]
        #[doc = " ```"]
        #[doc = " "]
        #[doc = " "]
        #[doc = " ```"]
        #[doc = " # fn main() -> Result<(), concrete::OutOfRangeError> {"]
        #[doc = " use concrete::prelude::*;"]
        #[doc = " use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint2};"]
        #[doc = " "]
        #[doc = " let config = ConfigBuilder::all_disabled()"]
        #[doc = "     .enable_default_uint2()"]
        #[doc = "     .build();"]
        #[doc = " let (keys, server_key) = generate_keys(config);"]
        #[doc = " "]
        #[doc = " let mut a = FheUint2::try_encrypt(2, &keys)?;"]
        #[doc = " let b = FheUint2::try_encrypt(1, &keys)?;"]
        #[doc = " "]
        #[doc = " set_server_key(server_key);"]
        #[doc = " "]
        #[doc = concat!(" a ", stringify!($op), " &b;")]
        #[doc = " let decrypted = a.decrypt(&keys);"]
        #[doc = " let mut expected = 2;"]
        #[doc = concat!(" expected ", stringify!($op), " 1;")]
        #[doc = " assert_eq!(decrypted, expected);"]
        #[doc = " # Ok(())"]
        #[doc = " # }"]
        #[doc = " ```"]
        impl<P, I> $trait_name<I> for GenericShortInt<P>
        where
            P: ShortIntegerParameter,
            ShortIntegerServerKey<P>: WithGlobalKey,
            I: Borrow<Self>,
        {
            fn $trait_method(&mut self, rhs: I) {
                ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|key| {
                    key.$key_method(&self, rhs.borrow())
                })
            }
        }
    }
);

// Scalar operations
macro_rules! short_int_impl_scalar_operation {
    ($trait_name:ident($trait_method:ident) => $key_method:ident) => {
        impl<P> $trait_name<u8> for &GenericShortInt<P>
        where
            P: ShortIntegerParameter,
            ShortIntegerServerKey<P>: WithGlobalKey,
        {
            type Output = GenericShortInt<P>;

            fn $trait_method(self, rhs: u8) -> Self::Output {
                ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|key| {
                    key.$key_method(self, rhs)
                })
            }
        }

        impl<P> $trait_name<u8> for GenericShortInt<P>
        where
            P: ShortIntegerParameter,
            ShortIntegerServerKey<P>: WithGlobalKey,
        {
            type Output = GenericShortInt<P>;

            fn $trait_method(self, rhs: u8) -> Self::Output {
                <&Self as $trait_name<u8>>::$trait_method(&self, rhs)
            }
        }

        impl<P> $trait_name<&GenericShortInt<P>> for u8
        where
            P: ShortIntegerParameter,
            ShortIntegerServerKey<P>: WithGlobalKey,
        {
            type Output = GenericShortInt<P>;

            fn $trait_method(self, rhs: &GenericShortInt<P>) -> Self::Output {
                <&GenericShortInt<P> as $trait_name<u8>>::$trait_method(rhs, self)
            }
        }

        impl<P> $trait_name<GenericShortInt<P>> for u8
        where
            P: ShortIntegerParameter,
            ShortIntegerServerKey<P>: WithGlobalKey,
        {
            type Output = GenericShortInt<P>;

            fn $trait_method(self, rhs: GenericShortInt<P>) -> Self::Output {
                <Self as $trait_name<&GenericShortInt<P>>>::$trait_method(self, &rhs)
            }
        }
    };
}

macro_rules! short_int_impl_scalar_operation_assign {
    ($trait_name:ident($trait_method:ident) => $key_method:ident) => {
        impl<P> $trait_name<u8> for GenericShortInt<P>
        where
            P: ShortIntegerParameter,
            ShortIntegerServerKey<P>: WithGlobalKey,
        {
            fn $trait_method(&mut self, rhs: u8) {
                ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|key| {
                    key.$key_method(self, rhs)
                })
            }
        }
    };
}

impl<P> Neg for GenericShortInt<P>
where
    P: ShortIntegerParameter,
    ShortIntegerServerKey<P>: WithGlobalKey,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|key| key.smart_neg(&self))
    }
}

impl<P> Neg for &GenericShortInt<P>
where
    P: ShortIntegerParameter,
    ShortIntegerServerKey<P>: WithGlobalKey,
{
    type Output = GenericShortInt<P>;

    fn neg(self) -> Self::Output {
        ShortIntegerServerKey::<P>::with_unwrapped_global_mut(|key| key.smart_neg(self))
    }
}

short_int_impl_operation!(Add(add,+) => smart_add);
short_int_impl_operation!(Sub(sub,-) => smart_sub);
short_int_impl_operation!(Mul(mul,*) => smart_mul);
short_int_impl_operation!(Div(div,/) => smart_div);
short_int_impl_operation!(BitAnd(bitand,&) => smart_bitand);
short_int_impl_operation!(BitOr(bitor,|) => smart_bitor);
short_int_impl_operation!(BitXor(bitxor,^) => smart_bitxor);

short_int_impl_operation_assign!(AddAssign(add_assign,+=) => smart_add_assign);
short_int_impl_operation_assign!(SubAssign(sub_assign,-=) => smart_sub_assign);
short_int_impl_operation_assign!(MulAssign(mul_assign,*=) => smart_mul_assign);
short_int_impl_operation_assign!(DivAssign(div_assign,/=) => smart_div_assign);
short_int_impl_operation_assign!(BitAndAssign(bitand_assign,&=) => smart_bitand_assign);
short_int_impl_operation_assign!(BitOrAssign(bitor_assign,|=) => smart_bitor_assign);
short_int_impl_operation_assign!(BitXorAssign(bitxor_assign,^=) => smart_bitxor_assign);

short_int_impl_scalar_operation!(Add(add) => smart_scalar_add);
short_int_impl_scalar_operation!(Sub(sub) => smart_scalar_sub);
short_int_impl_scalar_operation!(Mul(mul) => smart_scalar_mul);
short_int_impl_scalar_operation!(Div(div) => unchecked_scalar_div);
short_int_impl_scalar_operation!(Rem(rem) => unchecked_scalar_mod);
short_int_impl_scalar_operation!(Shl(shl) => smart_scalar_left_shift);
short_int_impl_scalar_operation!(Shr(shr) => unchecked_scalar_right_shift);

short_int_impl_scalar_operation_assign!(AddAssign(add_assign) => smart_scalar_add_assign);
short_int_impl_scalar_operation_assign!(SubAssign(sub_assign) => smart_scalar_sub_assign);
short_int_impl_scalar_operation_assign!(MulAssign(mul_assign) => smart_scalar_mul_assign);
