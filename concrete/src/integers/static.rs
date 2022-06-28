use concrete_integer::ciphertext::Ciphertext;
use concrete_integer::client_key::ClientKey as IntegerClientKey;
use concrete_integer::server_key::ServerKey;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, MulAssign,
    Neg, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey};
use crate::global_state::WithGlobalKey;
use crate::keys::RefKeyFromKeyChain;
use crate::shortints::ShortIntegerParameter;
use crate::traits::{FheDecrypt, FheEncrypt, FheTryEncrypt};
use crate::{ClientKey, FheUint2Parameters, OutOfRangeError};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::FheNumberConstant;
#[cfg(feature = "internal-keycache")]
use concrete_integer::keycache::KEY_CACHE;

/// A Generic FHE unsigned integer
///
/// The current representation for large integers is done by combining
/// multiple short integers of the same size.
///
/// The type parameter `P` is the type of shortint parameters.
///
/// `N` is the number of block of such shortints.
///
/// For example with both:
/// - P: [FheUint2Parameters]
/// - N: `4`
///
/// We will combine 4 shortint with 2 bits, to have a  `4 * 2 = 8` bit integer.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, doc(cfg(feature = "integers")))]
#[derive(Clone)]
pub struct GenericInteger<P: ShortIntegerParameter, const N: usize> {
    ciphertext: RefCell<Ciphertext>,
    _marker: PhantomData<P>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct GenericIntegerClientKey<P: ShortIntegerParameter, const N: usize> {
    key: IntegerClientKey,
    _marker: PhantomData<P>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub(super) struct GenericIntegerServerKey<P: ShortIntegerParameter, const N: usize> {
    key: ServerKey,
    _marker: PhantomData<P>,
}

macro_rules! static_int_type {
    (
        $(#[$outer:meta])*
        $name:ident {
            parameters: $parameters:ty,
            num_block: $num_block:expr,
            client_key_name: $client_key_name:ident,
            server_key_name: $server_key_name:ident,
            keychain_member: $($member:ident).*,
            type_variant: $enum_variant:expr,
        }
    ) => {
        pub(super) type $client_key_name = GenericIntegerClientKey<$parameters, $num_block>;
        pub(super) type $server_key_name = GenericIntegerServerKey<$parameters, $num_block>;

        $(#[$outer])*
        #[cfg_attr(doc, cfg(feature = "integers"))]
        pub type $name = GenericInteger<$parameters, $num_block>;

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

static_int_type! {
    #[doc="An unsigned integer type with 8 bits."]
    FheUint8 {
        parameters: FheUint2Parameters,
        num_block: 4,
        client_key_name: FheUint8ClientKey,
        server_key_name: FheUint8ServerKey,
        keychain_member: integer_key.uint8_key,
        type_variant: Type::FheUint8,
    }
}

static_int_type! {
    #[doc="An unsigned integer type with 12 bits."]
    FheUint12 {
        parameters: FheUint2Parameters,
        num_block: 6,
        client_key_name: FheUint12ClientKey,
        server_key_name: FheUint12ServerKey,
        keychain_member: integer_key.uint12_key,
        type_variant: Type::FheUint12,
    }
}

static_int_type! {
    #[doc="An unsigned integer type with 16 bits."]
    FheUint16 {
        parameters: FheUint2Parameters,
        num_block: 8,
        client_key_name: FheUint16ClientKey,
        server_key_name: FheUint16ServerKey,
        keychain_member: integer_key.uint16_key,
        type_variant: Type::FheUint16,
    }
}

impl<P, const N: usize> GenericIntegerClientKey<P, N>
where
    P: ShortIntegerParameter,
{
    fn encrypt(&self, value: u64) -> GenericInteger<P, N> {
        self.key.encrypt(value).into()
    }

    fn decrypt(&self, value: &GenericInteger<P, N>) -> u64 {
        self.key.decrypt(&value.ciphertext.borrow())
    }
}

impl<P, const N: usize> From<P> for GenericIntegerClientKey<P, N>
where
    P: ShortIntegerParameter,
{
    fn from(params: P) -> Self {
        #[cfg(feature = "internal-keycache")]
        let key = KEY_CACHE
            .get_from_params(params.into(), concrete_integer::client_key::VecLength(N))
            .0;
        #[cfg(not(feature = "internal-keycache"))]
        let key = IntegerClientKey::new(params.into(), N);
        Self {
            key,
            _marker: Default::default(),
        }
    }
}

impl<P, const N: usize> GenericIntegerServerKey<P, N>
where
    P: ShortIntegerParameter,
{
    pub(super) fn new(client_key: &GenericIntegerClientKey<P, N>) -> Self {
        #[cfg(feature = "internal-keycache")]
        let key = KEY_CACHE
            .get_from_params(
                client_key.key.parameters(),
                concrete_integer::client_key::VecLength(N),
            )
            .1;
        #[cfg(not(feature = "internal-keycache"))]
        let key = ServerKey::new(&client_key.key);
        Self {
            key,
            _marker: Default::default(),
        }
    }

    fn smart_neg(&self, lhs: &GenericInteger<P, N>) -> GenericInteger<P, N> {
        self.key.smart_neg(&mut lhs.ciphertext.borrow_mut()).into()
    }

    fn smart_add(
        &self,
        lhs: &GenericInteger<P, N>,
        rhs: &GenericInteger<P, N>,
    ) -> GenericInteger<P, N> {
        self.key
            .smart_add(
                &mut lhs.ciphertext.borrow_mut(),
                &mut rhs.ciphertext.borrow_mut(),
            )
            .into()
    }

    fn smart_sub(
        &self,
        lhs: &GenericInteger<P, N>,
        rhs: &GenericInteger<P, N>,
    ) -> GenericInteger<P, N> {
        self.key
            .smart_sub(
                &mut lhs.ciphertext.borrow_mut(),
                &mut rhs.ciphertext.borrow_mut(),
            )
            .into()
    }

    fn smart_mul(
        &self,
        lhs: &GenericInteger<P, N>,
        rhs: &GenericInteger<P, N>,
    ) -> GenericInteger<P, N> {
        self.key
            .smart_mul(
                &mut lhs.ciphertext.borrow_mut(),
                &mut rhs.ciphertext.borrow_mut(),
            )
            .into()
    }

    fn smart_bitand(
        &self,
        lhs: &GenericInteger<P, N>,
        rhs: &GenericInteger<P, N>,
    ) -> GenericInteger<P, N> {
        self.key
            .smart_bitand(
                &mut lhs.ciphertext.borrow_mut(),
                &mut rhs.ciphertext.borrow_mut(),
            )
            .into()
    }

    fn smart_bitor(
        &self,
        lhs: &GenericInteger<P, N>,
        rhs: &GenericInteger<P, N>,
    ) -> GenericInteger<P, N> {
        self.key
            .smart_bitor(
                &mut lhs.ciphertext.borrow_mut(),
                &mut rhs.ciphertext.borrow_mut(),
            )
            .into()
    }

    fn smart_bitxor(
        &self,
        lhs: &GenericInteger<P, N>,
        rhs: &GenericInteger<P, N>,
    ) -> GenericInteger<P, N> {
        self.key
            .smart_bitxor(
                &mut lhs.ciphertext.borrow_mut(),
                &mut rhs.ciphertext.borrow_mut(),
            )
            .into()
    }

    fn smart_add_assign(&self, lhs: &GenericInteger<P, N>, rhs: &GenericInteger<P, N>) {
        self.key.smart_add_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_sub_assign(&self, lhs: &GenericInteger<P, N>, rhs: &GenericInteger<P, N>) {
        self.key.smart_sub_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_mul_assign(&self, lhs: &GenericInteger<P, N>, rhs: &GenericInteger<P, N>) {
        self.key.smart_mul_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_bitand_assign(&self, lhs: &GenericInteger<P, N>, rhs: &GenericInteger<P, N>) {
        self.key.smart_bitand_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_bitor_assign(&self, lhs: &GenericInteger<P, N>, rhs: &GenericInteger<P, N>) {
        self.key.smart_bitor(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_bitxor_assign(&self, lhs: &GenericInteger<P, N>, rhs: &GenericInteger<P, N>) {
        self.key.smart_bitxor_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    fn smart_scalar_add(&self, lhs: &GenericInteger<P, N>, rhs: u64) -> GenericInteger<P, N> {
        self.key
            .smart_scalar_add(&mut lhs.ciphertext.borrow_mut(), rhs)
            .into()
    }

    fn smart_scalar_sub(&self, lhs: &GenericInteger<P, N>, rhs: u64) -> GenericInteger<P, N> {
        self.key
            .smart_scalar_sub(&mut lhs.ciphertext.borrow_mut(), rhs)
            .into()
    }

    fn smart_scalar_mul(&self, lhs: &GenericInteger<P, N>, rhs: u64) -> GenericInteger<P, N> {
        self.key
            .smart_scalar_mul(&mut lhs.ciphertext.borrow_mut(), rhs)
            .into()
    }

    fn unchecked_scalar_left_shift(
        &self,
        lhs: &GenericInteger<P, N>,
        rhs: u64,
    ) -> GenericInteger<P, N> {
        self.key
            .unchecked_scalar_left_shift(&lhs.ciphertext.borrow(), rhs as usize)
            .into()
    }

    fn unchecked_scalar_right_shift(
        &self,
        lhs: &GenericInteger<P, N>,
        rhs: u64,
    ) -> GenericInteger<P, N> {
        self.key
            .unchecked_scalar_right_shift(&lhs.ciphertext.borrow(), rhs as usize)
            .into()
    }

    fn smart_scalar_add_assign(&self, lhs: &GenericInteger<P, N>, rhs: u64) {
        self.key
            .smart_scalar_add_assign(&mut lhs.ciphertext.borrow_mut(), rhs);
    }

    fn smart_scalar_sub_assign(&self, lhs: &GenericInteger<P, N>, rhs: u64) {
        self.key
            .smart_scalar_sub_assign(&mut lhs.ciphertext.borrow_mut(), rhs);
    }

    fn smart_scalar_mul_assign(&self, lhs: &GenericInteger<P, N>, rhs: u64) {
        self.key
            .smart_scalar_mul_assign(&mut lhs.ciphertext.borrow_mut(), rhs);
    }

    fn unchecked_scalar_left_shift_assign(&self, lhs: &GenericInteger<P, N>, rhs: u64) {
        self.key
            .unchecked_scalar_left_shift_assign(&mut lhs.ciphertext.borrow_mut(), rhs as usize);
    }

    fn unchecked_scalar_right_shift_assign(&self, lhs: &GenericInteger<P, N>, rhs: u64) {
        self.key
            .unchecked_scalar_right_shift_assign(&mut lhs.ciphertext.borrow_mut(), rhs as usize);
    }
}

impl<P, const N: usize> From<Ciphertext> for GenericInteger<P, N>
where
    P: ShortIntegerParameter,
{
    fn from(ciphertext: Ciphertext) -> Self {
        Self {
            ciphertext: RefCell::new(ciphertext),
            _marker: Default::default(),
        }
    }
}

impl<P, const N: usize> GenericInteger<P, N>
where
    P: ShortIntegerParameter,
{
    pub const MIN: u64 = 0;
    pub const MAX: u64 = (1 << (P::MESSAGE_BIT_SIZE as u64 * N as u64)) - 1;

    pub const MODULUS: u64 = (1 << (P::MESSAGE_BIT_SIZE as u64 * N as u64));
}

impl<P, const N: usize> FheNumberConstant for GenericInteger<P, N>
where
    P: ShortIntegerParameter,
{
    const MIN: u64 = Self::MIN;
    const MAX: u64 = Self::MAX;
    const MODULUS: u64 = Self::MODULUS;
}

impl FheEncrypt<u8> for FheUint8 {
    #[track_caller]
    fn encrypt(value: u8, key: &ClientKey) -> Self {
        let key = GenericIntegerClientKey::unwrapped_ref_key(key);
        key.encrypt(value.into())
    }
}

impl FheDecrypt<u8> for FheUint8 {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> u8 {
        let key = GenericIntegerClientKey::unwrapped_ref_key(key);
        key.decrypt(self) as u8
    }
}

impl FheEncrypt<u16> for FheUint16 {
    #[track_caller]
    fn encrypt(value: u16, key: &ClientKey) -> Self {
        let key = FheUint16ClientKey::unwrapped_ref_key(key);
        key.encrypt(value.into())
    }
}

impl FheDecrypt<u16> for FheUint16 {
    #[track_caller]
    fn decrypt(&self, key: &ClientKey) -> u16 {
        let key = FheUint16ClientKey::unwrapped_ref_key(key);
        key.decrypt(self) as u16
    }
}

impl<P, const N: usize> FheDecrypt<u64> for GenericInteger<P, N>
where
    P: ShortIntegerParameter,
    GenericIntegerClientKey<P, N>: RefKeyFromKeyChain,
{
    fn decrypt(&self, key: &ClientKey) -> u64 {
        let key = GenericIntegerClientKey::<P, N>::unwrapped_ref_key(key);
        key.decrypt(self)
    }
}

impl<P, T, const N: usize> FheTryEncrypt<T> for GenericInteger<P, N>
where
    T: TryInto<u64>,
    P: ShortIntegerParameter,
    GenericIntegerClientKey<P, N>: RefKeyFromKeyChain,
{
    type Error = OutOfRangeError;

    fn try_encrypt(value: T, key: &ClientKey) -> Result<Self, Self::Error> {
        let value = value.try_into().map_err(|_err| OutOfRangeError)?;
        if value > Self::MAX {
            Err(OutOfRangeError)
        } else {
            let key = GenericIntegerClientKey::<P, N>::unwrapped_ref_key(key);
            Ok(key.encrypt(value))
        }
    }
}

macro_rules! generic_integer_impl_operation (
    ($trait_name:ident($trait_method:ident) => $key_method:ident) => {
        impl<P, B, const N: usize> $trait_name<B> for GenericInteger<P, N>
        where
            P: ShortIntegerParameter,
            B: Borrow<Self>,
            GenericIntegerServerKey<P, N>: WithGlobalKey,
        {
            type Output = Self;

            fn $trait_method(self, rhs: B) -> Self::Output {
                <&Self as $trait_name<B>>::$trait_method(&self, rhs)
            }
        }

        impl<P, B, const N: usize> $trait_name<B> for &GenericInteger<P, N>
        where
            P: ShortIntegerParameter,
            B: Borrow<GenericInteger<P, N>>,
            GenericIntegerServerKey<P, N>: WithGlobalKey,
        {
            type Output = GenericInteger<P, N>;

            fn $trait_method(self, rhs: B) -> Self::Output {
                GenericIntegerServerKey::<P, N>::with_unwrapped_global_mut(|key| {
                    key.$key_method(self, rhs.borrow())
                })
            }
        }
    }
);

macro_rules! generic_integer_impl_operation_assign (
    ($trait_name:ident($trait_method:ident, $op:tt) => $key_method:ident) => {
        impl<P, I, const N: usize> $trait_name<I> for GenericInteger<P, N>
        where
            P: ShortIntegerParameter,
            GenericIntegerServerKey<P, N>: WithGlobalKey,
            I: Borrow<Self>,
        {
            fn $trait_method(&mut self, rhs: I) {
                GenericIntegerServerKey::<P, N>::with_unwrapped_global_mut(|key| {
                    key.$key_method(&self, rhs.borrow())
                })
            }
        }
    }
);

macro_rules! generic_integer_impl_scalar_operation {
    ($trait_name:ident($trait_method:ident) => $key_method:ident($($scalar_type:ty),*)) => {
        $(
            impl<P, const N: usize> $trait_name<$scalar_type> for GenericInteger<P, N>
            where
                P: ShortIntegerParameter,
                GenericIntegerServerKey<P, N>: WithGlobalKey,
            {
                type Output = GenericInteger<P, N>;

                fn $trait_method(self, rhs: $scalar_type) -> Self::Output {
                    <&Self as $trait_name<$scalar_type>>::$trait_method(&self, rhs)
                }
            }

            impl<P, const N: usize> $trait_name<$scalar_type> for &GenericInteger<P, N>
            where
                P: ShortIntegerParameter,
                GenericIntegerServerKey<P, N>: WithGlobalKey,
            {
                type Output = GenericInteger<P, N>;

                fn $trait_method(self, rhs: $scalar_type) -> Self::Output {
                    GenericIntegerServerKey::<P, N>::with_unwrapped_global_mut(|key| {
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
            impl<P, const N: usize> $trait_name<$scalar_type> for GenericInteger<P, N>
                where
                    P: ShortIntegerParameter,
                    GenericIntegerServerKey<P, N>: WithGlobalKey,
            {
                fn $trait_method(&mut self, rhs: $scalar_type) {
                    GenericIntegerServerKey::<P, N>::with_unwrapped_global_mut(|key| {
                        key.$key_method(self, u64::from(rhs))
                    });
                }
            }
        )*
    }
}

impl<P, const N: usize> Neg for GenericInteger<P, N>
where
    P: ShortIntegerParameter,
    GenericIntegerServerKey<P, N>: WithGlobalKey,
{
    type Output = GenericInteger<P, N>;

    fn neg(self) -> Self::Output {
        <&Self as Neg>::neg(&self)
    }
}

impl<P, const N: usize> Neg for &GenericInteger<P, N>
where
    P: ShortIntegerParameter,
    GenericIntegerServerKey<P, N>: WithGlobalKey,
{
    type Output = GenericInteger<P, N>;

    fn neg(self) -> Self::Output {
        GenericIntegerServerKey::<P, N>::with_unwrapped_global_mut(|key| key.smart_neg(self))
    }
}

generic_integer_impl_operation!(Add(add) => smart_add);
generic_integer_impl_operation!(Sub(sub) => smart_sub);
generic_integer_impl_operation!(Mul(mul) => smart_mul);
generic_integer_impl_operation!(BitAnd(bitand) => smart_bitand);
generic_integer_impl_operation!(BitOr(bitor) => smart_bitor);
generic_integer_impl_operation!(BitXor(bitxor) => smart_bitxor);

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
