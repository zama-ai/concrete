use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::integers::parameters::EvaluationIntegerKey;

use super::client_key::GenericIntegerClientKey;
use super::parameters::IntegerParameter;

use concrete_integer::wopbs::WopbsKey;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct GenericIntegerServerKey<P: IntegerParameter> {
    pub(in crate::integers) inner: P::InnerServerKey,
    pub(in crate::integers) wopbs_key: WopbsKey,
    _marker: PhantomData<P>,
}

impl<P> GenericIntegerServerKey<P>
where
    P: IntegerParameter,
    P::InnerServerKey: EvaluationIntegerKey<P::InnerClientKey>,
{
    pub(super) fn new(client_key: &GenericIntegerClientKey<P>) -> Self {
        let inner = P::InnerServerKey::new(&client_key.inner);
        let wopbs_key = P::InnerServerKey::new_wopbs_key(&client_key.inner, &inner);
        Self {
            inner,
            wopbs_key,
            _marker: Default::default(),
        }
    }
}

pub(super) trait SmartNeg<Ciphertext> {
    type Output;
    fn smart_neg(&self, lhs: Ciphertext) -> Self::Output;
}

macro_rules! define_smart_server_key_op {
    ($op_name:ident) => {
        paste::paste! {
            pub(super) trait [< Smart $op_name >]<Lhs, Rhs> {
                type Output;

                fn [< smart_ $op_name:lower >](
                    &self,
                    lhs: Lhs,
                    rhs: Rhs,
                ) -> Self::Output;
            }

            pub(super) trait [< Smart $op_name Assign >]<Lhs, Rhs> {
                fn [< smart_ $op_name:lower _assign >](
                    &self,
                    lhs: &mut Lhs,
                    rhs: Rhs,
                );
            }
        }
    };
    ($($op:ident),*) => {
        $(
            define_smart_server_key_op!($op);
        )*
    };
}

define_smart_server_key_op!(Add, Sub, Mul, BitAnd, BitOr, BitXor, Shl, Shr);

macro_rules! impl_smart_op_for_concrete_integer_server_key {
    ($smart_trait:ident($smart_trait_fn:ident) => ($ciphertext:ty, $method:ident)) => {
        impl $smart_trait<&mut $ciphertext, &mut $ciphertext> for concrete_integer::ServerKey {
            type Output = $ciphertext;

            fn $smart_trait_fn(
                &self,
                lhs: &mut $ciphertext,
                rhs: &mut $ciphertext,
            ) -> Self::Output {
                self.$method(lhs, rhs)
            }
        }
    };
}

macro_rules! impl_smart_assign_op_for_concrete_integer_server_key {
    ($smart_trait:ident($smart_trait_fn:ident) => ($ciphertext:ty, $method:ident)) => {
        impl $smart_trait<$ciphertext, &mut $ciphertext> for concrete_integer::ServerKey {
            fn $smart_trait_fn(&self, lhs: &mut $ciphertext, rhs: &mut $ciphertext) {
                self.$method(lhs, rhs);
            }
        }
    };
}

macro_rules! impl_smart_scalar_op_for_concrete_integer_server_key {
    ($smart_trait:ident($smart_trait_fn:ident) => ($ciphertext:ty, $method:ident)) => {
        impl $smart_trait<&mut $ciphertext, u64> for concrete_integer::ServerKey {
            type Output = $ciphertext;

            fn $smart_trait_fn(&self, lhs: &mut $ciphertext, rhs: u64) -> Self::Output {
                self.$method(lhs, rhs.try_into().unwrap())
            }
        }
    };
}

macro_rules! impl_smart_scalar_assign_op_for_concrete_integer_server_key {
    ($smart_trait:ident($smart_trait_fn:ident) => ($ciphertext:ty, $method:ident)) => {
        impl $smart_trait<$ciphertext, u64> for concrete_integer::ServerKey {
            fn $smart_trait_fn(&self, lhs: &mut $ciphertext, rhs: u64) {
                // TODO fix this
                self.$method(lhs, rhs.try_into().unwrap());
            }
        }
    };
}

impl SmartNeg<&mut concrete_integer::RadixCiphertext> for concrete_integer::ServerKey {
    type Output = concrete_integer::RadixCiphertext;
    fn smart_neg(&self, lhs: &mut concrete_integer::RadixCiphertext) -> Self::Output {
        self.smart_neg(lhs)
    }
}

impl_smart_op_for_concrete_integer_server_key!(SmartAdd(smart_add) => (concrete_integer::RadixCiphertext, smart_add_parallelized));
impl_smart_op_for_concrete_integer_server_key!(SmartSub(smart_sub) => (concrete_integer::RadixCiphertext, smart_sub_parallelized));
impl_smart_op_for_concrete_integer_server_key!(SmartMul(smart_mul) => (concrete_integer::RadixCiphertext, smart_mul_parallelized));
impl_smart_op_for_concrete_integer_server_key!(SmartBitAnd(smart_bitand) => (concrete_integer::RadixCiphertext, smart_bitand_parallelized));
impl_smart_op_for_concrete_integer_server_key!(SmartBitOr(smart_bitor) => (concrete_integer::RadixCiphertext, smart_bitor_parallelized));
impl_smart_op_for_concrete_integer_server_key!(SmartBitXor(smart_bitxor) => (concrete_integer::RadixCiphertext, smart_bitxor_parallelized));

impl_smart_assign_op_for_concrete_integer_server_key!(SmartAddAssign(smart_add_assign) => (concrete_integer::RadixCiphertext, smart_add_assign_parallelized));
impl_smart_assign_op_for_concrete_integer_server_key!(SmartSubAssign(smart_sub_assign) => (concrete_integer::RadixCiphertext, smart_sub_assign_parallelized));
impl_smart_assign_op_for_concrete_integer_server_key!(SmartMulAssign(smart_mul_assign) => (concrete_integer::RadixCiphertext, smart_mul_assign_parallelized));
impl_smart_assign_op_for_concrete_integer_server_key!(SmartBitAndAssign(smart_bitand_assign) => (concrete_integer::RadixCiphertext, smart_bitand_assign_parallelized));
impl_smart_assign_op_for_concrete_integer_server_key!(SmartBitOrAssign(smart_bitor_assign) => (concrete_integer::RadixCiphertext, smart_bitor_assign_parallelized));
impl_smart_assign_op_for_concrete_integer_server_key!(SmartBitXorAssign(smart_bitxor_assign) => (concrete_integer::RadixCiphertext, smart_bitxor_assign_parallelized));

impl_smart_scalar_op_for_concrete_integer_server_key!(SmartAdd(smart_add) => (concrete_integer::RadixCiphertext, smart_scalar_add_parallelized));
impl_smart_scalar_op_for_concrete_integer_server_key!(SmartSub(smart_sub) => (concrete_integer::RadixCiphertext, smart_scalar_sub_parallelized));
impl_smart_scalar_op_for_concrete_integer_server_key!(SmartMul(smart_mul) => (concrete_integer::RadixCiphertext, smart_scalar_mul_parallelized));
impl_smart_scalar_op_for_concrete_integer_server_key!(SmartShl(smart_shl) => (concrete_integer::RadixCiphertext, unchecked_scalar_left_shift_parallelized));
impl_smart_scalar_op_for_concrete_integer_server_key!(SmartShr(smart_shr) => (concrete_integer::RadixCiphertext, unchecked_scalar_right_shift_parallelized));

impl_smart_scalar_assign_op_for_concrete_integer_server_key!(SmartAddAssign(smart_add_assign) => (concrete_integer::RadixCiphertext, smart_scalar_add_assign_parallelized));
impl_smart_scalar_assign_op_for_concrete_integer_server_key!(SmartSubAssign(smart_sub_assign) => (concrete_integer::RadixCiphertext, smart_scalar_sub_assign_parallelized));
impl_smart_scalar_assign_op_for_concrete_integer_server_key!(SmartMulAssign(smart_mul_assign) => (concrete_integer::RadixCiphertext, smart_scalar_mul_assign_parallelized));
impl_smart_scalar_assign_op_for_concrete_integer_server_key!(SmartShlAssign(smart_shl_assign) => (concrete_integer::RadixCiphertext, unchecked_scalar_left_shift_assign_parallelized));
impl_smart_scalar_assign_op_for_concrete_integer_server_key!(SmartShrAssign(smart_shr_assign) => (concrete_integer::RadixCiphertext, unchecked_scalar_right_shift_assign_parallelized));
