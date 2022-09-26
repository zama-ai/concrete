use concrete_integer::{CrtCiphertext, CrtClientKey, RadixCiphertext, RadixClientKey, ServerKey};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey};
use crate::global_state::WithGlobalKey;
use crate::integers::client_key::GenericIntegerClientKey;
use crate::integers::parameters::{FromParameters, IntegerParameter, PrivateIntegerKey};
use crate::integers::server_key::{
    GenericIntegerServerKey, SmartAdd, SmartAddAssign, SmartBitAnd, SmartBitAndAssign, SmartBitOr,
    SmartBitOrAssign, SmartBitXor, SmartBitXorAssign, SmartMul, SmartMulAssign, SmartNeg, SmartShl,
    SmartShlAssign, SmartShr, SmartShrAssign, SmartSub, SmartSubAssign,
};
use crate::keys::RefKeyFromKeyChain;
use crate::traits::DynamicFheEncryptor;
use crate::{ClientKey, CrtParameters, GenericInteger, RadixParameters};

/// Parameters for integers
///
/// Integers works by composing multiple shortints.
///
///Two decompositions are supported:
/// - Radix with [RadixParameters]
/// - Crt with [CrtParameters]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub enum DynIntegerParameters {
    Radix(RadixParameters),
    Crt(CrtParameters),
}

impl From<CrtParameters> for DynIntegerParameters {
    fn from(crt_params: CrtParameters) -> Self {
        Self::Crt(crt_params)
    }
}

impl From<RadixParameters> for DynIntegerParameters {
    fn from(radix_params: RadixParameters) -> Self {
        Self::Radix(radix_params)
    }
}

pub type DynInteger = GenericInteger<DynIntegerParameters>;
pub type DynIntegerServerKey = GenericIntegerServerKey<DynIntegerParameters>;
pub type DynIntegerClientKey = GenericIntegerClientKey<DynIntegerParameters>;

/// Id allows to retrieve the key for the dynamic integer
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IntegerTypeId(pub(in crate::integers) usize);

impl RefKeyFromKeyChain for IntegerTypeId {
    type Key = DynIntegerClientKey;

    fn ref_key(self, keys: &ClientKey) -> Result<&Self::Key, UninitializedClientKey> {
        keys.integer_key
            .custom_keys
            .get(self.0)
            .ok_or(UninitializedClientKey(Type::DynamicInteger))
    }
}

impl WithGlobalKey for IntegerTypeId {
    type Key = DynIntegerServerKey;

    fn with_global_mut<R, F>(self, func: F) -> Result<R, UninitializedServerKey>
    where
        F: FnOnce(&mut Self::Key) -> R,
    {
        crate::global_state::with_internal_keys_mut(|key| {
            key.integer_key
                .custom_keys
                .get_mut(self.0)
                .map(func)
                .ok_or(UninitializedServerKey(Type::DynamicInteger))
        })
    }
}

/// A dynamically defined integer type.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub enum DynInnerCiphertext {
    Radix(RadixCiphertext),
    Crt(CrtCiphertext),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub enum DynInnerClientKey {
    Radix(RadixClientKey),
    Crt(CrtClientKey),
}

impl AsRef<concrete_integer::ClientKey> for DynInnerClientKey {
    fn as_ref(&self) -> &concrete_integer::ClientKey {
        match self {
            Self::Crt(key) => key.as_ref(),
            Self::Radix(key) => key.as_ref(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct DynInnerServerKey {
    inner: ServerKey,
}

impl crate::integers::parameters::EvaluationIntegerKey<DynInnerClientKey> for DynInnerServerKey {
    fn new(client_key: &DynInnerClientKey) -> Self {
        let inner = match client_key {
            DynInnerClientKey::Radix(key) => ServerKey::new(key),
            DynInnerClientKey::Crt(key) => ServerKey::new(key),
        };
        Self { inner }
    }
}

impl FromParameters<DynIntegerParameters> for DynInnerClientKey {
    fn from_parameters(parameters: DynIntegerParameters) -> Self {
        match parameters {
            DynIntegerParameters::Radix(radix_params) => {
                Self::Radix(RadixClientKey::from_parameters(radix_params))
            }
            DynIntegerParameters::Crt(crt_params) => {
                Self::Crt(CrtClientKey::from_parameters(crt_params))
            }
        }
    }
}

impl From<DynIntegerParameters> for DynInnerClientKey {
    fn from(_: DynIntegerParameters) -> Self {
        todo!()
    }
}

impl PrivateIntegerKey for DynInnerClientKey {
    type Ciphertext = DynInnerCiphertext;

    fn encrypt(&self, value: u64) -> Self::Ciphertext {
        match self {
            DynInnerClientKey::Radix(key) => DynInnerCiphertext::Radix(key.encrypt(value)),
            DynInnerClientKey::Crt(key) => DynInnerCiphertext::Crt(key.encrypt(value)),
        }
    }

    fn decrypt(&self, ciphertext: &Self::Ciphertext) -> u64 {
        match (self, ciphertext) {
            (DynInnerClientKey::Radix(key), DynInnerCiphertext::Radix(ct)) => key.decrypt(ct),
            (DynInnerClientKey::Crt(key), DynInnerCiphertext::Crt(ct)) => key.decrypt(ct),
            (_, _) => panic!("Mismatch between ciphertext and client key representation"),
        }
    }
}

// Normal Ops

impl SmartNeg<&mut DynInnerCiphertext> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_neg(&self, lhs: &mut DynInnerCiphertext) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => DynInnerCiphertext::Radix(self.inner.smart_neg(lhs)),
            DynInnerCiphertext::Crt(_lhs) => todo!(),
        }
    }
}

impl SmartAdd<&mut DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_add(
        &self,
        lhs: &mut DynInnerCiphertext,
        rhs: &mut DynInnerCiphertext,
    ) -> DynInnerCiphertext {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                DynInnerCiphertext::Radix(self.inner.smart_add(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                DynInnerCiphertext::Crt(self.inner.smart_add_crt(lhs, rhs))
            }
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartSub<&mut DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_sub(
        &self,
        lhs: &mut DynInnerCiphertext,
        rhs: &mut DynInnerCiphertext,
    ) -> DynInnerCiphertext {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                DynInnerCiphertext::Radix(self.inner.smart_sub(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => todo!(),
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartMul<&mut DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_mul(
        &self,
        lhs: &mut DynInnerCiphertext,
        rhs: &mut DynInnerCiphertext,
    ) -> DynInnerCiphertext {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                DynInnerCiphertext::Radix(self.inner.smart_mul(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                DynInnerCiphertext::Crt(self.inner.smart_mul_crt(lhs, rhs))
            }
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartBitAnd<&mut DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_bitand(
        &self,
        lhs: &mut DynInnerCiphertext,
        rhs: &mut DynInnerCiphertext,
    ) -> DynInnerCiphertext {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                DynInnerCiphertext::Radix(self.inner.smart_bitand(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => todo!(),
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartBitOr<&mut DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_bitor(
        &self,
        lhs: &mut DynInnerCiphertext,
        rhs: &mut DynInnerCiphertext,
    ) -> DynInnerCiphertext {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                DynInnerCiphertext::Radix(self.inner.smart_bitor(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => todo!(),
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartBitXor<&mut DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_bitxor(
        &self,
        lhs: &mut DynInnerCiphertext,
        rhs: &mut DynInnerCiphertext,
    ) -> DynInnerCiphertext {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                DynInnerCiphertext::Radix(self.inner.smart_bitxor(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => todo!(),
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

// Assigned Normal Ops
impl SmartAddAssign<DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    fn smart_add_assign(&self, lhs: &mut DynInnerCiphertext, rhs: &mut DynInnerCiphertext) {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                self.inner.smart_add_assign(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                self.inner.smart_add_crt_assign(lhs, rhs)
            }
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartSubAssign<DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    fn smart_sub_assign(&self, lhs: &mut DynInnerCiphertext, rhs: &mut DynInnerCiphertext) {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                self.inner.smart_sub_assign(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => todo!(),
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartMulAssign<DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    fn smart_mul_assign(&self, lhs: &mut DynInnerCiphertext, rhs: &mut DynInnerCiphertext) {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                self.inner.smart_mul_assign(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                self.inner.smart_mul_crt_assign(lhs, rhs)
            }
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartBitAndAssign<DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    fn smart_bitand_assign(&self, lhs: &mut DynInnerCiphertext, rhs: &mut DynInnerCiphertext) {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                self.inner.smart_bitand_assign(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => todo!(),
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartBitOrAssign<DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    fn smart_bitor_assign(&self, lhs: &mut DynInnerCiphertext, rhs: &mut DynInnerCiphertext) {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                self.inner.smart_bitor_assign(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => todo!(),
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

impl SmartBitXorAssign<DynInnerCiphertext, &mut DynInnerCiphertext> for DynInnerServerKey {
    fn smart_bitxor_assign(&self, lhs: &mut DynInnerCiphertext, rhs: &mut DynInnerCiphertext) {
        match (lhs, rhs) {
            (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                self.inner.smart_bitxor_assign(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => todo!(),
            (_, _) => {
                panic!("Cannot mix Crt and Radix representation")
            }
        }
    }
}

// scalar ops
impl SmartAdd<&mut DynInnerCiphertext, u64> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_add(&self, lhs: &mut DynInnerCiphertext, rhs: u64) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => {
                DynInnerCiphertext::Radix(self.inner.smart_scalar_add(lhs, rhs))
            }
            DynInnerCiphertext::Crt(_lhs) => todo!(),
        }
    }
}

impl SmartSub<&mut DynInnerCiphertext, u64> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_sub(&self, lhs: &mut DynInnerCiphertext, rhs: u64) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => {
                DynInnerCiphertext::Radix(self.inner.smart_scalar_sub(lhs, rhs))
            }
            DynInnerCiphertext::Crt(_lhs) => todo!(),
        }
    }
}

impl SmartMul<&mut DynInnerCiphertext, u64> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_mul(&self, lhs: &mut DynInnerCiphertext, rhs: u64) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => {
                DynInnerCiphertext::Radix(self.inner.smart_scalar_mul(lhs, rhs))
            }
            DynInnerCiphertext::Crt(_lhs) => todo!(),
        }
    }
}

impl SmartShl<&mut DynInnerCiphertext, u64> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_shl(&self, lhs: &mut DynInnerCiphertext, rhs: u64) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => DynInnerCiphertext::Radix(
                self.inner
                    .unchecked_scalar_left_shift(lhs, rhs.try_into().unwrap()),
            ),
            DynInnerCiphertext::Crt(_lhs) => todo!(),
        }
    }
}

impl SmartShr<&mut DynInnerCiphertext, u64> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_shr(&self, lhs: &mut DynInnerCiphertext, rhs: u64) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => DynInnerCiphertext::Radix(
                self.inner
                    .unchecked_scalar_right_shift(lhs, rhs.try_into().unwrap()),
            ),
            DynInnerCiphertext::Crt(_lhs) => todo!(),
        }
    }
}

// scalar ops assign

impl SmartAddAssign<DynInnerCiphertext, u64> for DynInnerServerKey {
    fn smart_add_assign(&self, lhs: &mut DynInnerCiphertext, rhs: u64) {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => self.inner.smart_scalar_add_assign(lhs, rhs),
            DynInnerCiphertext::Crt(_lhs) => {
                todo!()
            }
        }
    }
}

impl SmartSubAssign<DynInnerCiphertext, u64> for DynInnerServerKey {
    fn smart_sub_assign(&self, lhs: &mut DynInnerCiphertext, rhs: u64) {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => self.inner.smart_scalar_sub_assign(lhs, rhs),
            DynInnerCiphertext::Crt(_lhs) => {
                todo!()
            }
        }
    }
}

impl SmartMulAssign<DynInnerCiphertext, u64> for DynInnerServerKey {
    fn smart_mul_assign(&self, lhs: &mut DynInnerCiphertext, rhs: u64) {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => self.inner.smart_scalar_mul_assign(lhs, rhs),
            DynInnerCiphertext::Crt(_lhs) => {
                todo!()
            }
        }
    }
}

impl SmartShlAssign<DynInnerCiphertext, u64> for DynInnerServerKey {
    fn smart_shl_assign(&self, lhs: &mut DynInnerCiphertext, rhs: u64) {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => self
                .inner
                .unchecked_scalar_left_shift_assign(lhs, rhs.try_into().unwrap()),
            DynInnerCiphertext::Crt(_lhs) => {
                todo!()
            }
        }
    }
}

impl SmartShrAssign<DynInnerCiphertext, u64> for DynInnerServerKey {
    fn smart_shr_assign(&self, lhs: &mut DynInnerCiphertext, rhs: u64) {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => self
                .inner
                .unchecked_scalar_right_shift_assign(lhs, rhs.try_into().unwrap()),
            DynInnerCiphertext::Crt(_lhs) => {
                todo!()
            }
        }
    }
}

impl IntegerParameter for DynIntegerParameters {
    type Id = IntegerTypeId;
    type InnerCiphertext = DynInnerCiphertext;
    type InnerClientKey = DynInnerClientKey;
    type InnerServerKey = DynInnerServerKey;
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
        let key = self.type_id.unwrapped_ref_key(key);
        let ciphertext = key.inner.encrypt(value);
        DynInteger::new(ciphertext, self.type_id)
    }
}
