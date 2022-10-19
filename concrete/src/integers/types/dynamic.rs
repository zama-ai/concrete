use concrete_integer::wopbs::WopbsKey;
use concrete_integer::{CrtCiphertext, CrtClientKey, RadixCiphertext, RadixClientKey, ServerKey};
use concrete_shortint::Parameters;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey};
use crate::global_state::WithGlobalKey;
use crate::integers::client_key::GenericIntegerClientKey;
use crate::integers::parameters::{
    EvaluationIntegerKey, FromParameters, IntegerParameter, PrivateIntegerKey,
};
use crate::integers::server_key::{
    GenericIntegerServerKey, SmartAdd, SmartAddAssign, SmartBitAnd, SmartBitAndAssign, SmartBitOr,
    SmartBitOrAssign, SmartBitXor, SmartBitXorAssign, SmartMul, SmartMulAssign, SmartNeg, SmartShl,
    SmartShlAssign, SmartShr, SmartShrAssign, SmartSub, SmartSubAssign,
};
use crate::keys::RefKeyFromKeyChain;
use crate::traits::{DynamicFheEncryptor, FheBootstrap};
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
        // Use full call syntax to make sure we make use of the key-cache
        let inner = match client_key {
            DynInnerClientKey::Radix(key) => {
                <ServerKey as EvaluationIntegerKey<RadixClientKey>>::new(key)
            }
            DynInnerClientKey::Crt(key) => {
                <ServerKey as EvaluationIntegerKey<CrtClientKey>>::new(key)
            }
        };
        Self { inner }
    }

    fn new_wopbs_key(
        client_key: &DynInnerClientKey,
        server_key: &Self,
        wopbs_block_parameters: concrete_shortint::Parameters,
    ) -> WopbsKey {
        match client_key {
            DynInnerClientKey::Radix(cks) => {
                <ServerKey as EvaluationIntegerKey<RadixClientKey>>::new_wopbs_key(
                    cks,
                    &server_key.inner,
                    wopbs_block_parameters,
                )
            }
            DynInnerClientKey::Crt(cks) => {
                <ServerKey as EvaluationIntegerKey<CrtClientKey>>::new_wopbs_key(
                    cks,
                    &server_key.inner,
                    wopbs_block_parameters,
                )
            }
        }
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

impl IntegerParameter for DynIntegerParameters {
    type Id = IntegerTypeId;
    type InnerCiphertext = DynInnerCiphertext;
    type InnerClientKey = DynInnerClientKey;
    type InnerServerKey = DynInnerServerKey;

    fn wopbs_block_parameters(&self) -> Parameters {
        match self {
            Self::Radix(radix_params) => radix_params.wopbs_block_parameters,
            Self::Crt(crt_params) => crt_params.wopbs_block_parameters,
        }
    }

    fn block_parameters(&self) -> Parameters {
        match self {
            Self::Radix(radix_params) => radix_params.block_parameters,
            Self::Crt(crt_params) => crt_params.block_parameters,
        }
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

impl FheBootstrap for DynInteger {
    fn map<F: Fn(u64) -> u64>(&self, func: F) -> Self {
        let new_ct = self
            .id
            .with_unwrapped_global_mut(|key| match &*self.ciphertext.borrow() {
                DynInnerCiphertext::Radix(ct) => {
                    let res = crate::integers::types::base::wopbs_radix(
                        &key.wopbs_key,
                        &key.inner.inner,
                        ct,
                        func,
                    );
                    DynInnerCiphertext::Radix(res)
                }
                DynInnerCiphertext::Crt(ct) => {
                    let res = crate::integers::types::base::wopbs_crt(
                        &key.wopbs_key,
                        &key.inner.inner,
                        ct,
                        func,
                    );
                    DynInnerCiphertext::Crt(res)
                }
            });

        Self::new(new_ct, self.id)
    }

    fn apply<F: Fn(u64) -> u64>(&mut self, func: F) {
        let res = self.map(func);
        self.ciphertext = res.ciphertext;
    }
}

impl DynInteger {
    pub fn bivariate_function<F: Fn(u64, u64) -> u64>(&self, rhs: &Self, func: F) -> Self {
        let res_ct = self.id.with_unwrapped_global_mut(|key| {
            match (&*self.ciphertext.borrow(), &*rhs.ciphertext.borrow()) {
                (DynInnerCiphertext::Radix(lhs), DynInnerCiphertext::Radix(rhs)) => {
                    let res = crate::integers::types::base::bivariate_wopbs_radix(
                        &key.wopbs_key,
                        &key.inner.inner,
                        lhs,
                        rhs,
                        func,
                    );
                    DynInnerCiphertext::Radix(res)
                }
                (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                    let res = crate::integers::types::base::bivariate_wopbs_crt(
                        &key.wopbs_key,
                        &key.inner.inner,
                        lhs,
                        rhs,
                        func,
                    );
                    DynInnerCiphertext::Crt(res)
                }
                (_, _) => panic!("Mismatch between ciphertext and client key representation"),
            }
        });

        Self::new(res_ct, self.id)
    }
}

// Normal Ops

impl SmartNeg<&mut DynInnerCiphertext> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_neg(&self, lhs: &mut DynInnerCiphertext) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => {
                DynInnerCiphertext::Radix(self.inner.smart_neg_parallelized(lhs))
            }
            DynInnerCiphertext::Crt(lhs) => {
                DynInnerCiphertext::Crt(self.inner.smart_crt_neg_parallelized(lhs))
            }
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
                DynInnerCiphertext::Radix(self.inner.smart_add_parallelized(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                DynInnerCiphertext::Crt(self.inner.smart_crt_add_parallelized(lhs, rhs))
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
                DynInnerCiphertext::Radix(self.inner.smart_sub_parallelized(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                DynInnerCiphertext::Crt(self.inner.smart_crt_sub_parallelized(lhs, rhs))
            }
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
                DynInnerCiphertext::Radix(self.inner.smart_mul_parallelized(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                DynInnerCiphertext::Crt(self.inner.smart_crt_mul_parallelized(lhs, rhs))
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
                DynInnerCiphertext::Radix(self.inner.smart_bitand_parallelized(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => {
                panic!("This operation is not supported for CRT representation")
            }
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
                DynInnerCiphertext::Radix(self.inner.smart_bitor_parallelized(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => {
                panic!("This operation is not supported for CRT representation")
            }
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
                DynInnerCiphertext::Radix(self.inner.smart_bitxor_parallelized(lhs, rhs))
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => {
                panic!("This operation is not supported for CRT representation")
            }
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
                self.inner.smart_add_assign_parallelized(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                self.inner.smart_crt_add_assign_parallelized(lhs, rhs)
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
                self.inner.smart_sub_assign_parallelized(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                self.inner.smart_crt_sub_assign_parallelized(lhs, rhs)
            }
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
                self.inner.smart_mul_assign_parallelized(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(lhs), DynInnerCiphertext::Crt(rhs)) => {
                self.inner.smart_crt_mul_assign_parallelized(lhs, rhs)
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
                self.inner.smart_bitand_assign_parallelized(lhs, rhs)
            }
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => {
                panic!("This operation is not supported for CRT representation")
            }
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
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => {
                panic!("This operation is not supported for CRT representation")
            }
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
            (DynInnerCiphertext::Crt(_lhs), DynInnerCiphertext::Crt(_rhs)) => {
                panic!("This operation is not supported for CRT representation")
            }
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
                DynInnerCiphertext::Radix(self.inner.smart_scalar_add_parallelized(lhs, rhs))
            }
            DynInnerCiphertext::Crt(lhs) => {
                DynInnerCiphertext::Crt(self.inner.smart_crt_scalar_add_parallelized(lhs, rhs))
            }
        }
    }
}

impl SmartSub<&mut DynInnerCiphertext, u64> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_sub(&self, lhs: &mut DynInnerCiphertext, rhs: u64) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => {
                DynInnerCiphertext::Radix(self.inner.smart_scalar_sub_parallelized(lhs, rhs))
            }
            DynInnerCiphertext::Crt(lhs) => {
                DynInnerCiphertext::Crt(self.inner.smart_crt_scalar_sub_parallelized(lhs, rhs))
            }
        }
    }
}

impl SmartMul<&mut DynInnerCiphertext, u64> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_mul(&self, lhs: &mut DynInnerCiphertext, rhs: u64) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => {
                DynInnerCiphertext::Radix(self.inner.smart_scalar_mul_parallelized(lhs, rhs))
            }
            DynInnerCiphertext::Crt(lhs) => {
                DynInnerCiphertext::Crt(self.inner.smart_crt_scalar_mul_parallelized(lhs, rhs))
            }
        }
    }
}

impl SmartShl<&mut DynInnerCiphertext, u64> for DynInnerServerKey {
    type Output = DynInnerCiphertext;

    fn smart_shl(&self, lhs: &mut DynInnerCiphertext, rhs: u64) -> Self::Output {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => DynInnerCiphertext::Radix(
                self.inner
                    .unchecked_scalar_left_shift_parallelized(lhs, rhs.try_into().unwrap()),
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
                    .unchecked_scalar_right_shift_parallelized(lhs, rhs.try_into().unwrap()),
            ),
            DynInnerCiphertext::Crt(_lhs) => todo!(),
        }
    }
}

// scalar ops assign

impl SmartAddAssign<DynInnerCiphertext, u64> for DynInnerServerKey {
    fn smart_add_assign(&self, lhs: &mut DynInnerCiphertext, rhs: u64) {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => {
                self.inner.smart_scalar_add_assign_parallelized(lhs, rhs)
            }
            DynInnerCiphertext::Crt(lhs) => self
                .inner
                .smart_crt_scalar_add_assign_parallelized(lhs, rhs),
        }
    }
}

impl SmartSubAssign<DynInnerCiphertext, u64> for DynInnerServerKey {
    fn smart_sub_assign(&self, lhs: &mut DynInnerCiphertext, rhs: u64) {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => {
                self.inner.smart_scalar_sub_assign_parallelized(lhs, rhs)
            }
            DynInnerCiphertext::Crt(lhs) => self
                .inner
                .smart_crt_scalar_sub_assign_parallelized(lhs, rhs),
        }
    }
}

impl SmartMulAssign<DynInnerCiphertext, u64> for DynInnerServerKey {
    fn smart_mul_assign(&self, lhs: &mut DynInnerCiphertext, rhs: u64) {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => {
                self.inner.smart_scalar_mul_assign_parallelized(lhs, rhs)
            }
            DynInnerCiphertext::Crt(lhs) => self
                .inner
                .smart_crt_scalar_mul_assign_parallelized(lhs, rhs),
        }
    }
}

impl SmartShlAssign<DynInnerCiphertext, u64> for DynInnerServerKey {
    fn smart_shl_assign(&self, lhs: &mut DynInnerCiphertext, rhs: u64) {
        match lhs {
            DynInnerCiphertext::Radix(lhs) => self
                .inner
                .unchecked_scalar_left_shift_assign_parallelized(lhs, rhs.try_into().unwrap()),
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
                .unchecked_scalar_right_shift_assign_parallelized(lhs, rhs.try_into().unwrap()),
            DynInnerCiphertext::Crt(_lhs) => {
                todo!()
            }
        }
    }
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
