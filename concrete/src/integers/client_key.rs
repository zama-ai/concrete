use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::integers::parameters::FromParameters;
use concrete_integer::{RadixCiphertext, RadixClientKey};

use super::parameters::{IntegerParameter, PrivateIntegerKey};

impl PrivateIntegerKey for RadixClientKey {
    type Ciphertext = RadixCiphertext;

    fn encrypt(&self, value: u64) -> Self::Ciphertext {
        self.encrypt(value)
    }

    fn decrypt(&self, ciphertext: &Self::Ciphertext) -> u64 {
        self.decrypt(ciphertext)
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct GenericIntegerClientKey<P: IntegerParameter> {
    pub(in crate::integers) inner: P::InnerClientKey,
    _marker: PhantomData<P>,
}

impl<P> From<P> for GenericIntegerClientKey<P>
where
    P: IntegerParameter,
{
    fn from(params: P) -> Self {
        let key = P::InnerClientKey::from_parameters(params);
        Self {
            inner: key,
            _marker: Default::default(),
        }
    }
}
