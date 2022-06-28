use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use concrete_shortint::ClientKey;

use super::{GenericShortInt, ShortIntegerParameter};

#[cfg(feature = "internal-keycache")]
use concrete_shortint::keycache::KEY_CACHE;

/// The key associated to a short integer type
///
/// Can encrypt and decrypt it.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct ShortIntegerClientKey<P: ShortIntegerParameter> {
    pub(super) key: ClientKey,
    _marker: PhantomData<P>,
}

impl<P> ShortIntegerClientKey<P>
where
    P: ShortIntegerParameter,
{
    pub(in crate::shortints) fn new(parameters: P) -> Self {
        #[cfg(feature = "internal-keycache")]
        let key = KEY_CACHE.get_from_param(parameters.into()).0;
        #[cfg(not(feature = "internal-keycache"))]
        let key = ClientKey::new(parameters.into());

        Self {
            key,
            _marker: Default::default(),
        }
    }

    pub(super) fn encrypt(&self, value: u8) -> GenericShortInt<P> {
        self.key.encrypt(u64::from(value)).into()
    }

    pub(super) fn decrypt(&self, value: &GenericShortInt<P>) -> u64 {
        self.key.decrypt(&value.ciphertext.borrow())
    }
}
