use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "internal-keycache")]
use concrete_shortint::keycache::KEY_CACHE;
use concrete_shortint::ClientKey;

use super::types::ShortIntegerParameter;

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
        let key = KEY_CACHE
            .get_from_param(parameters.into())
            .client_key()
            .clone();
        #[cfg(not(feature = "internal-keycache"))]
        let key = ClientKey::new(parameters.into());

        Self {
            key,
            _marker: Default::default(),
        }
    }
}
