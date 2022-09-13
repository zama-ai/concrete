use concrete_boolean::client_key::ClientKey;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::parameters::BooleanParameterSet;

#[cfg_attr(doc, cfg(feature = "booleans"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct GenericBoolClientKey<P>
where
    P: BooleanParameterSet,
{
    pub(in crate::booleans) key: ClientKey,
    _marker: std::marker::PhantomData<P>,
}

impl<P> GenericBoolClientKey<P>
where
    P: BooleanParameterSet,
{
    pub(in crate::booleans) fn new(parameters: P) -> Self {
        Self {
            key: ClientKey::new(&parameters.into()),
            _marker: Default::default(),
        }
    }
}
