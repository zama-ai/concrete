use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "internal-keycache")]
use concrete_integer::keycache::KEY_CACHE;
use concrete_integer::ClientKey;

use super::parameters::{IntegerParameter, IntegerParameterSet};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct GenericIntegerClientKey<P: IntegerParameter> {
    pub(in crate::integers) key: ClientKey,
    pub(in crate::integers) _marker: PhantomData<P>,
    #[cfg(feature = "internal-keycache")]
    pub(in crate::integers) params: IntegerParameterSet,
}

impl<P> From<P> for GenericIntegerClientKey<P>
where
    P: IntegerParameter,
{
    fn from(params: P) -> Self {
        let params: IntegerParameterSet = params.into();
        match params {
            IntegerParameterSet::Radix(radix_params) => {
                #[cfg(feature = "internal-keycache")]
                {
                    let key = KEY_CACHE
                        .get_from_params(
                            radix_params.block_parameters,
                            concrete_integer::client_key::VecLength(radix_params.num_block),
                        )
                        .0;
                    Self {
                        key,
                        _marker: Default::default(),
                        params,
                    }
                }
                #[cfg(not(feature = "internal-keycache"))]
                {
                    Self {
                        key: ClientKey::new(radix_params.block_parameters, radix_params.num_block),
                        _marker: Default::default(),
                    }
                }
            }
        }
    }
}
