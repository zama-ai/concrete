use concrete_boolean::parameters::BooleanParameters;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::booleans::client_key::GenericBoolClientKey;
use crate::booleans::parameters::BooleanParameterSet;
use crate::booleans::server_key::GenericBoolServerKey;
use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey};
use crate::keys::ClientKey;
use crate::FheBoolParameters;

use super::base::GenericBool;

// Has Overridable Operator:
// - and => BitAnd => &
// - not => Not => !
// - or => BitOr => |
// - xor => BitXor => ^
//
// Does Not have overridable operator:
// - mux -> But maybe by using a macro_rules with regular function we can have some sufficiently
//   nice syntax sugar
// - nand
// - nor
// - xnor should be Eq => ==,  But Eq requires to return a bool not a FHE bool So we cant do it
// - ||, && cannot be overloaded, maybe a well-crafted macro-rules that implements `if-else` could
//   bring this syntax sugar

/// The struct to identify the static boolean type
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Default)]
pub struct FheBoolId;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct StaticBoolParameters(pub(crate) FheBoolParameters);

impl From<StaticBoolParameters> for BooleanParameters {
    fn from(p: StaticBoolParameters) -> Self {
        p.0.into()
    }
}

impl From<FheBoolParameters> for StaticBoolParameters {
    fn from(p: FheBoolParameters) -> Self {
        Self(p)
    }
}

impl BooleanParameterSet for StaticBoolParameters {
    type Id = FheBoolId;
}

pub type FheBool = GenericBool<StaticBoolParameters>;
pub(in crate::booleans) type FheBoolClientKey = GenericBoolClientKey<StaticBoolParameters>;
pub(in crate::booleans) type FheBoolServerKey = GenericBoolServerKey<StaticBoolParameters>;

impl_with_global_key!(
    for FheBoolId {
        key_type: FheBoolServerKey,
        keychain_member: bool_key.key,
        type_variant: Type::FheBool,
    }
);

impl_ref_key_from_keychain!(
    for FheBoolId {
        key_type: FheBoolClientKey,
        keychain_member: bool_key.key,
        type_variant: Type::FheBool,
    }
);
