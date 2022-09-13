#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for 'radix' decomposition
///
/// Radix decomposition works by using multiple shortint blocks
/// with the same parameters to represent an integer.
///
/// For example, by taking 4 blocks with parameters
/// for 2bits shortints, with have a 4 * 2 = 8 bit integer.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct RadixParameters {
    pub block_parameters: concrete_shortint::Parameters,
    pub num_block: usize,
}

impl From<RadixParameters> for IntegerParameterSet {
    fn from(radix_params: RadixParameters) -> Self {
        Self::Radix(radix_params)
    }
}

/// Parameters for integers
///
/// Integers works by composing multiple shortints.
///
/// For now, only the radix decomposition is available
/// via the [RadixParameters]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub enum IntegerParameterSet {
    Radix(RadixParameters),
}

/// Trait to mark parameters type for integers
pub trait IntegerParameter: Copy + Into<IntegerParameterSet> {
    /// The Id allows to differentiate the different parameters
    /// as well as retrieving the corresponding client key and server key
    type Id: Copy;
}

/// Trait to mark parameters type for static integers
///
/// Static means the integer types with parameters provided by
/// the crate, so parameters for which we know the number of
/// bits the represent.
pub trait StaticIntegerParameter: IntegerParameter {
    const MESSAGE_BITS: usize;
}
