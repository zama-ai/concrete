//! Decomposition of numeric types.

#[cfg(test)]
mod tests;

use concrete_commons::{Numeric, SignedInteger, UnsignedInteger};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// The logarithm of the base used in a decomposition.
///
/// When decomposing an integer over powers of the $2^B$ basis, this type represents the $B$ value.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Deserialize, Serialize)]
pub struct DecompositionBaseLog(pub usize);

/// The number of levels used in a decomposition.
///
/// When decomposing an integer over the $l$ largest powers of the basis, this type represents
/// the $l$ value.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Deserialize, Serialize)]
pub struct DecompositionLevelCount(pub usize);

/// The level of a given member of a decomposition.
///
/// When decomposing an integer over the $l$ largest powers of the basis, this type represent the
/// level (in $[0,l)$) of the coefficient currently manipulated.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Deserialize, Serialize)]
pub struct DecompositionLevel(pub usize);

/// A trait for numeric types that can be decomposed using a signed decomposition.
pub trait SignedDecomposable: Sized {
    /// Rounds an unsigned integer to the closest element representable by the signed
    /// decomposition defined by `base_log` and `level`.
    fn round_to_closest_multiple(
        self,
        base_log: DecompositionBaseLog,
        level: DecompositionLevelCount,
    ) -> Self;

    /// This function only gives the level-th piece of the signed decomposition of a value.
    /// To do so, it needs the value, the decomposition parameters and the carry from the previous level
    /// We end up with coefficients in [-B/2, B/2[
    fn signed_decompose_one_level(
        self,
        previous_carry: Self,
        base_log: DecompositionBaseLog,
        level: DecompositionLevel,
    ) -> (Self, Self);

    /// Returns an element with some bits at the right place according to the `base_log` and `level`
    /// decomposition.
    fn set_val_at_level(self, base_log: DecompositionBaseLog, level: DecompositionLevel) -> Self;
}

macro_rules! implement {
    ($Type: tt) => {
        impl SignedDecomposable for $Type {
            fn round_to_closest_multiple(
                self,
                base_log: DecompositionBaseLog,
                level: DecompositionLevelCount,
            ) -> Self {
                // number of bit to throw out
                let shift: usize = <Self as Numeric>::BITS - level.0 * base_log.0;
                // get the first bit (MSB) to be thrown out
                let mask = 1 << (shift - 1);
                let b = (self & mask) >> (shift - 1);
                // do the truncation by shifting the MSB into LSB
                let mut res = self >> shift;
                // do the rounding
                res += b;
                // put back the MSB where they belong
                res <<= shift;
                return res;
            }
            fn signed_decompose_one_level(
                self,
                previous_carry: Self,
                base_log: DecompositionBaseLog,
                level: DecompositionLevel,
            ) -> (Self, Self) {
                let block_bit_mask: Self = (1 << base_log.0) - 1; // 000...000011...11 : with log '1'
                let msb_block_mask: Self = 1 << (base_log.0 - 1); //     000...000010...00 : the one is in the MSB
                let mut tmp = (self >> (<Self as Numeric>::BITS - base_log.0 * (level.0 + 1)))
                    & block_bit_mask;
                let mut carry: Self = tmp & msb_block_mask; // 0000...0001000 or 0000...0000000
                tmp = tmp.wrapping_add(previous_carry); //
                carry |= tmp & msb_block_mask; // 0000...0001000 or 0000...0000000
                let left = tmp.into_signed();
                let right = (carry << 1).into_signed();
                let res = (left - right).into_unsigned(); // res[i] = (tmp od as i32) - b;
                carry >>= (base_log.0 - 1); // 000...0001 or 000...0000
                return (res, carry);
            }
            fn set_val_at_level(
                self,
                base_log: DecompositionBaseLog,
                level: DecompositionLevel,
            ) -> Self {
                let mut res = 0;
                let shift: usize = <Self as Numeric>::BITS - (base_log.0 * (level.0 + 1));
                res += self << (shift);
                return res;
            }
        }
    };
}

implement!(u8);
implement!(u16);
implement!(u32);
implement!(u64);
implement!(u128);
