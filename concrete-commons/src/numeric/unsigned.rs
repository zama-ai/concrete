use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use crate::numeric::CastInto;

use super::{CastFrom, Numeric, SignedInteger};

/// A trait shared by all the unsigned integer types.
pub trait UnsignedInteger:
    Numeric
    + Ord
    + Eq
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Div<Self, Output = Self>
    + DivAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Rem<Self, Output = Self>
    + RemAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + BitAnd<Self, Output = Self>
    + BitAndAssign<Self>
    + BitOr<Self, Output = Self>
    + BitOrAssign<Self>
    + BitXor<Self, Output = Self>
    + BitXorAssign<Self>
    + Not<Output = Self>
    + Shl<usize, Output = Self>
    + ShlAssign<usize>
    + Shr<usize, Output = Self>
    + ShrAssign<usize>
    + CastFrom<f64>
    + CastInto<f64>
{
    /// The signed type of the same precision.
    type Signed: SignedInteger<Unsigned = Self> + CastFrom<Self>;
    /// Compute an addition, modulo the max of the type.
    #[must_use]
    fn wrapping_add(self, other: Self) -> Self;
    /// Compute a subtraction, modulo the max of the type.
    #[must_use]
    fn wrapping_sub(self, other: Self) -> Self;
    /// Compute a division, modulo the max of the type.
    #[must_use]
    fn wrapping_div(self, other: Self) -> Self;
    /// Compute a multiplication, modulo the max of the type.
    #[must_use]
    fn wrapping_mul(self, other: Self) -> Self;
    /// Compute a negation, modulo the max of the type.
    #[must_use]
    fn wrapping_neg(self) -> Self;
    /// Compute an exponentiation, modulo the max of the type.
    #[must_use]
    fn wrapping_pow(self, exp: u32) -> Self;
    /// Panic free shift-left operation.
    #[must_use]
    fn wrapping_shl(self, rhs: u32) -> Self;
    /// Panic free shift-right operation.
    #[must_use]
    fn wrapping_shr(self, rhs: u32) -> Self;
    /// Returns the casting of the current value to the signed type of the same size.
    fn into_signed(self) -> Self::Signed;
    /// Returns a bit representation of the integer, where blocks of length `block_length` are
    /// separated by whitespaces to increase the readability.
    fn to_bits_string(&self, block_length: usize) -> String;
}

macro_rules! implement {
    ($Type: tt, $SignedType:ty, $bits:expr) => {
        impl Numeric for $Type {
            const BITS: usize = $bits;
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const TWO: Self = 2;
            const MAX: Self = <$Type>::MAX;
        }
        impl UnsignedInteger for $Type {
            type Signed = $SignedType;
            fn into_signed(self) -> Self::Signed {
                Self::Signed::cast_from(self)
            }
            fn to_bits_string(&self, break_every: usize) -> String {
                let mut strn = match <$Type as Numeric>::BITS {
                    8 => format!("{:08b}", self),
                    16 => format!("{:016b}", self),
                    32 => format!("{:032b}", self),
                    64 => format!("{:064b}", self),
                    128 => format!("{:0128b}", self),
                    _ => unreachable!(),
                };
                for i in (1..(<$Type as Numeric>::BITS / break_every)).rev() {
                    strn.insert(i * break_every, ' ');
                }
                strn
            }
            fn wrapping_add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            fn wrapping_sub(self, other: Self) -> Self {
                self.wrapping_sub(other)
            }
            fn wrapping_div(self, other: Self) -> Self {
                self.wrapping_div(other)
            }
            fn wrapping_mul(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }
            fn wrapping_neg(self) -> Self {
                self.wrapping_neg()
            }
            fn wrapping_shl(self, rhs: u32) -> Self {
                self.wrapping_shl(rhs)
            }
            fn wrapping_shr(self, rhs: u32) -> Self {
                self.wrapping_shr(rhs)
            }
            fn wrapping_pow(self, exp: u32) -> Self {
                self.wrapping_pow(exp)
            }
        }
    };
}

implement!(u8, i8, 8);
implement!(u16, i16, 16);
implement!(u32, i32, 32);
implement!(u64, i64, 64);
implement!(u128, i128, 128);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_uint8_binary_rep() {
        let a: u8 = 100;
        let b = a.to_bits_string(4);
        assert_eq!(b, "0110 0100".to_string());
    }

    #[test]
    fn test_uint16_binary_rep() {
        let a: u16 = 25702;
        let b = a.to_bits_string(4);
        assert_eq!(b, "0110 0100 0110 0110".to_string());
    }

    #[test]
    fn test_uint32_binary_rep() {
        let a: u32 = 1684411356;
        let b = a.to_bits_string(4);
        assert_eq!(b, "0110 0100 0110 0110 0001 0011 1101 1100".to_string());
    }

    #[test]
    fn test_uint64_binary_rep() {
        let a: u64 = 7_234_491_689_707_068_824;
        let b = a.to_bits_string(4);
        assert_eq!(
            b,
            "0110 0100 0110 0110 0001 0011 1101 1100 \
                       1001 1111 1000 0001 0101 1101 1001 1000"
                .to_string()
        );
    }

    #[test]
    fn test_uint128_binary_rep() {
        let a: u128 = 124_282_366_920_938_463_463_374_121_543_098_288_434;
        let b = a.to_bits_string(4);
        assert_eq!(
            b,
            "0101 1101 0111 1111 1110 1001 1100 0111 \
                       1000 1110 0110 0010 0000 0101 1011 0000 \
                       1011 1000 0011 0000 0001 0000 1001 0110 \
                       0011 1010 0110 1101 1100 1001 0011 0010"
                .to_string()
        );
    }
}
