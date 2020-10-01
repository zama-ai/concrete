//! Types Module
//! * Contains definitions for the Torus type and other related types
//! * Contains also some material needed to manipulate such elements

#[cfg(test)]
mod tests;

#[cfg(not(any(feature = "cloud-computing")))]
pub const N_TESTS: usize = 2;

#[cfg(all(feature = "cloud-computing"))]
pub const N_TESTS: usize = 100;

// fftw
use fftw::plan::*;
use fftw::types::*;

// FTorus definition
pub type FTorus = f64;
pub const FTORUS_BIT: usize = 64;
pub const PI_TORUS: f64 = std::f64::consts::PI;

// CTorus definition
pub type CTorus = c64;
pub const CTORUS_BIT: usize = 128;
pub type C2CPlanTorus = C2CPlan64;

pub trait Types: Sized {
    type STorus;
    const TORUS_BIT: usize;
    const TORUS_MAX: Self;
    fn round_to_closest_multiple(x: Self, base_log: usize, level: usize) -> Self;
    fn torus_small_sign_decompose(res: &mut [Self], val: Self, base_log: usize);
    fn signed_decompose_one_level(
        value: Self,
        previous_carry: Self,
        base_log: usize,
        level: usize,
    ) -> (Self, Self);
    fn torus_to_f64(item: Self) -> f64;
    fn f64_to_torus(item: f64) -> Self;
    fn set_val_at_level_l(val: Self, base_log: usize, level_l: usize) -> Self;
    fn torus_binary_representation(n: Self, base_log: usize) -> String;
    fn torus_binary_comparison(t1: Self, t2: Self) -> String;
}

macro_rules! impl_trait_types {
    ($T:ty,$TB:expr,$TM:expr,$ST:ty) => {
        impl Types for $T {
            type STorus = $ST;
            const TORUS_BIT: usize = $TB;
            const TORUS_MAX: $T = $TM;

            /// Rounds a torus element to its closest torus element with only lv_tot * log_b MSB that can be different from zero
            /// # Comments
            /// * warning: panic when base_log*level >= TORUS_BIT if not in release mode
            /// # Example with binary representations:
            /// round_to_closest_multiple(1100100...0,2,2) -> 11010...0
            /// we will only keep 2*2 = 4 MSB
            /// so we can put a dot where the rounding happens, which is after the 4th MSB: 1100.1 is rounded to 1101
            /// # Arguments
            /// * `x` - element of the Torus to be rounded
            /// * `base_log` - number of bits for the base B (B=2^base_log)
            /// * `level` - number of blocks of the gadget matrix
            /// # Output
            /// * the rounded Torus element
            /// # Tests
            /// * test_round_to_closest_multiple: unit test
            /// * test_panic_round_to_closest_multiple: panic test when base_log*level == TORUS_BIT
            fn round_to_closest_multiple(x: $T, base_log: usize, level: usize) -> $T {
                // number of bit to throw out
                let shift: usize = Self::TORUS_BIT - level * base_log;

                // get the first bit (MSB) to be thrown out
                let mask: $T = 1 << (shift - 1);
                let b = (x & mask) >> (shift - 1);

                // do the truncation by shifting the MSB into LSB
                let mut res = x >> shift;

                // do the rounding
                res += b;

                // put back the MSB where they belong
                res = res << shift;
                return res;
            }

            /// Computes a signed decomposition of a Torus element
            /// The base is B = 2^base_log
            /// We end up with coefficients in [-B/2, B/2[
            /// # Comments
            /// * used for the gadget decomposition
            /// * len(res) equals level
            /// * takes into account the case where a block is equals to 11...11 (only ones) to get the right carry
            /// # Arguments
            /// * `res` - a tensor of signed integers (output)
            /// * `val` - the Torus element to be decomposed
            /// * `base_log` - number of bits for the base B (B=2^base_log)
            /// # Tests
            /// * test_torus_small_sign_decompose: unit test
            fn torus_small_sign_decompose(res: &mut [$T], val: $T, base_log: usize) {
                let mut tmp: $T;
                let mut carry: $T = 0;
                let mut previous_carry: $T;
                let block_bit_mask: $T = (1 << base_log) - 1; // 000...000011...11 : there are base_log ones in the LSB, it represents a block
                let msb_block_mask: $T = 1 << (base_log - 1); // 000...000010...00 : the one is in the MSB of the block

                // compute the decomposition from LSB to MSB (because of the carry)
                for i in (0..res.len()).rev() {
                    previous_carry = carry;
                    tmp = (val >> (Self::TORUS_BIT - base_log * (i + 1))) & block_bit_mask;
                    carry = tmp & msb_block_mask;
                    tmp = tmp.wrapping_add(previous_carry);
                    carry = carry | (tmp & msb_block_mask); // 0000...0001000 or 0000...0000000
                    res[i] = ((tmp as Self::STorus) - ((carry << 1) as Self::STorus)) as $T; // res[i] = (tmp as i32) - b;
                    carry = carry >> (base_log - 1); // 000...0001 or 000...0000
                }
            }

            /// This function only gives the level-th piece of the signed decomposition of a value.
            /// To do so, it needs the value, the decomposition parameters and the carry from the previous level
            /// We end up with coefficients in [-B/2, B/2[
            /// # Arguments
            /// * `value` - the Torus value we are decomposing
            /// * `previous_carry` - the carry form the level+1 level
            /// * `base_log` - number of bits for the base B (B=2^base_log)
            /// * `level` - the level of the piece of decomposition we want
            /// # Tests
            /// * test_rsd: unit test
            fn signed_decompose_one_level(
                value: $T,
                previous_carry: $T,
                base_log: usize,
                level: usize,
            ) -> ($T, $T) {
                let block_bit_mask: $T = (1 << base_log) - 1; // 000...000011...11 : with log '1'
                let msb_block_mask: $T = 1 << (base_log - 1); //     000...000010...00 : the one is in the MSB
                let mut tmp: $T =
                    (value >> (Self::TORUS_BIT - base_log * (level + 1))) & block_bit_mask;
                let mut carry: $T = tmp & msb_block_mask; // 0000...0001000 or 0000...0000000
                tmp = tmp.wrapping_add(previous_carry); //
                carry = carry | (tmp & msb_block_mask); // 0000...0001000 or 0000...0000000
                let res = (tmp as Self::STorus) - ((carry << 1) as Self::STorus); // res[i] = (tmp as i32) - b;
                carry = carry >> (base_log - 1); // 000...0001 or 000...0000
                return (res as $T, carry);
            }

            /// Converts a torus item to a f64 in \[0,1\[
            /// # Arguments
            /// * `item` - a Torus element
            /// # Output
            /// * a f64 in \[0,1\[
            fn torus_to_f64(item: $T) -> f64 {
                return (item as f64) / (f64::powi(2.0, Self::TORUS_BIT as i32));
            }

            /// Converts a f64 in \[0,1\[ to a Torus element
            /// # Arguments
            /// * `item` - a f64 in \[0,1\[
            /// # Output
            /// * a Torus element
            fn f64_to_torus(item: f64) -> $T {
                return (item * (f64::powi(2.0, Self::TORUS_BIT as i32))).round() as $T;
            }

            /// Returns a Torus element with a some bits a the right place according to the base_log and level decomposition
            /// # Arguments
            /// * `val` - a Torus element containing on its LSB some bit we want to move at a precise place
            /// * `base_log` - decomposition log2 base
            /// * `level_l` - the desired level
            /// # Output
            /// * a torus element build as desired
            fn set_val_at_level_l(val: $T, base_log: usize, level_l: usize) -> $T {
                let mut res: $T = 0;
                let shift: usize = Self::TORUS_BIT - (base_log * (level_l + 1));
                res = res + ((val as $T) << (shift));
                return res;
            }

            /// Output a string with a binary representation of n (a torus element)
            /// if base_log!=0 then bits are packed from MSB to LSD into words of logbase bits
            /// if base_log==0 then all bits are in one word
            /// # Arguments
            /// * `n` - the Torus element we want its representation
            /// * `base_log` - decomposition log2 base
            /// # Output
            /// * a String with the binary representation
            fn torus_binary_representation(n: $T, base_log: usize) -> String {
                let mut res = String::from("");
                for i in 0..Self::TORUS_BIT {
                    if base_log != 0 {
                        if (i % (base_log) == 0) & (i != 0) {
                            res.push(' ');
                        }
                    }
                    if 1 & (n >> (Self::TORUS_BIT - 1 - i)) == 0 {
                        res.push('0');
                    } else {
                        res.push('1');
                    }
                }
                return res;
            }

            /// Compares 2 Torus element and output a String with X when their bits differs
            /// # Arguments
            /// * `t1` - the first Torus element
            /// * `t2` - the second Torus element
            /// # Output
            /// * a String with the binary comparison
            fn torus_binary_comparison(t1: $T, t2: $T) -> String {
                let mut res = String::from("");
                let mut b1: bool;
                let mut b2: bool;
                for i in 0..Self::TORUS_BIT {
                    b1 = 1 & (t1 >> (Self::TORUS_BIT - 1 - i)) == 0;
                    b2 = 1 & (t2 >> (Self::TORUS_BIT - 1 - i)) == 0;
                    if b1 == b2 {
                        res.push(' ');
                    } else {
                        res.push('X');
                    }
                }
                return res;
            }
        }
    };
}

impl_trait_types!(u32, 32, std::u32::MAX, i32);
impl_trait_types!(u64, 64, std::u64::MAX, i64);
