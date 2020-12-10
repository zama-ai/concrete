//! Noise formulas for the RLWE operations considering that all slot have the same error variance

use crate::LWE;

pub trait RLWE: Sized {
    type STorus;
    fn scalar_polynomial_mult(variance: f64, scalar_polynomial: &[Self]) -> f64;
}

macro_rules! impl_trait_npe_rlwe {
    ($T:ty,$S:ty,$DOC:expr) => {
        impl RLWE for $T {
            type STorus = $S;

            /// Computes the variance of the error distribution after a multiplication between a RLWE sample and a scalar polynomial
            /// sigma_out^2 <- \Sum_i weight_i^2 * sigma^2
            /// Arguments
            /// * `variance` - the error variance in each slot of the input ciphertext
            /// * `scalar_polynomial` - a slice of Torus with the input weights
            /// Output
            /// * the error variance for each slot of the output ciphertext
            fn scalar_polynomial_mult(variance: f64, scalar_polynomial: &[Self]) -> f64 {
                return <$T as LWE>::multisum_uncorrelated(
                    &vec![variance; scalar_polynomial.len()],
                    scalar_polynomial,
                );
            }
        }
    };
}

impl_trait_npe_rlwe!(u32, i32, "type Torus = u32;");
impl_trait_npe_rlwe!(u64, i64, "type Torus = u64;");
