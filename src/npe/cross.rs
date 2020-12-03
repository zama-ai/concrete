//! Noise formulas for the cross-sample type operations
//! Those functions will be used in the cross-sample tests to check that
//! the noise behavior is consistent with the theory.

use crate::Types;

pub trait Cross: Sized {
    fn external_product(
        dimension: usize,
        l_gadget: usize,
        base_log: usize,
        polynomial_size: usize,
        var_trgsw: f64,
        var_trlwe: f64,
    ) -> f64;
    fn bootstrap(
        lwe_dimension: usize,
        rlwe_dimension: usize,
        l_gadget: usize,
        base_log: usize,
        polynomial_size: usize,
        var_bsk: f64,
    ) -> f64;
    fn cmux(
        var_rlwe_0: f64,
        var_rlwe_1: f64,
        var_trgsw: f64,
        dimension: usize,
        polynomial_size: usize,
        base_log: usize,
        l_gadget: usize,
    ) -> f64;
}

macro_rules! impl_trait_npe_cross {
    ($T:ty,$DOC:expr) => {
        impl Cross for $T {
            /// Return the variance of the external product given a set of parameters.
            /// To see how to use it, please refer to the test of the external product.
            /// Arguments
            /// * `dimension` - the size of the RLWE mask
            /// * `l_gadget` - number of elements for the Torus decomposition
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `polynomial_size` - number of coefficients of the polynomial e.g. degree + 1
            /// * `var_trgsw` - noise variance of the TRGSW
            /// * `var_trlwe` - noise variance of the TRLWE
            /// # Output
            /// * Returns the variance of the output RLWE
            /// # Warning
            /// * only correct for the external product inside a bootstrap
            /// # Example
            /// ```rust
            /// use concrete::npe::Cross ;
            #[doc = $DOC]
            /// // settings
            /// let dimension: usize = 3 ;
            /// let l_gadget: usize = 4 ;
            /// let base_log: usize = 7 ;
            /// let polynomial_size: usize = 1024 ;
            /// let var_trgsw: f64 = f64::powi(2., -38) ;
            /// let var_trlwe: f64 = f64::powi(2., -40) ;
            /// // Computing the noise
            /// let var_external_product = <Torus as Cross>::external_product(dimension, l_gadget, base_log, polynomial_size, var_trgsw, var_trlwe) ;
            /// ```
            fn external_product(
                dimension: usize,
                l_gadget: usize,
                base_log: usize,
                polynomial_size: usize,
                var_trgsw: f64,
                var_trlwe: f64,
            ) -> f64 {
                // norm 2 of the integer polynomial hidden in the TRGSW
                // for an external product inside a bootstrap, the integer polynomial is in fact
                // a constant polynomial equal to 0 or 1
                let norm_2_msg_trgsw = 1.;
                let b_g = 1 << base_log;
                let q_square = f64::powi(2., 2 * <$T as Types>::TORUS_BIT as i32);
                let res_1: f64 =
                    ((dimension + 1) * l_gadget * polynomial_size * (b_g * b_g + 2)) as f64 / 12.
                        * var_trgsw;

                let res_2: f64 = norm_2_msg_trgsw
                    * ((dimension * polynomial_size + 2) as f64
                        / (24. * f64::powi(b_g as f64, 2 * l_gadget as i32)) as f64
                        + (dimension * polynomial_size / 48 - 1 / 12) as f64 / q_square);

                let res_3: f64 = norm_2_msg_trgsw * var_trlwe;
                let res: f64 = res_1 + res_2 + res_3;
                return res;
            }

            /// Return the variance of the cmux given a set of parameters.
            /// To see how to use it, please refer to the test of the cmux.
            /// Arguments
            /// * `var_rlwe_0` - noise variance of the first TRLWE
            /// * `var_rlwe_1` - noise variance of the second TRLWE
            /// * `var_trgsw` - noise variance of the TRGSW
            /// * `dimension` - the size of the RLWE mask
            /// * `polynomial_size` - number of coefficients of the polynomial e.g. degree + 1
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `l_gadget` - number of elements for the Torus decomposition
            /// # Output
            /// * Returns the variance of the output RLWE
            /// # Warning
            /// * only correct for the cmux inside a bootstrap
            /// # Example
            /// ```rust
            /// use concrete::npe::Cross ;
            #[doc = $DOC]
            /// // settings
            /// let dimension: usize = 3 ;
            /// let l_gadget: usize = 4 ;
            /// let base_log: usize = 7 ;
            /// let polynomial_size: usize = 1024 ;
            /// let var_trgsw: f64 = f64::powi(2., -38) ;
            /// let var_trlwe_0: f64 = f64::powi(2., -40) ;
            /// let var_trlwe_1: f64 = f64::powi(2., -40) ;
            /// // Computing the noise
            /// let var_cmux = <Torus as Cross>::cmux(var_trlwe_0, var_trlwe_1, var_trgsw,  dimension, polynomial_size, base_log, l_gadget) ;
            /// ```
            fn cmux(
                var_rlwe_0: f64,
                var_rlwe_1: f64,
                var_trgsw: f64,
                dimension: usize,
                polynomial_size: usize,
                base_log: usize,
                l_gadget: usize,
            ) -> f64 {
                let var_external_product = Self::external_product(
                    dimension,
                    l_gadget,
                    base_log,
                    polynomial_size,
                    var_trgsw,
                    crate::npe::add_ciphertexts(var_rlwe_0, var_rlwe_1),
                );
                let var_cmux = crate::npe::add_ciphertexts(var_external_product, var_rlwe_0);
                return var_cmux;
            }

            /// Return the variance of output of a bootstrap given a set of parameters.
            /// To see how to use it, please refer to the test of the bootstrap.
            /// Arguments
            /// * `lwe_dimension` - size of the LWE mask
            /// * `rlwe_dimension` - size of the RLWE mask
            /// * `l_gadget` - number of elements for the Torus decomposition
            /// * `dimension` - the size of the RLWE mask
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `polynomial_size` - number of coefficients of the polynomial e.g. degree + 1
            /// * `var_bsk` - variance of the bootstrapping key
            /// # Output
            /// * Returns the variance of the output RLWE
            /// # Example
            /// ```rust
            /// use concrete::npe::Cross ;
            #[doc = $DOC]
            /// // settings
            /// let rlwe_dimension: usize = 3 ;
            /// let lwe_dimension: usize = 630 ;
            /// let l_gadget: usize = 4 ;
            /// let base_log: usize = 7 ;
            /// let polynomial_size: usize = 1024 ;
            /// let var_bsk: f64 = f64::powi(2., -38) ;
            /// // Computing the noise
            /// let var_bootstrap = <Torus as Cross>::bootstrap(lwe_dimension, rlwe_dimension, l_gadget, base_log, polynomial_size, var_bsk) ;
            /// ```
            fn bootstrap(
                lwe_dimension: usize,
                rlwe_dimension: usize,
                l_gadget: usize,
                base_log: usize,
                polynomial_size: usize,
                var_bsk: f64,
            ) -> f64 {
                let b_g = 1 << base_log;
                let q_square = f64::powi(2., 2 * <$T as Types>::TORUS_BIT as i32);
                let res_1: f64 = (lwe_dimension
                    * (rlwe_dimension + 1)
                    * l_gadget
                    * polynomial_size
                    * (b_g * b_g + 2)) as f64
                    / 12.
                    * var_bsk;

                let res_2: f64 = lwe_dimension as f64
                    * ((rlwe_dimension * polynomial_size + 2) as f64
                        / (24. * f64::powi(b_g as f64, 2 * l_gadget as i32)) as f64
                        + lwe_dimension as f64
                            * (rlwe_dimension * polynomial_size / 48 - 1 / 12) as f64
                            / q_square);

                let res: f64 = res_1 + res_2;
                return res;
            }
        }
    };
}

impl_trait_npe_cross!(u32, "type Torus = u32;");
impl_trait_npe_cross!(u64, "type Torus = u64;");

/// Computes tho variance of the error during a bootstrap due to the round on the LWE mask
/// # Argument
/// * `lwe_dimension` - size of the LWE mask
/// # Output
/// * Return the variance of the error
pub fn drift_index_lut(lwe_dimension: usize) -> f64 {
    (lwe_dimension as f64) / 16.0
}

#[warn(dead_code)]
pub fn bootstrap_then_key_switch(
    n: usize,
    l_bsk: usize,
    log_base_bsk: usize,
    stdev_bsk: f64,
    polynomial_size: usize,
    l_ks: usize,
    log_base_ks: usize,
    stdev_ks: f64,
) -> f64 {
    let res_1: f64 = (2 * n * l_bsk * polynomial_size * (1 << (2 * (log_base_bsk - 1)))) as f64
        * f64::powi(stdev_bsk, 2);
    let res_2: f64 = (n * (polynomial_size + 1)) as f64
        * (f64::powi(2.0, -2 * (log_base_bsk * l_bsk + 1) as i32));
    let res_3: f64 = polynomial_size as f64 * f64::powi(2.0, -2 * (log_base_ks * l_ks + 1) as i32);
    let res_4: f64 = polynomial_size as f64
        * l_ks as f64
        * f64::powi(2., log_base_ks as i32 - 1)
        * f64::powi(stdev_ks, 2);
    let res: f64 = res_1 + res_2 + res_3 + res_4;
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_test_bootstrap_then_key_switch() {
        let n: usize = 800; // 630;
        let l_bsk: usize = 3;
        let log_base_bsk: usize = 7;
        let stdev_bsk: f64 = f64::powi(2., -25);
        let polynomial_size: usize = 1 << 10;
        let l_ks: usize = 8;
        let log_base_ks: usize = 3;
        let stdev_ks: f64 = f64::powi(2., -19);
        bootstrap_then_key_switch(
            n,
            l_bsk,
            log_base_bsk,
            stdev_bsk,
            polynomial_size,
            l_ks,
            log_base_ks,
            stdev_ks,
        );
    }
}
