//! Noise formulas for the gsw-sample related operations
//! Those functions will be used in the gsw tests to check that
//! the noise behavior is consistent with the theory.

pub trait GSW: Sized {
    type STorus;
    fn external_product(
        dimension: usize,
        l_gadget: usize,
        base_log: usize,
        var_gsw: f64,
        var_lwe: f64,
    ) -> f64;

    fn cmux(
        var_lwe_0: f64,
        var_lwe_1: f64,
        var_gsw: f64,
        dimension: usize,
        base_log: usize,
        l_gadget: usize,
    ) -> f64;
}

macro_rules! impl_trait_npe_gsw {
    ($T:ty,$S:ty,$DOC:expr) => {
        impl GSW for $T {
            type STorus = $S;
            /// Return the variance of the external product given a set of parameters.
            /// To see how to use it, please refer to the test of the external product.
            /// Arguments
            /// * `dimension` - the size of the LWE mask
            /// * `l_gadget` - number of elements for the Torus decomposition
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `var_gsw` - noise variance of the GSW
            /// * `var_lwe` - noise variance of the LWE
            /// # Output
            /// * Returns the variance of the output LWE
            /// # Warning
            /// * only correct for the external product inside a cmux
            /// # Example
            /// ```rust
            /// use concrete_npe::GSW ;
            #[doc = $DOC]
            /// // settings
            /// let dimension: usize = 256 ;
            /// let l_gadget: usize = 4 ;
            /// let base_log: usize = 7 ;
            /// let var_gsw: f64 = f64::powi(2., -38) ;
            /// let var_lwe: f64 = f64::powi(2., -40) ;
            /// // Computing the noise
            /// let var_external_product = <Torus as GSW>::external_product(dimension, l_gadget,
            /// base_log, var_gsw, var_lwe) ;
            /// ```
            fn external_product(
                dimension: usize,
                l_gadget: usize,
                base_log: usize,
                var_gsw: f64,
                var_lwe: f64,
            ) -> f64 {
                // norm 2 of the integer hidden in the GSW
                // for an external product inside a cmux,
                // the integer is equal to 0 or 1
                let norm_2_msg_gsw = 1.;
                let b_g = 1 << base_log;
                let q_square = f64::powi(2., (2 * std::mem::size_of::<$T>() * 8) as i32);
                let res_1: f64 =
                    ((dimension + 1) * l_gadget * (b_g * b_g + 2)) as f64 / 12. * var_gsw;

                let res_2: f64 = norm_2_msg_gsw
                    * ((dimension + 2) as f64
                        / (24. * f64::powi(b_g as f64, 2 * l_gadget as i32)) as f64
                        + (dimension / 48 - 1 / 12) as f64 / q_square);

                let res_3: f64 = norm_2_msg_gsw * var_lwe;
                let res: f64 = res_1 + res_2 + res_3;
                return res;
            }

            /// Return the variance of the cmux given a set of parameters.
            /// To see how to use it, please refer to the test of the cmux.
            /// Arguments
            /// * `var_lwe_0` - noise variance of the first LWE
            /// * `var_lwe_1` - noise variance of the second LWE
            /// * `var_gsw` - noise variance of the GSW
            /// * `dimension` - the size of the LWE mask
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `l_gadget` - number of elements for the Torus decomposition
            /// # Output
            /// * Returns the variance of the output LWE
            /// # Example
            /// ```rust
            /// use concrete_npe::GSW ;
            #[doc = $DOC]
            /// // settings
            /// let dimension: usize = 256 ;
            /// let l_gadget: usize = 4 ;
            /// let base_log: usize = 7 ;
            /// let var_gsw: f64 = f64::powi(2., -38) ;
            /// let var_lwe_0: f64 = f64::powi(2., -40) ;
            /// let var_lwe_1: f64 = f64::powi(2., -40) ;
            /// // Computing the noise
            /// let var_cmux = <Torus as GSW>::cmux(var_lwe_0, var_lwe_1, var_gsw,
            /// dimension, base_log, l_gadget) ;
            /// ```
            fn cmux(
                var_lwe_0: f64,
                var_lwe_1: f64,
                var_gsw: f64,
                dimension: usize,
                base_log: usize,
                l_gadget: usize,
            ) -> f64 {
                let var_external_product = Self::external_product(
                    dimension,
                    l_gadget,
                    base_log,
                    var_gsw,
                    crate::add_ciphertexts(var_lwe_0, var_lwe_1),
                );
                let var_cmux = crate::add_ciphertexts(var_external_product, var_lwe_0);
                return var_cmux;
            }
        }
    };
}

impl_trait_npe_gsw!(u32, i32, "type Torus = u32;");
impl_trait_npe_gsw!(u64, i64, "type Torus = u64;");
