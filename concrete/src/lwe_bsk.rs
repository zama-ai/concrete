use std::fmt;

use backtrace::Backtrace;
use colored::Colorize;

use concrete_core::crypto::LweDimension;
use concrete_core::math::decomposition::{DecompositionBaseLog, DecompositionLevelCount};
use concrete_core::math::dispersion::StandardDev;
use concrete_core::math::polynomial::PolynomialSize;
use concrete_core::{
    crypto::{bootstrap::BootstrapKey, GlweSize},
    math::tensor::{AsMutTensor, AsRefTensor},
    math::{fft::Complex64, tensor::Tensor},
    numeric::Numeric,
};

use crate::error::CryptoAPIError;
use crate::Torus;

#[derive(Debug, PartialEq, Clone)]
pub struct LWEBSK {
    pub ciphertexts: BootstrapKey<Vec<Complex64>>,
    pub variance: f64,
    pub dimension: usize,
    pub polynomial_size: usize,
    pub base_log: usize,
    pub level: usize,
}

impl LWEBSK {
    /// Return the dimension of an LWE we can bootstrap with this key
    pub fn get_lwe_dimension(&self) -> usize {
        self.ciphertexts.as_tensor().len()
            / (usize::pow(self.dimension + 1, 2) * self.level * self.polynomial_size)
    }

    /// Return the log2 of the polynomial size of the RLWE involved in the bootstrap
    pub fn get_polynomial_size_log(&self) -> usize {
        f64::log2(self.polynomial_size as f64) as usize
    }

    /// Build a lookup table af a function from two encoders
    ///
    /// # Argument
    /// * `encoder_input` - the encoder of the input (of the bootstrap)
    /// * `encoder_output` - the encoder of the output (of the bootstrap)
    /// * `f` - a function
    ///
    /// # Output
    /// * a slice of Torus containing the lookup table
    pub fn generate_functional_look_up_table<F: Fn(f64) -> f64>(
        &self,
        encoder_input: &crate::Encoder,
        encoder_output: &crate::Encoder,
        f: F,
    ) -> Result<Vec<Torus>, CryptoAPIError> {
        // check that precision != 0
        if encoder_input.nb_bit_precision == 0 {
            return Err(PrecisionError!());
        }

        // check that the input encoder has at least 1 bit of padding
        if encoder_input.nb_bit_padding == 0 {
            return Err(NotEnoughPaddingError!(encoder_input.nb_bit_padding, 1));
        }

        // clone the input encoder and set nb_bit_padding to 1
        let mut encoder_input_clone = encoder_input.clone();
        encoder_input_clone.nb_bit_padding = 1;

        // allocation of the result
        let mut result: Vec<Torus> = vec![0; self.polynomial_size];

        // find the right index to start storing -val_i instead of val_i
        let minus_start_index: usize =
            self.polynomial_size - (self.polynomial_size >> (1 + encoder_input.nb_bit_precision));

        for (i, res) in result.iter_mut().enumerate() {
            // create a valid encoding from i
            let shift: usize = <Torus as Numeric>::BITS - self.get_polynomial_size_log() - 1;
            let encoded: Torus = (i as Torus) << shift;

            // decode the encoding
            let decoded: f64 = encoder_input_clone.decode_core(encoded)?;

            // apply the function
            let f_decoded: f64 = f(decoded);

            // encode the result
            let output_encoded: Torus =
                encoder_output.encode_outside_interval_operators(f_decoded)?;

            *res = if i < minus_start_index {
                output_encoded
            } else {
                output_encoded.wrapping_neg()
            };
        }
        Ok(result)
    }

    /// Build a lookup table for the identity function from two encoders
    ///
    /// # Argument
    /// * `encoder_input` - the encoder of the input (of the bootstrap)
    /// * `encoder_output` - the encoder of the output (of the bootstrap)
    ///
    /// # Output
    /// * a slice of Torus containing the lookup table
    pub fn generate_identity_look_up_table(
        &self,
        encoder_input: &crate::Encoder,
        encoder_output: &crate::Encoder,
    ) -> Result<Vec<Torus>, CryptoAPIError> {
        self.generate_functional_look_up_table(encoder_input, encoder_output, |x| x)
    }

    /// Create a valid bootstrapping key
    ///
    /// # Argument
    /// * `sk_before` - an LWE secret key (input for the bootstrap)
    /// * `sk_after` - an LWE secret key (output for the bootstrap)
    /// * `base_log` - the log2 of the decomposition base
    /// * `level` - the number of levels of the decomposition
    ///
    /// # Output
    /// * an LWEBSK
    pub fn new(
        sk_input: &crate::LWESecretKey,
        sk_output: &crate::RLWESecretKey,
        base_log: usize,
        level: usize,
    ) -> LWEBSK {
        // allocation for the bootstrapping key
        let mut coef_bsk = BootstrapKey::allocate(
            0_u64,
            GlweSize(sk_output.val.key_size().0 + 1),
            sk_output.val.polynomial_size(),
            DecompositionLevelCount(level),
            DecompositionBaseLog(base_log),
            sk_input.val.key_size(),
        );
        coef_bsk.fill_with_new_key(
            &sk_input.val,
            &sk_output.val,
            StandardDev::from_standard_dev(sk_output.std_dev),
        );
        let mut fourier_bsk = BootstrapKey::allocate(
            Complex64::new(0., 0.),
            GlweSize(sk_output.val.key_size().0 + 1),
            sk_output.val.polynomial_size(),
            DecompositionLevelCount(level),
            DecompositionBaseLog(base_log),
            sk_input.val.key_size(),
        );
        fourier_bsk.fill_with_forward_fourier(&coef_bsk);

        LWEBSK {
            ciphertexts: fourier_bsk,
            variance: f64::powi(sk_output.std_dev, 2),
            dimension: sk_output.dimension,
            polynomial_size: sk_output.polynomial_size,
            base_log,
            level,
        }
    }

    /// Create an empty bootstrapping key
    ///
    /// # Argument
    /// * `sk_before` - an LWE secret key (input for the bootstrap)
    /// * `sk_after` - an LWE secret key (output for the bootstrap)
    /// * `base_log` - the log2 of the decomposition base
    /// * `level` - the number of levels of the decomposition
    ///
    /// # Output
    /// * an LWEBSK
    pub fn zero(
        sk_input: &crate::LWESecretKey,
        sk_output: &crate::RLWESecretKey,
        base_log: usize,
        level: usize,
    ) -> LWEBSK {
        // allocation for the bootstrapping key
        let fourier_bsk = BootstrapKey::allocate(
            Complex64::new(0., 0.),
            GlweSize(sk_output.val.key_size().0 + 1),
            sk_output.val.polynomial_size(),
            DecompositionLevelCount(level),
            DecompositionBaseLog(base_log),
            sk_input.val.key_size(),
        );

        LWEBSK {
            ciphertexts: fourier_bsk,
            variance: f64::powi(sk_output.std_dev, 2),
            dimension: sk_output.dimension,
            polynomial_size: sk_output.polynomial_size,
            base_log,
            level,
        }
    }

    pub fn save(&self, path: &str) {
        let mut tensor = Tensor::allocate(0, self.ciphertexts.as_tensor().len() * 2 + 6);

        *tensor.get_element_mut(0) = self.variance.to_bits();
        *tensor.get_element_mut(1) = self.dimension as u64;
        *tensor.get_element_mut(2) = self.polynomial_size as u64;
        *tensor.get_element_mut(3) = self.base_log as u64;
        *tensor.get_element_mut(4) = self.level as u64;
        *tensor.get_element_mut(5) = self.ciphertexts.key_size().0 as u64;

        for (mut couple, c) in tensor
            .get_sub_mut(6..(self.ciphertexts.as_tensor().len() * 2 + 6))
            .subtensor_iter_mut(2)
            .zip(self.ciphertexts.as_tensor().iter())
        {
            *couple.get_element_mut(0) = c.re.to_bits();
            *couple.get_element_mut(1) = c.im.to_bits();
        }

        tensor.save_to_file(path).unwrap();
    }

    pub fn load(path: &str) -> crate::LWEBSK {
        let tensor = Tensor::load_from_file(path).expect("Failed to load file");

        let mut res = crate::LWEBSK {
            variance: f64::from_bits(*tensor.get_element(0)),
            dimension: *tensor.get_element(1) as usize,
            polynomial_size: *tensor.get_element(2) as usize,
            base_log: *tensor.get_element(3) as usize,
            level: *tensor.get_element(4) as usize,
            ciphertexts: BootstrapKey::allocate(
                Complex64::new(0., 0.),
                GlweSize(*tensor.get_element(1) as usize + 1),
                PolynomialSize(*tensor.get_element(2) as usize),
                DecompositionLevelCount(*tensor.get_element(4) as usize),
                DecompositionBaseLog(*tensor.get_element(3) as usize),
                LweDimension(*tensor.get_element(5) as usize),
            ),
        };

        for (couple, c) in tensor
            .get_sub(6..)
            .subtensor_iter(2)
            .zip(res.ciphertexts.as_mut_tensor().iter_mut())
        {
            *c = Complex64::new(
                f64::from_bits(*couple.get_element(0)),
                f64::from_bits(*couple.get_element(1)),
            );
        }
        res
    }
}

/// Print needed pieces of information about an LWEBSK
impl fmt::Display for LWEBSK {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = 2;
        let mut to_be_print: String = "".to_string();

        to_be_print += " LWEBSK {\n         -> samples = [";

        if self.ciphertexts.as_tensor().len() <= 2 * n {
            for elt in self.ciphertexts.as_tensor().iter() {
                to_be_print += &format!("{}, ", elt);
            }
            to_be_print += "]\n";
        } else {
            for elt in self.ciphertexts.as_tensor().get_sub(0..n).iter() {
                to_be_print += &format!("{}, ", elt);
            }
            to_be_print += "...";

            for elt in self
                .ciphertexts
                .as_tensor()
                .get_sub(self.ciphertexts.as_tensor().len() - n..)
                .iter()
            {
                to_be_print += &format!("{}, ", elt);
            }
            to_be_print += "]\n";
        }

        to_be_print += &format!("         -> variance = {}\n", self.variance);
        to_be_print += &format!("         -> dimension = {}\n", self.dimension);
        to_be_print =
            to_be_print + &format!("         -> polynomial_size = {}\n", self.polynomial_size);
        to_be_print += &format!("         -> base_log = {}\n", self.base_log);
        to_be_print += &format!("         -> level = {}\n", self.level);
        to_be_print += "       }";
        writeln!(f, "{}", to_be_print)
    }
}
