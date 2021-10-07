use crate::Torus;
use concrete_core::{
    crypto,
    math::tensor::Tensor,
    math::tensor::{AsMutTensor, AsRefTensor},
};
use serde::{Deserialize, Serialize};
use std::fmt;
use concrete_commons::dispersion::StandardDev;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};
use concrete_core::crypto::secret::generators::EncryptionRandomGenerator;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct LWEKSK {
    pub ciphertexts: crypto::lwe::LweKeyswitchKey<Vec<Torus>>,
    pub base_log: usize,
    pub level: usize,
    pub dimension_before: usize,
    pub dimension_after: usize,
    pub variance: f64,
}

impl LWEKSK {
    /// Generate an empty LWE key switching key
    ///
    /// # Argument
    /// * `sk_before` - an LWE secret key (input for the key switch)
    /// * `sk_after` - an LWE secret key (output for the key switch)
    /// * `base_log` - the log2 of the decomposition base
    /// * `level` - the number of levels of the decomposition
    ///
    /// # Output
    /// * an LWEKSK
    pub fn zero(
        sk_before: &crate::LWESecretKey,
        sk_after: &crate::LWESecretKey,
        base_log: usize,
        level: usize,
    ) -> LWEKSK {
        let ksk_ciphertexts = crypto::lwe::LweKeyswitchKey::allocate(
            0,
            DecompositionLevelCount(level),
            DecompositionBaseLog(base_log),
            LweDimension(sk_before.dimension),
            LweDimension(sk_after.dimension),
        );

        LWEKSK {
            ciphertexts: ksk_ciphertexts,
            base_log,
            level,
            dimension_before: sk_before.dimension,
            dimension_after: sk_after.dimension,
            variance: f64::powi(sk_after.std_dev, 2),
        }
    }

    /// Generate a valid LWE key switching key
    /// # Argument
    /// * `sk_before` - an LWE secret key (input for the key switch)
    /// * `sk_after` - an LWE secret key (output for the key switch)
    /// * `base_log` - the log2 of the decomposition base
    /// * `level` - the number of levels of the decomposition
    ///
    /// # Output
    /// * an LWEKSK
    pub fn new(
        sk_before: &crate::LWESecretKey,
        sk_after: &crate::LWESecretKey,
        base_log: usize,
        level: usize,
    ) -> LWEKSK {
        let mut ksk_ciphertexts = crypto::lwe::LweKeyswitchKey::allocate(
            0_u64,
            DecompositionLevelCount(level),
            DecompositionBaseLog(base_log),
            LweDimension(sk_before.dimension),
            LweDimension(sk_after.dimension),
        );

        ksk_ciphertexts.fill_with_keyswitch_key(
            &sk_before.val,
            &sk_after.val,
            StandardDev::from_standard_dev(sk_after.std_dev),
            &mut EncryptionRandomGenerator::new(None),
        );

        LWEKSK {
            ciphertexts: ksk_ciphertexts,
            base_log,
            level,
            dimension_before: sk_before.dimension,
            dimension_after: sk_after.dimension,
            variance: f64::powi(sk_after.std_dev, 2),
        }
    }

    pub fn save(&self, path: &str) {
        let mut tensor = Tensor::allocate(0, self.ciphertexts.as_tensor().len() + 6);

        *tensor.get_element_mut(0) = self.variance.to_bits();
        *tensor.get_element_mut(1) = self.dimension_before as u64;
        *tensor.get_element_mut(2) = self.dimension_after as u64;
        *tensor.get_element_mut(3) = self.base_log as u64;
        *tensor.get_element_mut(4) = self.level as u64;
        *tensor.get_element_mut(5) = self.ciphertexts.as_tensor().len() as u64;

        for (dst, src) in tensor
            .get_sub_mut(6..(self.ciphertexts.as_tensor().len() + 6))
            .iter_mut()
            .zip(self.ciphertexts.as_tensor().iter())
        {
            *dst = *src;
        }
        tensor.save_to_file(path).unwrap();
    }

    pub fn load(path: &str) -> crate::LWEKSK {
        let tensor: Tensor<Vec<u64>> = Tensor::load_from_file(path).unwrap();

        let mut res = crate::LWEKSK {
            variance: f64::from_bits(*tensor.get_element(0)),
            dimension_before: *tensor.get_element(1) as usize,
            dimension_after: *tensor.get_element(2) as usize,
            base_log: *tensor.get_element(3) as usize,
            level: *tensor.get_element(4) as usize,
            ciphertexts: crypto::lwe::LweKeyswitchKey::allocate(
                0,
                DecompositionLevelCount(*tensor.get_element(4) as usize),
                DecompositionBaseLog(*tensor.get_element(3) as usize),
                LweDimension(*tensor.get_element(1) as usize),
                LweDimension(*tensor.get_element(2) as usize),
            ),
        };

        res.ciphertexts
            .as_mut_tensor()
            .fill_with_one(&tensor.get_sub(6..), |v| *v);

        res
    }
}

/// Print needed pieces of information about an LWEKSK
impl fmt::Display for LWEKSK {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = 2;
        let mut to_be_print: String = "".to_string();

        to_be_print += " LWEBSK {\n         -> samples = [";

        if self.ciphertexts.as_tensor().len() <= 2 * n {
            for elt in self.ciphertexts.as_tensor().iter() {
                to_be_print = to_be_print + &format!("{}, ", *elt);
            }
        } else {
            for elt in self.ciphertexts.as_tensor().get_sub(0..n).iter() {
                to_be_print = to_be_print + &format!("{}, ", *elt);
            }
            to_be_print += "...";

            for elt in self
                .ciphertexts
                .as_tensor()
                .get_sub(self.ciphertexts.as_tensor().len() - n..)
                .iter()
            {
                to_be_print = to_be_print + &format!("{}, ", *elt);
            }
        }
        to_be_print += "]\n";

        to_be_print = to_be_print + &format!("         -> variance = {}\n", self.variance);
        to_be_print =
            to_be_print + &format!("         -> dimension before = {}\n", self.dimension_before);
        to_be_print =
            to_be_print + &format!("         -> dimension after = {}\n", self.dimension_after);

        to_be_print = to_be_print + &format!("         -> base_log = {}\n", self.base_log);
        to_be_print = to_be_print + &format!("         -> level = {}\n", self.level);
        to_be_print += "       }";
        writeln!(f, "{}", to_be_print)
    }
}
