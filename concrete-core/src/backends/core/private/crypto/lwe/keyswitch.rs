#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::numeric::SignedInteger;
use concrete_commons::parameters::{
    CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, LweDimension, LweSize,
};

use crate::backends::core::private::crypto::encoding::{Plaintext, PlaintextList};
use crate::backends::core::private::crypto::secret::generators::EncryptionRandomGenerator;
use crate::backends::core::private::crypto::secret::LweSecretKey;
use crate::backends::core::private::math::decomposition::{
    DecompositionLevel, DecompositionTerm, SignedDecomposer,
};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, ck_dim_eq, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;

use super::{LweCiphertext, LweList};

/// An Lwe Keyswithing key.
///
/// A keyswitching key allows to change the key of a cipher text. Lets assume the following
/// elements:
///
/// + The input key $s_{in}$ is composed of $n$ bits
/// + The output key $s_{out}$ is composed of $m$ bits
///
/// The keyswitch key will be composed of $m$ encryptions of each bits of the $s_{out}$ key, under
/// the key $s_{in}$; encryptions which will be stored as their decomposition over a given basis
/// $B_{ks}\in\mathbb{N}$, up to a level $l_{ks}\in\mathbb{N}$.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweKeyswitchKey<Cont> {
    tensor: Tensor<Cont>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    lwe_size: LweSize,
}

tensor_traits!(LweKeyswitchKey);

impl<Scalar> LweKeyswitchKey<Vec<Scalar>>
where
    Scalar: Copy,
{
    /// Allocates a keyswitching key whose masks and bodies are all `value`.
    ///
    /// # Note
    ///
    /// This function does *not* generate a keyswitch key, but merely allocates a container of the
    /// right size. See [`LweKeyswitchKey::fill_with_keyswitch_key`] to fill the container with a
    /// proper keyswitching key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension, LweSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::lwe::LweKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = LweKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     LweDimension(20),
    /// );
    /// assert_eq!(
    ///     ksk.decomposition_levels_count(),
    ///     DecompositionLevelCount(10)
    /// );
    /// assert_eq!(ksk.decomposition_base_log(), DecompositionBaseLog(16));
    /// assert_eq!(ksk.lwe_size(), LweSize(21));
    /// assert_eq!(ksk.before_key_size(), LweDimension(10));
    /// assert_eq!(ksk.after_key_size(), LweDimension(20));
    /// ```
    pub fn allocate(
        value: Scalar,
        decomp_size: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        input_size: LweDimension,
        output_size: LweDimension,
    ) -> Self {
        LweKeyswitchKey {
            tensor: Tensor::from_container(vec![
                value;
                decomp_size.0 * (output_size.0 + 1) * input_size.0
            ]),
            decomp_base_log,
            decomp_level_count: decomp_size,
            lwe_size: LweSize(output_size.0 + 1),
        }
    }
}

impl<Cont> LweKeyswitchKey<Cont> {
    /// Creates an LWE key switching key from a container.
    ///
    /// # Notes
    ///
    /// This method does not create a keyswitching key, but merely wrap the container in the proper
    /// type. It assumes that either the container already contains a proper keyswitching key, or
    /// that [`LweKeyswitchKey::fill_with_keyswitch_key`] will be called right after.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension, LweSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::lwe::LweKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let input_size = LweDimension(256);
    /// let output_size = LweDimension(35);
    /// let decomp_log_base = DecompositionBaseLog(7);
    /// let decomp_level_count = DecompositionLevelCount(4);
    ///
    /// let ksk = LweKeyswitchKey::from_container(
    ///     vec![0 as u8; input_size.0 * (output_size.0 + 1) * decomp_level_count.0],
    ///     decomp_log_base,
    ///     decomp_level_count,
    ///     output_size,
    /// );
    ///
    /// assert_eq!(ksk.decomposition_levels_count(), DecompositionLevelCount(4));
    /// assert_eq!(ksk.decomposition_base_log(), DecompositionBaseLog(7));
    /// assert_eq!(ksk.lwe_size(), LweSize(36));
    /// assert_eq!(ksk.before_key_size(), LweDimension(256));
    /// assert_eq!(ksk.after_key_size(), LweDimension(35));
    /// ```
    pub fn from_container(
        cont: Cont,
        decomp_base_log: DecompositionBaseLog,
        decomp_size: DecompositionLevelCount,
        output_size: LweDimension,
    ) -> LweKeyswitchKey<Cont>
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => output_size.0 + 1, decomp_size.0);
        LweKeyswitchKey {
            tensor,
            decomp_base_log,
            decomp_level_count: decomp_size,
            lwe_size: LweSize(output_size.0 + 1),
        }
    }

    /// Return the size of the output key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
    /// };
    /// use concrete_core::backends::core::private::crypto::lwe::LweKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = LweKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     LweDimension(20),
    /// );
    /// assert_eq!(ksk.after_key_size(), LweDimension(20));
    /// ```
    pub fn after_key_size(&self) -> LweDimension
    where
        Self: AsRefTensor,
    {
        LweDimension(self.lwe_size.0 - 1)
    }

    /// Returns the size of the ciphertexts encoding each level of the decomposition of each bits
    /// of the input key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension, LweSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::lwe::LweKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = LweKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     LweDimension(20),
    /// );
    /// assert_eq!(ksk.lwe_size(), LweSize(21));
    /// ```
    pub fn lwe_size(&self) -> LweSize
    where
        Self: AsRefTensor,
    {
        self.lwe_size
    }

    /// Returns the size of the input key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
    /// };
    /// use concrete_core::backends::core::private::crypto::lwe::LweKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = LweKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     LweDimension(20),
    /// );
    /// assert_eq!(ksk.before_key_size(), LweDimension(10));
    /// ```
    pub fn before_key_size(&self) -> LweDimension
    where
        Self: AsRefTensor,
    {
        LweDimension(self.as_tensor().len() / (self.lwe_size.0 * self.decomp_level_count.0))
    }

    /// Returns the number of levels used for the decomposition of the input key bits.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
    /// };
    /// use concrete_core::backends::core::private::crypto::lwe::LweKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = LweKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     LweDimension(20),
    /// );
    /// assert_eq!(
    ///     ksk.decomposition_levels_count(),
    ///     DecompositionLevelCount(10)
    /// );
    /// ```
    pub fn decomposition_levels_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        self.decomp_level_count
    }

    /// Returns the logarithm of the base used for the decomposition of the input key bits.
    ///
    /// Indeed, the basis used is always of the form $2^N$. This function returns $N$.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
    /// };
    /// use concrete_core::backends::core::private::crypto::lwe::LweKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = LweKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     LweDimension(20),
    /// );
    /// assert_eq!(ksk.decomposition_base_log(), DecompositionBaseLog(16));
    /// ```
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog
    where
        Self: AsRefTensor,
    {
        self.decomp_base_log
    }

    /// Fills the current keyswitch key container with an actual keyswitching key constructed from
    /// an input and an output key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension, LweSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::lwe::LweKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::LweSecretKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    ///
    /// let input_size = LweDimension(10);
    /// let output_size = LweDimension(20);
    /// let decomp_log_base = DecompositionBaseLog(3);
    /// let decomp_level_count = DecompositionLevelCount(5);
    /// let cipher_size = LweSize(55);
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    ///
    /// let input_key = LweSecretKey::generate_binary(input_size, &mut secret_generator);
    /// let output_key = LweSecretKey::generate_binary(output_size, &mut secret_generator);
    ///
    /// let mut ksk = LweKeyswitchKey::allocate(
    ///     0 as u32,
    ///     decomp_level_count,
    ///     decomp_log_base,
    ///     input_size,
    ///     output_size,
    /// );
    /// ksk.fill_with_keyswitch_key(&input_key, &output_key, noise, &mut encryption_generator);
    ///
    /// assert!(!ksk.as_tensor().iter().all(|a| *a == 0));
    /// ```
    pub fn fill_with_keyswitch_key<InKeyCont, OutKeyCont, Scalar>(
        &mut self,
        before_key: &LweSecretKey<BinaryKeyKind, InKeyCont>,
        after_key: &LweSecretKey<BinaryKeyKind, OutKeyCont>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsMutTensor<Element = Scalar>,
        LweSecretKey<BinaryKeyKind, InKeyCont>: AsRefTensor<Element = Scalar>,
        LweSecretKey<BinaryKeyKind, OutKeyCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We instantiate a buffer
        let mut messages = PlaintextList::from_container(vec![
            <Self as AsMutTensor>::Element::ZERO;
            self.decomp_level_count.0
        ]);

        // We retrieve decomposition arguments
        let decomp_level_count = self.decomp_level_count;
        let decomp_base_log = self.decomp_base_log;

        // loop over the before key blocks
        for (input_key_bit, keyswitch_key_block) in before_key
            .as_tensor()
            .iter()
            .zip(self.bit_decomp_iter_mut())
        {
            // We reset the buffer
            messages
                .as_mut_tensor()
                .fill_with_element(<Self as AsMutTensor>::Element::ZERO);

            // We fill the buffer with the powers of the key bits
            for (level, message) in (1..=decomp_level_count.0)
                .map(DecompositionLevel)
                .zip(messages.plaintext_iter_mut())
            {
                *message = Plaintext(
                    DecompositionTerm::new(level, decomp_base_log, *input_key_bit)
                        .to_recomposition_summand(),
                );
            }

            // We encrypt the buffer
            after_key.encrypt_lwe_list(
                &mut keyswitch_key_block.into_lwe_list(),
                &messages,
                noise_parameters,
                generator,
            );
        }
    }

    /// Iterates over borrowed `LweKeyBitDecomposition` elements.
    ///
    /// One `LweKeyBitDecomposition` being a set of lwe ciphertext, encrypting under the output
    /// key, the $l$ levels of the signed decomposition of a single bit of the input key.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use concrete_core::backends::core::private::crypto::{*, lwe::LweKeyswitchKey};
    /// use concrete_core::backends::core::private::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// let ksk = LweKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(15),
    ///     LweDimension(20)
    /// );
    /// for decomp in ksk.bit_decomp_iter() {
    ///     assert_eq!(decomp.lwe_size(), ksk.lwe_size());
    ///     assert_eq!(decomp.count().0, 10);
    /// }
    /// assert_eq!(ksk.bit_decomp_iter().count(), 15);
    /// ```
    pub(crate) fn bit_decomp_iter(
        &self,
    ) -> impl Iterator<Item = LweKeyBitDecomposition<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.lwe_size.0, self.decomp_level_count.0);
        let size = self.decomp_level_count.0 * self.lwe_size.0;
        let lwe_size = self.lwe_size;
        self.as_tensor()
            .subtensor_iter(size)
            .map(move |sub| LweKeyBitDecomposition::from_container(sub.into_container(), lwe_size))
    }

    /// Iterates over mutably borrowed `LweKeyBitDecomposition` elements.
    ///
    /// One `LweKeyBitDecomposition` being a set of lwe ciphertext, encrypting under the output
    /// key, the $l$ levels of the signed decomposition of a single bit of the input key.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use concrete_core::backends::core::private::crypto::{*, lwe::LweKeyswitchKey};
    /// use concrete_core::backends::core::private::math::tensor::{AsRefTensor, AsMutTensor};
    /// use concrete_core::backends::core::private::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// let mut ksk = LweKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(15),
    ///     LweDimension(20)
    /// );
    /// for mut decomp in ksk.bit_decomp_iter_mut() {
    ///     for mut ciphertext in decomp.ciphertext_iter_mut() {
    ///         ciphertext.as_mut_tensor().fill_with_element(0);
    ///     }
    /// }
    /// assert!(ksk.as_tensor().iter().all(|a| *a == 0));
    /// assert_eq!(ksk.bit_decomp_iter_mut().count(), 15);
    /// ```
    pub(crate) fn bit_decomp_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = LweKeyBitDecomposition<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.lwe_size.0, self.decomp_level_count.0);
        let chunks_size = self.decomp_level_count.0 * self.lwe_size.0;
        let lwe_size = self.lwe_size;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(move |sub| LweKeyBitDecomposition::from_container(sub.into_container(), lwe_size))
    }

    /// Switches the key of a signel Lwe ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension, LweSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::encoding::*;
    /// use concrete_core::backends::core::private::crypto::lwe::*;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::LweSecretKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    ///
    /// let input_size = LweDimension(1024);
    /// let output_size = LweDimension(1024);
    /// let decomp_log_base = DecompositionBaseLog(3);
    /// let decomp_level_count = DecompositionLevelCount(8);
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// let input_key = LweSecretKey::generate_binary(input_size, &mut secret_generator);
    /// let output_key = LweSecretKey::generate_binary(output_size, &mut secret_generator);
    ///
    /// let mut ksk = LweKeyswitchKey::allocate(
    ///     0 as u64,
    ///     decomp_level_count,
    ///     decomp_log_base,
    ///     input_size,
    ///     output_size,
    /// );
    /// ksk.fill_with_keyswitch_key(&input_key, &output_key, noise, &mut encryption_generator);
    ///
    /// let plaintext: Plaintext<u64> = Plaintext(1432154329994324);
    /// let mut ciphertext = LweCiphertext::allocate(0. as u64, LweSize(1025));
    /// let mut switched_ciphertext = LweCiphertext::allocate(0. as u64, LweSize(1025));
    /// input_key.encrypt_lwe(
    ///     &mut ciphertext,
    ///     &plaintext,
    ///     noise,
    ///     &mut encryption_generator,
    /// );
    ///
    /// ksk.keyswitch_ciphertext(&mut switched_ciphertext, &ciphertext);
    ///
    /// let mut decrypted = Plaintext(0 as u64);
    /// output_key.decrypt_lwe(&mut decrypted, &switched_ciphertext);
    /// ```
    pub fn keyswitch_ciphertext<InCont, OutCont, Scalar>(
        &self,
        after: &mut LweCiphertext<OutCont>,
        before: &LweCiphertext<InCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        LweCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
        LweCiphertext<InCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(self.before_key_size().0 => before.get_mask().mask_size().0);
        ck_dim_eq!(self.after_key_size().0 => after.get_mask().mask_size().0);

        // We reset the output
        after.as_mut_tensor().fill_with(|| Scalar::ZERO);

        // We copy the body
        *after.get_mut_body() = *before.get_body();

        // We allocate a boffer to hold the decomposition.
        let mut decomp = Tensor::allocate(Scalar::ZERO, self.decomp_level_count.0);

        // We instantiate a decomposer
        let decomposer = SignedDecomposer::new(self.decomp_base_log, self.decomp_level_count);

        for (block, before_mask) in self
            .bit_decomp_iter()
            .zip(before.get_mask().mask_element_iter())
        {
            let mask_rounded = decomposer.closest_representable(*before_mask);

            torus_small_sign_decompose(decomp.as_mut_slice(), mask_rounded, self.decomp_base_log.0);

            // loop over the number of levels
            for (level_key_cipher, decomposed) in block
                .as_tensor()
                .subtensor_iter(self.after_key_size().0 + 1)
                .zip(decomp.iter())
            {
                after
                    .as_mut_tensor()
                    .update_with_wrapping_sub_element_mul(&level_key_cipher, *decomposed);
            }
        }
    }

    pub fn keyswitch_list<InCont, OutCont, Scalar>(
        &self,
        output: &mut LweList<OutCont>,
        input: &LweList<InCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        LweList<InCont>: AsRefTensor<Element = Scalar>,
        LweList<OutCont>: AsMutTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(input.count().0 => output.count().0);
        // for each ciphertext, call mono_key_switch
        for (input_cipher, mut output_cipher) in
            input.ciphertext_iter().zip(output.ciphertext_iter_mut())
        {
            self.keyswitch_ciphertext(&mut output_cipher, &input_cipher);
        }
    }
}

/// The encryption of a single bit of the output key.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub(crate) struct LweKeyBitDecomposition<Cont> {
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) lwe_size: LweSize,
}

tensor_traits!(LweKeyBitDecomposition);

impl<Cont> LweKeyBitDecomposition<Cont> {
    /// Creates a key bit decomposition from a container.
    ///
    /// # Notes
    ///
    /// This method does not decompose a key bit in a basis, but merely wraps a container in the
    /// right structure. See [`LweKeyswitchKey::bit_decomp_iter`] for an iterator that returns key
    /// bit decompositions.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, lwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// assert_eq!(kbd.count(), CiphertextCount(15));
    /// assert_eq!(kbd.lwe_size(), LweSize(10));
    /// ```
    pub fn from_container(cont: Cont, lwe_size: LweSize) -> Self {
        LweKeyBitDecomposition {
            tensor: Tensor::from_container(cont),
            lwe_size,
        }
    }

    /// Returns the size of the lwe ciphertexts encoding each level of the key bit decomposition.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, lwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// assert_eq!(kbd.lwe_size(), LweSize(10));
    /// ```
    #[allow(dead_code)]
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_size
    }

    /// Returns the number of ciphertexts in the decomposition.
    ///
    /// Note that this is actually equals to the number of levels in the decomposition.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, lwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// assert_eq!(kbd.count(), CiphertextCount(15));
    /// ```
    #[allow(dead_code)]
    pub fn count(&self) -> CiphertextCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.lwe_size.0);
        CiphertextCount(self.as_tensor().len() / self.lwe_size.0)
    }

    /// Returns an iterator over borrowed `LweCiphertext`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, lwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// for ciphertext in kbd.ciphertext_iter(){
    ///     assert_eq!(ciphertext.lwe_size(), LweSize(10));
    /// }
    /// assert_eq!(kbd.ciphertext_iter().count(), 15);
    /// ```
    #[allow(dead_code)]
    pub fn ciphertext_iter(
        &self,
    ) -> impl Iterator<Item = LweCiphertext<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        self.as_tensor()
            .subtensor_iter(self.lwe_size.0)
            .map(|sub| LweCiphertext::from_container(sub.into_container()))
    }

    /// Returns an iterator over mutably borrowed `LweCiphertext`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, lwe::LweKeyBitDecomposition};
    /// use concrete_core::backends::core::private::math::tensor::{AsRefTensor, AsMutTensor};
    /// let mut kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// for mut ciphertext in kbd.ciphertext_iter_mut(){
    ///     ciphertext.as_mut_tensor().fill_with_element(9);
    /// }
    /// assert!(kbd.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(kbd.ciphertext_iter().count(), 15);
    /// ```
    #[allow(dead_code)]
    pub fn ciphertext_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = LweCiphertext<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.lwe_size.0;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(|sub| LweCiphertext::from_container(sub.into_container()))
    }

    /// Consumes the current key bit decomposition and returns an lwe list.
    ///
    /// Note that this operation is super cheap, as it merely rewraps the current container in an
    /// lwe list structure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, lwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// let list = kbd.into_lwe_list();
    /// assert_eq!(list.count(), CiphertextCount(15));
    /// assert_eq!(list.lwe_size(), LweSize(10));
    /// ```
    pub fn into_lwe_list(self) -> LweList<Cont> {
        LweList {
            tensor: self.tensor,
            lwe_size: self.lwe_size,
        }
    }
}

fn torus_small_sign_decompose<Scalar>(res: &mut [Scalar], val: Scalar, base_log: usize)
where
    Scalar: UnsignedTorus,
    Scalar::Signed: SignedInteger,
{
    let mut tmp: Scalar;
    let mut carry = Scalar::ZERO;
    let mut previous_carry: Scalar;
    let block_bit_mask: Scalar = (Scalar::ONE << base_log) - Scalar::ONE;
    let msb_block_mask: Scalar = Scalar::ONE << (base_log - 1);

    // compute the decomposition from LSB to MSB (because of the carry)
    for i in (0..res.len()).rev() {
        previous_carry = carry;
        tmp = (val >> (Scalar::BITS - base_log * (i + 1))) & block_bit_mask;
        carry = tmp & msb_block_mask;
        tmp = tmp.wrapping_add(previous_carry);
        carry |= tmp & msb_block_mask; // 0000...0001000 or 0000...0000000
        res[i] = ((tmp.into_signed()) - ((carry << 1).into_signed())).into_unsigned();
        carry >>= base_log - 1; // 000...0001 or 000...0000
    }
}
