#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{
    CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize,
    LweDimension, MonomialDegree, PlaintextCount, PolynomialSize,
};

use crate::backends::core::private::crypto::encoding::PlaintextList;
use crate::backends::core::private::crypto::lwe::{LweCiphertext, LweList};
use crate::backends::core::private::crypto::secret::generators::EncryptionRandomGenerator;
use crate::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
use crate::backends::core::private::math::decomposition::{
    torus_small_sign_decompose, DecompositionLevel, DecompositionTerm, SignedDecomposer,
};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, ck_dim_eq, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;

use super::{GlweCiphertext, GlweList};

/// A packing keyswitching key.
///
/// A packing keyswitching key allows to  pack several LWE ciphertexts into a single GLWE
/// ciphertext.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackingKeyswitchKey<Cont> {
    tensor: Tensor<Cont>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    output_glwe_size: GlweSize,
    output_polynomial_size: PolynomialSize,
}

tensor_traits!(PackingKeyswitchKey);

impl<Scalar> PackingKeyswitchKey<Vec<Scalar>>
where
    Scalar: Copy,
{
    /// Allocates a packing keyswitching key whose masks and bodies are all `value`.
    ///
    /// # Note
    ///
    /// This function does *not* generate a keyswitch key, but merely allocates a container of the
    /// right size. See [`PackingKeyswitchKey::fill_with_keyswitch_key`] to fill the container with
    /// a proper keyswitching key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, LweDimension,
    ///     LweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let pksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(
    ///     pksk.decomposition_level_count(),
    ///     DecompositionLevelCount(10)
    /// );
    /// assert_eq!(pksk.decomposition_base_log(), DecompositionBaseLog(16));
    /// assert_eq!(pksk.output_glwe_key_dimension(), GlweDimension(2));
    /// assert_eq!(pksk.input_lwe_key_dimension(), LweDimension(10));
    /// ```
    pub fn allocate(
        value: Scalar,
        decomp_size: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        input_dimension: LweDimension,
        output_dimension: GlweDimension,
        output_polynomial_size: PolynomialSize,
    ) -> Self {
        PackingKeyswitchKey {
            tensor: Tensor::from_container(vec![
                value;
                decomp_size.0
                    * output_dimension.to_glwe_size().0
                    * output_polynomial_size.0
                    * input_dimension.0
            ]),
            decomp_base_log,
            decomp_level_count: decomp_size,
            output_glwe_size: output_dimension.to_glwe_size(),
            output_polynomial_size,
        }
    }
}

impl<Cont> PackingKeyswitchKey<Cont> {
    /// Creates a packing keyswitching key from a container.
    ///
    /// # Notes
    ///
    /// This method does not create a packing keyswitch key, but merely wraps the container in
    /// the proper type. It assumes that either the container already contains a proper keyswitching
    /// key, or that [`PackingKeyswitchKey::fill_with_keyswitch_key`] will be called right
    /// after.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, LweDimension,
    ///     LweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let input_size = LweDimension(200);
    /// let output_size = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(256);
    /// let decomp_log_base = DecompositionBaseLog(7);
    /// let decomp_level_count = DecompositionLevelCount(4);
    ///
    /// let pksk = PackingKeyswitchKey::from_container(
    ///     vec![
    ///         0 as u8;
    ///         input_size.0 * (output_size.0 + 1) * polynomial_size.0 * decomp_level_count.0
    ///     ],
    ///     decomp_log_base,
    ///     decomp_level_count,
    ///     output_size,
    ///     polynomial_size,
    /// );
    ///
    /// assert_eq!(pksk.decomposition_level_count(), DecompositionLevelCount(4));
    /// assert_eq!(pksk.decomposition_base_log(), DecompositionBaseLog(7));
    /// assert_eq!(pksk.output_glwe_key_dimension(), GlweDimension(2));
    /// assert_eq!(pksk.input_lwe_key_dimension(), LweDimension(200));
    /// ```
    pub fn from_container(
        cont: Cont,
        decomp_base_log: DecompositionBaseLog,
        decomp_size: DecompositionLevelCount,
        output_glwe_dimension: GlweDimension,
        output_polynomial_size: PolynomialSize,
    ) -> PackingKeyswitchKey<Cont>
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => output_glwe_dimension.to_glwe_size().0 * output_polynomial_size.0, decomp_size.0);
        PackingKeyswitchKey {
            tensor,
            decomp_base_log,
            decomp_level_count: decomp_size,
            output_glwe_size: output_glwe_dimension.to_glwe_size(),
            output_polynomial_size,
        }
    }

    /// Returns the dimension of the output GLWE key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let pksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(pksk.output_glwe_key_dimension(), GlweDimension(2));
    /// ```
    pub fn output_glwe_key_dimension(&self) -> GlweDimension {
        self.output_glwe_size.to_glwe_dimension()
    }

    /// Returns the size of the polynomials composing the GLWE ciphertext
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, LweSize,
    ///     PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let pksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(pksk.output_polynomial_size(), PolynomialSize(256));
    /// ```
    pub fn output_polynomial_size(&self) -> PolynomialSize {
        self.output_polynomial_size
    }

    /// Returns the dimension of the input LWE key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let pksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(pksk.input_lwe_key_dimension(), LweDimension(10));
    /// ```
    pub fn input_lwe_key_dimension(&self) -> LweDimension
    where
        Self: AsRefTensor,
    {
        LweDimension(
            self.as_tensor().len()
                / (self.output_glwe_size.0
                    * self.output_polynomial_size.0
                    * self.decomp_level_count.0),
        )
    }

    /// Returns the number of levels used for the decomposition of the input key bits.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let pksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(
    ///     pksk.decomposition_level_count(),
    ///     DecompositionLevelCount(10)
    /// );
    /// ```
    pub fn decomposition_level_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        self.decomp_level_count
    }

    /// Returns the logarithm of the base used for the decomposition of the input key bits.
    ///
    /// Indeed, the basis used is always of the form $2^b$. This function returns $b$.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let pksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(pksk.decomposition_base_log(), DecompositionBaseLog(16));
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
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, LweSize,
    ///     PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    ///
    /// let input_size = LweDimension(10);
    /// let output_size = GlweDimension(3);
    /// let polynomial_size = PolynomialSize(256);
    /// let decomp_log_base = DecompositionBaseLog(3);
    /// let decomp_level_count = DecompositionLevelCount(5);
    /// let cipher_size = LweSize(55);
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    ///
    /// let input_key = LweSecretKey::generate_binary(input_size, &mut secret_generator);
    /// let output_key =
    ///     GlweSecretKey::generate_binary(output_size, polynomial_size, &mut secret_generator);
    ///
    /// let mut pksk = PackingKeyswitchKey::allocate(
    ///     0 as u32,
    ///     decomp_level_count,
    ///     decomp_log_base,
    ///     input_size,
    ///     output_size,
    ///     polynomial_size,
    /// );
    /// pksk.fill_with_packing_keyswitch_key(&input_key, &output_key, noise, &mut encryption_generator);
    ///
    /// assert!(!pksk.as_tensor().iter().all(|a| *a == 0));
    /// ```
    pub fn fill_with_packing_keyswitch_key<InKeyCont, OutKeyCont, Scalar>(
        &mut self,
        input_lwe_key: &LweSecretKey<BinaryKeyKind, InKeyCont>,
        output_glwe_key: &GlweSecretKey<BinaryKeyKind, OutKeyCont>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsMutTensor<Element = Scalar>,
        LweSecretKey<BinaryKeyKind, InKeyCont>: AsRefTensor<Element = Scalar>,
        GlweSecretKey<BinaryKeyKind, OutKeyCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We instantiate a buffer
        let mut messages = PlaintextList::from_container(vec![
            <Self as AsMutTensor>::Element::ZERO;
            self.decomp_level_count.0
                * self.output_polynomial_size.0
        ]);

        // We retrieve decomposition arguments
        let decomp_level_count = self.decomp_level_count;
        let decomp_base_log = self.decomp_base_log;
        let polynomial_size = self.output_polynomial_size;

        // loop over the before key blocks
        for (input_key_bit, keyswitch_key_block) in input_lwe_key
            .as_tensor()
            .iter()
            .zip(self.bit_decomp_iter_mut())
        {
            // We reset the buffer
            messages
                .as_mut_tensor()
                .fill_with_element(<Self as AsMutTensor>::Element::ZERO);

            // We fill the buffer with the powers of the key bits
            for (level, mut message) in (1..=decomp_level_count.0)
                .map(DecompositionLevel)
                .zip(messages.sublist_iter_mut(PlaintextCount(polynomial_size.0)))
            {
                *message.as_mut_tensor().first_mut() =
                    DecompositionTerm::new(level, decomp_base_log, *input_key_bit)
                        .to_recomposition_summand();
            }

            // We encrypt the buffer
            output_glwe_key.encrypt_glwe_list(
                &mut keyswitch_key_block.into_glwe_list(),
                &messages,
                noise_parameters,
                generator,
            );
        }
    }

    /// Iterates over borrowed `LweKeyBitDecomposition` elements.
    ///
    /// One `LweKeyBitDecomposition` being a set of LWE ciphertexts, encrypting under the output
    /// key, the $l$ levels of the signed decomposition of a single bit of the input key.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::PackingKeyswitchKey};
    /// use concrete_core::backends::core::private::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// let pksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(15),
    ///     LweDimension(20)
    /// );
    /// for decomp in pksk.bit_decomp_iter() {
    ///     assert_eq!(decomp.lwe_size(), pksk.lwe_size());
    ///     assert_eq!(decomp.count().0, 10);
    /// }
    /// assert_eq!(pksk.bit_decomp_iter().count(), 15);
    /// ```
    pub(crate) fn bit_decomp_iter(
        &self,
    ) -> impl Iterator<Item = LweKeyBitDecomposition<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.output_glwe_size.0 * self.output_polynomial_size.0, self.decomp_level_count.0);
        let size =
            self.decomp_level_count.0 * self.output_glwe_size.0 * self.output_polynomial_size.0;
        let glwe_size = self.output_glwe_size;
        let poly_size = self.output_polynomial_size;
        self.as_tensor().subtensor_iter(size).map(move |sub| {
            LweKeyBitDecomposition::from_container(sub.into_container(), glwe_size, poly_size)
        })
    }

    /// Iterates over mutably borrowed `LweKeyBitDecomposition` elements.
    ///
    /// One `LweKeyBitDecomposition` being a set of LWE ciphertexts, encrypting under the output
    /// key, the $l$ levels of the signed decomposition of a single bit of the input key.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::PackingKeyswitchKey};
    /// use concrete_core::backends::core::private::math::tensor::{AsRefTensor, AsMutTensor};
    /// use concrete_core::backends::core::private::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// let mut pksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(15),
    ///     LweDimension(20)
    /// );
    /// for mut decomp in pksk.bit_decomp_iter_mut() {
    ///     for mut ciphertext in decomp.ciphertext_iter_mut() {
    ///         ciphertext.as_mut_tensor().fill_with_element(0);
    ///     }
    /// }
    /// assert!(pksk.as_tensor().iter().all(|a| *a == 0));
    /// assert_eq!(pksk.bit_decomp_iter_mut().count(), 15);
    /// ```
    pub(crate) fn bit_decomp_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = LweKeyBitDecomposition<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.output_glwe_size.0 * self.output_polynomial_size.0, self.decomp_level_count.0);
        let chunks_size =
            self.decomp_level_count.0 * self.output_glwe_size.0 * self.output_polynomial_size.0;
        let glwe_size = self.output_glwe_size;
        let poly_size = self.output_polynomial_size;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(move |sub| {
                LweKeyBitDecomposition::from_container(sub.into_container(), glwe_size, poly_size)
            })
    }

    /// Keyswitches a single LWE ciphertext into a GLWE
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, LweDimension,
    ///     LweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::encoding::*;
    /// use concrete_core::backends::core::private::crypto::glwe::*;
    /// use concrete_core::backends::core::private::crypto::lwe::*;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    ///
    /// let input_size = LweDimension(1024);
    /// let output_size = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(256);
    /// let decomp_log_base = DecompositionBaseLog(3);
    /// let decomp_level_count = DecompositionLevelCount(8);
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// let input_key = LweSecretKey::generate_binary(input_size, &mut secret_generator);
    /// let output_key =
    ///     GlweSecretKey::generate_binary(output_size, polynomial_size, &mut secret_generator);
    ///
    /// let mut pksk = PackingKeyswitchKey::allocate(
    ///     0 as u64,
    ///     decomp_level_count,
    ///     decomp_log_base,
    ///     input_size,
    ///     output_size,
    ///     polynomial_size,
    /// );
    /// pksk.fill_with_packing_keyswitch_key(&input_key, &output_key, noise, &mut encryption_generator);
    ///
    /// let plaintext: Plaintext<u64> = Plaintext(1432154329994324);
    /// let mut ciphertext = LweCiphertext::allocate(0. as u64, LweSize(1025));
    /// let mut switched_ciphertext =
    ///     GlweCiphertext::allocate(0. as u64, PolynomialSize(256), GlweSize(3));
    /// input_key.encrypt_lwe(
    ///     &mut ciphertext,
    ///     &plaintext,
    ///     noise,
    ///     &mut encryption_generator,
    /// );
    ///
    /// pksk.keyswitch_ciphertext(&mut switched_ciphertext, &ciphertext);
    ///
    /// let mut decrypted = PlaintextList::from_container(vec![0 as u64; 256]);
    /// output_key.decrypt_glwe(&mut decrypted, &switched_ciphertext);
    /// ```
    pub fn keyswitch_ciphertext<InCont, OutCont, Scalar>(
        &self,
        after: &mut GlweCiphertext<OutCont>,
        before: &LweCiphertext<InCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        GlweCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
        LweCiphertext<InCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(self.input_lwe_key_dimension().0 => before.lwe_size().to_lwe_dimension().0);
        ck_dim_eq!(self.output_glwe_key_dimension().0 => after.size().to_glwe_dimension().0);

        // We reset the output
        after.as_mut_tensor().fill_with(|| Scalar::ZERO);

        // We copy the body
        *after.get_mut_body().tensor.as_mut_tensor().first_mut() = before.get_body().0;

        // We allocate a buffer to hold the decomposition.
        let mut decomp = Tensor::allocate(Scalar::ZERO, self.decomp_level_count.0);

        // We instantiate a decomposer
        let decomposer = SignedDecomposer::new(self.decomp_base_log, self.decomp_level_count);

        for (block, input_lwe_mask) in self
            .bit_decomp_iter()
            .zip(before.get_mask().mask_element_iter())
        {
            // We decompose
            let mask_rounded = decomposer.closest_representable(*input_lwe_mask);

            torus_small_sign_decompose(decomp.as_mut_slice(), mask_rounded, self.decomp_base_log.0);

            // Loop over the number of levels:
            // We compute the multiplication of a ciphertext from the keyswitching key with a
            // piece of the decomposition and subtract it to the buffer
            for (level_key_cipher, decomposed) in block
                .as_tensor()
                .subtensor_iter(self.output_glwe_size.0 * self.output_polynomial_size.0)
                .zip(decomp.iter())
            {
                after
                    .as_mut_tensor()
                    .update_with_wrapping_sub_element_mul(&level_key_cipher, *decomposed);
            }
        }
    }

    /// Packs several LweCiphertext into a single GlweCiphertext
    /// with a keyswitch technique
    pub fn packing_keyswitch<InCont, OutCont, Scalar>(
        &self,
        output: &mut GlweCiphertext<OutCont>,
        input: &LweList<InCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        LweList<InCont>: AsRefTensor<Element = Scalar>,
        GlweCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
        OutCont: Clone,
        Scalar: UnsignedTorus,
    {
        debug_assert!(input.count().0 <= output.polynomial_size().0);
        output.as_mut_tensor().fill_with_element(Scalar::ZERO);
        let mut buffer = output.clone();
        // for each ciphertext, call mono_key_switch
        for (degree, input_cipher) in input.ciphertext_iter().enumerate() {
            self.keyswitch_ciphertext(&mut buffer, &input_cipher);
            buffer
                .as_mut_polynomial_list()
                .polynomial_iter_mut()
                .for_each(|mut poly| {
                    poly.update_with_wrapping_monic_monomial_mul(MonomialDegree(degree))
                });
            output
                .as_mut_tensor()
                .update_with_wrapping_add(buffer.as_tensor());
        }
    }
}

/// The encryption of a single bit of the output key.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub(crate) struct LweKeyBitDecomposition<Cont> {
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) glwe_size: GlweSize,
    pub(crate) poly_size: PolynomialSize,
}

tensor_traits!(LweKeyBitDecomposition);

impl<Cont> LweKeyBitDecomposition<Cont> {
    /// Creates a key bit decomposition from a container.
    ///
    /// # Notes
    ///
    /// This method does not decompose a key bit in a basis, but merely wraps a container in the
    /// right structure. See [`PackingKeyswitchKey::bit_decomp_iter`] for an iterator that returns
    /// key bit decompositions.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 1500], GlweSize(10),
    /// PolynomialSize(10);
    /// assert_eq!(kbd.count(), CiphertextCount(15));
    /// assert_eq!(kbd.glwe_size(), GlweSize(10));
    /// assert_eq!(kbd.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn from_container(cont: Cont, glwe_size: GlweSize, poly_size: PolynomialSize) -> Self
    where
        Tensor<Cont>: AsRefSlice,
    {
        LweKeyBitDecomposition {
            tensor: Tensor::from_container(cont),
            glwe_size,
            poly_size,
        }
    }

    /// Returns the size of the GLWE ciphertexts encoding each level of the key bit decomposition.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// assert_eq!(kbd.lwe_size(), LweSize(10));
    /// ```
    #[allow(dead_code)]
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    /// Returns the size of the lwe ciphertexts encoding each level of the key bit decomposition.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// assert_eq!(kbd.lwe_size(), LweSize(10));
    /// ```
    #[allow(dead_code)]
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns the number of ciphertexts in the decomposition.
    ///
    /// Note that this is actually equals to the number of levels in the decomposition.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// assert_eq!(kbd.count(), CiphertextCount(15));
    /// ```
    #[allow(dead_code)]
    pub fn count(&self) -> CiphertextCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.glwe_size.0 * self.poly_size.0);
        CiphertextCount(self.as_tensor().len() / (self.glwe_size.0 * self.poly_size.0))
    }

    /// Returns an iterator over borrowed `GlweCiphertext`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// for ciphertext in kbd.ciphertext_iter(){
    ///     assert_eq!(ciphertext.lwe_size(), LweSize(10));
    /// }
    /// assert_eq!(kbd.ciphertext_iter().count(), 15);
    /// ```
    #[allow(dead_code)]
    pub fn ciphertext_iter(
        &self,
    ) -> impl Iterator<Item = GlweCiphertext<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        self.as_tensor()
            .subtensor_iter(self.glwe_size.0 * self.poly_size.0)
            .map(move |sub| GlweCiphertext::from_container(sub.into_container(), self.poly_size))
    }

    /// Returns an iterator over mutably borrowed `GlweCiphertext`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
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
    ) -> impl Iterator<Item = GlweCiphertext<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.glwe_size.0 * self.poly_size.0;
        let poly_size = self.poly_size;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(move |sub| GlweCiphertext::from_container(sub.into_container(), poly_size))
    }

    /// Consumes the current key bit decomposition and returns a GLWE.
    ///
    /// Note that this operation is super cheap, as it merely rewraps the current container in
    /// a GLWE structure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// let glwe = kbd.into_glwe_list();
    /// assert_eq!(list.count(), CiphertextCount(15));
    /// assert_eq!(list.lwe_size(), LweSize(10));
    /// ```
    pub fn into_glwe_list(self) -> GlweList<Cont> {
        GlweList {
            tensor: self.tensor,
            rlwe_size: self.glwe_size,
            poly_size: self.poly_size,
        }
    }
}
