use serde::{Deserialize, Serialize};

use crate::crypto::encoding::{Plaintext, PlaintextList};
use crate::crypto::lwe::{LweCiphertext, LweList};
use crate::crypto::secret::generators::{EncryptionRandomGenerator, SecretRandomGenerator};
use crate::math::random::{Gaussian, RandomGenerable};
use crate::math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor};
use crate::math::torus::UnsignedTorus;
use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::{
    BinaryKeyKind, GaussianKeyKind, KeyKind, TernaryKeyKind, UniformKeyKind,
};
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::LweDimension;
use std::marker::PhantomData;

/// A LWE secret key.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LweSecretKey<Kind, Cont>
where
    Kind: KeyKind,
{
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) kind: PhantomData<Kind>,
}

impl<Scalar> LweSecretKey<BinaryKeyKind, Vec<Scalar>>
where
    Scalar: UnsignedTorus,
{
    /// Generates a new binary secret key; e.g. allocates a storage and samples random values for
    /// the key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    /// let mut generator = SecretRandomGenerator::new(None);
    /// let secret_key: LweSecretKey<_, Vec<u32>> =
    ///     LweSecretKey::generate_binary(LweDimension(256), &mut generator);
    /// assert_eq!(secret_key.key_size(), LweDimension(256));
    /// ```
    pub fn generate_binary(size: LweDimension, generator: &mut SecretRandomGenerator) -> Self {
        LweSecretKey {
            tensor: generator.random_binary_tensor(size.0),
            kind: PhantomData,
        }
    }
}

impl<Scalar> LweSecretKey<TernaryKeyKind, Vec<Scalar>>
where
    Scalar: UnsignedTorus,
{
    /// Generates a new ternary secret key; e.g. allocates a storage and samples random values for
    /// the key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    /// let mut generator = SecretRandomGenerator::new(None);
    /// let secret_key: LweSecretKey<_, Vec<u32>> =
    ///     LweSecretKey::generate_ternary(LweDimension(256), &mut generator);
    /// assert_eq!(secret_key.key_size(), LweDimension(256));
    /// ```
    pub fn generate_ternary(size: LweDimension, generator: &mut SecretRandomGenerator) -> Self {
        LweSecretKey {
            tensor: generator.random_ternary_tensor(size.0),
            kind: PhantomData,
        }
    }
}

impl<Scalar> LweSecretKey<GaussianKeyKind, Vec<Scalar>>
where
    (Scalar, Scalar): RandomGenerable<Gaussian<f64>>,
    Scalar: UnsignedTorus,
{
    /// Generates a new gaussian secret key; e.g. allocates a storage and samples random values for
    /// the key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    /// let mut generator = SecretRandomGenerator::new(None);
    /// let secret_key: LweSecretKey<_, Vec<u32>> =
    ///     LweSecretKey::generate_gaussian(LweDimension(256), &mut generator);
    /// assert_eq!(secret_key.key_size(), LweDimension(256));
    /// ```
    pub fn generate_gaussian(size: LweDimension, generator: &mut SecretRandomGenerator) -> Self {
        LweSecretKey {
            tensor: generator.random_gaussian_tensor(size.0),
            kind: PhantomData,
        }
    }
}

impl<Scalar> LweSecretKey<UniformKeyKind, Vec<Scalar>>
where
    Scalar: UnsignedTorus,
{
    /// Generates a new gaussian secret key; e.g. allocates a storage and samples random values for
    /// the key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    /// let mut generator = SecretRandomGenerator::new(None);
    /// let secret_key: LweSecretKey<_, Vec<u32>> =
    ///     LweSecretKey::generate_uniform(LweDimension(256), &mut generator);
    /// assert_eq!(secret_key.key_size(), LweDimension(256));
    /// ```
    pub fn generate_uniform(size: LweDimension, generator: &mut SecretRandomGenerator) -> Self {
        LweSecretKey {
            tensor: generator.random_uniform_tensor(size.0),
            kind: PhantomData,
        }
    }
}

impl<Cont> LweSecretKey<BinaryKeyKind, Cont> {
    /// Creates a binary lwe secret key from a container.
    ///
    /// # Notes
    ///
    /// This method does not fill the container with random values to create a new key. It merely
    /// wraps a container into the appropriate type. See [`LweSecretKey::generate`] for a
    /// generation method.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    /// let secret_key = LweSecretKey::binary_from_container(vec![true; 256]);
    /// assert_eq!(secret_key.key_size(), LweDimension(256));
    /// ```
    pub fn binary_from_container(cont: Cont) -> Self
    where
        Cont: AsRefSlice,
    {
        LweSecretKey {
            tensor: Tensor::from_container(cont),
            kind: PhantomData,
        }
    }
}

impl<Cont> LweSecretKey<TernaryKeyKind, Cont> {
    /// Creates a ternary lwe secret key from a container.
    ///
    /// # Notes
    ///
    /// This method does not fill the container with random values to create a
    /// new key. It merely wraps a container into the appropriate type. See
    /// [`LweSecretKey::generate`] for a generation method.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    /// let secret_key = LweSecretKey::ternary_from_container(vec![true; 256]);
    /// assert_eq!(secret_key.key_size(), LweDimension(256));
    /// ```
    pub fn ternary_from_container(cont: Cont) -> Self
    where
        Cont: AsRefSlice,
    {
        LweSecretKey {
            tensor: Tensor::from_container(cont),
            kind: PhantomData,
        }
    }
}

impl<Cont> LweSecretKey<GaussianKeyKind, Cont> {
    /// Creates a gaussian lwe secret key from a container.
    ///
    /// # Notes
    ///
    /// This method does not fill the container with random values to create a new key. It merely
    /// wraps a container into the appropriate type. See [`LweSecretKey::generate`] for a
    /// generation method.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    /// let secret_key = LweSecretKey::gaussian_from_container(vec![true; 256]);
    /// assert_eq!(secret_key.key_size(), LweDimension(256));
    /// ```
    pub fn gaussian_from_container(cont: Cont) -> Self
    where
        Cont: AsRefSlice,
    {
        LweSecretKey {
            tensor: Tensor::from_container(cont),
            kind: PhantomData,
        }
    }
}

impl<Cont> LweSecretKey<UniformKeyKind, Cont> {
    /// Creates a uniform lwe secret key from a container.
    ///
    /// # Notes
    ///
    /// This method does not fill the container with random values to create a new key. It merely
    /// wraps a container into the appropriate type. See [`LweSecretKey::generate`] for a
    /// generation method.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    /// let secret_key = LweSecretKey::uniform_from_container(vec![true; 256]);
    /// assert_eq!(secret_key.key_size(), LweDimension(256));
    /// ```
    pub fn uniform_from_container(cont: Cont) -> Self
    where
        Cont: AsRefSlice,
    {
        LweSecretKey {
            tensor: Tensor::from_container(cont),
            kind: PhantomData,
        }
    }
}

impl<Kind, Cont> LweSecretKey<Kind, Cont>
where
    Kind: KeyKind,
{
    /// Returns the size of the secret key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    /// let secret_key = LweSecretKey::binary_from_container(vec![true; 256]);
    /// assert_eq!(secret_key.key_size(), LweDimension(256));
    /// ```
    pub fn key_size(&self) -> LweDimension
    where
        Self: AsRefTensor,
    {
        LweDimension(self.as_tensor().len())
    }

    /// Encrypts a single ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{LweDimension, LweSize};
    /// use concrete_core::crypto::encoding::*;
    /// use concrete_core::crypto::lwe::*;
    /// use concrete_core::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    ///
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key = LweSecretKey::generate_binary(LweDimension(256), &mut secret_generator);
    /// let encoder = RealEncoder {
    ///     offset: 0. as f32,
    ///     delta: 10.,
    /// };
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    ///
    /// let clear = Cleartext(2. as f32);
    /// let plain: Plaintext<u32> = encoder.encode(clear);
    /// let mut encrypted = LweCiphertext::allocate(0u32, LweSize(257));
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.encrypt_lwe(&mut encrypted, &plain, noise, &mut encryption_generator);
    ///
    /// let mut decrypted = Plaintext(0u32);
    /// secret_key.decrypt_lwe(&mut decrypted, &encrypted);
    /// let decoded = encoder.decode(decrypted);
    ///
    /// assert!((decoded.0 - clear.0).abs() < 0.1);
    /// ```
    pub fn encrypt_lwe<OutputCont, Scalar>(
        &self,
        output: &mut LweCiphertext<OutputCont>,
        encoded: &Plaintext<Scalar>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        LweCiphertext<OutputCont>: AsMutTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        let (output_body, mut output_masks) = output.get_mut_body_and_mask();

        // generate a uniformly random mask
        generator.fill_tensor_with_random_mask(&mut output_masks);

        // generate an error from the normal distribution described by std_dev
        output_body.0 = generator.random_noise(noise_parameters);

        // compute the multisum between the secret key and the mask
        output_body.0 = output_body
            .0
            .wrapping_add(output_masks.compute_multisum(&self));

        // add the encoded message
        output_body.0 = output_body.0.wrapping_add(encoded.0);
    }

    /// Encrypts a list of ciphertexts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, CleartextCount, LweDimension, LweSize, PlaintextCount,
    /// };
    /// use concrete_core::crypto::encoding::*;
    /// use concrete_core::crypto::lwe::*;
    /// use concrete_core::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    ///
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key = LweSecretKey::generate_binary(LweDimension(256), &mut secret_generator);
    /// let encoder = RealEncoder {
    ///     offset: 0. as f32,
    ///     delta: 10.,
    /// };
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    ///
    /// let clear_values = CleartextList::allocate(2. as f32, CleartextCount(100));
    /// let mut plain_values = PlaintextList::allocate(0u32, PlaintextCount(100));
    /// encoder.encode_list(&mut plain_values, &clear_values);
    /// let mut encrypted_values = LweList::allocate(0u32, LweSize(257), CiphertextCount(100));
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.encrypt_lwe_list(
    ///     &mut encrypted_values,
    ///     &plain_values,
    ///     noise,
    ///     &mut encryption_generator,
    /// );
    ///
    /// let mut decrypted_values = PlaintextList::allocate(0u32, PlaintextCount(100));
    /// secret_key.decrypt_lwe_list(&mut decrypted_values, &encrypted_values);
    /// let mut decoded_values = CleartextList::allocate(0. as f32, CleartextCount(100));
    /// encoder.decode_list(&mut decoded_values, &decrypted_values);
    /// for (clear, decoded) in clear_values
    ///     .cleartext_iter()
    ///     .zip(decoded_values.cleartext_iter())
    /// {
    ///     assert!((clear.0 - decoded.0).abs() < 0.1);
    /// }
    /// ```
    pub fn encrypt_lwe_list<OutputCont, InputCont, Scalar>(
        &self,
        output: &mut LweList<OutputCont>,
        encoded: &PlaintextList<InputCont>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        LweList<OutputCont>: AsMutTensor<Element = Scalar>,
        PlaintextList<InputCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        debug_assert!(
            output.count().0 == encoded.count().0,
            "Lwe cipher list size and encoded list size are not compatible"
        );
        for (mut cipher, message) in output.ciphertext_iter_mut().zip(encoded.plaintext_iter()) {
            self.encrypt_lwe(&mut cipher, message, noise_parameters.clone(), generator);
        }
    }

    /// Encrypts a single ciphertext with null masks.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{LweDimension, LweSize};
    /// use concrete_core::crypto::encoding::*;
    /// use concrete_core::crypto::lwe::*;
    /// use concrete_core::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    ///
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key = LweSecretKey::generate_binary(LweDimension(256), &mut secret_generator);
    /// let encoder = RealEncoder {
    ///     offset: 0. as f32,
    ///     delta: 10.,
    /// };
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    ///
    /// let clear = Cleartext(2. as f32);
    /// let plain: Plaintext<u32> = encoder.encode(clear);
    /// let mut encrypted = LweCiphertext::allocate(0u32, LweSize(257));
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.trivial_encrypt_lwe(&mut encrypted, &plain, noise, &mut encryption_generator);
    ///
    /// let mut decrypted = Plaintext(0u32);
    /// secret_key.decrypt_lwe(&mut decrypted, &encrypted);
    /// let decoded = encoder.decode(decrypted);
    ///
    /// assert!((decoded.0 - clear.0).abs() < 0.1);
    /// ```
    pub fn trivial_encrypt_lwe<OutputCont, Scalar>(
        &self,
        output: &mut LweCiphertext<OutputCont>,
        encoded: &Plaintext<Scalar>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        LweCiphertext<OutputCont>: AsMutTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        let (output_body, mut output_masks) = output.get_mut_body_and_mask();

        // generate a uniformly random mask
        output_masks
            .as_mut_tensor()
            .fill_with_element(<Scalar as Numeric>::ZERO);

        // generate an error from the normal distribution described by std_dev
        output_body.0 = generator.random_noise(noise_parameters);

        // compute the multisum between the secret key and the mask
        output_body.0 = output_body
            .0
            .wrapping_add(output_masks.compute_multisum(&self));

        // add the encoded message
        output_body.0 = output_body.0.wrapping_add(encoded.0);
    }

    /// Encrypts a list of ciphertexts with null masks.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, CleartextCount, LweDimension, LweSize, PlaintextCount,
    /// };
    /// use concrete_core::crypto::encoding::*;
    /// use concrete_core::crypto::lwe::*;
    /// use concrete_core::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::crypto::secret::*;
    /// use concrete_core::crypto::*;
    ///
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key = LweSecretKey::generate_binary(LweDimension(256), &mut secret_generator);
    /// let encoder = RealEncoder {
    ///     offset: 0. as f32,
    ///     delta: 10.,
    /// };
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    ///
    /// let clear_values = CleartextList::allocate(2. as f32, CleartextCount(100));
    /// let mut plain_values = PlaintextList::allocate(0u32, PlaintextCount(100));
    /// encoder.encode_list(&mut plain_values, &clear_values);
    /// let mut encrypted_values = LweList::allocate(0u32, LweSize(257), CiphertextCount(100));
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.trivial_encrypt_lwe_list(
    ///     &mut encrypted_values,
    ///     &plain_values,
    ///     noise,
    ///     &mut encryption_generator,
    /// );
    ///
    /// for ciphertext in encrypted_values.ciphertext_iter() {
    ///     for mask in ciphertext.get_mask().mask_element_iter() {
    ///         assert_eq!(*mask, 0);
    ///     }
    /// }
    ///
    /// let mut decrypted_values = PlaintextList::allocate(0u32, PlaintextCount(100));
    /// secret_key.decrypt_lwe_list(&mut decrypted_values, &encrypted_values);
    /// let mut decoded_values = CleartextList::allocate(0. as f32, CleartextCount(100));
    /// encoder.decode_list(&mut decoded_values, &decrypted_values);
    /// for (clear, decoded) in clear_values
    ///     .cleartext_iter()
    ///     .zip(decoded_values.cleartext_iter())
    /// {
    ///     assert!((clear.0 - decoded.0).abs() < 0.1);
    /// }
    /// ```
    pub fn trivial_encrypt_lwe_list<OutputCont, InputCont, Scalar>(
        &self,
        output: &mut LweList<OutputCont>,
        encoded: &PlaintextList<InputCont>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        LweList<OutputCont>: AsMutTensor<Element = Scalar>,
        PlaintextList<InputCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        debug_assert!(
            output.count().0 == encoded.count().0,
            "Lwe cipher list size and encoded list size are not compatible"
        );
        for (mut cipher, message) in output.ciphertext_iter_mut().zip(encoded.plaintext_iter()) {
            self.trivial_encrypt_lwe(&mut cipher, message, noise_parameters.clone(), generator);
        }
    }

    /// Decrypts a single ciphertext.
    ///
    /// See ['encrypt_lwe'] for an example.
    pub fn decrypt_lwe<Scalar, CipherCont>(
        &self,
        output: &mut Plaintext<Scalar>,
        cipher: &LweCiphertext<CipherCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        LweCiphertext<CipherCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        let (body, masks) = cipher.get_body_and_mask();
        // put body inside result
        output.0 = output.0.wrapping_add(body.0);
        // subtract the multisum between the key and the mask
        output.0 = output.0.wrapping_sub(masks.compute_multisum(&self));
    }

    /// Decrypts a list of ciphertexts.
    ///
    /// See ['encrypt_lwe_list'] for an example.
    pub fn decrypt_lwe_list<Scalar, EncodedCont, CipherCont>(
        &self,
        output: &mut PlaintextList<EncodedCont>,
        cipher: &LweList<CipherCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        PlaintextList<EncodedCont>: AsMutTensor<Element = Scalar>,
        LweList<CipherCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        debug_assert!(
            output.count().0 == cipher.count().0,
            "Tried to decrypt a list into one with incompatible size.Expected {} found {}",
            output.count().0,
            cipher.count().0
        );
        for (cipher, mut output) in cipher.ciphertext_iter().zip(output.plaintext_iter_mut()) {
            self.decrypt_lwe(&mut output, &cipher);
        }
    }
}

impl<Kind, Element, Cont> AsRefTensor for LweSecretKey<Kind, Cont>
where
    Kind: KeyKind,
    Cont: AsRefSlice<Element = Element>,
{
    type Element = Element;
    type Container = Cont;
    fn as_tensor(&self) -> &Tensor<Self::Container> {
        &self.tensor
    }
}

impl<Kind, Element, Cont> AsMutTensor for LweSecretKey<Kind, Cont>
where
    Kind: KeyKind,
    Cont: AsMutSlice<Element = Element>,
{
    type Element = Element;
    type Container = Cont;
    fn as_mut_tensor(&mut self) -> &mut Tensor<<Self as AsMutTensor>::Container> {
        &mut self.tensor
    }
}

impl<Kind, Cont> IntoTensor for LweSecretKey<Kind, Cont>
where
    Kind: KeyKind,
    Cont: AsRefSlice,
{
    type Element = <Cont as AsRefSlice>::Element;
    type Container = Cont;
    fn into_tensor(self) -> Tensor<Self::Container> {
        self.tensor
    }
}
