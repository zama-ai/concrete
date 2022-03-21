use crate::backends::core::private::crypto::encoding::{Plaintext, PlaintextList};
use crate::backends::core::private::crypto::ggsw::GgswCiphertext;
use crate::backends::core::private::crypto::glwe::{GlweCiphertext, GlweList};
use crate::backends::core::private::crypto::secret::LweSecretKey;
use crate::backends::core::private::math::tensor::{
    ck_dim_div, ck_dim_eq, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};

use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::math::polynomial::PolynomialList;
use crate::backends::core::private::math::random::{Gaussian, RandomGenerable};
use crate::backends::core::private::math::torus::UnsignedTorus;
use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::{
    BinaryKeyKind, GaussianKeyKind, KeyKind, TernaryKeyKind, UniformKeyKind,
};
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{GlweDimension, PlaintextCount, PolynomialSize};
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::Add;

/// A GLWE secret key
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweSecretKey<Kind, Container>
where
    Kind: KeyKind,
{
    tensor: Tensor<Container>,
    poly_size: PolynomialSize,
    kind: PhantomData<Kind>,
}

impl<Scalar> GlweSecretKey<BinaryKeyKind, Vec<Scalar>>
where
    Scalar: UnsignedTorus,
{
    /// Allocates a container for a new key, and fills it with random binary values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::key_kinds::BinaryKeyKind;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let mut generator = SecretRandomGenerator::new(None);
    /// let secret_key: GlweSecretKey<BinaryKeyKind, Vec<u32>> =
    ///     GlweSecretKey::generate_binary(GlweDimension(256), PolynomialSize(10), &mut generator);
    /// assert_eq!(secret_key.key_size(), GlweDimension(256));
    /// assert_eq!(secret_key.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn generate_binary(
        dimension: GlweDimension,
        poly_size: PolynomialSize,
        generator: &mut SecretRandomGenerator,
    ) -> Self {
        GlweSecretKey {
            tensor: generator.random_binary_tensor(poly_size.0 * dimension.0),
            poly_size,
            kind: PhantomData,
        }
    }
}

impl<Scalar> GlweSecretKey<TernaryKeyKind, Vec<Scalar>>
where
    Scalar: UnsignedTorus,
{
    /// Allocates a container for a new key, and fill it with random ternary values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::key_kinds::TernaryKeyKind;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key: GlweSecretKey<_, Vec<u32>> = GlweSecretKey::generate_ternary(
    ///     GlweDimension(256),
    ///     PolynomialSize(10),
    ///     &mut secret_generator,
    /// );
    /// assert_eq!(secret_key.key_size(), GlweDimension(256));
    /// assert_eq!(secret_key.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn generate_ternary(
        dimension: GlweDimension,
        poly_size: PolynomialSize,
        generator: &mut SecretRandomGenerator,
    ) -> Self {
        GlweSecretKey {
            tensor: generator.random_ternary_tensor(poly_size.0 * dimension.0),
            poly_size,
            kind: PhantomData,
        }
    }
}

impl<Scalar> GlweSecretKey<GaussianKeyKind, Vec<Scalar>>
where
    (Scalar, Scalar): RandomGenerable<Gaussian<f64>>,
    Scalar: UnsignedTorus,
{
    /// Allocates a container for a new key, and fill it with random gaussian values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::key_kinds::GaussianKeyKind;
    /// use concrete_commons::parameters::{GlweDimension, LweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key: GlweSecretKey<GaussianKeyKind, Vec<u32>> = GlweSecretKey::generate_gaussian(
    ///     GlweDimension(256),
    ///     PolynomialSize(10),
    ///     &mut secret_generator,
    /// );
    /// assert_eq!(secret_key.key_size(), GlweDimension(256));
    /// assert_eq!(secret_key.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn generate_gaussian(
        dimension: GlweDimension,
        poly_size: PolynomialSize,
        generator: &mut SecretRandomGenerator,
    ) -> Self {
        GlweSecretKey {
            tensor: generator.random_gaussian_tensor(poly_size.0 * dimension.0),
            poly_size,
            kind: PhantomData,
        }
    }
}

impl<Scalar> GlweSecretKey<UniformKeyKind, Vec<Scalar>>
where
    Scalar: UnsignedTorus,
{
    /// Allocates a container for a new key, and fill it with random uniform values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::key_kinds::UniformKeyKind;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key: GlweSecretKey<UniformKeyKind, Vec<u32>> = GlweSecretKey::generate_uniform(
    ///     GlweDimension(256),
    ///     PolynomialSize(10),
    ///     &mut secret_generator,
    /// );
    /// assert_eq!(secret_key.key_size(), GlweDimension(256));
    /// assert_eq!(secret_key.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn generate_uniform(
        dimension: GlweDimension,
        poly_size: PolynomialSize,
        generator: &mut SecretRandomGenerator,
    ) -> Self {
        GlweSecretKey {
            tensor: generator.random_uniform_tensor(poly_size.0 * dimension.0),
            poly_size,
            kind: PhantomData,
        }
    }
}

impl<Cont> GlweSecretKey<BinaryKeyKind, Cont> {
    /// Creates a binary key from a container.
    ///
    /// # Notes
    ///
    /// This method does not fill the container with random data. It merely wraps the container in
    /// the appropriate type. For a method that generate a new random key see
    /// [`GlweSecretKey::generate_binary`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let secret_key =
    ///     GlweSecretKey::binary_from_container(vec![0 as u8; 11 * 256], PolynomialSize(11));
    /// assert_eq!(secret_key.key_size(), GlweDimension(256));
    /// assert_eq!(secret_key.polynomial_size(), PolynomialSize(11));
    /// ```
    pub fn binary_from_container(cont: Cont, poly_size: PolynomialSize) -> Self
    where
        Cont: AsRefSlice,
    {
        ck_dim_div!(cont.as_slice().len() => poly_size.0);
        GlweSecretKey {
            tensor: Tensor::from_container(cont),
            poly_size,
            kind: PhantomData,
        }
    }
}

impl<Cont> GlweSecretKey<TernaryKeyKind, Cont> {
    /// Creates a ternary key from a container.
    ///
    /// # Notes
    ///
    /// This method does not fill the container with random data. It merely wraps the container in
    /// the appropriate type. For a method that generate a new random key see
    /// [`GlweSecretKey::generate_ternary`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let secret_key =
    ///     GlweSecretKey::ternary_from_container(vec![0 as u8; 11 * 256], PolynomialSize(11));
    /// assert_eq!(secret_key.key_size(), GlweDimension(256));
    /// assert_eq!(secret_key.polynomial_size(), PolynomialSize(11));
    /// ```
    pub fn ternary_from_container(cont: Cont, poly_size: PolynomialSize) -> Self
    where
        Cont: AsRefSlice,
    {
        ck_dim_div!(cont.as_slice().len() => poly_size.0);
        GlweSecretKey {
            tensor: Tensor::from_container(cont),
            poly_size,
            kind: PhantomData,
        }
    }
}

impl<Cont> GlweSecretKey<GaussianKeyKind, Cont> {
    /// Creates a gaussian key from a container.
    ///
    /// # Notes
    ///
    /// This method does not fill the container with random data. It merely wraps the container in
    /// the appropriate type. For a method that generate a new random key see
    /// [`GlweSecretKey::generate_gaussian`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let secret_key =
    ///     GlweSecretKey::binary_from_container(vec![0 as u8; 11 * 256], PolynomialSize(11));
    /// assert_eq!(secret_key.key_size(), GlweDimension(256));
    /// assert_eq!(secret_key.polynomial_size(), PolynomialSize(11));
    /// ```
    pub fn gaussian_from_container(cont: Cont, poly_size: PolynomialSize) -> Self
    where
        Cont: AsRefSlice,
    {
        ck_dim_div!(cont.as_slice().len() => poly_size.0);
        GlweSecretKey {
            tensor: Tensor::from_container(cont),
            poly_size,
            kind: PhantomData,
        }
    }
}

impl<Cont> GlweSecretKey<UniformKeyKind, Cont> {
    /// Creates a uniform key from a container.
    ///
    /// # Notes
    ///
    /// This method does not fill the container with random data. It merely wraps the container in
    /// the appropriate type. For a method that generate a new random key see
    /// [`GlweSecretKey::generate_uniform`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let secret_key =
    ///     GlweSecretKey::binary_from_container(vec![0 as u8; 11 * 256], PolynomialSize(11));
    /// assert_eq!(secret_key.key_size(), GlweDimension(256));
    /// assert_eq!(secret_key.polynomial_size(), PolynomialSize(11));
    /// ```
    pub fn uniform_from_container(cont: Cont, poly_size: PolynomialSize) -> Self
    where
        Cont: AsRefSlice,
    {
        ck_dim_div!(cont.as_slice().len() => poly_size.0);
        GlweSecretKey {
            tensor: Tensor::from_container(cont),
            poly_size,
            kind: PhantomData,
        }
    }
}

impl<Kind, Scalar> GlweSecretKey<Kind, Vec<Scalar>>
where
    Kind: KeyKind,
{
    /// Consumes the current GLWE secret key and turns it into an LWE secret key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, LweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::GlweSecretKey;
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let glwe_secret_key: GlweSecretKey<_, Vec<u32>> =
    ///     GlweSecretKey::generate_binary(GlweDimension(2), PolynomialSize(10), &mut secret_generator);
    /// let lwe_secret_key = glwe_secret_key.into_lwe_secret_key();
    /// assert_eq!(lwe_secret_key.key_size(), LweDimension(20))
    /// ```
    pub fn into_lwe_secret_key(self) -> LweSecretKey<Kind, Vec<Scalar>> {
        LweSecretKey {
            tensor: self.tensor,
            kind: PhantomData,
        }
    }
}

impl<Kind, Cont> GlweSecretKey<Kind, Cont>
where
    Kind: KeyKind,
{
    /// Returns the size of the secret key.
    ///
    /// This is equivalent to the number of masks in the [`GlweCiphertext`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key: GlweSecretKey<_, Vec<u32>> = GlweSecretKey::generate_binary(
    ///     GlweDimension(256),
    ///     PolynomialSize(10),
    ///     &mut secret_generator,
    /// );
    /// assert_eq!(secret_key.key_size(), GlweDimension(256));
    /// ```
    pub fn key_size(&self) -> GlweDimension
    where
        Self: AsRefTensor,
    {
        GlweDimension(self.as_tensor().len() / self.poly_size.0)
    }

    /// Returns the size of the secret key polynomials.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key: GlweSecretKey<_, Vec<u32>> = GlweSecretKey::generate_binary(
    ///     GlweDimension(256),
    ///     PolynomialSize(10),
    ///     &mut secret_generator,
    /// );
    /// assert_eq!(secret_key.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns a borrowed polynomial list from the current key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, PolynomialCount, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key: GlweSecretKey<_, Vec<u32>> = GlweSecretKey::generate_binary(
    ///     GlweDimension(256),
    ///     PolynomialSize(10),
    ///     &mut secret_generator,
    /// );
    /// let poly = secret_key.as_polynomial_list();
    /// assert_eq!(poly.polynomial_count(), PolynomialCount(256));
    /// assert_eq!(poly.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn as_polynomial_list(&self) -> PolynomialList<&[<Self as AsRefTensor>::Element]>
    where
        Self: AsRefTensor,
    {
        PolynomialList::from_container(self.as_tensor().as_slice(), self.poly_size)
    }

    /// Returns a mutably borrowed polynomial list from the current key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut secret_key: GlweSecretKey<_, Vec<u32>> = GlweSecretKey::generate_binary(
    ///     GlweDimension(256),
    ///     PolynomialSize(10),
    ///     &mut secret_generator,
    /// );
    /// let mut poly = secret_key.as_mut_polynomial_list();
    /// poly.as_mut_tensor().fill_with_element(1);
    /// assert!(secret_key.as_tensor().iter().all(|a| *a == 1));
    /// ```
    pub fn as_mut_polynomial_list(
        &mut self,
    ) -> PolynomialList<&mut [<Self as AsRefTensor>::Element]>
    where
        Self: AsMutTensor,
    {
        let poly_size = self.poly_size;
        PolynomialList::from_container(self.as_mut_tensor().as_mut_slice(), poly_size)
    }

    /// Encrypts a single GLWE ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::encoding::PlaintextList;
    /// use concrete_core::backends::core::private::crypto::glwe::GlweCiphertext;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key = GlweSecretKey::generate_binary(
    ///     GlweDimension(256),
    ///     PolynomialSize(5),
    ///     &mut secret_generator,
    /// );
    /// let noise = LogStandardDev::from_log_standard_dev(-25.);
    /// let plaintexts =
    ///     PlaintextList::from_container(vec![100000 as u32, 200000, 300000, 400000, 500000]);
    /// let mut ciphertext = GlweCiphertext::allocate(0 as u32, PolynomialSize(5), GlweSize(257));
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.encrypt_glwe(
    ///     &mut ciphertext,
    ///     &plaintexts,
    ///     noise,
    ///     &mut encryption_generator,
    /// );
    /// let mut decrypted = PlaintextList::from_container(vec![0 as u32, 0, 0, 0, 0]);
    /// secret_key.decrypt_glwe(&mut decrypted, &ciphertext);
    /// for (dec, plain) in decrypted.plaintext_iter().zip(plaintexts.plaintext_iter()) {
    ///     let d0 = dec.0.wrapping_sub(plain.0);
    ///     let d1 = plain.0.wrapping_sub(dec.0);
    ///     let dist = std::cmp::min(d0, d1);
    ///     assert!(dist < 400, "dist: {:?}", dist);
    /// }
    /// ```
    pub fn encrypt_glwe<Cont1, Cont2, Scalar>(
        &self,
        encrypted: &mut GlweCiphertext<Cont1>,
        encoded: &PlaintextList<Cont2>,
        noise_parameter: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        GlweCiphertext<Cont1>: AsMutTensor<Element = Scalar>,
        PlaintextList<Cont2>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(encoded.count().0 => encrypted.polynomial_size().0);
        ck_dim_eq!(encrypted.mask_size().0 => self.key_size().0);
        let (mut body, mut masks) = encrypted.get_mut_body_and_mask();
        generator.fill_tensor_with_random_noise(&mut body, noise_parameter);
        generator.fill_tensor_with_random_mask(&mut masks);
        body.as_mut_polynomial().update_with_wrapping_add_multisum(
            &masks.as_mut_polynomial_list(),
            &self.as_polynomial_list(),
        );
        body.as_mut_polynomial()
            .update_with_wrapping_add(&encoded.as_polynomial());
    }

    /// Encrypts a zero plaintext into a GLWE ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::encoding::PlaintextList;
    /// use concrete_core::backends::core::private::crypto::glwe::GlweCiphertext;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key = GlweSecretKey::generate_binary(
    ///     GlweDimension(256),
    ///     PolynomialSize(5),
    ///     &mut secret_generator,
    /// );
    /// let noise = LogStandardDev::from_log_standard_dev(-25.);
    /// let mut ciphertext = GlweCiphertext::allocate(0 as u32, PolynomialSize(5), GlweSize(257));
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.encrypt_zero_glwe(&mut ciphertext, noise, &mut encryption_generator);
    /// let mut decrypted = PlaintextList::from_container(vec![0 as u32, 0, 0, 0, 0]);
    /// secret_key.decrypt_glwe(&mut decrypted, &ciphertext);
    /// for dec in decrypted.plaintext_iter() {
    ///     let d0 = dec.0.wrapping_sub(0u32);
    ///     let d1 = 0u32.wrapping_sub(dec.0);
    ///     let dist = std::cmp::min(d0, d1);
    ///     assert!(dist < 500, "dist: {:?}", dist);
    /// }
    /// ```
    pub fn encrypt_zero_glwe<Scalar, Cont1>(
        &self,
        encrypted: &mut GlweCiphertext<Cont1>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        GlweCiphertext<Cont1>: AsMutTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(encrypted.mask_size().0 => self.key_size().0);
        let (mut body, mut masks) = encrypted.get_mut_body_and_mask();
        generator.fill_tensor_with_random_noise(&mut body, noise_parameters);
        generator.fill_tensor_with_random_mask(&mut masks);
        body.as_mut_polynomial().update_with_wrapping_add_multisum(
            &masks.as_mut_polynomial_list(),
            &self.as_polynomial_list(),
        );
    }

    /// Encrypts a list of GLWE ciphertexts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::encoding::PlaintextList;
    /// use concrete_core::backends::core::private::crypto::glwe::{GlweCiphertext, GlweList};
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key = GlweSecretKey::generate_binary(
    ///     GlweDimension(256),
    ///     PolynomialSize(2),
    ///     &mut secret_generator,
    /// );
    /// let noise = LogStandardDev::from_log_standard_dev(-25.);
    /// let plaintexts = PlaintextList::from_container(vec![1000 as u32, 2000, 3000, 4000]);
    /// let mut ciphertexts = GlweList::allocate(
    ///     0 as u32,
    ///     PolynomialSize(2),
    ///     GlweDimension(256),
    ///     CiphertextCount(2),
    /// );
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.encrypt_glwe_list(
    ///     &mut ciphertexts,
    ///     &plaintexts,
    ///     noise,
    ///     &mut encryption_generator,
    /// );
    /// let mut decrypted = PlaintextList::from_container(vec![0 as u32, 0, 0, 0]);
    /// secret_key.decrypt_glwe_list(&mut decrypted, &ciphertexts);
    /// for (dec, plain) in decrypted.plaintext_iter().zip(plaintexts.plaintext_iter()) {
    ///     let d0 = dec.0.wrapping_sub(plain.0);
    ///     let d1 = plain.0.wrapping_sub(dec.0);
    ///     let dist = std::cmp::min(d0, d1);
    ///     assert!(dist < 400, "dist: {:?}", dist);
    /// }
    /// ```
    pub fn encrypt_glwe_list<CiphCont, EncCont, Scalar>(
        &self,
        encrypt: &mut GlweList<CiphCont>,
        encoded: &PlaintextList<EncCont>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        GlweList<CiphCont>: AsMutTensor<Element = Scalar>,
        PlaintextList<EncCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
        for<'a> PlaintextList<&'a [Scalar]>: AsRefTensor<Element = Scalar>,
    {
        ck_dim_eq!(encrypt.ciphertext_count().0 * encrypt.polynomial_size().0 => encoded.count().0);
        ck_dim_eq!(encrypt.glwe_dimension().0 => self.key_size().0);

        let count = PlaintextCount(encrypt.polynomial_size().0);
        for (mut ciphertext, encoded) in encrypt
            .ciphertext_iter_mut()
            .zip(encoded.sublist_iter(count))
        {
            self.encrypt_glwe(&mut ciphertext, &encoded, noise_parameters, generator);
        }
    }

    /// Encrypts a list of GLWE ciphertexts, with a zero plaintext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::encoding::PlaintextList;
    /// use concrete_core::backends::core::private::crypto::glwe::{GlweCiphertext, GlweList};
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::*;
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key = GlweSecretKey::generate_binary(
    ///     GlweDimension(256),
    ///     PolynomialSize(2),
    ///     &mut secret_generator,
    /// );
    /// let noise = LogStandardDev::from_log_standard_dev(-25.);
    /// let mut ciphertexts = GlweList::allocate(
    ///     0 as u32,
    ///     PolynomialSize(2),
    ///     GlweDimension(256),
    ///     CiphertextCount(2),
    /// );
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.encrypt_zero_glwe_list(&mut ciphertexts, noise, &mut encryption_generator);
    /// let mut decrypted = PlaintextList::from_container(vec![0 as u32, 0, 0, 0]);
    /// secret_key.decrypt_glwe_list(&mut decrypted, &ciphertexts);
    /// for dec in decrypted.plaintext_iter() {
    ///     let d0 = dec.0.wrapping_sub(0u32);
    ///     let d1 = 0u32.wrapping_sub(dec.0);
    ///     let dist = std::cmp::min(d0, d1);
    ///     assert!(dist < 400, "dist: {:?}", dist);
    /// }
    /// ```
    pub fn encrypt_zero_glwe_list<Scalar, OutputCont>(
        &self,
        encrypted: &mut GlweList<OutputCont>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        GlweList<OutputCont>: AsMutTensor<Element = Scalar>,
        Scalar: UnsignedTorus + Add,
    {
        for mut ciphertext in encrypted.ciphertext_iter_mut() {
            self.encrypt_zero_glwe(&mut ciphertext, noise_parameters, generator);
        }
    }

    /// Decrypts a single GLWE ciphertext.
    ///
    /// See ['GlweSecretKey::encrypt_glwe`] for an example.
    pub fn decrypt_glwe<CiphCont, EncCont, Scalar>(
        &self,
        encoded: &mut PlaintextList<EncCont>,
        encrypted: &GlweCiphertext<CiphCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        PlaintextList<EncCont>: AsMutTensor<Element = Scalar>,
        GlweCiphertext<CiphCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus + Add,
    {
        ck_dim_eq!(encoded.count().0 => encrypted.polynomial_size().0);
        let (body, masks) = encrypted.get_body_and_mask();
        encoded
            .as_mut_tensor()
            .fill_with_one(body.as_tensor(), |a| *a);
        encoded
            .as_mut_polynomial()
            .update_with_wrapping_sub_multisum(
                &masks.as_polynomial_list(),
                &self.as_polynomial_list(),
            );
    }

    /// Decrypts a list of GLWE ciphertexts.
    ///
    /// See ['GlweSecretKey::encrypt_glwe_list`] for an example.
    pub fn decrypt_glwe_list<CiphCont, EncCont, Scalar>(
        &self,
        encoded: &mut PlaintextList<EncCont>,
        encrypted: &GlweList<CiphCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        PlaintextList<EncCont>: AsMutTensor<Element = Scalar>,
        GlweList<CiphCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus + Add,
        for<'a> PlaintextList<&'a mut [Scalar]>: AsMutTensor<Element = Scalar>,
    {
        ck_dim_eq!(encrypted.ciphertext_count().0 * encrypted.polynomial_size().0 => encoded.count().0);
        ck_dim_eq!(encrypted.glwe_dimension().0 => self.key_size().0);
        for (ciphertext, mut encoded) in encrypted
            .ciphertext_iter()
            .zip(encoded.sublist_iter_mut(PlaintextCount(encrypted.polynomial_size().0)))
        {
            self.decrypt_glwe(&mut encoded, &ciphertext);
        }
    }

    /// This function encrypts a message as a GGSW ciphertext.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::encoding::Plaintext;
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::GlweSecretKey;
    /// let mut generator = SecretRandomGenerator::new(None);
    /// let secret_key =
    ///     GlweSecretKey::generate_binary(GlweDimension(2), PolynomialSize(10), &mut generator);
    /// let mut ciphertext = GgswCiphertext::allocate(
    ///     0 as u32,
    ///     PolynomialSize(10),
    ///     GlweSize(3),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(7),
    /// );
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    /// let mut secret_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.encrypt_constant_ggsw(
    ///     &mut ciphertext,
    ///     &Plaintext(10),
    ///     noise,
    ///     &mut secret_generator,
    /// );
    /// ```
    pub fn encrypt_constant_ggsw<OutputCont, Scalar>(
        &self,
        encrypted: &mut GgswCiphertext<OutputCont>,
        encoded: &Plaintext<Scalar>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        GgswCiphertext<OutputCont>: AsMutTensor<Element = Scalar>,
        OutputCont: AsMutSlice<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(self.polynomial_size() => encrypted.polynomial_size());
        ck_dim_eq!(self.key_size() => encrypted.glwe_size().to_glwe_dimension());
        let gen_iter = generator
            .fork_ggsw_to_ggsw_levels::<Scalar>(
                encrypted.decomposition_level_count(),
                self.key_size().to_glwe_size(),
                self.poly_size,
            )
            .expect("Failed to split generator into ggsw levels");
        let base_log = encrypted.decomposition_base_log();
        for (mut matrix, mut generator) in encrypted.level_matrix_iter_mut().zip(gen_iter) {
            let decomposition = encoded.0
                * (Scalar::ONE
                    << (<Scalar as Numeric>::BITS
                        - (base_log.0 * (matrix.decomposition_level().0))));
            let gen_iter = generator
                .fork_ggsw_level_to_glwe::<Scalar>(self.key_size().to_glwe_size(), self.poly_size)
                .expect("Failed to split generator into rlwe");
            // We iterate over the rowe of the level matrix
            for ((index, row), mut generator) in matrix.row_iter_mut().enumerate().zip(gen_iter) {
                let mut rlwe_ct = row.into_glwe();
                // We issue a fresh  encryption of zero
                self.encrypt_zero_glwe(&mut rlwe_ct, noise_parameters, &mut generator);
                // We retrieve the row as a polynomial list
                let mut polynomial_list = rlwe_ct.into_polynomial_list();
                // We retrieve the polynomial in the diagonal
                let mut level_polynomial = polynomial_list.get_mut_polynomial(index);
                // We get the first coefficient
                let first_coef = level_polynomial.as_mut_tensor().first_mut();
                // We update the first coefficient
                *first_coef = first_coef.wrapping_add(decomposition);
            }
        }
    }

    /// This function encrypts a message as a GGSW ciphertext, using as many threads as possible.
    ///
    /// # Notes
    /// This method is hidden behind the "multithread" feature gate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::encoding::Plaintext;
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::GlweSecretKey;
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key =
    ///     GlweSecretKey::generate_binary(GlweDimension(2), PolynomialSize(10), &mut secret_generator);
    /// let mut ciphertext = GgswCiphertext::allocate(
    ///     0 as u32,
    ///     PolynomialSize(10),
    ///     GlweSize(3),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(7),
    /// );
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.par_encrypt_constant_ggsw(
    ///     &mut ciphertext,
    ///     &Plaintext(10),
    ///     noise,
    ///     &mut encryption_generator,
    /// );
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_encrypt_constant_ggsw<OutputCont, Scalar>(
        &self,
        encrypted: &mut GgswCiphertext<OutputCont>,
        encoded: &Plaintext<Scalar>,
        noise_parameters: impl DispersionParameter + Send + Sync,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        GgswCiphertext<OutputCont>: AsMutTensor<Element = Scalar>,
        OutputCont: AsMutSlice<Element = Scalar>,
        Scalar: UnsignedTorus + Send + Sync,
        Cont: Sync,
    {
        ck_dim_eq!(self.polynomial_size() => encrypted.polynomial_size());
        ck_dim_eq!(self.key_size() => encrypted.glwe_size().to_glwe_dimension());
        let generators = generator
            .par_fork_ggsw_to_ggsw_levels::<Scalar>(
                encrypted.decomposition_level_count(),
                self.key_size().to_glwe_size(),
                self.poly_size,
            )
            .expect("Failed to split generator into ggsw levels");
        let base_log = encrypted.decomposition_base_log();
        encrypted
            .par_level_matrix_iter_mut()
            .zip(generators)
            .for_each(move |(mut matrix, mut generator)| {
                let decomposition = encoded.0
                    * (Scalar::ONE
                        << (<Scalar as Numeric>::BITS
                            - (base_log.0 * (matrix.decomposition_level().0))));
                let gen_iter = generator
                    .par_fork_ggsw_level_to_glwe::<Scalar>(
                        self.key_size().to_glwe_size(),
                        self.poly_size,
                    )
                    .expect("Failed to split generator into glwe");
                // We iterate over the rowe of the level matrix
                matrix
                    .par_row_iter_mut()
                    .enumerate()
                    .zip(gen_iter)
                    .for_each(|((index, row), mut generator)| {
                        let mut rlwe_ct = row.into_glwe();
                        // We issue a fresh  encryption of zero
                        self.encrypt_zero_glwe(&mut rlwe_ct, noise_parameters, &mut generator);
                        // We retrieve the row as a polynomial list
                        let mut polynomial_list = rlwe_ct.into_polynomial_list();
                        // We retrieve the polynomial in the diagonal
                        let mut level_polynomial = polynomial_list.get_mut_polynomial(index);
                        // We get the first coefficient
                        let first_coef = level_polynomial.as_mut_tensor().first_mut();
                        // We update the first coefficient
                        *first_coef = first_coef.wrapping_add(decomposition);
                    })
            })
    }

    /// This function encrypts a message as a GGSW ciphertext whose rlwe masks are all zeros.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::encoding::Plaintext;
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::GlweSecretKey;
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let secret_key: GlweSecretKey<_, Vec<u32>> =
    ///     GlweSecretKey::generate_binary(GlweDimension(2), PolynomialSize(10), &mut secret_generator);
    /// let mut ciphertext = GgswCiphertext::allocate(
    ///     0 as u32,
    ///     PolynomialSize(10),
    ///     GlweSize(3),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(7),
    /// );
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// secret_key.trivial_encrypt_constant_ggsw(
    ///     &mut ciphertext,
    ///     &Plaintext(10),
    ///     noise,
    ///     &mut encryption_generator,
    /// );
    /// ```
    pub fn trivial_encrypt_constant_ggsw<OutputCont, Scalar>(
        &self,
        encrypted: &mut GgswCiphertext<OutputCont>,
        encoded: &Plaintext<Scalar>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        GgswCiphertext<OutputCont>: AsMutTensor<Element = Scalar>,
        OutputCont: AsMutSlice<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(self.polynomial_size() => encrypted.polynomial_size());
        ck_dim_eq!(self.key_size() => encrypted.glwe_size().to_glwe_dimension());
        // We fill the ggsw with trivial glwe encryptions of zero:
        for mut glwe in encrypted.as_mut_glwe_list().ciphertext_iter_mut() {
            let (mut body, mut mask) = glwe.get_mut_body_and_mask();
            mask.as_mut_tensor().fill_with_element(Scalar::ZERO);
            generator.fill_tensor_with_random_noise(&mut body, noise_parameters);
        }
        let base_log = encrypted.decomposition_base_log();
        for mut matrix in encrypted.level_matrix_iter_mut() {
            let decomposition = encoded.0
                * (Scalar::ONE
                    << (<Scalar as Numeric>::BITS
                        - (base_log.0 * (matrix.decomposition_level().0))));
            // We iterate over the rowe of the level matrix
            for (index, row) in matrix.row_iter_mut().enumerate() {
                let rlwe_ct = row.into_glwe();
                // We retrieve the row as a polynomial list
                let mut polynomial_list = rlwe_ct.into_polynomial_list();
                // We retrieve the polynomial in the diagonal
                let mut level_polynomial = polynomial_list.get_mut_polynomial(index);
                // We get the first coefficient
                let first_coef = level_polynomial.as_mut_tensor().first_mut();
                // We update the first coefficient
                *first_coef = first_coef.wrapping_add(decomposition);
            }
        }
    }
}

impl<Kind, Element, Cont> AsRefTensor for GlweSecretKey<Kind, Cont>
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

impl<Kind, Element, Cont> AsMutTensor for GlweSecretKey<Kind, Cont>
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

impl<Kind, Cont> IntoTensor for GlweSecretKey<Kind, Cont>
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
