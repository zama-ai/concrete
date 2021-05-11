//! Bootstrapping key
use fftw::array::AlignedVec;

use crate::crypto::encoding::Plaintext;
use crate::crypto::{LweDimension, UnsignedTorus};
use crate::math::decomposition::{DecompositionBaseLog, DecompositionLevelCount};
use crate::math::dispersion::DispersionParameter;
use crate::math::fft::{Complex64, Fft, FourierPolynomial};
use crate::math::polynomial::{Polynomial, PolynomialSize};
use crate::math::tensor::{AsMutTensor, AsRefSlice, AsRefTensor, Tensor};
use crate::numeric::Numeric;
use crate::{ck_dim_div, ck_dim_eq, tensor_traits};

use super::ggsw::GgswCiphertext;
use super::secret::{GlweSecretKey, LweSecretKey};
use super::GlweSize;
use crate::math::random::RandomGenerator;

mod serde;

/// A bootstrapping key
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapKey<Cont> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    rlwe_size: GlweSize,
    decomp_level: DecompositionLevelCount,
    decomp_base_log: DecompositionBaseLog,
}

tensor_traits!(BootstrapKey);

impl<Scalar> BootstrapKey<Vec<Scalar>> {
    /// Allocates a new bootstrapping key whose polynomials coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(9));
    /// assert_eq!(bsk.glwe_size(), GlweSize(7));
    /// assert_eq!(bsk.level_count(), DecompositionLevelCount(3));
    /// assert_eq!(bsk.base_log(), DecompositionBaseLog(5));
    /// assert_eq!(bsk.key_size(), LweDimension(4));
    /// ```
    pub fn allocate(
        value: Scalar,
        rlwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        key_size: LweDimension,
    ) -> BootstrapKey<Vec<Scalar>>
    where
        Scalar: Copy,
    {
        BootstrapKey {
            tensor: Tensor::from_container(vec![
                value;
                key_size.0
                    * decomp_level.0
                    * rlwe_size.0
                    * rlwe_size.0
                    * poly_size.0
            ]),
            decomp_level,
            decomp_base_log,
            rlwe_size,
            poly_size,
        }
    }
}

impl BootstrapKey<AlignedVec<Complex64>> {
    /// Allocates a new complex bootstrapping key whose polynomials coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::math::fft::Complex64;
    /// let bsk = BootstrapKey::allocate(
    ///     Complex64::new(9.,8.),
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(9));
    /// assert_eq!(bsk.glwe_size(), GlweSize(7));
    /// assert_eq!(bsk.level_count(), DecompositionLevelCount(3));
    /// assert_eq!(bsk.base_log(), DecompositionBaseLog(5));
    /// assert_eq!(bsk.key_size(), LweDimension(4));
    /// ```
    pub fn allocate_complex(
        value: Complex64,
        rlwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        key_size: LweDimension,
    ) -> Self {
        let mut tensor = Tensor::from_container(AlignedVec::new(
            key_size.0 * decomp_level.0 * rlwe_size.0 * rlwe_size.0 * poly_size.0,
        ));
        tensor.as_mut_tensor().fill_with_element(value);
        BootstrapKey {
            tensor,
            decomp_level,
            decomp_base_log,
            rlwe_size,
            poly_size,
        }
    }
}

impl<Cont> BootstrapKey<Cont> {
    /// Creates a bootstrapping key from an existing container of values.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweDimension};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// let vector = vec![0u32; 10*5*4*4*15];
    /// let bsk = BootstrapKey::from_container(
    ///     vector.as_slice(),
    ///     GlweSize(4),
    ///     PolynomialSize(10),    
    ///     DecompositionLevelCount(5),
    ///     DecompositionBaseLog(4)
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(10));
    /// assert_eq!(bsk.glwe_size(), GlweSize(4));
    /// assert_eq!(bsk.level_count(), DecompositionLevelCount(5));
    /// assert_eq!(bsk.base_log(), DecompositionBaseLog(4));
    /// assert_eq!(bsk.key_size(), LweDimension(15));
    /// ```
    pub fn from_container(
        cont: Cont,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
    ) -> BootstrapKey<Cont>
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() =>
            decomp_level.0,
            glwe_size.0 * glwe_size.0,
            poly_size.0
        );
        BootstrapKey {
            tensor,
            rlwe_size: glwe_size,
            poly_size,
            decomp_level,
            decomp_base_log,
        }
    }

    /// Returns the size of the polynomials used in the bootstrapping key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(9));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns the size of the GLWE ciphertexts used in the bootstrapping key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// assert_eq!(bsk.glwe_size(), GlweSize(7));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.rlwe_size
    }

    /// Returns the number of levels used to decompose the key bits.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// assert_eq!(bsk.level_count(), DecompositionLevelCount(3));
    /// ```
    pub fn level_count(&self) -> DecompositionLevelCount {
        self.decomp_level
    }

    /// Returns the logarithm of the base used to decompose the key bits.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// assert_eq!(bsk.base_log(), DecompositionBaseLog(5));
    /// ```
    pub fn base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns the size of the LWE encrypted key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// assert_eq!(bsk.key_size(), LweDimension(4));
    /// ```
    pub fn key_size(&self) -> LweDimension
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() =>
            self.poly_size.0,
            self.rlwe_size.0 * self.rlwe_size.0,
            self.decomp_level.0
        );
        LweDimension(
            self.as_tensor().len()
                / (self.rlwe_size.0 * self.rlwe_size.0 * self.poly_size.0 * self.decomp_level.0),
        )
    }

    /// Generate a new bootstrap key from the input parameters, and fills the current container
    /// with it.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension, GlweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::crypto::secret::{LweSecretKey, GlweSecretKey};
    /// use concrete_core::math::dispersion::LogStandardDev;
    /// use concrete_core::math::random::RandomGenerator;
    /// let mut generator = RandomGenerator::new(None);
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(6), PolynomialSize(9));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// let mut bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     glwe_dim.to_glwe_size(),
    ///     poly_size,
    ///     dec_lc,
    ///     dec_bl,
    ///     lwe_dim
    /// );
    /// let lwe_sk = LweSecretKey::generate(lwe_dim, &mut generator);
    /// let glwe_sk = GlweSecretKey::generate(glwe_dim, poly_size, &mut generator);
    /// bsk.fill_with_new_key(
    ///     &lwe_sk,
    ///     &glwe_sk,
    ///     LogStandardDev::from_log_standard_dev(-15.),
    ///     &mut generator
    /// );
    /// ```
    pub fn fill_with_new_key<LweCont, RlweCont, Scalar>(
        &mut self,
        lwe_secret_key: &LweSecretKey<LweCont>,
        glwe_secret_key: &GlweSecretKey<RlweCont>,
        noise_parameters: impl DispersionParameter,
        generator: &mut RandomGenerator,
    ) where
        Self: AsMutTensor<Element = Scalar>,
        LweSecretKey<LweCont>: AsRefTensor<Element = bool>,
        GlweSecretKey<RlweCont>: AsRefTensor<Element = bool>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(self.key_size().0 => lwe_secret_key.key_size().0);
        self.as_mut_tensor()
            .fill_with_element(<Scalar as Numeric>::ZERO);
        for (mut rgsw, sk_scalar) in self.ggsw_iter_mut().zip(lwe_secret_key.as_tensor().iter()) {
            let encoded = if *sk_scalar {
                Plaintext(Scalar::ONE)
            } else {
                Plaintext(Scalar::ZERO)
            };
            glwe_secret_key.encrypt_constant_ggsw(
                &mut rgsw,
                &encoded,
                noise_parameters.clone(),
                generator,
            );
        }
    }

    /// Generate a new bootstrap key from the input parameters, and fills the current container
    /// with it.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension, GlweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::crypto::secret::{LweSecretKey, GlweSecretKey};
    /// use concrete_core::math::dispersion::LogStandardDev;
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(6), PolynomialSize(9));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// use concrete_core::math::random::RandomGenerator;
    /// let mut generator = RandomGenerator::new(None);
    /// let mut bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     glwe_dim.to_glwe_size(),
    ///     poly_size,
    ///     dec_lc,
    ///     dec_bl,
    ///     lwe_dim
    /// );
    /// let lwe_sk = LweSecretKey::generate(lwe_dim, &mut generator);
    /// let glwe_sk = GlweSecretKey::generate(glwe_dim, poly_size, &mut generator);
    /// bsk.fill_with_new_trivial_key(
    ///     &lwe_sk,
    ///     &glwe_sk,  
    ///     LogStandardDev::from_log_standard_dev(-15.),
    ///     &mut generator
    /// );
    /// ```
    pub fn fill_with_new_trivial_key<LweCont, RlweCont, Scalar>(
        &mut self,
        lwe_secret_key: &LweSecretKey<LweCont>,
        rlwe_secret_key: &GlweSecretKey<RlweCont>,
        noise_parameters: impl DispersionParameter,
        generator: &mut RandomGenerator,
    ) where
        Self: AsMutTensor<Element = Scalar>,
        LweSecretKey<LweCont>: AsRefTensor<Element = bool>,
        GlweSecretKey<RlweCont>: AsRefTensor<Element = bool>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(self.key_size().0 => lwe_secret_key.key_size().0);
        for (mut rgsw, sk_scalar) in self.ggsw_iter_mut().zip(lwe_secret_key.as_tensor().iter()) {
            let encoded = if *sk_scalar {
                Plaintext(Scalar::ONE)
            } else {
                Plaintext(Scalar::ZERO)
            };
            rlwe_secret_key.trivial_encrypt_constant_ggsw(
                &mut rgsw,
                &encoded,
                noise_parameters.clone(),
                generator,
            );
        }
    }

    /// Returns an iterator over the borrowed GGSW ciphertext composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// for ggsw in bsk.ggsw_iter(){
    ///     assert_eq!(ggsw.polynomial_size(), PolynomialSize(9));
    ///     assert_eq!(ggsw.glwe_size(), GlweSize(7));
    ///     assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// }
    /// assert_eq!(bsk.ggsw_iter().count(), 4);
    /// ```
    pub fn ggsw_iter(
        &self,
    ) -> impl Iterator<Item = GgswCiphertext<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        let chunks_size =
            self.rlwe_size.0 * self.rlwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let rlwe_size = self.rlwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .map(move |tensor| {
                GgswCiphertext::from_container(
                    tensor.into_container(),
                    rlwe_size,
                    poly_size,
                    base_log,
                )
            })
    }

    /// Returns an iterator over the mutably borrowed GGSW ciphertext composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// for mut ggsw in bsk.ggsw_iter_mut(){
    ///     ggsw.as_mut_tensor().fill_with_element(0);
    /// }
    /// assert!(bsk.as_tensor().iter().all(|a| *a==0));
    /// assert_eq!(bsk.ggsw_iter_mut().count(), 4);
    /// ```
    pub fn ggsw_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = GgswCiphertext<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size =
            self.rlwe_size.0 * self.rlwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let rlwe_size = self.rlwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(move |tensor| {
                GgswCiphertext::from_container(
                    tensor.into_container(),
                    rlwe_size,
                    poly_size,
                    base_log,
                )
            })
    }

    /// Fills a complex bootstrapping key with the fourier transform of a bootstrapping key in
    /// coefficient domain.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::math::fft::Complex64;
    /// let bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// let mut frr_bsk = BootstrapKey::allocate_complex(
    ///     Complex64::new(0.,0.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// frr_bsk.fill_with_forward_fourier(&bsk);
    /// ```
    pub fn fill_with_forward_fourier<InputCont, Scalar>(
        &mut self,
        coef_bsk: &BootstrapKey<InputCont>,
    ) where
        Self: AsMutTensor<Element = Complex64>,
        BootstrapKey<InputCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We create an fft transformer
        let mut fft = Fft::new(self.poly_size);

        // We create an aligned buffer
        let mut fft_buffer = FourierPolynomial::allocate(Complex64::new(0., 0.), self.poly_size);

        // We transform every polynomial into
        for (mut fourier_poly, coef_poly) in self.fourier_poly_iter_mut().zip(coef_bsk.poly_iter())
        {
            fft.forward_as_torus(&mut fft_buffer, &coef_poly);
            fourier_poly
                .as_mut_tensor()
                .fill_with_one(fft_buffer.as_tensor(), |a| *a);
        }
    }

    /// For a complex bootstrapping key, returns an iterator over borrowed complex
    /// polynomials composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::math::fft::Complex64;
    /// let frr_bsk = BootstrapKey::allocate_complex(
    ///     Complex64::new(0.,0.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// for poly in frr_bsk.fourier_poly_iter(){
    ///     assert_eq!(poly.polynomial_size(), PolynomialSize(256));
    /// }
    /// assert_eq!(frr_bsk.fourier_poly_iter().count(), 7*7*3*4)
    /// ```
    pub fn fourier_poly_iter(&self) -> impl Iterator<Item = FourierPolynomial<&[Complex64]>>
    where
        Self: AsRefTensor<Element = Complex64>,
    {
        self.as_tensor()
            .subtensor_iter(self.poly_size.0)
            .map(|chunk| FourierPolynomial::from_container(chunk.into_container()))
    }

    /// For a complex bootstrapping key, returns an iterator over mutably borrowed complex
    /// polynomials composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::math::fft::Complex64;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut frr_bsk = BootstrapKey::allocate_complex(
    ///     Complex64::new(0.,0.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// for mut poly in frr_bsk.fourier_poly_iter_mut(){
    ///     poly.as_mut_tensor().fill_with_element(Complex64::new(5.,5.));
    /// }
    /// assert!(frr_bsk.as_tensor().iter().all(|a| *a==Complex64::new(5.,5.)));
    /// assert_eq!(frr_bsk.fourier_poly_iter_mut().count(), 7*7*3*4)
    /// ```
    pub fn fourier_poly_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = FourierPolynomial<&mut [Complex64]>>
    where
        Self: AsMutTensor<Element = Complex64>,
    {
        let poly_size = self.poly_size.0;
        self.as_mut_tensor()
            .subtensor_iter_mut(poly_size)
            .map(|chunk| FourierPolynomial::from_container(chunk.into_container()))
    }

    /// Returns an iterator over borrowed polynomials composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::math::fft::Complex64;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// for poly in bsk.poly_iter(){
    ///     assert_eq!(poly.polynomial_size(), PolynomialSize(256));
    /// }
    /// assert_eq!(bsk.poly_iter().count(), 7*7*3*4)
    /// ```
    pub fn poly_iter<'a, Coef>(&'a self) -> impl Iterator<Item = Polynomial<&[Coef]>>
    where
        Self: AsRefTensor<Element = Coef>,
        Coef: UnsignedTorus + 'a,
    {
        let poly_size = self.poly_size.0;
        self.as_tensor()
            .subtensor_iter(poly_size)
            .map(|chunk| Polynomial::from_container(chunk.into_container()))
    }

    /// Returns an iterator over mutably borrowed polynomials composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::bootstrap::BootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweSize, LweDimension};
    /// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::math::fft::Complex64;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut bsk = BootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4)
    /// );
    /// for mut poly in bsk.poly_iter_mut(){
    ///     poly.as_mut_tensor().fill_with_element(0u32);
    /// }
    /// assert!(bsk.as_tensor().iter().all(|a| *a==0));
    /// assert_eq!(bsk.poly_iter_mut().count(), 7*7*3*4)
    /// ```
    pub fn poly_iter_mut<'a, Coef>(&'a mut self) -> impl Iterator<Item = Polynomial<&mut [Coef]>>
    where
        Self: AsMutTensor<Element = Coef>,
        Coef: UnsignedTorus + 'a,
    {
        let poly_size = self.poly_size.0;
        self.as_mut_tensor()
            .subtensor_iter_mut(poly_size)
            .map(|chunk| Polynomial::from_container(chunk.into_container()))
    }
}
