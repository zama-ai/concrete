//! RGSW Tensor Operations
//! * Contains every function only related to tensors of RGSW samples

use crate::operators::crypto::{SecretKey, RLWE};
use crate::operators::math::FFT;
use crate::types::{C2CPlanTorus, CTorus};
use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
use itertools::{enumerate, izip};

pub trait RGSW: Sized {
    fn create_bootstrapping_key(
        trgsw: &mut [Self],
        base_log: usize,
        level: usize,
        rlwe_dimension: usize,
        polynomial_size: usize,
        std: f64,
        lwe_sk: &[Self],
        rlwe_sk: &[Self],
    );
    fn create_fourier_bootstrapping_key(
        trgsw_fft: &mut [CTorus],
        base_log: usize,
        level: usize,
        rlwe_dimension: usize,
        polynomial_size: usize,
        std: f64,
        lwe_sk: &[Self],
        rlwe_sk: &[Self],
    );
    fn create_trivial_bootstrapping_key(
        trgsw: &mut [Self],
        base_log: usize,
        level: usize,
        rlwe_dimension: usize,
        polynomial_size: usize,
        _std: f64,
        lwe_sk: &[Self],
        _rlwe_sk: &[Self],
    );
    fn create_trivial_fourier_bootstrapping_key(
        trgsw_fft: &mut [CTorus],
        base_log: usize,
        level: usize,
        rlwe_dimension: usize,
        polynomial_size: usize,
        std: f64,
        lwe_sk: &[Self],
        rlwe_sk: &[Self],
    );
}

macro_rules! impl_trait_rgsw {
    ($T:ty,$DOC:expr) => {
        impl RGSW for $T {
            /// Create a boostrapping key
            /// # Arguments
            /// * `trgsw` - TRGSW ciphertext (output)
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * `rlwe_dimension` - size of the mask of RLWE samples
            /// * `polynomial_size` - number of coefficients of the polynomial
            /// * `std`: standard deviation of the encryption noise
            /// * `lwe_sk`: secret key used to encrypt the sample we want to bootstrapp
            /// * `rlwe_sk`: secret key used to encrypt the RGSW
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::crypto::{rgsw, SecretKey, RGSW};
            /// use concrete_lib::operators::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let std: f64 = 0.00003;
            /// let lwe_dimension: usize = 256;
            /// let rlwe_dimension: usize = 20;
            /// let polynomial_size: usize = 128;
            /// let level: usize = 4;
            /// let base_log: usize = 2;
            /// let n_slots: usize = 1;
            ///
            /// // generates an LWE secret key
            /// let lwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(lwe_dimension, 1);
            /// let mut lwe_sk: Vec<Torus> = vec![0; lwe_sk_len];
            /// Tensor::uniform_random_default(&mut lwe_sk);
            ///
            /// // generates an RLWE secret key
            /// let rlwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(rlwe_dimension, polynomial_size);
            /// let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
            /// Tensor::uniform_random_default(&mut rlwe_sk);
            ///
            /// // sizes for the output
            /// let size: usize =
            ///     rgsw::get_trgsw_size(rlwe_dimension, polynomial_size, n_slots, level);
            ///
            /// // allocation of the output
            /// let mut trgsw: Vec<Torus> = vec![0; size];
            ///
            /// // creation of the bootstrapping key
            /// RGSW::create_bootstrapping_key(
            ///     &mut trgsw,
            ///     base_log,
            ///     level,
            ///     rlwe_dimension,
            ///     polynomial_size,
            ///     std,
            ///     &lwe_sk,
            ///     &rlwe_sk,
            /// );
            /// ```
            fn create_bootstrapping_key(
                trgsw: &mut [$T],
                base_log: usize,
                level: usize,
                rlwe_dimension: usize,
                polynomial_size: usize,
                std: f64,
                lwe_sk: &[$T],
                rlwe_sk: &[$T],
            ) {
                // fill with encryptions of zero
                RLWE::zero_encryption(trgsw, rlwe_sk, rlwe_dimension, polynomial_size, std);

                // computes needed sizes
                let trgsw_size: usize = get_trgsw_size(rlwe_dimension, polynomial_size, 1, level) ;

                // add gadget matrices
                for (i, trgsw_i) in enumerate(
                    trgsw
                        .chunks_mut(trgsw_size)
                ) {
                    let bit = SecretKey::get_bit(lwe_sk, i);
                    if rlwe_dimension == 1 {
                        RLWE::add_gadgetmatrix(
                            trgsw_i,
                            bit as $T,
                            rlwe_dimension,
                            polynomial_size,
                            base_log,
                            level,
                        );
                    } else {
                        RLWE::add_gadgetmatrix_generic(
                            trgsw_i,
                            bit as $T,
                            rlwe_dimension,
                            polynomial_size,
                            base_log,
                            level,
                        )
                    }
                }
            }

            /// Create a boostrapping key already in the Fourier domain
            /// # Comments
            /// * panics if there are no twiddle for the requested polynomial size
            /// # Arguments
            /// * `trgsw` - TRGSW ciphertext (output)
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * `rlwe_dimension` - size of the mask of RLWE samples
            /// * `polynomial_size` - number of coefficients of the polynomial
            /// * `std`: standard deviation of the encryption noise
            /// * `lwe_sk`: secret key used to encrypt the sample we want to bootstrapp
            /// * `rlwe_sk`: secret key used to encrypt the RGSW
            /// # Example
            /// ```rust
            /// use num_traits::Zero;
            /// use concrete_lib::operators::crypto::{rgsw, SecretKey, RGSW};
            /// use concrete_lib::operators::math::Tensor;
            /// use concrete_lib::types::{CTorus};
            ///
            /// type Torus = u32; // or u64
            ///
            /// // settings
            /// let std: f64 = 0.00003;
            /// let lwe_dimension: usize = 256;
            /// let rlwe_dimension: usize = 4;
            /// let polynomial_size: usize = 512;
            /// let level: usize = 4;
            /// let base_log: usize = 2;
            /// let n_slots: usize = 1;
            ///
            /// // generates an LWE secret key
            /// let lwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(lwe_dimension, 1);
            /// let mut lwe_sk: Vec<Torus> = vec![0; lwe_sk_len];
            /// Tensor::uniform_random_default(&mut lwe_sk);
            ///
            /// // generates an RLWE secret key
            /// let rlwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(rlwe_dimension, polynomial_size);
            /// let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
            /// Tensor::uniform_random_default(&mut rlwe_sk);
            ///
            /// // sizes for the output
            /// let size: usize =
            ///     rgsw::get_trgsw_size(rlwe_dimension, polynomial_size, n_slots, level) ;
            ///
            /// // allocation of the output
            /// let mut trgsw: Vec<CTorus> = vec![CTorus::zero(); size];
            ///
            /// // creation of the bootstrapping key
            /// RGSW::create_fourier_bootstrapping_key(
            ///     &mut trgsw,
            ///     base_log,
            ///     level,
            ///     rlwe_dimension,
            ///     polynomial_size,
            ///     std,
            ///     &lwe_sk,
            ///     &rlwe_sk,
            /// );
            /// ```
            fn create_fourier_bootstrapping_key(
                trgsw_fft: &mut [CTorus],
                base_log: usize,
                level: usize,
                rlwe_dimension: usize,
                polynomial_size: usize,
                std: f64,
                lwe_sk: &[$T],
                rlwe_sk: &[$T],
            ) {
                // fetch the twiddles (panics if there are no twiddle for the requested polynomial size)
                let twiddles = TWIDDLES_TORUS!(polynomial_size);

                // allocate temporary outputs
                let mut trgsw = vec![0 as $T; trgsw_fft.len()];

                // create a regular non fourier bootstrapping key
                Self::create_bootstrapping_key(
                    &mut trgsw,
                    base_log,
                    level,
                    rlwe_dimension,
                    polynomial_size,
                    std,
                    lwe_sk,
                    rlwe_sk,
                );

                // allocate temporary variables to put things into the fourier domain
                let mut fourier_polynomial_tmp = AlignedVec::<CTorus>::new(polynomial_size);
                let mut fourier_polynomial_cp = AlignedVec::<CTorus>::new(polynomial_size);

                // putting key into the fourier domain
                let mut fft = C2CPlanTorus::aligned(&[polynomial_size], Sign::Forward, Flag::Measure).unwrap();

                // masks
                for (fourier_polynomial, coeff_polynomial) in izip!(
                    trgsw_fft.chunks_mut(polynomial_size),
                    trgsw.chunks(polynomial_size)
                ) {
                    FFT::put_in_fft_domain_torus(
                        &mut fourier_polynomial_cp,
                        &mut fourier_polynomial_tmp,
                        coeff_polynomial,
                        twiddles,
                        &mut fft,
                    );

                    // copy the FFT result into a slice
                    for (coeff_cp, coeff) in fourier_polynomial_cp
                        .iter()
                        .zip(fourier_polynomial.iter_mut())
                    {
                        *coeff = *coeff_cp;
                    }
                }

            }

            /// Create a trivial boostrapping key
            /// # Arguments
            /// * `trgsw` - a TRGSW ciphertext (output)
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * * `rlwe_dimension` - size of the mask of RLWE samples
            /// * `polynomial_size` - number of coefficients of the polynomial
            /// * `_std`: standard deviation of the encryption noise
            /// * `lwe_sk`: secret key used to encrypt the sample we want to bootstrapp
            /// * `_rlwe_sk`: secret key used to encrypt the RGSW
            /// * `_lwe_sk_nb_bits` - number of bit of binary key
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::crypto::{rgsw, SecretKey, RGSW};
            /// use concrete_lib::operators::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let std: f64 = 0.00003;
            /// let lwe_dimension: usize = 256;
            /// let rlwe_dimension: usize = 20;
            /// let polynomial_size: usize = 128;
            /// let level: usize = 4;
            /// let base_log: usize = 2;
            /// let n_slots: usize = 1;
            ///
            /// // generates an LWE secret key
            /// let lwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(lwe_dimension, 1);
            /// let mut lwe_sk: Vec<Torus> = vec![0; lwe_sk_len];
            /// Tensor::uniform_random_default(&mut lwe_sk);
            ///
            /// // generates an RLWE secret key
            /// let rlwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(rlwe_dimension, polynomial_size);
            /// let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
            /// Tensor::uniform_random_default(&mut rlwe_sk);
            ///
            /// // sizes for the output
            /// let size: usize =
            ///     rgsw::get_trgsw_size(rlwe_dimension, polynomial_size, n_slots, level) ;
            ///
            /// // allocation of the output
            /// let mut trgsw: Vec<Torus> = vec![0; size];
            ///
            /// // creation of the bootstrapping key
            /// RGSW::create_trivial_bootstrapping_key(
            ///     &mut trgsw,
            ///     base_log,
            ///     level,
            ///     rlwe_dimension,
            ///     polynomial_size,
            ///     std,
            ///     &lwe_sk,
            ///     &rlwe_sk,
            /// );
            /// ```
            fn create_trivial_bootstrapping_key(
                trgsw: &mut [$T],
                base_log: usize,
                level: usize,
                rlwe_dimension: usize,
                polynomial_size: usize,
                _std: f64,
                lwe_sk: &[$T],
                _rlwe_sk: &[$T],
            ) {
                // RLWE::zero_encryption(trgsw, rlwe_sk, rlwe_dimension, polynomial_size, std);

                // computes needed sizes
                let trgsw_size: usize = get_trgsw_size(rlwe_dimension, polynomial_size, 1, level) ;

                // add gadget matrices
                for (i, trgsw_i) in enumerate(
                    trgsw
                        .chunks_mut(trgsw_size)
                ) {
                    let bit = SecretKey::get_bit(lwe_sk, i);
                    if rlwe_dimension == 1 {
                        RLWE::add_gadgetmatrix(
                            trgsw_i,
                            bit as $T,
                            rlwe_dimension,
                            polynomial_size,
                            base_log,
                            level,
                        );
                    } else {
                        RLWE::add_gadgetmatrix_generic(
                            trgsw_i,
                            bit as $T,
                            rlwe_dimension,
                            polynomial_size,
                            base_log,
                            level,
                        )
                    }
                }
            }

            /// Create a trivial bootstrapping key already in the fourier domain
            /// # Arguments
            /// * `trgsw` - TRGSW ciphertext (output)
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * `rlwe_dimension` - size of the mask of RLWE samples
            /// * `polynomial_size` - number of coefficients of the polynomial
            /// * `std`: standard deviation of the encryption noise
            /// * `lwe_sk`: secret key used to encrypt the sample we want to bootstrap
            /// * `rlwe_sk`: secret key used to encrypt the RGSW
            /// * `lwe_sk_nb_bits` - number of bit of binary key
            /// # Example
            /// ```rust
            /// use num_traits::Zero;
            /// use concrete_lib::operators::crypto::{rgsw, SecretKey, RGSW};
            /// use concrete_lib::operators::math::Tensor;
            /// use concrete_lib::types::{CTorus};
            ///
            /// type Torus = u32; // or u64
            ///
            /// // settings
            /// let std: f64 = 0.00003;
            /// let lwe_dimension: usize = 256;
            /// let rlwe_dimension: usize = 4;
            /// let polynomial_size: usize = 512;
            /// let level: usize = 4;
            /// let base_log: usize = 2;
            /// let n_slots: usize = 1;
            ///
            /// // generates an LWE secret key
            /// let lwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(lwe_dimension, 1);
            /// let mut lwe_sk: Vec<Torus> = vec![0; lwe_sk_len];
            /// Tensor::uniform_random_default(&mut lwe_sk);
            ///
            /// // generates an RLWE secret key
            /// let rlwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(rlwe_dimension, polynomial_size);
            /// let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
            /// Tensor::uniform_random_default(&mut rlwe_sk);
            ///
            /// // sizes for the output
            /// let size: usize =
            ///     rgsw::get_trgsw_size(rlwe_dimension, polynomial_size, n_slots, level) ;
            ///
            /// // allocation of the output
            /// let mut trgsw: Vec<CTorus> = vec![CTorus::zero(); size];
            ///
            /// // creation of the bootstrapping key
            /// RGSW::create_trivial_fourier_bootstrapping_key(
            ///     &mut trgsw,
            ///     base_log,
            ///     level,
            ///     rlwe_dimension,
            ///     polynomial_size,
            ///     std,
            ///     &lwe_sk,
            ///     &rlwe_sk,
            /// );
            /// ```
            fn create_trivial_fourier_bootstrapping_key(
                trgsw_fft: &mut [CTorus],
                base_log: usize,
                level: usize,
                rlwe_dimension: usize,
                polynomial_size: usize,
                std: f64,
                lwe_sk: &[$T],
                rlwe_sk: &[$T],
            ) {
                // fetch the twiddles (panics if there are no twiddle for the requested polynomial size)
                let twiddles = TWIDDLES_TORUS!(polynomial_size);

                // allocate temporary outputs
                let mut trgsw = vec![0 as $T; trgsw_fft.len()];

                // create a regular non fourier bootstrapping key
                Self::create_trivial_bootstrapping_key(
                    &mut trgsw,
                    base_log,
                    level,
                    rlwe_dimension,
                    polynomial_size,
                    std,
                    lwe_sk,
                    rlwe_sk,
                );

                // allocate temporary variables to put things into the fourier domain
                let mut fourier_polynomial_tmp = AlignedVec::<CTorus>::new(polynomial_size);
                let mut fourier_polynomial_cp = AlignedVec::<CTorus>::new(polynomial_size);

                // putting key into the fourier domain
                let mut fft = C2CPlanTorus::aligned(&[polynomial_size], Sign::Forward, Flag::Measure).unwrap();

                // masks
                for (fourier_polynomial, coeff_polynomial) in izip!(
                    trgsw_fft.chunks_mut(polynomial_size),
                    trgsw.chunks(polynomial_size)
                ) {
                    FFT::put_in_fft_domain_torus(
                        &mut fourier_polynomial_cp,
                        &mut fourier_polynomial_tmp,
                        coeff_polynomial,
                        twiddles,
                        &mut fft,
                    );

                    // copy the FFT result into a slice
                    for (coeff_cp, coeff) in fourier_polynomial_cp
                        .iter()
                        .zip(fourier_polynomial.iter_mut())
                    {
                        *coeff = *coeff_cp;
                    }
                }
            }
        }
    };
}

impl_trait_rgsw!(u32, "type Torus = u32;");
impl_trait_rgsw!(u64, "type Torus = u64;");

/// Returns the size of a TRGSW according to a set of parameters
/// # Arguments
/// * `dimension` - size of the RLWE mask
/// * `polynomial_size` - number of coefficients in polynomials
/// * `n_slots` - number of batched ciphertexts
/// * `level` - number of blocks of the gadget matrix
/// # Output
/// * the computed size as a usize
/// # Example
/// ```rust
/// use concrete_lib::operators::crypto::rgsw;
///
/// // settings
/// let n_slots: usize = 10;
/// let dimension: usize = 20;
/// let polynomial_size: usize = 128;
/// let level: usize = 4;
///
/// let size: usize = rgsw::get_trgsw_size(dimension, polynomial_size, n_slots, level);
/// ```
pub fn get_trgsw_size(
    dimension: usize,
    polynomial_size: usize,
    n_slots: usize,
    level: usize,
) -> usize {
    return dimension * (dimension + 1) * polynomial_size * level * n_slots
        + polynomial_size * level * (dimension + 1) * n_slots;
}
