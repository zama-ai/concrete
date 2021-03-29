use crate::crypto::encoding::PlaintextList;
use crate::crypto::glwe::GlweList;
use crate::crypto::secret::GlweSecretKey;
use crate::crypto::{CiphertextCount, GlweDimension, UnsignedTorus};
use crate::math::dispersion::LogStandardDev;
use crate::math::polynomial::PolynomialSize;
use crate::math::random;
use crate::test_tools::assert_delta_std_dev;
#[cfg(feature = "longtest")]
use crate::test_tools::{random_ciphertext_count, random_glwe_dimension, random_polynomial_size};

fn test_glwe<T: UnsignedTorus>(
    nb_ct: CiphertextCount,
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
) {
    // random settings
    let noise_parameter = LogStandardDev::from_log_standard_dev(-20.);

    // generates a secret key
    let sk = GlweSecretKey::generate(dimension, polynomial_size);

    // generates random plaintexts
    let plaintexts =
        PlaintextList::from_tensor(random::random_uniform_tensor(nb_ct.0 * polynomial_size.0));

    // encrypts
    let mut ciphertext = GlweList::allocate(T::ZERO, polynomial_size, dimension, nb_ct);
    sk.encrypt_glwe_list(&mut ciphertext, &plaintexts, noise_parameter.clone());

    // decrypts
    let mut decryptions =
        PlaintextList::from_tensor(random::random_uniform_tensor(nb_ct.0 * polynomial_size.0));
    sk.decrypt_glwe_list(&mut decryptions, &ciphertext);

    // test
    assert_delta_std_dev(&plaintexts, &decryptions, noise_parameter);
}

#[test]
fn test_glwe_encrypt_decrypt_u32() {
    #[cfg(not(feature = "longtest"))]
    test_glwe::<u32>(CiphertextCount(10), GlweDimension(20), PolynomialSize(32));
    #[cfg(feature = "longtest")]
    test_glwe::<u32>(
        random_ciphertext_count(100..200),
        random_glwe_dimension(100..200),
        random_polynomial_size(100..200),
    );
}

#[test]
fn test_glwe_encrypt_decrypt_u64() {
    #[cfg(not(feature = "longtest"))]
    test_glwe::<u64>(CiphertextCount(10), GlweDimension(20), PolynomialSize(64));
    #[cfg(feature = "longtest")]
    test_glwe::<u64>(
        random_ciphertext_count(100..200),
        random_glwe_dimension(100..200),
        random_polynomial_size(100..200),
    );
}
