use concrete_commons::dispersion::LogStandardDev;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, PlaintextCount};

use crate::backends::core::private::crypto::encoding::PlaintextList;
use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::crypto::secret::GlweSecretKey;
use crate::backends::core::private::math::tensor::{AsRefSlice, AsRefTensor};
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::{self};

use super::{StandardGgswCiphertext, StandardGgswSeededCiphertext};

fn test_seeded_ggsw<T: UnsignedTorus>() {
    // random settings
    let nb_ct = test_tools::random_ciphertext_count(10);
    let dimension = test_tools::random_glwe_dimension(5);
    let polynomial_size = test_tools::random_polynomial_size(200);
    let noise_parameters = LogStandardDev::from_log_standard_dev(-50.);
    let decomp_level = DecompositionLevelCount(3);
    let decomp_base_log = DecompositionBaseLog(7);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut generator = EncryptionRandomGenerator::new(None);

    // generates a secret key
    let sk = GlweSecretKey::generate_binary(dimension, polynomial_size, &mut secret_generator);

    // generates random plaintexts
    let plaintext_vector =
        PlaintextList::from_tensor(secret_generator.random_uniform_tensor(nb_ct.0));
    let mut seeded_decryptions = PlaintextList::from_container(vec![
        T::ZERO;
        polynomial_size.0
            * nb_ct.0
            * decomp_level.0
            * (dimension.0 + 1)
    ]);
    let mut control_decryptions = PlaintextList::from_container(vec![
        T::ZERO;
        polynomial_size.0
            * nb_ct.0
            * decomp_level.0
            * (dimension.0 + 1)
    ]);

    for (plaintext, (mut seeded_decryption, mut control_decryption)) in
        plaintext_vector.plaintext_iter().zip(
            seeded_decryptions
                .sublist_iter_mut(PlaintextCount(
                    polynomial_size.0 * decomp_level.0 * (dimension.0 + 1),
                ))
                .zip(control_decryptions.sublist_iter_mut(PlaintextCount(
                    polynomial_size.0 * decomp_level.0 * (dimension.0 + 1),
                ))),
        )
    {
        // encrypts
        let mut seeded_ggsw = StandardGgswSeededCiphertext::allocate(
            T::ZERO,
            polynomial_size,
            dimension.to_glwe_size(),
            decomp_level,
            decomp_base_log,
        );
        sk.encrypt_constant_seeded_ggsw(&mut seeded_ggsw, &plaintext, noise_parameters.clone());

        // expands
        let mut ggsw_expanded = StandardGgswCiphertext::allocate(
            T::ZERO,
            polynomial_size,
            dimension.to_glwe_size(),
            decomp_level,
            decomp_base_log,
        );
        seeded_ggsw.expand_into(&mut ggsw_expanded);

        // control encryption
        let mut ggsw = StandardGgswCiphertext::allocate(
            T::ZERO,
            polynomial_size,
            dimension.to_glwe_size(),
            decomp_level,
            decomp_base_log,
        );
        sk.encrypt_constant_ggsw(
            &mut ggsw,
            &plaintext,
            noise_parameters.clone(),
            &mut generator,
        );

        for ((seeded_glwe, control_glwe), (mut seeded_poly, mut control_poly)) in ggsw_expanded
            .as_glwe_list()
            .ciphertext_iter()
            .zip(ggsw.as_glwe_list().ciphertext_iter())
            .zip(
                seeded_decryption
                    .sublist_iter_mut(PlaintextCount(polynomial_size.0))
                    .zip(control_decryption.sublist_iter_mut(PlaintextCount(polynomial_size.0))),
            )
        {
            // decrypts
            sk.decrypt_glwe(&mut seeded_poly, &seeded_glwe);
            sk.decrypt_glwe(&mut control_poly, &control_glwe);

            // test
            for (coeff_control, coeff_seeded) in control_poly
                .as_tensor()
                .as_slice()
                .iter()
                .zip(seeded_poly.as_tensor().as_slice().iter())
            {
                let mut decoded_seeded = *coeff_seeded >> (T::BITS - 22);
                if decoded_seeded & T::ONE == T::ONE {
                    decoded_seeded += T::ONE;
                }
                decoded_seeded >>= 1;
                decoded_seeded %= T::ONE << 21;

                let mut decoded_control = *coeff_control >> (T::BITS - 22);
                if decoded_control & T::ONE == T::ONE {
                    decoded_control += T::ONE;
                }
                decoded_control >>= 1;
                decoded_control %= T::ONE << 21;
                assert_eq!(decoded_seeded, decoded_control);
            }
        }
    }
}

#[test]
fn test_seeded_ggsw_u32() {
    test_seeded_ggsw::<u32>()
}

#[test]
fn test_seeded_ggsw_u64() {
    test_seeded_ggsw::<u64>()
}
