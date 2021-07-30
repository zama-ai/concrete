use concrete_commons::{
    dispersion::LogStandardDev,
    parameters::{DecompositionBaseLog, DecompositionLevelCount, PolynomialSize},
};

use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::math::{tensor::AsRefSlice, torus::UnsignedTorus};
use crate::backends::core::private::{crypto::bootstrap::StandardSeededBootstrapKey, test_tools};
use crate::backends::core::private::{
    crypto::{
        bootstrap::StandardBootstrapKey,
        encoding::PlaintextList,
        secret::{GlweSecretKey, LweSecretKey},
    },
    math::tensor::AsRefTensor,
};

fn test_standard_seeded_bootstrap<T: UnsignedTorus>() {
    // random settings
    let glwe_dimension = test_tools::random_glwe_dimension(5);
    let lwe_dimension = test_tools::random_lwe_dimension(100);
    let polynomial_size = test_tools::random_polynomial_size(4);
    let polynomial_size = PolynomialSize(1 << (7 + polynomial_size.0));
    let noise_parameters = LogStandardDev::from_log_standard_dev(-50.);
    let decomp_level = DecompositionLevelCount(3);
    let decomp_base_log = DecompositionBaseLog(7);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut generator = EncryptionRandomGenerator::new(None);

    // generates a secret key
    let glwe_sk =
        GlweSecretKey::generate_binary(glwe_dimension, polynomial_size, &mut secret_generator);

    let lwe_sk = LweSecretKey::generate_binary(lwe_dimension, &mut secret_generator);

    // allocation and generation of the key in coef domain:
    let mut coef_bsk_seeded = StandardSeededBootstrapKey::allocate(
        T::ZERO,
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        decomp_level,
        decomp_base_log,
        lwe_dimension,
    );
    coef_bsk_seeded.fill_with_new_key(&lwe_sk, &glwe_sk, noise_parameters);

    // expansion of the bootstrapping key
    let mut coef_bsk_expanded = StandardBootstrapKey::allocate(
        T::ZERO,
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        decomp_level,
        decomp_base_log,
        lwe_dimension,
    );
    coef_bsk_seeded.expand_into(&mut coef_bsk_expanded);

    //control
    let mut coef_bsk_control = StandardBootstrapKey::allocate(
        T::ZERO,
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        decomp_level,
        decomp_base_log,
        lwe_dimension,
    );
    coef_bsk_control.fill_with_new_key(&lwe_sk, &glwe_sk, noise_parameters, &mut generator);

    for (seeded_ggsw, control_ggsw) in coef_bsk_expanded
        .ggsw_iter_mut()
        .zip(coef_bsk_control.ggsw_iter_mut())
    {
        for (seeded_glwe, control_glwe) in seeded_ggsw
            .as_glwe_list()
            .ciphertext_iter()
            .zip(control_ggsw.as_glwe_list().ciphertext_iter())
        {
            let mut seeded_poly = PlaintextList::from_container(vec![T::ZERO; polynomial_size.0]);
            let mut control_poly = PlaintextList::from_container(vec![T::ZERO; polynomial_size.0]);
            // decrypts
            glwe_sk.decrypt_glwe(&mut seeded_poly, &seeded_glwe);
            glwe_sk.decrypt_glwe(&mut control_poly, &control_glwe);

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
fn test_standard_seeded_bootstrap_u32() {
    test_standard_seeded_bootstrap::<u32>()
}

#[test]
fn test_standard_seeded_bootstrap_u64() {
    test_standard_seeded_bootstrap::<u64>()
}
