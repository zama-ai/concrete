#include <cstdint>
#include <gtest/gtest.h>
#include <setup_and_teardown.h>
#include <stdio.h>
#include <stdlib.h>

const unsigned REPETITIONS = 5;
const unsigned SAMPLES = 100;

typedef struct {
  int lwe_dimension;
  double noise_variance;
  int message_modulus;
  int carry_modulus;
  int number_of_inputs;
} LinearAlgebraTestParams;

class LinearAlgebraTestPrimitives_u64
    : public ::testing::TestWithParam<LinearAlgebraTestParams> {
protected:
  int lwe_dimension;
  double noise_variance;
  int message_modulus;
  int carry_modulus;
  int number_of_inputs;
  int payload_modulus;
  uint64_t delta;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *lwe_sk_array;
  uint64_t *d_lwe_in_1_ct;
  uint64_t *d_lwe_in_2_ct;
  uint64_t *d_plaintext_2;
  uint64_t *d_cleartext;
  uint64_t *d_lwe_out_ct;
  uint64_t *lwe_in_1_ct;
  uint64_t *lwe_in_2_ct;
  uint64_t *lwe_out_ct;
  uint64_t *plaintexts_1;
  uint64_t *plaintexts_2;
  int num_samples;

public:
  // Test arithmetic functions
  void SetUp() {
    stream = cuda_create_stream(0);

    // TestParams
    lwe_dimension = (int)GetParam().lwe_dimension;
    noise_variance = (double)GetParam().noise_variance;
    message_modulus = (int)GetParam().message_modulus;
    carry_modulus = (int)GetParam().carry_modulus;
    number_of_inputs = (int)GetParam().number_of_inputs;

    payload_modulus = message_modulus * carry_modulus;
    // Value of the shift we multiply our messages by
    // In this test we use a smaller delta to avoid an overflow during
    // multiplication
    delta =
        ((uint64_t)(1) << 63) / (uint64_t)(payload_modulus * payload_modulus);

    linear_algebra_setup(stream, &csprng, &lwe_sk_array, &d_lwe_in_1_ct,
                         &d_lwe_in_2_ct, &d_lwe_out_ct, &lwe_in_1_ct,
                         &lwe_in_2_ct, &lwe_out_ct, &plaintexts_1,
                         &plaintexts_2, &d_plaintext_2, &d_cleartext,
                         lwe_dimension, noise_variance, payload_modulus, delta,
                         number_of_inputs, REPETITIONS, SAMPLES, gpu_index);
  }

  void TearDown() {
    linear_algebra_teardown(
        stream, &csprng, &lwe_sk_array, &d_lwe_in_1_ct, &d_lwe_in_2_ct,
        &d_lwe_out_ct, &lwe_in_1_ct, &lwe_in_2_ct, &lwe_out_ct, &plaintexts_1,
        &plaintexts_2, &d_plaintext_2, &d_cleartext, gpu_index);
  }
};

TEST_P(LinearAlgebraTestPrimitives_u64, addition) {
  void *v_stream = (void *)stream;
  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    uint64_t *lwe_sk = lwe_sk_array + (ptrdiff_t)(r * lwe_dimension);
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t *d_lwe_1_in =
          d_lwe_in_1_ct +
          (ptrdiff_t)((r * SAMPLES * number_of_inputs + s * number_of_inputs) *
                      (lwe_dimension + 1));
      uint64_t *d_lwe_2_in =
          d_lwe_in_2_ct +
          (ptrdiff_t)((r * SAMPLES * number_of_inputs + s * number_of_inputs) *
                      (lwe_dimension + 1));
      // Execute addition
      cuda_add_lwe_ciphertext_vector_64(stream, gpu_index, (void *)d_lwe_out_ct,
                                        (void *)d_lwe_1_in, (void *)d_lwe_2_in,
                                        lwe_dimension, number_of_inputs);

      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                               number_of_inputs * (lwe_dimension + 1) *
                                   sizeof(uint64_t),
                               stream, gpu_index);
      cuda_synchronize_stream(v_stream);
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext_1 = plaintexts_1[r * SAMPLES * number_of_inputs +
                                            s * number_of_inputs + i];
        uint64_t plaintext_2 = plaintexts_2[r * SAMPLES * number_of_inputs +
                                            s * number_of_inputs + i];
        uint64_t decrypted = 0;
        concrete_cpu_decrypt_lwe_ciphertext_u64(
            lwe_sk, lwe_out_ct + i * (lwe_dimension + 1), lwe_dimension,
            &decrypted);
        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted & rounding_bit) << 1;
        uint64_t decoded = (decrypted + rounding) / delta;
        EXPECT_EQ(decoded, (plaintext_1 + plaintext_2) / delta)
            << "Repetition: " << r << ", sample: " << s;
      }
    }
  }
}

TEST_P(LinearAlgebraTestPrimitives_u64, plaintext_addition) {
  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    uint64_t *lwe_sk = lwe_sk_array + (ptrdiff_t)(r * lwe_dimension);
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t *d_lwe_1_slice =
          d_lwe_in_1_ct +
          (ptrdiff_t)((r * SAMPLES * number_of_inputs + s * number_of_inputs) *
                      (lwe_dimension + 1));
      uint64_t *d_plaintext_2_in =
          d_plaintext_2 +
          (ptrdiff_t)((r * SAMPLES * number_of_inputs + s * number_of_inputs));
      // Execute addition
      cuda_add_lwe_ciphertext_vector_plaintext_vector_64(
          stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_1_slice,
          (void *)d_plaintext_2_in, lwe_dimension, number_of_inputs);
      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                               number_of_inputs * (lwe_dimension + 1) *
                                   sizeof(uint64_t),
                               stream, gpu_index);
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext_1 = plaintexts_1[r * SAMPLES * number_of_inputs +
                                            s * number_of_inputs + i];
        uint64_t plaintext_2 = plaintexts_2[r * SAMPLES * number_of_inputs +
                                            s * number_of_inputs + i];
        uint64_t decrypted = 0;
        concrete_cpu_decrypt_lwe_ciphertext_u64(
            lwe_sk, lwe_out_ct + i * (lwe_dimension + 1), lwe_dimension,
            &decrypted);
        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted & rounding_bit) << 1;
        uint64_t decoded = (decrypted + rounding) / delta;
        EXPECT_EQ(decoded, (plaintext_1 + plaintext_2) / delta)
            << "Repetition: " << r << ", sample: " << s << " i: " << i << ") "
            << plaintext_1 / delta << " + " << plaintext_2 / delta;
      }
    }
  }
}

TEST_P(LinearAlgebraTestPrimitives_u64, cleartext_multiplication) {
  void *v_stream = (void *)stream;
  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    uint64_t *lwe_sk = lwe_sk_array + (ptrdiff_t)(r * lwe_dimension);
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t *d_lwe_1_slice =
          d_lwe_in_1_ct +
          (ptrdiff_t)((r * SAMPLES * number_of_inputs + s * number_of_inputs) *
                      (lwe_dimension + 1));
      uint64_t *d_cleartext_in =
          d_cleartext +
          (ptrdiff_t)((r * SAMPLES * number_of_inputs + s * number_of_inputs));
      // Execute cleartext multiplication
      cuda_mult_lwe_ciphertext_vector_cleartext_vector_64(
          stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_1_slice,
          (void *)d_cleartext_in, lwe_dimension, number_of_inputs);
      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                               number_of_inputs * (lwe_dimension + 1) *
                                   sizeof(uint64_t),
                               stream, gpu_index);
      cuda_synchronize_stream(v_stream);
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t cleartext_1 = plaintexts_1[r * SAMPLES * number_of_inputs +
                                            s * number_of_inputs + i] /
                               delta;
        uint64_t cleartext_2 = plaintexts_2[r * SAMPLES * number_of_inputs +
                                            s * number_of_inputs + i] /
                               delta;
        uint64_t decrypted = 0;
        concrete_cpu_decrypt_lwe_ciphertext_u64(
            lwe_sk, lwe_out_ct + i * (lwe_dimension + 1), lwe_dimension,
            &decrypted);
        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted & rounding_bit) << 1;
        uint64_t decoded = (decrypted + rounding) / delta;
        EXPECT_EQ(decoded, cleartext_1 * cleartext_2)
            << "Repetition: " << r << ", sample: " << s << " i: " << i
            << ", decrypted: " << decrypted;
      }
    }
  }
}

TEST_P(LinearAlgebraTestPrimitives_u64, negate) {
  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    uint64_t *lwe_sk = lwe_sk_array + (ptrdiff_t)(r * lwe_dimension);
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t *d_lwe_1_slice =
          d_lwe_in_1_ct +
          (ptrdiff_t)((r * SAMPLES * number_of_inputs + s * number_of_inputs) *
                      (lwe_dimension + 1));
      // Execute negate
      cuda_negate_lwe_ciphertext_vector_64(
          stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_1_slice,
          lwe_dimension, number_of_inputs);

      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                               number_of_inputs * (lwe_dimension + 1) *
                                   sizeof(uint64_t),
                               stream, gpu_index);
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext = plaintexts_1[r * SAMPLES * number_of_inputs +
                                          s * number_of_inputs + i];
        uint64_t decrypted = 0;
        concrete_cpu_decrypt_lwe_ciphertext_u64(
            lwe_sk, lwe_out_ct + i * (lwe_dimension + 1), lwe_dimension,
            &decrypted);
        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted & rounding_bit) << 1;
        uint64_t decoded = (decrypted + rounding) / delta;
        EXPECT_EQ(decoded, -plaintext / delta)
            << "Repetition: " << r << ", sample: " << s << " i: " << i;
      }
    }
  }
}

// Defines for which parameters set the linear algebra operations will be
// tested. It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<LinearAlgebraTestParams>
    linear_algebra_params_u64 = ::testing::Values(
        // n, lwe_std_dev, message_modulus, carry_modulus, number_of_inputs
        (LinearAlgebraTestParams){600, 7.52316384526264e-37, 2, 2, 10});

std::string
printParamName(::testing::TestParamInfo<LinearAlgebraTestParams> p) {
  LinearAlgebraTestParams params = p.param;

  return "n_" + std::to_string(params.lwe_dimension);
}

INSTANTIATE_TEST_CASE_P(LinearAlgebraInstantiation,
                        LinearAlgebraTestPrimitives_u64,
                        linear_algebra_params_u64, printParamName);
