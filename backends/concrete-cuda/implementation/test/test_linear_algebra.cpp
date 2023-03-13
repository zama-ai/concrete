#include "../include/device.h"
#include "../include/linear_algebra.h"
#include "concrete-cpu.h"
#include "utils.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <functional>
#include <stdio.h>
#include <stdlib.h>

const unsigned REPETITIONS = 5;
const unsigned SAMPLES = 100;

typedef struct {
  int lwe_dimension;
  double noise_variance;
  int message_modulus;
  int carry_modulus;
} LinearAlgebraTestParams;

class LinearAlgebraTestPrimitives_u64
    : public ::testing::TestWithParam<LinearAlgebraTestParams> {
protected:
  int lwe_dimension;
  double noise_variance;
  int message_modulus;
  int carry_modulus;
  int payload_modulus;
  uint64_t delta;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *lwe_sk_array;
  uint64_t *d_lwe_in_1_ct;
  uint64_t *d_lwe_in_2_ct;
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
    void *v_stream = (void *)stream;

    // TestParams
    lwe_dimension = (int)GetParam().lwe_dimension;
    noise_variance = (int)GetParam().noise_variance;
    message_modulus = (int)GetParam().message_modulus;
    carry_modulus = (int)GetParam().carry_modulus;

    payload_modulus = message_modulus * carry_modulus;
    // Value of the shift we multiply our messages by
    delta = ((uint64_t)(1) << 63) / (uint64_t)(payload_modulus);

    // Create a Csprng
    csprng =
        (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
    uint8_t seed[16] = {(uint8_t)0};
    concrete_cpu_construct_concrete_csprng(
        csprng, Uint128{.little_endian_bytes = {*seed}});

    // Generate the keys
    generate_lwe_secret_keys(&lwe_sk_array, lwe_dimension, csprng, REPETITIONS);
    plaintexts_1 = generate_plaintexts(payload_modulus, delta, 1, REPETITIONS, SAMPLES);
    plaintexts_2 = generate_plaintexts(payload_modulus, delta, 1, REPETITIONS, SAMPLES);

    d_lwe_in_1_ct = (uint64_t *)cuda_malloc_async(
        (lwe_dimension + 1) * sizeof(uint64_t), stream, gpu_index);
    d_lwe_in_2_ct = (uint64_t *)cuda_malloc_async(
        (lwe_dimension + 1) * sizeof(uint64_t), stream, gpu_index);
    d_lwe_out_ct = (uint64_t *)cuda_malloc_async(
        (lwe_dimension + 1) * sizeof(uint64_t), stream, gpu_index);

    lwe_in_1_ct = (uint64_t *)malloc((lwe_dimension + 1) * sizeof(uint64_t));
    lwe_in_2_ct = (uint64_t *)malloc((lwe_dimension + 1) * sizeof(uint64_t));
    lwe_out_ct = (uint64_t *)malloc((lwe_dimension + 1) * sizeof(uint64_t));

    cuda_synchronize_stream(v_stream);
  }

  void TearDown() {
    void *v_stream = (void *)stream;

    cuda_synchronize_stream(v_stream);
    concrete_cpu_destroy_concrete_csprng(csprng);
    free(csprng);
    cuda_drop_async(d_lwe_in_1_ct, stream, gpu_index);
    cuda_drop_async(d_lwe_in_2_ct, stream, gpu_index);
    cuda_drop_async(d_lwe_out_ct, stream, gpu_index);
    free(lwe_in_1_ct);
    free(lwe_in_2_ct);
    free(lwe_out_ct);
    free(lwe_sk_array);
    free(plaintexts_1);
    free(plaintexts_2);
  }
};

TEST_P(LinearAlgebraTestPrimitives_u64, addition) {
  void *v_stream = (void *)stream;
  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t plaintext_1 = plaintexts_1[r * SAMPLES + s];
      uint64_t plaintext_2 = plaintexts_2[r * SAMPLES + s];
      uint64_t *lwe_sk = lwe_sk_array + (ptrdiff_t)(r * lwe_dimension);
      concrete_cpu_encrypt_lwe_ciphertext_u64(lwe_sk, lwe_in_1_ct, plaintext_1,
                                              lwe_dimension, noise_variance,
                                              csprng, &CONCRETE_CSPRNG_VTABLE);
      concrete_cpu_encrypt_lwe_ciphertext_u64(lwe_sk, lwe_in_2_ct, plaintext_2,
                                              lwe_dimension, noise_variance,
                                              csprng, &CONCRETE_CSPRNG_VTABLE);
      cuda_synchronize_stream(v_stream);
      cuda_memcpy_async_to_gpu(d_lwe_in_1_ct, lwe_in_1_ct,
                               (lwe_dimension + 1) * sizeof(uint64_t), stream,
                               gpu_index);
      cuda_memcpy_async_to_gpu(d_lwe_in_2_ct, lwe_in_2_ct,
                               (lwe_dimension + 1) * sizeof(uint64_t), stream,
                               gpu_index);
      // Execute addition
      cuda_add_lwe_ciphertext_vector_64(
          stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
          (void *)d_lwe_in_2_ct, lwe_dimension, 1);

      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                               (lwe_dimension + 1) * sizeof(uint64_t), stream,
                               gpu_index);
      uint64_t decrypted = 0;
      concrete_cpu_decrypt_lwe_ciphertext_u64(lwe_sk, lwe_out_ct, lwe_dimension,
                                              &decrypted);
      // The bit before the message
      uint64_t rounding_bit = delta >> 1;
      // Compute the rounding bit
      uint64_t rounding = (decrypted & rounding_bit) << 1;
      uint64_t decoded = (decrypted + rounding) / delta;
      ASSERT_EQ(decoded, (plaintext_1 + plaintext_2) / delta);
      cuda_synchronize_stream(v_stream);
    }
  }
}

TEST_P(LinearAlgebraTestPrimitives_u64, plaintext_addition) {
  void *v_stream = (void *)stream;
  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t plaintext_1 = plaintexts_1[r * SAMPLES + s];
      uint64_t plaintext_2 = plaintexts_2[r * SAMPLES + s];
      uint64_t *lwe_sk = lwe_sk_array + (ptrdiff_t)(r * lwe_dimension);
      concrete_cpu_encrypt_lwe_ciphertext_u64(lwe_sk, lwe_in_1_ct, plaintext_1,
                                              lwe_dimension, noise_variance,
                                              csprng, &CONCRETE_CSPRNG_VTABLE);
      cuda_synchronize_stream(v_stream);
      cuda_memcpy_async_to_gpu(d_lwe_in_1_ct, lwe_in_1_ct,
                               (lwe_dimension + 1) * sizeof(uint64_t), stream,
                               gpu_index);
      cuda_memcpy_async_to_gpu(d_lwe_in_2_ct, &plaintext_2, sizeof(uint64_t),
                               stream, gpu_index);
      // Execute addition
      cuda_add_lwe_ciphertext_vector_plaintext_vector_64(
          stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
          (void *)d_lwe_in_2_ct, lwe_dimension, 1);
      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                               (lwe_dimension + 1) * sizeof(uint64_t), stream,
                               gpu_index);
      uint64_t decrypted = 0;
      concrete_cpu_decrypt_lwe_ciphertext_u64(lwe_sk, lwe_out_ct, lwe_dimension,
                                              &decrypted);
      // The bit before the message
      uint64_t rounding_bit = delta >> 1;
      // Compute the rounding bit
      uint64_t rounding = (decrypted & rounding_bit) << 1;
      uint64_t decoded = (decrypted + rounding) / delta;
      ASSERT_EQ(decoded, (plaintext_1 + plaintext_2) / delta);
      cuda_synchronize_stream(v_stream);
    }
  }
}

TEST_P(LinearAlgebraTestPrimitives_u64, plaintext_multiplication) {
  void *v_stream = (void *)stream;
  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t plaintext_1 = plaintexts_1[r * SAMPLES + s];
      uint64_t plaintext_2 = plaintexts_2[r * SAMPLES + s];
      uint64_t *lwe_sk = lwe_sk_array + (ptrdiff_t)(r * lwe_dimension);
      concrete_cpu_encrypt_lwe_ciphertext_u64(lwe_sk, lwe_in_1_ct, plaintext_1,
                                              lwe_dimension, noise_variance,
                                              csprng, &CONCRETE_CSPRNG_VTABLE);
      cuda_synchronize_stream(v_stream);
      cuda_memcpy_async_to_gpu(d_lwe_in_1_ct, lwe_in_1_ct,
                               (lwe_dimension + 1) * sizeof(uint64_t), stream,
                               gpu_index);
      cuda_memcpy_async_to_gpu(d_lwe_in_2_ct, &plaintext_1, sizeof(uint64_t),
                               stream, gpu_index);
      // Execute addition
      cuda_mult_lwe_ciphertext_vector_cleartext_vector_64(
          stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
          (void *)d_lwe_in_2_ct, lwe_dimension, 1);
      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                               (lwe_dimension + 1) * sizeof(uint64_t), stream,
                               gpu_index);
      uint64_t decrypted = 0;
      concrete_cpu_decrypt_lwe_ciphertext_u64(lwe_sk, lwe_out_ct, lwe_dimension,
                                              &decrypted);
      // The bit before the message
      uint64_t rounding_bit = delta >> 1;
      // Compute the rounding bit
      uint64_t rounding = (decrypted & rounding_bit) << 1;
      uint64_t decoded = (decrypted + rounding) / delta;
      ASSERT_EQ(decoded, (plaintext_1 * plaintext_2) / delta);
      cuda_synchronize_stream(v_stream);
    }
  }
}

TEST_P(LinearAlgebraTestPrimitives_u64, negate) {
  void *v_stream = (void *)stream;
  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t plaintext = plaintexts_1[r * SAMPLES + s];
      uint64_t *lwe_sk = lwe_sk_array + (ptrdiff_t)(r * lwe_dimension);
      concrete_cpu_encrypt_lwe_ciphertext_u64(lwe_sk, lwe_in_1_ct, plaintext,
                                              lwe_dimension, noise_variance,
                                              csprng, &CONCRETE_CSPRNG_VTABLE);
      cuda_synchronize_stream(v_stream);
      cuda_memcpy_async_to_gpu(d_lwe_in_1_ct, lwe_in_1_ct,
                               (lwe_dimension + 1) * sizeof(uint64_t), stream,
                               gpu_index);
      // Execute addition
      cuda_negate_lwe_ciphertext_vector_64(
          stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
          lwe_dimension, 1);

      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                               (lwe_dimension + 1) * sizeof(uint64_t), stream,
                               gpu_index);
      uint64_t decrypted = 0;
      concrete_cpu_decrypt_lwe_ciphertext_u64(lwe_sk, lwe_out_ct, lwe_dimension,
                                              &decrypted);
      // The bit before the message
      uint64_t rounding_bit = delta >> 1;
      // Compute the rounding bit
      uint64_t rounding = (decrypted & rounding_bit) << 1;
      uint64_t decoded = (decrypted + rounding) / delta;
      ASSERT_EQ(decoded, -plaintext / delta);
      cuda_synchronize_stream(v_stream);
    }
  }
}

// Defines for which parameters set the linear algebra operations will be
// tested. It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<LinearAlgebraTestParams>
    linear_algebra_params_u64 = ::testing::Values(
        // n, lwe_std_dev, message_modulus, carry_modulus
        (LinearAlgebraTestParams){600, 0.000007069849454709433, 4, 4});

std::string
printParamName(::testing::TestParamInfo<LinearAlgebraTestParams> p) {
  LinearAlgebraTestParams params = p.param;

  return "n_" + std::to_string(params.lwe_dimension);
}

INSTANTIATE_TEST_CASE_P(LinearAlgebraInstantiation,
                        LinearAlgebraTestPrimitives_u64,
                        linear_algebra_params_u64, printParamName);