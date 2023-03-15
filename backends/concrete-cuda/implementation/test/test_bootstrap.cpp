#include "../include/bootstrap.h"
#include "../include/device.h"
#include "concrete-cpu.h"
#include "utils.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <functional>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance;
  double glwe_modular_variance;
  int pbs_base_log;
  int pbs_level;
  int message_modulus;
  int carry_modulus;
  int number_of_inputs;
  int repetitions;
  int samples;
} BootstrapTestParams;

class BootstrapTestPrimitives_u64
    : public ::testing::TestWithParam<BootstrapTestParams> {
protected:
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance;
  double glwe_modular_variance;
  int pbs_base_log;
  int pbs_level;
  int message_modulus;
  int carry_modulus;
  int payload_modulus;
  int number_of_inputs;
  int repetitions;
  int samples;
  uint64_t delta;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;
  uint64_t *plaintexts;
  double *d_fourier_bsk_array;
  uint64_t *d_lut_pbs_identity;
  uint64_t *d_lut_pbs_indexes;
  uint64_t *d_lwe_ct_in_array;
  uint64_t *d_lwe_ct_out_array;

public:
  // Test arithmetic functions
  void SetUp() {
    stream = cuda_create_stream(0);
    void *v_stream = (void *)stream;

    // TestParams
    lwe_dimension = (int)GetParam().lwe_dimension;
    glwe_dimension = (int)GetParam().glwe_dimension;
    polynomial_size = (int)GetParam().polynomial_size;
    lwe_modular_variance = (int)GetParam().lwe_modular_variance;
    glwe_modular_variance = (int)GetParam().glwe_modular_variance;
    pbs_base_log = (int)GetParam().pbs_base_log;
    pbs_level = (int)GetParam().pbs_level;
    message_modulus = (int)GetParam().message_modulus;
    carry_modulus = (int)GetParam().carry_modulus;
    number_of_inputs = (int)GetParam().number_of_inputs;
    repetitions = (int)GetParam().repetitions;
    samples = (int)GetParam().samples;

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
    generate_lwe_secret_keys(&lwe_sk_in_array, lwe_dimension, csprng,
                             repetitions);
    generate_lwe_secret_keys(&lwe_sk_out_array,
                             glwe_dimension * polynomial_size, csprng,
                             repetitions);
    generate_lwe_bootstrap_keys(
        stream, gpu_index, &d_fourier_bsk_array, lwe_sk_in_array,
        lwe_sk_out_array, lwe_dimension, glwe_dimension, polynomial_size,
        pbs_level, pbs_base_log, csprng, glwe_modular_variance, repetitions);
    plaintexts = generate_plaintexts(payload_modulus, delta, number_of_inputs,
                                     repetitions, samples);

    // Create the LUT
    uint64_t *lut_pbs_identity = generate_identity_lut_pbs(
        polynomial_size, glwe_dimension, message_modulus, carry_modulus,
        [](int x) -> int { return x; });

    // Copy the LUT
    d_lut_pbs_identity = (uint64_t *)cuda_malloc_async(
        (glwe_dimension + 1) * polynomial_size * sizeof(uint64_t), stream,
        gpu_index);
    d_lut_pbs_indexes = (uint64_t *)cuda_malloc_async(
        number_of_inputs * sizeof(uint64_t), stream, gpu_index);
    cuda_synchronize_stream(v_stream);
    cuda_memset_async(d_lut_pbs_indexes, 0, number_of_inputs * sizeof(uint64_t),
                      stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_lut_pbs_identity, lut_pbs_identity,
                             polynomial_size * (glwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(v_stream);
    free(lut_pbs_identity);

    d_lwe_ct_out_array =
        (uint64_t *)cuda_malloc_async((glwe_dimension * polynomial_size + 1) *
                                          number_of_inputs * sizeof(uint64_t),
                                      stream, gpu_index);
    d_lwe_ct_in_array = (uint64_t *)cuda_malloc_async(
        (lwe_dimension + 1) * number_of_inputs * repetitions * samples *
            sizeof(uint64_t),
        stream, gpu_index);
    uint64_t *lwe_ct_in_array =
        (uint64_t *)malloc((lwe_dimension + 1) * number_of_inputs *
                           repetitions * samples * sizeof(uint64_t));
    // Create the input/output ciphertexts
    for (int r = 0; r < repetitions; r++) {
      uint64_t *lwe_sk_in = lwe_sk_in_array + (ptrdiff_t)(r * lwe_dimension);
      for (int s = 0; s < samples; s++) {
        for (int i = 0; i < number_of_inputs; i++) {
          uint64_t plaintext = plaintexts[r * samples * number_of_inputs +
                                          s * number_of_inputs + i];
          uint64_t *lwe_ct_in =
              lwe_ct_in_array + (ptrdiff_t)((r * samples * number_of_inputs +
                                             s * number_of_inputs + i) *
                                            (lwe_dimension + 1));
          concrete_cpu_encrypt_lwe_ciphertext_u64(
              lwe_sk_in, lwe_ct_in, plaintext, lwe_dimension,
              lwe_modular_variance, csprng, &CONCRETE_CSPRNG_VTABLE);
        }
      }
    }
    cuda_synchronize_stream(v_stream);
    cuda_memcpy_async_to_gpu(d_lwe_ct_in_array, lwe_ct_in_array,
                             repetitions * samples * number_of_inputs *
                                 (lwe_dimension + 1) * sizeof(uint64_t),
                             stream, gpu_index);
    free(lwe_ct_in_array);
  }

  void TearDown() {
    concrete_cpu_destroy_concrete_csprng(csprng);
    free(csprng);
    free(lwe_sk_in_array);
    free(lwe_sk_out_array);
    free(plaintexts);
    cuda_drop_async(d_fourier_bsk_array, stream, gpu_index);
    cuda_drop_async(d_lut_pbs_identity, stream, gpu_index);
    cuda_drop_async(d_lut_pbs_indexes, stream, gpu_index);
    cuda_drop_async(d_lwe_ct_in_array, stream, gpu_index);
    cuda_drop_async(d_lwe_ct_out_array, stream, gpu_index);
    cuda_destroy_stream(stream, gpu_index);
  }
};

TEST_P(BootstrapTestPrimitives_u64, amortized_bootstrap) {
  uint64_t *lwe_ct_out_array =
      (uint64_t *)malloc((glwe_dimension * polynomial_size + 1) *
                         number_of_inputs * sizeof(uint64_t));
  int8_t *pbs_buffer = nullptr;
  scratch_cuda_bootstrap_amortized_64(
      stream, gpu_index, &pbs_buffer, glwe_dimension, polynomial_size,
      number_of_inputs, cuda_get_max_shared_memory(gpu_index), true);
  int bsk_size = (glwe_dimension + 1) * (glwe_dimension + 1) * pbs_level *
                 polynomial_size * (lwe_dimension + 1);
  // Here execute the PBS
  for (int r = 0; r < repetitions; r++) {
    double *d_fourier_bsk = d_fourier_bsk_array + (ptrdiff_t)(bsk_size * r);
    uint64_t *lwe_sk_out =
        lwe_sk_out_array + (ptrdiff_t)(r * glwe_dimension * polynomial_size);
    for (int s = 0; s < samples; s++) {
      uint64_t *d_lwe_ct_in =
          d_lwe_ct_in_array +
          (ptrdiff_t)((r * samples * number_of_inputs + s * number_of_inputs) *
                      (lwe_dimension + 1));
      // Execute PBS
      cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
          stream, gpu_index, (void *)d_lwe_ct_out_array,
          (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
          (void *)d_lwe_ct_in, (void *)d_fourier_bsk, pbs_buffer, lwe_dimension,
          glwe_dimension, polynomial_size, pbs_base_log, pbs_level,
          number_of_inputs, 1, 0, cuda_get_max_shared_memory(gpu_index));
      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_ct_out_array, d_lwe_ct_out_array,
                               (glwe_dimension * polynomial_size + 1) *
                                   number_of_inputs * sizeof(uint64_t),
                               stream, gpu_index);

      for (int j = 0; j < number_of_inputs; j++) {
        uint64_t *result =
            lwe_ct_out_array +
            (ptrdiff_t)(j * (glwe_dimension * polynomial_size + 1));
        uint64_t plaintext = plaintexts[r * samples * number_of_inputs +
                                        s * number_of_inputs + j];
        uint64_t decrypted = 0;
        concrete_cpu_decrypt_lwe_ciphertext_u64(
            lwe_sk_out, result, glwe_dimension * polynomial_size, &decrypted);
        EXPECT_NE(decrypted, plaintext);
        // let err = (decrypted >= plaintext) ? decrypted - plaintext :
        // plaintext
        // - decrypted;
        // error_sample_vec.push(err);

        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted & rounding_bit) << 1;
        uint64_t decoded = (decrypted + rounding) / delta;
        EXPECT_EQ(decoded, plaintext / delta);
      }
    }
  }
  cleanup_cuda_bootstrap_amortized(stream, gpu_index, &pbs_buffer);
  free(lwe_ct_out_array);
}

TEST_P(BootstrapTestPrimitives_u64, low_latency_bootstrap) {
  int number_of_sm = 0;
  cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, 0);
  if (number_of_inputs > number_of_sm * 4 / (glwe_dimension + 1) / pbs_level)
    GTEST_SKIP() << "The Low Latency PBS does not support this configuration";
  uint64_t *lwe_ct_out_array =
      (uint64_t *)malloc((glwe_dimension * polynomial_size + 1) *
                         number_of_inputs * sizeof(uint64_t));
  int8_t *pbs_buffer = nullptr;
  scratch_cuda_bootstrap_low_latency_64(
      stream, gpu_index, &pbs_buffer, glwe_dimension, polynomial_size,
      pbs_level, number_of_inputs, cuda_get_max_shared_memory(gpu_index), true);
  int bsk_size = (glwe_dimension + 1) * (glwe_dimension + 1) * pbs_level *
                 polynomial_size * (lwe_dimension + 1);
  // Here execute the PBS
  for (int r = 0; r < repetitions; r++) {
    double *d_fourier_bsk = d_fourier_bsk_array + (ptrdiff_t)(bsk_size * r);
    uint64_t *lwe_sk_out =
        lwe_sk_out_array + (ptrdiff_t)(r * glwe_dimension * polynomial_size);
    for (int s = 0; s < samples; s++) {
      uint64_t *d_lwe_ct_in =
          d_lwe_ct_in_array +
          (ptrdiff_t)((r * samples * number_of_inputs + s * number_of_inputs) *
                      (lwe_dimension + 1));
      // Execute PBS
      cuda_bootstrap_low_latency_lwe_ciphertext_vector_64(
          stream, gpu_index, (void *)d_lwe_ct_out_array,
          (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
          (void *)d_lwe_ct_in, (void *)d_fourier_bsk, pbs_buffer, lwe_dimension,
          glwe_dimension, polynomial_size, pbs_base_log, pbs_level,
          number_of_inputs, 1, 0, cuda_get_max_shared_memory(gpu_index));
      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_ct_out_array, d_lwe_ct_out_array,
                               (glwe_dimension * polynomial_size + 1) *
                                   number_of_inputs * sizeof(uint64_t),
                               stream, gpu_index);

      for (int j = 0; j < number_of_inputs; j++) {
        uint64_t *result =
            lwe_ct_out_array +
            (ptrdiff_t)(j * (glwe_dimension * polynomial_size + 1));
        uint64_t plaintext = plaintexts[r * samples * number_of_inputs +
                                        s * number_of_inputs + j];
        uint64_t decrypted = 0;
        concrete_cpu_decrypt_lwe_ciphertext_u64(
            lwe_sk_out, result, glwe_dimension * polynomial_size, &decrypted);
        EXPECT_NE(decrypted, plaintext);
        // let err = (decrypted >= plaintext) ? decrypted - plaintext :
        // plaintext
        // - decrypted;
        // error_sample_vec.push(err);

        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted & rounding_bit) << 1;
        uint64_t decoded = (decrypted + rounding) / delta;
        EXPECT_EQ(decoded, plaintext / delta);
      }
    }
  }
  cleanup_cuda_bootstrap_low_latency(stream, gpu_index, &pbs_buffer);
  free(lwe_ct_out_array);
}

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<BootstrapTestParams> pbs_params_u64 =
    ::testing::Values(
        // n, k, N, lwe_variance, glwe_variance, pbs_base_log, pbs_level,
        // message_modulus, carry_modulus, number_of_inputs, repetitions,
        // samples
        (BootstrapTestParams){567, 5, 256, 0.000007069849454709433,
                              0.00000000000000029403601535432533, 15, 1, 2, 1,
                              5, 2, 50},
        (BootstrapTestParams){623, 6, 256, 7.52316384526264e-37,
                              7.52316384526264e-37, 9, 3, 2, 2, 5, 2, 50},
        (BootstrapTestParams){694, 3, 512, 0.000007069849454709433,
                              0.00000000000000029403601535432533, 18, 1, 2, 1,
                              5, 2, 50},
        (BootstrapTestParams){769, 2, 1024, 0.000007069849454709433,
                              0.00000000000000029403601535432533, 23, 1, 2, 1,
                              5, 2, 50},
        (BootstrapTestParams){754, 1, 2048, 0.000007069849454709433,
                              0.00000000000000029403601535432533, 23, 1, 4, 1,
                              5, 2, 50},
        (BootstrapTestParams){847, 1, 4096, 0.000007069849454709433,
                              0.00000000000000029403601535432533, 2, 12, 2, 1,
                              2, 1, 50},
        (BootstrapTestParams){881, 1, 8192, 0.000007069849454709433,
                              0.00000000000000029403601535432533, 22, 1, 2, 1,
                              2, 1, 25});

std::string printParamName(::testing::TestParamInfo<BootstrapTestParams> p) {
  BootstrapTestParams params = p.param;

  return "n_" + std::to_string(params.lwe_dimension) + "_k_" +
         std::to_string(params.glwe_dimension) + "_N_" +
         std::to_string(params.polynomial_size) + "_pbs_base_log_" +
         std::to_string(params.pbs_base_log) + "_pbs_level_" +
         std::to_string(params.pbs_level) + "_number_of_inputs_" +
         std::to_string(params.number_of_inputs);
}

INSTANTIATE_TEST_CASE_P(BootstrapInstantiation, BootstrapTestPrimitives_u64,
                        pbs_params_u64, printParamName);
