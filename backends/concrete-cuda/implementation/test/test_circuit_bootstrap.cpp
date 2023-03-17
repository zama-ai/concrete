#include "../include/circuit_bootstrap.h"
#include "../include/device.h"
#include "concrete-cpu.h"
#include "utils.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

const unsigned REPETITIONS = 2;
const unsigned SAMPLES = 50;

typedef struct {
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance;
  double glwe_modular_variance;
  int pbs_base_log;
  int pbs_level;
  int pksk_base_log;
  int pksk_level;
  int cbs_base_log;
  int cbs_level;
  int number_of_inputs;
} CircuitBootstrapTestParams;

class CircuitBootstrapTestPrimitives_u64
    : public ::testing::TestWithParam<CircuitBootstrapTestParams> {
protected:
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance;
  double glwe_modular_variance;
  int pbs_base_log;
  int pbs_level;
  int pksk_base_log;
  int pksk_level;
  int cbs_base_log;
  int cbs_level;
  int number_of_inputs;
  int number_of_bits_of_message_including_padding;
  int ggsw_size;
  uint64_t delta;
  int delta_log;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;
  uint64_t *lwe_in_ct;
  uint64_t *ggsw_out_ct;
  uint64_t *plaintexts;
  double *d_fourier_bsk_array;
  uint64_t *d_pksk_array;
  uint64_t *d_lwe_in_ct;
  uint64_t *d_ggsw_out_ct;
  uint64_t *d_lut_vector_indexes;
  int8_t *cbs_buffer;

public:
  // Test arithmetic functions
  void SetUp() {
    stream = cuda_create_stream(0);

    // TestParams
    lwe_dimension = (int)GetParam().lwe_dimension;
    glwe_dimension = (int)GetParam().glwe_dimension;
    polynomial_size = (int)GetParam().polynomial_size;
    lwe_modular_variance = (double)GetParam().lwe_modular_variance;
    glwe_modular_variance = (double)GetParam().glwe_modular_variance;
    pbs_base_log = (int)GetParam().pbs_base_log;
    pbs_level = (int)GetParam().pbs_level;
    pksk_base_log = (int)GetParam().pksk_base_log;
    pksk_level = (int)GetParam().pksk_level;
    cbs_base_log = (int)GetParam().cbs_base_log;
    cbs_level = (int)GetParam().cbs_level;
    number_of_inputs = (int)GetParam().number_of_inputs;
    // We generate binary messages
    number_of_bits_of_message_including_padding = 2;
    delta_log = 60;
    delta = (uint64_t)(1) << delta_log;
    ggsw_size = cbs_level * (glwe_dimension + 1) * (glwe_dimension + 1) *
                polynomial_size;

    // Create a Csprng
    csprng =
        (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
    uint8_t seed[16] = {(uint8_t)0};
    concrete_cpu_construct_concrete_csprng(
        csprng, Uint128{.little_endian_bytes = {*seed}});

    // Generate the keys
    generate_lwe_secret_keys(&lwe_sk_in_array, lwe_dimension, csprng,
                             REPETITIONS);
    generate_lwe_secret_keys(&lwe_sk_out_array,
                             glwe_dimension * polynomial_size, csprng,
                             REPETITIONS);
    generate_lwe_bootstrap_keys(
        stream, gpu_index, &d_fourier_bsk_array, lwe_sk_in_array,
        lwe_sk_out_array, lwe_dimension, glwe_dimension, polynomial_size,
        pbs_level, pbs_base_log, csprng, glwe_modular_variance, REPETITIONS);
    generate_lwe_private_functional_keyswitch_key_lists(
        stream, gpu_index, &d_pksk_array, lwe_sk_out_array, lwe_sk_out_array,
        glwe_dimension * polynomial_size, glwe_dimension, polynomial_size,
        pksk_level, pksk_base_log, csprng, lwe_modular_variance, REPETITIONS);
    plaintexts =
        generate_plaintexts(number_of_bits_of_message_including_padding, delta,
                            number_of_inputs, REPETITIONS, SAMPLES);

    d_ggsw_out_ct = (uint64_t *)cuda_malloc_async(
        number_of_inputs * ggsw_size * sizeof(uint64_t), stream, gpu_index);

    d_lwe_in_ct = (uint64_t *)cuda_malloc_async(
        number_of_inputs * (lwe_dimension + 1) * sizeof(uint64_t), stream,
        gpu_index);

    lwe_in_ct = (uint64_t *)malloc(number_of_inputs * (lwe_dimension + 1) *
                                   sizeof(uint64_t));
    ggsw_out_ct = (uint64_t *)malloc(ggsw_size * sizeof(uint64_t));
    // Execute cbs scratch
    scratch_cuda_circuit_bootstrap_64(
        stream, gpu_index, &cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, cbs_level, number_of_inputs,
        cuda_get_max_shared_memory(gpu_index), true);
    // Build LUT vector indexes
    uint64_t *h_lut_vector_indexes =
        (uint64_t *)malloc(number_of_inputs * cbs_level * sizeof(uint64_t));
    for (int index = 0; index < cbs_level * number_of_inputs; index++) {
      h_lut_vector_indexes[index] = index % cbs_level;
    }
    d_lut_vector_indexes = (uint64_t *)cuda_malloc_async(
        number_of_inputs * cbs_level * sizeof(uint64_t), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_lut_vector_indexes, h_lut_vector_indexes,
                             number_of_inputs * cbs_level * sizeof(uint64_t),
                             stream, gpu_index);
    free(h_lut_vector_indexes);
  }

  void TearDown() {
    void *v_stream = (void *)stream;

    cuda_synchronize_stream(v_stream);
    concrete_cpu_destroy_concrete_csprng(csprng);
    free(csprng);
    free(lwe_sk_in_array);
    free(lwe_sk_out_array);
    free(plaintexts);
    free(lwe_in_ct);
    free(ggsw_out_ct);
    cleanup_cuda_circuit_bootstrap(stream, gpu_index, &cbs_buffer);
    cuda_drop_async(d_fourier_bsk_array, stream, gpu_index);
    cuda_drop_async(d_pksk_array, stream, gpu_index);
    cuda_drop_async(d_lwe_in_ct, stream, gpu_index);
    cuda_drop_async(d_ggsw_out_ct, stream, gpu_index);
    cuda_drop_async(d_lut_vector_indexes, stream, gpu_index);
    cuda_destroy_stream(stream, gpu_index);
  }
};

TEST_P(CircuitBootstrapTestPrimitives_u64, circuit_bootstrap) {
  void *v_stream = (void *)stream;
  for (uint r = 0; r < REPETITIONS; r++) {
    int bsk_size = (glwe_dimension + 1) * (glwe_dimension + 1) * pbs_level *
                   polynomial_size * (lwe_dimension + 1);
    double *d_fourier_bsk = d_fourier_bsk_array + (ptrdiff_t)(bsk_size * r);
    int pksk_list_size = pksk_level * (glwe_dimension + 1) * polynomial_size *
                         (glwe_dimension * polynomial_size + 1) *
                         (glwe_dimension + 1);
    uint64_t *d_pksk_list = d_pksk_array + (ptrdiff_t)(pksk_list_size * r);
    uint64_t *lwe_in_sk = lwe_sk_in_array + (ptrdiff_t)(lwe_dimension * r);
    uint64_t *lwe_sk_out =
        lwe_sk_out_array + (ptrdiff_t)(r * glwe_dimension * polynomial_size);
    for (uint s = 0; s < SAMPLES; s++) {
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext = plaintexts[r * SAMPLES * number_of_inputs +
                                        s * number_of_inputs + i];
        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_in_sk, lwe_in_ct + i * (lwe_dimension + 1), plaintext,
            lwe_dimension, lwe_modular_variance, csprng,
            &CONCRETE_CSPRNG_VTABLE);
      }
      cuda_synchronize_stream(v_stream);
      cuda_memcpy_async_to_gpu(d_lwe_in_ct, lwe_in_ct,
                               number_of_inputs * (lwe_dimension + 1) *
                                   sizeof(uint64_t),
                               stream, gpu_index);

      // Execute circuit bootstrap
      cuda_circuit_bootstrap_64(
          stream, gpu_index, (void *)d_ggsw_out_ct, (void *)d_lwe_in_ct,
          (void *)d_fourier_bsk, (void *)d_pksk_list,
          (void *)d_lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
          glwe_dimension, lwe_dimension, pbs_level, pbs_base_log, pksk_level,
          pksk_base_log, cbs_level, cbs_base_log, number_of_inputs,
          cuda_get_max_shared_memory(gpu_index));

      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext = plaintexts[r * SAMPLES * number_of_inputs +
                                        s * number_of_inputs + i];
        uint64_t *decrypted =
            (uint64_t *)malloc(polynomial_size * (glwe_dimension + 1) *
                               cbs_level * sizeof(uint64_t));
        // Copy result back
        cuda_memcpy_async_to_cpu(ggsw_out_ct, d_ggsw_out_ct + i * ggsw_size,
                                 ggsw_size * sizeof(uint64_t), stream,
                                 gpu_index);
        cuda_synchronize_stream(v_stream);

        uint64_t multiplying_factor = -(plaintext >> delta_log);
        for (int l = 1; l < cbs_level + 1; l++) {
          for (int j = 0; j < glwe_dimension; j++) {
            uint64_t *res = decrypted + (ptrdiff_t)((l - 1) * polynomial_size *
                                                        (glwe_dimension + 1) +
                                                    j * polynomial_size);
            uint64_t *glwe_ct_out =
                ggsw_out_ct +
                (ptrdiff_t)((l - 1) * polynomial_size * (glwe_dimension + 1) *
                                (glwe_dimension + 1) +
                            j * polynomial_size * (glwe_dimension + 1));
            concrete_cpu_decrypt_glwe_ciphertext_u64(
                lwe_sk_out, res, glwe_ct_out, glwe_dimension, polynomial_size);

            for (int k = 0; k < polynomial_size; k++) {
              uint64_t expected_decryption =
                  lwe_sk_out[j * polynomial_size + k] * multiplying_factor;
              expected_decryption >>= (64 - cbs_base_log * l);
              uint64_t decoded_plaintext =
                  closest_representable(res[k], l, cbs_base_log) >>
                  (64 - cbs_base_log * l);
              EXPECT_EQ(expected_decryption, decoded_plaintext);
            }
          }
        }
        // Check last glwe on last level
        uint64_t *res =
            decrypted + (ptrdiff_t)((cbs_level - 1) * polynomial_size *
                                        (glwe_dimension + 1) +
                                    glwe_dimension * polynomial_size);
        uint64_t *glwe_ct_out =
            ggsw_out_ct +
            (ptrdiff_t)((cbs_level - 1) * polynomial_size *
                            (glwe_dimension + 1) * (glwe_dimension + 1) +
                        glwe_dimension * polynomial_size *
                            (glwe_dimension + 1));
        concrete_cpu_decrypt_glwe_ciphertext_u64(
            lwe_sk_out, res, glwe_ct_out, glwe_dimension, polynomial_size);

        for (int k = 0; k < polynomial_size; k++) {
          uint64_t expected_decryption = (k == 0) ? plaintext / delta : 0;
          uint64_t decoded_plaintext =
              closest_representable(res[k], cbs_level, cbs_base_log) >>
              (64 - cbs_base_log * cbs_level);
          EXPECT_EQ(expected_decryption, decoded_plaintext);
        }
        free(decrypted);
      }
    }
  }
}

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<CircuitBootstrapTestParams> cbs_params_u64 =
    ::testing::Values(
        // n, k, N, lwe_variance, glwe_variance, pbs_base_log, pbs_level,
        // pksk_base_log, pksk_level, cbs_base_log, cbs_level, number_of_inputs
        (CircuitBootstrapTestParams){10, 2, 512, 7.52316384526264e-37,
                                     7.52316384526264e-37, 11, 2, 15, 2, 10, 1,
                                     10});

std::string
printParamName(::testing::TestParamInfo<CircuitBootstrapTestParams> p) {
  CircuitBootstrapTestParams params = p.param;

  return "n_" + std::to_string(params.lwe_dimension) + "_k_" +
         std::to_string(params.glwe_dimension) + "_N_" +
         std::to_string(params.polynomial_size) + "_pbs_base_log_" +
         std::to_string(params.pbs_base_log) + "_pbs_level_" +
         std::to_string(params.pbs_level) + "_pksk_base_log_" +
         std::to_string(params.pksk_base_log) + "_pksk_level_" +
         std::to_string(params.pksk_level) + "_cbs_base_log_" +
         std::to_string(params.cbs_base_log) + "_cbs_level_" +
         std::to_string(params.cbs_level);
}

INSTANTIATE_TEST_CASE_P(CircuitBootstrapInstantiation,
                        CircuitBootstrapTestPrimitives_u64, cbs_params_u64,
                        printParamName);
