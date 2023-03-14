#include "../include/bootstrap.h"
#include "../include/device.h"
#include "concrete-cpu.h"
#include "utils.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

const unsigned REPETITIONS = 5;
const unsigned SAMPLES = 10;

typedef struct {
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance;
  double glwe_modular_variance;
  int pbs_base_log;
  int pbs_level;
  int ks_base_log;
  int ks_level;
  int pksk_base_log;
  int pksk_level;
  int cbs_base_log;
  int cbs_level;
  int tau;
} WopBootstrapTestParams;

class WopBootstrapTestPrimitives_u64
    : public ::testing::TestWithParam<WopBootstrapTestParams> {
protected:
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance;
  double glwe_modular_variance;
  int pbs_base_log;
  int pbs_level;
  int ks_base_log;
  int ks_level;
  int pksk_base_log;
  int pksk_level;
  int cbs_base_log;
  int cbs_level;
  int tau;
  int p;
  uint64_t delta;
  uint32_t cbs_delta_log;
  int delta_log;
  int delta_log_lut;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;
  uint64_t *lwe_in_ct_array;
  uint64_t *lwe_out_ct_array;
  uint64_t *plaintexts;
  double *d_fourier_bsk_array;
  uint64_t *d_ksk_array;
  uint64_t *d_pksk_array;
  uint64_t *d_lwe_ct_in_array;
  uint64_t *d_lwe_ct_out_array;
  uint64_t *d_lut_vector;
  int8_t *wop_pbs_buffer;
  int input_lwe_dimension;

public:
  // Test arithmetic functions
  void SetUp() {
    stream = cuda_create_stream(0);
    void *v_stream = (void *)stream;

    // TestParams
    lwe_dimension = (int)GetParam().lwe_dimension;
    glwe_dimension = (int)GetParam().glwe_dimension;
    polynomial_size = (int)GetParam().polynomial_size;
    lwe_modular_variance = (double)GetParam().lwe_modular_variance;
    glwe_modular_variance = (double)GetParam().glwe_modular_variance;
    pbs_base_log = (int)GetParam().pbs_base_log;
    pbs_level = (int)GetParam().pbs_level;
    ks_base_log = (int)GetParam().ks_base_log;
    ks_level = (int)GetParam().ks_level;
    pksk_base_log = (int)GetParam().pksk_base_log;
    pksk_level = (int)GetParam().pksk_level;
    cbs_base_log = (int)GetParam().cbs_base_log;
    cbs_level = (int)GetParam().cbs_level;
    tau = (int)GetParam().tau;
    p = 10 / tau;
    delta_log = 64 - p;
    delta_log_lut = delta_log;
    delta = (uint64_t)(1) << delta_log;

    // Create a Csprng
    csprng =
        (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
    uint8_t seed[16] = {(uint8_t)0};
    concrete_cpu_construct_concrete_csprng(
        csprng, Uint128{.little_endian_bytes = {*seed}});

    input_lwe_dimension = glwe_dimension * polynomial_size;
    // Generate the keys
    generate_lwe_secret_keys(&lwe_sk_in_array, input_lwe_dimension, csprng, REPETITIONS);
    generate_lwe_secret_keys(&lwe_sk_out_array, lwe_dimension, csprng, REPETITIONS);
    generate_lwe_keyswitch_keys(stream, gpu_index, &d_ksk_array,
                                lwe_sk_in_array, lwe_sk_out_array,
                                input_lwe_dimension, lwe_dimension, ks_level,
                                ks_base_log, csprng, lwe_modular_variance, REPETITIONS);
    generate_lwe_bootstrap_keys(
        stream, gpu_index, &d_fourier_bsk_array, lwe_sk_out_array,
        lwe_sk_in_array, lwe_dimension, glwe_dimension, polynomial_size,
        pbs_level, pbs_base_log, csprng, glwe_modular_variance, REPETITIONS);
    generate_lwe_private_functional_keyswitch_key_lists(
        stream, gpu_index, &d_pksk_array, lwe_sk_in_array, lwe_sk_in_array,
        input_lwe_dimension, glwe_dimension, polynomial_size, pksk_level,
        pksk_base_log, csprng, lwe_modular_variance, REPETITIONS);
    plaintexts = generate_plaintexts(p, delta, tau, REPETITIONS, SAMPLES);

    // LUT creation
    int lut_size = polynomial_size;
    int lut_num = tau << (tau * p - (int)log2(polynomial_size)); // r

    uint64_t *big_lut =
        (uint64_t *)malloc(lut_num * lut_size * sizeof(uint64_t));
    for (int t = tau - 1; t >= 0; t--) {
      uint64_t *small_lut = big_lut + (ptrdiff_t)(t * (1 << (tau * p)));
      for (uint64_t value = 0; value < (uint64_t)(1 << (tau * p)); value++) {
        int nbits = t * p;
        uint64_t x = (value >> nbits) & (uint64_t)((1 << p) - 1);
        small_lut[value] =
            ((x % (uint64_t)(1 << (64 - delta_log))) << delta_log_lut);
      }
    }
    d_lut_vector = (uint64_t *)cuda_malloc_async(
        lut_num * lut_size * sizeof(uint64_t), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_lut_vector, big_lut,
                             lut_num * lut_size * sizeof(uint64_t), stream,
                             gpu_index);
    // Execute scratch
    scratch_cuda_wop_pbs_64(stream, gpu_index, &wop_pbs_buffer,
                            (uint32_t *)&delta_log, &cbs_delta_log,
                            glwe_dimension, lwe_dimension, polynomial_size,
                            cbs_level, pbs_level, p, p, tau,
                            cuda_get_max_shared_memory(gpu_index), true);
    // Allocate input
    d_lwe_ct_in_array = (uint64_t *)cuda_malloc_async(
        (input_lwe_dimension + 1) * tau * sizeof(uint64_t), stream, gpu_index);
    // Allocate output
    d_lwe_ct_out_array = (uint64_t *)cuda_malloc_async(
        (input_lwe_dimension + 1) * tau * sizeof(uint64_t), stream, gpu_index);
    lwe_in_ct_array =
        (uint64_t *)malloc((input_lwe_dimension + 1) * tau * sizeof(uint64_t));
    lwe_out_ct_array =
        (uint64_t *)malloc((input_lwe_dimension + 1) * tau * sizeof(uint64_t));

    cuda_synchronize_stream(v_stream);
    free(big_lut);
  }

  void TearDown() {
    void *v_stream = (void *)stream;

    cuda_synchronize_stream(v_stream);
    concrete_cpu_destroy_concrete_csprng(csprng);
    free(csprng);
    free(lwe_sk_in_array);
    free(lwe_sk_out_array);
    free(plaintexts);
    free(lwe_in_ct_array);
    free(lwe_out_ct_array);
    cleanup_cuda_circuit_bootstrap_vertical_packing(stream, gpu_index,
                                                    &wop_pbs_buffer);
    cuda_drop_async(d_fourier_bsk_array, stream, gpu_index);
    cuda_drop_async(d_ksk_array, stream, gpu_index);
    cuda_drop_async(d_pksk_array, stream, gpu_index);
    cuda_drop_async(d_lwe_ct_in_array, stream, gpu_index);
    cuda_drop_async(d_lwe_ct_out_array, stream, gpu_index);
    cuda_drop_async(d_lut_vector, stream, gpu_index);
    cuda_destroy_stream(stream, gpu_index);
  }
};

TEST_P(WopBootstrapTestPrimitives_u64, wop_pbs) {
  void *v_stream = (void *)stream;
  int input_lwe_dimension = glwe_dimension * polynomial_size;
  int bsk_size = (glwe_dimension + 1) * (glwe_dimension + 1) * pbs_level *
                 polynomial_size * (lwe_dimension + 1);
  int ksk_size =
      ks_level * (lwe_dimension + 1) * glwe_dimension * polynomial_size;
  int pksk_list_size = pksk_level * (glwe_dimension + 1) * polynomial_size *
                       (glwe_dimension * polynomial_size + 1) *
                       (glwe_dimension + 1);
  for (uint r = 0; r < REPETITIONS; r++) {
    double *d_fourier_bsk = d_fourier_bsk_array + (ptrdiff_t)(bsk_size * r);
    uint64_t *lwe_sk_in =
        lwe_sk_in_array + (ptrdiff_t)(r * glwe_dimension * polynomial_size);
    uint64_t *d_ksk = d_ksk_array + (ptrdiff_t)(ksk_size * r);
    uint64_t *d_pksk_list = d_pksk_array + (ptrdiff_t)(pksk_list_size * r);
    for (uint s = 0; s < SAMPLES; s++) {
      for (int t = 0; t < tau; t++) {
        uint64_t plaintext = plaintexts[r * SAMPLES * tau + s * tau + t];
        uint64_t *lwe_in_ct =
            lwe_in_ct_array + (ptrdiff_t)(t * (input_lwe_dimension + 1));
        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_sk_in, lwe_in_ct, plaintext, input_lwe_dimension,
            lwe_modular_variance, csprng, &CONCRETE_CSPRNG_VTABLE);
      }
      cuda_memcpy_async_to_gpu(d_lwe_ct_in_array, lwe_in_ct_array,
                               (input_lwe_dimension + 1) * tau *
                                   sizeof(uint64_t),
                               stream, gpu_index);

      // Execute wop pbs
      cuda_wop_pbs_64(stream, gpu_index, (void *)d_lwe_ct_out_array,
                      (void *)d_lwe_ct_in_array, (void *)d_lut_vector,
                      (void *)d_fourier_bsk, (void *)d_ksk, (void *)d_pksk_list,
                      wop_pbs_buffer, cbs_delta_log, glwe_dimension,
                      lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
                      ks_base_log, ks_level, pksk_base_log, pksk_level,
                      cbs_base_log, cbs_level, p, p, delta_log, tau,
                      cuda_get_max_shared_memory(gpu_index));

      //// Copy result back
       cuda_memcpy_async_to_cpu(lwe_out_ct_array, d_lwe_ct_out_array,
      (input_lwe_dimension + 1) * tau * sizeof(uint64_t), stream, gpu_index);
       cuda_synchronize_stream(v_stream);

       for (int i = 0; i < tau; i++) {
        uint64_t plaintext = plaintexts[r * SAMPLES * tau + s * tau + i];
        uint64_t *result_ct =
            lwe_out_ct_array + (ptrdiff_t)(i * (input_lwe_dimension + 1));
        uint64_t decrypted_message = 0;
        concrete_cpu_decrypt_lwe_ciphertext_u64(
            lwe_sk_in, result_ct, input_lwe_dimension, &decrypted_message);
        // Round after decryption
        uint64_t decrypted =
            closest_representable(decrypted_message, 1, p) >> delta_log;
        uint64_t expected = plaintext >> delta_log;
        EXPECT_EQ(decrypted, expected);
      }
    }
  }
}

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<WopBootstrapTestParams> wop_pbs_params_u64 =
    ::testing::Values(
        // n, k, N, lwe_variance, glwe_variance, pbs_base_log, pbs_level,
        // ks_base_log, ks_level, tau
        (WopBootstrapTestParams){481, 2, 512, 7.52316384526264e-37,
                                                    7.52316384526264e-37, 4,
                                                    9, 1, 9, 4, 9, 6, 4, 1}
//        (WopBootstrapTestParams){481, 2, 512, 7.52316384526264e-37,
//                                 7.52316384526264e-37, 4, 9, 1, 9, 4, 9, 6, 4,
//                                 2} ,
//        (WopBootstrapTestParams){481, 2, 1024, 7.52316384526264e-37,
//                                                    7.52316384526264e-37, 4,
//                                                    9, 1, 9, 4, 9, 6, 4, 1},
//        (WopBootstrapTestParams){481, 2, 1024, 7.52316384526264e-37,
//                                                    7.52316384526264e-37, 4,
//                                                    9, 1, 9, 4, 9, 6, 4, 2}
    );

std::string printParamName(::testing::TestParamInfo<WopBootstrapTestParams> p) {
  WopBootstrapTestParams params = p.param;

  std::string message = "Unknown_parameter_set";
  if (params.polynomial_size == 512) {
    // When log_2_poly_size == 9 we have a cmux tree done with a single cmux.
    message = "wop_pbs_cmux_tree_with_single_cmux_n_" +
              std::to_string(params.lwe_dimension) + "_k_" +
              std::to_string(params.glwe_dimension) + "_N_" +
              std::to_string(params.polynomial_size) + "_tau_" +
              std::to_string(params.tau);
  } else if (params.polynomial_size == 1024) {
    // When log_2_poly_size == 10 the VP skips the cmux tree.
    message = "wop_pbs_without_cmux_tree_n_" +
              std::to_string(params.lwe_dimension) + "_k_" +
              std::to_string(params.glwe_dimension) + "_N_" +
              std::to_string(params.polynomial_size) + "_tau_" +
              std::to_string(params.tau);
  }
  return message;
}

INSTANTIATE_TEST_CASE_P(WopBootstrapInstantiation,
                        WopBootstrapTestPrimitives_u64, wop_pbs_params_u64,
                        printParamName);
