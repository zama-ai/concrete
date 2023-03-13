#include <cstdint>
#include <gtest/gtest.h>
#include <setup_and_teardown.h>
#include <stdio.h>
#include <stdlib.h>

const unsigned REPETITIONS = 2;
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
  int cbs_delta_log;
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
    input_lwe_dimension = glwe_dimension * polynomial_size;

    wop_pbs_setup(
        stream, &csprng, &lwe_sk_in_array, &lwe_sk_out_array, &d_ksk_array,
        &d_fourier_bsk_array, &d_pksk_array, &plaintexts, &d_lwe_ct_in_array,
        &d_lwe_ct_out_array, &d_lut_vector, &wop_pbs_buffer, lwe_dimension,
        glwe_dimension, polynomial_size, lwe_modular_variance,
        glwe_modular_variance, ks_base_log, ks_level, pksk_base_log, pksk_level,
        pbs_base_log, pbs_level, cbs_level, p, &delta_log, &cbs_delta_log,
        &delta_log_lut, &delta, tau, REPETITIONS, SAMPLES, gpu_index);
  }

  void TearDown() {
    wop_pbs_teardown(stream, csprng, lwe_sk_in_array, lwe_sk_out_array,
                     d_ksk_array, d_fourier_bsk_array, d_pksk_array, plaintexts,
                     d_lwe_ct_in_array, d_lut_vector, d_lwe_ct_out_array,
                     wop_pbs_buffer, gpu_index);
  }
};

TEST_P(WopBootstrapTestPrimitives_u64, wop_pbs) {
  void *v_stream = (void *)stream;
  uint64_t *lwe_out_ct_array =
      (uint64_t *)malloc((input_lwe_dimension + 1) * tau * sizeof(uint64_t));
  int bsk_size = (glwe_dimension + 1) * (glwe_dimension + 1) * pbs_level *
                 polynomial_size * (lwe_dimension + 1);
  int ksk_size =
      ks_level * (lwe_dimension + 1) * glwe_dimension * polynomial_size;
  int pksk_list_size = pksk_level * (glwe_dimension + 1) * polynomial_size *
                       (glwe_dimension * polynomial_size + 1) *
                       (glwe_dimension + 1);
  for (uint r = 0; r < REPETITIONS; r++) {
    double *d_fourier_bsk = d_fourier_bsk_array + (ptrdiff_t)(bsk_size * r);
    uint64_t *d_ksk = d_ksk_array + (ptrdiff_t)(ksk_size * r);
    uint64_t *d_pksk_list = d_pksk_array + (ptrdiff_t)(pksk_list_size * r);
    uint64_t *lwe_sk_in =
        lwe_sk_in_array + (ptrdiff_t)(input_lwe_dimension * r);
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t *d_lwe_ct_in =
          d_lwe_ct_in_array + (ptrdiff_t)((r * SAMPLES * tau + s * tau) *
                                          (input_lwe_dimension + 1));

      // Execute wop pbs
      cuda_wop_pbs_64(
          stream, gpu_index, (void *)d_lwe_ct_out_array, (void *)d_lwe_ct_in,
          (void *)d_lut_vector, (void *)d_fourier_bsk, (void *)d_ksk,
          (void *)d_pksk_list, wop_pbs_buffer, cbs_delta_log, glwe_dimension,
          lwe_dimension, polynomial_size, pbs_base_log, pbs_level, ks_base_log,
          ks_level, pksk_base_log, pksk_level, cbs_base_log, cbs_level, p, p,
          delta_log, tau, cuda_get_max_shared_memory(gpu_index));

      //// Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct_array, d_lwe_ct_out_array,
                               (input_lwe_dimension + 1) * tau *
                                   sizeof(uint64_t),
                               stream, gpu_index);
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
                                 7.52316384526264e-37, 4, 9, 1, 9, 4, 9, 6, 4,
                                 1}
        //        (WopBootstrapTestParams){481, 2, 512, 7.52316384526264e-37,
        //                                 7.52316384526264e-37, 4, 9, 1, 9, 4,
        //                                 9, 6, 4, 2} ,
        //        (WopBootstrapTestParams){481, 2, 1024, 7.52316384526264e-37,
        //                                                    7.52316384526264e-37,
        //                                                    4, 9, 1, 9, 4, 9,
        //                                                    6, 4, 1},
        //        (WopBootstrapTestParams){481, 2, 1024, 7.52316384526264e-37,
        //                                                    7.52316384526264e-37,
        //                                                    4, 9, 1, 9, 4, 9,
        //                                                    6, 4, 2}
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
