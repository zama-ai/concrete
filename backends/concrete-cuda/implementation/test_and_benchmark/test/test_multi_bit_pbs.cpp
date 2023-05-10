#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <setup_and_teardown.h>
#include <utils.h>

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
  int grouping_factor;
  int repetitions;
  int samples;
} MultiBitPBSTestParams;

class MultiBitPBSTestPrimitives_u64
    : public ::testing::TestWithParam<MultiBitPBSTestParams> {
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
  int grouping_factor;
  uint64_t delta;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;
  uint64_t *plaintexts;
  uint64_t *d_bsk_array;
  uint64_t *d_lut_pbs_identity;
  uint64_t *d_lut_pbs_indexes;
  uint64_t *d_lwe_ct_in_array;
  uint64_t *d_lwe_ct_out_array;
  uint64_t *lwe_ct_out_array;
  int8_t *pbs_buffer;

public:
  // Test arithmetic functions
  void SetUp() {
    stream = cuda_create_stream(0);

    // TestParams
    lwe_dimension = (int)GetParam().lwe_dimension;
    glwe_dimension = (int)GetParam().glwe_dimension;
    polynomial_size = (int)GetParam().polynomial_size;
    grouping_factor = (int)GetParam().grouping_factor;
    lwe_modular_variance = (double)GetParam().lwe_modular_variance;
    glwe_modular_variance = (double)GetParam().glwe_modular_variance;
    pbs_base_log = (int)GetParam().pbs_base_log;
    pbs_level = (int)GetParam().pbs_level;
    message_modulus = (int)GetParam().message_modulus;
    carry_modulus = (int)GetParam().carry_modulus;
    number_of_inputs = (int)GetParam().number_of_inputs;
    repetitions = (int)GetParam().repetitions;
    samples = (int)GetParam().samples;

    multi_bit_pbs_setup(
        stream, &csprng, &lwe_sk_in_array, &lwe_sk_out_array, &d_bsk_array,
        &plaintexts, &d_lut_pbs_identity, &d_lut_pbs_indexes,
        &d_lwe_ct_in_array, &d_lwe_ct_out_array, &pbs_buffer, lwe_dimension,
        glwe_dimension, polynomial_size, grouping_factor, lwe_modular_variance,
        glwe_modular_variance, pbs_base_log, pbs_level, message_modulus,
        carry_modulus, &payload_modulus, &delta, number_of_inputs, repetitions,
        samples, gpu_index);

    lwe_ct_out_array =
        (uint64_t *)malloc((glwe_dimension * polynomial_size + 1) *
                           number_of_inputs * sizeof(uint64_t));
  }

  void TearDown() {
    free(lwe_ct_out_array);
    multi_bit_pbs_teardown(stream, csprng, lwe_sk_in_array, lwe_sk_out_array,
                           d_bsk_array, plaintexts, d_lut_pbs_identity,
                           d_lut_pbs_indexes, d_lwe_ct_in_array,
                           d_lwe_ct_out_array, &pbs_buffer, gpu_index);
  }
};

TEST_P(MultiBitPBSTestPrimitives_u64, multi_bit_pbs) {

  int bsk_size = (lwe_dimension / grouping_factor) * pbs_level *
                 (glwe_dimension + 1) * (glwe_dimension + 1) * polynomial_size *
                 (1 << grouping_factor);
  // Here execute the PBS
  for (int r = 0; r < repetitions; r++) {
    uint64_t *d_bsk = d_bsk_array + (ptrdiff_t)(bsk_size * r);
    uint64_t *lwe_sk_out =
        lwe_sk_out_array + (ptrdiff_t)(r * glwe_dimension * polynomial_size);
    for (int s = 0; s < samples; s++) {
      uint64_t *d_lwe_ct_in =
          d_lwe_ct_in_array +
          (ptrdiff_t)((r * samples * number_of_inputs + s * number_of_inputs) *
                      (lwe_dimension + 1));
      // Execute PBS
      cuda_multi_bit_pbs_lwe_ciphertext_vector_64(
          stream, gpu_index, (void *)d_lwe_ct_out_array,
          (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
          (void *)d_lwe_ct_in, (void *)d_bsk, pbs_buffer, lwe_dimension,
          glwe_dimension, polynomial_size, grouping_factor, pbs_base_log,
          pbs_level, number_of_inputs, 1, 0,
          cuda_get_max_shared_memory(gpu_index));
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
        EXPECT_NE(decrypted, plaintext)
            << "Repetition: " << r << ", sample: " << s << ", input: " << j;
        // let err = (decrypted >= plaintext) ? decrypted - plaintext :
        // plaintext
        // - decrypted;
        // error_sample_vec.push(err);

        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted & rounding_bit) << 1;
        uint64_t decoded = (decrypted + rounding) / delta;
        EXPECT_EQ(decoded, plaintext / delta)
            << "Repetition: " << r << ", sample: " << s << ", input: " << j;
      }
    }
  }
  // cleanup_cuda_multi_bit_pbs(stream, gpu_index, &pbs_buffer);
}

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<MultiBitPBSTestParams> multipbs_params_u64 =
    ::testing::Values(
        // fast test
        (MultiBitPBSTestParams){16, 1, 256, 1.3880686109937e-11,
                                1.1919984450689246e-23, 23, 1, 2, 2, 1, 2, 1,
                                2},
        (MultiBitPBSTestParams){16, 1, 256, 1.3880686109937e-11,
                                1.1919984450689246e-23, 23, 1, 2, 2, 128, 2, 1,
                                2},
        // 4_bits_multi_bit_group_2
        (MultiBitPBSTestParams){818, 1, 2048, 1.3880686109937e-11,
                                1.1919984450689246e-23, 22, 1, 2, 2, 1, 2, 1,
                                1},
        (MultiBitPBSTestParams){818, 1, 2048, 1.3880686109937e-15,
                                1.1919984450689246e-24, 22, 1, 2, 2, 128, 2, 1,
                                1},
        // 4_bits_multi_bit_group_3
        (MultiBitPBSTestParams){888, 1, 2048, 4.9571231961752025e-12,
                                9.9409770026944e-32, 21, 1, 2, 2, 1, 3, 1, 1},
        (MultiBitPBSTestParams){888, 1, 2048, 4.9571231961752025e-12,
                                9.9409770026944e-32, 21, 1, 2, 2, 128, 3, 1, 1},

        (MultiBitPBSTestParams){742, 1, 2048, 4.9571231961752025e-12,
                                9.9409770026944e-32, 23, 1, 2, 2, 128, 2, 1, 1},
        (MultiBitPBSTestParams){744, 1, 2048, 4.9571231961752025e-12,
                                9.9409770026944e-32, 23, 1, 2, 2, 1, 3, 1, 1},
        (MultiBitPBSTestParams){744, 1, 2048, 4.9571231961752025e-12,
                                9.9409770026944e-32, 23, 1, 2, 2, 5, 3, 1, 1},
        (MultiBitPBSTestParams){744, 1, 2048, 4.9571231961752025e-12,
                                9.9409770026944e-32, 23, 1, 2, 2, 128, 3, 1,
                                1});
std::string printParamName(::testing::TestParamInfo<MultiBitPBSTestParams> p) {
  MultiBitPBSTestParams params = p.param;

  return "n_" + std::to_string(params.lwe_dimension) + "_k_" +
         std::to_string(params.glwe_dimension) + "_N_" +
         std::to_string(params.polynomial_size) + "_pbs_base_log_" +
         std::to_string(params.pbs_base_log) + "_pbs_level_" +
         std::to_string(params.pbs_level) + "_grouping_factor_" +
         std::to_string(params.grouping_factor) + "_number_of_inputs_" +
         std::to_string(params.number_of_inputs);
}

INSTANTIATE_TEST_CASE_P(MultiBitPBSInstantiation, MultiBitPBSTestPrimitives_u64,
                        multipbs_params_u64, printParamName);
