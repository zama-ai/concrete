#include <cstdint>
#include <gtest/gtest.h>
#include <setup_and_teardown.h>
#include <stdio.h>
#include <stdlib.h>

const unsigned REPETITIONS = 2;
const unsigned SAMPLES = 50;

typedef struct {
  int input_lwe_dimension;
  int output_lwe_dimension;
  double noise_variance;
  int ksk_base_log;
  int ksk_level;
  int message_modulus;
  int carry_modulus;
  int number_of_inputs;
} KeyswitchTestParams;

class KeyswitchTestPrimitives_u64
    : public ::testing::TestWithParam<KeyswitchTestParams> {
protected:
  int input_lwe_dimension;
  int output_lwe_dimension;
  double noise_variance;
  int ksk_base_log;
  int ksk_level;
  int message_modulus;
  int carry_modulus;
  int number_of_inputs;
  int payload_modulus;
  uint64_t delta;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;
  uint64_t *plaintexts;
  uint64_t *d_ksk_array;
  uint64_t *d_lwe_ct_out_array;
  uint64_t *d_lwe_ct_in_array;
  uint64_t *lwe_in_ct;
  uint64_t *lwe_out_ct;

public:
  // Test arithmetic functions
  void SetUp() {
    stream = cuda_create_stream(0);

    // TestParams
    input_lwe_dimension = (int)GetParam().input_lwe_dimension;
    output_lwe_dimension = (int)GetParam().output_lwe_dimension;
    noise_variance = (double)GetParam().noise_variance;
    ksk_base_log = (int)GetParam().ksk_base_log;
    ksk_level = (int)GetParam().ksk_level;
    message_modulus = (int)GetParam().message_modulus;
    carry_modulus = (int)GetParam().carry_modulus;
    number_of_inputs = (int)GetParam().number_of_inputs;

    keyswitch_setup(stream, &csprng, &lwe_sk_in_array, &lwe_sk_out_array,
                    &d_ksk_array, &plaintexts, &d_lwe_ct_in_array,
                    &d_lwe_ct_out_array, input_lwe_dimension,
                    output_lwe_dimension, noise_variance, ksk_base_log,
                    ksk_level, message_modulus, carry_modulus, &payload_modulus,
                    &delta, number_of_inputs, REPETITIONS, SAMPLES, gpu_index);
  }

  void TearDown() {
    keyswitch_teardown(stream, csprng, lwe_sk_in_array, lwe_sk_out_array,
                       d_ksk_array, plaintexts, d_lwe_ct_in_array,
                       d_lwe_ct_out_array, gpu_index);
  }
};

TEST_P(KeyswitchTestPrimitives_u64, keyswitch) {
  uint64_t *lwe_out_ct = (uint64_t *)malloc(
      (output_lwe_dimension + 1) * number_of_inputs * sizeof(uint64_t));
  for (uint r = 0; r < REPETITIONS; r++) {
    uint64_t *lwe_out_sk =
        lwe_sk_out_array + (ptrdiff_t)(r * output_lwe_dimension);
    int ksk_size = ksk_level * (output_lwe_dimension + 1) * input_lwe_dimension;
    uint64_t *d_ksk = d_ksk_array + (ptrdiff_t)(ksk_size * r);
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t *d_lwe_ct_in =
          d_lwe_ct_in_array +
          (ptrdiff_t)((r * SAMPLES * number_of_inputs + s * number_of_inputs) *
                      (input_lwe_dimension + 1));
      // Execute keyswitch
      cuda_keyswitch_lwe_ciphertext_vector_64(
          stream, gpu_index, (void *)d_lwe_ct_out_array, (void *)d_lwe_ct_in,
          (void *)d_ksk, input_lwe_dimension, output_lwe_dimension,
          ksk_base_log, ksk_level, number_of_inputs);

      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_ct_out_array,
                               number_of_inputs * (output_lwe_dimension + 1) *
                                   sizeof(uint64_t),
                               stream, gpu_index);
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext = plaintexts[r * SAMPLES * number_of_inputs +
                                        s * number_of_inputs + i];
        uint64_t decrypted = 0;
        concrete_cpu_decrypt_lwe_ciphertext_u64(
            lwe_out_sk, lwe_out_ct + i * (output_lwe_dimension + 1),
            output_lwe_dimension, &decrypted);
        EXPECT_NE(decrypted, plaintext);
        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted & rounding_bit) << 1;
        uint64_t decoded = (decrypted + rounding) / delta;
        EXPECT_EQ(decoded, plaintext / delta);
      }
    }
  }
  free(lwe_out_ct);
}

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<KeyswitchTestParams> ksk_params_u64 =
    ::testing::Values(
        // n, k*N, noise_variance, ks_base_log, ks_level,
        // message_modulus, carry_modulus, number_of_inputs
        (KeyswitchTestParams){567, 1280, 2.9802322387695312e-18, 3, 3, 2, 1,
                              10},
        (KeyswitchTestParams){694, 1536, 2.9802322387695312e-18, 4, 3, 2, 1,
                              10},
        (KeyswitchTestParams){769, 2048, 2.9802322387695312e-18, 4, 3, 2, 1,
                              10},
        (KeyswitchTestParams){754, 2048, 2.9802322387695312e-18, 3, 5, 2, 1,
                              10},
        (KeyswitchTestParams){847, 4096, 2.9802322387695312e-18, 4, 4, 2, 1,
                              10},
        (KeyswitchTestParams){881, 8192, 2.9802322387695312e-18, 3, 6, 2, 1,
                              10});

std::string printParamName(::testing::TestParamInfo<KeyswitchTestParams> p) {
  KeyswitchTestParams params = p.param;

  return "na_" + std::to_string(params.input_lwe_dimension) + "_nb_" +
         std::to_string(params.output_lwe_dimension) + "_baselog_" +
         std::to_string(params.ksk_base_log) + "_ksk_level_" +
         std::to_string(params.ksk_level);
}

INSTANTIATE_TEST_CASE_P(KeyswitchInstantiation, KeyswitchTestPrimitives_u64,
                        ksk_params_u64, printParamName);
