#include "../include/device.h"
#include "../include/keyswitch.h"
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
  int input_lwe_dimension;
  int output_lwe_dimension;
  double noise_variance;
  int ksk_base_log;
  int ksk_level;
  int message_modulus;
  int carry_modulus;
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
  int payload_modulus;
  uint64_t delta;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;
  uint64_t *plaintexts;
  uint64_t *d_ksk_array;
  uint64_t *d_lwe_out_ct;
  uint64_t *d_lwe_in_ct;
  uint64_t *lwe_in_ct;
  uint64_t *lwe_out_ct;
  int num_samples;

public:
  // Test arithmetic functions
  void SetUp() {
    stream = cuda_create_stream(0);
    void *v_stream = (void *)stream;

    // TestParams
    input_lwe_dimension = (int)GetParam().input_lwe_dimension;
    output_lwe_dimension = (int)GetParam().output_lwe_dimension;
    noise_variance = (int)GetParam().noise_variance;
    ksk_base_log = (int)GetParam().ksk_base_log;
    ksk_level = (int)GetParam().ksk_level;
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
    generate_lwe_secret_keys(&lwe_sk_in_array, input_lwe_dimension, csprng, REPETITIONS);
    generate_lwe_secret_keys(&lwe_sk_out_array, output_lwe_dimension, csprng, REPETITIONS);
    generate_lwe_keyswitch_keys(
        stream, gpu_index, &d_ksk_array, lwe_sk_in_array, lwe_sk_out_array,
        input_lwe_dimension, output_lwe_dimension, ksk_level, ksk_base_log,
        csprng, noise_variance, REPETITIONS);
    plaintexts = generate_plaintexts(payload_modulus, delta, 1, REPETITIONS, SAMPLES);

    d_lwe_out_ct = (uint64_t *)cuda_malloc_async(
        (output_lwe_dimension + 1) * sizeof(uint64_t), stream, gpu_index);

    d_lwe_in_ct = (uint64_t *)cuda_malloc_async(
        (input_lwe_dimension + 1) * sizeof(uint64_t), stream, gpu_index);

    lwe_in_ct =
        (uint64_t *)malloc((input_lwe_dimension + 1) * sizeof(uint64_t));
    lwe_out_ct =
        (uint64_t *)malloc((output_lwe_dimension + 1) * sizeof(uint64_t));

    cuda_synchronize_stream(v_stream);
  }

  void TearDown() {
    void *v_stream = (void *)stream;

    cuda_synchronize_stream(v_stream);
    concrete_cpu_destroy_concrete_csprng(csprng);
    free(csprng);
    cuda_drop_async(d_lwe_in_ct, stream, gpu_index);
    cuda_drop_async(d_lwe_out_ct, stream, gpu_index);
    free(lwe_in_ct);
    free(lwe_out_ct);
    free(lwe_sk_in_array);
    free(lwe_sk_out_array);
    free(plaintexts);
    cuda_drop_async(d_ksk_array, stream, gpu_index);
    cuda_destroy_stream(stream, gpu_index);
  }
};

TEST_P(KeyswitchTestPrimitives_u64, keyswitch) {
  void *v_stream = (void *)stream;
  for (uint r = 0; r < REPETITIONS; r++) {
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t plaintext = plaintexts[r * SAMPLES + s];
      uint64_t *lwe_in_sk =
          lwe_sk_in_array + (ptrdiff_t)(r * input_lwe_dimension);
      uint64_t *lwe_out_sk =
          lwe_sk_out_array + (ptrdiff_t)(r * output_lwe_dimension);
      int ksk_size =
          ksk_level * (output_lwe_dimension + 1) * input_lwe_dimension;
      uint64_t *d_ksk = d_ksk_array + (ptrdiff_t)(ksk_size * r);
      concrete_cpu_encrypt_lwe_ciphertext_u64(
          lwe_in_sk, lwe_in_ct, plaintext, input_lwe_dimension, noise_variance,
          csprng, &CONCRETE_CSPRNG_VTABLE);
      cuda_synchronize_stream(v_stream);
      cuda_memcpy_async_to_gpu(d_lwe_in_ct, lwe_in_ct,
                               (input_lwe_dimension + 1) * sizeof(uint64_t),
                               stream, gpu_index);
      // Execute keyswitch
      cuda_keyswitch_lwe_ciphertext_vector_64(
          stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_ct,
          (void *)d_ksk, input_lwe_dimension, output_lwe_dimension,
          ksk_base_log, ksk_level, 1);

      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                               (output_lwe_dimension + 1) * sizeof(uint64_t),
                               stream, gpu_index);
      uint64_t decrypted = 0;
      concrete_cpu_decrypt_lwe_ciphertext_u64(lwe_out_sk, lwe_out_ct,
                                              output_lwe_dimension, &decrypted);
      EXPECT_NE(decrypted, plaintext);
      // let err = (decrypted >= plaintext) ? decrypted - plaintext : plaintext
      // - decrypted;
      // error_sample_vec.push(err);

      // The bit before the message
      uint64_t rounding_bit = delta >> 1;
      // Compute the rounding bit
      uint64_t rounding = (decrypted & rounding_bit) << 1;
      uint64_t decoded = (decrypted + rounding) / delta;
      ASSERT_EQ(decoded, plaintext / delta);
    }
  }
}

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<KeyswitchTestParams> ksk_params_u64 =
    ::testing::Values(
        // n, k*N, noise_variance, ks_base_log, ks_level,
        // message_modulus, carry_modulus
        // 1 bit message 0 bit carry parameters
        (KeyswitchTestParams){567, 1280, 2.9802322387695312e-08, 3, 3, 2, 1},
        // 3 bits message 0 bit carry parameters
        (KeyswitchTestParams){694, 1536, 2.9802322387695312e-08, 4, 3, 4, 1},
        // 4 bits message 0 bit carry parameters
        (KeyswitchTestParams){769, 2048, 2.9802322387695312e-08, 4, 3, 5, 1},
        // 5 bits message 0 bit carry parameters
        (KeyswitchTestParams){754, 2048, 2.9802322387695312e-08, 3, 5, 6, 1},
        // 6 bits message 0 bit carry parameters
        (KeyswitchTestParams){847, 4096, 2.9802322387695312e-08, 4, 4, 7, 1},
        // 7 bits message 0 bit carry parameters
        (KeyswitchTestParams){881, 8192, 2.9802322387695312e-08, 3, 6, 8, 1});

std::string printParamName(::testing::TestParamInfo<KeyswitchTestParams> p) {
  KeyswitchTestParams params = p.param;

  return "na_" + std::to_string(params.input_lwe_dimension) + "_nb_" +
         std::to_string(params.output_lwe_dimension) + "_baselog_" +
         std::to_string(params.ksk_base_log) + "_ksk_level_" +
         std::to_string(params.ksk_level);
}

INSTANTIATE_TEST_CASE_P(KeyswitchInstantiation, KeyswitchTestPrimitives_u64,
                        ksk_params_u64, printParamName);
