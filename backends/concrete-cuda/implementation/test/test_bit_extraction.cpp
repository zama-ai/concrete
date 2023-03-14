#include "../include/bit_extraction.h"
#include "../include/device.h"
#include "concrete-cpu.h"
#include "utils.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

const unsigned REPETITIONS = 5;
const unsigned SAMPLES = 100;

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
  int number_of_bits_of_message_including_padding;
  int number_of_bits_to_extract;
  int number_of_inputs;
} BitExtractionTestParams;

class BitExtractionTestPrimitives_u64
    : public ::testing::TestWithParam<BitExtractionTestParams> {
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
  int number_of_bits_of_message_including_padding;
  int number_of_bits_to_extract;
  int number_of_inputs;
  uint64_t delta;
  int delta_log;
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
  uint64_t *d_lwe_in_ct_array;
  uint64_t *d_lwe_out_ct_array;
  int8_t *bit_extract_buffer;
  int input_lwe_dimension;
  int output_lwe_dimension;

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
    number_of_bits_of_message_including_padding =
        (int)GetParam().number_of_bits_of_message_including_padding;
    number_of_bits_to_extract = (int)GetParam().number_of_bits_to_extract;
    number_of_inputs = (int)GetParam().number_of_inputs;
    delta_log = 64 - number_of_bits_of_message_including_padding;
    delta = (uint64_t)(1) << delta_log;

    // Create a Csprng
    csprng =
        (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
    uint8_t seed[16] = {(uint8_t)0};
    concrete_cpu_construct_concrete_csprng(
        csprng, Uint128{.little_endian_bytes = {*seed}});

    input_lwe_dimension = glwe_dimension * polynomial_size;
    output_lwe_dimension = lwe_dimension;
    // Generate the keys
    generate_lwe_secret_keys(&lwe_sk_in_array, input_lwe_dimension, csprng, REPETITIONS);
    generate_lwe_secret_keys(&lwe_sk_out_array, output_lwe_dimension, csprng, REPETITIONS);
    generate_lwe_keyswitch_keys(
        stream, gpu_index, &d_ksk_array, lwe_sk_in_array, lwe_sk_out_array,
        input_lwe_dimension, output_lwe_dimension, ks_level, ks_base_log,
        csprng, lwe_modular_variance, REPETITIONS);
    generate_lwe_bootstrap_keys(
        stream, gpu_index, &d_fourier_bsk_array, lwe_sk_out_array,
        lwe_sk_in_array, output_lwe_dimension, glwe_dimension, polynomial_size,
        pbs_level, pbs_base_log, csprng, glwe_modular_variance, REPETITIONS);
    plaintexts = generate_plaintexts(
        number_of_bits_of_message_including_padding, delta, number_of_inputs, REPETITIONS, SAMPLES);

    d_lwe_out_ct_array = (uint64_t *)cuda_malloc_async(
        (output_lwe_dimension + 1) * number_of_bits_to_extract *
            number_of_inputs * sizeof(uint64_t),
        stream, gpu_index);

    d_lwe_in_ct_array = (uint64_t *)cuda_malloc_async(
        (input_lwe_dimension + 1) * number_of_inputs * sizeof(uint64_t), stream,
        gpu_index);

    lwe_in_ct_array = (uint64_t *)malloc((input_lwe_dimension + 1) *
                                         number_of_inputs * sizeof(uint64_t));
    lwe_out_ct_array = (uint64_t *)malloc((output_lwe_dimension + 1) *
                                          number_of_bits_to_extract *
                                          number_of_inputs * sizeof(uint64_t));
    // Execute scratch
    scratch_cuda_extract_bits_64(stream, gpu_index, &bit_extract_buffer,
                                 glwe_dimension, lwe_dimension, polynomial_size,
                                 pbs_level, number_of_inputs,
                                 cuda_get_max_shared_memory(gpu_index), true);
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
    cleanup_cuda_extract_bits(stream, gpu_index, &bit_extract_buffer);
    cuda_drop_async(d_fourier_bsk_array, stream, gpu_index);
    cuda_drop_async(d_ksk_array, stream, gpu_index);
    cuda_drop_async(d_lwe_in_ct_array, stream, gpu_index);
    cuda_drop_async(d_lwe_out_ct_array, stream, gpu_index);
    cuda_destroy_stream(stream, gpu_index);
  }
};

TEST_P(BitExtractionTestPrimitives_u64, bit_extraction) {
  void *v_stream = (void *)stream;
  int bsk_size = (glwe_dimension + 1) * (glwe_dimension + 1) * pbs_level *
                 polynomial_size * (output_lwe_dimension + 1);
  int ksk_size =
      ks_level * input_lwe_dimension * (output_lwe_dimension + 1);
  for (uint r = 0; r < REPETITIONS; r++) {
    double *d_fourier_bsk = d_fourier_bsk_array + (ptrdiff_t)(bsk_size * r);
    uint64_t *d_ksk = d_ksk_array + (ptrdiff_t)(ksk_size * r);
    uint64_t *lwe_in_sk =
        lwe_sk_in_array + (ptrdiff_t)(input_lwe_dimension * r);
    uint64_t *lwe_sk_out = lwe_sk_out_array + (ptrdiff_t)(r * output_lwe_dimension);
    for (uint s = 0; s < SAMPLES; s++) {
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext = plaintexts[r * SAMPLES * number_of_inputs +
                                        s * number_of_inputs + i];
        uint64_t *lwe_in_ct =
            lwe_in_ct_array +
            (ptrdiff_t)(
                i * (input_lwe_dimension + 1));
        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_in_sk, lwe_in_ct, plaintext, input_lwe_dimension,
            lwe_modular_variance, csprng, &CONCRETE_CSPRNG_VTABLE);
      }
      cuda_memcpy_async_to_gpu(d_lwe_in_ct_array, lwe_in_ct_array,
                               (input_lwe_dimension + 1) *
                                   number_of_inputs * sizeof(uint64_t),
                               stream, gpu_index);

      // Execute bit extract
      cuda_extract_bits_64(
          stream, gpu_index, (void *)d_lwe_out_ct_array,
          (void *)d_lwe_in_ct_array, bit_extract_buffer, (void *)d_ksk,
          (void *)d_fourier_bsk, number_of_bits_to_extract, delta_log,
          input_lwe_dimension, output_lwe_dimension, glwe_dimension,
          polynomial_size, pbs_base_log, pbs_level, ks_base_log, ks_level,
          number_of_inputs, cuda_get_max_shared_memory(gpu_index));

      // Copy result back
      cuda_memcpy_async_to_cpu(lwe_out_ct_array, d_lwe_out_ct_array,
                               (output_lwe_dimension + 1) * number_of_bits_to_extract *
                                   number_of_inputs * sizeof(uint64_t),
                               stream, gpu_index);
      cuda_synchronize_stream(v_stream);
      for (int j = 0; j < number_of_inputs; j++) {
        uint64_t *result_array =
            lwe_out_ct_array +
            (ptrdiff_t)(j * number_of_bits_to_extract * (output_lwe_dimension + 1));
        uint64_t plaintext = plaintexts[r * SAMPLES * number_of_inputs +
                                        s * number_of_inputs + j];
        for (int i = 0; i < number_of_bits_to_extract; i++) {
          uint64_t *result_ct =
              result_array + (ptrdiff_t)((number_of_bits_to_extract - 1 - i) *
                                         (output_lwe_dimension + 1));
          uint64_t decrypted_message = 0;
          concrete_cpu_decrypt_lwe_ciphertext_u64(
              lwe_sk_out, result_ct, output_lwe_dimension, &decrypted_message);
          // Round after decryption
          uint64_t decrypted_rounded =
              closest_representable(decrypted_message, 1, 1);
          // Bring back the extracted bit found in the MSB in the LSB
          uint64_t decrypted_extract_bit = decrypted_rounded >> 63;
          uint64_t expected = ((plaintext >> delta_log) >> i) & (uint64_t)(1);
          EXPECT_EQ(decrypted_extract_bit, expected);
        }
      }
    }
  }
}

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<BitExtractionTestParams>
    bit_extract_params_u64 = ::testing::Values(
        // n, k, N, lwe_variance, glwe_variance, pbs_base_log, pbs_level,
        // ks_base_log, ks_level, number_of_message_bits,
        // number_of_bits_to_extract, number_of_inputs
        (BitExtractionTestParams){585, 1, 1024, 7.52316384526264e-37,
                                  7.52316384526264e-37, 10, 2, 4, 7, 5, 5, 1},
        (BitExtractionTestParams){481, 1, 1024, 7.52316384526264e-37,
                                  7.52316384526264e-37, 4, 7, 1, 9, 5, 5, 1});

std::string
printParamName(::testing::TestParamInfo<BitExtractionTestParams> p) {
  BitExtractionTestParams params = p.param;

  return "n_" + std::to_string(params.lwe_dimension) + "_k_" +
         std::to_string(params.glwe_dimension) + "_N_" +
         std::to_string(params.polynomial_size) + "_pbs_base_log_" +
         std::to_string(params.pbs_base_log) + "_pbs_level_" +
         std::to_string(params.pbs_level) + "_ks_base_log_" +
         std::to_string(params.ks_base_log) + "_ks_level_" +
         std::to_string(params.ks_level) + "_number_of_message_bits_" +
         std::to_string(params.number_of_bits_of_message_including_padding) +
         "_number_of_bits_to_extract_" +
         std::to_string(params.number_of_bits_to_extract) +
         "_number_of_inputs_" + std::to_string(params.number_of_inputs);
}

INSTANTIATE_TEST_CASE_P(BitExtractionInstantiation,
                        BitExtractionTestPrimitives_u64, bit_extract_params_u64,
                        printParamName);