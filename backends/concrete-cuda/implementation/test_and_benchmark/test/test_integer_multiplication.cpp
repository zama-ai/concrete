#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <setup_and_teardown.h>
#include <utils.h>
const bool USE_MULTI_GPU = false;

typedef struct {
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance;
  double glwe_modular_variance;
  int pbs_base_log;
  int pbs_level;
  int ksk_base_log;
  int ksk_level;
  int total_message_bits;
  int number_of_blocks;
  int message_modulus;
  int carry_modulus;
  int repetitions;
  int samples;
  PBS_TYPE pbs_type;
} IntegerMultiplicationTestParams;

class IntegerMultiplicationTestPrimitives_u64
    : public ::testing::TestWithParam<IntegerMultiplicationTestParams> {
protected:
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance;
  double glwe_modular_variance;
  int pbs_base_log;
  int pbs_level;
  int ksk_base_log;
  int ksk_level;
  int total_message_bits;
  int number_of_blocks;
  int message_modulus;
  int carry_modulus;
  int repetitions;
  int samples;
  PBS_TYPE pbs_type;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t delta;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;
  uint64_t *plaintexts_1;
  uint64_t *plaintexts_2;
  uint64_t *expected;
  void *d_bsk_array;
  uint64_t *d_ksk_array;
  uint64_t *d_lwe_ct_in_array_1;
  uint64_t *d_lwe_ct_in_array_2;
  uint64_t *d_lwe_ct_out_array;
  int_mul_memory<uint64_t> *mem_ptr;

public:
  // Test arithmetic functions
  void SetUp() {
    // TestParams
    lwe_dimension = (int)GetParam().lwe_dimension;
    glwe_dimension = (int)GetParam().glwe_dimension;
    polynomial_size = (int)GetParam().polynomial_size;
    lwe_modular_variance = (double)GetParam().lwe_modular_variance;
    glwe_modular_variance = (double)GetParam().glwe_modular_variance;
    pbs_base_log = (int)GetParam().pbs_base_log;
    pbs_level = (int)GetParam().pbs_level;
    ksk_base_log = (int)GetParam().ksk_base_log;
    ksk_level = (int)GetParam().ksk_level;
    total_message_bits = (int)GetParam().total_message_bits;
    number_of_blocks = (int)GetParam().number_of_blocks;
    message_modulus = (int)GetParam().message_modulus;
    carry_modulus = (int)GetParam().carry_modulus;
    repetitions = (int)GetParam().repetitions;
    samples = (int)GetParam().samples;
    pbs_type = (PBS_TYPE)GetParam().pbs_type;
    mem_ptr = new int_mul_memory<uint64_t>;
    stream = cuda_create_stream(gpu_index);

    integer_multiplication_setup(
        stream, &csprng, &lwe_sk_in_array, &lwe_sk_out_array, &d_bsk_array,
        &d_ksk_array, &plaintexts_1, &plaintexts_2, &d_lwe_ct_in_array_1,
        &d_lwe_ct_in_array_2, &d_lwe_ct_out_array, mem_ptr, lwe_dimension,
        glwe_dimension, polynomial_size, lwe_modular_variance,
        glwe_modular_variance, pbs_base_log, pbs_level, ksk_base_log, ksk_level,
        total_message_bits, number_of_blocks, message_modulus, carry_modulus,
        &delta, repetitions, samples, pbs_type, gpu_index);

    expected = (uint64_t *)malloc(repetitions * samples * number_of_blocks *
                                  sizeof(uint64_t));
    printf("Message 1: %lu, %lu, Message 2: %lu, %lu\n",
           plaintexts_1[0] / delta, plaintexts_1[1] / delta,
           plaintexts_2[0] / delta, plaintexts_2[1] / delta);
    for (int r = 0; r < repetitions; r++) {
      for (int s = 0; s < samples; s++) {
        uint64_t message_1 = 0;
        uint64_t message_2 = 0;
        for (int i = 0; i < number_of_blocks; i++) {
          message_1 += std::pow(message_modulus, i) *
                       plaintexts_1[r * samples * number_of_blocks +
                                    s * number_of_blocks + i] /
                       delta;
          message_2 += std::pow(message_modulus, i) *
                       plaintexts_2[r * samples * number_of_blocks +
                                    s * number_of_blocks + i] /
                       delta;
        }
        uint64_t expected_result =
            (message_1 * message_2) % (1 << total_message_bits);
        for (int i = number_of_blocks - 1; i >= 0; i--) {
          uint64_t coef = expected_result / std::pow(message_modulus, i);
          expected[i] = coef;
          expected_result -= coef * std::pow(message_modulus, i);
          printf("Expected [%d] = %lu\n", i, expected[i]);
        }
      }
    }
  }

  void TearDown() {
    free(expected);
    integer_multiplication_teardown(
        stream, csprng, lwe_sk_in_array, lwe_sk_out_array, d_bsk_array,
        d_ksk_array, plaintexts_1, plaintexts_2, d_lwe_ct_in_array_1,
        d_lwe_ct_in_array_2, d_lwe_ct_out_array, mem_ptr);
    cuda_synchronize_stream(stream);
    cuda_destroy_stream(stream, gpu_index);
  }
};

TEST_P(IntegerMultiplicationTestPrimitives_u64, integer_multiplication) {

  int bsk_size = (glwe_dimension + 1) * (glwe_dimension + 1) * pbs_level *
                 polynomial_size * (lwe_dimension + 1);
  int ksk_size =
      ksk_level * (lwe_dimension + 1) * glwe_dimension * polynomial_size;

  uint64_t *lwe_ct_out_array =
      (uint64_t *)malloc((glwe_dimension * polynomial_size + 1) *
                         number_of_blocks * sizeof(uint64_t));
  uint64_t *decrypted = (uint64_t *)malloc(number_of_blocks * sizeof(uint64_t));
  for (int r = 0; r < repetitions; r++) {
    void *d_bsk = d_bsk_array + (ptrdiff_t)(bsk_size * r);
    uint64_t *d_ksk = d_ksk_array + (ptrdiff_t)(ksk_size * r);
    uint64_t *lwe_sk =
        lwe_sk_in_array + (ptrdiff_t)(glwe_dimension * polynomial_size * r);
    for (int s = 0; s < samples; s++) {
      uint64_t *d_lwe_ct_in_1 =
          d_lwe_ct_in_array_1 +
          (ptrdiff_t)((r * samples * number_of_blocks + s * number_of_blocks) *
                      (glwe_dimension * polynomial_size + 1));
      uint64_t *d_lwe_ct_in_2 =
          d_lwe_ct_in_array_2 +
          (ptrdiff_t)((r * samples * number_of_blocks + s * number_of_blocks) *
                      (glwe_dimension * polynomial_size + 1));
      uint32_t ct_degree_out = 0;
      uint32_t ct_degree_left = 0;
      uint32_t ct_degree_right = 0;
      int8_t *mult_buffer = NULL;
      // Execute integer mult
      if (USE_MULTI_GPU) {

        // //for debug
        // int8_t *pbs_buffer;
        // scratch_cuda_multi_bit_pbs_64(
        //   stream, gpu_index, &pbs_buffer, lwe_dimension, glwe_dimension,
        // polynomial_size, pbs_level, 3, 16, number_of_blocks *
        // number_of_blocks, cuda_get_max_shared_memory(gpu_index), true);
        // mem_ptr->pbs_buffer = pbs_buffer;
        // cuda_integer_mult_radix_ciphertext_kb_64(
        //     stream, gpu_index, d_lwe_ct_out_array, d_lwe_ct_in_1,
        //     d_lwe_ct_in_2, &ct_degree_out, &ct_degree_left, &ct_degree_right,
        //     d_bsk, d_ksk, (void *)mem_ptr, message_modulus, carry_modulus,
        //     glwe_dimension, lwe_dimension, polynomial_size, pbs_base_log,
        //     pbs_level, ksk_base_log, ksk_level, number_of_blocks,
        //     cuda_get_max_shared_memory(gpu_index));

        // ///////////////////

        scratch_cuda_integer_mult_radix_ciphertext_kb_64_multi_gpu(
            mem_ptr, d_bsk, d_ksk, message_modulus, carry_modulus,
            glwe_dimension, lwe_dimension, polynomial_size, pbs_base_log,
            pbs_level, ksk_base_log, ksk_level, number_of_blocks, pbs_type,
            cuda_get_max_shared_memory(gpu_index), true);

        cuda_integer_mult_radix_ciphertext_kb_64_multi_gpu(
            d_lwe_ct_out_array, d_lwe_ct_in_1, d_lwe_ct_in_2, &ct_degree_out,
            &ct_degree_left, &ct_degree_right, d_bsk, d_ksk, (void *)mem_ptr,
            message_modulus, carry_modulus, glwe_dimension, lwe_dimension,
            polynomial_size, pbs_base_log, pbs_level, ksk_base_log, ksk_level,
            number_of_blocks, pbs_type, cuda_get_max_shared_memory(gpu_index));

      } else {
        scratch_cuda_integer_mult_radix_ciphertext_kb_64(
            stream, gpu_index, (void *)mem_ptr, message_modulus, carry_modulus,
            glwe_dimension, lwe_dimension, polynomial_size, pbs_base_log,
            pbs_level, ksk_base_log, ksk_level, number_of_blocks, pbs_type,
            cuda_get_max_shared_memory(gpu_index), true);

        cuda_integer_mult_radix_ciphertext_kb_64(
            stream, gpu_index, d_lwe_ct_out_array, d_lwe_ct_in_1, d_lwe_ct_in_2,
            &ct_degree_out, &ct_degree_left, &ct_degree_right, d_bsk, d_ksk,
            (void *)mem_ptr, message_modulus, carry_modulus, glwe_dimension,
            lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
            ksk_base_log, ksk_level, number_of_blocks, pbs_type,
            cuda_get_max_shared_memory(gpu_index));
      }

      cuda_memcpy_async_to_cpu(lwe_ct_out_array, d_lwe_ct_out_array,
                               (glwe_dimension * polynomial_size + 1) *
                                   number_of_blocks * sizeof(uint64_t),
                               stream, gpu_index);

      // Process result
      decrypt_integer_u64_blocks(lwe_ct_out_array, lwe_sk, &decrypted,
                                 glwe_dimension * polynomial_size,
                                 number_of_blocks, delta, message_modulus);
      printf("decrypted[0]: %lu\n", decrypted[0]);
      printf("decrypted[1]: %lu\n", decrypted[1]);
      for (int i = 0; i < number_of_blocks; i++) {
        ASSERT_EQ(decrypted[i], expected[i])
            << "Repetition: " << r << ", sample: " << s;
      }
    }
  }
  free(lwe_ct_out_array);
  free(decrypted);
}

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<IntegerMultiplicationTestParams>
    integer_mult_params_u64 = ::testing::Values(
        // n, k, N, lwe_variance, glwe_variance, pbs_base_log, pbs_level,
        // ksk_base_log, ksk_level,
        // total_message_bits, number_of_blocks, message_modulus,
        // carry_modulus, repetitions, samples
        // SHORTINT_PARAM_MESSAGE_2_CARRY_2
        // The total number of bits of message should not exceed 64 to be
        // able to use a uint64_t representation for the result calculation
        // in clear
        (IntegerMultiplicationTestParams){744, 1, 2048, 4.478453795193731e-11,
                                          8.645717832544903e-32, 23, 1, 3, 5, 4,
                                          2, 4, 4, 1, 1, AMORTIZED});
std::string
printParamName(::testing::TestParamInfo<IntegerMultiplicationTestParams> p) {
  IntegerMultiplicationTestParams params = p.param;

  return "n_" + std::to_string(params.lwe_dimension) + "_k_" +
         std::to_string(params.glwe_dimension) + "_N_" +
         std::to_string(params.polynomial_size) + "_pbs_base_log_" +
         std::to_string(params.pbs_base_log) + "_pbs_level_" +
         std::to_string(params.pbs_level) + "_number_of_blocks_" +
         std::to_string(params.number_of_blocks) + "_message_modulus_" +
         std::to_string(params.message_modulus) + "_carry_modulus_" +
         std::to_string(params.carry_modulus);
}

INSTANTIATE_TEST_CASE_P(IntegerMultiplicationInstantiation,
                        IntegerMultiplicationTestPrimitives_u64,
                        integer_mult_params_u64, printParamName);
