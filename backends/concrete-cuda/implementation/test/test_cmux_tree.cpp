#include "../include/device.h"
#include "../include/vertical_packing.h"
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
  int glwe_dimension;
  int polynomial_size;
  int r;
  int tau;
  double glwe_modular_variance;
  int base_log;
  int level_count;
  int delta_log;
} CMUXTreeTestParams;

class CMUXTreeTestPrimitives_u64
    : public ::testing::TestWithParam<CMUXTreeTestParams> {
protected:
  int glwe_dimension;
  int polynomial_size;
  int r_lut;
  int tau;
  double glwe_modular_variance;
  int base_log;
  int level_count;
  uint64_t delta;
  int delta_log;
  Csprng *csprng;
  uint64_t *plaintexts;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *glwe_sk;
  uint64_t *d_lut_identity;

public:
  // Test arithmetic functions
  void SetUp() {
    stream = cuda_create_stream(0);
    void *v_stream = (void *)stream;

    // TestParams
    glwe_dimension = (int)GetParam().glwe_dimension;
    polynomial_size = (int)GetParam().polynomial_size;
    r_lut = (int)GetParam().r;
    tau = (int)GetParam().tau;
    glwe_modular_variance = (int)GetParam().glwe_modular_variance;
    base_log = (int)GetParam().base_log;
    level_count = (int)GetParam().level_count;
    delta_log = (int)GetParam().delta_log;

    // Value of the shift we multiply our messages by
    delta = ((uint64_t)(1) << delta_log);

    // Create a Csprng
    csprng =
        (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
    uint8_t seed[16] = {(uint8_t)0};
    concrete_cpu_construct_concrete_csprng(
        csprng, Uint128{.little_endian_bytes = {*seed}});

    // Generate the keys
    generate_glwe_secret_keys(&glwe_sk, glwe_dimension, polynomial_size,
                              csprng, REPETITIONS);
    plaintexts = generate_plaintexts(r_lut, 1, 1, REPETITIONS, SAMPLES);

    // Create the LUT
    int num_lut = (1 << r_lut);
    d_lut_identity = (uint64_t *)cuda_malloc_async(
        polynomial_size * num_lut * tau * sizeof(uint64_t), stream, gpu_index);
    uint64_t *lut_cmux_tree_identity = generate_identity_lut_cmux_tree(
        polynomial_size, num_lut, tau, delta_log);

    // Copy all LUTs
    cuda_memcpy_async_to_gpu(d_lut_identity, lut_cmux_tree_identity,
                             polynomial_size * num_lut * tau * sizeof(uint64_t),
                             stream, gpu_index);

    cuda_synchronize_stream(v_stream);
    free(lut_cmux_tree_identity);
  }

  void TearDown() {
    cuda_synchronize_stream(stream);
    concrete_cpu_destroy_concrete_csprng(csprng);
    free(plaintexts);
    free(csprng);
    cuda_drop_async(d_lut_identity, stream, gpu_index);
    cuda_destroy_stream(stream, gpu_index);
  }
};

TEST_P(CMUXTreeTestPrimitives_u64, cmux_tree) {
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;
  int glwe_size = (glwe_dimension + 1) * polynomial_size;
  uint64_t *d_ggsw_bit_array = (uint64_t *)cuda_malloc_async(
      r_lut * ggsw_size * sizeof(uint64_t), stream, gpu_index);
  uint64_t *d_results = (uint64_t *)cuda_malloc_async(
      tau * glwe_size * sizeof(uint64_t), stream, gpu_index);
  uint64_t *results = (uint64_t *)malloc(tau * glwe_size * sizeof(uint64_t));
  uint64_t *ggsw = (uint64_t *)malloc(ggsw_size * sizeof(uint64_t));

  int8_t *cmux_tree_buffer = nullptr;
  scratch_cuda_cmux_tree_64(stream, gpu_index, &cmux_tree_buffer,
                            glwe_dimension, polynomial_size, level_count, r_lut,
                            tau, cuda_get_max_shared_memory(gpu_index), true);

  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t witness = plaintexts[r * SAMPLES + s];

      // Instantiate the GGSW m^tree ciphertexts
      // We need r GGSW ciphertexts
      // Bit decomposition of the value from MSB to LSB
      uint64_t *bit_array = bit_decompose_value(witness, r_lut);

      for (int i = 0; i < r_lut; i++) {
        uint64_t *d_ggsw_slice = d_ggsw_bit_array + i * ggsw_size;
        concrete_cpu_encrypt_ggsw_ciphertext_u64(
            glwe_sk, ggsw, bit_array[i], glwe_dimension, polynomial_size,
            level_count, base_log, glwe_modular_variance, csprng,
            &CONCRETE_CSPRNG_VTABLE);
        cuda_memcpy_async_to_gpu(d_ggsw_slice, ggsw,
                                 ggsw_size * sizeof(uint64_t), stream,
                                 gpu_index);
      }
      cuda_synchronize_stream(stream);

      // Execute scratch/CMUX tree/cleanup
      cuda_cmux_tree_64(stream, gpu_index, (void *)d_results,
                        (void *)d_ggsw_bit_array, (void *)d_lut_identity,
                        cmux_tree_buffer, glwe_dimension, polynomial_size,
                        base_log, level_count, r_lut, tau,
                        cuda_get_max_shared_memory(gpu_index));

      // Copy result back
      cuda_memcpy_async_to_cpu(results, d_results,
                               tau * glwe_size * sizeof(uint64_t), stream,
                               gpu_index);
      cuda_synchronize_stream(stream);
      for (int tree = 0; tree < tau; tree++) {
        uint64_t *result = results + tree * glwe_size;
        uint64_t *decrypted =
            (uint64_t *)malloc(polynomial_size * sizeof(uint64_t));
        concrete_cpu_decrypt_glwe_ciphertext_u64(
            glwe_sk, decrypted, result, glwe_dimension, polynomial_size);
        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted[0] & rounding_bit) << 1;
        uint64_t decoded = (decrypted[0] + rounding) / delta;
        EXPECT_EQ(decoded, (witness + tree) % (1 << (64 - delta_log)));
        free(decrypted);
      }
      free(bit_array);
    }
  }
  cuda_synchronize_stream(stream);
  cleanup_cuda_cmux_tree(stream, gpu_index, &cmux_tree_buffer);
  free(ggsw);

  cuda_drop_async(d_ggsw_bit_array, stream, gpu_index);
}

int glwe_dimension;
int polynomial_size;
double glwe_modular_variance;
int base_log;
int level_count;
int message_modulus;
int carry_modulus;

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<CMUXTreeTestParams> cmux_tree_params_u64 =
    ::testing::Values(
        // k, N, r, tau, glwe_variance, base_log, level_count, delta_log
        (CMUXTreeTestParams){2, 256, 10, 6, 0.00000000000000029403601535432533,
                             6, 3, 60});

std::string printParamName(::testing::TestParamInfo<CMUXTreeTestParams> p) {
  CMUXTreeTestParams params = p.param;

  return "k_" + std::to_string(params.glwe_dimension) + "_N_" +
         std::to_string(params.polynomial_size) + "_tau_" +
         std::to_string(params.tau) + "_base_log_" +
         std::to_string(params.base_log) + "_level_count_" +
         std::to_string(params.level_count);
}

INSTANTIATE_TEST_CASE_P(CMUXTreeInstantiation, CMUXTreeTestPrimitives_u64,
                        cmux_tree_params_u64, printParamName);