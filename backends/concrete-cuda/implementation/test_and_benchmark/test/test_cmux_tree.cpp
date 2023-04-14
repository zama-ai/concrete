#include <cmath>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <setup_and_teardown.h>
#include <stdlib.h>

const unsigned REPETITIONS = 5;
const unsigned SAMPLES = 50;

typedef struct {
  int glwe_dimension;
  int polynomial_size;
  int p; // number_of_bits_to_extract
  int tau;
  double glwe_modular_variance;
  int base_log;
  int level_count;
} CMUXTreeTestParams;

class CMUXTreeTestPrimitives_u64
    : public ::testing::TestWithParam<CMUXTreeTestParams> {
protected:
  int glwe_dimension;
  int polynomial_size;
  int p;
  int tau;
  double glwe_modular_variance;
  int base_log;
  int level_count;
  uint64_t delta;
  uint32_t delta_log;
  Csprng *csprng;
  uint64_t *plaintexts;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *glwe_sk;
  uint64_t *d_lut_identity;
  int8_t *cmux_tree_buffer = nullptr;
  uint64_t *d_ggsw_bit_array;
  uint64_t *d_glwe_out;
  uint64_t *glwe_out;

public:
  // Test arithmetic functions
  void SetUp() {
    stream = cuda_create_stream(0);

    // TestParams
    glwe_dimension = (int)GetParam().glwe_dimension;
    polynomial_size = (int)GetParam().polynomial_size;
    p = (int)GetParam().p;
    tau = (int)GetParam().tau;
    glwe_modular_variance = (int)GetParam().glwe_modular_variance;
    base_log = (int)GetParam().base_log;
    level_count = (int)GetParam().level_count;

    cmux_tree_setup(stream, &csprng, &glwe_sk, &d_lut_identity, &plaintexts,
                    &d_ggsw_bit_array, &cmux_tree_buffer, &d_glwe_out,
                    glwe_dimension, polynomial_size, base_log, level_count,
                    glwe_modular_variance, p, tau, &delta_log, REPETITIONS,
                    SAMPLES, gpu_index);

    // Value of the shift we multiply our messages by
    delta = ((uint64_t)(1) << delta_log);

    glwe_out = (uint64_t *)malloc(tau * (glwe_dimension + 1) * polynomial_size *
                                  sizeof(uint64_t));
  }

  void TearDown() {
    free(glwe_out);
    cmux_tree_teardown(stream, &csprng, &glwe_sk, &d_lut_identity, &plaintexts,
                       &d_ggsw_bit_array, &cmux_tree_buffer, &d_glwe_out,
                       gpu_index);
  }
};

TEST_P(CMUXTreeTestPrimitives_u64, cmux_tree) {
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;
  int glwe_size = (glwe_dimension + 1) * polynomial_size;
  uint32_t r_lut = tau * p - log2(polynomial_size);

  // Here execute the PBS
  for (uint r = 0; r < REPETITIONS; r++) {
    for (uint s = 0; s < SAMPLES; s++) {
      uint64_t witness = plaintexts[r * SAMPLES + s];

      uint64_t *d_ggsw_bit_array_slice =
          d_ggsw_bit_array +
          (ptrdiff_t)((r * SAMPLES * r_lut + s * r_lut) * ggsw_size);

      // Execute CMUX tree
      cuda_cmux_tree_64(stream, gpu_index, (void *)d_glwe_out,
                        (void *)d_ggsw_bit_array_slice, (void *)d_lut_identity,
                        cmux_tree_buffer, glwe_dimension, polynomial_size,
                        base_log, level_count, (1 << (tau * p)), tau,
                        cuda_get_max_shared_memory(gpu_index));

      // Copy result back
      cuda_memcpy_async_to_cpu(glwe_out, d_glwe_out,
                               tau * glwe_size * sizeof(uint64_t), stream,
                               gpu_index);
      cuda_synchronize_stream(stream);
      for (int tree = 0; tree < tau; tree++) {
        uint64_t *result = glwe_out + tree * glwe_size;
        uint64_t *decrypted =
            (uint64_t *)malloc(polynomial_size * sizeof(uint64_t));
        concrete_cpu_decrypt_glwe_ciphertext_u64(
            glwe_sk, decrypted, result, glwe_dimension, polynomial_size);
        // The bit before the message
        uint64_t rounding_bit = delta >> 1;
        // Compute the rounding bit
        uint64_t rounding = (decrypted[0] & rounding_bit) << 1;
        uint64_t decoded = (decrypted[0] + rounding) / delta;
        EXPECT_EQ(decoded, witness % (1 << p))
            << "Repetition: " << r << ", sample: " << s << ", tree: " << tree;
        free(decrypted);
      }
    }
  }
  cuda_synchronize_stream(stream);
}

// Defines for which parameters set the PBS will be tested.
// It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<CMUXTreeTestParams> cmux_tree_params_u64 =
    ::testing::Values(
        // k, N, p, tau, glwe_variance, base_log, level_count
        (CMUXTreeTestParams){2, 256, 10, 1, 2.9403601535432533e-16, 6, 3},
        (CMUXTreeTestParams){2, 512, 13, 1, 2.9403601535432533e-16, 6, 3},
        (CMUXTreeTestParams){1, 1024, 11, 1, 2.9403601535432533e-16, 6, 3});

std::string printParamName(::testing::TestParamInfo<CMUXTreeTestParams> p) {
  CMUXTreeTestParams params = p.param;

  return "k_" + std::to_string(params.glwe_dimension) + "_N_" +
         std::to_string(params.polynomial_size) + "_tau_" +
         std::to_string(params.tau) + "_p_" + std::to_string(params.p) +
         "_base_log_" + std::to_string(params.base_log) + "_level_count_" +
         std::to_string(params.level_count);
}

INSTANTIATE_TEST_CASE_P(CMUXTreeInstantiation, CMUXTreeTestPrimitives_u64,
                        cmux_tree_params_u64, printParamName);
