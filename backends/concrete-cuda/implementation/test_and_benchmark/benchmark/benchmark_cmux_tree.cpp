#include <benchmark/benchmark.h>
#include <cstdint>
#include <functional>
#include <setup_and_teardown.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int glwe_dimension;
  int polynomial_size;
  int r;
  int tau;
  int base_log;
  int level_count;
} CMUXTreeBenchmarkParams;

class CMUXTree_u64 : public benchmark::Fixture {
protected:
  int glwe_dimension;
  int polynomial_size;
  int r_lut;
  int tau;
  double glwe_modular_variance = 0.00000000000000029403601535432533;
  int base_log;
  int level_count;
  uint64_t delta;
  int delta_log = 60;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *d_lut_identity;
  uint64_t *d_ggsw_bit_array;
  uint64_t *plaintexts;
  uint64_t *d_glwe_out;
  uint64_t *glwe_sk;
  int8_t *cmux_tree_buffer = nullptr;

public:
  // Test arithmetic functions
  void SetUp(const ::benchmark::State &state) {
    stream = cuda_create_stream(0);

    // TestParams
    glwe_dimension = state.range(0);
    polynomial_size = state.range(1);
    r_lut = state.range(2);
    tau = state.range(3);
    base_log = state.range(4);
    level_count = state.range(5);

    // Value of the shift we multiply our messages by
    delta = ((uint64_t)(1) << delta_log);

    cmux_tree_setup(stream, &csprng, &glwe_sk, &d_lut_identity, &plaintexts,
                    &d_ggsw_bit_array, &cmux_tree_buffer, &d_glwe_out,
                    glwe_dimension, polynomial_size, base_log, level_count,
                    glwe_modular_variance, r_lut, tau, delta_log, 1, 1,
                    gpu_index);
  }

  void TearDown() {
    cmux_tree_teardown(stream, &csprng, &glwe_sk, &d_lut_identity, &plaintexts,
                       &d_ggsw_bit_array, &cmux_tree_buffer, &d_glwe_out,
                       gpu_index);
  }
};

BENCHMARK_DEFINE_F(CMUXTree_u64, ConcreteCuda_CMUXTree)(benchmark::State &st) {
  for (auto _ : st) {
    // Execute scratch/CMUX tree/cleanup
    cuda_cmux_tree_64(stream, gpu_index, (void *)d_glwe_out,
                      (void *)d_ggsw_bit_array, (void *)d_lut_identity,
                      cmux_tree_buffer, glwe_dimension, polynomial_size,
                      base_log, level_count, r_lut, tau,
                      cuda_get_max_shared_memory(gpu_index));
    cuda_synchronize_stream(stream);
  }
}

// k, N, r, tau, base_log, level_count
static void CMUXTreeBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  std::vector<CMUXTreeBenchmarkParams> params = {
      (CMUXTreeBenchmarkParams){2, 256, 10, 6, 6, 3},
  };

  // Add to the list of parameters to benchmark
  for (auto x : params)
    b->Args({x.glwe_dimension, x.polynomial_size, x.r, x.tau, x.base_log,
             x.level_count});
}

BENCHMARK_REGISTER_F(CMUXTree_u64, ConcreteCuda_CMUXTree)
    ->Apply(CMUXTreeBenchmarkGenerateParams);
