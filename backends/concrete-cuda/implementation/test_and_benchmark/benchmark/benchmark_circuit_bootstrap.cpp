#include <benchmark/benchmark.h>
#include <cstdint>
#include <setup_and_teardown.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  int pbs_base_log;
  int pbs_level;
  int pksk_base_log;
  int pksk_level;
  int cbs_base_log;
  int cbs_level;
  int number_of_inputs;
} CircuitBootstrapBenchmarkParams;

class CircuitBootstrap_u64 : public benchmark::Fixture {
protected:
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance = 7.52316384526264e-37;
  double glwe_modular_variance = 7.52316384526264e-37;
  int pbs_base_log;
  int pbs_level;
  int pksk_base_log;
  int pksk_level;
  int cbs_base_log;
  int cbs_level;
  int number_of_inputs;
  int number_of_bits_of_message_including_padding;
  int ggsw_size;
  uint64_t delta;
  int delta_log;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *lwe_sk_in;
  uint64_t *lwe_sk_out;
  uint64_t *plaintexts;
  double *d_fourier_bsk;
  uint64_t *d_pksk;
  uint64_t *d_lwe_ct_in_array;
  uint64_t *d_ggsw_ct_out_array;
  uint64_t *d_lut_vector_indexes;
  int8_t *cbs_buffer;

public:
  // Test arithmetic functions
  void SetUp(const ::benchmark::State &state) {
    stream = cuda_create_stream(0);

    // TestParams
    lwe_dimension = state.range(0);
    glwe_dimension = state.range(1);
    polynomial_size = state.range(2);
    pbs_base_log = state.range(3);
    pbs_level = state.range(4);
    pksk_base_log = state.range(5);
    pksk_level = state.range(6);
    cbs_base_log = state.range(7);
    cbs_level = state.range(8);
    number_of_inputs = state.range(9);

    // We generate binary messages
    number_of_bits_of_message_including_padding = 2;
    ggsw_size = cbs_level * (glwe_dimension + 1) * (glwe_dimension + 1) *
                polynomial_size;
    circuit_bootstrap_setup(
        stream, &csprng, &lwe_sk_in, &lwe_sk_out, &d_fourier_bsk, &d_pksk,
        &plaintexts, &d_lwe_ct_in_array, &d_ggsw_ct_out_array,
        &d_lut_vector_indexes, &cbs_buffer, lwe_dimension, glwe_dimension,
        polynomial_size, lwe_modular_variance, glwe_modular_variance,
        pksk_base_log, pksk_level, pbs_base_log, pbs_level, cbs_level,
        number_of_bits_of_message_including_padding, ggsw_size, &delta_log,
        &delta, number_of_inputs, 1, 1, gpu_index);
  }

  void TearDown(const ::benchmark::State &state) {
    circuit_bootstrap_teardown(stream, csprng, lwe_sk_in, lwe_sk_out,
                               d_fourier_bsk, d_pksk, plaintexts,
                               d_lwe_ct_in_array, d_lut_vector_indexes,
                               d_ggsw_ct_out_array, cbs_buffer, gpu_index);
  }
};

BENCHMARK_DEFINE_F(CircuitBootstrap_u64, ConcreteCuda_CircuitBootstrap)
(benchmark::State &st) {
  for (auto _ : st) {
    // Execute circuit bootstrap
    cuda_circuit_bootstrap_64(
        stream, gpu_index, (void *)d_ggsw_ct_out_array,
        (void *)d_lwe_ct_in_array, (void *)d_fourier_bsk, (void *)d_pksk,
        (void *)d_lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, pbs_level, pbs_base_log, pksk_level,
        pksk_base_log, cbs_level, cbs_base_log, number_of_inputs,
        cuda_get_max_shared_memory(gpu_index));
    cuda_synchronize_stream(stream);
  }
}

static void
CircuitBootstrapBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  std::vector<CircuitBootstrapBenchmarkParams> params = {
      (CircuitBootstrapBenchmarkParams){10, 2, 512, 11, 2, 15, 2, 10, 1, 100}};

  // Add to the list of parameters to benchmark
  for (auto x : params)
    b->Args({x.lwe_dimension, x.glwe_dimension, x.polynomial_size,
             x.pbs_base_log, x.pbs_level, x.pksk_base_log, x.pksk_level,
             x.cbs_base_log, x.cbs_level, x.number_of_inputs});
}

BENCHMARK_REGISTER_F(CircuitBootstrap_u64, ConcreteCuda_CircuitBootstrap)
    ->Apply(CircuitBootstrapBenchmarkGenerateParams);
