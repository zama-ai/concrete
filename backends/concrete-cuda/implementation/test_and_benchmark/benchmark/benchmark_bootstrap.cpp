#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <setup_and_teardown.h>

typedef struct {
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  int pbs_base_log;
  int pbs_level;
  int input_lwe_ciphertext_count;
} BootstrapBenchmarkParams;

class BootstrapBenchmark_u64 : public benchmark::Fixture {
protected:
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  int input_lwe_ciphertext_count;
  double lwe_modular_variance = 0.000007069849454709433;
  double glwe_modular_variance = 0.00000000000000029403601535432533;
  int pbs_base_log;
  int pbs_level;
  int message_modulus = 4;
  int carry_modulus = 4;
  int payload_modulus;
  uint64_t delta;
  double *d_fourier_bsk_array;
  uint64_t *d_lut_pbs_identity;
  uint64_t *d_lut_pbs_indexes;
  uint64_t *d_lwe_ct_in_array;
  uint64_t *d_lwe_ct_out_array;
  uint64_t *lwe_ct_array;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;
  uint64_t *plaintexts;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  int8_t *amortized_pbs_buffer;
  int8_t *lowlat_pbs_buffer;

public:
  void SetUp(const ::benchmark::State &state) {
    stream = cuda_create_stream(0);

    lwe_dimension = state.range(0);
    glwe_dimension = state.range(1);
    polynomial_size = state.range(2);
    pbs_base_log = state.range(3);
    pbs_level = state.range(4);
    input_lwe_ciphertext_count = state.range(5);

    bootstrap_setup(stream, &csprng, &lwe_sk_in_array, &lwe_sk_out_array,
                    &d_fourier_bsk_array, &plaintexts, &d_lut_pbs_identity,
                    &d_lut_pbs_indexes, &d_lwe_ct_in_array, &d_lwe_ct_out_array,
                    &amortized_pbs_buffer, &lowlat_pbs_buffer, lwe_dimension,
                    glwe_dimension, polynomial_size, lwe_modular_variance,
                    glwe_modular_variance, pbs_base_log, pbs_level,
                    message_modulus, carry_modulus, &payload_modulus, &delta,
                    input_lwe_ciphertext_count, 1, 1, gpu_index);

    // We keep the following for the benchmarks with copies
    lwe_ct_array = (uint64_t *)malloc(
        (lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint64_t));
  }

  void TearDown() {
    bootstrap_teardown(stream, csprng, lwe_sk_in_array, lwe_sk_out_array,
                       d_fourier_bsk_array, plaintexts, d_lut_pbs_identity,
                       d_lut_pbs_indexes, d_lwe_ct_in_array, d_lwe_ct_out_array,
                       amortized_pbs_buffer, lowlat_pbs_buffer, gpu_index);
    free(lwe_ct_array);
  }
};

BENCHMARK_DEFINE_F(BootstrapBenchmark_u64, AmortizedPBS)(benchmark::State &st) {
  void *v_stream = (void *)stream;

  for (auto _ : st) {
    // Execute PBS
    cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_ct_out_array,
        (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
        (void *)d_lwe_ct_in_array, (void *)d_fourier_bsk_array,
        amortized_pbs_buffer, lwe_dimension, glwe_dimension, polynomial_size,
        pbs_base_log, pbs_level, input_lwe_ciphertext_count,
        input_lwe_ciphertext_count, 0, cuda_get_max_shared_memory(gpu_index));
    cuda_synchronize_stream(v_stream);
  }
}

BENCHMARK_DEFINE_F(BootstrapBenchmark_u64, CopiesPlusAmortizedPBS)
(benchmark::State &st) {
  void *v_stream = (void *)stream;

  for (auto _ : st) {
    cuda_memcpy_async_to_gpu(d_lwe_ct_in_array, lwe_ct_array,
                             (lwe_dimension + 1) * input_lwe_ciphertext_count *
                                 sizeof(uint64_t),
                             stream, gpu_index);

    // Execute PBS
    cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_ct_out_array,
        (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
        (void *)d_lwe_ct_in_array, (void *)d_fourier_bsk_array,
        amortized_pbs_buffer, lwe_dimension, glwe_dimension, polynomial_size,
        pbs_base_log, pbs_level, input_lwe_ciphertext_count,
        input_lwe_ciphertext_count, 0, cuda_get_max_shared_memory(gpu_index));

    cuda_memcpy_async_to_cpu(lwe_ct_array, d_lwe_ct_out_array,
                             (lwe_dimension + 1) * input_lwe_ciphertext_count *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(v_stream);
  }
}

BENCHMARK_DEFINE_F(BootstrapBenchmark_u64, LowLatencyPBS)
(benchmark::State &st) {
  for (auto _ : st) {
    // Execute PBS
    cuda_bootstrap_low_latency_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_ct_out_array,
        (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
        (void *)d_lwe_ct_in_array, (void *)d_fourier_bsk_array,
        lowlat_pbs_buffer, lwe_dimension, glwe_dimension, polynomial_size,
        pbs_base_log, pbs_level, 1, 1, 0,
        cuda_get_max_shared_memory(gpu_index));
    cuda_synchronize_stream(stream);
  }
}

BENCHMARK_DEFINE_F(BootstrapBenchmark_u64, CopiesPlusLowLatencyPBS)
(benchmark::State &st) {
  void *v_stream = (void *)stream;

  for (auto _ : st) {
    cuda_memcpy_async_to_gpu(d_lwe_ct_in_array, lwe_ct_array,
                             (lwe_dimension + 1) * input_lwe_ciphertext_count *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    // Execute PBS
    cuda_bootstrap_low_latency_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_ct_out_array,
        (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
        (void *)d_lwe_ct_in_array, (void *)d_fourier_bsk_array,
        lowlat_pbs_buffer, lwe_dimension, glwe_dimension, polynomial_size,
        pbs_base_log, pbs_level, 1, 1, 0,
        cuda_get_max_shared_memory(gpu_index));

    cuda_memcpy_async_to_cpu(lwe_ct_array, d_lwe_ct_out_array,
                             (lwe_dimension + 1) * input_lwe_ciphertext_count *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(v_stream);
  }
}

static void
BootstrapBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  // lwe_dimension, glwe_dimension, polynomial_size, pbs_base_log, pbs_level,
  // input_lwe_ciphertext_count
  std::vector<BootstrapBenchmarkParams> params = {
      (BootstrapBenchmarkParams){567, 5, 256, 15, 1, 1},
      (BootstrapBenchmarkParams){577, 6, 256, 12, 3, 1},
      (BootstrapBenchmarkParams){553, 4, 512, 12, 3, 1},
      (BootstrapBenchmarkParams){769, 2, 1024, 23, 1, 1},
      (BootstrapBenchmarkParams){714, 2, 1024, 15, 2, 1},
      (BootstrapBenchmarkParams){694, 2, 1024, 8, 5, 1},
      (BootstrapBenchmarkParams){881, 1, 8192, 22, 1, 1},
      (BootstrapBenchmarkParams){879, 1, 8192, 11, 3, 1},
  };

  // Add to the list of parameters to benchmark
  for (auto x : params)
    for (int num_samples = 1; num_samples <= 10000; num_samples *= 10) {
      b->Args({x.lwe_dimension, x.glwe_dimension, x.polynomial_size,
               x.pbs_base_log, x.pbs_level, num_samples});
    }
}

BENCHMARK_REGISTER_F(BootstrapBenchmark_u64, AmortizedPBS)
    ->Apply(BootstrapBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(BootstrapBenchmark_u64, LowLatencyPBS)
    ->Apply(BootstrapBenchmarkGenerateParams);

BENCHMARK_REGISTER_F(BootstrapBenchmark_u64, CopiesPlusAmortizedPBS)
    ->Apply(BootstrapBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(BootstrapBenchmark_u64, CopiesPlusLowLatencyPBS)
    ->Apply(BootstrapBenchmarkGenerateParams);
