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

class Bootstrap_u64 : public benchmark::Fixture {
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

public:
  void SetUp(const ::benchmark::State &state) {
    cudaDeviceSynchronize();
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
                    lwe_dimension, glwe_dimension, polynomial_size,
                    lwe_modular_variance, glwe_modular_variance, pbs_base_log,
                    pbs_level, message_modulus, carry_modulus, &payload_modulus,
                    &delta, input_lwe_ciphertext_count, 1, 1, gpu_index);

    // We keep the following for the benchmarks with copies
    lwe_ct_array = (uint64_t *)malloc(
        (lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint64_t));
  }

  void TearDown(const ::benchmark::State &state) {
    bootstrap_teardown(stream, csprng, lwe_sk_in_array, lwe_sk_out_array,
                       d_fourier_bsk_array, plaintexts, d_lut_pbs_identity,
                       d_lut_pbs_indexes, d_lwe_ct_in_array, d_lwe_ct_out_array,
                       gpu_index);
    free(lwe_ct_array);
    cudaDeviceSynchronize();
    cudaDeviceReset();
  }
};

BENCHMARK_DEFINE_F(Bootstrap_u64, ConcreteCuda_AmortizedPBS)
(benchmark::State &st) {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  uint64_t buffer_size = get_buffer_size_bootstrap_amortized_64(
      glwe_dimension, polynomial_size, input_lwe_ciphertext_count,
      cuda_get_max_shared_memory(gpu_index));
  if (buffer_size > free)
    st.SkipWithError("Not enough free memory in the device. Skipping...");

  int8_t *pbs_buffer;
  scratch_cuda_bootstrap_amortized_64(
      stream, gpu_index, &pbs_buffer, glwe_dimension, polynomial_size,
      input_lwe_ciphertext_count, cuda_get_max_shared_memory(gpu_index), true);

  for (auto _ : st) {
    // Execute PBS
    cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_ct_out_array,
        (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
        (void *)d_lwe_ct_in_array, (void *)d_fourier_bsk_array, pbs_buffer,
        lwe_dimension, glwe_dimension, polynomial_size, pbs_base_log, pbs_level,
        input_lwe_ciphertext_count, input_lwe_ciphertext_count, 0,
        cuda_get_max_shared_memory(gpu_index));
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(input_lwe_ciphertext_count / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
  cleanup_cuda_bootstrap_amortized(stream, gpu_index, &pbs_buffer);
  cuda_synchronize_stream(stream);
}

BENCHMARK_DEFINE_F(Bootstrap_u64, ConcreteCuda_CopiesPlusAmortizedPBS)
(benchmark::State &st) {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  uint64_t buffer_size = get_buffer_size_bootstrap_amortized_64(
      glwe_dimension, polynomial_size, input_lwe_ciphertext_count,
      cuda_get_max_shared_memory(gpu_index));
  if (buffer_size > free)
    st.SkipWithError("Not enough free memory in the device. Skipping...");

  int8_t *pbs_buffer;
  scratch_cuda_bootstrap_amortized_64(
      stream, gpu_index, &pbs_buffer, glwe_dimension, polynomial_size,
      input_lwe_ciphertext_count, cuda_get_max_shared_memory(gpu_index), true);

  for (auto _ : st) {
    cuda_memcpy_async_to_gpu(d_lwe_ct_in_array, lwe_ct_array,
                             (lwe_dimension + 1) * input_lwe_ciphertext_count *
                                 sizeof(uint64_t),
                             stream, gpu_index);

    // Execute PBS
    cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_ct_out_array,
        (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
        (void *)d_lwe_ct_in_array, (void *)d_fourier_bsk_array, pbs_buffer,
        lwe_dimension, glwe_dimension, polynomial_size, pbs_base_log, pbs_level,
        input_lwe_ciphertext_count, input_lwe_ciphertext_count, 0,
        cuda_get_max_shared_memory(gpu_index));

    cuda_memcpy_async_to_cpu(lwe_ct_array, d_lwe_ct_out_array,
                             (lwe_dimension + 1) * input_lwe_ciphertext_count *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(input_lwe_ciphertext_count / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
  cleanup_cuda_bootstrap_amortized(stream, gpu_index, &pbs_buffer);
  cuda_synchronize_stream(stream);
}

BENCHMARK_DEFINE_F(Bootstrap_u64, ConcreteCuda_LowLatencyPBS)
(benchmark::State &st) {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  uint64_t buffer_size = get_buffer_size_bootstrap_low_latency_64(
      glwe_dimension, polynomial_size, pbs_level, input_lwe_ciphertext_count,
      cuda_get_max_shared_memory(gpu_index));

  if (buffer_size > free)
    st.SkipWithError("Not enough free memory in the device. Skipping...");
  if (!verify_cuda_bootstrap_low_latency_grid_size_64(
          glwe_dimension, polynomial_size, pbs_level,
          input_lwe_ciphertext_count, cuda_get_max_shared_memory(gpu_index)))
    st.SkipWithError(
        "Not enough SM on device to run this configuration. Skipping...");

  int8_t *pbs_buffer;
  scratch_cuda_bootstrap_low_latency_64(
      stream, gpu_index, &pbs_buffer, glwe_dimension, polynomial_size,
      pbs_level, input_lwe_ciphertext_count,
      cuda_get_max_shared_memory(gpu_index), true);

  for (auto _ : st) {
    // Execute PBS
    cuda_bootstrap_low_latency_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_ct_out_array,
        (void *)d_lut_pbs_identity, (void *)d_lut_pbs_indexes,
        (void *)d_lwe_ct_in_array, (void *)d_fourier_bsk_array, pbs_buffer,
        lwe_dimension, glwe_dimension, polynomial_size, pbs_base_log, pbs_level,
        input_lwe_ciphertext_count, 1, 0,
        cuda_get_max_shared_memory(gpu_index));
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(input_lwe_ciphertext_count / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
  cleanup_cuda_bootstrap_low_latency(stream, gpu_index, &pbs_buffer);
  cuda_synchronize_stream(stream);
}

static void
AmortizedBootstrapBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  // lwe_dimension, glwe_dimension, polynomial_size, pbs_base_log, pbs_level,
  // input_lwe_ciphertext_count
  std::vector<BootstrapBenchmarkParams> params = {
      // BOOLEAN_DEFAULT_PARAMETERS
      (BootstrapBenchmarkParams){777, 3, 512, 18, 1, 1000},
      // BOOLEAN_TFHE_LIB_PARAMETERS
      (BootstrapBenchmarkParams){830, 2, 1024, 23, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_0
      (BootstrapBenchmarkParams){678, 5, 256, 15, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_1
      (BootstrapBenchmarkParams){684, 3, 512, 18, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_2_CARRY_0
      (BootstrapBenchmarkParams){656, 2, 512, 8, 2, 1000},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_2
      // SHORTINT_PARAM_MESSAGE_2_CARRY_1
      // SHORTINT_PARAM_MESSAGE_3_CARRY_0
      (BootstrapBenchmarkParams){742, 2, 1024, 23, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_3
      // SHORTINT_PARAM_MESSAGE_2_CARRY_2
      // SHORTINT_PARAM_MESSAGE_3_CARRY_1
      // SHORTINT_PARAM_MESSAGE_4_CARRY_0
      (BootstrapBenchmarkParams){745, 1, 2048, 23, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_5_CARRY_0
      // SHORTINT_PARAM_MESSAGE_3_CARRY_2
      (BootstrapBenchmarkParams){807, 1, 4096, 22, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_6_CARRY_0
      (BootstrapBenchmarkParams){915, 1, 8192, 22, 1, 100},
      // SHORTINT_PARAM_MESSAGE_3_CARRY_3
      //(BootstrapBenchmarkParams){864, 1, 8192, 15, 2, 100},
      // SHORTINT_PARAM_MESSAGE_4_CARRY_3
      // SHORTINT_PARAM_MESSAGE_7_CARRY_0
      (BootstrapBenchmarkParams){930, 1, 16384, 15, 2, 100},
  };

  // Add to the list of parameters to benchmark
  for (auto x : params) {
    b->Args({x.lwe_dimension, x.glwe_dimension, x.polynomial_size,
             x.pbs_base_log, x.pbs_level, x.input_lwe_ciphertext_count});
  }
}


static void
LowLatencyBootstrapBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  // lwe_dimension, glwe_dimension, polynomial_size, pbs_base_log, pbs_level,
  // input_lwe_ciphertext_count
  std::vector<BootstrapBenchmarkParams> params = {
      // BOOLEAN_DEFAULT_PARAMETERS
      (BootstrapBenchmarkParams){777, 3, 512, 18, 1, 1},
      // BOOLEAN_TFHE_LIB_PARAMETERS
      (BootstrapBenchmarkParams){830, 2, 1024, 23, 1, 1},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_0
      (BootstrapBenchmarkParams){678, 5, 256, 15, 1, 1},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_1
      (BootstrapBenchmarkParams){684, 3, 512, 18, 1, 1},
      // SHORTINT_PARAM_MESSAGE_2_CARRY_0
      (BootstrapBenchmarkParams){656, 2, 512, 8, 2, 1},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_2
      // SHORTINT_PARAM_MESSAGE_2_CARRY_1
      // SHORTINT_PARAM_MESSAGE_3_CARRY_0
      (BootstrapBenchmarkParams){742, 2, 1024, 23, 1, 1},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_3
      // SHORTINT_PARAM_MESSAGE_2_CARRY_2
      // SHORTINT_PARAM_MESSAGE_3_CARRY_1
      // SHORTINT_PARAM_MESSAGE_4_CARRY_0
      (BootstrapBenchmarkParams){745, 1, 2048, 23, 1, 1},
      // SHORTINT_PARAM_MESSAGE_5_CARRY_0
      // SHORTINT_PARAM_MESSAGE_3_CARRY_2
      (BootstrapBenchmarkParams){807, 1, 4096, 22, 1, 1},
      // SHORTINT_PARAM_MESSAGE_6_CARRY_0
      (BootstrapBenchmarkParams){915, 1, 8192, 22, 1, 1},
      // SHORTINT_PARAM_MESSAGE_3_CARRY_3
      //(BootstrapBenchmarkParams){864, 1, 8192, 15, 2, 1},
      // SHORTINT_PARAM_MESSAGE_4_CARRY_3
      // SHORTINT_PARAM_MESSAGE_7_CARRY_0
      (BootstrapBenchmarkParams){930, 1, 16384, 15, 2, 1},
  };

  // Add to the list of parameters to benchmark
  for (auto x : params) {
    b->Args({x.lwe_dimension, x.glwe_dimension, x.polynomial_size,
             x.pbs_base_log, x.pbs_level, x.input_lwe_ciphertext_count});
  }
}

BENCHMARK_REGISTER_F(Bootstrap_u64, ConcreteCuda_AmortizedPBS)
    ->Apply(AmortizedBootstrapBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(Bootstrap_u64, ConcreteCuda_LowLatencyPBS)
    ->Apply(LowLatencyBootstrapBenchmarkGenerateParams);

BENCHMARK_REGISTER_F(Bootstrap_u64, ConcreteCuda_CopiesPlusAmortizedPBS)
    ->Apply(AmortizedBootstrapBenchmarkGenerateParams);
