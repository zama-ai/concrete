#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
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
  std::vector<double *> d_fourier_bsk_array;
  std::vector<uint64_t *> d_lut_pbs_identity;
  std::vector<uint64_t *> d_lut_pbs_indexes;
  std::vector<uint64_t *> d_lwe_ct_in_array;
  std::vector<uint64_t *> d_lwe_ct_out_array;
  uint64_t *lwe_ct_array;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;
  uint64_t *plaintexts;
  Csprng *csprng;
  std::vector<int8_t *> pbs_buffer;
  int num_gpus;
  std::vector<cudaStream_t *> streams;
  std::vector<int> input_lwe_ciphertext_count_per_gpu;

public:
  void SetUp(const ::benchmark::State &state) {
    lwe_dimension = state.range(0);
    glwe_dimension = state.range(1);
    polynomial_size = state.range(2);
    pbs_base_log = state.range(3);
    pbs_level = state.range(4);
    input_lwe_ciphertext_count = state.range(5);

    num_gpus = std::min(cuda_get_number_of_gpus(), input_lwe_ciphertext_count);

    for (int gpu_index = 0; gpu_index < num_gpus; gpu_index++) {
      cudaSetDevice(gpu_index);
      cudaStream_t *stream = cuda_create_stream(gpu_index);
      streams.push_back(stream);
      int input_lwe_ciphertext_count_on_gpu = number_of_inputs_on_gpu(
          gpu_index, input_lwe_ciphertext_count, num_gpus);

      double *d_fourier_bsk_array_per_gpu;
      uint64_t *d_lut_pbs_identity_per_gpu;
      uint64_t *d_lut_pbs_indexes_per_gpu;
      uint64_t *d_lwe_ct_in_array_per_gpu;
      uint64_t *d_lwe_ct_out_array_per_gpu;
      int8_t *pbs_buffer_per_gpu;

      bootstrap_classical_setup(
          stream, &csprng, &lwe_sk_in_array, &lwe_sk_out_array,
          &d_fourier_bsk_array_per_gpu, &plaintexts,
          &d_lut_pbs_identity_per_gpu, &d_lut_pbs_indexes_per_gpu,
          &d_lwe_ct_in_array_per_gpu, &d_lwe_ct_out_array_per_gpu,
          lwe_dimension, glwe_dimension, polynomial_size, lwe_modular_variance,
          glwe_modular_variance, pbs_base_log, pbs_level, message_modulus,
          carry_modulus, &payload_modulus, &delta,
          input_lwe_ciphertext_count_on_gpu, 1, 1, gpu_index);
      size_t free, total;
      cudaMemGetInfo(&free, &total);
      uint64_t buffer_size = get_buffer_size_bootstrap_low_latency_64(
          glwe_dimension, polynomial_size, pbs_level,
          input_lwe_ciphertext_count_on_gpu,
          cuda_get_max_shared_memory(gpu_index));

      assert(buffer_size > free);
      scratch_cuda_bootstrap_low_latency_64(
          stream, gpu_index, &pbs_buffer_per_gpu, glwe_dimension,
          polynomial_size, pbs_level, input_lwe_ciphertext_count_on_gpu,
          cuda_get_max_shared_memory(gpu_index), true);

      d_fourier_bsk_array.push_back(d_fourier_bsk_array_per_gpu);
      d_lut_pbs_identity.push_back(d_lut_pbs_identity_per_gpu);
      d_lut_pbs_indexes.push_back(d_lut_pbs_indexes_per_gpu);
      d_lwe_ct_in_array.push_back(d_lwe_ct_in_array_per_gpu);
      d_lwe_ct_out_array.push_back(d_lwe_ct_out_array_per_gpu);
      pbs_buffer.push_back(pbs_buffer_per_gpu);
      input_lwe_ciphertext_count_per_gpu.push_back(
          input_lwe_ciphertext_count_on_gpu);
    }

    // We keep the following for the benchmarks with copies
    lwe_ct_array = (uint64_t *)malloc(
        (lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint64_t));
  }

  void TearDown(const ::benchmark::State &state) {
    concrete_cpu_destroy_concrete_csprng(csprng);
    free(csprng);
    free(lwe_sk_in_array);
    free(lwe_sk_out_array);
    free(plaintexts);

    for (int gpu_index = 0; gpu_index < num_gpus; gpu_index++) {
      cudaSetDevice(gpu_index);
      cleanup_cuda_bootstrap_low_latency(streams[gpu_index], gpu_index,
                                         &pbs_buffer[gpu_index]);
      cuda_drop_async(d_fourier_bsk_array[gpu_index], streams[gpu_index],
                      gpu_index);
      cuda_drop_async(d_lut_pbs_identity[gpu_index], streams[gpu_index],
                      gpu_index);
      cuda_drop_async(d_lut_pbs_indexes[gpu_index], streams[gpu_index],
                      gpu_index);
      cuda_drop_async(d_lwe_ct_in_array[gpu_index], streams[gpu_index],
                      gpu_index);
      cuda_drop_async(d_lwe_ct_out_array[gpu_index], streams[gpu_index],
                      gpu_index);
      cuda_synchronize_stream(streams[gpu_index]);
      cuda_destroy_stream(streams[gpu_index], gpu_index);
    }
    d_fourier_bsk_array.clear();
    d_lut_pbs_identity.clear();
    d_lut_pbs_indexes.clear();
    d_lwe_ct_in_array.clear();
    d_lwe_ct_out_array.clear();
    pbs_buffer.clear();
    input_lwe_ciphertext_count_per_gpu.clear();
    streams.clear();
    cudaDeviceReset();
  }
};

BENCHMARK_DEFINE_F(Bootstrap_u64, ConcreteCuda_LowLatencyPBS)
(benchmark::State &st) {

  for (auto _ : st) {
#pragma omp parallel for
    for (int gpu_index = 0; gpu_index < num_gpus; gpu_index++) {
      // Execute PBS
      cuda_bootstrap_low_latency_lwe_ciphertext_vector_64(
          streams[gpu_index], gpu_index, (void *)d_lwe_ct_out_array[gpu_index],
          (void *)d_lut_pbs_identity[gpu_index],
          (void *)d_lut_pbs_indexes[gpu_index],
          (void *)d_lwe_ct_in_array[gpu_index],
          (void *)d_fourier_bsk_array[gpu_index], pbs_buffer[gpu_index],
          lwe_dimension, glwe_dimension, polynomial_size, pbs_base_log,
          pbs_level, input_lwe_ciphertext_count_per_gpu[gpu_index], 1, 0,
          cuda_get_max_shared_memory(gpu_index));
    }
    for (int gpu_index = 0; gpu_index < num_gpus; gpu_index++) {
      cudaSetDevice(gpu_index);
      cuda_synchronize_stream(streams[gpu_index]);
    }
  }
  st.counters["Throughput"] =
      benchmark::Counter(input_lwe_ciphertext_count / get_aws_cost_per_second(),
                         benchmark::Counter::kIsIterationInvariantRate);
}

static void
BootstrapBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  // lwe_dimension, glwe_dimension, polynomial_size, pbs_base_log, pbs_level,
  // input_lwe_ciphertext_count
  std::vector<BootstrapBenchmarkParams> params = {
      // BOOLEAN_DEFAULT_PARAMETERS
      (BootstrapBenchmarkParams){777, 3, 512, 18, 1, 1},
      (BootstrapBenchmarkParams){777, 3, 512, 18, 1, 1000},
      // BOOLEAN_TFHE_LIB_PARAMETERS
      (BootstrapBenchmarkParams){830, 2, 1024, 23, 1, 1},
      (BootstrapBenchmarkParams){830, 2, 1024, 23, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_0
      (BootstrapBenchmarkParams){678, 5, 256, 15, 1, 1},
      (BootstrapBenchmarkParams){678, 5, 256, 15, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_1
      (BootstrapBenchmarkParams){684, 3, 512, 18, 1, 1},
      (BootstrapBenchmarkParams){684, 3, 512, 18, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_2_CARRY_0
      (BootstrapBenchmarkParams){656, 2, 512, 8, 2, 1},
      (BootstrapBenchmarkParams){656, 2, 512, 8, 2, 1000},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_2
      // SHORTINT_PARAM_MESSAGE_2_CARRY_1
      // SHORTINT_PARAM_MESSAGE_3_CARRY_0
      (BootstrapBenchmarkParams){742, 2, 1024, 23, 1, 1},
      (BootstrapBenchmarkParams){742, 2, 1024, 23, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_1_CARRY_3
      // SHORTINT_PARAM_MESSAGE_2_CARRY_2
      // SHORTINT_PARAM_MESSAGE_3_CARRY_1
      // SHORTINT_PARAM_MESSAGE_4_CARRY_0
      (BootstrapBenchmarkParams){745, 1, 2048, 23, 1, 1},
      (BootstrapBenchmarkParams){745, 1, 2048, 23, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_5_CARRY_0
      // SHORTINT_PARAM_MESSAGE_3_CARRY_2
      (BootstrapBenchmarkParams){807, 1, 4096, 22, 1, 1},
      (BootstrapBenchmarkParams){807, 1, 4096, 22, 1, 1000},
      // SHORTINT_PARAM_MESSAGE_6_CARRY_0
      (BootstrapBenchmarkParams){915, 1, 8192, 22, 1, 1},
      (BootstrapBenchmarkParams){915, 1, 8192, 22, 1, 100},
      // SHORTINT_PARAM_MESSAGE_3_CARRY_3
      //(BootstrapBenchmarkParams){864, 1, 8192, 15, 2, 100},
      // SHORTINT_PARAM_MESSAGE_4_CARRY_3
      // SHORTINT_PARAM_MESSAGE_7_CARRY_0
      (BootstrapBenchmarkParams){930, 1, 16384, 15, 2, 1},
      (BootstrapBenchmarkParams){930, 1, 16384, 15, 2, 100},
  };

  // Add to the list of parameters to benchmark
  for (auto x : params) {
    b->Args({x.lwe_dimension, x.glwe_dimension, x.polynomial_size,
             x.pbs_base_log, x.pbs_level, x.input_lwe_ciphertext_count});
  }
}

BENCHMARK_REGISTER_F(Bootstrap_u64, ConcreteCuda_LowLatencyPBS)
    ->Apply(BootstrapBenchmarkGenerateParams);
