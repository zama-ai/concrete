#include <benchmark/benchmark.h>
#include <cstdint>
#include <setup_and_teardown.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int input_lwe_dimension;
  int output_lwe_dimension;
  int ksk_base_log;
  int ksk_level;
  int number_of_inputs;
} KeyswitchBenchmarkParams;

class Keyswitch_u64 : public benchmark::Fixture {
protected:
  int input_lwe_dimension;
  int output_lwe_dimension;
  double noise_variance = 2.9802322387695312e-08;
  int ksk_base_log;
  int ksk_level;
  int message_modulus = 4;
  int carry_modulus = 4;
  int payload_modulus;
  int number_of_inputs;
  uint64_t delta;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *plaintexts;
  uint64_t *d_ksk_array;
  uint64_t *d_lwe_out_ct_array;
  uint64_t *d_lwe_in_ct_array;
  uint64_t *lwe_sk_in_array;
  uint64_t *lwe_sk_out_array;

public:
  // Test arithmetic functions
  void SetUp(const ::benchmark::State &state) {
    stream = cuda_create_stream(0);

    // TestParams
    input_lwe_dimension = state.range(0);
    output_lwe_dimension = state.range(1);
    ksk_base_log = state.range(2);
    ksk_level = state.range(3);
    number_of_inputs = state.range(4);

    keyswitch_setup(stream, &csprng, &lwe_sk_in_array, &lwe_sk_out_array,
                    &d_ksk_array, &plaintexts, &d_lwe_in_ct_array,
                    &d_lwe_out_ct_array, input_lwe_dimension,
                    output_lwe_dimension, noise_variance, ksk_base_log,
                    ksk_level, message_modulus, carry_modulus, &payload_modulus,
                    &delta, number_of_inputs, 1, 1, gpu_index);
  }

  void TearDown(const ::benchmark::State &state) {
    keyswitch_teardown(stream, csprng, lwe_sk_in_array, lwe_sk_out_array,
                       d_ksk_array, plaintexts, d_lwe_in_ct_array,
                       d_lwe_out_ct_array, gpu_index);
  }
};

BENCHMARK_DEFINE_F(Keyswitch_u64, ConcreteCuda_Keyswitch)
(benchmark::State &st) {
  for (auto _ : st) {
    // Execute keyswitch
    cuda_keyswitch_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct_array,
        (void *)d_lwe_in_ct_array, (void *)d_ksk_array, input_lwe_dimension,
        output_lwe_dimension, ksk_base_log, ksk_level, number_of_inputs);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] =
      benchmark::Counter(number_of_inputs / get_aws_cost_per_second(),
                         benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(Keyswitch_u64, ConcreteCuda_CopiesPlusKeyswitch)
(benchmark::State &st) {
  uint64_t *lwe_in_ct = (uint64_t *)malloc(
      number_of_inputs * (input_lwe_dimension + 1) * sizeof(uint64_t));
  uint64_t *lwe_out_ct = (uint64_t *)malloc(
      number_of_inputs * (output_lwe_dimension + 1) * sizeof(uint64_t));
  void *v_stream = (void *)stream;
  for (auto _ : st) {
    cuda_memcpy_async_to_gpu(d_lwe_in_ct_array, lwe_in_ct,
                             number_of_inputs * (input_lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    // Execute keyswitch
    cuda_keyswitch_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct_array,
        (void *)d_lwe_in_ct_array, (void *)d_ksk_array, input_lwe_dimension,
        output_lwe_dimension, ksk_base_log, ksk_level, number_of_inputs);
    cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct_array,
                             number_of_inputs * (output_lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(v_stream);
  }
  st.counters["Throughput"] =
      benchmark::Counter(number_of_inputs / get_aws_cost_per_second(),
                         benchmark::Counter::kIsIterationInvariantRate);
  free(lwe_in_ct);
  free(lwe_out_ct);
}

static void
KeyswitchBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  // na, nb, base_log, level, number_of_inputs
  std::vector<KeyswitchBenchmarkParams> params = {
      (KeyswitchBenchmarkParams){600, 1024, 3, 8, 1000},
  };

  // Add to the list of parameters to benchmark
  for (auto x : params)
    b->Args({x.input_lwe_dimension, x.output_lwe_dimension, x.ksk_base_log,
             x.ksk_level, x.number_of_inputs});
}

BENCHMARK_REGISTER_F(Keyswitch_u64, ConcreteCuda_Keyswitch)
    ->Apply(KeyswitchBenchmarkGenerateParams);

BENCHMARK_REGISTER_F(Keyswitch_u64, ConcreteCuda_CopiesPlusKeyswitch)
    ->Apply(KeyswitchBenchmarkGenerateParams);
