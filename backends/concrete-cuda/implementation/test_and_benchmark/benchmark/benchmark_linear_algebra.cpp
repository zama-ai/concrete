#include <benchmark/benchmark.h>
#include <cstdint>
#include <setup_and_teardown.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int lwe_dimension;
  int input_lwe_ciphertext_count;
} LinearAlgebraBenchmarkParams;

class LinearAlgebra_u64 : public benchmark::Fixture {
protected:
  int lwe_dimension;
  double noise_variance = 2.9802322387695312e-08;
  int ksk_base_log;
  int ksk_level;
  int message_modulus = 4;
  int carry_modulus = 4;
  int num_samples;
  uint64_t delta;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *d_lwe_in_1_ct;
  uint64_t *d_lwe_in_2_ct;
  uint64_t *d_lwe_out_ct;
  uint64_t *plaintexts_1;
  uint64_t *plaintexts_2;
  uint64_t *d_plaintext_2;
  uint64_t *d_cleartext;
  uint64_t *lwe_in_1_ct;
  uint64_t *lwe_in_2_ct;
  uint64_t *lwe_out_ct;
  uint64_t *lwe_sk_array;

public:
  // Test arithmetic functions
  void SetUp(const ::benchmark::State &state) {
    stream = cuda_create_stream(0);

    // TestParams
    lwe_dimension = state.range(0);
    num_samples = state.range(1);

    int payload_modulus = message_modulus * carry_modulus;
    // Value of the shift we multiply our messages by
    delta = ((uint64_t)(1) << 63) / (uint64_t)(payload_modulus);

    linear_algebra_setup(
        stream, &csprng, &lwe_sk_array, &d_lwe_in_1_ct, &d_lwe_in_2_ct,
        &d_lwe_out_ct, &lwe_in_1_ct, &lwe_in_2_ct, &lwe_out_ct, &plaintexts_1,
        &plaintexts_2, &d_plaintext_2, &d_cleartext, lwe_dimension,
        noise_variance, payload_modulus, delta, num_samples, 1, 1, gpu_index);
  }

  void TearDown(const ::benchmark::State &state) {
    linear_algebra_teardown(
        stream, &csprng, &lwe_sk_array, &d_lwe_in_1_ct, &d_lwe_in_2_ct,
        &d_lwe_out_ct, &lwe_in_1_ct, &lwe_in_2_ct, &lwe_out_ct, &plaintexts_1,
        &plaintexts_2, &d_plaintext_2, &d_cleartext, gpu_index);
  }
};

BENCHMARK_DEFINE_F(LinearAlgebra_u64, ConcreteCuda_Addition)
(benchmark::State &st) {
  // Execute addition
  for (auto _ : st) {
    cuda_add_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
        (void *)d_lwe_in_2_ct, lwe_dimension, num_samples);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(num_samples / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(LinearAlgebra_u64, ConcreteCuda_CopiesPlusAddition)
(benchmark::State &st) {
  // Execute addition
  for (auto _ : st) {

    cuda_memcpy_async_to_gpu(d_lwe_in_1_ct, lwe_in_1_ct,
                             num_samples * (lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_lwe_in_2_ct, lwe_in_2_ct,
                             num_samples * (lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_add_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
        (void *)d_lwe_in_2_ct, lwe_dimension, num_samples);

    cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                             num_samples * (lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(num_samples / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(LinearAlgebra_u64, ConcreteCuda_PlaintextAddition)
(benchmark::State &st) {
  for (auto _ : st) {
    // Execute addition
    cuda_add_lwe_ciphertext_vector_plaintext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
        (void *)d_plaintext_2, lwe_dimension, num_samples);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(num_samples / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(LinearAlgebra_u64, ConcreteCuda_CopiesPlusPlaintextAddition)
(benchmark::State &st) {
  for (auto _ : st) {

    cuda_memcpy_async_to_gpu(d_lwe_in_1_ct, lwe_in_1_ct,
                             num_samples * (lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_plaintext_2, plaintexts_2,
                             num_samples * sizeof(uint64_t), stream, gpu_index);
    // Execute addition
    cuda_add_lwe_ciphertext_vector_plaintext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
        (void *)d_plaintext_2, lwe_dimension, num_samples);

    cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                             num_samples * (lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(num_samples / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(LinearAlgebra_u64, ConcreteCuda_CleartextMultiplication)
(benchmark::State &st) {
  for (auto _ : st) {
    // Execute addition
    cuda_mult_lwe_ciphertext_vector_cleartext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
        (void *)d_cleartext, lwe_dimension, num_samples);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(num_samples / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(LinearAlgebra_u64,
                   ConcreteCuda_CopiesPlusCleartextMultiplication)
(benchmark::State &st) {
  for (auto _ : st) {
    cuda_memcpy_async_to_gpu(d_lwe_in_1_ct, lwe_in_1_ct,
                             num_samples * (lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_cleartext, plaintexts_2,
                             num_samples * sizeof(uint64_t), stream, gpu_index);
    // Execute addition
    cuda_mult_lwe_ciphertext_vector_cleartext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
        (void *)d_cleartext, lwe_dimension, num_samples);

    cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                             num_samples * (lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(num_samples / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(LinearAlgebra_u64, ConcreteCuda_Negation)
(benchmark::State &st) {
  for (auto _ : st) {
    // Execute addition
    cuda_negate_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
        lwe_dimension, num_samples);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(num_samples / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK_DEFINE_F(LinearAlgebra_u64, ConcreteCuda_CopiesPlusNegation)
(benchmark::State &st) {
  for (auto _ : st) {
    cuda_memcpy_async_to_gpu(d_lwe_in_1_ct, lwe_in_1_ct,
                             num_samples * (lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    // Execute addition
    cuda_negate_lwe_ciphertext_vector_64(
        stream, gpu_index, (void *)d_lwe_out_ct, (void *)d_lwe_in_1_ct,
        lwe_dimension, num_samples);

    cuda_memcpy_async_to_cpu(lwe_out_ct, d_lwe_out_ct,
                             num_samples * (lwe_dimension + 1) *
                                 sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(stream);
  }
  st.counters["Throughput"] = benchmark::Counter(num_samples / get_aws_cost_per_second(),
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

static void
LinearAlgebraBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  // n, input_lwe_ciphertext_count
  std::vector<LinearAlgebraBenchmarkParams> params = {
      (LinearAlgebraBenchmarkParams){600, 100},
  };

  // Add to the list of parameters to benchmark
  for (auto x : params)
    b->Args({x.lwe_dimension, x.input_lwe_ciphertext_count});
}

BENCHMARK_REGISTER_F(LinearAlgebra_u64, ConcreteCuda_Addition)
    ->Apply(LinearAlgebraBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(LinearAlgebra_u64, ConcreteCuda_CopiesPlusAddition)
    ->Apply(LinearAlgebraBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(LinearAlgebra_u64, ConcreteCuda_PlaintextAddition)
    ->Apply(LinearAlgebraBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(LinearAlgebra_u64,
                     ConcreteCuda_CopiesPlusPlaintextAddition)
    ->Apply(LinearAlgebraBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(LinearAlgebra_u64, ConcreteCuda_CleartextMultiplication)
    ->Apply(LinearAlgebraBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(LinearAlgebra_u64,
                     ConcreteCuda_CopiesPlusCleartextMultiplication)
    ->Apply(LinearAlgebraBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(LinearAlgebra_u64, ConcreteCuda_Negation)
    ->Apply(LinearAlgebraBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(LinearAlgebra_u64, ConcreteCuda_CopiesPlusNegation)
    ->Apply(LinearAlgebraBenchmarkGenerateParams);
