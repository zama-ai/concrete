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
  int ks_base_log;
  int ks_level;
  int number_of_bits_of_message_including_padding;
  int number_of_bits_to_extract;
  int number_of_inputs;
} BitExtractionBenchmarkParams;

class BitExtraction_u64 : public benchmark::Fixture {
protected:
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  double lwe_modular_variance = 7.52316384526264e-37;
  double glwe_modular_variance = 7.52316384526264e-37;
  int pbs_base_log;
  int pbs_level;
  int ks_base_log;
  int ks_level;
  int number_of_bits_of_message_including_padding;
  int number_of_bits_to_extract;
  int number_of_inputs;
  uint64_t delta;
  int delta_log;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *plaintexts;
  double *d_fourier_bsk;
  uint64_t *d_ksk;
  uint64_t *d_lwe_ct_in_array;
  uint64_t *d_lwe_ct_out_array;
  int8_t *bit_extract_buffer;
  uint64_t *lwe_sk_in;
  uint64_t *lwe_sk_out;

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
    ks_base_log = state.range(5);
    ks_level = state.range(6);
    number_of_bits_of_message_including_padding = state.range(7);
    number_of_bits_to_extract = state.range(8);
    number_of_inputs = state.range(9);

    bit_extraction_setup(
        stream, &csprng, &lwe_sk_in, &lwe_sk_out, &d_fourier_bsk, &d_ksk,
        &plaintexts, &d_lwe_ct_in_array, &d_lwe_ct_out_array,
        &bit_extract_buffer, lwe_dimension, glwe_dimension, polynomial_size,
        lwe_modular_variance, glwe_modular_variance, ks_base_log, ks_level,
        pbs_base_log, pbs_level, number_of_bits_of_message_including_padding,
        number_of_bits_to_extract, &delta_log, &delta, number_of_inputs, 1, 1,
        gpu_index);
  }

  void TearDown(const ::benchmark::State &state) {
    bit_extraction_teardown(stream, csprng, lwe_sk_in, lwe_sk_out,
                            d_fourier_bsk, d_ksk, plaintexts, d_lwe_ct_in_array,
                            d_lwe_ct_out_array, bit_extract_buffer, gpu_index);
  }
};

BENCHMARK_DEFINE_F(BitExtraction_u64, ConcreteCuda_BitExtraction)
(benchmark::State &st) {
  for (auto _ : st) {
    // Execute bit extract
    cuda_extract_bits_64(
        stream, gpu_index, (void *)d_lwe_ct_out_array,
        (void *)d_lwe_ct_in_array, bit_extract_buffer, (void *)d_ksk,
        (void *)d_fourier_bsk, number_of_bits_to_extract, delta_log,
        glwe_dimension * polynomial_size, lwe_dimension, glwe_dimension,
        polynomial_size, pbs_base_log, pbs_level, ks_base_log, ks_level,
        number_of_inputs, cuda_get_max_shared_memory(gpu_index));
    cuda_synchronize_stream(stream);
  }
}

static void
BitExtractionBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  std::vector<BitExtractionBenchmarkParams> params = {
      (BitExtractionBenchmarkParams){585, 1, 1024, 10, 2, 4, 7, 5, 5, 1} //,
  };

  // Add to the list of parameters to benchmark
  for (auto x : params)
    b->Args({x.lwe_dimension, x.glwe_dimension, x.polynomial_size,
             x.pbs_base_log, x.pbs_level, x.ks_base_log, x.ks_level,
             x.number_of_bits_of_message_including_padding,
             x.number_of_bits_to_extract, x.number_of_inputs});
}

BENCHMARK_REGISTER_F(BitExtraction_u64, ConcreteCuda_BitExtraction)
    ->Apply(BitExtractionBenchmarkGenerateParams);
