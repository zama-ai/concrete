#include <benchmark/benchmark.h>
#include <cstdint>
#include <setup_and_teardown.h>
#include <stdio.h>
#include <stdlib.h>

const unsigned MAX_TAU = 4;

typedef struct {
  int lwe_dimension;
  int glwe_dimension;
  int polynomial_size;
  int pbs_base_log;
  int pbs_level;
  int ks_base_log;
  int ks_level;
  int pksk_base_log;
  int pksk_level;
  int cbs_base_log;
  int cbs_level;
  int tau;
  int p;
} WopPBSBenchmarkParams;

class WopPBS_u64 : public benchmark::Fixture {
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
  int pksk_base_log;
  int pksk_level;
  int cbs_base_log;
  int cbs_level;
  int tau;
  uint32_t p_array[MAX_TAU];
  int input_lwe_dimension;
  uint64_t delta_array[MAX_TAU];
  int cbs_delta_log;
  uint32_t delta_log_array[MAX_TAU];
  int delta_log_lut;
  Csprng *csprng;
  cudaStream_t *stream;
  int gpu_index = 0;
  uint64_t *plaintexts;
  double *d_fourier_bsk;
  uint64_t *lwe_sk_in;
  uint64_t *lwe_sk_out;
  uint64_t *d_ksk;
  uint64_t *d_pksk;
  uint64_t *d_lwe_ct_in_array;
  uint64_t *d_lwe_ct_out_array;
  uint64_t *d_lut_vector;
  int8_t *wop_pbs_buffer;
  uint64_t *lwe_ct_in_array;
  uint64_t *lwe_ct_out_array;

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
    pksk_base_log = state.range(7);
    pksk_level = state.range(8);
    cbs_base_log = state.range(9);
    cbs_level = state.range(10);
    tau = state.range(11);
    p_array[0] = state.range(12);
    wop_pbs_setup(stream, &csprng, &lwe_sk_in, &lwe_sk_out, &d_ksk,
                  &d_fourier_bsk, &d_pksk, &plaintexts, &d_lwe_ct_in_array,
                  &d_lwe_ct_out_array, &d_lut_vector, &wop_pbs_buffer,
                  lwe_dimension, glwe_dimension, polynomial_size,
                  lwe_modular_variance, glwe_modular_variance, ks_base_log,
                  ks_level, pksk_base_log, pksk_level, pbs_base_log, pbs_level,
                  cbs_level, p_array, delta_log_array, &cbs_delta_log,
                  delta_array, tau, 1, 1, gpu_index);

    // We keep the following for the benchmarks with copies
    lwe_ct_in_array = (uint64_t *)malloc(
        (glwe_dimension * polynomial_size + 1) * tau * sizeof(uint64_t));
    lwe_ct_out_array = (uint64_t *)malloc(
        (glwe_dimension * polynomial_size + 1) * tau * sizeof(uint64_t));
    for (int i = 0; i < tau; i++) {
      uint64_t plaintext = plaintexts[i];
      uint64_t *lwe_ct_in =
          lwe_ct_in_array +
          (ptrdiff_t)(i * (glwe_dimension * polynomial_size + 1));
      concrete_cpu_encrypt_lwe_ciphertext_u64(
          lwe_sk_in, lwe_ct_in, plaintext, glwe_dimension * polynomial_size,
          lwe_modular_variance, csprng, &CONCRETE_CSPRNG_VTABLE);
    }
  }

  void TearDown(const ::benchmark::State &state) {
    wop_pbs_teardown(stream, csprng, lwe_sk_in, lwe_sk_out, d_ksk,
                     d_fourier_bsk, d_pksk, plaintexts, d_lwe_ct_in_array,
                     d_lut_vector, d_lwe_ct_out_array, wop_pbs_buffer,
                     gpu_index);
    free(lwe_ct_in_array);
    free(lwe_ct_out_array);
  }
};

BENCHMARK_DEFINE_F(WopPBS_u64, ConcreteCuda_WopPBS)(benchmark::State &st) {
  for (auto _ : st) {
    // Execute wop pbs
    cuda_wop_pbs_64(stream, gpu_index, (void *)d_lwe_ct_out_array,
                    (void *)d_lwe_ct_in_array, (void *)d_lut_vector,
                    (void *)d_fourier_bsk, (void *)d_ksk, (void *)d_pksk,
                    wop_pbs_buffer, cbs_delta_log, glwe_dimension,
                    lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
                    ks_base_log, ks_level, pksk_base_log, pksk_level,
                    cbs_base_log, cbs_level, p_array, p_array, delta_log_array,
                    tau, cuda_get_max_shared_memory(gpu_index));
    cuda_synchronize_stream(stream);
  }
}

BENCHMARK_DEFINE_F(WopPBS_u64, ConcreteCuda_CopiesPlusWopPBS)
(benchmark::State &st) {
  for (auto _ : st) {
    cuda_memcpy_async_to_gpu(d_lwe_ct_in_array, lwe_ct_in_array,
                             (input_lwe_dimension + 1) * tau * sizeof(uint64_t),
                             stream, gpu_index);
    // Execute wop pbs
    cuda_wop_pbs_64(stream, gpu_index, (void *)d_lwe_ct_out_array,
                    (void *)d_lwe_ct_in_array, (void *)d_lut_vector,
                    (void *)d_fourier_bsk, (void *)d_ksk, (void *)d_pksk,
                    wop_pbs_buffer, cbs_delta_log, glwe_dimension,
                    lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
                    ks_base_log, ks_level, pksk_base_log, pksk_level,
                    cbs_base_log, cbs_level, p_array, p_array, delta_log_array,
                    tau, cuda_get_max_shared_memory(gpu_index));

    cuda_memcpy_async_to_cpu(lwe_ct_out_array, d_lwe_ct_out_array,
                             (input_lwe_dimension + 1) * tau * sizeof(uint64_t),
                             stream, gpu_index);
    cuda_synchronize_stream(stream);
  }
}

static void WopPBSBenchmarkGenerateParams(benchmark::internal::Benchmark *b) {
  // Define the parameters to benchmark
  // n, k, N, lwe_variance, glwe_variance, pbs_base_log, pbs_level,
  // ks_base_log, ks_level, tau, p
  std::vector<WopPBSBenchmarkParams> params = {
      (WopPBSBenchmarkParams){481, 2, 512, 4, 9, 1, 9, 4, 9, 6, 4, 1, 10},
      //// INTEGER_PARAM_MESSAGE_4_CARRY_4_16_BITS
      //(WopPBSBenchmarkParams){481, 1, 2048, 9, 4, 1, 9, 9, 4, 6, 4, 1, 8},
      //// INTEGER_PARAM_MESSAGE_2_CARRY_2_16_BITS
      //(WopPBSBenchmarkParams){493, 1, 2048, 16, 2, 2, 5, 16, 2, 6, 4, 1, 4},
  };

  // Add to the list of parameters to benchmark
  for (auto x : params)
    b->Args({x.lwe_dimension, x.glwe_dimension, x.polynomial_size,
             x.pbs_base_log, x.pbs_level, x.ks_base_log, x.ks_level,
             x.pksk_base_log, x.pksk_level, x.cbs_base_log, x.cbs_level, x.tau,
             x.p});
}

BENCHMARK_REGISTER_F(WopPBS_u64, ConcreteCuda_WopPBS)
    ->Apply(WopPBSBenchmarkGenerateParams);
BENCHMARK_REGISTER_F(WopPBS_u64, ConcreteCuda_CopiesPlusWopPBS)
    ->Apply(WopPBSBenchmarkGenerateParams);
