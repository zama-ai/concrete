#ifndef CNCRT_KS_H
#define CNCRT_KS_H

#include "crypto/gadget.cuh"
#include "crypto/torus.cuh"
#include "polynomial/polynomial.cuh"
#include <thread>
#include <vector>

template <typename Torus>
__device__ Torus *get_ith_block(Torus *ksk, int i, int level,
                                uint32_t lwe_dimension_out,
                                uint32_t level_count) {
  int pos = i * level_count * (lwe_dimension_out + 1) +
            level * (lwe_dimension_out + 1);
  Torus *ptr = &ksk[pos];
  return ptr;
}

// blockIdx.y represents single lwe ciphertext
// blockIdx.x represents chunk of lwe ciphertext,
// chunk_count = glwe_size * polynomial_size / threads.
// each threads will responsible to process only lwe_size times multiplication
template <typename Torus>
__global__ void
fp_keyswitch(Torus *glwe_array_out, Torus *lwe_array_in, Torus *fp_ksk_array,
             uint32_t lwe_dimension_in, uint32_t glwe_dimension,
             uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
             uint32_t number_of_input_lwe, uint32_t number_of_keys) {
  size_t tid = threadIdx.x;

  size_t glwe_size = (glwe_dimension + 1);
  size_t lwe_size = (lwe_dimension_in + 1);

  // number of coefficients in a single fp-ksk
  size_t ksk_size = lwe_size * level_count * glwe_size * polynomial_size;

  // number of coefficients inside fp-ksk block for each lwe_input coefficient
  size_t ksk_block_size = glwe_size * polynomial_size * level_count;

  size_t ciphertext_id = blockIdx.y;
  // number of coefficients processed inside single block
  size_t coef_per_block = blockDim.x;
  size_t chunk_id = blockIdx.x;
  size_t ksk_id = ciphertext_id % number_of_keys;

  extern __shared__ char sharedmem[];

  // result accumulator, shared memory is used because of frequent access
  Torus *local_glwe_chunk = (Torus *)sharedmem;

  // current input lwe ciphertext
  auto cur_input_lwe = &lwe_array_in[ciphertext_id * lwe_size];
  // current output glwe ciphertext
  auto cur_output_glwe =
      &glwe_array_out[ciphertext_id * glwe_size * polynomial_size];
  // current out glwe chunk, will be processed inside single block
  auto cur_glwe_chunk = &cur_output_glwe[chunk_id * coef_per_block];

  // fp key used for current ciphertext
  auto cur_ksk = &fp_ksk_array[ksk_id * ksk_size];

  // set shared mem accumulator to 0
  local_glwe_chunk[tid] = 0;

  // iterate through each coefficient of  input lwe
  for (size_t i = 0; i <= lwe_dimension_in; i++) {
    Torus a_i =
        round_to_closest_multiple(cur_input_lwe[i], base_log, level_count);

    Torus state = a_i >> (sizeof(Torus) * 8 - base_log * level_count);
    Torus mod_b_mask = (1ll << base_log) - 1ll;

    // block of key for current lwe coefficient (cur_input_lwe[i])
    auto ksk_block = &cur_ksk[i * ksk_block_size];

    // iterate through levels, calculating decomposition in reverse order
    for (size_t j = 0; j < level_count; j++) {
      auto ksk_glwe =
          &ksk_block[(level_count - j - 1) * glwe_size * polynomial_size];
      auto ksk_glwe_chunk = &ksk_glwe[chunk_id * coef_per_block];
      Torus decomposed = decompose_one<Torus>(state, mod_b_mask, base_log);
      local_glwe_chunk[tid] -= decomposed * ksk_glwe_chunk[tid];
    }
  }
  cur_glwe_chunk[tid] = local_glwe_chunk[tid];
}

/*
 * keyswitch kernel
 * Each thread handles a piece of the following equation:
 * $$GLWE_s2(\Delta.m+e) = (0,0,..,0,b) - \sum_{i=0,k-1} <Dec(a_i),
 * (GLWE_s2(s1_i q/beta),..,GLWE(s1_i q/beta^l)>$$ where k is the dimension of
 * the GLWE ciphertext. If the polynomial dimension in GLWE is > 1, this
 * equation is solved for each polynomial coefficient. where Dec denotes the
 * decomposition with base beta and l levels and the inner product is done
 * between the decomposition of a_i and l GLWE encryptions of s1_i q/\beta^j,
 * with j in [1,l] We obtain a GLWE encryption of Delta.m (with Delta the
 * scaling factor) under key s2 instead of s1, with an increased noise
 *
 */
template <typename Torus>
__global__ void keyswitch(Torus *lwe_array_out, Torus *lwe_array_in, Torus *ksk,
                          uint32_t lwe_dimension_in, uint32_t lwe_dimension_out,
                          uint32_t base_log, uint32_t level_count,
                          int lwe_lower, int lwe_upper, int cutoff) {
  int tid = threadIdx.x;

  extern __shared__ char sharedmem[];

  Torus *local_lwe_array_out = (Torus *)sharedmem;

  auto block_lwe_array_in =
      get_chunk(lwe_array_in, blockIdx.x, lwe_dimension_in + 1);
  auto block_lwe_array_out =
      get_chunk(lwe_array_out, blockIdx.x, lwe_dimension_out + 1);

  auto gadget = GadgetMatrixSingle<Torus>(base_log, level_count);

  int lwe_part_per_thd;
  if (tid < cutoff) {
    lwe_part_per_thd = lwe_upper;
  } else {
    lwe_part_per_thd = lwe_lower;
  }
  __syncthreads();

  for (int k = 0; k < lwe_part_per_thd; k++) {
    int idx = tid + k * blockDim.x;
    local_lwe_array_out[idx] = 0;
  }

  if (tid == 0) {
    local_lwe_array_out[lwe_dimension_out] =
        block_lwe_array_in[lwe_dimension_in];
  }

  for (int i = 0; i < lwe_dimension_in; i++) {

    __syncthreads();

    Torus a_i =
        round_to_closest_multiple(block_lwe_array_in[i], base_log, level_count);

    Torus state = a_i >> (sizeof(Torus) * 8 - base_log * level_count);
    Torus mask_mod_b = (1ll << base_log) - 1ll;

    for (int j = 0; j < level_count; j++) {
      auto ksk_block = get_ith_block(ksk, i, level_count - j - 1,
                                     lwe_dimension_out, level_count);
      Torus decomposed = decompose_one<Torus>(state, mask_mod_b, base_log);
      for (int k = 0; k < lwe_part_per_thd; k++) {
        int idx = tid + k * blockDim.x;
        local_lwe_array_out[idx] -= (Torus)ksk_block[idx] * decomposed;
      }
    }
  }

  for (int k = 0; k < lwe_part_per_thd; k++) {
    int idx = tid + k * blockDim.x;
    block_lwe_array_out[idx] = local_lwe_array_out[idx];
  }
}

/// assume lwe_array_in in the gpu
template <typename Torus>
__host__ void cuda_keyswitch_lwe_ciphertext_vector(
    void *v_stream, uint32_t gpu_index, Torus *lwe_array_out,
    Torus *lwe_array_in, Torus *ksk, uint32_t lwe_dimension_in,
    uint32_t lwe_dimension_out, uint32_t base_log, uint32_t level_count,
    uint32_t num_samples) {

  cudaSetDevice(gpu_index);
  constexpr int ideal_threads = 128;

  int lwe_dim = lwe_dimension_out + 1;
  int lwe_lower, lwe_upper, cutoff;
  if (lwe_dim % ideal_threads == 0) {
    lwe_lower = lwe_dim / ideal_threads;
    lwe_upper = lwe_dim / ideal_threads;
    cutoff = 0;
  } else {
    int y =
        ceil((double)lwe_dim / (double)ideal_threads) * ideal_threads - lwe_dim;
    cutoff = ideal_threads - y;
    lwe_lower = lwe_dim / ideal_threads;
    lwe_upper = (int)ceil((double)lwe_dim / (double)ideal_threads);
  }

  int lwe_size_after = (lwe_dimension_out + 1) * num_samples;

  int shared_mem = sizeof(Torus) * (lwe_dimension_out + 1);

  auto stream = static_cast<cudaStream_t *>(v_stream);
  cudaMemsetAsync(lwe_array_out, 0, sizeof(Torus) * lwe_size_after, *stream);

  dim3 grid(num_samples, 1, 1);
  dim3 threads(ideal_threads, 1, 1);

  cudaFuncSetAttribute(keyswitch<Torus>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);

  keyswitch<<<grid, threads, shared_mem, *stream>>>(
      lwe_array_out, lwe_array_in, ksk, lwe_dimension_in, lwe_dimension_out,
      base_log, level_count, lwe_lower, lwe_upper, cutoff);
  check_cuda_error(cudaGetLastError());
}

template <typename Torus>
__host__ void cuda_fp_keyswitch_lwe_to_glwe(
    void *v_stream, uint32_t gpu_index, Torus *glwe_array_out,
    Torus *lwe_array_in, Torus *fp_ksk_array, uint32_t lwe_dimension_in,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t base_log,
    uint32_t level_count, uint32_t number_of_input_lwe,
    uint32_t number_of_keys) {
  cudaSetDevice(gpu_index);
  int threads = 256;
  int glwe_accumulator_size = (glwe_dimension + 1) * polynomial_size;
  dim3 blocks(glwe_accumulator_size / threads, number_of_input_lwe, 1);

  int shared_mem = sizeof(Torus) * threads;
  auto stream = static_cast<cudaStream_t *>(v_stream);
  fp_keyswitch<<<blocks, threads, shared_mem, *stream>>>(
      glwe_array_out, lwe_array_in, fp_ksk_array, lwe_dimension_in,
      glwe_dimension, polynomial_size, base_log, level_count,
      number_of_input_lwe, number_of_keys);
}

#endif
