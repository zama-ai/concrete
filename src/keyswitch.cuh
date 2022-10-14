#ifndef CNCRT_KS_H
#define CNCRT_KS_H

#include "crypto/gadget.cuh"
#include "crypto/torus.cuh"
#include "polynomial/polynomial.cuh"
#include <thread>
#include <vector>

template <typename Torus>
__device__ Torus *get_ith_block(Torus *ksk, int i, int level,
                                uint32_t lwe_dimension_after,
                                uint32_t l_gadget) {
  int pos = i * l_gadget * (lwe_dimension_after + 1) +
            level * (lwe_dimension_after + 1);
  Torus *ptr = &ksk[pos];
  return ptr;
}

template <typename Torus>
__device__ Torus decompose_one(Torus &state, Torus mod_b_mask, int base_log) {
  Torus res = state & mod_b_mask;
  state >>= base_log;
  Torus carry = ((res - 1ll) | state) & res;
  carry >>= base_log - 1;
  state += carry;
  res -= carry << base_log;
  return res;
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
__global__ void keyswitch(Torus *lwe_out, Torus *lwe_in, Torus *ksk,
                          uint32_t lwe_dimension_before,
                          uint32_t lwe_dimension_after, uint32_t base_log,
                          uint32_t l_gadget, int lwe_lower, int lwe_upper,
                          int cutoff) {
  int tid = threadIdx.x;

  extern __shared__ char sharedmem[];

  Torus *local_lwe_out = (Torus *)sharedmem;

  auto block_lwe_in = get_chunk(lwe_in, blockIdx.x, lwe_dimension_before + 1);
  auto block_lwe_out = get_chunk(lwe_out, blockIdx.x, lwe_dimension_after + 1);

  auto gadget = GadgetMatrixSingle<Torus>(base_log, l_gadget);

  int lwe_part_per_thd;
  if (tid < cutoff) {
    lwe_part_per_thd = lwe_upper;
  } else {
    lwe_part_per_thd = lwe_lower;
  }
  __syncthreads();

  for (int k = 0; k < lwe_part_per_thd; k++) {
    int idx = tid + k * blockDim.x;
    local_lwe_out[idx] = 0;
  }

  if (tid == 0) {
    local_lwe_out[lwe_dimension_after] = block_lwe_in[lwe_dimension_before];
  }

  for (int i = 0; i < lwe_dimension_before; i++) {

    __syncthreads();

    Torus a_i = round_to_closest_multiple(block_lwe_in[i], base_log, l_gadget);

    Torus state = a_i >> (sizeof(Torus) * 8 - base_log * l_gadget);
    Torus mod_b_mask = (1ll << base_log) - 1ll;

    for (int j = 0; j < l_gadget; j++) {
      auto ksk_block = get_ith_block(ksk, i, l_gadget - j - 1,
                                     lwe_dimension_after, l_gadget);
      Torus decomposed = decompose_one<Torus>(state, mod_b_mask, base_log);
      for (int k = 0; k < lwe_part_per_thd; k++) {
        int idx = tid + k * blockDim.x;
        local_lwe_out[idx] -= (Torus)ksk_block[idx] * decomposed;
      }
    }
  }

  for (int k = 0; k < lwe_part_per_thd; k++) {
    int idx = tid + k * blockDim.x;
    block_lwe_out[idx] = local_lwe_out[idx];
  }
}

/// assume lwe_in in the gpu
template <typename Torus>
__host__ void cuda_keyswitch_lwe_ciphertext_vector(
    void *v_stream, Torus *lwe_out, Torus *lwe_in, Torus *ksk,
    uint32_t lwe_dimension_before, uint32_t lwe_dimension_after,
    uint32_t base_log, uint32_t l_gadget, uint32_t num_samples) {

  constexpr int ideal_threads = 128;

  int lwe_dim = lwe_dimension_after + 1;
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

  int lwe_size_after = (lwe_dimension_after + 1) * num_samples;

  int shared_mem = sizeof(Torus) * (lwe_dimension_after + 1);

  cudaMemset(lwe_out, 0, sizeof(Torus) * lwe_size_after);

  dim3 grid(num_samples, 1, 1);
  dim3 threads(ideal_threads, 1, 1);

  cudaFuncSetAttribute(keyswitch<Torus>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);

  auto stream = static_cast<cudaStream_t *>(v_stream);
  keyswitch<<<grid, threads, shared_mem, *stream>>>(
      lwe_out, lwe_in, ksk, lwe_dimension_before, lwe_dimension_after, base_log,
      l_gadget, lwe_lower, lwe_upper, cutoff);

  cudaStreamSynchronize(*stream);
}

#endif
