#ifndef CNCRT_CRYPTO_H
#define CNCRT_CRPYTO_H

#include "polynomial/polynomial.cuh"
#include <cstdint>

#pragma once
template <typename T, class params> class GadgetMatrix {
private:
  uint32_t level_count;
  uint32_t base_log;
  uint32_t mask;
  uint32_t halfbg;
  T offset;
  int current_level;
  T mask_mod_b;
  T *state;

public:
  __device__ GadgetMatrix(uint32_t base_log, uint32_t level_count, T *state)
      : base_log(base_log), level_count(level_count), state(state) {

    mask_mod_b = (1ll << base_log) - 1ll;
    current_level = level_count;
    int tid = threadIdx.x;
    for (int i = 0; i < params::opt; i++) {
      state[tid] >>= (sizeof(T) * 8 - base_log * level_count);
      tid += params::degree / params::opt;
    }
    synchronize_threads_in_block();
  }

  __device__ void decompose_and_compress_next(double2 *result) {
    int tid = threadIdx.x;
    current_level -= 1;
    for (int i = 0; i < params::opt / 2; i++) {
      T res_re = state[tid * 2] & mask_mod_b;
      T res_im = state[tid * 2 + 1] & mask_mod_b;
      state[tid * 2] >>= base_log;
      state[tid * 2 + 1] >>= base_log;
      T carry_re = ((res_re - 1ll) | state[tid * 2]) & res_re;
      T carry_im = ((res_im - 1ll) | state[tid * 2 + 1]) & res_im;
      carry_re >>= (base_log - 1);
      carry_im >>= (base_log - 1);
      state[tid * 2] += carry_re;
      state[tid * 2 + 1] += carry_im;
      res_re -= carry_re << base_log;
      res_im -= carry_im << base_log;

      result[tid].x = (int32_t)res_re;
      result[tid].y = (int32_t)res_im;

      tid += params::degree / params::opt;
    }
    synchronize_threads_in_block();
  }
  __device__ void decompose_and_compress_level(double2 *result, int level) {
    for (int i = 0; i < level_count - level; i++)
      decompose_and_compress_next(result);
  }
};

template <typename T> class GadgetMatrixSingle {
private:
  uint32_t level_count;
  uint32_t base_log;
  uint32_t mask;
  uint32_t halfbg;
  T offset;

public:
  __device__ GadgetMatrixSingle(uint32_t base_log, uint32_t level_count)
      : base_log(base_log), level_count(level_count) {
    uint32_t bg = 1 << base_log;
    this->halfbg = bg / 2;
    this->mask = bg - 1;
    T temp = 0;
    for (int i = 0; i < this->level_count; i++) {
      temp += 1ULL << (sizeof(T) * 8 - (i + 1) * this->base_log);
    }
    this->offset = temp * this->halfbg;
  }

  __device__ T decompose_one_level_single(T element, uint32_t level) {
    T s = element + this->offset;
    uint32_t decal = (sizeof(T) * 8 - (level + 1) * this->base_log);
    T temp1 = (s >> decal) & this->mask;
    return (T)(temp1 - this->halfbg);
  }
};

template <typename Torus>
__device__ Torus decompose_one(Torus &state, Torus mask_mod_b, int base_log) {
  Torus res = state & mask_mod_b;
  state >>= base_log;
  Torus carry = ((res - 1ll) | state) & res;
  carry >>= base_log - 1;
  state += carry;
  res -= carry << base_log;
  return res;
}

#endif // CNCRT_CRPYTO_H
