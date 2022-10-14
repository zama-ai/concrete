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

public:
  __device__ GadgetMatrix(uint32_t base_log, uint32_t level_count)
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

  template <typename V, typename U>
  __device__ void decompose_one_level(Polynomial<V, params> &result,
                                      Polynomial<U, params> &polynomial,
                                      uint32_t level) {
    int tid = threadIdx.x;
    for (int i = 0; i < params::opt; i++) {
      T s = polynomial.coefficients[tid] + this->offset;
      uint32_t decal = (sizeof(T) * 8 - (level + 1) * this->base_log);
      T temp1 = (s >> decal) & this->mask;
      result.coefficients[tid] = (V)(temp1 - this->halfbg);
      tid += params::degree / params::opt;
    }
  }
  template <typename V, typename U>
  __device__ void decompose_one_level(V *result, U *polynomial,
                                      uint32_t level) {
    int tid = threadIdx.x;
    for (int i = 0; i < params::opt; i++) {
      T s = polynomial[tid] + this->offset;
      uint32_t decal = (sizeof(T) * 8 - (level + 1) * this->base_log);
      T temp1 = (s >> decal) & this->mask;
      result[tid] = (V)(temp1 - this->halfbg);
      tid += params::degree / params::opt;
    }
  }

  __device__ T decompose_one_level_single(T element, uint32_t level) {
    T s = element + this->offset;
    uint32_t decal = (sizeof(T) * 8 - (level + 1) * this->base_log);
    T temp1 = (s >> decal) & this->mask;
    return (T)(temp1 - this->halfbg);
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

#endif // CNCRT_CRPYTO_H
