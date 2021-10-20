#ifndef CNCRT_TORUS_H
#define CNCRT_TORUS_H

#include "types/int128.cuh"
#include <limits>

template <typename Torus>
__device__ inline Torus typecast_double_to_torus(double x) {
  if constexpr (sizeof(Torus) < 8) {
    // this simple cast works up to 32 bits, afterwards we must do it manually
    long long ret = x;
    return (Torus)ret;
  } else {
    int128 nnnn = make_int128_from_float(x);
    uint64_t lll = nnnn.lo_;
    return lll;
  }
  // nvcc doesn't get it that the if {} else {} above should always return
  // something, and complains that this function might return nothing, so we
  // put this useless return here
  return 0;
}

template <typename T>
__device__ inline T round_to_closest_multiple(T x, uint32_t base_log,
                                              uint32_t l_gadget) {
  T shift = sizeof(T) * 8 - l_gadget * base_log;
  T mask = 1ll << (shift - 1);
  T b = (x & mask) >> (shift - 1);
  T res = x >> shift;
  res += b;
  res <<= shift;
  return res;
}

template <typename T>
__device__ __forceinline__ T rescale_torus_element(T element,
                                                   uint32_t log_shift) {
  // todo(Joao): not sure if this works
  // return element >> log_shift;
  return round((double)element / (double(std::numeric_limits<T>::max()) + 1.0) *
               (double)log_shift);
}

#endif // CNCRT_TORUS_H