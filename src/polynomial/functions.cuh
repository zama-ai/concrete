#ifndef GPU_POLYNOMIAL_FUNCTIONS
#define GPU_POLYNOMIAL_FUNCTIONS
#include "helper_cuda.h"
#include "utils/timer.cuh"

/*
 *  function compresses decomposed buffer into half size complex buffer for fft
 */
template <class params>
__device__ void real_to_complex_compressed(int16_t *src, double2 *dst) {
  int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    dst[tid].x = __int2double_rn(src[2 * tid]);
    dst[tid].y = __int2double_rn(src[2 * tid + 1]);
    tid += params::degree / params::opt;
  }
}

/*
 * copy source polynomial to specific slice of batched polynomials
 * used only in low latency version
 */
template <typename T, class params>
__device__ void copy_into_ith_polynomial_low_lat(T *source, T *dst, int i) {
  int tid = threadIdx.x;
  int begin = i * (params::degree / 2 + 1);
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    dst[tid + begin] = source[tid];
    tid = tid + params::degree / params::opt;
  }

  if (threadIdx.x == 0) {
    dst[params::degree / 2 + begin] = source[params::degree / 2];
  }
}

/*
 * accumulates source polynomial into specific slice of batched polynomial
 * used only in low latency version
 */
template <typename T, class params>
__device__ void add_polynomial_inplace_low_lat(T *source, T *dst, int p_id) {
  int tid = threadIdx.x;
  int begin = p_id * (params::degree / 2 + 1);
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    dst[tid] += source[tid + begin];
    tid = tid + params::degree / params::opt;
  }

  if (threadIdx.x == 0) {
    dst[params::degree / 2] += source[params::degree / 2 + begin];
  }
}

/*
 * Performs acc = acc * (X^ä + 1) if zeroAcc = false
 * Performs acc = 0 if zeroAcc
 * takes single buffer and calculates inplace.
 */
template <typename T, int elems_per_thread, int block_size>
__device__ void divide_by_monomial_negacyclic_inplace(T *accumulator, T *input,
                                                      uint32_t j,
                                                      bool zeroAcc) {
  int tid = threadIdx.x;
  constexpr int degree = block_size * elems_per_thread;
  if (zeroAcc) {
    for (int i = 0; i < elems_per_thread; i++) {
      accumulator[tid] = 0;
      tid += block_size;
    }
  } else {
    tid = threadIdx.x;
    for (int i = 0; i < elems_per_thread; i++) {
      if (j < degree) {
        if (tid < degree - j) {
          accumulator[tid] = input[tid + j];
        } else {
          accumulator[tid] = -input[tid - degree + j];
        }
      } else {
        uint32_t jj = j - degree;
        if (tid < degree - jj) {
          accumulator[tid] = -input[tid + jj];
        } else {
          accumulator[tid] = input[tid - degree + jj];
        }
      }
      tid += block_size;
    }
  }
}

/*
 * Performs result_acc = acc * (X^ä - 1) - acc
 * takes single buffer as input returns single rotated buffer
 */
template <typename T, int elems_per_thread, int block_size>
__device__ void
multiply_by_monomial_negacyclic_and_sub_polynomial(T *acc, T *result_acc,
                                                   uint32_t j) {
  int tid = threadIdx.x;
  constexpr int degree = block_size * elems_per_thread;
  for (int i = 0; i < elems_per_thread; i++) {
    if (j < degree) {
      if (tid < j) {
        result_acc[tid] = -acc[tid - j + degree] - acc[tid];
      } else {
        result_acc[tid] = acc[tid - j] - acc[tid];
      }
    } else {
      uint32_t jj = j - degree;
      if (tid < jj) {
        result_acc[tid] = acc[tid - jj + degree] - acc[tid];

      } else {
        result_acc[tid] = -acc[tid - jj] - acc[tid];
      }
    }
    tid += block_size;
  }
}

/*
 * performs a rounding to increase accuracy of the PBS
 * calculates inplace.
 */
template <typename T, int elems_per_thread, int block_size>
__device__ void round_to_closest_multiple_inplace(T *rotated_acc, int base_log,
                                                  int level_count) {
  int tid = threadIdx.x;
  for (int i = 0; i < elems_per_thread; i++) {

    T x_acc = rotated_acc[tid];
    T shift = sizeof(T) * 8 - level_count * base_log;
    T mask = 1ll << (shift - 1);
    T b_acc = (x_acc & mask) >> (shift - 1);
    T res_acc = x_acc >> shift;
    res_acc += b_acc;
    res_acc <<= shift;
    rotated_acc[tid] = res_acc;
    tid = tid + block_size;
  }
}

template <typename Torus, class params>
__device__ void add_to_torus(double2 *m_values, Torus *result) {
  Torus mx = (sizeof(Torus) == 4) ? UINT32_MAX : UINT64_MAX;
  int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    double v1 = m_values[tid].x;
    double v2 = m_values[tid].y;

    double frac = v1 - floor(v1);
    frac *= mx;
    double carry = frac - floor(frac);
    frac += (carry >= 0.5);

    Torus V1 = 0;
    typecast_double_to_torus<Torus>(frac, V1);

    frac = v2 - floor(v2);
    frac *= mx;
    carry = frac - floor(v2);
    frac += (carry >= 0.5);

    Torus V2 = 0;
    typecast_double_to_torus<Torus>(frac, V2);

    result[tid * 2] += V1;
    result[tid * 2 + 1] += V2;
    tid = tid + params::degree / params::opt;
  }
}

template <typename Torus, class params>
__device__ void sample_extract_body(Torus *lwe_array_out, Torus *accumulator) {
  // Set first coefficient of the accumulator as the body of the LWE sample
  lwe_array_out[params::degree] = accumulator[0];
}

template <typename Torus, class params>
__device__ void sample_extract_mask(Torus *lwe_array_out, Torus *accumulator) {
  // Set ACC = -ACC
  // accumulator.negate_inplace();

  int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    accumulator[tid] = -accumulator[tid];
    tid = tid + params::degree / params::opt;
  }
  synchronize_threads_in_block();

  // Reverse the accumulator
  // accumulator.reverse_inplace();

  tid = threadIdx.x;
  Torus result[params::opt];
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    result[i] = accumulator[params::degree - tid - 1];
    tid = tid + params::degree / params::opt;
  }
  synchronize_threads_in_block();
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    accumulator[tid] = result[i];
    tid = tid + params::degree / params::opt;
  }
  synchronize_threads_in_block();

  // Perform ACC * X
  // accumulator.multiply_by_monomial_negacyclic_inplace(1);

  tid = threadIdx.x;
  result[params::opt];
  for (int i = 0; i < params::opt; i++) {
    if (1 < params::degree) {
      if (tid < 1)
        result[i] = -accumulator[tid - 1 + params::degree];
      else
        result[i] = accumulator[tid - 1];
    } else {
      uint32_t jj = 1 - (uint32_t)params::degree;
      if (tid < jj)
        result[i] = accumulator[tid - jj + params::degree];
      else
        result[i] = -accumulator[tid - jj];
    }
    tid += params::degree / params::opt;
  }
  synchronize_threads_in_block();
  tid = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    accumulator[tid] = result[i];
    tid += params::degree / params::opt;
  }
  synchronize_threads_in_block();

  // Copy to the mask of the LWE sample
  // accumulator.copy_into(lwe_array_out);

  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    lwe_array_out[tid] = accumulator[tid];
    tid = tid + params::degree / params::opt;
  }
}

#endif
