#ifndef GPU_POLYNOMIAL_FUNCTIONS
#define GPU_POLYNOMIAL_FUNCTIONS
#include "device.h"
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
 * Receives num_poly  concatenated polynomials of type T. For each:
 *
 * Performs acc = acc * (X^ä + 1) if zeroAcc = false
 * Performs acc = 0 if zeroAcc
 * takes single buffer and calculates inplace.
 */
template <typename T, int elems_per_thread, int block_size>
__device__ void divide_by_monomial_negacyclic_inplace(T *accumulator, T *input,
                                                      uint32_t j, bool zeroAcc,
                                                      uint32_t num_poly) {
  constexpr int degree = block_size * elems_per_thread;
  for (int z = 0; z < num_poly; z++) {
    T *accumulator_slice = (T *)accumulator + (ptrdiff_t)(z * degree);
    T *input_slice = (T *)input + (ptrdiff_t)(z * degree);

    int tid = threadIdx.x;
    if (zeroAcc) {
      for (int i = 0; i < elems_per_thread; i++) {
        accumulator_slice[tid] = 0;
        tid += block_size;
      }
    } else {
      tid = threadIdx.x;
      for (int i = 0; i < elems_per_thread; i++) {
        if (j < degree) {
          if (tid < degree - j) {
            accumulator_slice[tid] = input_slice[tid + j];
          } else {
            accumulator_slice[tid] = -input_slice[tid - degree + j];
          }
        } else {
          uint32_t jj = j - degree;
          if (tid < degree - jj) {
            accumulator_slice[tid] = -input_slice[tid + jj];
          } else {
            accumulator_slice[tid] = input_slice[tid - degree + jj];
          }
        }
        tid += block_size;
      }
    }
  }
}

/*
 * Receives num_poly  concatenated polynomials of type T. For each:
 *
 * Performs result_acc = acc * (X^ä - 1) - acc
 * takes single buffer as input and returns a single rotated buffer
 */
template <typename T, int elems_per_thread, int block_size>
__device__ void multiply_by_monomial_negacyclic_and_sub_polynomial(
    T *acc, T *result_acc, uint32_t j, uint32_t num_poly) {
  constexpr int degree = block_size * elems_per_thread;
  for (int z = 0; z < num_poly; z++) {
    T *acc_slice = (T *)acc + (ptrdiff_t)(z * degree);
    T *result_acc_slice = (T *)result_acc + (ptrdiff_t)(z * degree);
    int tid = threadIdx.x;
    for (int i = 0; i < elems_per_thread; i++) {
      if (j < degree) {
        if (tid < j) {
          result_acc_slice[tid] = -acc_slice[tid - j + degree] - acc_slice[tid];
        } else {
          result_acc_slice[tid] = acc_slice[tid - j] - acc_slice[tid];
        }
      } else {
        uint32_t jj = j - degree;
        if (tid < jj) {
          result_acc_slice[tid] = acc_slice[tid - jj + degree] - acc_slice[tid];

        } else {
          result_acc_slice[tid] = -acc_slice[tid - jj] - acc_slice[tid];
        }
      }
      tid += block_size;
    }
  }
}

/*
 * Receives num_poly  concatenated polynomials of type T. For each performs a
 * rounding to increase accuracy of the PBS. Calculates inplace.
 */
template <typename T, int elems_per_thread, int block_size>
__device__ void round_to_closest_multiple_inplace(T *rotated_acc, int base_log,
                                                  int level_count,
                                                  uint32_t num_poly) {
  constexpr int degree = block_size * elems_per_thread;
  for (int z = 0; z < num_poly; z++) {
    T *rotated_acc_slice = (T *)rotated_acc + (ptrdiff_t)(z * degree);
    int tid = threadIdx.x;
    for (int i = 0; i < elems_per_thread; i++) {
      T x_acc = rotated_acc_slice[tid];
      T shift = sizeof(T) * 8 - level_count * base_log;
      T mask = 1ll << (shift - 1);
      T b_acc = (x_acc & mask) >> (shift - 1);
      T res_acc = x_acc >> shift;
      res_acc += b_acc;
      res_acc <<= shift;
      rotated_acc_slice[tid] = res_acc;
      tid = tid + block_size;
    }
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

    result[tid] += V1;
    result[tid + params::degree / 2] += V2;
    tid = tid + params::degree / params::opt;
  }
}

// Extracts the body of a GLWE with dimension glwe_dimension
template <typename Torus, class params>
__device__ void sample_extract_body(Torus *lwe_array_out, Torus *accumulator,
                                    uint32_t glwe_dimension) {
  // Set first coefficient of the accumulator as the body of the LWE sample
  lwe_array_out[glwe_dimension * params::degree] =
      accumulator[glwe_dimension * params::degree];
}

// Extracts the mask of a GLWE with dimension glwe_dimension
template <typename Torus, class params>
__device__ void sample_extract_mask(Torus *lwe_array_out, Torus *accumulator,
                                    uint32_t glwe_dimension) {
  for (int z = 0; z < glwe_dimension; z++) {
    Torus *lwe_array_out_slice =
        (Torus *)lwe_array_out + (ptrdiff_t)(z * params::degree);
    Torus *accumulator_slice =
        (Torus *)accumulator + (ptrdiff_t)(z * params::degree);

    // Set ACC = -ACC
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      accumulator_slice[tid] = -accumulator_slice[tid];
      tid = tid + params::degree / params::opt;
    }
    synchronize_threads_in_block();

    // Reverse the accumulator
    tid = threadIdx.x;
    Torus result[params::opt];
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      result[i] = accumulator_slice[params::degree - tid - 1];
      tid = tid + params::degree / params::opt;
    }
    synchronize_threads_in_block();
    tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      accumulator_slice[tid] = result[i];
      tid = tid + params::degree / params::opt;
    }
    synchronize_threads_in_block();

    // Perform ACC * X
    // (equivalent to multiply_by_monomial_negacyclic_inplace(1))
    tid = threadIdx.x;
    result[params::opt];
    for (int i = 0; i < params::opt; i++) {
      if (1 < params::degree) {
        if (tid < 1)
          result[i] = -accumulator_slice[tid - 1 + params::degree];
        else
          result[i] = accumulator_slice[tid - 1];
      } else {
        uint32_t jj = 1 - (uint32_t)params::degree;
        if (tid < jj)
          result[i] = accumulator_slice[tid - jj + params::degree];
        else
          result[i] = -accumulator_slice[tid - jj];
      }
      tid += params::degree / params::opt;
    }
    synchronize_threads_in_block();
    tid = threadIdx.x;
    for (int i = 0; i < params::opt; i++) {
      accumulator_slice[tid] = result[i];
      tid += params::degree / params::opt;
    }
    synchronize_threads_in_block();

    // Copy to the mask of the LWE sample
    tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      lwe_array_out_slice[tid] = accumulator_slice[tid];
      tid = tid + params::degree / params::opt;
    }
  }
}

#endif
