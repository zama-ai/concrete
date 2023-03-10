#ifndef CNCRT_POLYNOMIAL_H
#define CNCRT_POLYNOMIAL_H

#include "complex/operations.cuh"
#include "crypto/torus.cuh"
#include "device.h"
#include "fft/bnsmfft.cuh"
#include "parameters.cuh"
#include "utils/timer.cuh"
#include <cassert>
#include <cstdint>

#define PI 3.141592653589793238462643383279502884197

template <typename T>
__device__ T *get_chunk(T *data, int chunk_num, int chunk_size) {
  int pos = chunk_num * chunk_size;
  T *ptr = &data[pos];
  return ptr;
}

class ExtraMemory {
public:
  uint32_t m_size;
  __device__ ExtraMemory(uint32_t size) : m_size(size) {}
};

template <typename T, class params> class Polynomial;

template <typename T, class params> class Vector;

template <typename FT, class params> class Twiddles;

template <typename T, class params> class Polynomial {
public:
  T *coefficients;
  uint32_t degree;

  __device__ Polynomial(T *coefficients, uint32_t degree)
      : coefficients(coefficients), degree(degree) {}

  __device__ Polynomial(int8_t *memory, uint32_t degree)
      : coefficients((T *)memory), degree(degree) {}

  __host__ void copy_to_host(T *dest) {
    cudaMemcpyAsync(dest, this->coefficients, sizeof(T) * params::degree,
                    cudaMemcpyDeviceToHost);
  }

  __device__ T get_coefficient(int i) { return this->coefficients[i]; }

  __device__ int8_t *reuse_memory() { return (int8_t *)coefficients; }

  __device__ void copy_coefficients_from(Polynomial<T, params> &source,
                                         int begin_dest = 0,
                                         int begin_src = 0) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      this->coefficients[tid + begin_dest] = source.coefficients[tid];
      tid = tid + params::degree / params::opt;
    }
  }

  __device__ void fill_with(T value) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      coefficients[tid] = value;
      tid += params::degree / params::opt;
    }
  }

  __device__ void multiply_by_monomial_negacyclic(Polynomial<T, params> &result,
                                                  uint32_t j) {
    int tid = threadIdx.x;
    for (int i = 0; i < params::opt; i++) {
      if (j < params::degree) {
        if (tid < j)
          result.coefficients[tid] =
              -this->coefficients[tid - j + params::degree];
        else
          result.coefficients[tid] = this->coefficients[tid - j];
      } else {
        uint32_t jj = j - params::degree;
        if (tid < jj)
          result.coefficients[tid] =
              this->coefficients[tid - jj + params::degree];
        else
          result.coefficients[tid] = -this->coefficients[tid - jj];
      }
      tid += params::degree / params::opt;
    }
  }

  __device__ void multiply_by_monomial_negacyclic_inplace(uint32_t j) {
    int tid = threadIdx.x;
    T result[params::opt];
    for (int i = 0; i < params::opt; i++) {
      if (j < params::degree) {
        if (tid < j)
          result[i] = -this->coefficients[tid - j + params::degree];
        else
          result[i] = this->coefficients[tid - j];
      } else {
        uint32_t jj = j - params::degree;
        if (tid < jj)
          result[i] = this->coefficients[tid - jj + params::degree];
        else
          result[i] = -this->coefficients[tid - jj];
      }
      tid += params::degree / params::opt;
    }
    synchronize_threads_in_block();
    tid = threadIdx.x;
    for (int i = 0; i < params::opt; i++) {
      coefficients[tid] = result[i];
      tid += params::degree / params::opt;
    }
    synchronize_threads_in_block();
  }

  __device__ void multiply_by_monomial_negacyclic_and_sub_polynomial(
      Polynomial<T, params> &result, uint32_t j) {
    int tid = threadIdx.x;
    for (int i = 0; i < params::opt; i++) {
      if (j < params::degree) {
        if (tid < j)
          result.coefficients[tid] =
              -this->coefficients[tid - j + params::degree] -
              this->coefficients[tid];
        else
          result.coefficients[tid] =
              this->coefficients[tid - j] - this->coefficients[tid];
      } else {
        uint32_t jj = j - params::degree;
        if (tid < jj)
          result.coefficients[tid] =
              this->coefficients[tid - jj + params::degree] -
              this->coefficients[tid];
        else
          result.coefficients[tid] =
              -this->coefficients[tid - jj] - this->coefficients[tid];
      }
      tid += params::degree / params::opt;
    }
  }

  __device__ void divide_by_monomial_negacyclic(Polynomial<T, params> &result,
                                                uint32_t j) {
    int tid = threadIdx.x;
    for (int i = 0; i < params::opt; i++) {
      if (j < params::degree) {
        if (tid < params::degree - j) {
          result.coefficients[tid] = this->coefficients[tid + j];
        } else {
          result.coefficients[tid] =
              -this->coefficients[tid - params::degree + j];
        }
      } else {
        uint32_t jj = j - params::degree;
        if (tid < params::degree - jj) {
          result.coefficients[tid] = -this->coefficients[tid + jj];
        } else {
          result.coefficients[tid] =
              this->coefficients[tid - params::degree + jj];
        }
      }
      tid += params::degree / params::opt;
    }
  }

  __device__ void divide_by_monomial_negacyclic_inplace(uint32_t j) {
    int tid = threadIdx.x;
    T result[params::opt];
    for (int i = 0; i < params::opt; i++) {
      if (j < params::degree) {
        if (tid < params::degree - j) {
          result[i] = this->coefficients[tid + j];
        } else {
          result[i] = -this->coefficients[tid - params::degree + j];
        }
      } else {
        uint32_t jj = j - params::degree;
        if (tid < params::degree - jj) {
          result[i] = -this->coefficients[tid + jj];
        } else {
          result[i] = this->coefficients[tid - params::degree + jj];
        }
      }
      tid += params::degree / params::opt;
    }
    tid = threadIdx.x;
    for (int i = 0; i < params::opt; i++) {
      coefficients[tid] = result[i];
      tid = tid + params::degree / params::opt;
    }
  }

  __device__ void round_to_closest_multiple_inplace(uint32_t base_log,
                                                    uint32_t level_count) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {

      T x = coefficients[tid];
      T shift = sizeof(T) * 8 - level_count * base_log;
      T mask = 1ll << (shift - 1);
      T b = (x & mask) >> (shift - 1);
      T res = x >> shift;
      res += b;
      res <<= shift;
      coefficients[tid] = res;
      tid = tid + params::degree / params::opt;
    }
  }

  __device__ void multiply_by_scalar_inplace(T scalar) {
    int tid = threadIdx.x;
    const int grid_dim = blockDim.x;
    const int slices = params::degree / grid_dim;
    const int jump = grid_dim;
    for (int i = 0; i < slices; i++) {
      this->coefficients[tid] *= scalar;
      tid += jump;
    }
  }

  __device__ void add_scalar_inplace(T scalar) {
    int tid = threadIdx.x;
    const int grid_dim = blockDim.x;
    const int slices = params::degree / grid_dim;
    const int jump = grid_dim;
    for (int i = 0; i < slices; i++) {
      this->coefficients[tid] += scalar;
      tid += jump;
    }
  }

  __device__ void sub_scalar_inplace(T scalar) {
    int tid = threadIdx.x;
    const int grid_dim = blockDim.x;
    const int slices = params::degree / grid_dim;
    const int jump = grid_dim;
    for (int i = 0; i < slices; i++) {
      this->coefficients[tid] -= scalar;
      tid += jump;
    }
  }

  __device__ void sub_polynomial_inplace(Polynomial<T, params> &rhs) {
    int tid = threadIdx.x;
    const int grid_dim = blockDim.x;
    const int slices = params::degree / grid_dim;
    const int jump = grid_dim;
    for (int i = 0; i < slices; i++) {
      this->coefficients[tid] -= rhs.coefficients[tid];
      tid += jump;
    }
  }

  __device__ void negate_inplace() {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      coefficients[tid] = -coefficients[tid];
      tid = tid + params::degree / params::opt;
    }
    synchronize_threads_in_block();
  }

  __device__ void copy_into(Vector<T, params> &vec) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      vec.m_data[tid] = coefficients[tid];
      tid = tid + params::degree / params::opt;
    }
  }

  __device__ void copy_reversed_into(Vector<T, params> &vec) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      vec.m_data[tid] = coefficients[params::degree - tid - 1];
      tid = tid + params::degree / params::opt;
    }
  }

  __device__ void reverse_inplace() {
    int tid = threadIdx.x;
    T result[params::opt];
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      result[i] = coefficients[params::degree - tid - 1];
      tid = tid + params::degree / params::opt;
    }
    synchronize_threads_in_block();
    tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      coefficients[tid] = result[i];
      tid = tid + params::degree / params::opt;
    }
    synchronize_threads_in_block();
  }
};
template <typename T, class params> class Vector {
public:
  T *m_data;
  uint32_t m_size;

  __device__ Vector(T *elements, uint32_t size)
      : m_data(elements), m_size(size) {}

  __host__ Vector() {}

  __device__ T &operator[](int i) { return m_data[i]; }

  __device__ Vector<T, params> get_chunk(int chunk_num, int chunk_size) {
    int pos = chunk_num * chunk_size;
    T *ptr = &m_data[pos];
    return Vector<T, params>(ptr, chunk_size);
  }

  __host__ void copy_to_device(T *source, uint32_t elements) {
    cudaMemcpyAsync(m_data, source, sizeof(T) * elements,
                    cudaMemcpyHostToDevice);
  }

  __host__ void copy_to_host(T *dest) {
    cudaMemcpyAsync(dest, m_data, sizeof(T) * m_size, cudaMemcpyDeviceToHost);
  }

  __host__ void copy_to_host(T *dest, int elements) {
    cudaMemcpyAsync(dest, m_data, sizeof(T) * elements, cudaMemcpyDeviceToHost);
  }

  __device__ T get_ith_element(int i) { return m_data[i]; }

  __device__ T get_last_element() { return m_data[m_size - 1]; }

  __device__ void set_last_element(T elem) { m_data[m_size - 1] = elem; }

  __device__ void operator-=(const Vector<T, params> &rhs) {
    assert(m_size == rhs->m_size);
    int tid = threadIdx.x;
    int pos = tid;
    int total = m_size / blockDim.x + 1;
    for (int i = 0; i < total; i++) {
      if (pos < m_size)
        m_data[pos] -= rhs.m_data[pos];
      pos += blockDim.x;
    }
  }

  __device__ void operator*=(const T &rhs) {
    int tid = threadIdx.x;
    int pos = tid;
    int total = m_size / blockDim.x + 1;
    for (int i = 0; i < total; i++) {
      if (pos < m_size)
        m_data[pos] *= rhs;
      pos += blockDim.x;
    }
  }
};

template <typename FT, class params> class Twiddles {
public:
  Vector<FT, params> twiddles2, twiddles3, twiddles4, twiddles5, twiddles6,
      twiddles7, twiddles8, twiddles9, twiddles10;

  __device__
  Twiddles(Vector<FT, params> &twiddles2, Vector<FT, params> &twiddles3,
           Vector<FT, params> &twiddles4, Vector<FT, params> &twiddles5,
           Vector<FT, params> &twiddles6, Vector<FT, params> &twiddles7,
           Vector<FT, params> &twiddles8, Vector<FT, params> &twiddles9,
           Vector<FT, params> &twiddles10)
      : twiddles2(twiddles2), twiddles3(twiddles3), twiddles4(twiddles4),
        twiddles5(twiddles5), twiddles6(twiddles6), twiddles7(twiddles7),
        twiddles8(twiddles8), twiddles9(twiddles9), twiddles10(twiddles10) {}
};

#endif // CNCRT_POLYNOMIAL_H
