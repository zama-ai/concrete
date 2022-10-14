#ifndef CNCRT_POLYNOMIAL_H
#define CNCRT_POLYNOMIAL_H

#include "complex/operations.cuh"
#include "crypto/torus.cuh"
#include "fft/bnsmfft.cuh"
#include "fft/smfft.cuh"
#include "parameters.cuh"
#include "utils/memory.cuh"
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
template <typename T, class params> class PolynomialFourier;

template <typename T, class params> class Polynomial;

template <typename T, class params> class Vector;

template <typename FT, class params> class Twiddles;

template <typename T, class params> class VectorPolynomial {
public:
  T *m_data;
  uint32_t m_num_polynomials;

  __device__ VectorPolynomial(T *data, uint32_t num_polynomials)
      : m_data(data), m_num_polynomials(num_polynomials) {}

  __device__ VectorPolynomial(SharedMemory &shmem, uint32_t num_polynomials)
      : m_num_polynomials(num_polynomials) {
    shmem.get_allocation(&m_data, m_num_polynomials * params::degree);
  }

  __device__ VectorPolynomial<T, params> get_chunk(int chunk_num,
                                                   int chunk_size) {
    int pos = chunk_num * chunk_size;
    T *ptr = &m_data[pos];

    return VectorPolynomial<T, params>(ptr, chunk_size / params::degree);
  }

  __host__ VectorPolynomial() {}

  __host__ VectorPolynomial(DeviceMemory &dmem, uint32_t num_polynomials,
                            int device)
      : m_num_polynomials(num_polynomials) {
    dmem.get_allocation(&m_data, m_num_polynomials * params::degree, device);
  }

  __host__ VectorPolynomial(DeviceMemory &dmem, T *source,
                            uint32_t num_polynomials, int device)
      : m_num_polynomials(num_polynomials) {
    dmem.get_allocation_and_copy_async(
        &m_data, source, m_num_polynomials * params::degree, device);
  }

  __host__ void copy_to_host(T *dest) {
    cudaMemcpyAsync(dest, m_data,
                    sizeof(T) * m_num_polynomials * params::degree,
                    cudaMemcpyDeviceToHost);
  }

  __device__ void copy_into(Polynomial<T, params> &dest,
                            int polynomial_number = 0) {
    int tid = threadIdx.x;
    int begin = polynomial_number * params::degree;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      dest.coefficients[tid] = m_data[tid + begin];
      tid = tid + params::degree / params::opt;
    }
    synchronize_threads_in_block();
  }

  __device__ void copy_into_ith_polynomial(PolynomialFourier<T, params> &source,
                                           int i) {
    int tid = threadIdx.x;
    int begin = i * (params::degree / 2 + 1);
#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      this->m_data[tid + begin] = source.m_values[tid];
      tid = tid + params::degree / params::opt;
    }

    if (threadIdx.x == 0) {
      this->m_data[params::degree / 2 + begin] =
          source.m_values[params::degree / 2];
    }
  }

  __device__ void split_into_polynomials(Polynomial<T, params> &first,
                                         Polynomial<T, params> &second) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      first.coefficients[tid] = m_data[tid];
      second.coefficients[tid] = m_data[tid + params::degree];
      tid = tid + params::degree / params::opt;
    }
  }
};

template <typename T, class params> class PolynomialFourier {
public:
  T *m_values;
  uint32_t degree;

  __device__ __host__ PolynomialFourier(T *m_values) : m_values(m_values) {}

  __device__ PolynomialFourier(SharedMemory &shmem) : degree(degree) {
    shmem.get_allocation(&this->m_values, params::degree);
  }

  __device__ PolynomialFourier(SharedMemory &shmem, ExtraMemory extra_memory)
      : degree(degree) {
    shmem.get_allocation(&this->m_values, params::degree + extra_memory.m_size);
  }
  __device__ PolynomialFourier(SharedMemory &shmem, uint32_t degree)
      : degree(degree) {
    shmem.get_allocation(&this->m_values, degree);
  }

  __host__ PolynomialFourier(DeviceMemory &dmem, int device) : degree(degree) {
    dmem.get_allocation(&this->m_values, params::degree, device);
  }

  __device__ char *reuse_memory() { return (char *)m_values; }
  __device__ void copy_from(PolynomialFourier<T, params> &source, int begin) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      this->m_values[tid + begin] = source.m_values[tid];
      tid = tid + params::degree / params::opt;
    }
  }
  __device__ void fill_with(T value) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      m_values[tid] = value;
      tid += params::degree / params::opt;
    }
  }

  __device__ void swap_quarters_inplace() {
    int tid = threadIdx.x;
    int s1 = params::quarter;
    int s2 = params::three_quarters;

    T tmp = m_values[s2 + tid];
    m_values[s2 + tid] = m_values[s1 + tid];
    m_values[s1 + tid] = tmp;
  }

  __device__ void add_polynomial_inplace(VectorPolynomial<T, params> &source,
                                         int polynomial_number) {
    int tid = threadIdx.x;
    int begin = polynomial_number * (params::degree / 2 + 1);
#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      this->m_values[tid] += source.m_data[tid + begin];
      tid = tid + params::degree / params::opt;
    }

    if (threadIdx.x == 0) {
      this->m_values[params::degree / 2] +=
          source.m_data[params::degree / 2 + begin];
    }
  }

  __device__ T &operator[](int i) { return m_values[i]; }
};

template <typename T, class params> class Polynomial {
public:
  T *coefficients;
  uint32_t degree;

  __device__ Polynomial(T *coefficients, uint32_t degree)
      : coefficients(coefficients), degree(degree) {}

  __device__ Polynomial(char *memory, uint32_t degree)
      : coefficients((T *)memory), degree(degree) {}

  __device__ Polynomial(SharedMemory &shmem, uint32_t degree) : degree(degree) {
    shmem.get_allocation(&this->coefficients, degree);
  }

  __host__ Polynomial(DeviceMemory &dmem, uint32_t degree, int device)
      : degree(degree) {
    dmem.get_allocation(&this->coefficients, params::degree, device);
  }

  __host__ Polynomial(DeviceMemory &dmem, T *source, uint32_t degree,
                      int device)
      : degree(degree) {
    dmem.get_allocation_and_copy_async(&this->coefficients, source,
                                       params::degree, device);
  }

  __host__ void copy_to_host(T *dest) {
    cudaMemcpyAsync(dest, this->coefficients, sizeof(T) * params::degree,
                    cudaMemcpyDeviceToHost);
  }

  __device__ T get_coefficient(int i) { return this->coefficients[i]; }

  __device__ char *reuse_memory() { return (char *)coefficients; }

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
                                                    uint32_t l_gadget) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {

      T x = coefficients[tid];
      T shift = sizeof(T) * 8 - l_gadget * base_log;
      T mask = 1ll << (shift - 1);
      T b = (x & mask) >> (shift - 1);
      T res = x >> shift;
      res += b;
      res <<= shift;
      coefficients[tid] = res;
      tid = tid + params::degree / params::opt;
    }
  }

  __device__ void
  to_complex_compressed(PolynomialFourier<double2, params> &dest) {

    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      dest.m_values[tid].x = (double)coefficients[2 * tid];
      dest.m_values[tid].y = (double)coefficients[2 * tid + 1];
      tid += params::degree / params::opt;
    }
  }

  __device__ void to_complex(PolynomialFourier<double2, params> &dest) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt; i++) {
      dest.m_values[tid].x = (double)coefficients[tid];
      dest.m_values[tid].y = 0.0;
      tid += params::degree / params::opt;
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

  template <typename V>
  __device__ Vector(SharedMemory &shmem, V src, int size) : m_size(size) {
    shmem.get_allocation(&m_data, m_size);
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt && tid < m_size; i++) {
      if (tid > m_size)
        continue;
      m_data[tid] = src[tid];
      tid += params::degree / params::opt;
    }
  }

  __device__ Vector(SharedMemory &shmem, uint32_t size) : m_size(size) {
    shmem.get_allocation(&m_data, m_size);
  }

  __host__ Vector() {}

  __host__ Vector(DeviceMemory &dmem, uint32_t size, int device)
      : m_size(size) {
    dmem.get_allocation(&m_data, m_size, device);
  }

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

  __host__ Vector(DeviceMemory &dmem, T *source, uint32_t size_source,
                  int device)
      : m_size(size_source) {
    dmem.get_allocation_and_copy_async(&m_data, source, m_size, device);
  }

  __host__ Vector(DeviceMemory &dmem, T *source, uint32_t allocation_size,
                  uint32_t copy_size, int device)
      : m_size(allocation_size) {
    if (copy_size > allocation_size) {
      printf("warning: copying more than allocation");
    }
    dmem.get_allocation_and_copy_async(&m_data, source, m_size, copy_size,
                                       device);
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
