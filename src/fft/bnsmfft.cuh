#ifndef GPU_BOOTSTRAP_FFT_1024_CUH
#define GPU_BOOTSTRAP_FFT_1024_CUH

#include "complex/operations.cuh"
#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include "twiddles.cuh"

/*
 * Direct negacyclic FFT:
 *   - before the FFT the N real coefficients are stored into a
 *     N/2 sized complex with the even coefficients in the real part
 *     and the odd coefficients in the imaginary part. This is referred to
 *     as the half-size FFT
 *   - when calling BNSMFFT_direct for the forward negacyclic FFT of PBS,
 *     opt is divided by 2 because the butterfly pattern is always applied
 *     between pairs of coefficients
 *   - instead of twisting each coefficient A_j before the FFT by
 *     multiplying by the w^j roots of unity (aka twiddles, w=exp(-i pi /N)),
 *     the FFT is modified, and for each level k of the FFT the twiddle:
 *     w_j,k = exp(-i pi j/2^k)
 *     is replaced with:
 *     \zeta_j,k = exp(-i pi (2j-1)/2^k)
 *   - this technique also implies a correction of the
 *     complex obtained after the FFT, which is done in the
 * forward_negacyclic_fft_inplace function of bootstrap.cuh
 */
template <class params> __device__ void NSMFFT_direct(double2 *A) {

  /* We don't make bit reverse here, since twiddles are already reversed
   *  Each thread is always in charge of "opt/2" pairs of coefficients,
   *  which is why we always loop through N/2 by N/opt strides
   *  The pragma unroll instruction tells the compiler to unroll the
   *  full loop, which should increase performance
   */

  size_t tid = threadIdx.x;
  size_t twid_id;
  size_t t = params::degree / 2;
  size_t m = 1;
  size_t i1, i2;
  double2 u, v, w;
  // level 1
  // we don't make actual complex multiplication on level1 since we have only
  // one twiddle, it's real and image parts are equal, so we can multiply
  // it with simpler operations
#pragma unroll
  for (size_t i = 0; i < params::opt / 2; ++i) {
    i1 = tid;
    i2 = tid + t;
    u = A[i1];
    v.x = (A[i2].x - A[i2].y) * 0.707106781186547461715008466854;
    v.y = (A[i2].x + A[i2].y) * 0.707106781186547461715008466854;
    A[i1].x += v.x;
    A[i1].y += v.y;

    A[i2].x = u.x - v.x;
    A[i2].y = u.y - v.y;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  size_t iter = 1;
  // for levels more than 1
  // from here none of the twiddles have equal real and imag part, so
  // complete complex multiplication has to be done
  // here we have more than one twiddles
  while (t > 1) {
    iter++;
    tid = threadIdx.x;
    t >>= 1;
    m <<= 1;
#pragma unroll
    for (size_t i = 0; i < params::opt / 2; ++i) {
      twid_id = tid / t;
      i1 = 2 * t * twid_id + (tid & (t - 1));
      i2 = i1 + t;
      w = negtwiddles[twid_id + m];
      u = A[i1];
      v.x = A[i2].x * w.x - A[i2].y * w.y;
      v.y = A[i2].y * w.x + A[i2].x * w.y;
      A[i1].x += v.x;
      A[i1].y += v.y;
      A[i2].x = u.x - v.x;
      A[i2].y = u.y - v.y;
      tid += params::degree / params::opt;
    }
    __syncthreads();
  }
}

/*
 * negacyclic inverse fft
 */
template <class params> __device__ void NSMFFT_inverse(double2 *A) {

  /* We don't make bit reverse here, since twiddles are already reversed
   *  Each thread is always in charge of "opt/2" pairs of coefficients,
   *  which is why we always loop through N/2 by N/opt strides
   *  The pragma unroll instruction tells the compiler to unroll the
   *  full loop, which should increase performance
   */

  size_t tid = threadIdx.x;
  size_t twid_id;
  size_t m = params::degree;
  size_t t = 1;
  size_t i1, i2;
  double2 u, w;

  tid = threadIdx.x;
  for (size_t i = 0; i < params::opt; ++i) {
    A[tid].x *= 1. / params::degree;
    A[tid].y *= 1. / params::degree;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // none of the twiddles have equal real and imag part, so
  // complete complex multiplication has to be done
  // here we have more than one twiddles
  while (m > 1) {
    tid = threadIdx.x;
    m >>= 1;
#pragma unroll
    for (size_t i = 0; i < params::opt / 2; ++i) {
      twid_id = tid / t;
      i1 = 2 * t * twid_id + (tid & (t - 1));
      i2 = i1 + t;
      w = negtwiddles[twid_id + m];
      u.x = A[i1].x - A[i2].x;
      u.y = A[i1].y - A[i2].y;
      A[i1].x += A[i2].x;
      A[i1].y += A[i2].y;

      A[i2].x = u.x * w.x + u.y * w.y;
      A[i2].y = u.y * w.x - u.x * w.y;
      tid += params::degree / params::opt;
    }
    t <<= 1;
    __syncthreads();
  }
}

/*
 * global batch fft
 * does fft in half size
 * unrolling halfsize fft result in half size + 1 eleemnts
 * this function must be called with actual degree
 * function takes as input already compressed input
 */
template <class params, sharedMemDegree SMD>
__global__ void batch_NSMFFT(double2 *d_input, double2 *d_output,
                             double2 *buffer) {
  extern __shared__ double2 sharedMemoryFFT[];
  double2 *fft = (SMD == NOSM) ? &buffer[blockIdx.x * params::degree / 2]
                               : sharedMemoryFFT;
  int tid = threadIdx.x;

#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    fft[tid] = d_input[blockIdx.x * (params::degree / 2) + tid];
    tid = tid + params::degree / params::opt;
  }
  __syncthreads();
  NSMFFT_direct<HalfDegree<params>>(fft);
  __syncthreads();

  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    d_output[blockIdx.x * (params::degree / 2) + tid] = fft[tid];
    tid = tid + params::degree / params::opt;
  }
}

#endif // GPU_BOOTSTRAP_FFT_1024_CUH
