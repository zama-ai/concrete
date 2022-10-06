#ifndef GPU_BOOTSTRAP_FFT_1024_CUH
#define GPU_BOOTSTRAP_FFT_1024_CUH

#include "complex/operations.cuh"
#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include "twiddles.cuh"

/*
 * bit reverse
 *  coefficient bits are reversed based on precalculated indexes
 *  SW1 and SW2
 */
template <class params> __device__ void bit_reverse_inplace(double2 *A) {
  int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    short sw1 = SW1[tid];
    short sw2 = SW2[tid];
    double2 tmp = A[sw1];
    A[sw1] = A[sw2];
    A[sw2] = tmp;
    tid += params::degree / params::opt;
  }
}

/*
 *  negacyclic twiddle
 *  returns negacyclic twiddle based on degree and index
 *  twiddles are precalculated inside negTwids{3..13} arrays
 */
template <int degree> __device__ double2 negacyclic_twiddle(int j) {
  double2 twid;
  switch (degree) {
  case 512:
    twid = negTwids9[j];
    break;
  case 1024:
    twid = negTwids10[j];
    break;
  case 2048:
    twid = negTwids11[j];
    break;
  case 4096:
    twid = negTwids12[j];
    break;
  case 8192:
    twid = negTwids13[j];
  default:
    twid.x = 0;
    twid.y = 0;
    break;
  }
  return twid;
}

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
  /* First, reverse the bits of the input complex
   *  The bit reversal for half-size FFT has been stored into the
   *  SW1 and SW2 arrays beforehand
   *  Each thread is always in charge of "opt/2" pairs of coefficients,
   *  which is why we always loop through N/2 by N/opt strides
   *  The pragma unroll instruction tells the compiler to unroll the
   *  full loop, which should increase performance
   */
  bit_reverse_inplace<params>(A);
  __syncthreads();

  // Now we go through all the levels of the FFT one by one
  // (instead of recursively)
  // first iteration: k=1, zeta=i for all coefficients
  int tid = threadIdx.x;
  int i1, i2;
  double2 u, v;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 2
    i1 = tid << 1;
    i2 = i1 + 1;
    u = A[i1];
    // v = i*A[i2]
    v.y = A[i2].x;
    v.x = -A[i2].y;
    // A[i1] <- A[i1] + i*A[i2]
    // A[i2] <- A[i1] - i*A[i2]
    A[i1] += v;
    A[i2] = u - v;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // second iteration: apply the butterfly pattern
  // between groups of 4 coefficients
  // k=2, \zeta=exp(i pi/4) for even coefficients and
  // exp(3 i pi / 4) for odd coefficients
  tid = threadIdx.x;
  // odd = 0 for even coefficients, 1 for odd coefficients
  int odd = tid & 1;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 2
    // i1=2*tid if tid is even and 2*tid-1 if it is odd
    i1 = (tid << 1) - odd;
    i2 = i1 + 2;

    double a = A[i2].x;
    double b = A[i2].y;
    u = A[i1];

    // \zeta_j,2 = exp(-i pi (2j-1)/4) -> j=0: exp(i pi/4) or j=1: exp(-i pi/4)
    // \zeta_even = sqrt(2)/2 + i * sqrt(2)/2 = sqrt(2)/2*(1+i)
    // \zeta_odd  = sqrt(2)/2 - i * sqrt(2)/2 = sqrt(2)/2*(1-i)

    // v_j = \zeta_j * (a+i*b)
    // v_even = sqrt(2)/2*((a-b)+i*(a+b))
    // v_odd = sqrt(2)/2*(a+b+i*(b-a))
    v.x =
        (odd) ? (-0.707106781186548) * (a + b) : (0.707106781186548) * (a - b);
    v.y = (odd) ? (0.707106781186548) * (a - b) : (0.707106781186548) * (a + b);

    // v.x = (0.707106781186548 * odd) * (a + b) + (0.707106781186548 * (!odd))
    // * (a - b); v.y = (0.707106781186548 * odd) * (b - a) + (0.707106781186548
    // * (!odd)) * (a + b);

    A[i1] = u + v;
    A[i2] = u - v;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // third iteration
  // from k=3 on, we have to do the full complex multiplication
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 4
    // rem is the remainder of tid/4. tid takes values:
    // 0, 1, 2, 3, 4, 5, 6, 7, ... N/4
    // then rem takes values:
    // 0, 1, 2, 3, 0, 1, 2, 3, ... N/4
    // and striding by 4 will allow us to cover all
    // the coefficients correctly
    int rem = tid & 3;
    i1 = (tid << 1) - rem;
    i2 = i1 + 4;

    double2 w = negTwids3[rem];
    u = A[i1], v = A[i2] * w;

    A[i1] = u + v;
    A[i2] = u - v;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // 4_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 8
    // rem is the remainder of tid/8. tid takes values:
    // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... N/4
    // then rem takes values:
    // 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, ... N/4
    // and striding by 8 will allow us to cover all
    // the coefficients correctly
    int rem = tid & 7;
    i1 = (tid << 1) - rem;
    i2 = i1 + 8;

    double2 w = negTwids4[rem];
    u = A[i1], v = A[i2] * w;
    A[i1] = u + v;
    A[i2] = u - v;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // 5_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 16
    // rem is the remainder of tid/16
    // and the same logic as for previous iterations applies
    int rem = tid & 15;
    i1 = (tid << 1) - rem;
    i2 = i1 + 16;
    double2 w = negTwids5[rem];
    u = A[i1], v = A[i2] * w;
    A[i1] = u + v;
    A[i2] = u - v;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // 6_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 32
    // rem is the remainder of tid/32
    // and the same logic as for previous iterations applies
    int rem = tid & 31;
    i1 = (tid << 1) - rem;
    i2 = i1 + 32;
    double2 w = negTwids6[rem];
    u = A[i1], v = A[i2] * w;
    A[i1] = u + v;
    A[i2] = u - v;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // 7_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 64
    // rem is the remainder of tid/64
    // and the same logic as for previous iterations applies
    int rem = tid & 63;
    i1 = (tid << 1) - rem;
    i2 = i1 + 64;
    double2 w = negTwids7[rem];
    u = A[i1], v = A[i2] * w;
    A[i1] = u + v;
    A[i2] = u - v;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // 8_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 128
    // rem is the remainder of tid/128
    // and the same logic as for previous iterations applies
    int rem = tid & 127;
    i1 = (tid << 1) - rem;
    i2 = i1 + 128;
    double2 w = negTwids8[rem];
    u = A[i1], v = A[i2] * w;
    A[i1] = u + v;
    A[i2] = u - v;
    tid += params::degree / params::opt;
  }
  __syncthreads();
  if constexpr (params::log2_degree > 8) {
    // 9_th iteration
    tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      // the butterfly pattern is applied to each pair
      // of coefficients, with a stride of 256
      // rem is the remainder of tid/256
      // and the same logic as for previous iterations applies
      int rem = tid & 255;
      i1 = (tid << 1) - rem;
      i2 = i1 + 256;
      double2 w = negTwids9[rem];
      u = A[i1], v = A[i2] * w;
      A[i1] = u + v;
      A[i2] = u - v;
      tid += params::degree / params::opt;
    }
    __syncthreads();
  }
  if constexpr (params::log2_degree > 9) {
    // 10_th iteration
    tid = threadIdx.x;
    //#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      // the butterfly pattern is applied to each pair
      // of coefficients, with a stride of 512
      // rem is the remainder of tid/512
      // and the same logic as for previous iterations applies
      int rem = tid & 511;
      i1 = (tid << 1) - rem;
      i2 = i1 + 512;
      double2 w = negTwids10[rem];
      u = A[i1], v = A[i2] * w;
      A[i1] = u + v;
      A[i2] = u - v;
      tid += params::degree / params::opt;
    }
    __syncthreads();
  }

  if constexpr (params::log2_degree > 10) {
    // 11_th iteration
    tid = threadIdx.x;
    //#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      // the butterfly pattern is applied to each pair
      // of coefficients, with a stride of 1024
      // rem is the remainder of tid/1024
      // and the same logic as for previous iterations applies
      int rem = tid & 1023;
      i1 = (tid << 1) - rem;
      i2 = i1 + 1024;
      double2 w = negTwids11[rem];
      u = A[i1], v = A[i2] * w;
      A[i1] = u + v;
      A[i2] = u - v;
      tid += params::degree / params::opt;
    }
    __syncthreads();
  }

  if constexpr (params::log2_degree > 11) {
    // 12_th iteration
    tid = threadIdx.x;
    //#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      // the butterfly pattern is applied to each pair
      // of coefficients, with a stride of 2048
      // rem is the remainder of tid/2048
      // and the same logic as for previous iterations applies
      int rem = tid & 2047;
      i1 = (tid << 1) - rem;
      i2 = i1 + 2048;
      double2 w = negTwids12[rem];
      u = A[i1], v = A[i2] * w;
      A[i1] = u + v;
      A[i2] = u - v;
      tid += params::degree / params::opt;
    }
    __syncthreads();
  }
  // Real polynomials handled should not exceed a degree of 8192
}

/*
 * negacyclic inverse fft
 */
template <class params> __device__ void NSMFFT_inverse(double2 *A) {
  /* First, reverse the bits of the input complex
   *  The bit reversal for half-size FFT has been stored into the
   *  SW1 and SW2 arrays beforehand
   *  Each thread is always in charge of "opt/2" pairs of coefficients,
   *  which is why we always loop through N/2 by N/opt strides
   *  The pragma unroll instruction tells the compiler to unroll the
   *  full loop, which should increase performance
   */
  int tid;
  int i1, i2;
  double2 u, v;
  if constexpr (params::log2_degree > 11) {
    // 12_th iteration
    tid = threadIdx.x;
    //#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      // the butterfly pattern is applied to each pair
      // of coefficients, with a stride of 2048
      // rem is the remainder of tid/2048
      // and the same logic as for previous iterations applies
      int rem = tid & 2047;
      i1 = (tid << 1) - rem;
      i2 = i1 + 2048;
      double2 w = conjugate(negTwids12[rem]);
      u = A[i1], v = A[i2];
      A[i1] = (u + v) * 0.5;
      A[i2] = (u - v) * w * 0.5;
      tid += params::degree / params::opt;
    }
    __syncthreads();
  }

  if constexpr (params::log2_degree > 10) {
    // 11_th iteration
    tid = threadIdx.x;
    //#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      // the butterfly pattern is applied to each pair
      // of coefficients, with a stride of 1024
      // rem is the remainder of tid/1024
      // and the same logic as for previous iterations applies
      int rem = tid & 1023;
      i1 = (tid << 1) - rem;
      i2 = i1 + 1024;
      double2 w = conjugate(negTwids11[rem]);
      u = A[i1], v = A[i2];
      A[i1] = (u + v) * 0.5;
      A[i2] = (u - v) * w * 0.5;
      tid += params::degree / params::opt;
    }
    __syncthreads();
  }

  if constexpr (params::log2_degree > 9) {
    // 10_th iteration
    tid = threadIdx.x;
    //#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      // the butterfly pattern is applied to each pair
      // of coefficients, with a stride of 512
      // rem is the remainder of tid/512
      // and the same logic as for previous iterations applies
      int rem = tid & 511;
      i1 = (tid << 1) - rem;
      i2 = i1 + 512;
      double2 w = conjugate(negTwids10[rem]);
      u = A[i1], v = A[i2];
      A[i1] = (u + v) * 0.5;
      A[i2] = (u - v) * w * 0.5;
      tid += params::degree / params::opt;
    }
    __syncthreads();
  }

  if constexpr (params::log2_degree > 8) {
    // 9_th iteration
    tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      // the butterfly pattern is applied to each pair
      // of coefficients, with a stride of 256
      // rem is the remainder of tid/256
      // and the same logic as for previous iterations applies
      int rem = tid & 255;
      i1 = (tid << 1) - rem;
      i2 = i1 + 256;
      double2 w = conjugate(negTwids9[rem]);
      u = A[i1], v = A[i2];
      A[i1] = (u + v) * 0.5;
      A[i2] = (u - v) * w * 0.5;
      tid += params::degree / params::opt;
    }
    __syncthreads();
  }

  // 8_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 128
    // rem is the remainder of tid/128
    // and the same logic as for previous iterations applies
    int rem = tid & 127;
    i1 = (tid << 1) - rem;
    i2 = i1 + 128;
    double2 w = conjugate(negTwids8[rem]);
    u = A[i1], v = A[i2];
    A[i1] = (u + v) * 0.5;
    A[i2] = (u - v) * w * 0.5;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // 7_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 64
    // rem is the remainder of tid/64
    // and the same logic as for previous iterations applies
    int rem = tid & 63;
    i1 = (tid << 1) - rem;
    i2 = i1 + 64;
    double2 w = conjugate(negTwids7[rem]);
    u = A[i1], v = A[i2];
    A[i1] = (u + v) * 0.5;
    A[i2] = (u - v) * w * 0.5;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // 6_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 32
    // rem is the remainder of tid/32
    // and the same logic as for previous iterations applies
    int rem = tid & 31;
    i1 = (tid << 1) - rem;
    i2 = i1 + 32;
    double2 w = conjugate(negTwids6[rem]);
    u = A[i1], v = A[i2];
    A[i1] = (u + v) * 0.5;
    A[i2] = (u - v) * w * 0.5;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // 5_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 16
    // rem is the remainder of tid/16
    // and the same logic as for previous iterations applies
    int rem = tid & 15;
    i1 = (tid << 1) - rem;
    i2 = i1 + 16;
    double2 w = conjugate(negTwids5[rem]);
    u = A[i1], v = A[i2];
    A[i1] = (u + v) * 0.5;
    A[i2] = (u - v) * w * 0.5;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // 4_th iteration
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 8
    // rem is the remainder of tid/8. tid takes values:
    // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... N/4
    // then rem takes values:
    // 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, ... N/4
    // and striding by 8 will allow us to cover all
    // the coefficients correctly
    int rem = tid & 7;
    i1 = (tid << 1) - rem;
    i2 = i1 + 8;

    double2 w = conjugate(negTwids4[rem]);
    u = A[i1], v = A[i2];
    A[i1] = (u + v) * 0.5;
    A[i2] = (u - v) * w * 0.5;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // third iteration
  // from k=3 on, we have to do the full complex multiplication
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 4
    // rem is the remainder of tid/4. tid takes values:
    // 0, 1, 2, 3, 4, 5, 6, 7, ... N/4
    // then rem takes values:
    // 0, 1, 2, 3, 0, 1, 2, 3, ... N/4
    // and striding by 4 will allow us to cover all
    // the coefficients correctly
    int rem = tid & 3;
    i1 = (tid << 1) - rem;
    i2 = i1 + 4;

    double2 w = conjugate(negTwids3[rem]);
    u = A[i1], v = A[i2];
    A[i1] = (u + v) * 0.5;
    A[i2] = (u - v) * w * 0.5;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // second iteration: apply the butterfly pattern
  // between groups of 4 coefficients
  // k=2, \zeta=exp(i pi/4) for even coefficients and
  // exp(3 i pi / 4) for odd coefficients
  tid = threadIdx.x;
  // odd = 0 for even coefficients, 1 for odd coefficients
  int odd = tid & 1;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 2
    // i1=2*tid if tid is even and 2*tid-1 if it is odd
    i1 = (tid << 1) - odd;
    i2 = i1 + 2;

    double2 w;
    if (odd) {
      w.x = -0.707106781186547461715008466854;
      w.y = -0.707106781186547572737310929369;
    } else {
      w.x = 0.707106781186547461715008466854;
      w.y = -0.707106781186547572737310929369;
    }

    u = A[i1], v = A[i2];
    A[i1] = (u + v) * 0.5;
    A[i2] = (u - v) * w * 0.5;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  // Now we go through all the levels of the FFT one by one
  // (instead of recursively)
  // first iteration: k=1, zeta=i for all coefficients
  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    // the butterfly pattern is applied to each pair
    // of coefficients, with a stride of 2
    i1 = tid << 1;
    i2 = i1 + 1;
    double2 w = {0, -1};
    u = A[i1], v = A[i2];
    A[i1] = (u + v) * 0.5;
    A[i2] = (u - v) * w * 0.5;
    tid += params::degree / params::opt;
  }
  __syncthreads();

  bit_reverse_inplace<params>(A);
  __syncthreads();
  // Real polynomials handled should not exceed a degree of 8192
}

/*
 *  correction after direct fft
 *  does not use extra shared memory for recovering
 *  correction is done using registers.
 *  based on Pascal's paper
 */
template <class params>
__device__ void correction_direct_fft_inplace(double2 *x) {
  constexpr int threads = params::degree / params::opt;
  int tid = threadIdx.x;
  double2 left[params::opt / 4];
  double2 right[params::opt / 4];
#pragma unroll
  for (int i = 0; i < params::opt / 4; i++) {
    left[i] = x[tid + i * threads];
  }
#pragma unroll
  for (int i = 0; i < params::opt / 4; i++) {
    right[i] = x[params::degree / 2 - (tid + i * threads + 1)];
  }
#pragma unroll
  for (int i = 0; i < params::opt / 4; i++) {
    double2 tw = negacyclic_twiddle<params::degree>(tid + i * threads);
    double add_RE = left[i].x + right[i].x;
    double sub_RE = left[i].x - right[i].x;
    double add_IM = left[i].y + right[i].y;
    double sub_IM = left[i].y - right[i].y;

    double tmp1 = add_IM * tw.x + sub_RE * tw.y;
    double tmp2 = -sub_RE * tw.x + add_IM * tw.y;
    x[tid + i * threads].x = (add_RE + tmp1) * 0.5;
    x[tid + i * threads].y = (sub_IM + tmp2) * 0.5;
    x[params::degree / 2 - (tid + i * threads + 1)].x = (add_RE - tmp1) * 0.5;
    x[params::degree / 2 - (tid + i * threads + 1)].y = (-sub_IM + tmp2) * 0.5;
  }
}

/*
 *  correction before inverse fft
 *  does not use extra shared memory for recovering
 *  correction is done using registers.
 *  based on Pascal's paper
 */
template <class params>
__device__ void correction_inverse_fft_inplace(double2 *x) {
  constexpr int threads = params::degree / params::opt;
  int tid = threadIdx.x;
  double2 left[params::opt / 4];
  double2 right[params::opt / 4];
#pragma unroll
  for (int i = 0; i < params::opt / 4; i++) {
    left[i] = x[tid + i * threads];
  }
#pragma unroll
  for (int i = 0; i < params::opt / 4; i++) {
    right[i] = x[params::degree / 2 - (tid + i * threads + 1)];
  }
#pragma unroll
  for (int i = 0; i < params::opt / 4; i++) {
    double2 tw = negacyclic_twiddle<params::degree>(tid + i * threads);
    double add_RE = left[i].x + right[i].x;
    double sub_RE = left[i].x - right[i].x;
    double add_IM = left[i].y + right[i].y;
    double sub_IM = left[i].y - right[i].y;

    double tmp1 = add_IM * tw.x - sub_RE * tw.y;
    double tmp2 = sub_RE * tw.x + add_IM * tw.y;
    x[tid + i * threads].x = (add_RE - tmp1) * 0.5;
    x[tid + i * threads].y = (sub_IM + tmp2) * 0.5;
    x[params::degree / 2 - (tid + i * threads + 1)].x = (add_RE + tmp1) * 0.5;
    x[params::degree / 2 - (tid + i * threads + 1)].y = (-sub_IM + tmp2) * 0.5;
  }
}

/*
 * global batch fft
 * does fft in half size
 * unrolling halfsize fft result in half size + 1 eleemnts
 * this function must be called with actual degree
 * function takes as input already compressed input
 */
template <class params>
__global__ void batch_NSMFFT(double2 *d_input, double2 *d_output) {
  extern __shared__ double2 sharedMemoryFFT[];
  double2 *fft = sharedMemoryFFT;

  int tid = threadIdx.x;

#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    fft[tid] = d_input[blockIdx.x * (params::degree / 2) + tid];
    tid = tid + params::degree / params::opt;
  }
  __syncthreads();
  NSMFFT_direct<HalfDegree<params>>(fft);
  __syncthreads();
  correction_direct_fft_inplace<params>(fft);
  __syncthreads();

  tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt / 2; i++) {
    d_output[blockIdx.x * (params::degree / 2) + tid] = fft[tid];
    tid = tid + params::degree / params::opt;
  }
}

#endif // GPU_BOOTSTRAP_FFT_1024_CUH
