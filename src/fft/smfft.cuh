/*
#ifndef GPU_BOOTSTRAP_SMFFT_CUH
#define GPU_BOOTSTRAP_SMFFT_CUH

#include "../complex/operations.cuh"
#include "twiddles.cuh"

__device__ __inline__ double2 Get_W_value_inverse(int index) {
  double2 ctemp = _gTwiddles[index];
  ctemp.y = -ctemp.y;
  return (ctemp);
}
template <class params>
__device__ double2 Get_after_inverse_fft_twiddle(int index) {
  double2 ctemp;
  switch (params::degree) {
  case 512:
    ctemp = INVERSE_TWIDDLES_512[index];
    break;
  case 1024:
    ctemp = gTwiddles1024[index];
    ctemp.x /= params::degree;
    ctemp.y /= -params::degree;
    break;
  default:
    break;
  }

  return ctemp;
}

__device__ __inline__ double shfl(double *value, int par) {
#if (CUDART_VERSION >= 9000)
  return (__shfl_sync(0xffffffff, (*value), par));
#else
  return (__shfl((*value), par));
#endif
}

__device__ __inline__ double shfl_xor(double *value, int par) {
#if (CUDART_VERSION >= 9000)
  return (__shfl_xor_sync(0xffffffff, (*value), par));
#else
  return (__shfl_xor((*value), par));
#endif
}

__device__ __inline__ double shfl_down(double *value, int par) {
#if (CUDART_VERSION >= 9000)
  return (__shfl_down_sync(0xffffffff, (*value), par));
#else
  return (__shfl_down((*value), par));
#endif
}

__device__ __inline__ void
reorder_16_register(double2 *A_DFT_value, double2 *B_DFT_value,
                    double2 *C_DFT_value, double2 *D_DFT_value, int *local_id) {
  double2 Af2temp, Bf2temp, Cf2temp, Df2temp;
  unsigned int target = (((unsigned int)__brev(((*local_id) & 15))) >> (28)) +
                        16 * ((*local_id) >> 4);
  Af2temp.x = shfl(&(A_DFT_value->x), target);
  Af2temp.y = shfl(&(A_DFT_value->y), target);
  Bf2temp.x = shfl(&(B_DFT_value->x), target);
  Bf2temp.y = shfl(&(B_DFT_value->y), target);
  Cf2temp.x = shfl(&(C_DFT_value->x), target);
  Cf2temp.y = shfl(&(C_DFT_value->y), target);
  Df2temp.x = shfl(&(D_DFT_value->x), target);
  Df2temp.y = shfl(&(D_DFT_value->y), target);
  __syncwarp();
  (*A_DFT_value) = Af2temp;
  (*B_DFT_value) = Bf2temp;
  (*C_DFT_value) = Cf2temp;
  (*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_32_register(double2 *A_DFT_value,
                                               double2 *B_DFT_value,
                                               double2 *C_DFT_value,
                                               double2 *D_DFT_value) {
  double2 Af2temp, Bf2temp, Cf2temp, Df2temp;
  unsigned int target = ((unsigned int)__brev(threadIdx.x)) >> (27);
  Af2temp.x = shfl(&(A_DFT_value->x), target);
  Af2temp.y = shfl(&(A_DFT_value->y), target);
  Bf2temp.x = shfl(&(B_DFT_value->x), target);
  Bf2temp.y = shfl(&(B_DFT_value->y), target);
  Cf2temp.x = shfl(&(C_DFT_value->x), target);
  Cf2temp.y = shfl(&(C_DFT_value->y), target);
  Df2temp.x = shfl(&(D_DFT_value->x), target);
  Df2temp.y = shfl(&(D_DFT_value->y), target);
  __syncwarp();
  (*A_DFT_value) = Af2temp;
  (*B_DFT_value) = Bf2temp;
  (*C_DFT_value) = Cf2temp;
  (*D_DFT_value) = Df2temp;
}

template <class params>
__device__ __inline__ void
reorder_512(double2 *s_input, double2 *A_DFT_value, double2 *B_DFT_value,
            double2 *C_DFT_value, double2 *D_DFT_value) {
  int local_id = threadIdx.x & (params::warp - 1);
  int warp_id = threadIdx.x / params::warp;

  // reorder elements within warp so we can save them in semi-transposed manner
  // into shared memory
  reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

  // reorder elements within warp so we can save them in semi-transposed manner
  // into shared memory
  __syncthreads();
  unsigned int sm_store_pos =
      (local_id >> 1) + 16 * (local_id & 1) + warp_id * 132;
  s_input[sm_store_pos] = *A_DFT_value;
  s_input[sm_store_pos + 33] = *B_DFT_value;
  s_input[66 + sm_store_pos] = *C_DFT_value;
  s_input[66 + sm_store_pos + 33] = *D_DFT_value;

  __syncthreads();

  // Read shared memory to get reordered input
  unsigned int sm_read_pos = (local_id & 15) * 32 + local_id + warp_id * 4;
  __syncthreads();
  *A_DFT_value = s_input[sm_read_pos + 0];
  *B_DFT_value = s_input[sm_read_pos + 1];
  *C_DFT_value = s_input[sm_read_pos + 2];
  *D_DFT_value = s_input[sm_read_pos + 3];

  __syncthreads();
  reorder_16_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value,
                      &local_id);

  __syncthreads();
}

template <class params>
__device__ __inline__ void
reorder_1024(double2 *s_input, double2 *A_DFT_value, double2 *B_DFT_value,
             double2 *C_DFT_value, double2 *D_DFT_value) {
  int local_id = threadIdx.x & (params::warp - 1);
  int warp_id = threadIdx.x / params::warp;

  // reorder elements within params::warp so we can save them in semi-transposed
  // manner into shared memory
  reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

  // reorder elements within params::warp so we can save them in semi-transposed
  // manner into shared memory
  __syncthreads();
  unsigned int sm_store_pos =
      (local_id >> 0) + 32 * (local_id & 0) + warp_id * 132;
  s_input[sm_store_pos] = *A_DFT_value;
  s_input[sm_store_pos + 33] = *B_DFT_value;
  s_input[66 + sm_store_pos] = *C_DFT_value;
  s_input[66 + sm_store_pos + 33] = *D_DFT_value;

  // Read shared memory to get reordered input
  unsigned int sm_read_pos = (local_id & 31) * 32 + local_id + warp_id * 4;
  __syncthreads();
  *A_DFT_value = s_input[sm_read_pos + 0];
  *B_DFT_value = s_input[sm_read_pos + 1];
  *C_DFT_value = s_input[sm_read_pos + 2];
  *D_DFT_value = s_input[sm_read_pos + 3];

  __syncthreads();
  reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}

__device__ bool printOnce = true;

template <class params> __device__ void do_SMFFT_CT_DIT(double2 *s_input) {
  double2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
  double2 W;
  double2 Aftemp, Bftemp, Cftemp, Dftemp;

  int j, m_param;
  int parity, itemp;
  int A_read_index, B_read_index, C_read_index, D_read_index;
  int PoT, PoTp1, q;

  int local_id = threadIdx.x & (params::warp - 1);
  int warp_id = threadIdx.x / params::warp;
  A_DFT_value = s_input[local_id + (warp_id << 2) * params::warp];
  B_DFT_value =
      s_input[local_id + (warp_id << 2) * params::warp + params::warp];
  C_DFT_value =
      s_input[local_id + (warp_id << 2) * params::warp + 2 * params::warp];
  D_DFT_value =
      s_input[local_id + (warp_id << 2) * params::warp + 3 * params::warp];

  switch (params::log2_degree) {
  case 9:
    reorder_512<params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value,
                        &D_DFT_value);
    break;
  case 10:
    reorder_1024<params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value,
                         &D_DFT_value);
    break;
  // case 11:
  //	reorder_2048<params, opt>(s_input, &A_DFT_value, &B_DFT_value,
  //&C_DFT_value, &D_DFT_value); 	break;
  default:
    break;
  }

  //----> FFT
  PoT = 1;
  PoTp1 = 2;

  //--> First iteration
  itemp = local_id & 1;
  parity = (1 - itemp * 2);

  A_DFT_value.x = parity * A_DFT_value.x + shfl_xor(&A_DFT_value.x, 1);
  A_DFT_value.y = parity * A_DFT_value.y + shfl_xor(&A_DFT_value.y, 1);
  B_DFT_value.x = parity * B_DFT_value.x + shfl_xor(&B_DFT_value.x, 1);
  B_DFT_value.y = parity * B_DFT_value.y + shfl_xor(&B_DFT_value.y, 1);
  C_DFT_value.x = parity * C_DFT_value.x + shfl_xor(&C_DFT_value.x, 1);
  C_DFT_value.y = parity * C_DFT_value.y + shfl_xor(&C_DFT_value.y, 1);
  D_DFT_value.x = parity * D_DFT_value.x + shfl_xor(&D_DFT_value.x, 1);
  D_DFT_value.y = parity * D_DFT_value.y + shfl_xor(&D_DFT_value.y, 1);

  //--> Second through Fifth iteration (no synchronization)
  PoT = 2;
  PoTp1 = 4;
  for (q = 1; q < 5; q++) {
    m_param = (local_id & (PoTp1 - 1));
    itemp = m_param >> q;
    parity = ((itemp << 1) - 1);
    if (params::fft_direction)
      W = Get_W_value_inverse((q - 1) * 257 + itemp * m_param);
    else
      W = _gTwiddles[(q - 1) * 257 + itemp * m_param];
    Aftemp.x = W.x * A_DFT_value.x - W.y * A_DFT_value.y;
    Aftemp.y = W.x * A_DFT_value.y + W.y * A_DFT_value.x;
    Bftemp.x = W.x * B_DFT_value.x - W.y * B_DFT_value.y;
    Bftemp.y = W.x * B_DFT_value.y + W.y * B_DFT_value.x;
    Cftemp.x = W.x * C_DFT_value.x - W.y * C_DFT_value.y;
    Cftemp.y = W.x * C_DFT_value.y + W.y * C_DFT_value.x;
    Dftemp.x = W.x * D_DFT_value.x - W.y * D_DFT_value.y;
    Dftemp.y = W.x * D_DFT_value.y + W.y * D_DFT_value.x;

    A_DFT_value.x = Aftemp.x + parity * shfl_xor(&Aftemp.x, PoT);
    A_DFT_value.y = Aftemp.y + parity * shfl_xor(&Aftemp.y, PoT);
    B_DFT_value.x = Bftemp.x + parity * shfl_xor(&Bftemp.x, PoT);
    B_DFT_value.y = Bftemp.y + parity * shfl_xor(&Bftemp.y, PoT);
    C_DFT_value.x = Cftemp.x + parity * shfl_xor(&Cftemp.x, PoT);
    C_DFT_value.y = Cftemp.y + parity * shfl_xor(&Cftemp.y, PoT);
    D_DFT_value.x = Dftemp.x + parity * shfl_xor(&Dftemp.x, PoT);
    D_DFT_value.y = Dftemp.y + parity * shfl_xor(&Dftemp.y, PoT);

    PoT = PoT << 1;
    PoTp1 = PoTp1 << 1;
  }

  itemp = local_id + (warp_id << 2) * params::warp;
  s_input[itemp] = A_DFT_value;
  s_input[itemp + params::warp] = B_DFT_value;
  s_input[itemp + 2 * params::warp] = C_DFT_value;
  s_input[itemp + 3 * params::warp] = D_DFT_value;

  for (q = 5; q < (params::log2_degree - 1); q++) {
    __syncthreads();
    m_param = threadIdx.x & (PoT - 1);
    j = threadIdx.x >> q;

    if (params::fft_direction)
      W = Get_W_value_inverse((q - 1) * 257 + m_param);
    else
      W = _gTwiddles[(q - 1) * 257 + m_param];

    A_read_index = j * (PoTp1 << 1) + m_param;
    B_read_index = j * (PoTp1 << 1) + m_param + PoT;
    C_read_index = j * (PoTp1 << 1) + m_param + PoTp1;
    D_read_index = j * (PoTp1 << 1) + m_param + 3 * PoT;

    Aftemp = s_input[A_read_index];
    Bftemp = s_input[B_read_index];
    A_DFT_value.x = Aftemp.x + W.x * Bftemp.x - W.y * Bftemp.y;
    A_DFT_value.y = Aftemp.y + W.x * Bftemp.y + W.y * Bftemp.x;
    B_DFT_value.x = Aftemp.x - W.x * Bftemp.x + W.y * Bftemp.y;
    B_DFT_value.y = Aftemp.y - W.x * Bftemp.y - W.y * Bftemp.x;

    Cftemp = s_input[C_read_index];
    Dftemp = s_input[D_read_index];
    C_DFT_value.x = Cftemp.x + W.x * Dftemp.x - W.y * Dftemp.y;
    C_DFT_value.y = Cftemp.y + W.x * Dftemp.y + W.y * Dftemp.x;
    D_DFT_value.x = Cftemp.x - W.x * Dftemp.x + W.y * Dftemp.y;
    D_DFT_value.y = Cftemp.y - W.x * Dftemp.y - W.y * Dftemp.x;

    s_input[A_read_index] = A_DFT_value;
    s_input[B_read_index] = B_DFT_value;
    s_input[C_read_index] = C_DFT_value;
    s_input[D_read_index] = D_DFT_value;

    PoT = PoT << 1;
    PoTp1 = PoTp1 << 1;
  }

  // last iteration
  if (params::log2_degree > 6) {
    __syncthreads();
    m_param = threadIdx.x;

    if (params::fft_direction)
      W = Get_W_value_inverse((q - 1) * 257 + m_param);
    else
      W = _gTwiddles[(q - 1) * 257 + m_param];

    A_read_index = m_param;
    B_read_index = m_param + PoT;
    C_read_index = m_param + (PoT >> 1);
    D_read_index = m_param + 3 * (PoT >> 1);

    Aftemp = s_input[A_read_index];
    Bftemp = s_input[B_read_index];
    A_DFT_value.x = Aftemp.x + W.x * Bftemp.x - W.y * Bftemp.y;
    A_DFT_value.y = Aftemp.y + W.x * Bftemp.y + W.y * Bftemp.x;
    B_DFT_value.x = Aftemp.x - W.x * Bftemp.x + W.y * Bftemp.y;
    B_DFT_value.y = Aftemp.y - W.x * Bftemp.y - W.y * Bftemp.x;

    Cftemp = s_input[C_read_index];
    Dftemp = s_input[D_read_index];
    C_DFT_value.x = Cftemp.x + W.y * Dftemp.x + W.x * Dftemp.y;
    C_DFT_value.y = Cftemp.y + W.y * Dftemp.y - W.x * Dftemp.x;
    D_DFT_value.x = Cftemp.x - W.y * Dftemp.x - W.x * Dftemp.y;
    D_DFT_value.y = Cftemp.y - W.y * Dftemp.y + W.x * Dftemp.x;

    s_input[A_read_index] = A_DFT_value;
    s_input[B_read_index] = B_DFT_value;
    s_input[C_read_index] = C_DFT_value;
    s_input[D_read_index] = D_DFT_value;
  }
}

template <class params>
__global__ void SMFFT_DIT_external(double2 *d_input, double2 *d_output) {
  __syncthreads();

  extern __shared__ double2 sharedmemBSK[];

  double2 *s_input = sharedmemBSK;

  int cTid = threadIdx.x * params::opt;
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    double2 tmp;
    switch (params::degree) {
    case 512:
      tmp = INVERSE_TWIDDLES_512[cTid];
      tmp.x *= params::degree;
      tmp.y *= -params::degree;
      break;
    case 1024:
      tmp = gTwiddles1024[cTid];
      break;
    default:
      break;
    }

    d_input[blockIdx.x * params::degree + cTid] *= tmp;
    cTid++;
  }

  __syncthreads();

  s_input[threadIdx.x] = d_input[threadIdx.x + blockIdx.x * params::degree];
  s_input[threadIdx.x + params::quarter] =
      d_input[threadIdx.x + blockIdx.x * params::degree + params::quarter];
  s_input[threadIdx.x + params::half] =
      d_input[threadIdx.x + blockIdx.x * params::degree + params::half];
  s_input[threadIdx.x + params::three_quarters] =
      d_input[threadIdx.x + blockIdx.x * params::degree +
              params::three_quarters];

  __syncthreads();

  do_SMFFT_CT_DIT<params>(s_input);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < 1024; i++)
      printf("smfft[%u] %.10f %.10f\n", i, s_input[i].x, s_input[i].y);
  }
  __syncthreads();



  __syncthreads();
  d_output[threadIdx.x + blockIdx.x * params::degree] = s_input[threadIdx.x];
  d_output[threadIdx.x + blockIdx.x * params::degree + params::quarter] =
      s_input[threadIdx.x + params::quarter];
  d_output[threadIdx.x + blockIdx.x * params::degree + params::half] =
      s_input[threadIdx.x + params::half];
  d_output[threadIdx.x + blockIdx.x * params::degree + params::three_quarters] =
      s_input[threadIdx.x + params::three_quarters];

  __syncthreads();
}

#endif // GPU_BOOTSTRAP_SMFFT_CUH

*/