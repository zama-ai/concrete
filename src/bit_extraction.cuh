#ifndef BIT_EXTRACT_H
#define BIT_EXTRACT_H

#include "cooperative_groups.h"

#include "../include/helper_cuda.h"
#include "bootstrap.h"
#include "bootstrap_low_latency.cuh"
#include "device.h"
#include "keyswitch.cuh"
#include "polynomial/parameters.cuh"
#include "utils/timer.cuh"

/*
 * Function copies batch lwe input to two different buffers,
 * one is shifted by value
 * one is copied without any modification
 * works for ciphertexts with sizes supported by params::degree
 */
template <typename Torus, class params>
__global__ void copy_and_shift_lwe(Torus *dst_copy, Torus *dst_shift,
                                   Torus *src, Torus value) {
  int blockId = blockIdx.x;
  int tid = threadIdx.x;
  auto cur_dst_copy = &dst_copy[blockId * (params::degree + 1)];
  auto cur_dst_shift = &dst_shift[blockId * (params::degree + 1)];
  auto cur_src = &src[blockId * (params::degree + 1)];

#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    cur_dst_copy[tid] = cur_src[tid];
    cur_dst_shift[tid] = cur_src[tid] * value;
    tid += params::degree / params::opt;
  }

  if (threadIdx.x == params::degree / params::opt - 1) {
    cur_dst_copy[params::degree] = cur_src[params::degree];
    cur_dst_shift[params::degree] = cur_src[params::degree] * value;
  }
}

/*
 * Function copies batch of lwe to lwe when size is not supported by
 * params::degree
 */
template <typename Torus>
__global__ void copy_small_lwe(Torus *dst, Torus *src, uint32_t small_lwe_size,
                               uint32_t number_of_bits, uint32_t lwe_id) {

  size_t blockId = blockIdx.x;
  size_t threads_per_block = blockDim.x;
  size_t opt = small_lwe_size / threads_per_block;
  size_t rem = small_lwe_size & (threads_per_block - 1);

  auto cur_lwe_list = &dst[blockId * small_lwe_size * number_of_bits];
  auto cur_dst = &cur_lwe_list[lwe_id * small_lwe_size];
  auto cur_src = &src[blockId * small_lwe_size];

  size_t tid = threadIdx.x;
  for (int i = 0; i < opt; i++) {
    cur_dst[tid] = cur_src[tid];
    tid += threads_per_block;
  }

  if (threadIdx.x < rem)
    cur_dst[tid] = cur_src[tid];
}

/*
 * Function used to wrapping add value on the body of ciphertexts,
 * should be called with blocksize.x = 1;
 * blickIdx.x refers id of ciphertext
 * NOTE: check if putting thi functionality in copy_small_lwe or fill_pbs_lut
 * is faster
 */
template <typename Torus>
__global__ void add_to_body(Torus *lwe, size_t lwe_dimension, Torus value) {
  lwe[blockIdx.x * (lwe_dimension + 1) + lwe_dimension] += value;
}

/*
 * Add alpha where alpha = delta*2^{bit_idx-1} to end up with an encryption of 0
 * if the extracted bit was 0 and 1 in the other case
 * Remove the extracted bit from the state LWE to get a 0 at the extracted bit
 * location.
 * Shift on padding bit for next iteration, that's why
 * alpha= 1ll << (ciphertext_n_bits - delta_log - bit_idx - 2) is used
 * instead of alpha= 1ll << (ciphertext_n_bits - delta_log - bit_idx - 1)
 */
template <typename Torus, class params>
__global__ void add_sub_and_mul_lwe(Torus *shifted_lwe, Torus *state_lwe,
                                    Torus *pbs_lwe_array_out, Torus add_value,
                                    Torus mul_value) {
  size_t tid = threadIdx.x;
  size_t blockId = blockIdx.x;
  auto cur_shifted_lwe = &shifted_lwe[blockId * (params::degree + 1)];
  auto cur_state_lwe = &state_lwe[blockId * (params::degree + 1)];
  auto cur_pbs_lwe_array_out =
      &pbs_lwe_array_out[blockId * (params::degree + 1)];
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    cur_shifted_lwe[tid] = cur_state_lwe[tid] -= cur_pbs_lwe_array_out[tid];
    cur_shifted_lwe[tid] *= mul_value;
    tid += params::degree / params::opt;
  }

  if (threadIdx.x == params::degree / params::opt - 1) {
    cur_shifted_lwe[params::degree] = cur_state_lwe[params::degree] -=
        (cur_pbs_lwe_array_out[params::degree] + add_value);
    cur_shifted_lwe[params::degree] *= mul_value;
  }
}

/*
 * Fill lut(only body) for the current bit, equivalent to trivial encryption as
 * msk is 0s
 * blockIdx.x refers id of lut vector
 */
template <typename Torus, class params>
__global__ void fill_lut_body_for_current_bit(Torus *lut, Torus value) {

  Torus *cur_poly = &lut[blockIdx.x * 2 * params::degree + params::degree];
  size_t tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    cur_poly[tid] = value;
    tid += params::degree / params::opt;
  }
}

/*
 * Host function for cuda extract bits.
 * it executes device functions in specific order and manages
 * parallelism
 */
template <typename Torus, class params>
__host__ void host_extract_bits(
    void *v_stream, uint32_t gpu_index, Torus *list_lwe_array_out,
    Torus *lwe_array_in, Torus *lwe_array_in_buffer,
    Torus *lwe_array_in_shifted_buffer, Torus *lwe_array_out_ks_buffer,
    Torus *lwe_array_out_pbs_buffer, Torus *lut_pbs,
    uint32_t *lut_vector_indexes, Torus *ksk, double2 *fourier_bsk,
    uint32_t number_of_bits, uint32_t delta_log, uint32_t lwe_dimension_in,
    uint32_t lwe_dimension_out, uint32_t base_log_bsk, uint32_t level_count_bsk,
    uint32_t base_log_ksk, uint32_t level_count_ksk, uint32_t number_of_samples,
    uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);
  uint32_t ciphertext_n_bits = sizeof(Torus) * 8;

  int blocks = 1;
  int threads = params::degree / params::opt;

  // shift lwe on padding bit and copy in new buffer
  copy_and_shift_lwe<Torus, params><<<blocks, threads, 0, *stream>>>(
      lwe_array_in_buffer, lwe_array_in_shifted_buffer, lwe_array_in,
      1ll << (ciphertext_n_bits - delta_log - 1));
  checkCudaErrors(cudaGetLastError());

  for (int bit_idx = 0; bit_idx < number_of_bits; bit_idx++) {
    cuda_keyswitch_lwe_ciphertext_vector(
        v_stream, gpu_index, lwe_array_out_ks_buffer,
        lwe_array_in_shifted_buffer, ksk, lwe_dimension_in, lwe_dimension_out,
        base_log_ksk, level_count_ksk, 1);

    copy_small_lwe<<<1, 256, 0, *stream>>>(
        list_lwe_array_out, lwe_array_out_ks_buffer, lwe_dimension_out + 1,
        number_of_bits, number_of_bits - bit_idx - 1);
    checkCudaErrors(cudaGetLastError());

    if (bit_idx == number_of_bits - 1) {
      break;
    }

    // Add q/4 to center the error while computing a negacyclic LUT
    add_to_body<Torus><<<1, 1, 0, *stream>>>(lwe_array_out_ks_buffer,
                                             lwe_dimension_out,
                                             1ll << (ciphertext_n_bits - 2));
    checkCudaErrors(cudaGetLastError());

    // Fill lut for the current bit (equivalent to trivial encryption as mask is
    // 0s) The LUT is filled with -alpha in each coefficient where alpha =
    // delta*2^{bit_idx-1}
    fill_lut_body_for_current_bit<Torus, params>
        <<<blocks, threads, 0, *stream>>>(
            lut_pbs, 0ll - 1ll << (delta_log - 1 + bit_idx));
    checkCudaErrors(cudaGetLastError());

    host_bootstrap_low_latency<Torus, params>(
        v_stream, gpu_index, lwe_array_out_pbs_buffer, lut_pbs,
        lut_vector_indexes, lwe_array_out_ks_buffer, fourier_bsk,
        lwe_dimension_out, lwe_dimension_in, base_log_bsk, level_count_bsk,
        number_of_samples, 1, max_shared_memory);

    // Add alpha where alpha = delta*2^{bit_idx-1} to end up with an encryption
    // of 0 if the extracted bit was 0 and 1 in the other case
    add_sub_and_mul_lwe<Torus, params><<<1, threads, 0, *stream>>>(
        lwe_array_in_shifted_buffer, lwe_array_in_buffer,
        lwe_array_out_pbs_buffer, 1ll << (delta_log - 1 + bit_idx),
        1ll << (ciphertext_n_bits - delta_log - bit_idx - 2));
    checkCudaErrors(cudaGetLastError());
  }
}

#endif // BIT_EXTRACT_H
