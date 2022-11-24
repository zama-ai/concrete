#ifndef WOP_PBS_H
#define WOP_PBS_H

#include "cooperative_groups.h"

#include "../include/helper_cuda.h"
#include "bootstrap.h"
#include "bootstrap_low_latency.cuh"
#include "complex/operations.cuh"
#include "crypto/ggsw.cuh"
#include "crypto/torus.cuh"
#include "device.h"
#include "fft/bnsmfft.cuh"
#include "fft/smfft.cuh"
#include "fft/twiddles.cuh"
#include "keyswitch.cuh"
#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial.cuh"
#include "polynomial/polynomial_math.cuh"
#include "utils/memory.cuh"
#include "utils/timer.cuh"

template <class params> __device__ void fft(double2 *output, int16_t *input) {
  synchronize_threads_in_block();

  // Reduce the size of the FFT to be performed by storing
  // the real-valued polynomial into a complex polynomial
  real_to_complex_compressed<params>(input, output);
  synchronize_threads_in_block();

  // Switch to the FFT space
  NSMFFT_direct<HalfDegree<params>>(output);
  synchronize_threads_in_block();

  correction_direct_fft_inplace<params>(output);
  synchronize_threads_in_block();
}

template <class params> __device__ void ifft_inplace(double2 *data) {
  synchronize_threads_in_block();

  correction_inverse_fft_inplace<params>(data);
  synchronize_threads_in_block();

  NSMFFT_inverse<HalfDegree<params>>(data);
  synchronize_threads_in_block();
}

/*
 * Receives an array of GLWE ciphertexts and two indexes to ciphertexts in this
 * array, and an array of GGSW ciphertexts with a index to one ciphertext in it.
 * Compute a CMUX with these operands and writes the output to a particular
 * index of glwe_array_out.
 *
 * This function needs polynomial_size threads per block.
 *
 * - glwe_array_out: An array where the result should be written to.
 * - glwe_array_in: An array where the GLWE inputs are stored.
 * - ggsw_in: An array where the GGSW input is stored. In the fourier domain.
 * - selected_memory: An array to be used for the accumulators. Can be in the
 * shared memory or global memory.
 * - output_idx: The index of the output where the glwe ciphertext should be
 * written.
 * - input_idx1: The index of the first glwe ciphertext we will use.
 * - input_idx2: The index of the second glwe ciphertext we will use.
 * - glwe_dim: This is k.
 * - polynomial_size: size of the polynomials. This is N.
 * - base_log: log base used for the gadget matrix - B = 2^base_log (~8)
 * - level_count: number of decomposition levels in the gadget matrix (~4)
 * - ggsw_idx: The index of the GGSW we will use.
 */
template <typename Torus, typename STorus, class params>
__device__ void
cmux(Torus *glwe_array_out, Torus *glwe_array_in, double2 *ggsw_in,
     char *selected_memory, uint32_t output_idx, uint32_t input_idx1,
     uint32_t input_idx2, uint32_t glwe_dim, uint32_t polynomial_size,
     uint32_t base_log, uint32_t level_count, uint32_t ggsw_idx) {

  // Define glwe_sub
  Torus *glwe_sub_mask = (Torus *)selected_memory;
  Torus *glwe_sub_body = (Torus *)glwe_sub_mask + (ptrdiff_t)polynomial_size;

  int16_t *glwe_mask_decomposed = (int16_t *)(glwe_sub_body + polynomial_size);
  int16_t *glwe_body_decomposed =
      (int16_t *)glwe_mask_decomposed + (ptrdiff_t)polynomial_size;

  double2 *mask_res_fft = (double2 *)(glwe_body_decomposed + polynomial_size);
  double2 *body_res_fft =
      (double2 *)mask_res_fft + (ptrdiff_t)polynomial_size / 2;

  double2 *glwe_fft =
      (double2 *)body_res_fft + (ptrdiff_t)(polynomial_size / 2);

  GadgetMatrix<Torus, params> gadget(base_log, level_count);

  /////////////////////////////////////

  // glwe2-glwe1

  // Copy m0 to shared memory to preserve data
  auto m0_mask = &glwe_array_in[input_idx1 * (glwe_dim + 1) * polynomial_size];
  auto m0_body = m0_mask + polynomial_size;

  // Just gets the pointer for m1 on global memory
  auto m1_mask = &glwe_array_in[input_idx2 * (glwe_dim + 1) * polynomial_size];
  auto m1_body = m1_mask + polynomial_size;

  // Mask
  sub_polynomial<Torus, params>(glwe_sub_mask, m1_mask, m0_mask);
  // Body
  sub_polynomial<Torus, params>(glwe_sub_body, m1_body, m0_body);

  synchronize_threads_in_block();

  // Initialize the polynomial multiplication via FFT arrays
  // The polynomial multiplications happens at the block level
  // and each thread handles two or more coefficients
  int pos = threadIdx.x;
  for (int j = 0; j < params::opt / 2; j++) {
    mask_res_fft[pos].x = 0;
    mask_res_fft[pos].y = 0;
    body_res_fft[pos].x = 0;
    body_res_fft[pos].y = 0;
    pos += params::degree / params::opt;
  }

  // Subtract each glwe operand, decompose the resulting
  // polynomial coefficients to multiply each decomposed level
  // with the corresponding part of the LUT
  for (int level = 0; level < level_count; level++) {

    // Decomposition
    gadget.decompose_one_level(glwe_mask_decomposed, glwe_sub_mask, level);
    gadget.decompose_one_level(glwe_body_decomposed, glwe_sub_body, level);

    // First, perform the polynomial multiplication for the mask
    synchronize_threads_in_block();
    fft<params>(glwe_fft, glwe_mask_decomposed);

    // External product and accumulate
    // Get the piece necessary for the multiplication
    auto mask_fourier = get_ith_mask_kth_block(
        ggsw_in, ggsw_idx, 0, level, polynomial_size, glwe_dim, level_count);
    auto body_fourier = get_ith_body_kth_block(
        ggsw_in, ggsw_idx, 0, level, polynomial_size, glwe_dim, level_count);

    synchronize_threads_in_block();

    // Perform the coefficient-wise product
    synchronize_threads_in_block();
    polynomial_product_accumulate_in_fourier_domain<params, double2>(
        mask_res_fft, glwe_fft, mask_fourier);
    polynomial_product_accumulate_in_fourier_domain<params, double2>(
        body_res_fft, glwe_fft, body_fourier);

    // Now handle the polynomial multiplication for the body
    // in the same way
    synchronize_threads_in_block();
    fft<params>(glwe_fft, glwe_body_decomposed);

    // External product and accumulate
    // Get the piece necessary for the multiplication
    mask_fourier = get_ith_mask_kth_block(
        ggsw_in, ggsw_idx, 1, level, polynomial_size, glwe_dim, level_count);
    body_fourier = get_ith_body_kth_block(
        ggsw_in, ggsw_idx, 1, level, polynomial_size, glwe_dim, level_count);

    synchronize_threads_in_block();

    polynomial_product_accumulate_in_fourier_domain<params, double2>(
        mask_res_fft, glwe_fft, mask_fourier);
    polynomial_product_accumulate_in_fourier_domain<params, double2>(
        body_res_fft, glwe_fft, body_fourier);
  }

  // IFFT
  synchronize_threads_in_block();
  ifft_inplace<params>(mask_res_fft);
  ifft_inplace<params>(body_res_fft);
  synchronize_threads_in_block();

  // Write the output
  Torus *mb_mask =
      &glwe_array_out[output_idx * (glwe_dim + 1) * polynomial_size];
  Torus *mb_body = mb_mask + polynomial_size;

  int tid = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    mb_mask[tid] = m0_mask[tid];
    mb_body[tid] = m0_body[tid];
    tid += params::degree / params::opt;
  }

  add_to_torus<Torus, params>(mask_res_fft, mb_mask);
  add_to_torus<Torus, params>(body_res_fft, mb_body);
}

/**
 * Computes several CMUXes using an array of GLWE ciphertexts and a single GGSW
 * ciphertext. The GLWE ciphertexts are picked two-by-two in sequence. Each
 * thread block computes a single CMUX.
 *
 * - glwe_array_out: An array where the result should be written to.
 * - glwe_array_in: An array where the GLWE inputs are stored.
 * - ggsw_in: An array where the GGSW input is stored. In the fourier domain.
 * - device_mem: An pointer for the global memory in case the shared memory is
 * not big enough to store the accumulators.
 * - device_memory_size_per_block: Memory size needed to store all accumulators
 * for a single block.
 * - glwe_dim: This is k.
 * - polynomial_size: size of the polynomials. This is N.
 * - base_log: log base used for the gadget matrix - B = 2^base_log (~8)
 * - level_count: number of decomposition levels in the gadget matrix (~4)
 * - ggsw_idx: The index of the GGSW we will use.
 */
template <typename Torus, typename STorus, class params, sharedMemDegree SMD>
__global__ void
device_batch_cmux(Torus *glwe_array_out, Torus *glwe_array_in, double2 *ggsw_in,
                  char *device_mem, size_t device_memory_size_per_block,
                  uint32_t glwe_dim, uint32_t polynomial_size,
                  uint32_t base_log, uint32_t level_count, uint32_t ggsw_idx) {

  int cmux_idx = blockIdx.x;
  int output_idx = cmux_idx;
  int input_idx1 = (cmux_idx << 1);
  int input_idx2 = (cmux_idx << 1) + 1;

  // We use shared memory for intermediate result
  extern __shared__ char sharedmem[];
  char *selected_memory;

  if constexpr (SMD == FULLSM)
    selected_memory = sharedmem;
  else
    selected_memory = &device_mem[blockIdx.x * device_memory_size_per_block];

  cmux<Torus, STorus, params>(glwe_array_out, glwe_array_in, ggsw_in,
                              selected_memory, output_idx, input_idx1,
                              input_idx2, glwe_dim, polynomial_size, base_log,
                              level_count, ggsw_idx);
}
/*
 * This kernel executes the CMUX tree used by the hybrid packing of the WoPBS.
 *
 * Uses shared memory for intermediate results
 *
 *  - v_stream: The CUDA stream that should be used.
 *  - glwe_array_out: A device array for the output GLWE ciphertext.
 *  - ggsw_in: A device array for the GGSW ciphertexts used in each layer.
 *  - lut_vector: A device array for the GLWE ciphertexts used in the first
 * layer.
 * -  polynomial_size: size of the polynomials. This is N.
 *  - base_log: log base used for the gadget matrix - B = 2^base_log (~8)
 *  - level_count: number of decomposition levels in the gadget matrix (~4)
 *  - r: Number of layers in the tree.
 */
template <typename Torus, typename STorus, class params>
void host_cmux_tree(void *v_stream, Torus *glwe_array_out, Torus *ggsw_in,
                    Torus *lut_vector, uint32_t glwe_dimension,
                    uint32_t polynomial_size, uint32_t base_log,
                    uint32_t level_count, uint32_t r,
                    uint32_t max_shared_memory) {
  // This should be refactored to pass the gpu index as a parameter
  uint32_t gpu_index = 0;

  auto stream = static_cast<cudaStream_t *>(v_stream);
  int num_lut = (1 << r);

  cuda_initialize_twiddles(polynomial_size, 0);

  int memory_needed_per_block =
      sizeof(Torus) * polynomial_size +       // glwe_sub_mask
      sizeof(Torus) * polynomial_size +       // glwe_sub_body
      sizeof(int16_t) * polynomial_size +     // glwe_mask_decomposed
      sizeof(int16_t) * polynomial_size +     // glwe_body_decomposed
      sizeof(double2) * polynomial_size / 2 + // mask_res_fft
      sizeof(double2) * polynomial_size / 2 + // body_res_fft
      sizeof(double2) * polynomial_size / 2;  // glwe_fft

  dim3 thds(polynomial_size / params::opt, 1, 1);

  //////////////////////
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;

  double2 *d_ggsw_fft_in = (double2 *)cuda_malloc_async(
      r * ggsw_size * sizeof(double), *stream, gpu_index);

  batch_fft_ggsw_vector<Torus, STorus, params>(v_stream, d_ggsw_fft_in, ggsw_in,
                                               r, glwe_dimension,
                                               polynomial_size, level_count);

  //////////////////////

  // Allocate global memory in case parameters are too large
  char *d_mem;
  if (max_shared_memory < memory_needed_per_block) {
    d_mem = (char *)cuda_malloc_async(memory_needed_per_block * (1 << (r - 1)),
                                      *stream, gpu_index);
  } else {
    checkCudaErrors(cudaFuncSetAttribute(
        device_batch_cmux<Torus, STorus, params, FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, memory_needed_per_block));
    checkCudaErrors(
        cudaFuncSetCacheConfig(device_batch_cmux<Torus, STorus, params, FULLSM>,
                               cudaFuncCachePreferShared));
  }

  // Allocate buffers
  int glwe_size = (glwe_dimension + 1) * polynomial_size;

  Torus *d_buffer1 = (Torus *)cuda_malloc_async(
      num_lut * glwe_size * sizeof(Torus), *stream, gpu_index);
  Torus *d_buffer2 = (Torus *)cuda_malloc_async(
      num_lut * glwe_size * sizeof(Torus), *stream, gpu_index);

  checkCudaErrors(cudaMemcpyAsync(d_buffer1, lut_vector,
                                  num_lut * glwe_size * sizeof(Torus),
                                  cudaMemcpyDeviceToDevice, *stream));

  Torus *output;
  // Run the cmux tree
  for (int layer_idx = 0; layer_idx < r; layer_idx++) {
    output = (layer_idx % 2 ? d_buffer1 : d_buffer2);
    Torus *input = (layer_idx % 2 ? d_buffer2 : d_buffer1);

    int num_cmuxes = (1 << (r - 1 - layer_idx));
    dim3 grid(num_cmuxes, 1, 1);

    // walks horizontally through the leafs
    if (max_shared_memory < memory_needed_per_block)
      device_batch_cmux<Torus, STorus, params, NOSM>
          <<<grid, thds, memory_needed_per_block, *stream>>>(
              output, input, d_ggsw_fft_in, d_mem, memory_needed_per_block,
              glwe_dimension, // k
              polynomial_size, base_log, level_count,
              layer_idx // r
          );
    else
      device_batch_cmux<Torus, STorus, params, FULLSM>
          <<<grid, thds, memory_needed_per_block, *stream>>>(
              output, input, d_ggsw_fft_in, d_mem, memory_needed_per_block,
              glwe_dimension, // k
              polynomial_size, base_log, level_count,
              layer_idx // r
          );
  }

  checkCudaErrors(
      cudaMemcpyAsync(glwe_array_out, output,
                      (glwe_dimension + 1) * polynomial_size * sizeof(Torus),
                      cudaMemcpyDeviceToDevice, *stream));

  // We only need synchronization to assert that data is in glwe_array_out
  // before returning. Memory release can be added to the stream and processed
  // later.
  checkCudaErrors(cudaStreamSynchronize(*stream));

  // Free memory
  cuda_drop_async(d_ggsw_fft_in, *stream, gpu_index);
  cuda_drop_async(d_buffer1, *stream, gpu_index);
  cuda_drop_async(d_buffer2, *stream, gpu_index);
  if (max_shared_memory < memory_needed_per_block)
    cuda_drop_async(d_mem, *stream, gpu_index);
}

// only works for big lwe for ks+bs case
// state_lwe_buffer is copied from big lwe input
// shifted_lwe_buffer is scalar multiplication of lwe input
// blockIdx.x refers to input ciphertext id
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

// only works for small lwe in ks+bs case
// function copies lwe when length is not a power of two
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

// only used in extract bits for one ciphertext
// should be called with one block and one thread
// NOTE: check if putting this functionality in copy_small_lwe or
// fill_pbs_lut vector is faster
template <typename Torus>
__global__ void add_to_body(Torus *lwe, size_t lwe_dimension, Torus value) {
  lwe[blockIdx.x * (lwe_dimension + 1) + lwe_dimension] += value;
}

// Fill lut(only body) for the current bit (equivalent to trivial encryption as
// mask is 0s)
// The LUT is filled with -alpha in each coefficient where alpha =
// delta*2^{bit_idx-1}
template <typename Torus, class params>
__global__ void fill_lut_body_for_current_bit(Torus *lut, Torus value) {
  Torus *cur_poly = &lut[params::degree];
  size_t tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    cur_poly[tid] = value;
    tid += params::degree / params::opt;
  }
}

// Add alpha where alpha = delta*2^{bit_idx-1} to end up with an encryption of 0
// if the extracted bit was 0 and 1 in the other case
//
// Remove the extracted bit from the state LWE to get a 0 at the extracted bit
// location.
//
// Shift on padding bit for next iteration, that's why
// alpha= 1ll << (ciphertext_n_bits - delta_log - bit_idx - 2) is used
// instead of alpha= 1ll << (ciphertext_n_bits - delta_log - bit_idx - 1)
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

template <typename Torus, class params>
__host__ void host_extract_bits(
    void *v_stream, Torus *list_lwe_array_out, Torus *lwe_array_in,
    Torus *lwe_array_in_buffer, Torus *lwe_array_in_shifted_buffer,
    Torus *lwe_array_out_ks_buffer, Torus *lwe_array_out_pbs_buffer,
    Torus *lut_pbs, uint32_t *lut_vector_indexes, Torus *ksk,
    double2 *fourier_bsk, uint32_t number_of_bits, uint32_t delta_log,
    uint32_t lwe_dimension_in, uint32_t lwe_dimension_out,
    uint32_t base_log_bsk, uint32_t level_count_bsk, uint32_t base_log_ksk,
    uint32_t level_count_ksk, uint32_t number_of_samples) {

  auto stream = static_cast<cudaStream_t *>(v_stream);
  uint32_t ciphertext_n_bits = sizeof(Torus) * 8;

  int blocks = 1;
  int threads = params::degree / params::opt;

  copy_and_shift_lwe<Torus, params><<<blocks, threads, 0, *stream>>>(
      lwe_array_in_buffer, lwe_array_in_shifted_buffer, lwe_array_in,
      1ll << (ciphertext_n_bits - delta_log - 1));
  checkCudaErrors(cudaGetLastError());

  for (int bit_idx = 0; bit_idx < number_of_bits; bit_idx++) {
    cuda_keyswitch_lwe_ciphertext_vector(
        v_stream, lwe_array_out_ks_buffer, lwe_array_in_shifted_buffer, ksk,
        lwe_dimension_in, lwe_dimension_out, base_log_ksk, level_count_ksk, 1);

    copy_small_lwe<<<1, 256, 0, *stream>>>(
        list_lwe_array_out, lwe_array_out_ks_buffer, lwe_dimension_out + 1,
        number_of_bits, number_of_bits - bit_idx - 1);
    checkCudaErrors(cudaGetLastError());

    if (bit_idx == number_of_bits - 1) {
      break;
    }

    add_to_body<Torus><<<1, 1, 0, *stream>>>(lwe_array_out_ks_buffer,
                                             lwe_dimension_out,
                                             1ll << (ciphertext_n_bits - 2));
    checkCudaErrors(cudaGetLastError());

    fill_lut_body_for_current_bit<Torus, params>
        <<<blocks, threads, 0, *stream>>>(
            lut_pbs, 0ll - 1ll << (delta_log - 1 + bit_idx));
    checkCudaErrors(cudaGetLastError());

    host_bootstrap_low_latency<Torus, params>(
        v_stream, lwe_array_out_pbs_buffer, lut_pbs, lut_vector_indexes,
        lwe_array_out_ks_buffer, fourier_bsk, lwe_dimension_out,
        lwe_dimension_in, base_log_bsk, level_count_bsk, number_of_samples, 1);

    add_sub_and_mul_lwe<Torus, params><<<1, threads, 0, *stream>>>(
        lwe_array_in_shifted_buffer, lwe_array_in_buffer,
        lwe_array_out_pbs_buffer, 1ll << (delta_log - 1 + bit_idx),
        1ll << (ciphertext_n_bits - delta_log - bit_idx - 2));
    checkCudaErrors(cudaGetLastError());
  }
}

/*
 * Receives "tau" GLWE ciphertexts as LUTs and "mbr_size" GGSWs. Each block
 * computes the blind rotation loop + sample extraction for a single LUT.
 * Writes the lwe output to lwe_out.
 *
 * This function needs polynomial_size/params::opt threads per block and tau
 * blocks
 *
 * - lwe_out: An array of lwe ciphertexts. The outcome is written here.
 * - glwe_in: An array of "tau" GLWE ciphertexts. These are the LUTs.
 * - ggsw_in: An array of "mbr_size" GGSWs in the fourier domain.
 * - mbr_size: The number of GGSWs.
 * - glwe_dim: This is k.
 * - polynomial_size: size of the polynomials. This is N.
 * - base_log: log base used for the gadget matrix - B = 2^base_log (~8)
 * - l_gadget: number of decomposition levels in the gadget matrix (~4)
 * - device_memory_size_per_sample: Amount of (shared/global) memory used for
 * the accumulators.
 * - device_mem: An array to be used for the accumulators. Can be in the shared
 * memory or global memory.
 */
template <typename Torus, typename STorus, class params, sharedMemDegree SMD>
__global__ void device_blind_rotation_and_sample_extraction(
    Torus *lwe_out, Torus *glwe_in, double2 *ggsw_in, // m^BR
    uint32_t mbr_size, uint32_t glwe_dim, uint32_t polynomial_size,
    uint32_t base_log, uint32_t l_gadget, size_t device_memory_size_per_sample,
    char *device_mem) {

  // We use shared memory for intermediate result
  extern __shared__ char sharedmem[];
  char *selected_memory;

  if constexpr (SMD == FULLSM)
    selected_memory = sharedmem;
  else
    selected_memory = &device_mem[blockIdx.x * device_memory_size_per_sample];

  Torus *accumulator_c0 = (Torus *)selected_memory;
  Torus *accumulator_c1 = (Torus *)accumulator_c0 + 2 * polynomial_size;

  // Input LUT
  auto mi = &glwe_in[blockIdx.x * (glwe_dim + 1) * polynomial_size];
  int tid = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    accumulator_c0[tid] = mi[tid];
    accumulator_c0[tid + params::degree] = mi[tid + params::degree];
    tid += params::degree / params::opt;
  }

  int monomial_degree = 0;
  for (int i = mbr_size - 1; i >= 0; i--) {
    synchronize_threads_in_block();

    // Compute x^ai * ACC
    // Body
    divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                          params::degree / params::opt>(
        accumulator_c1, accumulator_c0, (1 << monomial_degree), false);
    // Mask
    divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                          params::degree / params::opt>(
        accumulator_c1 + polynomial_size, accumulator_c0 + polynomial_size,
        (1 << monomial_degree), false);

    monomial_degree += 1;

    // ACC = CMUX ( Ci, x^ai * ACC, ACC )
    synchronize_threads_in_block();
    cmux<Torus, STorus, params>(accumulator_c0, accumulator_c0, ggsw_in,
                                (char *)(accumulator_c0 + 4 * polynomial_size),
                                0, 0, 1, glwe_dim, polynomial_size, base_log,
                                l_gadget, i);
  }
  synchronize_threads_in_block();

  // Write the output
  auto block_lwe_out = &lwe_out[blockIdx.x * (polynomial_size + 1)];

  // The blind rotation for this block is over
  // Now we can perform the sample extraction: for the body it's just
  // the resulting constant coefficient of the accumulator
  // For the mask it's more complicated
  sample_extract_mask<Torus, params>(block_lwe_out, accumulator_c0);
  sample_extract_body<Torus, params>(block_lwe_out,
                                     accumulator_c0 + polynomial_size);
}

template <typename Torus, typename STorus, class params>
void host_blind_rotate_and_sample_extraction(
    void *v_stream, Torus *lwe_out, Torus *ggsw_in, Torus *lut_vector,
    uint32_t mbr_size, uint32_t tau, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t l_gadget,
    uint32_t max_shared_memory) {
  // This should be refactored to pass the gpu index as a parameter
  uint32_t gpu_index = 0;

  assert(glwe_dimension ==
         1); // For larger k we will need to adjust the mask size

  auto stream = static_cast<cudaStream_t *>(v_stream);

  int memory_needed_per_block =
      sizeof(Torus) * polynomial_size +       // accumulator_c0 mask
      sizeof(Torus) * polynomial_size +       // accumulator_c0 body
      sizeof(Torus) * polynomial_size +       // accumulator_c1 mask
      sizeof(Torus) * polynomial_size +       // accumulator_c1 body
      sizeof(Torus) * polynomial_size +       // glwe_sub_mask
      sizeof(Torus) * polynomial_size +       // glwe_sub_body
      sizeof(int16_t) * polynomial_size +     // glwe_mask_decomposed
      sizeof(int16_t) * polynomial_size +     // glwe_body_decomposed
      sizeof(double2) * polynomial_size / 2 + // mask_res_fft
      sizeof(double2) * polynomial_size / 2 + // body_res_fft
      sizeof(double2) * polynomial_size / 2;  // glwe_fft

  char *d_mem;
  if (max_shared_memory < memory_needed_per_block)
    d_mem = (char *)cuda_malloc_async(memory_needed_per_block * tau, *stream,
                                      gpu_index);
  else {
    checkCudaErrors(cudaFuncSetAttribute(
        device_blind_rotation_and_sample_extraction<Torus, STorus, params,
                                                    FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, memory_needed_per_block));
    checkCudaErrors(cudaFuncSetCacheConfig(
        device_blind_rotation_and_sample_extraction<Torus, STorus, params,
                                                    FULLSM>,
        cudaFuncCachePreferShared));
  }

  // Applying the FFT on m^br
  int ggsw_size =
      polynomial_size * (glwe_dimension + 1) * (glwe_dimension + 1) * l_gadget;
  double2 *d_ggsw_fft_in = (double2 *)cuda_malloc_async(
      mbr_size * ggsw_size * sizeof(double), *stream, gpu_index);

  batch_fft_ggsw_vector<Torus, STorus, params>(v_stream, d_ggsw_fft_in, ggsw_in,
                                               mbr_size, glwe_dimension,
                                               polynomial_size, l_gadget);
  checkCudaErrors(cudaGetLastError());

  //
  dim3 thds(polynomial_size / params::opt, 1, 1);
  dim3 grid(tau, 1, 1);

  if (max_shared_memory < memory_needed_per_block)
    device_blind_rotation_and_sample_extraction<Torus, STorus, params, NOSM>
        <<<grid, thds, 0, *stream>>>(lwe_out, lut_vector, d_ggsw_fft_in,
                                     mbr_size,
                                     glwe_dimension, // k
                                     polynomial_size, base_log, l_gadget,
                                     memory_needed_per_block, d_mem);
  else
    device_blind_rotation_and_sample_extraction<Torus, STorus, params, FULLSM>
        <<<grid, thds, memory_needed_per_block, *stream>>>(
            lwe_out, lut_vector, d_ggsw_fft_in, mbr_size,
            glwe_dimension, // k
            polynomial_size, base_log, l_gadget, memory_needed_per_block,
            d_mem);
  checkCudaErrors(cudaGetLastError());

  //
  cuda_drop_async(d_ggsw_fft_in, *stream, gpu_index);
  if (max_shared_memory < memory_needed_per_block)
    cuda_drop_async(d_mem, *stream, gpu_index);
}

#endif // WO_PBS_H
