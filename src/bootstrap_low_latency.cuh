#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

#ifndef LOWLAT_PBS_H
#define LOWLAT_PBS_H

#include "cooperative_groups.h"

#include "../include/helper_cuda.h"
#include "bootstrap.h"
#include "complex/operations.cuh"
#include "crypto/gadget.cuh"
#include "crypto/torus.cuh"
#include "device.h"
#include "fft/bnsmfft.cuh"
#include "fft/smfft.cuh"
#include "fft/twiddles.cuh"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial.cuh"
#include "polynomial/polynomial_math.cuh"
#include "utils/memory.cuh"
#include "utils/timer.cuh"

// Cooperative groups are used in the low latency PBS
using namespace cooperative_groups;
namespace cg = cooperative_groups;

template <typename Torus, class params>
__device__ void
mul_ggsw_glwe(Torus *accumulator, double2 *fft, double2 *mask_join_buffer,
              double2 *body_join_buffer, double2 *bootstrapping_key,
              int polynomial_size, int level_count, int iteration,
              grid_group &grid) {

  // Switch to the FFT space
  NSMFFT_direct<HalfDegree<params>>(fft);
  synchronize_threads_in_block();

  correction_direct_fft_inplace<params>(fft);
  synchronize_threads_in_block();

  // Get the pieces of the bootstrapping key that will be needed for the
  // external product; blockIdx.x is the ID of the block that's executing
  // this function, so we end up getting the lines of the bootstrapping key
  // needed to perform the external product in this block (corresponding to
  // the same decomposition level)

  auto bsk_mask_slice =
      get_ith_mask_kth_block(bootstrapping_key, iteration, blockIdx.y,
                             blockIdx.x, polynomial_size, 1, level_count);
  auto bsk_body_slice =
      get_ith_body_kth_block(bootstrapping_key, iteration, blockIdx.y,
                             blockIdx.x, polynomial_size, 1, level_count);

  // Perform the matrix multiplication between the GGSW and the GLWE,
  // each block operating on a single level for mask and body

  auto first_processed_bsk =
      (blockIdx.y == 0) ? bsk_mask_slice : bsk_body_slice;
  auto second_processed_bsk =
      (blockIdx.y == 0) ? bsk_body_slice : bsk_mask_slice;

  auto first_processed_acc =
      (blockIdx.y == 0) ? &mask_join_buffer[params::degree / 2 * blockIdx.x]
                        : &body_join_buffer[params::degree / 2 * blockIdx.x];
  auto second_processed_acc =
      (blockIdx.y == 0) ? &body_join_buffer[params::degree / 2 * blockIdx.x]
                        : &mask_join_buffer[params::degree / 2 * blockIdx.x];

  int tid = threadIdx.x;

  // first product
  for (int i = 0; i < params::opt / 2; i++) {
    first_processed_acc[tid] = fft[tid] * first_processed_bsk[tid];
    tid += params::degree / params::opt;
  }

  grid.sync();
  tid = threadIdx.x;
  // second product
  for (int i = 0; i < params::opt / 2; i++) {
    second_processed_acc[tid] += fft[tid] * second_processed_bsk[tid];
    tid += params::degree / params::opt;
  }

  // -----------------------------------------------------------------

  // All blocks are synchronized here; after this sync, *_join_buffer has the
  // values needed from every other block
  grid.sync();

  auto src_acc = (blockIdx.y == 0) ? mask_join_buffer : body_join_buffer;

  // copy first product into fft buffer
  tid = threadIdx.x;
  for (int i = 0; i < params::opt / 2; i++) {
    fft[tid] = src_acc[tid];
    tid += params::degree / params::opt;
  }
  synchronize_threads_in_block();

  // accumulate rest of the products into fft buffer
  for (int l = 1; l < gridDim.x; l++) {
    auto cur_src_acc = &src_acc[l * params::degree / 2];
    tid = threadIdx.x;
    for (int i = 0; i < params::opt / 2; i++) {
      fft[tid] += cur_src_acc[tid];
      tid += params::degree / params::opt;
    }
  }

  synchronize_threads_in_block();

  correction_inverse_fft_inplace<params>(fft);
  synchronize_threads_in_block();

  // Perform the inverse FFT on the result of the GGSW x GLWE and add to the
  // accumulator
  NSMFFT_inverse<HalfDegree<params>>(fft);
  synchronize_threads_in_block();

  add_to_torus<Torus, params>(fft, accumulator);

  __syncthreads();
}

template <typename Torus, class params, sharedMemDegree SMD>
/*
 * Kernel launched by the low latency version of the
 * bootstrapping, that uses cooperative groups
 * lwe_array_out vector of output lwe s, with length
 * (polynomial_size+1)*num_samples lut_vector - vector of look up tables with
 * length  polynomial_size * num_samples lut_vector_indexes - mapping between
 * lwe_array_in and lut_vector lwe_array_in
 * - vector of lwe inputs with length (lwe_dimension + 1) * num_samples
 *
 */
__global__ void device_bootstrap_low_latency(
    Torus *lwe_array_out, Torus *lut_vector, Torus *lwe_array_in,
    double2 *bootstrapping_key, double2 *mask_join_buffer,
    double2 *body_join_buffer, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t base_log, uint32_t level_count, char *device_mem,
    int device_memory_size_per_block) {

  grid_group grid = this_grid();

  // We use shared memory for the polynomials that are used often during the
  // bootstrap, since shared memory is kept in L1 cache and accessing it is
  // much faster than global memory
  extern __shared__ char sharedmem[];
  char *selected_memory;
  int block_index =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

  if constexpr (SMD == FULLSM)
    selected_memory = sharedmem;
  else
    selected_memory = &device_mem[block_index * device_memory_size_per_block];

  Torus *accumulator = (Torus *)selected_memory;
  Torus *accumulator_rotated =
      (Torus *)accumulator + (ptrdiff_t)polynomial_size;
  double2 *accumulator_fft =
      (double2 *)accumulator_rotated +
      polynomial_size / (sizeof(double2) / sizeof(Torus));
  if constexpr (SMD == PARTIALSM)
    accumulator_fft = (double2 *)sharedmem;

  // The third dimension of the block is used to determine on which ciphertext
  // this block is operating, in the case of batch bootstraps
  auto block_lwe_array_in = &lwe_array_in[blockIdx.z * (lwe_dimension + 1)];

  auto block_lut_vector = &lut_vector[blockIdx.z * params::degree * 2];

  auto block_mask_join_buffer =
      &mask_join_buffer[blockIdx.z * level_count * params::degree / 2];
  auto block_body_join_buffer =
      &body_join_buffer[blockIdx.z * level_count * params::degree / 2];

  // Since the space is L1 cache is small, we use the same memory location for
  // the rotated accumulator and the fft accumulator, since we know that the
  // rotated array is not in use anymore by the time we perform the fft

  // Put "b" in [0, 2N[
  Torus b_hat = 0;
  rescale_torus_element(block_lwe_array_in[lwe_dimension], b_hat,
                        2 * params::degree);

  if (blockIdx.y == 0) {
    divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                          params::degree / params::opt>(
        accumulator, block_lut_vector, b_hat, false);
  } else {
    divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                          params::degree / params::opt>(
        accumulator, &block_lut_vector[params::degree], b_hat, false);
  }

  for (int i = 0; i < lwe_dimension; i++) {
    synchronize_threads_in_block();

    // Put "a" in [0, 2N[
    Torus a_hat = 0;
    rescale_torus_element(block_lwe_array_in[i], a_hat,
                          2 * params::degree); // 2 * params::log2_degree + 1);

    // Perform ACC * (X^ä - 1)
    multiply_by_monomial_negacyclic_and_sub_polynomial<
        Torus, params::opt, params::degree / params::opt>(
        accumulator, accumulator_rotated, a_hat);

    // Perform a rounding to increase the accuracy of the
    // bootstrapped ciphertext
    round_to_closest_multiple_inplace<Torus, params::opt,
                                      params::degree / params::opt>(
        accumulator_rotated, base_log, level_count);

    synchronize_threads_in_block();

    // Decompose the accumulator. Each block gets one level of the
    // decomposition, for the mask and the body (so block 0 will have the
    // accumulator decomposed at level 0, 1 at 1, etc.)
    GadgetMatrix<Torus, params> gadget_acc(base_log, level_count,
                                           accumulator_rotated);
    gadget_acc.decompose_and_compress_level(accumulator_fft, blockIdx.x);

    // We are using the same memory space for accumulator_fft and
    // accumulator_rotated, so we need to synchronize here to make sure they
    // don't modify the same memory space at the same time
    synchronize_threads_in_block();
    // Perform G^-1(ACC) * GGSW -> GLWE
    mul_ggsw_glwe<Torus, params>(accumulator, accumulator_fft,
                                 block_mask_join_buffer, block_body_join_buffer,
                                 bootstrapping_key, polynomial_size,
                                 level_count, i, grid);
  }

  auto block_lwe_array_out = &lwe_array_out[blockIdx.z * (polynomial_size + 1)];

  if (blockIdx.x == 0 && blockIdx.y == 0) {
    // Perform a sample extract. At this point, all blocks have the result, but
    // we do the computation at block 0 to avoid waiting for extra blocks, in
    // case they're not synchronized
    sample_extract_mask<Torus, params>(block_lwe_array_out, accumulator);
  } else if (blockIdx.x == 0 && blockIdx.y == 1) {
    sample_extract_body<Torus, params>(block_lwe_array_out, accumulator);
  }
}

/*
 * Host wrapper to the low latency version
 * of bootstrapping
 */
template <typename Torus, class params>
__host__ void host_bootstrap_low_latency(
    void *v_stream, uint32_t gpu_index, Torus *lwe_array_out, Torus *lut_vector,
    uint32_t *lut_vector_indexes, Torus *lwe_array_in,
    double2 *bootstrapping_key, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t num_lut_vectors,
    uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  int buffer_size_per_gpu = level_count * input_lwe_ciphertext_count *
                            polynomial_size / 2 * sizeof(double2);
  double2 *mask_buffer_fft =
      (double2 *)cuda_malloc_async(buffer_size_per_gpu, *stream, gpu_index);
  double2 *body_buffer_fft =
      (double2 *)cuda_malloc_async(buffer_size_per_gpu, *stream, gpu_index);

  // With SM each block corresponds to either the mask or body, no need to
  // duplicate data for each
  int SM_FULL = sizeof(Torus) * polynomial_size +      // accumulator_rotated
                sizeof(Torus) * polynomial_size +      // accumulator
                sizeof(double2) * polynomial_size / 2; // accumulator fft

  int SM_PART =
      sizeof(double2) * polynomial_size / 2; // accumulator fft mask & body

  int DM_FULL = SM_FULL;

  int DM_PART = DM_FULL - SM_PART;

  char *d_mem;

  int thds = polynomial_size / params::opt;
  dim3 grid(level_count, 2, input_lwe_ciphertext_count);

  void *kernel_args[12];
  kernel_args[0] = &lwe_array_out;
  kernel_args[1] = &lut_vector;
  kernel_args[2] = &lwe_array_in;
  kernel_args[3] = &bootstrapping_key;
  kernel_args[4] = &mask_buffer_fft;
  kernel_args[5] = &body_buffer_fft;
  kernel_args[6] = &lwe_dimension;
  kernel_args[7] = &polynomial_size;
  kernel_args[8] = &base_log;
  kernel_args[9] = &level_count;
  kernel_args[10] = &d_mem;

  if (max_shared_memory < SM_PART) {
    kernel_args[11] = &DM_FULL;
    checkCudaErrors(cudaGetLastError());
    d_mem = (char *)cuda_malloc_async(DM_FULL * input_lwe_ciphertext_count *
                                          level_count * 2,
                                      *stream, gpu_index);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaLaunchCooperativeKernel(
        (void *)device_bootstrap_low_latency<Torus, params, NOSM>, grid, thds,
        (void **)kernel_args, 0, *stream));
  } else if (max_shared_memory < SM_FULL) {
    kernel_args[11] = &DM_PART;
    d_mem = (char *)cuda_malloc_async(DM_PART * input_lwe_ciphertext_count *
                                          level_count * 2,
                                      *stream, gpu_index);
    checkCudaErrors(cudaFuncSetAttribute(
        device_bootstrap_low_latency<Torus, params, PARTIALSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SM_PART));
    cudaFuncSetCacheConfig(
        device_bootstrap_low_latency<Torus, params, PARTIALSM>,
        cudaFuncCachePreferShared);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaLaunchCooperativeKernel(
        (void *)device_bootstrap_low_latency<Torus, params, PARTIALSM>, grid,
        thds, (void **)kernel_args, SM_PART, *stream));

  } else {
    int DM_NONE = 0;
    kernel_args[11] = &DM_NONE;
    d_mem = (char *)cuda_malloc_async(0, *stream, gpu_index);
    checkCudaErrors(cudaFuncSetAttribute(
        device_bootstrap_low_latency<Torus, params, FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SM_FULL));
    cudaFuncSetCacheConfig(device_bootstrap_low_latency<Torus, params, FULLSM>,
                           cudaFuncCachePreferShared);
    checkCudaErrors(cudaLaunchCooperativeKernel(
        (void *)device_bootstrap_low_latency<Torus, params, FULLSM>, grid, thds,
        (void **)kernel_args, SM_FULL, *stream));
  }

  checkCudaErrors(cudaGetLastError());
  // Synchronize the streams before copying the result to lwe_array_out at the
  // right place
  cudaStreamSynchronize(*stream);
  cuda_drop_async(mask_buffer_fft, *stream, gpu_index);
  cuda_drop_async(body_buffer_fft, *stream, gpu_index);
  cuda_drop_async(d_mem, *stream, gpu_index);
}

#endif // LOWLAT_PBS_H
