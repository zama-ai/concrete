#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

#ifndef LOWLAT_PBS_H
#define LOWLAT_PBS_H

#include "cooperative_groups.h"

#include "bootstrap.h"
#include "complex/operations.cuh"
#include "crypto/gadget.cuh"
#include "crypto/torus.cuh"
#include "device.h"
#include "fft/bnsmfft.cuh"
#include "fft/twiddles.cuh"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial.cuh"
#include "polynomial/polynomial_math.cuh"
#include "utils/timer.cuh"

// Cooperative groups are used in the low latency PBS
using namespace cooperative_groups;
namespace cg = cooperative_groups;

template <typename Torus, class params>
__device__ void mul_ggsw_glwe(Torus *accumulator, double2 *fft,
                              double2 *join_buffer, double2 *bootstrapping_key,
                              int polynomial_size, uint32_t glwe_dimension,
                              int level_count, int iteration,
                              grid_group &grid) {

  // Switch to the FFT space
  NSMFFT_direct<HalfDegree<params>>(fft);
  synchronize_threads_in_block();

  // Get the pieces of the bootstrapping key that will be needed for the
  // external product; blockIdx.x is the ID of the block that's executing
  // this function, so we end up getting the lines of the bootstrapping key
  // needed to perform the external product in this block (corresponding to
  // the same decomposition level)
  auto bsk_slice = get_ith_mask_kth_block(
      bootstrapping_key, iteration, blockIdx.y, blockIdx.x, polynomial_size,
      glwe_dimension, level_count);

  // Selects all GLWEs in a particular decomposition level
  auto level_join_buffer =
      join_buffer + blockIdx.x * (glwe_dimension + 1) * params::degree / 2;

  // Perform the matrix multiplication between the GGSW and the GLWE,
  // each block operating on a single level for mask and body

  // The first product is used to initialize level_join_buffer
  auto bsk_poly = bsk_slice + blockIdx.y * params::degree / 2;
  auto buffer_slice = level_join_buffer + blockIdx.y * params::degree / 2;

  int tid = threadIdx.x;
  for (int i = 0; i < params::opt / 2; i++) {
    buffer_slice[tid] = fft[tid] * bsk_poly[tid];
    tid += params::degree / params::opt;
  }

  grid.sync();

  // Continues multiplying fft by every polynomial in that particular bsk level
  // Each y-block accumulates in a different polynomial at each iteration
  for (int j = 1; j < (glwe_dimension + 1); j++) {
    int idx = (j + blockIdx.y) % (glwe_dimension + 1);

    auto bsk_poly = bsk_slice + idx * params::degree / 2;
    auto buffer_slice = level_join_buffer + idx * params::degree / 2;

    int tid = threadIdx.x;
    for (int i = 0; i < params::opt / 2; i++) {
      buffer_slice[tid] += fft[tid] * bsk_poly[tid];
      tid += params::degree / params::opt;
    }
    grid.sync();
  }

  // -----------------------------------------------------------------
  // All blocks are synchronized here; after this sync, level_join_buffer has
  // the values needed from every other block

  auto src_acc = join_buffer + blockIdx.y * params::degree / 2;

  // copy first product into fft buffer
  tid = threadIdx.x;
  for (int i = 0; i < params::opt / 2; i++) {
    fft[tid] = src_acc[tid];
    tid += params::degree / params::opt;
  }
  synchronize_threads_in_block();

  // accumulate rest of the products into fft buffer
  for (int l = 1; l < gridDim.x; l++) {
    auto cur_src_acc = &src_acc[l * (glwe_dimension + 1) * params::degree / 2];
    tid = threadIdx.x;
    for (int i = 0; i < params::opt / 2; i++) {
      fft[tid] += cur_src_acc[tid];
      tid += params::degree / params::opt;
    }
  }

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
 *
 * - lwe_array_out: vector of output lwe s, with length
 * (glwe_dimension * polynomial_size+1)*num_samples
 * - lut_vector: vector of look up tables with
 * length  (glwe_dimension+1) * polynomial_size * num_samples
 * - lut_vector_indexes: mapping between lwe_array_in and lut_vector
 * lwe_array_in: vector of lwe inputs with length (lwe_dimension + 1) *
 * num_samples
 *
 * Each y-block computes one element of the lwe_array_out.
 */
__global__ void device_bootstrap_low_latency(
    Torus *lwe_array_out, Torus *lut_vector, Torus *lut_vector_indexes,
    Torus *lwe_array_in, double2 *bootstrapping_key, double2 *join_buffer,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t base_log,
    uint32_t level_count, int8_t *device_mem,
    int device_memory_size_per_block) {

  grid_group grid = this_grid();

  // We use shared memory for the polynomials that are used often during the
  // bootstrap, since shared memory is kept in L1 cache and accessing it is
  // much faster than global memory
  extern __shared__ int8_t sharedmem[];
  int8_t *selected_memory;
  int block_index =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  uint32_t glwe_dimension = gridDim.y - 1;

  if constexpr (SMD == FULLSM)
    selected_memory = sharedmem;
  else
    selected_memory = &device_mem[block_index * device_memory_size_per_block];

  // We always compute the pointer with most restrictive alignment to avoid
  // alignment issues
  double2 *accumulator_fft = (double2 *)selected_memory;
  Torus *accumulator =
      (Torus *)accumulator_fft +
      (ptrdiff_t)(sizeof(double2) * polynomial_size / 2 / sizeof(Torus));
  Torus *accumulator_rotated =
      (Torus *)accumulator + (ptrdiff_t)polynomial_size;

  if constexpr (SMD == PARTIALSM)
    accumulator_fft = (double2 *)sharedmem;

  // The third dimension of the block is used to determine on which ciphertext
  // this block is operating, in the case of batch bootstraps
  Torus *block_lwe_array_in = &lwe_array_in[blockIdx.z * (lwe_dimension + 1)];

  Torus *block_lut_vector = &lut_vector[lut_vector_indexes[blockIdx.z] *
                                        params::degree * (glwe_dimension + 1)];

  double2 *block_join_buffer =
      &join_buffer[blockIdx.z * level_count * (glwe_dimension + 1) *
                   params::degree / 2];
  // Since the space is L1 cache is small, we use the same memory location for
  // the rotated accumulator and the fft accumulator, since we know that the
  // rotated array is not in use anymore by the time we perform the fft

  // Put "b" in [0, 2N[
  Torus b_hat = 0;
  rescale_torus_element(block_lwe_array_in[lwe_dimension], b_hat,
                        2 * params::degree);

  divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                        params::degree / params::opt>(
      accumulator, &block_lut_vector[blockIdx.y * params::degree], b_hat,
      false);

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
    mul_ggsw_glwe<Torus, params>(
        accumulator, accumulator_fft, block_join_buffer, bootstrapping_key,
        polynomial_size, glwe_dimension, level_count, i, grid);

    synchronize_threads_in_block();
  }

  auto block_lwe_array_out =
      &lwe_array_out[blockIdx.z * (glwe_dimension * polynomial_size + 1) +
                     blockIdx.y * polynomial_size];

  if (blockIdx.x == 0 && blockIdx.y < glwe_dimension) {
    // Perform a sample extract. At this point, all blocks have the result, but
    // we do the computation at block 0 to avoid waiting for extra blocks, in
    // case they're not synchronized
    sample_extract_mask<Torus, params>(block_lwe_array_out, accumulator);
  } else if (blockIdx.x == 0 && blockIdx.y == glwe_dimension) {
    sample_extract_body<Torus, params>(block_lwe_array_out, accumulator, 0);
  }
}

template <typename Torus>
__host__ __device__ int
get_buffer_size_full_sm_bootstrap_low_latency(uint32_t polynomial_size) {
  return sizeof(Torus) * polynomial_size +      // accumulator_rotated
         sizeof(Torus) * polynomial_size +      // accumulator
         sizeof(double2) * polynomial_size / 2; // accumulator fft
}

template <typename Torus>
__host__ __device__ int
get_buffer_size_partial_sm_bootstrap_low_latency(uint32_t polynomial_size) {
  return sizeof(double2) * polynomial_size / 2; // accumulator fft mask & body
}

template <typename Torus>
__host__ __device__ int get_buffer_size_bootstrap_low_latency(
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory) {

  int full_sm =
      get_buffer_size_full_sm_bootstrap_low_latency<Torus>(polynomial_size);
  int partial_sm =
      get_buffer_size_partial_sm_bootstrap_low_latency<Torus>(polynomial_size);
  int partial_dm = full_sm - partial_sm;
  int full_dm = full_sm;
  int device_mem = 0;
  if (max_shared_memory < partial_sm) {
    device_mem = full_dm * input_lwe_ciphertext_count * level_count *
                 (glwe_dimension + 1);
  } else if (max_shared_memory < full_sm) {
    device_mem = partial_dm * input_lwe_ciphertext_count * level_count *
                 (glwe_dimension + 1);
  }
  return device_mem + (glwe_dimension + 1) * level_count *
                          input_lwe_ciphertext_count * polynomial_size / 2 *
                          sizeof(double2);
}

template <typename Torus, typename STorus, typename params>
__host__ void scratch_bootstrap_low_latency(
    void *v_stream, uint32_t gpu_index, int8_t **pbs_buffer,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {
  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  int full_sm =
      get_buffer_size_full_sm_bootstrap_low_latency<Torus>(polynomial_size);
  int partial_sm =
      get_buffer_size_partial_sm_bootstrap_low_latency<Torus>(polynomial_size);
  if (max_shared_memory >= partial_sm && max_shared_memory < full_sm) {
    check_cuda_error(cudaFuncSetAttribute(
        device_bootstrap_low_latency<Torus, params, PARTIALSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, partial_sm));
    cudaFuncSetCacheConfig(
        device_bootstrap_low_latency<Torus, params, PARTIALSM>,
        cudaFuncCachePreferShared);
    check_cuda_error(cudaGetLastError());
  } else if (max_shared_memory >= partial_sm) {
    check_cuda_error(cudaFuncSetAttribute(
        device_bootstrap_low_latency<Torus, params, FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, full_sm));
    cudaFuncSetCacheConfig(device_bootstrap_low_latency<Torus, params, FULLSM>,
                           cudaFuncCachePreferShared);
    check_cuda_error(cudaGetLastError());
  }
  if (allocate_gpu_memory) {
    int buffer_size = get_buffer_size_bootstrap_low_latency<Torus>(
        glwe_dimension, polynomial_size, level_count,
        input_lwe_ciphertext_count, max_shared_memory);
    *pbs_buffer = (int8_t *)cuda_malloc_async(buffer_size, stream, gpu_index);
    check_cuda_error(cudaGetLastError());
  }
}

/*
 * Host wrapper to the low latency version
 * of bootstrapping
 */
template <typename Torus, class params>
__host__ void host_bootstrap_low_latency(
    void *v_stream, uint32_t gpu_index, Torus *lwe_array_out, Torus *lut_vector,
    Torus *lut_vector_indexes, Torus *lwe_array_in, double2 *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t num_lut_vectors,
    uint32_t max_shared_memory) {

  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  // With SM each block corresponds to either the mask or body, no need to
  // duplicate data for each
  int full_sm =
      get_buffer_size_full_sm_bootstrap_low_latency<Torus>(polynomial_size);

  int partial_sm =
      get_buffer_size_partial_sm_bootstrap_low_latency<Torus>(polynomial_size);

  int full_dm = full_sm;

  int partial_dm = full_dm - partial_sm;

  int8_t *d_mem = pbs_buffer;
  double2 *buffer_fft =
      (double2 *)d_mem +
      (ptrdiff_t)(get_buffer_size_bootstrap_low_latency<Torus>(
                      glwe_dimension, polynomial_size, level_count,
                      input_lwe_ciphertext_count, max_shared_memory) /
                      sizeof(double2) -
                  (glwe_dimension + 1) * level_count *
                      input_lwe_ciphertext_count * polynomial_size / 2);

  int thds = polynomial_size / params::opt;
  dim3 grid(level_count, glwe_dimension + 1, input_lwe_ciphertext_count);

  void *kernel_args[12];
  kernel_args[0] = &lwe_array_out;
  kernel_args[1] = &lut_vector;
  kernel_args[2] = &lut_vector_indexes;
  kernel_args[3] = &lwe_array_in;
  kernel_args[4] = &bootstrapping_key;
  kernel_args[5] = &buffer_fft;
  kernel_args[6] = &lwe_dimension;
  kernel_args[7] = &polynomial_size;
  kernel_args[8] = &base_log;
  kernel_args[9] = &level_count;
  kernel_args[10] = &d_mem;

  if (max_shared_memory < partial_sm) {
    kernel_args[11] = &full_dm;
    check_cuda_error(cudaLaunchCooperativeKernel(
        (void *)device_bootstrap_low_latency<Torus, params, NOSM>, grid, thds,
        (void **)kernel_args, 0, *stream));
  } else if (max_shared_memory < full_sm) {
    kernel_args[11] = &partial_dm;
    check_cuda_error(cudaLaunchCooperativeKernel(
        (void *)device_bootstrap_low_latency<Torus, params, PARTIALSM>, grid,
        thds, (void **)kernel_args, partial_sm, *stream));

  } else {
    int no_dm = 0;
    kernel_args[11] = &no_dm;
    check_cuda_error(cudaLaunchCooperativeKernel(
        (void *)device_bootstrap_low_latency<Torus, params, FULLSM>, grid, thds,
        (void **)kernel_args, full_sm, *stream));
  }

  check_cuda_error(cudaGetLastError());
}

#endif // LOWLAT_PBS_H
