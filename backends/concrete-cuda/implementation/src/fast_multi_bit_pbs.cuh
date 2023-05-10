#ifndef FASTMULTIBIT_PBS_H
#define FASTMULTIBIT_PBS_H

#include "bootstrap.h"
#include "complex/operations.cuh"
#include "cooperative_groups.h"
#include "crypto/gadget.cuh"
#include "crypto/ggsw.cuh"
#include "crypto/torus.cuh"
#include "device.h"
#include "fft/bnsmfft.cuh"
#include "fft/twiddles.cuh"
#include "multi_bit_pbs.cuh"
#include "multi_bit_pbs.h"
#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial.cuh"
#include "polynomial/polynomial_math.cuh"
#include "utils/timer.cuh"
#include <vector>

template <typename Torus, class params>
__global__ void device_multi_bit_bootstrap_fast_accumulate(
    Torus *lwe_array_out, Torus *lut_vector, Torus *lut_vector_indexes,
    Torus *lwe_array_in, double2 *keybundle_array, double2 *join_buffer,
    Torus *global_accumulator, uint32_t lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t grouping_factor, uint32_t lwe_offset, uint32_t lwe_chunk_size,
    uint32_t keybundle_size_per_input) {

  grid_group grid = this_grid();

  // We use shared memory for the polynomials that are used often during the
  // bootstrap, since shared memory is kept in L1 cache and accessing it is
  // much faster than global memory
  extern __shared__ int8_t sharedmem[];
  int8_t *selected_memory;

  selected_memory = sharedmem;

  // We always compute the pointer with most restrictive alignment to avoid
  // alignment issues
  double2 *accumulator_fft = (double2 *)selected_memory;
  Torus *accumulator =
      (Torus *)accumulator_fft +
      (ptrdiff_t)(sizeof(double2) * polynomial_size / 2 / sizeof(Torus));

  // The third dimension of the block is used to determine on which ciphertext
  // this block is operating, in the case of batch bootstraps
  Torus *block_lwe_array_in = &lwe_array_in[blockIdx.z * (lwe_dimension + 1)];

  Torus *block_lut_vector = &lut_vector[lut_vector_indexes[blockIdx.z] *
                                        params::degree * (glwe_dimension + 1)];

  double2 *block_join_buffer =
      &join_buffer[blockIdx.z * level_count * (glwe_dimension + 1) *
                   params::degree / 2];

  Torus *global_slice =
      global_accumulator +
      (blockIdx.y + blockIdx.z * (glwe_dimension + 1)) * params::degree;

  double2 *keybundle = keybundle_array +
                       // select the input
                       blockIdx.z * keybundle_size_per_input;

  if (lwe_offset == 0) {
    // Put "b" in [0, 2N[
    Torus b_hat = 0;
    rescale_torus_element(block_lwe_array_in[lwe_dimension], b_hat,
                          2 * params::degree);

    divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                          params::degree / params::opt>(
        accumulator, &block_lut_vector[blockIdx.y * params::degree], b_hat,
        false);
  } else {
    // Load the accumulator calculated in previous iterations
    copy_polynomial<Torus, params::opt, params::degree / params::opt>(
        global_slice, accumulator);
  }

  for (int i = 0; (i + lwe_offset) < lwe_dimension && i < lwe_chunk_size; i++) {
    // Decompose the accumulator. Each block gets one level of the
    // decomposition, for the mask and the body (so block 0 will have the
    // accumulator decomposed at level 0, 1 at 1, etc.)
    GadgetMatrix<Torus, params> gadget_acc(base_log, level_count, accumulator);
    gadget_acc.decompose_and_compress_level(accumulator_fft, blockIdx.x);

    // We are using the same memory space for accumulator_fft and
    // accumulator_rotated, so we need to synchronize here to make sure they
    // don't modify the same memory space at the same time
    synchronize_threads_in_block();

    // Perform G^-1(ACC) * GGSW -> GLWE
    mul_ggsw_glwe<Torus, params>(accumulator, accumulator_fft,
                                 block_join_buffer, keybundle, polynomial_size,
                                 glwe_dimension, level_count, i, grid);

    synchronize_threads_in_block();
  }

  if (lwe_offset + lwe_chunk_size >= (lwe_dimension / grouping_factor)) {
    auto block_lwe_array_out =
        &lwe_array_out[blockIdx.z * (glwe_dimension * polynomial_size + 1) +
                       blockIdx.y * polynomial_size];

    if (blockIdx.x == 0 && blockIdx.y < glwe_dimension) {
      // Perform a sample extract. At this point, all blocks have the result,
      // but we do the computation at block 0 to avoid waiting for extra blocks,
      // in case they're not synchronized
      sample_extract_mask<Torus, params>(block_lwe_array_out, accumulator);
    } else if (blockIdx.x == 0 && blockIdx.y == glwe_dimension) {
      sample_extract_body<Torus, params>(block_lwe_array_out, accumulator, 0);
    }
  } else {
    // Load the accumulator calculated in previous iterations
    copy_polynomial<Torus, params::opt, params::degree / params::opt>(
        accumulator, global_slice);
  }
}

template <typename Torus>
__host__ __device__ uint64_t
get_buffer_size_full_sm_fast_multibit_bootstrap(uint32_t polynomial_size) {
  return sizeof(Torus) * polynomial_size * 2; // accumulator
}

template <typename Torus>
__host__ __device__ uint64_t get_buffer_size_fast_multibit_bootstrap(
    uint32_t lwe_dimension, uint32_t glwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t input_lwe_ciphertext_count,
    uint32_t grouping_factor, uint32_t lwe_chunk_size,
    uint32_t max_shared_memory) {

  uint64_t buffer_size = 0;
  buffer_size += 2 * input_lwe_ciphertext_count * lwe_chunk_size * level_count *
                 (glwe_dimension + 1) * (glwe_dimension + 1) *
                 (polynomial_size / 2) * sizeof(double2); // keybundle fft
  buffer_size += input_lwe_ciphertext_count * (glwe_dimension + 1) *
                 level_count * (polynomial_size / 2) *
                 sizeof(double2); // join buffer
  buffer_size += input_lwe_ciphertext_count * (glwe_dimension + 1) *
                 polynomial_size * sizeof(Torus); // global_accumulator

  return buffer_size + buffer_size % sizeof(double2);
}

template <typename Torus, typename STorus, typename params>
__host__ void scratch_fast_multi_bit_pbs(
    void *v_stream, uint32_t gpu_index, int8_t **pbs_buffer,
    uint32_t lwe_dimension, uint32_t glwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t input_lwe_ciphertext_count,
    uint32_t grouping_factor, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {
  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  uint64_t full_sm_keybundle =
      get_buffer_size_full_sm_multibit_bootstrap_keybundle<Torus>(
          polynomial_size);
  uint64_t full_sm_accumulate =
      get_buffer_size_full_sm_fast_multibit_bootstrap<Torus>(polynomial_size);

  check_cuda_error(cudaFuncSetAttribute(
      device_multi_bit_bootstrap_keybundle<Torus, params>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, full_sm_keybundle));
  cudaFuncSetCacheConfig(device_multi_bit_bootstrap_keybundle<Torus, params>,
                         cudaFuncCachePreferShared);
  check_cuda_error(cudaGetLastError());

  check_cuda_error(cudaFuncSetAttribute(
      device_multi_bit_bootstrap_fast_accumulate<Torus, params>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, full_sm_accumulate));
  cudaFuncSetCacheConfig(
      device_multi_bit_bootstrap_fast_accumulate<Torus, params>,
      cudaFuncCachePreferShared);
  check_cuda_error(cudaGetLastError());

  if (allocate_gpu_memory) {
    uint32_t lwe_chunk_size = get_lwe_chunk_size(input_lwe_ciphertext_count);

    uint64_t buffer_size = get_buffer_size_fast_multibit_bootstrap<Torus>(
        lwe_dimension, glwe_dimension, polynomial_size, level_count,
        input_lwe_ciphertext_count, grouping_factor, lwe_chunk_size,
        max_shared_memory);
    *pbs_buffer = (int8_t *)cuda_malloc_async(buffer_size, stream, gpu_index);
    check_cuda_error(cudaGetLastError());
  }
}

template <typename Torus, typename STorus, class params>
__host__ void host_fast_multi_bit_pbs(
    void *v_stream, uint32_t gpu_index, Torus *lwe_array_out, Torus *lut_vector,
    Torus *lut_vector_indexes, Torus *lwe_array_in, uint64_t *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t grouping_factor, uint32_t base_log,
    uint32_t level_count, uint32_t num_samples, uint32_t num_lut_vectors,
    uint32_t lwe_idx, uint32_t max_shared_memory) {
  uint32_t lwe_chunk_size = get_lwe_chunk_size(num_samples);

  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);
  uint32_t number_of_chunks =
      (lwe_dimension / grouping_factor) / lwe_chunk_size;
  number_of_chunks += ((lwe_dimension / grouping_factor) % lwe_chunk_size) != 0;

  assert(lwe_chunk_size * number_of_chunks >=
         (lwe_dimension / grouping_factor));
  double2 *keybundle_fft = (double2 *)pbs_buffer;
  double2 *buffer_fft = (double2 *)keybundle_fft +
                        2 * num_samples * lwe_chunk_size * level_count *
                            (glwe_dimension + 1) * (glwe_dimension + 1) *
                            (polynomial_size / 2);
  Torus *global_accumulator =
      (Torus *)buffer_fft +
      (ptrdiff_t)(sizeof(double2) * num_samples * (glwe_dimension + 1) *
                  level_count * (polynomial_size / 2) / sizeof(Torus));

  dim3 grid_accumulate(level_count, glwe_dimension + 1, num_samples);
  dim3 thds(polynomial_size / params::opt, 1, 1);

  uint64_t full_sm_keybundle =
      get_buffer_size_full_sm_multibit_bootstrap_keybundle<Torus>(
          polynomial_size);
  uint64_t full_sm_accumulate =
      get_buffer_size_full_sm_fast_multibit_bootstrap<Torus>(polynomial_size);

  uint32_t keybundle_size_per_input =
      lwe_chunk_size * level_count * (glwe_dimension + 1) *
      (glwe_dimension + 1) * (polynomial_size / 2);

  // Creates all streams
  cudaStream_t *alternative_stream = cuda_create_stream(gpu_index);

  cudaStream_t *current_stream = stream;
  cudaStream_t *next_stream = alternative_stream;

  dim3 grid_keybundle(num_samples * lwe_chunk_size,
                      (glwe_dimension + 1) * (glwe_dimension + 1), level_count);
  device_multi_bit_bootstrap_keybundle<Torus, params>
      <<<grid_keybundle, thds, full_sm_keybundle, *current_stream>>>(
          lwe_array_in, keybundle_fft, bootstrapping_key, lwe_dimension,
          glwe_dimension, polynomial_size, grouping_factor, base_log,
          level_count, 0, lwe_chunk_size, keybundle_size_per_input);
  check_cuda_error(cudaGetLastError());

  void *kernel_args[16];
  kernel_args[0] = &lwe_array_out;
  kernel_args[1] = &lut_vector;
  kernel_args[2] = &lut_vector_indexes;
  kernel_args[3] = &lwe_array_in;
  kernel_args[5] = &buffer_fft;
  kernel_args[6] = &global_accumulator;
  kernel_args[7] = &lwe_dimension;
  kernel_args[8] = &glwe_dimension;
  kernel_args[9] = &polynomial_size;
  kernel_args[10] = &base_log;
  kernel_args[11] = &level_count;
  kernel_args[12] = &grouping_factor;
  kernel_args[15] = &keybundle_size_per_input;

  uint32_t lwe_offset = 0;
  for (int i = 0; i < number_of_chunks; i++) {
    uint32_t chunk_size = std::min(
        lwe_chunk_size, (lwe_dimension / grouping_factor) - lwe_offset);

    dim3 grid_keybundle(num_samples * chunk_size,
                        (glwe_dimension + 1) * (glwe_dimension + 1),
                        level_count);
    auto keybundle_next =
        keybundle_fft + ((i + 1) & 1) * num_samples * keybundle_size_per_input;
    device_multi_bit_bootstrap_keybundle<Torus, params>
        <<<grid_keybundle, thds, full_sm_keybundle, *next_stream>>>(
            lwe_array_in, keybundle_next, bootstrapping_key, lwe_dimension,
            glwe_dimension, polynomial_size, grouping_factor, base_log,
            level_count, lwe_offset + lwe_chunk_size, chunk_size,
            keybundle_size_per_input);
    check_cuda_error(cudaGetLastError());

    auto keybundle_current =
        keybundle_fft + (i & 1) * num_samples * keybundle_size_per_input;

    kernel_args[4] = &keybundle_current;
    kernel_args[13] = &lwe_offset;
    kernel_args[14] = &chunk_size;

    check_cuda_error(cudaLaunchCooperativeKernel(
        (void *)device_multi_bit_bootstrap_fast_accumulate<Torus, params>,
        grid_accumulate, thds, (void **)kernel_args, full_sm_accumulate,
        *current_stream));
    cuda_synchronize_stream(current_stream);
    std::swap(next_stream, current_stream);

    lwe_offset += chunk_size;
  }

  cuda_synchronize_stream(stream);
  cuda_synchronize_stream(alternative_stream);
  cuda_destroy_stream(alternative_stream, gpu_index);
}

// Verify if the grid size for the low latency kernel satisfies the cooperative
// group constraints
template <typename Torus, class params>
__host__ bool
verify_cuda_bootstrap_fast_multi_bit_grid_size(int glwe_dimension,
                                               int level_count, int num_samples,
                                               uint32_t max_shared_memory) {

  // Calculate the dimension of the kernel
  uint64_t full_sm =
      get_buffer_size_full_sm_fast_multibit_bootstrap<Torus>(params::degree);

  int thds = params::degree / params::opt;

  // Get the maximum number of active blocks per streaming multiprocessors
  int number_of_blocks = level_count * (glwe_dimension + 1) * num_samples;
  int max_active_blocks_per_sm;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_per_sm,
      (void *)device_multi_bit_bootstrap_fast_accumulate<Torus, params>, thds,
      full_sm);

  // Get the number of streaming multiprocessors
  int number_of_sm = 0;
  cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, 0);
  return number_of_blocks <= max_active_blocks_per_sm * number_of_sm;
}
#endif // FASTMULTIBIT_PBS_H
