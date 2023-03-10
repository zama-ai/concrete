#ifndef VERTICAL_PACKING_CUH
#define VERTICAL_PACKING_CUH

#include "bootstrap.h"
#include "complex/operations.cuh"
#include "crypto/gadget.cuh"
#include "crypto/ggsw.cuh"
#include "crypto/torus.cuh"
#include "device.h"
#include "fft/bnsmfft.cuh"
#include "fft/twiddles.cuh"
#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial.cuh"
#include "polynomial/polynomial_math.cuh"
#include "utils/timer.cuh"

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
     int8_t *selected_memory, uint32_t output_idx, uint32_t input_idx1,
     uint32_t input_idx2, uint32_t glwe_dim, uint32_t polynomial_size,
     uint32_t base_log, uint32_t level_count, uint32_t ggsw_idx) {

  // Define glwe_sub
  Torus *glwe_sub = (Torus *)selected_memory;

  double2 *res_fft =
      (double2 *)glwe_sub +
      (glwe_dim + 1) * polynomial_size / (sizeof(double2) / sizeof(Torus));

  double2 *glwe_fft =
      (double2 *)res_fft + (ptrdiff_t)((glwe_dim + 1) * polynomial_size / 2);

  /////////////////////////////////////

  // glwe2-glwe1

  // Gets the pointers for the global memory
  auto m0 = &glwe_array_in[input_idx1 * (glwe_dim + 1) * polynomial_size];
  auto m1 = &glwe_array_in[input_idx2 * (glwe_dim + 1) * polynomial_size];

  // Subtraction: m1-m0
  for (int i = 0; i < (glwe_dim + 1); i++) {
    auto glwe_sub_slice = glwe_sub + i * params::degree;
    auto m0_slice = m0 + i * params::degree;
    auto m1_slice = m1 + i * params::degree;
    sub_polynomial<Torus, params>(glwe_sub_slice, m1_slice, m0_slice);
  }

  // Initialize the polynomial multiplication via FFT arrays
  // The polynomial multiplications happens at the block level
  // and each thread handles two or more coefficients
  int pos = threadIdx.x;
  for (int i = 0; i < (glwe_dim + 1); i++)
    for (int j = 0; j < params::opt / 2; j++) {
      res_fft[pos].x = 0;
      res_fft[pos].y = 0;
      pos += params::degree / params::opt;
    }

  synchronize_threads_in_block();
  GadgetMatrix<Torus, params> gadget(base_log, level_count, glwe_sub,
                                     glwe_dim + 1);

  // Subtract each glwe operand, decompose the resulting
  // polynomial coefficients to multiply each decomposed level
  // with the corresponding part of the LUT
  for (int level = level_count - 1; level >= 0; level--) {
    // Decomposition
    for (int i = 0; i < (glwe_dim + 1); i++) {
      gadget.decompose_and_compress_next_polynomial(glwe_fft, i);

      // First, perform the polynomial multiplication
      NSMFFT_direct<HalfDegree<params>>(glwe_fft);

      // External product and accumulate
      // Get the piece necessary for the multiplication
      auto bsk_slice = get_ith_mask_kth_block(
          ggsw_in, ggsw_idx, i, level, polynomial_size, glwe_dim, level_count);

      // Perform the coefficient-wise product
      for (int j = 0; j < (glwe_dim + 1); j++) {
        auto bsk_poly = bsk_slice + j * params::degree / 2;
        auto res_fft_poly = res_fft + j * params::degree / 2;
        polynomial_product_accumulate_in_fourier_domain<params, double2>(
            res_fft_poly, glwe_fft, bsk_poly);
      }
    }
    synchronize_threads_in_block();
  }

  // IFFT
  synchronize_threads_in_block();
  for (int i = 0; i < (glwe_dim + 1); i++) {
    auto res_fft_slice = res_fft + i * params::degree / 2;
    NSMFFT_inverse<HalfDegree<params>>(res_fft_slice);
  }
  synchronize_threads_in_block();

  // Write the output
  Torus *mb = &glwe_array_out[output_idx * (glwe_dim + 1) * polynomial_size];

  int tid = threadIdx.x;
  for (int i = 0; i < (glwe_dim + 1); i++)
    for (int j = 0; j < params::opt; j++) {
      mb[tid] = m0[tid];
      tid += params::degree / params::opt;
    }

  for (int i = 0; i < (glwe_dim + 1); i++) {
    auto res_fft_slice = res_fft + i * params::degree / 2;
    auto mb_slice = mb + i * params::degree;
    add_to_torus<Torus, params>(res_fft_slice, mb_slice);
  }
}

// Appends zeroed paddings between each LUT
template <typename Torus, class params>
__host__ void add_padding_to_lut_async(Torus *lut_out, Torus *lut_in,
                                       uint32_t glwe_dimension,
                                       uint32_t num_lut, cudaStream_t *stream) {
  check_cuda_error(cudaMemsetAsync(lut_out, 0,
                                   num_lut * (glwe_dimension + 1) *
                                       params::degree * sizeof(Torus),
                                   *stream));
  for (int i = 0; i < num_lut; i++)
    check_cuda_error(cudaMemcpyAsync(
        lut_out + ((glwe_dimension + 1) * i + glwe_dimension) * params::degree,
        lut_in + i * params::degree, params::degree * sizeof(Torus),
        cudaMemcpyDeviceToDevice, *stream));
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
__global__ void device_batch_cmux(Torus *glwe_array_out, Torus *glwe_array_in,
                                  double2 *ggsw_in, int8_t *device_mem,
                                  size_t device_memory_size_per_block,
                                  uint32_t glwe_dim, uint32_t polynomial_size,
                                  uint32_t base_log, uint32_t level_count,
                                  uint32_t ggsw_idx, uint32_t num_lut) {

  // We are running gridDim.y cmux trees in parallel
  int tree_idx = blockIdx.y;
  int tree_offset = tree_idx * num_lut * (glwe_dim + 1) * polynomial_size;

  auto block_glwe_array_out = glwe_array_out + tree_offset;
  auto block_glwe_array_in = glwe_array_in + tree_offset;

  // The x-axis handles a single cmux tree. Each block computes one cmux.
  int cmux_idx = blockIdx.x;
  int output_idx = cmux_idx;
  int input_idx1 = (cmux_idx << 1);
  int input_idx2 = (cmux_idx << 1) + 1;

  // We use shared memory for intermediate result
  extern __shared__ int8_t sharedmem[];
  int8_t *selected_memory;

  if constexpr (SMD == FULLSM)
    selected_memory = sharedmem;
  else
    selected_memory = &device_mem[(blockIdx.x + blockIdx.y * gridDim.x) *
                                  device_memory_size_per_block];

  cmux<Torus, STorus, params>(block_glwe_array_out, block_glwe_array_in,
                              ggsw_in, selected_memory, output_idx, input_idx1,
                              input_idx2, glwe_dim, polynomial_size, base_log,
                              level_count, ggsw_idx);
}

template <typename Torus>
__host__ __device__ int
get_memory_needed_per_block_cmux_tree(uint32_t glwe_dimension,
                                      uint32_t polynomial_size) {
  return sizeof(Torus) * polynomial_size * (glwe_dimension + 1) + // glwe_sub
         sizeof(double2) * polynomial_size / 2 *
             (glwe_dimension + 1) +             // res_fft
         sizeof(double2) * polynomial_size / 2; // glwe_fft
}

template <typename Torus>
__host__ __device__ int
get_buffer_size_cmux_tree(uint32_t glwe_dimension, uint32_t polynomial_size,
                          uint32_t level_count, uint32_t r, uint32_t tau,
                          uint32_t max_shared_memory) {

  int memory_needed_per_block = get_memory_needed_per_block_cmux_tree<Torus>(
      glwe_dimension, polynomial_size);
  int num_lut = (1 << r);
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;
  int glwe_size = (glwe_dimension + 1) * polynomial_size;
  int device_mem = 0;
  if (max_shared_memory < memory_needed_per_block) {
    device_mem = memory_needed_per_block * (1 << (r - 1)) * tau;
  }
  if (max_shared_memory < polynomial_size * sizeof(double)) {
    device_mem += polynomial_size * sizeof(double);
  }
  int buffer_size = r * ggsw_size * sizeof(double) +
                    num_lut * tau * glwe_size * sizeof(Torus) +
                    num_lut * tau * glwe_size * sizeof(Torus) + device_mem;
  return buffer_size + buffer_size % sizeof(double2);
}

template <typename Torus, typename STorus, typename params>
__host__ void
scratch_cmux_tree(void *v_stream, uint32_t gpu_index, int8_t **cmux_tree_buffer,
                  uint32_t glwe_dimension, uint32_t polynomial_size,
                  uint32_t level_count, uint32_t r, uint32_t tau,
                  uint32_t max_shared_memory, bool allocate_gpu_memory) {
  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  int memory_needed_per_block = get_memory_needed_per_block_cmux_tree<Torus>(
      glwe_dimension, polynomial_size);
  if (max_shared_memory >= memory_needed_per_block) {
    check_cuda_error(cudaFuncSetAttribute(
        device_batch_cmux<Torus, STorus, params, FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, memory_needed_per_block));
    check_cuda_error(
        cudaFuncSetCacheConfig(device_batch_cmux<Torus, STorus, params, FULLSM>,
                               cudaFuncCachePreferShared));
  }

  if (allocate_gpu_memory) {
    int buffer_size = get_buffer_size_cmux_tree<Torus>(
        glwe_dimension, polynomial_size, level_count, r, tau,
        max_shared_memory);
    *cmux_tree_buffer =
        (int8_t *)cuda_malloc_async(buffer_size, stream, gpu_index);
    check_cuda_error(cudaGetLastError());
  }
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
 *  - tau: The quantity of CMUX trees that should be executed
 */
template <typename Torus, typename STorus, class params>
__host__ void
host_cmux_tree(void *v_stream, uint32_t gpu_index, Torus *glwe_array_out,
               Torus *ggsw_in, Torus *lut_vector, int8_t *cmux_tree_buffer,
               uint32_t glwe_dimension, uint32_t polynomial_size,
               uint32_t base_log, uint32_t level_count, uint32_t r,
               uint32_t tau, uint32_t max_shared_memory) {
  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  int num_lut = (1 << r);
  if (r == 0) {
    // Simply copy the LUTs
    add_padding_to_lut_async<Torus, params>(glwe_array_out, lut_vector,
                                            glwe_dimension, tau, stream);
    return;
  }

  int memory_needed_per_block = get_memory_needed_per_block_cmux_tree<Torus>(
      glwe_dimension, polynomial_size);

  dim3 thds(polynomial_size / params::opt, 1, 1);

  //////////////////////
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;
  int glwe_size = (glwe_dimension + 1) * polynomial_size;

  // Define the buffers
  // Always define the buffers with strongest memory alignment constraints first
  // d_buffer1 and d_buffer2 are aligned with Torus, so they're defined last
  double2 *d_ggsw_fft_in = (double2 *)cmux_tree_buffer;
  int8_t *d_mem =
      (int8_t *)d_ggsw_fft_in + (ptrdiff_t)(r * ggsw_size * sizeof(double));
  int8_t *d_mem_fft = d_mem;
  if (max_shared_memory < memory_needed_per_block) {
    d_mem_fft =
        d_mem + (ptrdiff_t)(memory_needed_per_block * (1 << (r - 1)) * tau);
  }
  int8_t *d_buffer1 = d_mem_fft;
  if (max_shared_memory < polynomial_size * sizeof(double)) {
    d_buffer1 = d_mem_fft + (ptrdiff_t)(polynomial_size * sizeof(double));
  }
  int8_t *d_buffer2 =
      d_buffer1 + (ptrdiff_t)(num_lut * tau * glwe_size * sizeof(Torus));

  //////////////////////

  batch_fft_ggsw_vector<Torus, STorus, params>(
      stream, d_ggsw_fft_in, ggsw_in, d_mem_fft, r, glwe_dimension,
      polynomial_size, level_count, gpu_index, max_shared_memory);

  add_padding_to_lut_async<Torus, params>(
      (Torus *)d_buffer1, lut_vector, glwe_dimension, num_lut * tau, stream);

  Torus *output;
  // Run the cmux tree
  for (int layer_idx = 0; layer_idx < r; layer_idx++) {
    output = (layer_idx % 2 ? (Torus *)d_buffer1 : (Torus *)d_buffer2);
    Torus *input = (layer_idx % 2 ? (Torus *)d_buffer2 : (Torus *)d_buffer1);

    int num_cmuxes = (1 << (r - 1 - layer_idx));
    dim3 grid(num_cmuxes, tau, 1);

    // walks horizontally through the leaves
    if (max_shared_memory < memory_needed_per_block) {
      device_batch_cmux<Torus, STorus, params, NOSM>
          <<<grid, thds, 0, *stream>>>(output, input, d_ggsw_fft_in, d_mem,
                                       memory_needed_per_block,
                                       glwe_dimension, // k
                                       polynomial_size, base_log, level_count,
                                       layer_idx, // r
                                       num_lut);
    } else {
      device_batch_cmux<Torus, STorus, params, FULLSM>
          <<<grid, thds, memory_needed_per_block, *stream>>>(
              output, input, d_ggsw_fft_in, d_mem, memory_needed_per_block,
              glwe_dimension, // k
              polynomial_size, base_log, level_count,
              layer_idx, // r
              num_lut);
    }
    check_cuda_error(cudaGetLastError());
  }

  for (int i = 0; i < tau; i++) {
    check_cuda_error(cudaMemcpyAsync(
        glwe_array_out + i * glwe_size, output + i * num_lut * glwe_size,
        glwe_size * sizeof(Torus), cudaMemcpyDeviceToDevice, *stream));
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
 * - level_count: number of decomposition levels in the gadget matrix (~4)
 * - device_memory_size_per_sample: Amount of (shared/global) memory used for
 * the accumulators.
 * - device_mem: An array to be used for the accumulators. Can be in the shared
 * memory or global memory.
 */
template <typename Torus, typename STorus, class params, sharedMemDegree SMD>
__global__ void device_blind_rotation_and_sample_extraction(
    Torus *lwe_out, Torus *glwe_in, double2 *ggsw_in, // m^BR
    uint32_t mbr_size, uint32_t glwe_dim, uint32_t polynomial_size,
    uint32_t base_log, uint32_t level_count,
    size_t device_memory_size_per_sample, int8_t *device_mem) {

  // We use shared memory for intermediate result
  extern __shared__ int8_t sharedmem[];
  int8_t *selected_memory;

  if constexpr (SMD == FULLSM)
    selected_memory = sharedmem;
  else
    selected_memory = &device_mem[blockIdx.x * device_memory_size_per_sample];

  Torus *accumulator_c0 = (Torus *)selected_memory;
  Torus *accumulator_c1 =
      (Torus *)accumulator_c0 + (glwe_dim + 1) * polynomial_size;
  int8_t *cmux_memory =
      (int8_t *)(accumulator_c1 + (glwe_dim + 1) * polynomial_size);

  // Input LUT
  auto mi = &glwe_in[blockIdx.x * (glwe_dim + 1) * polynomial_size];
  int tid = threadIdx.x;
  for (int i = 0; i < (glwe_dim + 1); i++)
    for (int j = 0; j < params::opt; j++) {
      accumulator_c0[tid] = mi[tid];
      tid += params::degree / params::opt;
    }

  int monomial_degree = 0;
  for (int i = mbr_size - 1; i >= 0; i--) {
    synchronize_threads_in_block();

    // Compute x^ai * ACC
    // Mask and Body
    divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                          params::degree / params::opt>(
        accumulator_c1, accumulator_c0, (1 << monomial_degree), false,
        (glwe_dim + 1));

    monomial_degree += 1;

    // ACC = CMUX ( Ci, x^ai * ACC, ACC )
    synchronize_threads_in_block();
    cmux<Torus, STorus, params>(accumulator_c0, accumulator_c0, ggsw_in,
                                cmux_memory, 0, 0, 1, glwe_dim, polynomial_size,
                                base_log, level_count, i);
  }
  synchronize_threads_in_block();

  // Write the output
  auto block_lwe_out = &lwe_out[blockIdx.x * (glwe_dim * polynomial_size + 1)];

  // The blind rotation for this block is over
  // Now we can perform the sample extraction: for the body it's just
  // the resulting constant coefficient of the accumulator
  // For the mask it's more complicated
  sample_extract_mask<Torus, params>(block_lwe_out, accumulator_c0, glwe_dim);
  sample_extract_body<Torus, params>(block_lwe_out, accumulator_c0, glwe_dim);
}

template <typename Torus>
__host__ __device__ int
get_memory_needed_per_block_blind_rotation_sample_extraction(
    uint32_t glwe_dimension, uint32_t polynomial_size) {
  return sizeof(Torus) * polynomial_size *
             (glwe_dimension + 1) + // accumulator_c0
         sizeof(Torus) * polynomial_size *
             (glwe_dimension + 1) + // accumulator_c1
         +get_memory_needed_per_block_cmux_tree<Torus>(glwe_dimension,
                                                       polynomial_size);
}

template <typename Torus>
__host__ __device__ int get_buffer_size_blind_rotation_sample_extraction(
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t mbr_size, uint32_t tau, uint32_t max_shared_memory) {

  int memory_needed_per_block =
      get_memory_needed_per_block_blind_rotation_sample_extraction<Torus>(
          glwe_dimension, polynomial_size);
  int device_mem = 0;
  if (max_shared_memory < memory_needed_per_block) {
    device_mem = memory_needed_per_block * tau;
  }
  if (max_shared_memory < polynomial_size * sizeof(double)) {
    device_mem += polynomial_size * sizeof(double);
  }
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;
  int buffer_size = mbr_size * ggsw_size * sizeof(double) // d_ggsw_fft_in
                    + device_mem;
  return buffer_size + buffer_size % sizeof(double2);
}

template <typename Torus, typename STorus, typename params>
__host__ void scratch_blind_rotation_sample_extraction(
    void *v_stream, uint32_t gpu_index, int8_t **br_se_buffer,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t mbr_size, uint32_t tau, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {
  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  int memory_needed_per_block =
      get_memory_needed_per_block_blind_rotation_sample_extraction<Torus>(
          glwe_dimension, polynomial_size);
  if (max_shared_memory >= memory_needed_per_block) {
    check_cuda_error(cudaFuncSetAttribute(
        device_blind_rotation_and_sample_extraction<Torus, STorus, params,
                                                    FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, memory_needed_per_block));
    check_cuda_error(cudaFuncSetCacheConfig(
        device_blind_rotation_and_sample_extraction<Torus, STorus, params,
                                                    FULLSM>,
        cudaFuncCachePreferShared));
  }

  if (allocate_gpu_memory) {
    int buffer_size = get_buffer_size_blind_rotation_sample_extraction<Torus>(
        glwe_dimension, polynomial_size, level_count, mbr_size, tau,
        max_shared_memory);
    *br_se_buffer = (int8_t *)cuda_malloc_async(buffer_size, stream, gpu_index);
    check_cuda_error(cudaGetLastError());
  }
}

template <typename Torus, typename STorus, class params>
__host__ void host_blind_rotate_and_sample_extraction(
    void *v_stream, uint32_t gpu_index, Torus *lwe_out, Torus *ggsw_in,
    Torus *lut_vector, int8_t *br_se_buffer, uint32_t mbr_size, uint32_t tau,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t base_log,
    uint32_t level_count, uint32_t max_shared_memory) {

  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  int memory_needed_per_block =
      get_memory_needed_per_block_blind_rotation_sample_extraction<Torus>(
          glwe_dimension, polynomial_size);

  // Prepare the buffers
  // Here all the buffers have double2 alignment
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;
  double2 *d_ggsw_fft_in = (double2 *)br_se_buffer;
  int8_t *d_mem_fft = (int8_t *)d_ggsw_fft_in +
                      (ptrdiff_t)(mbr_size * ggsw_size * sizeof(double));
  int8_t *d_mem = d_mem_fft;
  if (max_shared_memory < polynomial_size * sizeof(double)) {
    d_mem = d_mem_fft + (ptrdiff_t)(polynomial_size * sizeof(double));
  }
  // Apply the FFT on m^br
  batch_fft_ggsw_vector<Torus, STorus, params>(
      stream, d_ggsw_fft_in, ggsw_in, d_mem_fft, mbr_size, glwe_dimension,
      polynomial_size, level_count, gpu_index, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  dim3 thds(polynomial_size / params::opt, 1, 1);
  dim3 grid(tau, 1, 1);

  if (max_shared_memory < memory_needed_per_block)
    device_blind_rotation_and_sample_extraction<Torus, STorus, params, NOSM>
        <<<grid, thds, 0, *stream>>>(lwe_out, lut_vector, d_ggsw_fft_in,
                                     mbr_size,
                                     glwe_dimension, // k
                                     polynomial_size, base_log, level_count,
                                     memory_needed_per_block, d_mem);
  else
    device_blind_rotation_and_sample_extraction<Torus, STorus, params, FULLSM>
        <<<grid, thds, memory_needed_per_block, *stream>>>(
            lwe_out, lut_vector, d_ggsw_fft_in, mbr_size,
            glwe_dimension, // k
            polynomial_size, base_log, level_count, memory_needed_per_block,
            d_mem);
  check_cuda_error(cudaGetLastError());
}
#endif // VERTICAL_PACKING_CUH
