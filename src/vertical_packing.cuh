#ifndef VERTICAL_PACKING_H
#define VERTICAL_PACKING_H

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

template <class params> __device__ void fft(double2 *output) {
  // Switch to the FFT space
  NSMFFT_direct<HalfDegree<params>>(output);
  synchronize_threads_in_block();
}

template <class params> __device__ void ifft_inplace(double2 *data) {
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
     int8_t *selected_memory, uint32_t output_idx, uint32_t input_idx1,
     uint32_t input_idx2, uint32_t glwe_dim, uint32_t polynomial_size,
     uint32_t base_log, uint32_t level_count, uint32_t ggsw_idx) {

  // Define glwe_sub
  Torus *glwe_sub_mask = (Torus *)selected_memory;
  Torus *glwe_sub_body = (Torus *)glwe_sub_mask + (ptrdiff_t)polynomial_size;

  double2 *mask_res_fft = (double2 *)glwe_sub_body +
                          polynomial_size / (sizeof(double2) / sizeof(Torus));
  double2 *body_res_fft =
      (double2 *)mask_res_fft + (ptrdiff_t)polynomial_size / 2;

  double2 *glwe_fft =
      (double2 *)body_res_fft + (ptrdiff_t)(polynomial_size / 2);

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

  GadgetMatrix<Torus, params> gadget_mask(base_log, level_count, glwe_sub_mask,
                                          1);
  GadgetMatrix<Torus, params> gadget_body(base_log, level_count, glwe_sub_body,
                                          1);
  // Subtract each glwe operand, decompose the resulting
  // polynomial coefficients to multiply each decomposed level
  // with the corresponding part of the LUT
  for (int level = level_count - 1; level >= 0; level--) {

    // Decomposition
    gadget_mask.decompose_and_compress_next(glwe_fft);

    // First, perform the polynomial multiplication for the mask
    fft<params>(glwe_fft);

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

    gadget_body.decompose_and_compress_next(glwe_fft);
    fft<params>(glwe_fft);

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
        lut_out + (2 * i + 1) * params::degree, lut_in + i * params::degree,
        params::degree * sizeof(Torus), cudaMemcpyDeviceToDevice, *stream));
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

  cmux<Torus, STorus, params>(
      glwe_array_out + tree_offset, glwe_array_in + tree_offset, ggsw_in,
      selected_memory, output_idx, input_idx1, input_idx2, glwe_dim,
      polynomial_size, base_log, level_count, ggsw_idx);
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
__host__ void host_cmux_tree(void *v_stream, uint32_t gpu_index,
                             Torus *glwe_array_out, Torus *ggsw_in,
                             Torus *lut_vector, uint32_t glwe_dimension,
                             uint32_t polynomial_size, uint32_t base_log,
                             uint32_t level_count, uint32_t r, uint32_t tau,
                             uint32_t max_shared_memory) {
  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  int num_lut = (1 << r);
  if (r == 0) {
    // Simply copy the LUTs
    add_padding_to_lut_async<Torus, params>(glwe_array_out, lut_vector,
                                            glwe_dimension, tau, stream);
    return;
  }

  int memory_needed_per_block =
      sizeof(Torus) * polynomial_size +       // glwe_sub_mask
      sizeof(Torus) * polynomial_size +       // glwe_sub_body
      sizeof(double2) * polynomial_size / 2 + // mask_res_fft
      sizeof(double2) * polynomial_size / 2 + // body_res_fft
      sizeof(double2) * polynomial_size / 2;  // glwe_fft

  dim3 thds(polynomial_size / params::opt, 1, 1);

  //////////////////////
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;

  double2 *d_ggsw_fft_in = (double2 *)cuda_malloc_async(
      r * ggsw_size * sizeof(double), stream, gpu_index);

  batch_fft_ggsw_vector<Torus, STorus, params>(
      stream, d_ggsw_fft_in, ggsw_in, r, glwe_dimension, polynomial_size,
      level_count, gpu_index, max_shared_memory);

  //////////////////////

  // Allocate global memory in case parameters are too large
  int8_t *d_mem;
  if (max_shared_memory < memory_needed_per_block) {
    d_mem = (int8_t *)cuda_malloc_async(
        memory_needed_per_block * (1 << (r - 1)) * tau, stream, gpu_index);
  } else {
    check_cuda_error(cudaFuncSetAttribute(
        device_batch_cmux<Torus, STorus, params, FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, memory_needed_per_block));
    check_cuda_error(
        cudaFuncSetCacheConfig(device_batch_cmux<Torus, STorus, params, FULLSM>,
                               cudaFuncCachePreferShared));
  }

  // Allocate buffers
  int glwe_size = (glwe_dimension + 1) * polynomial_size;

  Torus *d_buffer1 = (Torus *)cuda_malloc_async(
      num_lut * tau * glwe_size * sizeof(Torus), stream, gpu_index);
  Torus *d_buffer2 = (Torus *)cuda_malloc_async(
      num_lut * tau * glwe_size * sizeof(Torus), stream, gpu_index);

  add_padding_to_lut_async<Torus, params>(d_buffer1, lut_vector, glwe_dimension,
                                          num_lut * tau, stream);

  Torus *output;
  // Run the cmux tree
  for (int layer_idx = 0; layer_idx < r; layer_idx++) {
    output = (layer_idx % 2 ? d_buffer1 : d_buffer2);
    Torus *input = (layer_idx % 2 ? d_buffer2 : d_buffer1);

    int num_cmuxes = (1 << (r - 1 - layer_idx));
    dim3 grid(num_cmuxes, tau, 1);

    // walks horizontally through the leaves
    if (max_shared_memory < memory_needed_per_block)
      device_batch_cmux<Torus, STorus, params, NOSM>
          <<<grid, thds, 0, *stream>>>(output, input, d_ggsw_fft_in, d_mem,
                                       memory_needed_per_block,
                                       glwe_dimension, // k
                                       polynomial_size, base_log, level_count,
                                       layer_idx, // r
                                       num_lut);
    else
      device_batch_cmux<Torus, STorus, params, FULLSM>
          <<<grid, thds, memory_needed_per_block, *stream>>>(
              output, input, d_ggsw_fft_in, d_mem, memory_needed_per_block,
              glwe_dimension, // k
              polynomial_size, base_log, level_count,
              layer_idx, // r
              num_lut);
    check_cuda_error(cudaGetLastError());
  }

  for (int i = 0; i < tau; i++)
    check_cuda_error(cudaMemcpyAsync(
        glwe_array_out + i * glwe_size, output + i * num_lut * glwe_size,
        glwe_size * sizeof(Torus), cudaMemcpyDeviceToDevice, *stream));

  // Free memory
  cuda_drop_async(d_ggsw_fft_in, stream, gpu_index);
  cuda_drop_async(d_buffer1, stream, gpu_index);
  cuda_drop_async(d_buffer2, stream, gpu_index);
  if (max_shared_memory < memory_needed_per_block)
    cuda_drop_async(d_mem, stream, gpu_index);
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
        accumulator_c1, accumulator_c0, (1 << monomial_degree), false, 1);
    // Mask
    divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                          params::degree / params::opt>(
        accumulator_c1 + polynomial_size, accumulator_c0 + polynomial_size,
        (1 << monomial_degree), false, 1);

    monomial_degree += 1;

    // ACC = CMUX ( Ci, x^ai * ACC, ACC )
    synchronize_threads_in_block();
    cmux<Torus, STorus, params>(
        accumulator_c0, accumulator_c0, ggsw_in,
        (int8_t *)(accumulator_c0 + 4 * polynomial_size), 0, 0, 1, glwe_dim,
        polynomial_size, base_log, level_count, i);
  }
  synchronize_threads_in_block();

  // Write the output
  auto block_lwe_out = &lwe_out[blockIdx.x * (polynomial_size + 1)];

  // The blind rotation for this block is over
  // Now we can perform the sample extraction: for the body it's just
  // the resulting constant coefficient of the accumulator
  // For the mask it's more complicated
  sample_extract_mask<Torus, params>(block_lwe_out, accumulator_c0, 1);
  sample_extract_body<Torus, params>(block_lwe_out, accumulator_c0, 1);
}

template <typename Torus, typename STorus, class params>
__host__ void host_blind_rotate_and_sample_extraction(
    void *v_stream, uint32_t gpu_index, Torus *lwe_out, Torus *ggsw_in,
    Torus *lut_vector, uint32_t mbr_size, uint32_t tau, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t max_shared_memory) {

  cudaSetDevice(gpu_index);
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
      sizeof(double2) * polynomial_size / 2 + // mask_res_fft
      sizeof(double2) * polynomial_size / 2 + // body_res_fft
      sizeof(double2) * polynomial_size / 2;  // glwe_fft

  int8_t *d_mem;
  if (max_shared_memory < memory_needed_per_block)
    d_mem = (int8_t *)cuda_malloc_async(memory_needed_per_block * tau, stream,
                                        gpu_index);
  else {
    check_cuda_error(cudaFuncSetAttribute(
        device_blind_rotation_and_sample_extraction<Torus, STorus, params,
                                                    FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, memory_needed_per_block));
    check_cuda_error(cudaFuncSetCacheConfig(
        device_blind_rotation_and_sample_extraction<Torus, STorus, params,
                                                    FULLSM>,
        cudaFuncCachePreferShared));
  }

  // Applying the FFT on m^br
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;
  double2 *d_ggsw_fft_in = (double2 *)cuda_malloc_async(
      mbr_size * ggsw_size * sizeof(double), stream, gpu_index);

  batch_fft_ggsw_vector<Torus, STorus, params>(
      stream, d_ggsw_fft_in, ggsw_in, mbr_size, glwe_dimension, polynomial_size,
      level_count, gpu_index, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  //
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

  //
  cuda_drop_async(d_ggsw_fft_in, stream, gpu_index);
  if (max_shared_memory < memory_needed_per_block)
    cuda_drop_async(d_mem, stream, gpu_index);
}
#endif // VERTICAL_PACKING_H
