#ifndef CBS_CUH
#define CBS_CUH

#include "bit_extraction.cuh"
#include "bootstrap.h"
#include "bootstrap_amortized.cuh"
#include "device.h"
#include "keyswitch.cuh"
#include "polynomial/parameters.cuh"
#include "utils/timer.cuh"

/*
 * scalar multiplication to value for batch of lwe_ciphertext
 * works for any size of lwe input
 * blockIdx.x refers to input ciphertext it
 */
template <typename Torus, class params>
__global__ void shift_lwe_cbs(Torus *dst_shift, Torus *src, Torus value,
                              size_t lwe_size) {

  size_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  size_t threads_per_block = blockDim.x;
  size_t opt = lwe_size / threads_per_block;
  size_t rem = lwe_size & (threads_per_block - 1);

  auto cur_dst = &dst_shift[blockId * lwe_size];
  auto cur_src = &src[blockIdx.y * lwe_size];

  size_t tid = threadIdx.x;
  for (size_t i = 0; i < opt; i++) {
    cur_dst[tid] = cur_src[tid] * value;
    tid += threads_per_block;
  }

  if (threadIdx.x < rem)
    cur_dst[tid] = cur_src[tid] * value;
}

/*
 * Fill lut, equivalent to trivial encryption as mask is 0s.
 * The LUT is filled with -alpha in each coefficient where
 * alpha = 2^{log(q) - 1 - base_log * level}
 * blockIdx.x refers to lut id
 * value is not passed and calculated inside function because lut id is one
 * of the variable.
 */
template <typename Torus, class params>
__global__ void fill_lut_body_for_cbs(Torus *lut, uint32_t ciphertext_n_bits,
                                      uint32_t base_log_cbs) {

  Torus *cur_mask = &lut[blockIdx.x * 2 * params::degree];
  Torus *cur_poly = &lut[blockIdx.x * 2 * params::degree + params::degree];
  size_t tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    cur_mask[tid] = 0;
    cur_poly[tid] =
        0ll -
        (1ll << (ciphertext_n_bits - 1 - base_log_cbs * (blockIdx.x + 1)));
    tid += params::degree / params::opt;
  }
}

/*
 * copy pbs result (glwe_dimension + 1) times to be an input of fp-ks
 * each of the input ciphertext from lwe_src is  copied (glwe_dimension + 1)
 * times inside lwe_dst, and then value is added to the body.
 * blockIdx.x refers to destination lwe ciphertext id: 'dst_lwe_id'
 * 'src_lwe_id' = 'dst_lwe_id' / (glwe_dimension + 1)
 *
 * example: glwe_dimension = 1
 *                 src_0         ...          src_n
 *                  / \                        / \
 *                 /   \                      /   \
 *             dst_0  dst_1               dst_2n  dst_2n+1
 */
template <typename Torus, class params>
__global__ void copy_add_lwe_cbs(Torus *lwe_dst, Torus *lwe_src,
                                 uint32_t ciphertext_n_bits,
                                 uint32_t base_log_cbs, uint32_t level_cbs) {
  size_t tid = threadIdx.x;
  size_t dst_lwe_id = blockIdx.x;
  size_t src_lwe_id = dst_lwe_id / 2;
  size_t cur_cbs_level = src_lwe_id % level_cbs + 1;

  auto cur_src = &lwe_src[src_lwe_id * (params::degree + 1)];
  auto cur_dst = &lwe_dst[dst_lwe_id * (params::degree + 1)];
#pragma unroll
  for (int i = 0; i < params::opt; i++) {
    cur_dst[tid] = cur_src[tid];
    tid += params::degree / params::opt;
  }
  Torus val = 1ll << (ciphertext_n_bits - 1 - base_log_cbs * cur_cbs_level);
  if (threadIdx.x == 0) {
    cur_dst[params::degree] = cur_src[params::degree] + val;
  }
}

template <typename Torus>
__host__ __device__ int
get_buffer_size_cbs(uint32_t glwe_dimension, uint32_t lwe_dimension,
                    uint32_t polynomial_size, uint32_t level_count_cbs,
                    uint32_t number_of_inputs) {

  return number_of_inputs * level_count_cbs * (glwe_dimension + 1) *
             (polynomial_size + 1) *
             sizeof(Torus) + // lwe_array_in_fp_ks_buffer
         number_of_inputs * level_count_cbs * (polynomial_size + 1) *
             sizeof(Torus) + // lwe_array_out_pbs_buffer
         number_of_inputs * level_count_cbs * (lwe_dimension + 1) *
             sizeof(Torus) + // lwe_array_in_shifted_buffer
         level_count_cbs * (glwe_dimension + 1) * polynomial_size *
             sizeof(Torus); // lut_vector_cbs
}

template <typename Torus, typename STorus, typename params>
__host__ void scratch_circuit_bootstrap(
    void *v_stream, uint32_t gpu_index, int8_t **cbs_buffer,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t level_count_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory, bool allocate_gpu_memory) {

  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  int pbs_count = number_of_inputs * level_count_cbs;
  // allocate and initialize device pointers for circuit bootstrap and vertical
  // packing
  if (allocate_gpu_memory) {
    int buffer_size =
        get_buffer_size_cbs<Torus>(glwe_dimension, lwe_dimension,
                                   polynomial_size, level_count_cbs,
                                   number_of_inputs) +
        get_buffer_size_bootstrap_amortized<Torus>(
            glwe_dimension, polynomial_size, pbs_count, max_shared_memory);
    *cbs_buffer = (int8_t *)cuda_malloc_async(buffer_size, stream, gpu_index);
  }

  scratch_bootstrap_amortized<Torus, STorus, params>(
      v_stream, gpu_index, cbs_buffer, glwe_dimension, polynomial_size,
      pbs_count, max_shared_memory, false);
}

/*
 * Host function for cuda circuit bootstrap.
 * It executes device functions in specific order and manages
 * parallelism
 */
template <typename Torus, class params>
__host__ void host_circuit_bootstrap(
    void *v_stream, uint32_t gpu_index, Torus *ggsw_out, Torus *lwe_array_in,
    double2 *fourier_bsk, Torus *fp_ksk_array, Torus *lut_vector_indexes,
    int8_t *cbs_buffer, uint32_t delta_log, uint32_t polynomial_size,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t level_bsk,
    uint32_t base_log_bsk, uint32_t level_pksk, uint32_t base_log_pksk,
    uint32_t level_cbs, uint32_t base_log_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory) {
  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  uint32_t ciphertext_n_bits = sizeof(Torus) * 8;
  uint32_t lwe_size = lwe_dimension + 1;
  int pbs_count = number_of_inputs * level_cbs;

  dim3 blocks(level_cbs, number_of_inputs, 1);
  int threads = 256;

  Torus *lwe_array_in_fp_ks_buffer = (Torus *)cbs_buffer;
  Torus *lwe_array_out_pbs_buffer =
      (Torus *)lwe_array_in_fp_ks_buffer +
      (ptrdiff_t)(number_of_inputs * level_cbs * (glwe_dimension + 1) *
                  (polynomial_size + 1));
  Torus *lwe_array_in_shifted_buffer =
      (Torus *)lwe_array_out_pbs_buffer +
      (ptrdiff_t)(number_of_inputs * level_cbs * (polynomial_size + 1));
  Torus *lut_vector =
      (Torus *)lwe_array_in_shifted_buffer +
      (ptrdiff_t)(number_of_inputs * level_cbs * (lwe_dimension + 1));
  int8_t *pbs_buffer =
      (int8_t *)lut_vector + (ptrdiff_t)(level_cbs * (glwe_dimension + 1) *
                                         polynomial_size * sizeof(Torus));

  // Shift message LSB on padding bit, at this point we expect to have messages
  // with only 1 bit of information
  shift_lwe_cbs<Torus, params><<<blocks, threads, 0, *stream>>>(
      lwe_array_in_shifted_buffer, lwe_array_in,
      1LL << (ciphertext_n_bits - delta_log - 1), lwe_size);

  // Add q/4 to center the error while computing a negacyclic LUT
  add_to_body<Torus>
      <<<pbs_count, 1, 0, *stream>>>(lwe_array_in_shifted_buffer, lwe_dimension,
                                     1ll << (ciphertext_n_bits - 2));
  // Fill lut (equivalent to trivial encryption as mask is 0s)
  // The LUT is filled with -alpha in each coefficient where
  // alpha = 2^{log(q) - 1 - base_log * level}
  fill_lut_body_for_cbs<Torus, params>
      <<<level_cbs, params::degree / params::opt, 0, *stream>>>(
          lut_vector, ciphertext_n_bits, base_log_cbs);

  // Applying a negacyclic LUT on a ciphertext with one bit of message in the
  // MSB and no bit of padding
  host_bootstrap_amortized<Torus, params>(
      v_stream, gpu_index, lwe_array_out_pbs_buffer, lut_vector,
      lut_vector_indexes, lwe_array_in_shifted_buffer, fourier_bsk, pbs_buffer,
      glwe_dimension, lwe_dimension, polynomial_size, base_log_bsk, level_bsk,
      pbs_count, level_cbs, 0, max_shared_memory);

  dim3 copy_grid(pbs_count * (glwe_dimension + 1), 1, 1);
  dim3 copy_block(params::degree / params::opt, 1, 1);
  // Add q/4 to center the error while computing a negacyclic LUT
  // copy pbs result (glwe_dimension + 1) times to be an input of fp-ks
  copy_add_lwe_cbs<Torus, params><<<copy_grid, copy_block, 0, *stream>>>(
      lwe_array_in_fp_ks_buffer, lwe_array_out_pbs_buffer, ciphertext_n_bits,
      base_log_cbs, level_cbs);

  cuda_fp_keyswitch_lwe_to_glwe(
      v_stream, gpu_index, ggsw_out, lwe_array_in_fp_ks_buffer, fp_ksk_array,
      polynomial_size, glwe_dimension, polynomial_size, base_log_pksk,
      level_pksk, pbs_count * (glwe_dimension + 1), glwe_dimension + 1);
}

#endif // CBS_CUH
