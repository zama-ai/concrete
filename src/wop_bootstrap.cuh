#ifndef WOP_PBS_H
#define WOP_PBS_H

#include "cooperative_groups.h"

#include "bit_extraction.cuh"
#include "bootstrap.h"
#include "circuit_bootstrap.cuh"
#include "device.h"
#include "utils/kernel_dimensions.cuh"
#include "utils/timer.cuh"
#include "vertical_packing.cuh"

template <typename Torus, class params>
__global__ void device_build_lut(Torus *lut_out, Torus *lut_in,
                                 uint32_t glwe_dimension, uint32_t lut_number) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < glwe_dimension * params::degree * lut_number) {
    int lut_index = index / (glwe_dimension * params::degree);
    for (int j = 0; j < glwe_dimension; j++) {
      lut_out[index + lut_index * (glwe_dimension + 1) * params::degree +
              j * params::degree] = 0;
    }
    lut_out[index + lut_index * (glwe_dimension + 1) * params::degree +
            glwe_dimension * params::degree] = lut_in[index];
  }
}

template <typename Torus>
__host__ __device__ int
get_buffer_size_cbs_vp(uint32_t glwe_dimension, uint32_t lwe_dimension,
                       uint32_t polynomial_size, uint32_t level_count_cbs,
                       uint32_t number_of_inputs, uint32_t tau) {

  int ggsw_size = level_count_cbs * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * polynomial_size;
  return number_of_inputs * level_count_cbs *
             sizeof(Torus)                              // lut_vector_indexes
         + number_of_inputs * ggsw_size * sizeof(Torus) // ggsw_out
         +
         number_of_inputs * level_count_cbs * (glwe_dimension + 1) *
             (polynomial_size + 1) * sizeof(Torus) // lwe_array_in_fp_ks_buffer
         + number_of_inputs * level_count_cbs * (polynomial_size + 1) *
               sizeof(Torus) // lwe_array_out_pbs_buffer
         + number_of_inputs * level_count_cbs * (lwe_dimension + 1) *
               sizeof(Torus) // lwe_array_in_shifted_buffer
         + level_count_cbs * (glwe_dimension + 1) * polynomial_size *
               sizeof(Torus) // lut_vector_cbs
         + tau * (glwe_dimension + 1) * polynomial_size *
               sizeof(Torus); // glwe_array_out
}

template <typename Torus, typename STorus, typename params>
__host__ void scratch_circuit_bootstrap_vertical_packing(
    void *v_stream, uint32_t gpu_index, int8_t **cbs_vp_buffer,
    uint32_t *cbs_delta_log, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t level_count_cbs,
    uint32_t number_of_inputs, uint32_t tau, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {

  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  // Allocate lut vector indexes on the CPU first to avoid blocking the stream
  Torus *h_lut_vector_indexes =
      (Torus *)malloc(number_of_inputs * level_count_cbs * sizeof(Torus));
  uint32_t r = number_of_inputs - params::log2_degree;
  // allocate and initialize device pointers for circuit bootstrap and vertical
  // packing
  if (allocate_gpu_memory) {
    int buffer_size = get_buffer_size_cbs_vp<Torus>(
        glwe_dimension, lwe_dimension, polynomial_size, level_count_cbs,
        number_of_inputs, tau);
    *cbs_vp_buffer =
        (int8_t *)cuda_malloc_async(buffer_size, stream, gpu_index);
  }
  // indexes of lut vectors for cbs
  for (uint index = 0; index < level_count_cbs * number_of_inputs; index++) {
    h_lut_vector_indexes[index] = index % level_count_cbs;
  }
  // lut_vector_indexes is the first buffer in the cbs_vp_buffer
  cuda_memcpy_async_to_gpu((Torus *)*cbs_vp_buffer, h_lut_vector_indexes,
                           number_of_inputs * level_count_cbs * sizeof(Torus),
                           stream, gpu_index);
  check_cuda_error(cudaStreamSynchronize(*stream));
  free(h_lut_vector_indexes);
  check_cuda_error(cudaGetLastError());

  uint32_t bits = sizeof(Torus) * 8;
  *cbs_delta_log = (bits - 1);
}

/*
 * Cleanup functions free the necessary data on the GPU and on the CPU.
 * Data that lives on the CPU is prefixed with `h_`. This cleanup function thus
 * frees the data for the circuit bootstrap and vertical packing on GPU:
 * - ggsw_out
 * - lwe_array_in_fp_ks_buffer
 * - lwe_array_out_pbs_buffer
 * - lwe_array_in_shifted buffer
 * - lut_vector_cbs
 * - lut_vector_indexes
 */
__host__ void
cleanup_circuit_bootstrap_vertical_packing(void *v_stream, uint32_t gpu_index,
                                           int8_t **cbs_vp_buffer) {

  auto stream = static_cast<cudaStream_t *>(v_stream);
  // Free memory
  cuda_drop_async(*cbs_vp_buffer, stream, gpu_index);
}

// number_of_inputs is the total number of LWE ciphertexts passed to CBS + VP,
// i.e. tau * p where tau is the number of LUTs (the original number of LWEs
// before bit extraction) and p is the number of extracted bits
template <typename Torus, typename STorus, class params>
__host__ void host_circuit_bootstrap_vertical_packing(
    void *v_stream, uint32_t gpu_index, Torus *lwe_array_out,
    Torus *lwe_array_in, Torus *lut_vector, double2 *fourier_bsk,
    Torus *cbs_fpksk, int8_t *cbs_vp_buffer, uint32_t cbs_delta_log,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t base_log_bsk, uint32_t level_count_bsk, uint32_t base_log_pksk,
    uint32_t level_count_pksk, uint32_t base_log_cbs, uint32_t level_count_cbs,
    uint32_t number_of_inputs, uint32_t tau, uint32_t max_shared_memory) {

  int ggsw_size = level_count_cbs * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * polynomial_size;
  Torus *lut_vector_indexes = (Torus *)cbs_vp_buffer;
  Torus *ggsw_out = (Torus *)lut_vector_indexes +
                    (ptrdiff_t)(number_of_inputs * level_count_cbs);
  Torus *lwe_array_in_fp_ks_buffer =
      (Torus *)ggsw_out + (ptrdiff_t)(number_of_inputs * ggsw_size);
  Torus *lwe_array_out_pbs_buffer =
      (Torus *)lwe_array_in_fp_ks_buffer +
      (ptrdiff_t)(number_of_inputs * level_count_cbs * (glwe_dimension + 1) *
                  (polynomial_size + 1));
  Torus *lwe_array_in_shifted_buffer =
      (Torus *)lwe_array_out_pbs_buffer +
      (ptrdiff_t)(number_of_inputs * level_count_cbs * (polynomial_size + 1));
  Torus *lut_vector_cbs =
      (Torus *)lwe_array_in_shifted_buffer +
      (ptrdiff_t)(number_of_inputs * level_count_cbs * (lwe_dimension + 1));
  Torus *glwe_array_out =
      (Torus *)lut_vector_cbs +
      (ptrdiff_t)(level_count_cbs * (glwe_dimension + 1) * polynomial_size);

  host_circuit_bootstrap<Torus, params>(
      v_stream, gpu_index, ggsw_out, lwe_array_in, fourier_bsk, cbs_fpksk,
      lwe_array_in_shifted_buffer, lut_vector_cbs, lut_vector_indexes,
      lwe_array_out_pbs_buffer, lwe_array_in_fp_ks_buffer, cbs_delta_log,
      polynomial_size, glwe_dimension, lwe_dimension, level_count_bsk,
      base_log_bsk, level_count_pksk, base_log_pksk, level_count_cbs,
      base_log_cbs, number_of_inputs, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  // number_of_inputs = tau * p is the total number of GGSWs
  // split the vec of GGSW in two, the msb GGSW is for the CMux tree and the
  // lsb GGSW is for the last blind rotation.
  uint32_t r = number_of_inputs - params::log2_degree;
  // CMUX Tree
  // r = tau * p - log2(N)
  host_cmux_tree<Torus, STorus, params>(
      v_stream, gpu_index, glwe_array_out, ggsw_out, lut_vector, glwe_dimension,
      polynomial_size, base_log_cbs, level_count_cbs, r, tau,
      max_shared_memory);
  check_cuda_error(cudaGetLastError());

  // Blind rotation + sample extraction
  // mbr = tau * p - r = log2(N)
  Torus *br_ggsw = (Torus *)ggsw_out +
                   (ptrdiff_t)(r * level_count_cbs * (glwe_dimension + 1) *
                               (glwe_dimension + 1) * polynomial_size);
  host_blind_rotate_and_sample_extraction<Torus, STorus, params>(
      v_stream, gpu_index, lwe_array_out, br_ggsw, glwe_array_out,
      number_of_inputs - r, tau, glwe_dimension, polynomial_size, base_log_cbs,
      level_count_cbs, max_shared_memory);
}

template <typename Torus>
__host__ __device__ int
get_buffer_size_wop_pbs(uint32_t glwe_dimension, uint32_t lwe_dimension,
                        uint32_t polynomial_size, uint32_t level_count_cbs,
                        uint32_t number_of_bits_of_message_including_padding,
                        uint32_t number_of_bits_to_extract,
                        uint32_t number_of_inputs) {

  return sizeof(Torus) // lut_vector_indexes
         + ((glwe_dimension + 1) * polynomial_size) * sizeof(Torus) // lut_pbs
         + (polynomial_size + 1) * sizeof(Torus) // lwe_array_in_buffer
         + (polynomial_size + 1) * sizeof(Torus) // lwe_array_in_shifted_buffer
         + (lwe_dimension + 1) * sizeof(Torus)   // lwe_array_out_ks_buffer
         + (polynomial_size + 1) * sizeof(Torus) // lwe_array_out_pbs_buffer
         + (lwe_dimension + 1) *                 // lwe_array_out_bit_extract
               (number_of_bits_of_message_including_padding) * sizeof(Torus);
}

template <typename Torus, typename STorus, typename params>
__host__ void
scratch_wop_pbs(void *v_stream, uint32_t gpu_index, int8_t **wop_pbs_buffer,
                uint32_t *delta_log, uint32_t *cbs_delta_log,
                uint32_t glwe_dimension, uint32_t lwe_dimension,
                uint32_t polynomial_size, uint32_t level_count_cbs,
                uint32_t number_of_bits_of_message_including_padding,
                uint32_t number_of_bits_to_extract, uint32_t number_of_inputs,
                uint32_t max_shared_memory) {

  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  int wop_pbs_buffer_size = get_buffer_size_wop_pbs<Torus>(
      glwe_dimension, lwe_dimension, polynomial_size, level_count_cbs,
      number_of_bits_of_message_including_padding, number_of_bits_to_extract,
      number_of_inputs);
  uint32_t cbs_vp_number_of_inputs =
      number_of_inputs * number_of_bits_to_extract;
  uint32_t tau = number_of_inputs;
  uint32_t r = cbs_vp_number_of_inputs - params::log2_degree;
  int buffer_size = get_buffer_size_cbs_vp<Torus>(
                        glwe_dimension, lwe_dimension, polynomial_size,
                        level_count_cbs, cbs_vp_number_of_inputs, tau) +
                    wop_pbs_buffer_size;

  *wop_pbs_buffer = (int8_t *)cuda_malloc_async(buffer_size, stream, gpu_index);

  // indexes of lut vectors for bit extract
  Torus h_lut_vector_indexes = 0;
  // lut_vector_indexes is the first array in the wop_pbs buffer
  cuda_memcpy_async_to_gpu(*wop_pbs_buffer, (int8_t *)&h_lut_vector_indexes,
                           sizeof(Torus), stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t ciphertext_total_bits_count = sizeof(Torus) * 8;
  *delta_log =
      ciphertext_total_bits_count - number_of_bits_of_message_including_padding;
  int8_t *cbs_vp_buffer =
      (int8_t *)*wop_pbs_buffer + (ptrdiff_t)wop_pbs_buffer_size;
  scratch_circuit_bootstrap_vertical_packing<Torus, STorus, params>(
      v_stream, gpu_index, &cbs_vp_buffer, cbs_delta_log, glwe_dimension,
      lwe_dimension, polynomial_size, level_count_cbs,
      number_of_inputs * number_of_bits_to_extract, number_of_inputs,
      max_shared_memory, false);
}

/*
 * Cleanup functions free the necessary data on the GPU and on the CPU.
 * Data that lives on the CPU is prefixed with `h_`. This cleanup function thus
 * frees the data for the wop PBS on GPU in wop_pbs_buffer
 */
__host__ void cleanup_wop_pbs(void *v_stream, uint32_t gpu_index,
                              int8_t **wop_pbs_buffer) {
  auto stream = static_cast<cudaStream_t *>(v_stream);
  cuda_drop_async(*wop_pbs_buffer, stream, gpu_index);
}

template <typename Torus, typename STorus, class params>
__host__ void host_wop_pbs(
    void *v_stream, uint32_t gpu_index, Torus *lwe_array_out,
    Torus *lwe_array_in, Torus *lut_vector, double2 *fourier_bsk, Torus *ksk,
    Torus *cbs_fpksk, int8_t *wop_pbs_buffer, uint32_t cbs_delta_log,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t base_log_bsk, uint32_t level_count_bsk, uint32_t base_log_ksk,
    uint32_t level_count_ksk, uint32_t base_log_pksk, uint32_t level_count_pksk,
    uint32_t base_log_cbs, uint32_t level_count_cbs,
    uint32_t number_of_bits_of_message_including_padding,
    uint32_t number_of_bits_to_extract, uint32_t delta_log,
    uint32_t number_of_inputs, uint32_t max_shared_memory) {

  // lut_vector_indexes is the first array in the wop_pbs buffer
  Torus *lut_vector_indexes = (Torus *)wop_pbs_buffer;
  Torus *lut_pbs = (Torus *)lut_vector_indexes + (ptrdiff_t)(1);
  Torus *lwe_array_in_buffer =
      (Torus *)lut_pbs + (ptrdiff_t)((glwe_dimension + 1) * polynomial_size);
  Torus *lwe_array_in_shifted_buffer =
      (Torus *)lwe_array_in_buffer + (ptrdiff_t)(polynomial_size + 1);
  Torus *lwe_array_out_ks_buffer =
      (Torus *)lwe_array_in_shifted_buffer + (ptrdiff_t)(polynomial_size + 1);
  Torus *lwe_array_out_pbs_buffer =
      (Torus *)lwe_array_out_ks_buffer + (ptrdiff_t)(lwe_dimension + 1);
  Torus *lwe_array_out_bit_extract =
      (Torus *)lwe_array_out_pbs_buffer + (ptrdiff_t)(polynomial_size + 1);
  host_extract_bits<Torus, params>(
      v_stream, gpu_index, lwe_array_out_bit_extract, lwe_array_in,
      lwe_array_in_buffer, lwe_array_in_shifted_buffer, lwe_array_out_ks_buffer,
      lwe_array_out_pbs_buffer, lut_pbs, lut_vector_indexes, ksk, fourier_bsk,
      number_of_bits_to_extract, delta_log, polynomial_size, lwe_dimension,
      glwe_dimension, base_log_bsk, level_count_bsk, base_log_ksk,
      level_count_ksk, number_of_inputs, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  int8_t *cbs_vp_buffer =
      (int8_t *)wop_pbs_buffer +
      (ptrdiff_t)get_buffer_size_wop_pbs<Torus>(
          glwe_dimension, lwe_dimension, polynomial_size, level_count_cbs,
          number_of_bits_of_message_including_padding,
          number_of_bits_to_extract, number_of_inputs);
  host_circuit_bootstrap_vertical_packing<Torus, STorus, params>(
      v_stream, gpu_index, lwe_array_out, lwe_array_out_bit_extract, lut_vector,
      fourier_bsk, cbs_fpksk, cbs_vp_buffer, cbs_delta_log, glwe_dimension,
      lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
      base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
      number_of_inputs * number_of_bits_to_extract, number_of_inputs,
      max_shared_memory);
  check_cuda_error(cudaGetLastError());
}
#endif // WOP_PBS_H
