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
get_buffer_size_cbs_vp(uint32_t glwe_dimension, uint32_t polynomial_size,
                       uint32_t level_count_cbs, uint32_t tau,
                       uint32_t number_of_inputs) {

  int ggsw_size = level_count_cbs * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * polynomial_size;
  return number_of_inputs * level_count_cbs *
             sizeof(Torus) +                            // lut_vector_indexes
         number_of_inputs * ggsw_size * sizeof(Torus) + // ggsw_out_cbs
         tau * (glwe_dimension + 1) * polynomial_size *
             sizeof(Torus); // glwe_array_out_cmux_tree
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
  uint32_t mbr_size = number_of_inputs - r;
  // allocate and initialize device pointers for circuit bootstrap and vertical
  // packing
  if (allocate_gpu_memory) {
    int buffer_size =
        get_buffer_size_cbs_vp<Torus>(glwe_dimension, polynomial_size,
                                      level_count_cbs, tau, number_of_inputs) +
        get_buffer_size_cbs<Torus>(glwe_dimension, lwe_dimension,
                                   polynomial_size, level_count_cbs,
                                   number_of_inputs) +
        get_buffer_size_bootstrap_amortized<Torus>(
            glwe_dimension, polynomial_size, number_of_inputs * level_count_cbs,
            max_shared_memory) +
        get_buffer_size_cmux_tree<Torus>(glwe_dimension, polynomial_size,
                                         level_count_cbs, r, tau,
                                         max_shared_memory) +
        get_buffer_size_blind_rotation_sample_extraction<Torus>(
            glwe_dimension, polynomial_size, level_count_cbs, mbr_size, tau,
            max_shared_memory);
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
  scratch_circuit_bootstrap<Torus, STorus, params>(
      v_stream, gpu_index, cbs_vp_buffer, glwe_dimension, lwe_dimension,
      polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
      false);
  scratch_cmux_tree<Torus, STorus, params>(
      v_stream, gpu_index, cbs_vp_buffer, glwe_dimension, polynomial_size,
      level_count_cbs, r, tau, max_shared_memory, false);
  scratch_blind_rotation_sample_extraction<Torus, STorus, params>(
      v_stream, gpu_index, cbs_vp_buffer, glwe_dimension, polynomial_size,
      level_count_cbs, mbr_size, tau, max_shared_memory, false);
}

/*
 * Cleanup functions free the necessary data on the GPU and on the CPU.
 * Data that lives on the CPU is prefixed with `h_`. This cleanup function thus
 * frees the data for the circuit bootstrap and vertical packing on GPU
 * contained in cbs_vp_buffer
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
  int8_t *cbs_buffer =
      (int8_t *)lut_vector_indexes +
      (ptrdiff_t)(number_of_inputs * level_count_cbs * sizeof(Torus));
  int8_t *ggsw_out_cbs =
      cbs_buffer +
      (ptrdiff_t)(get_buffer_size_cbs<Torus>(glwe_dimension, lwe_dimension,
                                             polynomial_size, level_count_cbs,
                                             number_of_inputs) +
                  get_buffer_size_bootstrap_amortized<Torus>(
                      glwe_dimension, polynomial_size,
                      number_of_inputs * level_count_cbs, max_shared_memory));
  host_circuit_bootstrap<Torus, params>(
      v_stream, gpu_index, (Torus *)ggsw_out_cbs, lwe_array_in, fourier_bsk,
      cbs_fpksk, lut_vector_indexes, cbs_buffer, cbs_delta_log, polynomial_size,
      glwe_dimension, lwe_dimension, level_count_bsk, base_log_bsk,
      level_count_pksk, base_log_pksk, level_count_cbs, base_log_cbs,
      number_of_inputs, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  // number_of_inputs = tau * p is the total number of GGSWs
  // split the vec of GGSW in two, the msb GGSW is for the CMux tree and the
  // lsb GGSW is for the last blind rotation.
  uint32_t r = number_of_inputs - params::log2_degree;
  int8_t *cmux_tree_buffer =
      ggsw_out_cbs + (ptrdiff_t)(number_of_inputs * ggsw_size * sizeof(Torus));
  int8_t *glwe_array_out_cmux_tree =
      cmux_tree_buffer + (ptrdiff_t)(get_buffer_size_cmux_tree<Torus>(
                             glwe_dimension, polynomial_size, level_count_cbs,
                             r, tau, max_shared_memory));
  // CMUX Tree
  // r = tau * p - log2(N)
  host_cmux_tree<Torus, STorus, params>(
      v_stream, gpu_index, (Torus *)glwe_array_out_cmux_tree,
      (Torus *)ggsw_out_cbs, lut_vector, cmux_tree_buffer, glwe_dimension,
      polynomial_size, base_log_cbs, level_count_cbs, r, tau,
      max_shared_memory);
  check_cuda_error(cudaGetLastError());

  // Blind rotation + sample extraction
  // mbr = tau * p - r = log2(N)
  Torus *br_ggsw = (Torus *)ggsw_out_cbs +
                   (ptrdiff_t)(r * level_count_cbs * (glwe_dimension + 1) *
                               (glwe_dimension + 1) * polynomial_size);
  int8_t *br_se_buffer =
      glwe_array_out_cmux_tree +
      (ptrdiff_t)(tau * (glwe_dimension + 1) * polynomial_size * sizeof(Torus));
  host_blind_rotate_and_sample_extraction<Torus, STorus, params>(
      v_stream, gpu_index, lwe_array_out, br_ggsw,
      (Torus *)glwe_array_out_cmux_tree, br_se_buffer, number_of_inputs - r,
      tau, glwe_dimension, polynomial_size, base_log_cbs, level_count_cbs,
      max_shared_memory);
}

template <typename Torus>
__host__ __device__ int
get_buffer_size_wop_pbs(uint32_t lwe_dimension,
                        uint32_t number_of_bits_of_message_including_padding) {

  return (lwe_dimension + 1) * (number_of_bits_of_message_including_padding) *
         sizeof(Torus); // lwe_array_out_bit_extract
}

template <typename Torus, typename STorus, typename params>
__host__ void
scratch_wop_pbs(void *v_stream, uint32_t gpu_index, int8_t **wop_pbs_buffer,
                uint32_t *delta_log, uint32_t *cbs_delta_log,
                uint32_t glwe_dimension, uint32_t lwe_dimension,
                uint32_t polynomial_size, uint32_t level_count_cbs,
                uint32_t level_count_bsk,
                uint32_t number_of_bits_of_message_including_padding,
                uint32_t number_of_bits_to_extract, uint32_t number_of_inputs,
                uint32_t max_shared_memory, bool allocate_gpu_memory) {

  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  int bit_extract_buffer_size =
      get_buffer_size_extract_bits<Torus>(glwe_dimension, lwe_dimension,
                                          polynomial_size, number_of_inputs) +
      get_buffer_size_bootstrap_low_latency<Torus>(
          glwe_dimension, polynomial_size, level_count_bsk, number_of_inputs,
          max_shared_memory);
  uint32_t cbs_vp_number_of_inputs =
      number_of_inputs * number_of_bits_to_extract;
  uint32_t tau = number_of_inputs;
  uint32_t r = cbs_vp_number_of_inputs - params::log2_degree;
  uint32_t mbr_size = cbs_vp_number_of_inputs - r;
  if (allocate_gpu_memory) {
    int buffer_size =
        bit_extract_buffer_size +
        get_buffer_size_wop_pbs<Torus>(
            lwe_dimension, number_of_bits_of_message_including_padding) +
        get_buffer_size_cbs_vp<Torus>(glwe_dimension, polynomial_size,
                                      level_count_cbs, tau,
                                      cbs_vp_number_of_inputs) +
        get_buffer_size_cbs<Torus>(glwe_dimension, lwe_dimension,
                                   polynomial_size, level_count_cbs,
                                   cbs_vp_number_of_inputs) +
        get_buffer_size_bootstrap_amortized<Torus>(
            glwe_dimension, polynomial_size,
            cbs_vp_number_of_inputs * level_count_cbs, max_shared_memory) +
        get_buffer_size_cmux_tree<Torus>(glwe_dimension, polynomial_size,
                                         level_count_cbs, r, tau,
                                         max_shared_memory) +
        get_buffer_size_blind_rotation_sample_extraction<Torus>(
            glwe_dimension, polynomial_size, level_count_cbs, mbr_size, tau,
            max_shared_memory);

    *wop_pbs_buffer =
        (int8_t *)cuda_malloc_async(buffer_size, stream, gpu_index);
  }
  uint32_t ciphertext_total_bits_count = sizeof(Torus) * 8;
  *delta_log =
      ciphertext_total_bits_count - number_of_bits_of_message_including_padding;

  int8_t *bit_extract_buffer =
      (int8_t *)*wop_pbs_buffer +
      (ptrdiff_t)(get_buffer_size_wop_pbs<Torus>(
          lwe_dimension, number_of_bits_of_message_including_padding));
  scratch_extract_bits<Torus, STorus, params>(
      v_stream, gpu_index, &bit_extract_buffer, glwe_dimension, lwe_dimension,
      polynomial_size, level_count_bsk, number_of_inputs, max_shared_memory,
      false);

  int8_t *cbs_vp_buffer =
      bit_extract_buffer + (ptrdiff_t)bit_extract_buffer_size;
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

  int8_t *bit_extract_buffer = wop_pbs_buffer;
  int8_t *lwe_array_out_bit_extract =
      bit_extract_buffer +
      (ptrdiff_t)(get_buffer_size_extract_bits<Torus>(
                      glwe_dimension, lwe_dimension, polynomial_size,
                      number_of_inputs) +
                  get_buffer_size_bootstrap_low_latency<Torus>(
                      glwe_dimension, polynomial_size, level_count_bsk,
                      number_of_inputs, max_shared_memory));
  host_extract_bits<Torus, params>(
      v_stream, gpu_index, (Torus *)lwe_array_out_bit_extract, lwe_array_in,
      bit_extract_buffer, ksk, fourier_bsk, number_of_bits_to_extract,
      delta_log, glwe_dimension * polynomial_size, lwe_dimension,
      glwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
      base_log_ksk, level_count_ksk, number_of_inputs, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  int8_t *cbs_vp_buffer =
      lwe_array_out_bit_extract +
      (ptrdiff_t)(get_buffer_size_wop_pbs<Torus>(
          lwe_dimension, number_of_bits_of_message_including_padding));
  host_circuit_bootstrap_vertical_packing<Torus, STorus, params>(
      v_stream, gpu_index, lwe_array_out, (Torus *)lwe_array_out_bit_extract,
      lut_vector, fourier_bsk, cbs_fpksk, cbs_vp_buffer, cbs_delta_log,
      glwe_dimension, lwe_dimension, polynomial_size, base_log_bsk,
      level_count_bsk, base_log_pksk, level_count_pksk, base_log_cbs,
      level_count_cbs, number_of_inputs * number_of_bits_to_extract,
      number_of_inputs, max_shared_memory);
  check_cuda_error(cudaGetLastError());
}
#endif // WOP_PBS_H
