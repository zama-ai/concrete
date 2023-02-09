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

// number_of_inputs is the total number of LWE ciphertexts passed to CBS + VP,
// i.e. tau * p where tau is the number of LUTs (the original number of LWEs
// before bit extraction) and p is the number of extracted bits
template <typename Torus, typename STorus, class params>
__host__ void host_circuit_bootstrap_vertical_packing(
    void *v_stream, uint32_t gpu_index, Torus *lwe_array_out,
    Torus *lwe_array_in, Torus *lut_vector, double2 *fourier_bsk,
    Torus *cbs_fpksk, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t base_log_bsk, uint32_t level_count_bsk,
    uint32_t base_log_pksk, uint32_t level_count_pksk, uint32_t base_log_cbs,
    uint32_t level_count_cbs, uint32_t number_of_inputs, uint32_t tau,
    uint32_t max_shared_memory) {

  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  // allocate and initialize device pointers for circuit bootstrap
  // output ggsw array for cbs
  int ggsw_size = level_count_cbs * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * polynomial_size;
  Torus *ggsw_out = (Torus *)cuda_malloc_async(
      number_of_inputs * ggsw_size * sizeof(Torus), stream, gpu_index);
  // input lwe array for fp-ks
  Torus *lwe_array_in_fp_ks_buffer = (Torus *)cuda_malloc_async(
      number_of_inputs * level_count_cbs * (glwe_dimension + 1) *
          (polynomial_size + 1) * sizeof(Torus),
      stream, gpu_index);
  // buffer for pbs output
  Torus *lwe_array_out_pbs_buffer =
      (Torus *)cuda_malloc_async(number_of_inputs * level_count_cbs *
                                     (polynomial_size + 1) * sizeof(Torus),
                                 stream, gpu_index);
  // vector for shifted lwe input
  Torus *lwe_array_in_shifted_buffer = (Torus *)cuda_malloc_async(
      number_of_inputs * level_count_cbs * (lwe_dimension + 1) * sizeof(Torus),
      stream, gpu_index);
  // lut vector buffer for cbs
  Torus *lut_vector_cbs = (Torus *)cuda_malloc_async(
      level_count_cbs * (glwe_dimension + 1) * polynomial_size * sizeof(Torus),
      stream, gpu_index);
  // indexes of lut vectors for cbs
  uint32_t *h_lut_vector_indexes =
      (uint32_t *)malloc(number_of_inputs * level_count_cbs * sizeof(uint32_t));
  for (uint index = 0; index < level_count_cbs * number_of_inputs; index++) {
    h_lut_vector_indexes[index] = index % level_count_cbs;
  }
  uint32_t *lut_vector_indexes = (uint32_t *)cuda_malloc_async(
      number_of_inputs * level_count_cbs * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(
      lut_vector_indexes, h_lut_vector_indexes,
      number_of_inputs * level_count_cbs * sizeof(uint32_t), stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  uint32_t bits = sizeof(Torus) * 8;
  uint32_t delta_log = (bits - 1);

  host_circuit_bootstrap<Torus, params>(
      v_stream, gpu_index, ggsw_out, lwe_array_in, fourier_bsk, cbs_fpksk,
      lwe_array_in_shifted_buffer, lut_vector_cbs, lut_vector_indexes,
      lwe_array_out_pbs_buffer, lwe_array_in_fp_ks_buffer, delta_log,
      polynomial_size, glwe_dimension, lwe_dimension, level_count_bsk,
      base_log_bsk, level_count_pksk, base_log_pksk, level_count_cbs,
      base_log_cbs, number_of_inputs, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  // Free memory
  cuda_drop_async(lwe_array_in_fp_ks_buffer, stream, gpu_index);
  cuda_drop_async(lwe_array_in_shifted_buffer, stream, gpu_index);
  cuda_drop_async(lwe_array_out_pbs_buffer, stream, gpu_index);
  cuda_drop_async(lut_vector_cbs, stream, gpu_index);
  cuda_drop_async(lut_vector_indexes, stream, gpu_index);
  free(h_lut_vector_indexes);

  // number_of_inputs = tau * p is the total number of GGSWs
  // split the vec of GGSW in two, the msb GGSW is for the CMux tree and the
  // lsb GGSW is for the last blind rotation.
  uint32_t r = number_of_inputs - params::log2_degree;
  Torus *glwe_array_out = (Torus *)cuda_malloc_async(
      tau * (glwe_dimension + 1) * polynomial_size * sizeof(Torus), stream,
      gpu_index);
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

  cuda_drop_async(glwe_array_out, stream, gpu_index);
  cuda_drop_async(ggsw_out, stream, gpu_index);
}

template <typename Torus, typename STorus, class params>
__host__ void host_wop_pbs(
    void *v_stream, uint32_t gpu_index, Torus *lwe_array_out,
    Torus *lwe_array_in, Torus *lut_vector, double2 *fourier_bsk, Torus *ksk,
    Torus *cbs_fpksk, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t base_log_bsk, uint32_t level_count_bsk,
    uint32_t base_log_ksk, uint32_t level_count_ksk, uint32_t base_log_pksk,
    uint32_t level_count_pksk, uint32_t base_log_cbs, uint32_t level_count_cbs,
    uint32_t number_of_bits_of_message_including_padding,
    uint32_t number_of_bits_to_extract, uint32_t number_of_inputs,
    uint32_t max_shared_memory) {

  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  // let mut h_lut_vector_indexes = vec![0 as u32; 1];
  // indexes of lut vectors for bit extract
  uint32_t *h_lut_vector_indexes = (uint32_t *)malloc(sizeof(uint32_t));
  h_lut_vector_indexes[0] = 0;
  uint32_t *lut_vector_indexes =
      (uint32_t *)cuda_malloc_async(sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(lut_vector_indexes, h_lut_vector_indexes,
                           sizeof(uint32_t), stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  Torus *lut_pbs = (Torus *)cuda_malloc_async(
      (2 * polynomial_size) * sizeof(Torus), stream, gpu_index);
  Torus *lwe_array_in_buffer = (Torus *)cuda_malloc_async(
      (polynomial_size + 1) * sizeof(Torus), stream, gpu_index);
  Torus *lwe_array_in_shifted_buffer = (Torus *)cuda_malloc_async(
      (polynomial_size + 1) * sizeof(Torus), stream, gpu_index);
  Torus *lwe_array_out_ks_buffer = (Torus *)cuda_malloc_async(
      (lwe_dimension + 1) * sizeof(Torus), stream, gpu_index);
  Torus *lwe_array_out_pbs_buffer = (Torus *)cuda_malloc_async(
      (polynomial_size + 1) * sizeof(Torus), stream, gpu_index);
  Torus *lwe_array_out_bit_extract = (Torus *)cuda_malloc_async(
      (lwe_dimension + 1) * (number_of_bits_of_message_including_padding) *
          sizeof(Torus),
      stream, gpu_index);
  uint32_t ciphertext_n_bits = sizeof(Torus) * 8;
  uint32_t delta_log =
      ciphertext_n_bits - number_of_bits_of_message_including_padding;
  host_extract_bits<Torus, params>(
      v_stream, gpu_index, lwe_array_out_bit_extract, lwe_array_in,
      lwe_array_in_buffer, lwe_array_in_shifted_buffer, lwe_array_out_ks_buffer,
      lwe_array_out_pbs_buffer, lut_pbs, lut_vector_indexes, ksk, fourier_bsk,
      number_of_bits_to_extract, delta_log, polynomial_size, lwe_dimension,
      glwe_dimension, base_log_bsk, level_count_bsk, base_log_ksk,
      level_count_ksk, number_of_inputs, max_shared_memory);
  check_cuda_error(cudaGetLastError());
  cuda_drop_async(lut_pbs, stream, gpu_index);
  cuda_drop_async(lut_vector_indexes, stream, gpu_index);
  cuda_drop_async(lwe_array_in_buffer, stream, gpu_index);
  cuda_drop_async(lwe_array_in_shifted_buffer, stream, gpu_index);
  cuda_drop_async(lwe_array_out_ks_buffer, stream, gpu_index);
  cuda_drop_async(lwe_array_out_pbs_buffer, stream, gpu_index);

  host_circuit_bootstrap_vertical_packing<Torus, STorus, params>(
      v_stream, gpu_index, lwe_array_out, lwe_array_out_bit_extract, lut_vector,
      fourier_bsk, cbs_fpksk, glwe_dimension, lwe_dimension, polynomial_size,
      base_log_bsk, level_count_bsk, base_log_pksk, level_count_pksk,
      base_log_cbs, level_count_cbs,
      number_of_inputs * number_of_bits_to_extract, number_of_inputs,
      max_shared_memory);

  check_cuda_error(cudaGetLastError());
  cuda_drop_async(lwe_array_out_bit_extract, stream, gpu_index);
}
#endif // WOP_PBS_H
