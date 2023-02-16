#include "circuit_bootstrap.cuh"
#include "circuit_bootstrap.h"

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the circuit bootstrap on 32 bits inputs, into `cbs_buffer`. It also
 * configures SM options on the GPU in case FULLSM mode is going to be used.
 */
void scratch_cuda_circuit_bootstrap_32(
    void *v_stream, uint32_t gpu_index, int8_t **cbs_buffer,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t level_count_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory, bool allocate_gpu_memory) {

  switch (polynomial_size) {
  case 512:
    scratch_circuit_bootstrap<uint32_t, int32_t, Degree<512>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 1024:
    scratch_circuit_bootstrap<uint32_t, int32_t, Degree<1024>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 2048:
    scratch_circuit_bootstrap<uint32_t, int32_t, Degree<2048>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 4096:
    scratch_circuit_bootstrap<uint32_t, int32_t, Degree<4096>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 8192:
    scratch_circuit_bootstrap<uint32_t, int32_t, Degree<8192>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the circuit bootstrap on 32 bits inputs, into `cbs_buffer`. It also
 * configures SM options on the GPU in case FULLSM mode is going to be used.
 */
void scratch_cuda_circuit_bootstrap_64(
    void *v_stream, uint32_t gpu_index, int8_t **cbs_buffer,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t level_count_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory, bool allocate_gpu_memory) {

  switch (polynomial_size) {
  case 512:
    scratch_circuit_bootstrap<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 1024:
    scratch_circuit_bootstrap<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 2048:
    scratch_circuit_bootstrap<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 4096:
    scratch_circuit_bootstrap<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 8192:
    scratch_circuit_bootstrap<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count_cbs, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/*
 * Perform circuit bootstrapping for the batch of 32 bit LWE ciphertexts.
 * Head out to the equivalent operation on 64 bits for more details.
 */
void cuda_circuit_bootstrap_32(
    void *v_stream, uint32_t gpu_index, void *ggsw_out, void *lwe_array_in,
    void *fourier_bsk, void *fp_ksk_array, void *lut_vector_indexes,
    int8_t *cbs_buffer, uint32_t delta_log, uint32_t polynomial_size,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t level_bsk,
    uint32_t base_log_bsk, uint32_t level_pksk, uint32_t base_log_pksk,
    uint32_t level_cbs, uint32_t base_log_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory) {
  assert(("Error (GPU circuit bootstrap): polynomial_size should be one of "
          "512, 1024, 2048, 4096, 8192",
          polynomial_size == 512 || polynomial_size == 1024 ||
              polynomial_size == 2048 || polynomial_size == 4096 ||
              polynomial_size == 8192));
  // The number of samples should be lower than the number of streaming
  // multiprocessors divided by (4 * (k + 1) * l) (the factor 4 being related
  // to the occupancy of 50%). The only supported value for k is 1, so
  // k + 1 = 2 for now.
  int number_of_sm = 0;
  cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, 0);
  assert(("Error (GPU extract bits): the number of input LWEs must be lower or "
          "equal to the "
          "number of streaming multiprocessors on the device divided by 8 * "
          "level_count_bsk",
          number_of_inputs <= number_of_sm / 4. / 2. / level_bsk));
  switch (polynomial_size) {
  case 512:
    host_circuit_bootstrap<uint32_t, Degree<512>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  case 1024:
    host_circuit_bootstrap<uint32_t, Degree<1024>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  case 2048:
    host_circuit_bootstrap<uint32_t, Degree<2048>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  case 4096:
    host_circuit_bootstrap<uint32_t, Degree<4096>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  case 8192:
    host_circuit_bootstrap<uint32_t, Degree<8192>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  default:
    break;
  }
}

/*
 * Perform circuit bootstrapping on a batch of 64 bit input LWE ciphertexts.
 * - `v_stream` is a void pointer to the Cuda stream to be used in the kernel
 * launch
 * - `gpu_index` is the index of the GPU to be used in the kernel launch
 *  - 'ggsw_out' output batch of ggsw with size:
 * 'number_of_inputs' * 'level_cbs' * ('glwe_dimension' + 1)^2 *
 * polynomial_size * sizeof(u64)
 *  - 'lwe_array_in' input batch of lwe ciphertexts, with size:
 * 'number_of_inputs' * '(lwe_dimension' + 1) * sizeof(u64)
 *  - 'fourier_bsk' bootstrapping key in fourier domain with size:
 * 'lwe_dimension' * 'level_bsk' * ('glwe_dimension' + 1)^2 *
 * 'polynomial_size' / 2 * sizeof(double2)
 *  - 'fp_ksk_array' batch of fp-keyswitch keys with size:
 * ('polynomial_size' + 1) * 'level_pksk' * ('glwe_dimension' + 1)^2 *
 * 'polynomial_size' * sizeof(u64)
 *  - 'cbs_buffer': buffer used during calculations, it is not an actual
 *  inputs of the function, just allocated memory for calculation
 *  process, like this, memory can be allocated once and can be used as much
 *  as needed for different calls of circuit_bootstrap function
 *
 * This function calls a wrapper to a device kernel that performs the
 * circuit bootstrap. The kernel is templatized based on integer discretization
 * and polynomial degree.
 */
void cuda_circuit_bootstrap_64(
    void *v_stream, uint32_t gpu_index, void *ggsw_out, void *lwe_array_in,
    void *fourier_bsk, void *fp_ksk_array, void *lut_vector_indexes,
    int8_t *cbs_buffer, uint32_t delta_log, uint32_t polynomial_size,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t level_bsk,
    uint32_t base_log_bsk, uint32_t level_pksk, uint32_t base_log_pksk,
    uint32_t level_cbs, uint32_t base_log_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory) {
  assert(("Error (GPU circuit bootstrap): polynomial_size should be one of "
          "512, 1024, 2048, 4096, 8192",
          polynomial_size == 512 || polynomial_size == 1024 ||
              polynomial_size == 2048 || polynomial_size == 4096 ||
              polynomial_size == 8192));
  // The number of samples should be lower than the number of streaming
  // multiprocessors divided by (4 * (k + 1) * l) (the factor 4 being related
  // to the occupancy of 50%). The only supported value for k is 1, so
  // k + 1 = 2 for now.
  int number_of_sm = 0;
  cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, 0);
  assert(("Error (GPU extract bits): the number of input LWEs must be lower or "
          "equal to the "
          "number of streaming multiprocessors on the device divided by 8 * "
          "level_count_bsk",
          number_of_inputs <= number_of_sm / 4. / 2. / level_bsk));
  // The number of samples should be lower than the number of streaming
  switch (polynomial_size) {
  case 512:
    host_circuit_bootstrap<uint64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  case 1024:
    host_circuit_bootstrap<uint64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  case 2048:
    host_circuit_bootstrap<uint64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  case 4096:
    host_circuit_bootstrap<uint64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  case 8192:
    host_circuit_bootstrap<uint64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lut_vector_indexes, cbs_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_inputs,
        max_shared_memory);
    break;
  default:
    break;
  }
}

/*
 * This cleanup function frees the data for the circuit bootstrap on GPU in
 * cbs_buffer for 32 or 64 bits inputs.
 */
void cleanup_cuda_circuit_bootstrap(void *v_stream, uint32_t gpu_index,
                                    int8_t **cbs_buffer) {
  auto stream = static_cast<cudaStream_t *>(v_stream);
  // Free memory
  cuda_drop_async(*cbs_buffer, stream, gpu_index);
}
