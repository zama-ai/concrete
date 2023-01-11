#include "circuit_bootstrap.cuh"

/*
 * Perform circuit bootstrapping for the batch of 32 bit LWE ciphertexts.
 * Head out to the equivalent operation on 64 bits for more details.
 */
void cuda_circuit_bootstrap_32(
    void *v_stream, uint32_t gpu_index, void *ggsw_out, void *lwe_array_in,
    void *fourier_bsk, void *fp_ksk_array, void *lwe_array_in_shifted_buffer,
    void *lut_vector, void *lut_vector_indexes, void *lwe_array_out_pbs_buffer,
    void *lwe_array_in_fp_ks_buffer, uint32_t delta_log,
    uint32_t polynomial_size, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t level_bsk, uint32_t base_log_bsk, uint32_t level_pksk,
    uint32_t base_log_pksk, uint32_t level_cbs, uint32_t base_log_cbs,
    uint32_t number_of_samples, uint32_t max_shared_memory) {
  assert(("Error (GPU circuit bootstrap): glwe_dimension should be equal to 1",
          glwe_dimension == 1));
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
          number_of_samples <= number_of_sm / 4. / 2. / level_bsk));
  switch (polynomial_size) {
  case 512:
    host_circuit_bootstrap<uint32_t, Degree<512>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lwe_array_in_shifted_buffer, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_out_pbs_buffer,
        (uint32_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
        max_shared_memory);
    break;
  case 1024:
    host_circuit_bootstrap<uint32_t, Degree<1024>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lwe_array_in_shifted_buffer, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_out_pbs_buffer,
        (uint32_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
        max_shared_memory);
    break;
  case 2048:
    host_circuit_bootstrap<uint32_t, Degree<2048>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lwe_array_in_shifted_buffer, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_out_pbs_buffer,
        (uint32_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
        max_shared_memory);
    break;
  case 4096:
    host_circuit_bootstrap<uint32_t, Degree<4096>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lwe_array_in_shifted_buffer, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_out_pbs_buffer,
        (uint32_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
        max_shared_memory);
    break;
  case 8192:
    host_circuit_bootstrap<uint32_t, Degree<8192>>(
        v_stream, gpu_index, (uint32_t *)ggsw_out, (uint32_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint32_t *)fp_ksk_array,
        (uint32_t *)lwe_array_in_shifted_buffer, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_out_pbs_buffer,
        (uint32_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
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
 * 'number_of_samples' * 'level_cbs' * ('glwe_dimension' + 1)^2 *
 * polynomial_size * sizeof(u64)
 *  - 'lwe_array_in' input batch of lwe ciphertexts, with size:
 * 'number_of_samples' * '(lwe_dimension' + 1) * sizeof(u64)
 *  - 'fourier_bsk' bootstrapping key in fourier domain with size:
 * 'lwe_dimension' * 'level_bsk' * ('glwe_dimension' + 1)^2 *
 * 'polynomial_size' / 2 * sizeof(double2)
 *  - 'fp_ksk_array' batch of fp-keyswitch keys with size:
 * ('polynomial_size' + 1) * 'level_pksk' * ('glwe_dimension' + 1)^2 *
 * 'polynomial_size' * sizeof(u64)
 *  The following 5 parameters are used during calculations, they are not actual
 *  inputs of the function, they are just allocated memory for calculation
 *  process, like this, memory can be allocated once and can be used as much
 *  as needed for different calls of circuit_bootstrap function
 *  - 'lwe_array_in_shifted_buffer' with size:
 * 'number_of_samples' * 'level_cbs' * ('lwe_dimension' + 1) * sizeof(u64)
 *  - 'lut_vector' with size:
 * 'level_cbs' * ('glwe_dimension' + 1) * 'polynomial_size' * sizeof(u64)
 *  - 'lut_vector_indexes' stores the index corresponding to which test
 *  vector to use
 *  - 'lwe_array_out_pbs_buffer' with size
 * 'number_of_samples' * 'level_cbs' * ('polynomial_size' + 1) * sizeof(u64)
 *  - 'lwe_array_in_fp_ks_buffer' with size
 * 'number_of_samples' * 'level_cbs' * ('glwe_dimension' + 1) *
 * ('polynomial_size' + 1) * sizeof(u64)
 *
 * This function calls a wrapper to a device kernel that performs the
 * circuit bootstrap. The kernel is templatized based on integer discretization
 * and polynomial degree.
 */
void cuda_circuit_bootstrap_64(
    void *v_stream, uint32_t gpu_index, void *ggsw_out, void *lwe_array_in,
    void *fourier_bsk, void *fp_ksk_array, void *lwe_array_in_shifted_buffer,
    void *lut_vector, void *lut_vector_indexes, void *lwe_array_out_pbs_buffer,
    void *lwe_array_in_fp_ks_buffer, uint32_t delta_log,
    uint32_t polynomial_size, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t level_bsk, uint32_t base_log_bsk, uint32_t level_pksk,
    uint32_t base_log_pksk, uint32_t level_cbs, uint32_t base_log_cbs,
    uint32_t number_of_samples, uint32_t max_shared_memory) {
  assert(("Error (GPU circuit bootstrap): glwe_dimension should be equal to 1",
          glwe_dimension == 1));
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
          number_of_samples <= number_of_sm / 4. / 2. / level_bsk));
  // The number of samples should be lower than the number of streaming
  switch (polynomial_size) {
  case 512:
    host_circuit_bootstrap<uint64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lwe_array_in_shifted_buffer, (uint64_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint64_t *)lwe_array_out_pbs_buffer,
        (uint64_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
        max_shared_memory);
    break;
  case 1024:
    host_circuit_bootstrap<uint64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lwe_array_in_shifted_buffer, (uint64_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint64_t *)lwe_array_out_pbs_buffer,
        (uint64_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
        max_shared_memory);
    break;
  case 2048:
    host_circuit_bootstrap<uint64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lwe_array_in_shifted_buffer, (uint64_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint64_t *)lwe_array_out_pbs_buffer,
        (uint64_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
        max_shared_memory);
    break;
  case 4096:
    host_circuit_bootstrap<uint64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lwe_array_in_shifted_buffer, (uint64_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint64_t *)lwe_array_out_pbs_buffer,
        (uint64_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
        max_shared_memory);
    break;
  case 8192:
    host_circuit_bootstrap<uint64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)ggsw_out, (uint64_t *)lwe_array_in,
        (double2 *)fourier_bsk, (uint64_t *)fp_ksk_array,
        (uint64_t *)lwe_array_in_shifted_buffer, (uint64_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint64_t *)lwe_array_out_pbs_buffer,
        (uint64_t *)lwe_array_in_fp_ks_buffer, delta_log, polynomial_size,
        glwe_dimension, lwe_dimension, level_bsk, base_log_bsk, level_pksk,
        base_log_pksk, level_cbs, base_log_cbs, number_of_samples,
        max_shared_memory);
    break;
  default:
    break;
  }
}
