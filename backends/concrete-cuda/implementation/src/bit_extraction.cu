#include "bit_extraction.cuh"

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the bit extraction on 32 bits inputs, into `cbs_buffer`. It also
 * configures SM options on the GPU in case FULLSM mode is going to be used.
 */
void scratch_cuda_extract_bits_32(
    void *v_stream, uint32_t gpu_index, int8_t **bit_extract_buffer,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t number_of_inputs, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {

  switch (polynomial_size) {
  case 256:
    scratch_extract_bits<uint32_t, int32_t, Degree<256>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 512:
    scratch_extract_bits<uint32_t, int32_t, Degree<512>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 1024:
    scratch_extract_bits<uint32_t, int32_t, Degree<1024>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 2048:
    scratch_extract_bits<uint32_t, int32_t, Degree<2048>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 4096:
    scratch_extract_bits<uint32_t, int32_t, Degree<4096>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 8192:
    scratch_extract_bits<uint32_t, int32_t, Degree<8192>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the bit extraction on 64 bits inputs, into `cbs_buffer`. It also
 * configures SM options on the GPU in case FULLSM mode is going to be used.
 */
void scratch_cuda_extract_bits_64(
    void *v_stream, uint32_t gpu_index, int8_t **bit_extract_buffer,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t number_of_inputs, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {

  switch (polynomial_size) {
  case 256:
    scratch_extract_bits<uint64_t, int64_t, Degree<256>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 512:
    scratch_extract_bits<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 1024:
    scratch_extract_bits<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 2048:
    scratch_extract_bits<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 4096:
    scratch_extract_bits<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  case 8192:
    scratch_extract_bits<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, bit_extract_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, level_count, number_of_inputs, max_shared_memory,
        allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/* Perform bit extract on a batch of 32 bit LWE ciphertexts.
 * See the corresponding function on 64 bit LWE ciphertexts for more details.
 */
void cuda_extract_bits_32(void *v_stream, uint32_t gpu_index,
                          void *list_lwe_array_out, void *lwe_array_in,
                          int8_t *bit_extract_buffer, void *ksk,
                          void *fourier_bsk, uint32_t number_of_bits,
                          uint32_t delta_log, uint32_t lwe_dimension_in,
                          uint32_t lwe_dimension_out, uint32_t glwe_dimension,
                          uint32_t polynomial_size, uint32_t base_log_bsk,
                          uint32_t level_count_bsk, uint32_t base_log_ksk,
                          uint32_t level_count_ksk, uint32_t number_of_samples,
                          uint32_t max_shared_memory) {
  assert(("Error (GPU extract bits): base log should be <= 32",
          base_log_bsk <= 32));
  assert(("Error (GPU extract bits): polynomial_size should be one of "
          "256, 512, 1024, 2048, 4096, 8192",
          polynomial_size == 256 || polynomial_size == 512 ||
              polynomial_size == 1024 || polynomial_size == 2048 ||
              polynomial_size == 4096 || polynomial_size == 8192));
  // The number of samples should be lower than four time the number of
  // streaming multiprocessors divided by (4 * (k + 1) * l) (the factor 4 being
  // related to the occupancy of 50%). The only supported value for k is 1, so
  // k + 1 = 2 for now.
  int number_of_sm = 0;
  cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, 0);
  assert(("Error (GPU extract bits): the number of input LWEs must be lower or "
          "equal to the "
          "number of streaming multiprocessors on the device divided by 8 * "
          "level_count_bsk",
          number_of_samples <= number_of_sm / 4. / 2. / level_count_bsk));

  switch (polynomial_size) {
  case 256:
    host_extract_bits<uint32_t, Degree<256>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, bit_extract_buffer, (uint32_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 512:
    host_extract_bits<uint32_t, Degree<512>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, bit_extract_buffer, (uint32_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 1024:
    host_extract_bits<uint32_t, Degree<1024>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, bit_extract_buffer, (uint32_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 2048:
    host_extract_bits<uint32_t, Degree<2048>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, bit_extract_buffer, (uint32_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 4096:
    host_extract_bits<uint32_t, Degree<4096>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, bit_extract_buffer, (uint32_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 8192:
    host_extract_bits<uint32_t, Degree<8192>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, bit_extract_buffer, (uint32_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  default:
    break;
  }
}

/* Perform bit extract on a batch of 64 bit lwe ciphertexts.
 * - `v_stream` is a void pointer to the Cuda stream to be used in the kernel
 * launch
 * - `gpu_index` is the index of the GPU to be used in the kernel launch
 *  - 'number_of_bits' will be extracted from each ciphertext
 * starting at the bit number 'delta_log' (0-indexed) included.
 * Output bits are ordered from the MSB to LSB. Every extracted bit is
 * represented as an LWE ciphertext, containing the encryption of the bit scaled
 * by q/2.
 *  - 'list_lwe_array_out' output batch LWE ciphertexts for each bit of every
 * input ciphertext
 *  - 'lwe_array_in' batch of input LWE ciphertexts, with size -
 * ('lwe_dimension_in' + 1) * number_of_samples * sizeof(u64)
 * The following 5 parameters are used during calculations, they are not actual
 * inputs of the function they are just allocated memory for calculation
 * process, like this, memory can be allocated once and can be used as much
 * as needed for different calls of extract_bit function.
 *  - 'lwe_array_in_buffer' same size as 'lwe_array_in'
 *  - 'lwe_array_in_shifted_buffer' same size as 'lwe_array_in'
 *  - 'lwe_array_out_ks_buffer'  with size:
 * ('lwe_dimension_out' + 1) * number_of_samples * sizeof(u64)
 *  - 'lwe_array_out_pbs_buffer' same size as 'lwe_array_in'
 *  - 'lut_pbs' with size:
 * (glwe_dimension + 1) * (lwe_dimension_in + 1) * sizeof(u64)
 * The other inputs are:
 *  - 'lut_vector_indexes' stores the index corresponding to which test
 * vector to use
 *  - 'ksk' keyswitch key
 *  - 'fourier_bsk'  complex compressed bsk in fourier domain
 *  - 'lwe_dimension_in' input LWE ciphertext dimension, supported input
 * dimensions are: {256, 512, 1024,2048, 4096, 8192}
 *  - 'lwe_dimension_out' output LWE ciphertext dimension
 *  - 'glwe_dimension' GLWE dimension,  only glwe_dimension = 1 is supported
 * for now
 *  - 'base_log_bsk' base_log for bootstrapping
 *  - 'level_count_bsk' decomposition level count for bootstrapping
 *  - 'base_log_ksk' base_log for keyswitch
 *  - 'level_count_ksk' decomposition level for keyswitch
 *  - 'number_of_samples' number of input LWE ciphertexts
 *  - 'max_shared_memory' maximum amount of shared memory to be used inside
 * device functions
 *
 * This function will call corresponding template of wrapper host function which
 * will manage the calls of device functions.
 */
void cuda_extract_bits_64(void *v_stream, uint32_t gpu_index,
                          void *list_lwe_array_out, void *lwe_array_in,
                          int8_t *bit_extract_buffer, void *ksk,
                          void *fourier_bsk, uint32_t number_of_bits,
                          uint32_t delta_log, uint32_t lwe_dimension_in,
                          uint32_t lwe_dimension_out, uint32_t glwe_dimension,
                          uint32_t polynomial_size, uint32_t base_log_bsk,
                          uint32_t level_count_bsk, uint32_t base_log_ksk,
                          uint32_t level_count_ksk, uint32_t number_of_samples,
                          uint32_t max_shared_memory) {
  assert(("Error (GPU extract bits): base log should be <= 64",
          base_log_bsk <= 64));
  assert(("Error (GPU extract bits): polynomial_size should be one of "
          "256, 512, 1024, 2048, 4096, 8192",
          polynomial_size == 256 || polynomial_size == 512 ||
              polynomial_size == 1024 || polynomial_size == 2048 ||
              polynomial_size == 4096 || polynomial_size == 8192));
  // The number of samples should be lower than four time the number of
  // streaming multiprocessors divided by (4 * (k + 1) * l) (the factor 4 being
  // related to the occupancy of 50%). The only supported value for k is 1, so
  // k + 1 = 2 for now.
  int number_of_sm = 0;
  cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, 0);
  assert(("Error (GPU extract bits): the number of input LWEs must be lower or "
          "equal to the "
          "number of streaming multiprocessors on the device divided by 8 * "
          "level_count_bsk",
          number_of_samples <= number_of_sm / 4. / 2. / level_count_bsk));

  switch (polynomial_size) {
  case 256:
    host_extract_bits<uint64_t, Degree<256>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, bit_extract_buffer, (uint64_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 512:
    host_extract_bits<uint64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, bit_extract_buffer, (uint64_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 1024:
    host_extract_bits<uint64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, bit_extract_buffer, (uint64_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 2048:
    host_extract_bits<uint64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, bit_extract_buffer, (uint64_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 4096:
    host_extract_bits<uint64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, bit_extract_buffer, (uint64_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  case 8192:
    host_extract_bits<uint64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, bit_extract_buffer, (uint64_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log, lwe_dimension_in,
        lwe_dimension_out, glwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, number_of_samples,
        max_shared_memory);
    break;
  default:
    break;
  }
}

/*
 * This cleanup function frees the data for the bit extraction on GPU in
 * bit_extract_buffer for 32 or 64 bits inputs.
 */
void cleanup_cuda_extract_bits(void *v_stream, uint32_t gpu_index,
                               int8_t **bit_extract_buffer) {
  auto stream = static_cast<cudaStream_t *>(v_stream);
  // Free memory
  cuda_drop_async(*bit_extract_buffer, stream, gpu_index);
}
