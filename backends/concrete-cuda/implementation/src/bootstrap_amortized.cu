#include "bootstrap_amortized.cuh"

/*
 * Returns the buffer size for 64 bits executions
 */
uint64_t get_buffer_size_bootstrap_amortized_64(
    uint32_t glwe_dimension, uint32_t polynomial_size,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory) {
  return get_buffer_size_bootstrap_amortized<uint64_t>(
      glwe_dimension, polynomial_size, input_lwe_ciphertext_count,
      max_shared_memory);
}

/*
 * Runs standard checks to validate the inputs
 */
void checks_fast_bootstrap_amortized(int polynomial_size) {
  assert(
      ("Error (GPU amortized PBS): polynomial size should be one of 256, 512, "
       "1024, 2048, 4096, 8192, 16384",
       polynomial_size == 256 || polynomial_size == 512 ||
           polynomial_size == 1024 || polynomial_size == 2048 ||
           polynomial_size == 4096 || polynomial_size == 8192 ||
           polynomial_size == 16384));
}

/*
 * Runs standard checks to validate the inputs
 */
void checks_bootstrap_amortized(int nbits, int base_log, int polynomial_size) {
  assert(("Error (GPU amortized PBS): base log should be <= nbits",
          base_log <= nbits));
  checks_fast_bootstrap_amortized(polynomial_size);
}

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the amortized PBS on 32 bits inputs, into `pbs_buffer`. It also
 * configures SM options on the GPU in case FULLSM or PARTIALSM mode is going to
 * be used.
 */
void scratch_cuda_bootstrap_amortized_32(void *v_stream, uint32_t gpu_index,
                                         int8_t **pbs_buffer,
                                         uint32_t glwe_dimension,
                                         uint32_t polynomial_size,
                                         uint32_t input_lwe_ciphertext_count,
                                         uint32_t max_shared_memory,
                                         bool allocate_gpu_memory) {
  checks_fast_bootstrap_amortized(polynomial_size);

  switch (polynomial_size) {
  case 256:
    scratch_bootstrap_amortized<uint32_t, int32_t, AmortizedDegree<256>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 512:
    scratch_bootstrap_amortized<uint32_t, int32_t, AmortizedDegree<512>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 1024:
    scratch_bootstrap_amortized<uint32_t, int32_t, AmortizedDegree<1024>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 2048:
    scratch_bootstrap_amortized<uint32_t, int32_t, AmortizedDegree<2048>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 4096:
    scratch_bootstrap_amortized<uint32_t, int32_t, AmortizedDegree<4096>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 8192:
    scratch_bootstrap_amortized<uint32_t, int32_t, AmortizedDegree<8192>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 16384:
    scratch_bootstrap_amortized<uint32_t, int32_t, AmortizedDegree<16384>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the amortized PBS on 64 bits inputs, into `pbs_buffer`. It also
 * configures SM options on the GPU in case FULLSM or PARTIALSM mode is going to
 * be used.
 */
void scratch_cuda_bootstrap_amortized_64(void *v_stream, uint32_t gpu_index,
                                         int8_t **pbs_buffer,
                                         uint32_t glwe_dimension,
                                         uint32_t polynomial_size,
                                         uint32_t input_lwe_ciphertext_count,
                                         uint32_t max_shared_memory,
                                         bool allocate_gpu_memory) {
  checks_fast_bootstrap_amortized(polynomial_size);

  switch (polynomial_size) {
  case 256:
    scratch_bootstrap_amortized<uint64_t, int64_t, AmortizedDegree<256>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 512:
    scratch_bootstrap_amortized<uint64_t, int64_t, AmortizedDegree<512>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 1024:
    scratch_bootstrap_amortized<uint64_t, int64_t, AmortizedDegree<1024>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 2048:
    scratch_bootstrap_amortized<uint64_t, int64_t, AmortizedDegree<2048>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 4096:
    scratch_bootstrap_amortized<uint64_t, int64_t, AmortizedDegree<4096>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 8192:
    scratch_bootstrap_amortized<uint64_t, int64_t, AmortizedDegree<8192>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  case 16384:
    scratch_bootstrap_amortized<uint64_t, int64_t, AmortizedDegree<16384>>(
        v_stream, gpu_index, pbs_buffer, glwe_dimension, polynomial_size,
        input_lwe_ciphertext_count, max_shared_memory, allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/* Perform the programmable bootstrapping on a batch of input u32 LWE
 * ciphertexts. See the corresponding operation on 64 bits for more details.
 */
void cuda_bootstrap_amortized_lwe_ciphertext_vector_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lut_vector,
    void *lut_vector_indexes, void *lwe_array_in, void *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t num_samples, uint32_t num_lut_vectors, uint32_t lwe_idx,
    uint32_t max_shared_memory) {

  checks_bootstrap_amortized(32, base_log, polynomial_size);

  switch (polynomial_size) {
  case 256:
    host_bootstrap_amortized<uint32_t, AmortizedDegree<256>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 512:
    host_bootstrap_amortized<uint32_t, AmortizedDegree<512>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 1024:
    host_bootstrap_amortized<uint32_t, AmortizedDegree<1024>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 2048:
    host_bootstrap_amortized<uint32_t, AmortizedDegree<2048>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 4096:
    host_bootstrap_amortized<uint32_t, AmortizedDegree<4096>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 8192:
    host_bootstrap_amortized<uint32_t, AmortizedDegree<8192>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 16384:
    host_bootstrap_amortized<uint32_t, AmortizedDegree<16384>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out, (uint32_t *)lut_vector,
        (uint32_t *)lut_vector_indexes, (uint32_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  default:
    break;
  }
}

/* Perform the programmable bootstrapping on a batch of input u64 LWE
 * ciphertexts. This functions performs best for large numbers of inputs (> 10).
 * - `v_stream` is a void pointer to the Cuda stream to be used in the kernel
 * launch
 * - `gpu_index` is the index of the GPU to be used in the kernel launch
 *  - lwe_array_out: output batch of num_samples bootstrapped ciphertexts c =
 * (a0,..an-1,b) where n is the LWE dimension
 *  - lut_vector: should hold as many test vectors of size polynomial_size
 * as there are input ciphertexts, but actually holds
 * num_lut_vectors vectors to reduce memory usage
 *  - lut_vector_indexes: stores the index corresponding to
 * which test vector of lut_vector to use for each LWE input in
 * lwe_array_in
 *  - lwe_array_in: input batch of num_samples LWE ciphertexts, containing n
 * mask values + 1 body value
 *  - bootstrapping_key: GGSW encryption of the LWE secret key sk1
 * under secret key sk2
 * bsk = Z + sk1 H
 * where H is the gadget matrix and Z is a matrix (k+1).l
 * containing GLWE encryptions of 0 under sk2.
 * bsk is thus a tensor of size (k+1)^2.l.N.n
 * where l is the number of decomposition levels and
 * k is the GLWE dimension, N is the polynomial size for
 * GLWE. The polynomial size for GLWE and the test vector
 * are the same because they have to be in the same ring
 * to be multiplied.
 * - input_lwe_dimension: size of the Torus vector used to encrypt the input
 * LWE ciphertexts - referred to as n above (~ 600)
 * - polynomial_size: size of the test polynomial (test vector) and size of the
 * GLWE polynomials (~1024) (where `size` refers to the polynomial degree + 1).
 * - base_log: log of the base used for the gadget matrix - B = 2^base_log (~8)
 * - level_count: number of decomposition levels in the gadget matrix (~4)
 * - num_samples: number of encrypted input messages
 * - num_lut_vectors: parameter to set the actual number of test vectors to be
 * used
 * - lwe_idx: the index of the LWE input to consider for the GPU of index
 * gpu_index. In case of multi-GPU computing, it is assumed that only a part of
 * the input LWE array is copied to each GPU, but the whole LUT array is copied
 * (because the case when the number of LUTs is smaller than the number of input
 * LWEs is not trivial to take into account in the data repartition on the
 * GPUs). `lwe_idx` is used to determine which LUT to consider for a given LWE
 * input in the LUT array `lut_vector`.
 *  - 'max_shared_memory' maximum amount of shared memory to be used inside
 * device functions
 *
 * This function calls a wrapper to a device kernel that performs the
 * bootstrapping:
 * 	- the kernel is templatized based on integer discretization and
 * polynomial degree
 * 	- num_samples blocks of threads are launched, where each thread is going
 * to handle one or more polynomial coefficients at each stage:
 * 		- perform the blind rotation
 * 		- round the result
 * 		- decompose into level_count levels, then for each level:
 * 		  - switch to the FFT domain
 * 		  - multiply with the bootstrapping key
 * 		  - come back to the coefficients representation
 * 	- between each stage a synchronization of the threads is necessary
 * 	- in case the device has enough shared memory, temporary arrays used for
 * the different stages (accumulators) are stored into the shared memory
 * 	- the accumulators serve to combine the results for all decomposition
 * levels
 * 	- the constant memory (64K) is used for storing the roots of identity
 * values for the FFT
 */
void cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lut_vector,
    void *lut_vector_indexes, void *lwe_array_in, void *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t num_samples, uint32_t num_lut_vectors, uint32_t lwe_idx,
    uint32_t max_shared_memory) {

  checks_bootstrap_amortized(64, base_log, polynomial_size);

  switch (polynomial_size) {
  case 256:
    host_bootstrap_amortized<uint64_t, AmortizedDegree<256>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out, (uint64_t *)lut_vector,
        (uint64_t *)lut_vector_indexes, (uint64_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 512:
    host_bootstrap_amortized<uint64_t, AmortizedDegree<512>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out, (uint64_t *)lut_vector,
        (uint64_t *)lut_vector_indexes, (uint64_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 1024:
    host_bootstrap_amortized<uint64_t, AmortizedDegree<1024>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out, (uint64_t *)lut_vector,
        (uint64_t *)lut_vector_indexes, (uint64_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 2048:
    host_bootstrap_amortized<uint64_t, AmortizedDegree<2048>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out, (uint64_t *)lut_vector,
        (uint64_t *)lut_vector_indexes, (uint64_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 4096:
    host_bootstrap_amortized<uint64_t, AmortizedDegree<4096>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out, (uint64_t *)lut_vector,
        (uint64_t *)lut_vector_indexes, (uint64_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 8192:
    host_bootstrap_amortized<uint64_t, AmortizedDegree<8192>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out, (uint64_t *)lut_vector,
        (uint64_t *)lut_vector_indexes, (uint64_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  case 16384:
    host_bootstrap_amortized<uint64_t, AmortizedDegree<16384>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out, (uint64_t *)lut_vector,
        (uint64_t *)lut_vector_indexes, (uint64_t *)lwe_array_in,
        (double2 *)bootstrapping_key, pbs_buffer, glwe_dimension, lwe_dimension,
        polynomial_size, base_log, level_count, num_samples, num_lut_vectors,
        lwe_idx, max_shared_memory);
    break;
  default:
    break;
  }
}

/*
 * This cleanup function frees the data for the amortized PBS on GPU in
 * pbs_buffer for 32 or 64 bits inputs.
 */
void cleanup_cuda_bootstrap_amortized(void *v_stream, uint32_t gpu_index,
                                      int8_t **pbs_buffer) {
  auto stream = static_cast<cudaStream_t *>(v_stream);
  // Free memory
  cuda_drop_async(*pbs_buffer, stream, gpu_index);
}
