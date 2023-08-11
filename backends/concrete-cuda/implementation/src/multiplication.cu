#include "multiplication.cuh"

/*
 * Perform the multiplication of a u32 input LWE ciphertext vector with a u32
 * cleartext vector. See the equivalent operation on u64 data for more details.
 */
void cuda_mult_lwe_ciphertext_vector_cleartext_vector_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *cleartext_array_in, uint32_t input_lwe_dimension,
    uint32_t input_lwe_ciphertext_count) {

  host_cleartext_multiplication(
      v_stream, gpu_index, static_cast<uint32_t *>(lwe_array_out),
      static_cast<uint32_t *>(lwe_array_in),
      static_cast<uint32_t *>(cleartext_array_in), input_lwe_dimension,
      input_lwe_ciphertext_count);
}
/*
 * Perform the multiplication of a u64 input LWE ciphertext vector with a u64
 * input cleartext vector.
 * - `v_stream` is a void pointer to the Cuda stream to be used in the kernel
 * launch
 * - `gpu_index` is the index of the GPU to be used in the kernel launch
 * - `lwe_array_out` is an array of size
 * `(input_lwe_dimension + 1) * input_lwe_ciphertext_count` that should have
 * been allocated on the GPU before calling this function, and that will hold
 * the result of the computation.
 * - `lwe_array_in` is the LWE ciphertext vector used as input, it should have
 * been allocated and initialized before calling this function. It has the same
 * size as the output array.
 * - `cleartext_array_in` is the cleartext vector used as input, it should have
 * been allocated and initialized before calling this function. It should be of
 * size `input_lwe_ciphertext_count`.
 * - `input_lwe_dimension` is the number of mask elements in the input and
 * output LWE ciphertext vectors
 * - `input_lwe_ciphertext_count` is the number of ciphertexts contained in the
 * input LWE ciphertext vector, as well as in the output. It is also the number
 * of cleartexts in the input cleartext vector.
 *
 * Each cleartext of the input cleartext vector is multiplied to the mask and
 * body of the corresponding LWE ciphertext in the LWE ciphertext vector. The
 * result of the operation is stored in the output LWE ciphertext vector. The
 * two input vectors are unchanged. This function is a wrapper to a device
 * function that performs the operation on the GPU.
 */
void cuda_mult_lwe_ciphertext_vector_cleartext_vector_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *cleartext_array_in, uint32_t input_lwe_dimension,
    uint32_t input_lwe_ciphertext_count) {

  host_cleartext_multiplication(
      v_stream, gpu_index, static_cast<uint64_t *>(lwe_array_out),
      static_cast<uint64_t *>(lwe_array_in),
      static_cast<uint64_t *>(cleartext_array_in), input_lwe_dimension,
      input_lwe_ciphertext_count);
}


/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the integer radix multiplication in keyswitch->bootstrap order.
 */
void scratch_cuda_integer_mult_radix_ciphertext_kb_64(
    void *v_stream, uint32_t gpu_index, void *mem_ptr, uint32_t message_modulus,
    uint32_t carry_modulus, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level,
    uint32_t ks_base_log, uint32_t ks_level, uint32_t num_blocks,
    PBS_TYPE pbs_type, uint32_t max_shared_memory, bool allocate_gpu_memory) {
  switch (polynomial_size) {
  case 2048:
    scratch_cuda_integer_mult_radix_ciphertext_kb<uint64_t, Degree<2048>>(
        v_stream, gpu_index, (int_mul_memory<uint64_t> *)mem_ptr,
        message_modulus, carry_modulus, glwe_dimension, lwe_dimension,
        polynomial_size, pbs_base_log, pbs_level, ks_base_log, ks_level,
        num_blocks, pbs_type, max_shared_memory, allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/*
 * Computes a multiplication between two 64 bit radix lwe ciphertexts
 * encrypting integer values. keyswitch -> bootstrap pattern is used, function
 * works for single pair of radix ciphertexts, 'v_stream' can be used for
 * parallelization
 * - 'v_stream' is a void pointer to the Cuda stream to be used in the kernel
 * launch
 * - 'gpu_index' is the index of the GPU to be used in the kernel launch
 * - 'radix_lwe_out' is 64 bit radix big lwe ciphertext, product of
 * multiplication
 * - 'radix_lwe_left' left radix big lwe ciphertext
 * - 'radix_lwe_right' right radix big lwe ciphertext
 * - 'ct_degree_out' degree for each lwe ciphertext block for out
 * RadixCiphertext
 * - 'ct_degree_left' degree for each lwe ciphertext block for left
 * RadixCiphertext
 * - 'ct_degree_right' degree for each lwe ciphertext block for right
 * RadixCiphertext
 * - 'bsk' bootstrapping key in fourier domain
 * - 'ksk' keyswitching key
 * - 'mem_ptr'
 * - 'message_modulus' message_modulus
 * - 'carry_modulus' carry_modulus
 * - 'glwe_dimension' glwe_dimension
 * - 'lwe_dimension' is the dimension of small lwe ciphertext
 * - 'polynomial_size' polynomial size
 * - 'pbs_base_log' base log used in the pbs
 * - 'pbs_level' decomposition level count used in the pbs
 * - 'ks_level' decomposition level count used in the keyswitch
 * - 'num_blocks' is the number of big lwe ciphertext blocks inside radix
 * ciphertext
 * - 'pbs_type' selects which PBS implementation should be used
 * - 'max_shared_memory' maximum shared memory per cuda block
 */
void cuda_integer_mult_radix_ciphertext_kb_64(
    void *v_stream, uint32_t gpu_index, void *radix_lwe_out,
    void *radix_lwe_left, void *radix_lwe_right, uint32_t *ct_degree_out,
    uint32_t *ct_degree_left, uint32_t *ct_degree_right, void *bsk, void *ksk,
    void *mem_ptr, uint32_t message_modulus, uint32_t carry_modulus,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t pbs_base_log, uint32_t pbs_level, uint32_t ks_base_log,
    uint32_t ks_level, uint32_t num_blocks, PBS_TYPE pbs_type,
    uint32_t max_shared_memory) {

  switch (polynomial_size) {
  case 2048:
    host_integer_mult_radix_kb<uint64_t, int64_t, AmortizedDegree<2048>>(
        v_stream, gpu_index, (uint64_t *)radix_lwe_out,
        (uint64_t *)radix_lwe_left, (uint64_t *)radix_lwe_right, ct_degree_out,
        ct_degree_left, ct_degree_right, bsk, (uint64_t *)ksk,
        (int_mul_memory<uint64_t> *)mem_ptr, message_modulus, carry_modulus,
        glwe_dimension, lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
        ks_base_log, ks_level, num_blocks, pbs_type, max_shared_memory);
    break;
  default:
    break;
  }
}

void scratch_cuda_integer_mult_radix_ciphertext_kb_64_multi_gpu(
    void *mem_ptr, void *bsk, void *ksk, uint32_t message_modulus,
    uint32_t carry_modulus, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level,
    uint32_t ks_base_log, uint32_t ks_level, uint32_t num_blocks,
    PBS_TYPE pbs_type, uint32_t max_shared_memory, bool allocate_gpu_memory) {
  switch (polynomial_size) {
  case 2048:
    scratch_cuda_integer_mult_radix_ciphertext_kb_multi_gpu<uint64_t,
                                                            Degree<2048>>(
        (int_mul_memory<uint64_t> *)mem_ptr, (uint64_t *)bsk, (uint64_t *)ksk,
        message_modulus, carry_modulus, glwe_dimension, lwe_dimension,
        polynomial_size, pbs_base_log, pbs_level, ks_base_log, ks_level,
        num_blocks, pbs_type, max_shared_memory, allocate_gpu_memory);
    break;
  default:
    break;
  }
}

void cuda_integer_mult_radix_ciphertext_kb_64_multi_gpu(
    void *radix_lwe_out, void *radix_lwe_left, void *radix_lwe_right,
    uint32_t *ct_degree_out, uint32_t *ct_degree_left,
    uint32_t *ct_degree_right, void *bsk, void *ksk, void *mem_ptr,
    uint32_t message_modulus, uint32_t carry_modulus, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t pbs_base_log,
    uint32_t pbs_level, uint32_t ks_base_log, uint32_t ks_level,
    uint32_t num_blocks, PBS_TYPE pbs_type, uint32_t max_shared_memory) {

  switch (polynomial_size) {
  case 2048:
    host_integer_mult_radix_kb_multi_gpu<uint64_t, int64_t, Degree<2048>>(
        (uint64_t *)radix_lwe_out, (uint64_t *)radix_lwe_left,
        (uint64_t *)radix_lwe_right, ct_degree_out, ct_degree_left,
        ct_degree_right, (uint64_t *)bsk, (uint64_t *)ksk,
        (int_mul_memory<uint64_t> *)mem_ptr, message_modulus, carry_modulus,
        glwe_dimension, lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
        ks_base_log, ks_level, num_blocks, max_shared_memory);
    break;
  default:
    break;
  }
}
