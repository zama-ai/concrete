#include "vertical_packing.cuh"
#include "vertical_packing.h"
#include <cassert>

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the Cmux tree on 32 bits inputs, into `cmux_tree_buffer`. It also configures
 * SM options on the GPU in case FULLSM mode is going to be used.
 */
void scratch_cuda_cmux_tree_32(void *v_stream, uint32_t gpu_index,
                               int8_t **cmux_tree_buffer,
                               uint32_t glwe_dimension,
                               uint32_t polynomial_size, uint32_t level_count,
                               uint32_t r, uint32_t tau,
                               uint32_t max_shared_memory,
                               bool allocate_gpu_memory) {

  switch (polynomial_size) {
  case 512:
    scratch_cmux_tree<uint32_t, int32_t, Degree<512>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 1024:
    scratch_cmux_tree<uint32_t, int32_t, Degree<1024>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 2048:
    scratch_cmux_tree<uint32_t, int32_t, Degree<2048>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 4096:
    scratch_cmux_tree<uint32_t, int32_t, Degree<4096>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 8192:
    scratch_cmux_tree<uint32_t, int32_t, Degree<8192>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the Cmux tree on 64 bits inputs, into `cmux_tree_buffer`. It also configures
 * SM options on the GPU in case FULLSM mode is going to be used.
 */
void scratch_cuda_cmux_tree_64(void *v_stream, uint32_t gpu_index,
                               int8_t **cmux_tree_buffer,
                               uint32_t glwe_dimension,
                               uint32_t polynomial_size, uint32_t level_count,
                               uint32_t r, uint32_t tau,
                               uint32_t max_shared_memory,
                               bool allocate_gpu_memory) {
  switch (polynomial_size) {
  case 512:
    scratch_cmux_tree<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 1024:
    scratch_cmux_tree<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 2048:
    scratch_cmux_tree<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 4096:
    scratch_cmux_tree<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 8192:
    scratch_cmux_tree<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, cmux_tree_buffer, glwe_dimension, polynomial_size,
        level_count, r, tau, max_shared_memory, allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/*
 * Perform cmux tree on a batch of 32-bit input GGSW ciphertexts.
 * Check the equivalent function for 64-bit inputs for more details.
 */
void cuda_cmux_tree_32(void *v_stream, uint32_t gpu_index, void *glwe_array_out,
                       void *ggsw_in, void *lut_vector,
                       int8_t *cmux_tree_buffer, uint32_t glwe_dimension,
                       uint32_t polynomial_size, uint32_t base_log,
                       uint32_t level_count, uint32_t r, uint32_t tau,
                       uint32_t max_shared_memory) {

  assert(("Error (GPU Cmux tree): base log should be <= 32", base_log <= 32));
  assert(("Error (GPU Cmux tree): polynomial size should be one of 512, 1024, "
          "2048, 4096, 8192",
          polynomial_size == 512 || polynomial_size == 1024 ||
              polynomial_size == 2048 || polynomial_size == 4096 ||
              polynomial_size == 8192));
  // For larger k we will need to adjust the mask size
  assert(("Error (GPU Cmux tree): r, the number of layers in the tree, should "
          "be >= 1 ",
          r >= 1));

  switch (polynomial_size) {
  case 512:
    host_cmux_tree<uint32_t, int32_t, Degree<512>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  case 1024:
    host_cmux_tree<uint32_t, int32_t, Degree<1024>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  case 2048:
    host_cmux_tree<uint32_t, int32_t, Degree<2048>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  case 4096:
    host_cmux_tree<uint32_t, int32_t, Degree<4096>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  case 8192:
    host_cmux_tree<uint32_t, int32_t, Degree<8192>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  default:
    break;
  }
}

/*
 * Perform Cmux tree on a batch of 64-bit input GGSW ciphertexts
 * - `v_stream` is a void pointer to the Cuda stream to be used in the kernel
 * launch
 * - `gpu_index` is the index of the GPU to be used in the kernel launch
 *  - 'glwe_array_out' output batch of GLWE buffer for Cmux tree, 'tau' GLWE's
 * will be the output of the function
 *  - 'ggsw_in' batch of input GGSW ciphertexts, function expects 'r' GGSW
 * ciphertexts as input.
 *  - 'lut_vector' batch of test vectors (LUTs) there should be 2^r LUTs
 * inside 'lut_vector' parameter
 *  - 'glwe_dimension' GLWE dimension, supported values: {1}
 *  - 'polynomial_size' size of the test polynomial, supported values: {512,
 * 1024, 2048, 4096, 8192}
 *  - 'base_log' base log parameter for cmux block
 *  - 'level_count' decomposition level for cmux block
 *  - 'r' number of input GGSW ciphertexts
 *  - 'tau' number of input LWE ciphertext which were used to generate GGSW
 * ciphertexts stored in 'ggsw_in', it is also an amount of output GLWE
 * ciphertexts
 *  - 'max_shared_memory' maximum shared memory amount to be used for cmux
 *  kernel
 *
 * This function calls a wrapper to a device kernel that performs the
 * Cmux tree. The kernel is templatized based on integer discretization and
 * polynomial degree.
 */
void cuda_cmux_tree_64(void *v_stream, uint32_t gpu_index, void *glwe_array_out,
                       void *ggsw_in, void *lut_vector,
                       int8_t *cmux_tree_buffer, uint32_t glwe_dimension,
                       uint32_t polynomial_size, uint32_t base_log,
                       uint32_t level_count, uint32_t r, uint32_t tau,
                       uint32_t max_shared_memory) {

  assert(("Error (GPU Cmux tree): base log should be <= 64", base_log <= 64));
  assert(("Error (GPU Cmux tree): polynomial size should be one of 512, 1024, "
          "2048, 4096, 8192",
          polynomial_size == 512 || polynomial_size == 1024 ||
              polynomial_size == 2048 || polynomial_size == 4096 ||
              polynomial_size == 8192));
  // For larger k we will need to adjust the mask size
  assert(("Error (GPU Cmux tree): r, the number of layers in the tree, should "
          "be >= 1 ",
          r >= 1));

  switch (polynomial_size) {
  case 512:
    host_cmux_tree<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  case 1024:
    host_cmux_tree<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  case 2048:
    host_cmux_tree<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  case 4096:
    host_cmux_tree<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  case 8192:
    host_cmux_tree<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, cmux_tree_buffer, glwe_dimension,
        polynomial_size, base_log, level_count, r, tau, max_shared_memory);
    break;
  default:
    break;
  }
}

/*
 * This cleanup function frees the data for the Cmux tree on GPU in
 * cmux_tree_buffer for 32 or 64 bits inputs.
 */
void cleanup_cuda_cmux_tree(void *v_stream, uint32_t gpu_index,
                            int8_t **cmux_tree_buffer) {
  auto stream = static_cast<cudaStream_t *>(v_stream);
  // Free memory
  cuda_drop_async(*cmux_tree_buffer, stream, gpu_index);
}

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the Cmux tree on 32 bits inputs, into `br_se_buffer`. It also configures
 * SM options on the GPU in case FULLSM mode is going to be used.
 */
void scratch_cuda_blind_rotation_sample_extraction_32(
    void *v_stream, uint32_t gpu_index, int8_t **br_se_buffer,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t mbr_size, uint32_t tau, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {

  switch (polynomial_size) {
  case 512:
    scratch_blind_rotation_sample_extraction<uint32_t, int32_t, Degree<512>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 1024:
    scratch_blind_rotation_sample_extraction<uint32_t, int32_t, Degree<1024>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 2048:
    scratch_blind_rotation_sample_extraction<uint32_t, int32_t, Degree<2048>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 4096:
    scratch_blind_rotation_sample_extraction<uint32_t, int32_t, Degree<4096>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 8192:
    scratch_blind_rotation_sample_extraction<uint32_t, int32_t, Degree<8192>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/*
 * This scratch function allocates the necessary amount of data on the GPU for
 * the Cmux tree on 64 bits inputs, into `br_se_buffer`. It also configures
 * SM options on the GPU in case FULLSM mode is going to be used.
 */
void scratch_cuda_blind_rotation_sample_extraction_64(
    void *v_stream, uint32_t gpu_index, int8_t **br_se_buffer,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t mbr_size, uint32_t tau, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {

  switch (polynomial_size) {
  case 512:
    scratch_blind_rotation_sample_extraction<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 1024:
    scratch_blind_rotation_sample_extraction<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 2048:
    scratch_blind_rotation_sample_extraction<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 4096:
    scratch_blind_rotation_sample_extraction<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  case 8192:
    scratch_blind_rotation_sample_extraction<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, br_se_buffer, glwe_dimension, polynomial_size,
        level_count, mbr_size, tau, max_shared_memory, allocate_gpu_memory);
    break;
  default:
    break;
  }
}

/*
 * Performs blind rotation on batch of 64-bit input ggsw ciphertexts
 * - `v_stream` is a void pointer to the Cuda stream to be used in the kernel
 * launch
 * - `gpu_index` is the index of the GPU to be used in the kernel launch
 *  - 'lwe_out'  batch of output lwe ciphertexts, there should be 'tau'
 * ciphertexts inside 'lwe_out'
 *  - 'ggsw_in' batch of input ggsw ciphertexts, function expects 'mbr_size'
 * ggsw ciphertexts inside 'ggsw_in'
 *  - 'lut_vector' list of test vectors, function expects 'tau' test vectors
 * inside 'lut_vector' parameter
 *  - 'glwe_dimension' glwe dimension, supported values : {1}
 *  - 'polynomial_size' size of test polynomial supported sizes: {512, 1024,
 * 2048, 4096, 8192}
 *  - 'base_log' base log parameter
 *  - 'l_gadget' decomposition level
 *  - 'max_shared_memory' maximum number of shared memory to be used in
 * device functions(kernels)
 *
 * This function calls a wrapper to a device kernel that performs the
 * blind rotation and sample extraction. The kernel is templatized based on
 * integer discretization and polynomial degree.
 */
void cuda_blind_rotate_and_sample_extraction_64(
    void *v_stream, uint32_t gpu_index, void *lwe_out, void *ggsw_in,
    void *lut_vector, int8_t *br_se_buffer, uint32_t mbr_size, uint32_t tau,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t base_log,
    uint32_t l_gadget, uint32_t max_shared_memory) {

  switch (polynomial_size) {
  case 512:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, br_se_buffer, mbr_size, tau, glwe_dimension,
        polynomial_size, base_log, l_gadget, max_shared_memory);
    break;
  case 1024:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, br_se_buffer, mbr_size, tau, glwe_dimension,
        polynomial_size, base_log, l_gadget, max_shared_memory);
    break;
  case 2048:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, br_se_buffer, mbr_size, tau, glwe_dimension,
        polynomial_size, base_log, l_gadget, max_shared_memory);
    break;
  case 4096:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, br_se_buffer, mbr_size, tau, glwe_dimension,
        polynomial_size, base_log, l_gadget, max_shared_memory);
    break;
  case 8192:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, br_se_buffer, mbr_size, tau, glwe_dimension,
        polynomial_size, base_log, l_gadget, max_shared_memory);
    break;
  }
}

/*
 * This cleanup function frees the data for the blind rotation and sample
 * extraction on GPU in br_se_buffer for 32 or 64 bits inputs.
 */
void cleanup_cuda_blind_rotation_sample_extraction(void *v_stream,
                                                   uint32_t gpu_index,
                                                   int8_t **br_se_buffer) {
  auto stream = static_cast<cudaStream_t *>(v_stream);
  // Free memory
  cuda_drop_async(*br_se_buffer, stream, gpu_index);
}
