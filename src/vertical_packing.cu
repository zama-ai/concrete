#include "vertical_packing.cuh"

void cuda_cmux_tree_32(void *v_stream, uint32_t gpu_index, void *glwe_array_out,
                       void *ggsw_in, void *lut_vector, uint32_t glwe_dimension,
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
  assert(("Error (GPU Cmux tree): glwe_dimension should be equal to 1",
          glwe_dimension == 1));
  assert(("Error (GPU Cmux tree): r, the number of layers in the tree, should "
          "be >= 1 ",
          r >= 1));

  switch (polynomial_size) {
  case 512:
    host_cmux_tree<uint32_t, int32_t, Degree<512>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  case 1024:
    host_cmux_tree<uint32_t, int32_t, Degree<1024>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  case 2048:
    host_cmux_tree<uint32_t, int32_t, Degree<2048>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  case 4096:
    host_cmux_tree<uint32_t, int32_t, Degree<4096>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  case 8192:
    host_cmux_tree<uint32_t, int32_t, Degree<8192>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  default:
    break;
  }
}

void cuda_cmux_tree_64(void *v_stream, uint32_t gpu_index, void *glwe_array_out,
                       void *ggsw_in, void *lut_vector, uint32_t glwe_dimension,
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
  assert(("Error (GPU Cmux tree): glwe_dimension should be equal to 1",
          glwe_dimension == 1));
  assert(("Error (GPU Cmux tree): r, the number of layers in the tree, should "
          "be >= 1 ",
          r >= 1));

  switch (polynomial_size) {
  case 512:
    host_cmux_tree<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  case 1024:
    host_cmux_tree<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  case 2048:
    host_cmux_tree<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  case 4096:
    host_cmux_tree<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  case 8192:
    host_cmux_tree<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, tau, max_shared_memory);
    break;
  default:
    break;
  }
}

void cuda_blind_rotate_and_sample_extraction_64(
    void *v_stream, uint32_t gpu_index, void *lwe_out, void *ggsw_in,
    void *lut_vector, uint32_t mbr_size, uint32_t tau, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t l_gadget,
    uint32_t max_shared_memory) {

  switch (polynomial_size) {
  case 512:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, mbr_size, tau, glwe_dimension, polynomial_size,
        base_log, l_gadget, max_shared_memory);
    break;
  case 1024:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, mbr_size, tau, glwe_dimension, polynomial_size,
        base_log, l_gadget, max_shared_memory);
    break;
  case 2048:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, mbr_size, tau, glwe_dimension, polynomial_size,
        base_log, l_gadget, max_shared_memory);
    break;
  case 4096:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, mbr_size, tau, glwe_dimension, polynomial_size,
        base_log, l_gadget, max_shared_memory);
    break;
  case 8192:
    host_blind_rotate_and_sample_extraction<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)lwe_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, mbr_size, tau, glwe_dimension, polynomial_size,
        base_log, l_gadget, max_shared_memory);
    break;
  }
}
