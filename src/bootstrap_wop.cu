#include "bootstrap_wop.cuh"

void cuda_cmux_tree_32(void *v_stream, uint32_t gpu_index, void *glwe_array_out,
                       void *ggsw_in, void *lut_vector, uint32_t glwe_dimension,
                       uint32_t polynomial_size, uint32_t base_log,
                       uint32_t level_count, uint32_t r,
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
        level_count, r, max_shared_memory);
    break;
  case 1024:
    host_cmux_tree<uint32_t, int32_t, Degree<1024>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, max_shared_memory);
    break;
  case 2048:
    host_cmux_tree<uint32_t, int32_t, Degree<2048>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, max_shared_memory);
    break;
  case 4096:
    host_cmux_tree<uint32_t, int32_t, Degree<4096>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, max_shared_memory);
    break;
  case 8192:
    host_cmux_tree<uint32_t, int32_t, Degree<8192>>(
        v_stream, gpu_index, (uint32_t *)glwe_array_out, (uint32_t *)ggsw_in,
        (uint32_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, max_shared_memory);
    break;
  default:
    break;
  }
}

void cuda_cmux_tree_64(void *v_stream, uint32_t gpu_index, void *glwe_array_out,
                       void *ggsw_in, void *lut_vector, uint32_t glwe_dimension,
                       uint32_t polynomial_size, uint32_t base_log,
                       uint32_t level_count, uint32_t r,
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
        level_count, r, max_shared_memory);
    break;
  case 1024:
    host_cmux_tree<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, max_shared_memory);
    break;
  case 2048:
    host_cmux_tree<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, max_shared_memory);
    break;
  case 4096:
    host_cmux_tree<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, max_shared_memory);
    break;
  case 8192:
    host_cmux_tree<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)glwe_array_out, (uint64_t *)ggsw_in,
        (uint64_t *)lut_vector, glwe_dimension, polynomial_size, base_log,
        level_count, r, max_shared_memory);
    break;
  default:
    break;
  }
}

void cuda_extract_bits_32(
    void *v_stream, uint32_t gpu_index, void *list_lwe_array_out,
    void *lwe_array_in, void *lwe_array_in_buffer,
    void *lwe_array_in_shifted_buffer, void *lwe_array_out_ks_buffer,
    void *lwe_array_out_pbs_buffer, void *lut_pbs, void *lut_vector_indexes,
    void *ksk, void *fourier_bsk, uint32_t number_of_bits, uint32_t delta_log,
    uint32_t lwe_dimension_in, uint32_t lwe_dimension_out,
    uint32_t glwe_dimension, uint32_t base_log_bsk, uint32_t level_count_bsk,
    uint32_t base_log_ksk, uint32_t level_count_ksk, uint32_t number_of_samples,
    uint32_t max_shared_memory) {
  assert(("Error (GPU extract bits): base log should be <= 32",
          base_log_bsk <= 32));
  assert(("Error (GPU extract bits): glwe_dimension should be equal to 1",
          glwe_dimension == 1));
  assert(("Error (GPU extract bits): lwe_dimension_in should be one of "
          "512, 1024, 2048, 4096, 8192",
          lwe_dimension_in == 512 || lwe_dimension_in == 1024 ||
              lwe_dimension_in == 2048 || lwe_dimension_in == 4096 ||
              lwe_dimension_in == 8192));
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
          number_of_samples <= number_of_sm / 4. / 2. / level_count_bsk));

  switch (lwe_dimension_in) {
  case 512:
    host_extract_bits<uint32_t, Degree<512>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lwe_array_in_buffer,
        (uint32_t *)lwe_array_in_shifted_buffer,
        (uint32_t *)lwe_array_out_ks_buffer,
        (uint32_t *)lwe_array_out_pbs_buffer, (uint32_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint32_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
    break;
  case 1024:
    host_extract_bits<uint32_t, Degree<1024>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lwe_array_in_buffer,
        (uint32_t *)lwe_array_in_shifted_buffer,
        (uint32_t *)lwe_array_out_ks_buffer,
        (uint32_t *)lwe_array_out_pbs_buffer, (uint32_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint32_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
    break;
  case 2048:
    host_extract_bits<uint32_t, Degree<2048>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lwe_array_in_buffer,
        (uint32_t *)lwe_array_in_shifted_buffer,
        (uint32_t *)lwe_array_out_ks_buffer,
        (uint32_t *)lwe_array_out_pbs_buffer, (uint32_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint32_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
    break;
  case 4096:
    host_extract_bits<uint32_t, Degree<4096>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lwe_array_in_buffer,
        (uint32_t *)lwe_array_in_shifted_buffer,
        (uint32_t *)lwe_array_out_ks_buffer,
        (uint32_t *)lwe_array_out_pbs_buffer, (uint32_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint32_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
    break;
  case 8192:
    host_extract_bits<uint32_t, Degree<8192>>(
        v_stream, gpu_index, (uint32_t *)list_lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lwe_array_in_buffer,
        (uint32_t *)lwe_array_in_shifted_buffer,
        (uint32_t *)lwe_array_out_ks_buffer,
        (uint32_t *)lwe_array_out_pbs_buffer, (uint32_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint32_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
    break;
  default:
    break;
  }
}

void cuda_extract_bits_64(
    void *v_stream, uint32_t gpu_index, void *list_lwe_array_out,
    void *lwe_array_in, void *lwe_array_in_buffer,
    void *lwe_array_in_shifted_buffer, void *lwe_array_out_ks_buffer,
    void *lwe_array_out_pbs_buffer, void *lut_pbs, void *lut_vector_indexes,
    void *ksk, void *fourier_bsk, uint32_t number_of_bits, uint32_t delta_log,
    uint32_t lwe_dimension_in, uint32_t lwe_dimension_out,
    uint32_t glwe_dimension, uint32_t base_log_bsk, uint32_t level_count_bsk,
    uint32_t base_log_ksk, uint32_t level_count_ksk, uint32_t number_of_samples,
    uint32_t max_shared_memory) {
  assert(("Error (GPU extract bits): base log should be <= 64",
          base_log_bsk <= 64));
  assert(("Error (GPU extract bits): glwe_dimension should be equal to 1",
          glwe_dimension == 1));
  assert(("Error (GPU extract bits): lwe_dimension_in should be one of "
          "512, 1024, 2048, 4096, 8192",
          lwe_dimension_in == 512 || lwe_dimension_in == 1024 ||
              lwe_dimension_in == 2048 || lwe_dimension_in == 4096 ||
              lwe_dimension_in == 8192));
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
          number_of_samples <= number_of_sm / 4. / 2. / level_count_bsk));

  switch (lwe_dimension_in) {
  case 512:
    host_extract_bits<uint64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lwe_array_in_buffer,
        (uint64_t *)lwe_array_in_shifted_buffer,
        (uint64_t *)lwe_array_out_ks_buffer,
        (uint64_t *)lwe_array_out_pbs_buffer, (uint64_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint64_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
    break;
  case 1024:
    host_extract_bits<uint64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lwe_array_in_buffer,
        (uint64_t *)lwe_array_in_shifted_buffer,
        (uint64_t *)lwe_array_out_ks_buffer,
        (uint64_t *)lwe_array_out_pbs_buffer, (uint64_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint64_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
    break;
  case 2048:
    host_extract_bits<uint64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lwe_array_in_buffer,
        (uint64_t *)lwe_array_in_shifted_buffer,
        (uint64_t *)lwe_array_out_ks_buffer,
        (uint64_t *)lwe_array_out_pbs_buffer, (uint64_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint64_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
    break;
  case 4096:
    host_extract_bits<uint64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lwe_array_in_buffer,
        (uint64_t *)lwe_array_in_shifted_buffer,
        (uint64_t *)lwe_array_out_ks_buffer,
        (uint64_t *)lwe_array_out_pbs_buffer, (uint64_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint64_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
    break;
  case 8192:
    host_extract_bits<uint64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)list_lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lwe_array_in_buffer,
        (uint64_t *)lwe_array_in_shifted_buffer,
        (uint64_t *)lwe_array_out_ks_buffer,
        (uint64_t *)lwe_array_out_pbs_buffer, (uint64_t *)lut_pbs,
        (uint32_t *)lut_vector_indexes, (uint64_t *)ksk, (double2 *)fourier_bsk,
        number_of_bits, delta_log, lwe_dimension_in, lwe_dimension_out,
        base_log_bsk, level_count_bsk, base_log_ksk, level_count_ksk,
        number_of_samples, max_shared_memory);
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

void cuda_circuit_bootstrap_vertical_packing_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *fourier_bsk, void *cbs_fpksk, void *lut_vector,
    uint32_t polynomial_size, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t level_count_bsk, uint32_t base_log_bsk, uint32_t level_count_pksk,
    uint32_t base_log_pksk, uint32_t level_count_cbs, uint32_t base_log_cbs,
    uint32_t number_of_inputs, uint32_t lut_number,
    uint32_t max_shared_memory) {
  assert(("Error (GPU circuit bootstrap): glwe_dimension should be equal to 1",
          glwe_dimension == 1));
  assert(("Error (GPU circuit bootstrap): polynomial_size should be one of "
          "512, 1024, 2048, 4096, 8192",
          polynomial_size == 512 || polynomial_size == 1024 ||
              polynomial_size == 2048 || polynomial_size == 4096 ||
              polynomial_size == 8192));
  // The number of inputs should be lower than the number of streaming
  // multiprocessors divided by (4 * (k + 1) * l) (the factor 4 being related
  // to the occupancy of 50%). The only supported value for k is 1, so
  // k + 1 = 2 for now.
  int number_of_sm = 0;
  cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, 0);
  assert(("Error (GPU extract bits): the number of input LWEs must be lower or "
          "equal to the "
          "number of streaming multiprocessors on the device divided by 8 * "
          "level_count_bsk",
          number_of_inputs <= number_of_sm / 4. / 2. / level_count_bsk));
  switch (polynomial_size) {
  case 512:
    host_circuit_bootstrap_vertical_packing<uint32_t, int32_t, Degree<512>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lut_vector,
        (double2 *)fourier_bsk, (uint32_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  case 1024:
    host_circuit_bootstrap_vertical_packing<uint32_t, int32_t, Degree<1024>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lut_vector,
        (double2 *)fourier_bsk, (uint32_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  case 2048:
    host_circuit_bootstrap_vertical_packing<uint32_t, int32_t, Degree<2048>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lut_vector,
        (double2 *)fourier_bsk, (uint32_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  case 4096:
    host_circuit_bootstrap_vertical_packing<uint32_t, int32_t, Degree<4096>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lut_vector,
        (double2 *)fourier_bsk, (uint32_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  case 8192:
    host_circuit_bootstrap_vertical_packing<uint32_t, int32_t, Degree<8192>>(
        v_stream, gpu_index, (uint32_t *)lwe_array_out,
        (uint32_t *)lwe_array_in, (uint32_t *)lut_vector,
        (double2 *)fourier_bsk, (uint32_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  default:
    break;
  }
}

void cuda_circuit_bootstrap_vertical_packing_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *fourier_bsk, void *cbs_fpksk, void *lut_vector,
    uint32_t polynomial_size, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t level_count_bsk, uint32_t base_log_bsk, uint32_t level_count_pksk,
    uint32_t base_log_pksk, uint32_t level_count_cbs, uint32_t base_log_cbs,
    uint32_t number_of_inputs, uint32_t lut_number,
    uint32_t max_shared_memory) {
  assert(("Error (GPU circuit bootstrap): glwe_dimension should be equal to 1",
          glwe_dimension == 1));
  assert(("Error (GPU circuit bootstrap): polynomial_size should be one of "
          "512, 1024, 2048, 4096, 8192",
          polynomial_size == 512 || polynomial_size == 1024 ||
              polynomial_size == 2048 || polynomial_size == 4096 ||
              polynomial_size == 8192));
  // The number of inputs should be lower than the number of streaming
  // multiprocessors divided by (4 * (k + 1) * l) (the factor 4 being related
  // to the occupancy of 50%). The only supported value for k is 1, so
  // k + 1 = 2 for now.
  int number_of_sm = 0;
  cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, 0);
  assert(("Error (GPU extract bits): the number of input LWEs must be lower or "
          "equal to the "
          "number of streaming multiprocessors on the device divided by 8 * "
          "level_count_bsk",
          number_of_inputs <= number_of_sm / 4. / 2. / level_count_bsk));
  switch (polynomial_size) {
  case 512:
    host_circuit_bootstrap_vertical_packing<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  case 1024:
    host_circuit_bootstrap_vertical_packing<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  case 2048:
    host_circuit_bootstrap_vertical_packing<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  case 4096:
    host_circuit_bootstrap_vertical_packing<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  case 8192:
    host_circuit_bootstrap_vertical_packing<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)cbs_fpksk, glwe_dimension,
        lwe_dimension, polynomial_size, base_log_bsk, level_count_bsk,
        base_log_pksk, level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_inputs, lut_number, max_shared_memory);
    break;
  default:
    break;
  }
}
