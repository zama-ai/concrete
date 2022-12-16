#include "wop_bootstrap.cuh"

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

void cuda_wop_pbs_64(void *v_stream, uint32_t gpu_index, void *lwe_array_out,
                     void *lwe_array_in, void *lut_vector, void *fourier_bsk,
                     void *ksk, void *cbs_fpksk, uint32_t glwe_dimension,
                     uint32_t lwe_dimension, uint32_t polynomial_size,
                     uint32_t base_log_bsk, uint32_t level_count_bsk,
                     uint32_t base_log_ksk, uint32_t level_count_ksk,
                     uint32_t base_log_pksk, uint32_t level_count_pksk,
                     uint32_t base_log_cbs, uint32_t level_count_cbs,
                     uint32_t number_of_bits_of_message_including_padding,
                     uint32_t number_of_bits_to_extract,
                     uint32_t number_of_inputs, uint32_t max_shared_memory) {
  assert(("Error (GPU WOP PBS): glwe_dimension should be equal to 1",
          glwe_dimension == 1));
  assert(("Error (GPU WOP PBS): polynomial_size should be one of "
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
  assert(("Error (GPU WOP PBS): the number of input LWEs must be lower or "
          "equal to the "
          "number of streaming multiprocessors on the device divided by 8 * "
          "level_count_bsk",
          number_of_inputs <= number_of_sm / 4. / 2. / level_count_bsk));
  switch (polynomial_size) {
  case 512:
    host_wop_pbs<uint64_t, int64_t, Degree<512>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)ksk, (uint64_t *)cbs_fpksk,
        glwe_dimension, lwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, base_log_pksk,
        level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_bits_of_message_including_padding, number_of_bits_to_extract,
        number_of_inputs, max_shared_memory);
    break;
  case 1024:
    host_wop_pbs<uint64_t, int64_t, Degree<1024>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)ksk, (uint64_t *)cbs_fpksk,
        glwe_dimension, lwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, base_log_pksk,
        level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_bits_of_message_including_padding, number_of_bits_to_extract,
        number_of_inputs, max_shared_memory);
    break;
  case 2048:
    host_wop_pbs<uint64_t, int64_t, Degree<2048>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)ksk, (uint64_t *)cbs_fpksk,
        glwe_dimension, lwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, base_log_pksk,
        level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_bits_of_message_including_padding, number_of_bits_to_extract,
        number_of_inputs, max_shared_memory);
    break;
  case 4096:
    host_wop_pbs<uint64_t, int64_t, Degree<4096>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)ksk, (uint64_t *)cbs_fpksk,
        glwe_dimension, lwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, base_log_pksk,
        level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_bits_of_message_including_padding, number_of_bits_to_extract,
        number_of_inputs, max_shared_memory);
    break;
  case 8192:
    host_wop_pbs<uint64_t, int64_t, Degree<8192>>(
        v_stream, gpu_index, (uint64_t *)lwe_array_out,
        (uint64_t *)lwe_array_in, (uint64_t *)lut_vector,
        (double2 *)fourier_bsk, (uint64_t *)ksk, (uint64_t *)cbs_fpksk,
        glwe_dimension, lwe_dimension, polynomial_size, base_log_bsk,
        level_count_bsk, base_log_ksk, level_count_ksk, base_log_pksk,
        level_count_pksk, base_log_cbs, level_count_cbs,
        number_of_bits_of_message_including_padding, number_of_bits_to_extract,
        number_of_inputs, max_shared_memory);
    break;
  default:
    break;
  }
}
