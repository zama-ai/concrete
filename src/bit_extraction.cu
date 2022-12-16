#include "bit_extraction.cuh"

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
  // The number of samples should be lower than 4 time the number of streaming
  // multiprocessors divided by ((k + 1) * l) (the factor 4 being related
  // to the occupancy of 50%). The only supported value for k is 1, so
  // k + 1 = 2 for now.
  int number_of_sm = 0;
  cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, 0);
  assert(("Error (GPU extract bits): the number of input LWEs must be lower or "
          "equal to the "
          "number of streaming multiprocessors on the device divided by 8 * "
          "level_count_bsk",
          number_of_samples <= number_of_sm * 4. / 2. / level_count_bsk));

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
          number_of_samples <= number_of_sm * 4. / 2. / level_count_bsk));

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
