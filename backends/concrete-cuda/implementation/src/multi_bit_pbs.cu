#include "fast_multi_bit_pbs.cuh"
#include "multi_bit_pbs.cuh"
#include "multi_bit_pbs.h"
#include "polynomial/parameters.cuh"

void checks_multi_bit_pbs(int polynomial_size) {
  assert(
      ("Error (GPU multi-bit PBS): polynomial size should be one of 256, 512, "
       "1024, 2048, 4096, 8192, 16384",
       polynomial_size == 256 || polynomial_size == 512 ||
           polynomial_size == 1024 || polynomial_size == 2048 ||
           polynomial_size == 4096 || polynomial_size == 8192 ||
           polynomial_size == 16384));
}

void cuda_multi_bit_pbs_lwe_ciphertext_vector_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lut_vector,
    void *lut_vector_indexes, void *lwe_array_in, void *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t grouping_factor, uint32_t base_log,
    uint32_t level_count, uint32_t num_samples, uint32_t num_lut_vectors,
    uint32_t lwe_idx, uint32_t max_shared_memory) {

  checks_multi_bit_pbs(polynomial_size);

  switch (polynomial_size) {
  case 256:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<256>>(
            glwe_dimension, level_count, num_samples, max_shared_memory)) {
      host_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<256>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    } else {
      host_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<256>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    }
    break;
  case 512:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<512>>(
            glwe_dimension, level_count, num_samples, max_shared_memory)) {
      host_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<512>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    } else {
      host_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<512>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    }
    break;
  case 1024:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<1024>>(
            glwe_dimension, level_count, num_samples, max_shared_memory)) {
      host_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<1024>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    } else {
      host_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<1024>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    }
    break;
  case 2048:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<2048>>(
            glwe_dimension, level_count, num_samples, max_shared_memory)) {
      host_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<2048>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    } else {
      host_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<2048>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    }
    break;
  case 4096:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<4096>>(
            glwe_dimension, level_count, num_samples, max_shared_memory)) {
      host_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<4096>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    } else {
      host_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<4096>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    }
    break;
  case 8192:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<8192>>(
            glwe_dimension, level_count, num_samples, max_shared_memory)) {
      host_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<8192>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    } else {
      host_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<8192>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    }
    break;
  case 16384:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<16384>>(
            glwe_dimension, level_count, num_samples, max_shared_memory)) {
      host_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<16384>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    } else {
      host_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<16384>>(
          v_stream, gpu_index, (uint64_t *)lwe_array_out,
          (uint64_t *)lut_vector, (uint64_t *)lut_vector_indexes,
          (uint64_t *)lwe_array_in, (uint64_t *)bootstrapping_key, pbs_buffer,
          glwe_dimension, lwe_dimension, polynomial_size, grouping_factor,
          base_log, level_count, num_samples, num_lut_vectors, lwe_idx,
          max_shared_memory);
    }
    break;
  default:
    break;
  }
}

void scratch_cuda_multi_bit_pbs_64(
    void *v_stream, uint32_t gpu_index, int8_t **pbs_buffer,
    uint32_t lwe_dimension, uint32_t glwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t grouping_factor,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {

  switch (polynomial_size) {
  case 256:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<256>>(
            glwe_dimension, level_count, input_lwe_ciphertext_count,
            max_shared_memory)) {
      scratch_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<256>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    } else {
      scratch_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<256>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    }
    break;
  case 512:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<512>>(
            glwe_dimension, level_count, input_lwe_ciphertext_count,
            max_shared_memory)) {
      scratch_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<512>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    } else {
      scratch_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<512>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    }
    break;
  case 1024:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<1024>>(
            glwe_dimension, level_count, input_lwe_ciphertext_count,
            max_shared_memory)) {
      scratch_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<1024>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    } else {
      scratch_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<1024>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    }
    break;
  case 2048:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<2048>>(
            glwe_dimension, level_count, input_lwe_ciphertext_count,
            max_shared_memory)) {
      scratch_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<2048>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    } else {
      scratch_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<2048>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    }
    break;
  case 4096:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<4096>>(
            glwe_dimension, level_count, input_lwe_ciphertext_count,
            max_shared_memory)) {
      scratch_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<4096>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    } else {
      scratch_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<4096>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    }
    break;
  case 8192:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<8192>>(
            glwe_dimension, level_count, input_lwe_ciphertext_count,
            max_shared_memory)) {
      scratch_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<8192>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    } else {
      scratch_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<8192>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    }
    break;
  case 16384:
    if (verify_cuda_bootstrap_fast_multi_bit_grid_size<uint64_t,
                                                       AmortizedDegree<16384>>(
            glwe_dimension, level_count, input_lwe_ciphertext_count,
            max_shared_memory)) {
      scratch_fast_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<16384>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    } else {
      scratch_multi_bit_pbs<uint64_t, int64_t, AmortizedDegree<16384>>(
          v_stream, gpu_index, pbs_buffer, lwe_dimension, glwe_dimension,
          polynomial_size, level_count, input_lwe_ciphertext_count,
          grouping_factor, max_shared_memory, allocate_gpu_memory);
    }
    break;
  default:
    break;
  }
}

void cleanup_cuda_multi_bit_pbs(void *v_stream, uint32_t gpu_index,
                                int8_t **pbs_buffer) {
  auto stream = static_cast<cudaStream_t *>(v_stream);
  // Free memory
  cuda_drop_async(*pbs_buffer, stream, gpu_index);
}

__host__ uint32_t get_lwe_chunk_size(uint32_t num_samples) {

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0); // Assuming device 0

  const char *v100Name = "V100"; // Known name of V100 GPU
  const char *a100Name = "A100"; // Known name of V100 GPU

  if (std::strstr(deviceProp.name, v100Name) != nullptr) {
    // Tesla V100
    switch (cuda_get_number_of_gpus()) {
    case 4:
      if (num_samples < 4)
        return 7;
      else if (num_samples < 8)
        return 4;
      else if (num_samples < 16)
        return 2;
      else if (num_samples < 4096)
        return 7;
      else if (num_samples < 8192)
        return 5;
      else if (num_samples < 16384)
        return 3;
      else
        return 1;
    case 8:
      if (num_samples < 4)
        return 7;
      else if (num_samples < 8)
        return 4;
      else if (num_samples < 16)
        return 35;
      else if (num_samples < 32)
        return 39;
      else if (num_samples < 512)
        return 16;
      else if (num_samples < 1024)
        return 14;
      else if (num_samples < 2048)
        return 15;
      else
        return 12;
    default:
      // 1 GPU
      if (num_samples < 4)
        return 7;
      else if (num_samples < 8)
        return 4;
      else if (num_samples < 16)
        return 2;
      else if (num_samples < 4096)
        return 7;
      else if (num_samples < 8192)
        return 5;
      else if (num_samples < 16384)
        return 3;
      else
        return 1;
    }
  } else if (std::strstr(deviceProp.name, a100Name) != nullptr) {
    // Tesla A100
    switch (cuda_get_number_of_gpus()) {
    case 8:
      if (num_samples < 8)
        return 19;
      else if (num_samples < 16)
        return 12;
      else if (num_samples < 64)
        return 19;
      else if (num_samples < 128)
        return 16;
      else if (num_samples < 512)
        return 19;
      else if (num_samples < 1024)
        return 18;
      else
        return 17;
    default:
      // 1 GPU
      if (num_samples < 4)
        return 11;
      else if (num_samples < 8)
        return 6;
      else if (num_samples < 16)
        return 13;
      else if (num_samples < 64)
        return 19;
      else if (num_samples < 128)
        return 1;
      else if (num_samples < 512)
        return 19;
      else if (num_samples < 1024)
        return 17;
      else if (num_samples < 8192)
        return 19;
      else if (num_samples < 16384)
        return 12;
      else
        return 9;
    }
  } else {
    if (num_samples < 4)
      return 7;
    else if (num_samples < 8)
      return 4;
    else if (num_samples < 16)
      return 2;
    else if (num_samples < 4096)
      return 7;
    else if (num_samples < 8192)
      return 5;
    else if (num_samples < 16384)
      return 3;
    else
      return 1;
  }
}

__host__ uint64_t get_max_buffer_size_multibit_bootstrap(
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t max_input_lwe_ciphertext_count) {

  uint64_t max_buffer_size = 0;
  for (uint32_t input_lwe_ciphertext_count = 1;
       input_lwe_ciphertext_count <= max_input_lwe_ciphertext_count;
       input_lwe_ciphertext_count++) {
    max_buffer_size = std::max(
        max_buffer_size, get_buffer_size_multibit_bootstrap<uint64_t>(
                             glwe_dimension, polynomial_size, level_count,
                             input_lwe_ciphertext_count,
                             get_lwe_chunk_size(input_lwe_ciphertext_count)));
  }

  return max_buffer_size;
}
