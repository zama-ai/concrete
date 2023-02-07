#ifndef CUDA_BOOLEAN_GATES_CU
#define CUDA_BOOLEAN_GATES_CU

#include "bootstrap.h"
#include "device.h"
#include "keyswitch.h"
#include "linear_algebra.h"

constexpr int PLAINTEXT_TRUE{1 << (32 - 3)};
constexpr int PLAINTEXT_FALSE{7 << (32 - 3)};

extern "C" void cuda_boolean_not_32(void *v_stream, uint32_t gpu_index,
                                    void *lwe_array_out, void *lwe_array_in,
                                    uint32_t input_lwe_dimension,
                                    uint32_t input_lwe_ciphertext_count) {

  cuda_negate_lwe_ciphertext_vector_32(v_stream, gpu_index, lwe_array_out,
                                       lwe_array_in, input_lwe_dimension,
                                       input_lwe_ciphertext_count);
}

extern "C" void cuda_boolean_and_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out,
    void *lwe_array_in_1, void *lwe_array_in_2, void *bootstrapping_key,
    void *ksk, uint32_t input_lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level_count,
    uint32_t ks_base_log, uint32_t ks_level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  uint32_t *lwe_buffer_1 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  // 1. Add the two ciphertexts
  cuda_add_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_buffer_1, lwe_array_in_1, lwe_array_in_2,
      input_lwe_dimension, input_lwe_ciphertext_count);
  // 2. Add "false" plaintext, where "false" is 7 << (32 - 3)
  uint32_t *h_false_plaintext_array =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_false_plaintext_array[index] = PLAINTEXT_FALSE;
  }
  uint32_t *false_plaintext_array = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(false_plaintext_array, h_false_plaintext_array,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  uint32_t *lwe_buffer_2 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_add_lwe_ciphertext_vector_plaintext_vector_32(
      v_stream, gpu_index, lwe_buffer_2, lwe_buffer_1, false_plaintext_array,
      input_lwe_dimension, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_buffer_1, stream, gpu_index);
  cuda_drop_async(false_plaintext_array, stream, gpu_index);
  free(h_false_plaintext_array);

  // 3. Compute a PBS with the LUT created below
  uint32_t *h_pbs_lut = (uint32_t *)malloc((glwe_dimension + 1) *
                                           polynomial_size * sizeof(uint32_t));
  for (uint index = 0; index < (glwe_dimension + 1) * polynomial_size;
       index++) {
    h_pbs_lut[index] =
        index < (glwe_dimension * polynomial_size) ? 0 : PLAINTEXT_TRUE;
  }
  uint32_t *pbs_lut = (uint32_t *)cuda_malloc_async(
      (glwe_dimension + 1) * polynomial_size * sizeof(uint32_t), stream,
      gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut, h_pbs_lut,
                           (glwe_dimension + 1) * polynomial_size *
                               sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *h_pbs_lut_indexes =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_pbs_lut_indexes[index] = 0;
  }
  uint32_t *pbs_lut_indexes = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut_indexes, h_pbs_lut_indexes,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *lwe_pbs_buffer = (uint32_t *)cuda_malloc_async(
      (glwe_dimension * polynomial_size + 1) * input_lwe_ciphertext_count *
          sizeof(uint32_t),
      stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  cuda_bootstrap_amortized_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_pbs_buffer, pbs_lut, pbs_lut_indexes,
      lwe_buffer_2, bootstrapping_key, input_lwe_dimension, glwe_dimension,
      polynomial_size, pbs_base_log, pbs_level_count,
      input_lwe_ciphertext_count, 1, 0, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  cuda_drop_async(lwe_buffer_2, stream, gpu_index);
  cuda_drop_async(pbs_lut, stream, gpu_index);
  cuda_drop_async(pbs_lut_indexes, stream, gpu_index);
  free(h_pbs_lut);
  free(h_pbs_lut_indexes);

  cuda_keyswitch_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_array_out, lwe_pbs_buffer, ksk,
      glwe_dimension * polynomial_size, input_lwe_dimension, ks_base_log,
      ks_level_count, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_pbs_buffer, stream, gpu_index);
}

extern "C" void cuda_boolean_nand_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out,
    void *lwe_array_in_1, void *lwe_array_in_2, void *bootstrapping_key,
    void *ksk, uint32_t input_lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level_count,
    uint32_t ks_base_log, uint32_t ks_level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  uint32_t *lwe_buffer_1 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  // 1. Add the two ciphertexts
  cuda_add_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_buffer_1, lwe_array_in_1, lwe_array_in_2,
      input_lwe_dimension, input_lwe_ciphertext_count);
  // 2. Negate ciphertext
  uint32_t *lwe_buffer_2 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_negate_lwe_ciphertext_vector_32(v_stream, gpu_index, lwe_buffer_2,
                                       lwe_buffer_1, input_lwe_dimension,
                                       input_lwe_ciphertext_count);
  cuda_drop_async(lwe_buffer_1, stream, gpu_index);
  // 3. Add "true" plaintext, where "true" is 1 << (32 - 3)
  uint32_t *h_true_plaintext_array =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_true_plaintext_array[index] = PLAINTEXT_TRUE;
  }
  uint32_t *true_plaintext_array = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(true_plaintext_array, h_true_plaintext_array,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  uint32_t *lwe_buffer_3 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_add_lwe_ciphertext_vector_plaintext_vector_32(
      v_stream, gpu_index, lwe_buffer_3, lwe_buffer_2, true_plaintext_array,
      input_lwe_dimension, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_buffer_2, stream, gpu_index);
  cuda_drop_async(true_plaintext_array, stream, gpu_index);
  free(h_true_plaintext_array);

  // 3. Compute a PBS with the LUT created below
  uint32_t *h_pbs_lut = (uint32_t *)malloc((glwe_dimension + 1) *
                                           polynomial_size * sizeof(uint32_t));
  for (uint index = 0; index < (glwe_dimension + 1) * polynomial_size;
       index++) {
    h_pbs_lut[index] =
        index < (glwe_dimension * polynomial_size) ? 0 : PLAINTEXT_TRUE;
  }
  uint32_t *pbs_lut = (uint32_t *)cuda_malloc_async(
      (glwe_dimension + 1) * polynomial_size * sizeof(uint32_t), stream,
      gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut, h_pbs_lut,
                           (glwe_dimension + 1) * polynomial_size *
                               sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *h_pbs_lut_indexes =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_pbs_lut_indexes[index] = 0;
  }
  uint32_t *pbs_lut_indexes = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut_indexes, h_pbs_lut_indexes,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *lwe_pbs_buffer = (uint32_t *)cuda_malloc_async(
      (glwe_dimension * polynomial_size + 1) * input_lwe_ciphertext_count *
          sizeof(uint32_t),
      stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  cuda_bootstrap_amortized_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_pbs_buffer, pbs_lut, pbs_lut_indexes,
      lwe_buffer_3, bootstrapping_key, input_lwe_dimension, glwe_dimension,
      polynomial_size, pbs_base_log, pbs_level_count,
      input_lwe_ciphertext_count, 1, 0, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  cuda_drop_async(lwe_buffer_3, stream, gpu_index);
  cuda_drop_async(pbs_lut, stream, gpu_index);
  cuda_drop_async(pbs_lut_indexes, stream, gpu_index);
  free(h_pbs_lut);
  free(h_pbs_lut_indexes);

  cuda_keyswitch_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_array_out, lwe_pbs_buffer, ksk,
      glwe_dimension * polynomial_size, input_lwe_dimension, ks_base_log,
      ks_level_count, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_pbs_buffer, stream, gpu_index);
}

extern "C" void cuda_boolean_nor_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out,
    void *lwe_array_in_1, void *lwe_array_in_2, void *bootstrapping_key,
    void *ksk, uint32_t input_lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level_count,
    uint32_t ks_base_log, uint32_t ks_level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  uint32_t *lwe_buffer_1 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  // 1. Add the two ciphertexts
  cuda_add_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_buffer_1, lwe_array_in_1, lwe_array_in_2,
      input_lwe_dimension, input_lwe_ciphertext_count);
  // 2. Negate ciphertext
  uint32_t *lwe_buffer_2 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_negate_lwe_ciphertext_vector_32(v_stream, gpu_index, lwe_buffer_2,
                                       lwe_buffer_1, input_lwe_dimension,
                                       input_lwe_ciphertext_count);
  cuda_drop_async(lwe_buffer_1, stream, gpu_index);
  // 3. Add "false" plaintext
  uint32_t *h_false_plaintext_array =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_false_plaintext_array[index] = PLAINTEXT_FALSE;
  }
  uint32_t *false_plaintext_array = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(false_plaintext_array, h_false_plaintext_array,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  uint32_t *lwe_buffer_3 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_add_lwe_ciphertext_vector_plaintext_vector_32(
      v_stream, gpu_index, lwe_buffer_3, lwe_buffer_2, false_plaintext_array,
      input_lwe_dimension, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_buffer_2, stream, gpu_index);
  cuda_drop_async(false_plaintext_array, stream, gpu_index);
  free(h_false_plaintext_array);

  // 3. Compute a PBS with the LUT created below
  uint32_t *h_pbs_lut = (uint32_t *)malloc((glwe_dimension + 1) *
                                           polynomial_size * sizeof(uint32_t));
  for (uint index = 0; index < (glwe_dimension + 1) * polynomial_size;
       index++) {
    h_pbs_lut[index] =
        index < (glwe_dimension * polynomial_size) ? 0 : PLAINTEXT_TRUE;
  }
  uint32_t *pbs_lut = (uint32_t *)cuda_malloc_async(
      (glwe_dimension + 1) * polynomial_size * sizeof(uint32_t), stream,
      gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut, h_pbs_lut,
                           (glwe_dimension + 1) * polynomial_size *
                               sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *h_pbs_lut_indexes =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_pbs_lut_indexes[index] = 0;
  }
  uint32_t *pbs_lut_indexes = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut_indexes, h_pbs_lut_indexes,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *lwe_pbs_buffer = (uint32_t *)cuda_malloc_async(
      (glwe_dimension * polynomial_size + 1) * input_lwe_ciphertext_count *
          sizeof(uint32_t),
      stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  cuda_bootstrap_amortized_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_pbs_buffer, pbs_lut, pbs_lut_indexes,
      lwe_buffer_3, bootstrapping_key, input_lwe_dimension, glwe_dimension,
      polynomial_size, pbs_base_log, pbs_level_count,
      input_lwe_ciphertext_count, 1, 0, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  cuda_drop_async(lwe_buffer_3, stream, gpu_index);
  cuda_drop_async(pbs_lut, stream, gpu_index);
  cuda_drop_async(pbs_lut_indexes, stream, gpu_index);
  free(h_pbs_lut);
  free(h_pbs_lut_indexes);

  cuda_keyswitch_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_array_out, lwe_pbs_buffer, ksk,
      glwe_dimension * polynomial_size, input_lwe_dimension, ks_base_log,
      ks_level_count, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_pbs_buffer, stream, gpu_index);
}

extern "C" void cuda_boolean_or_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out,
    void *lwe_array_in_1, void *lwe_array_in_2, void *bootstrapping_key,
    void *ksk, uint32_t input_lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level_count,
    uint32_t ks_base_log, uint32_t ks_level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  uint32_t *lwe_buffer_1 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  // 1. Add the two ciphertexts
  cuda_add_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_buffer_1, lwe_array_in_1, lwe_array_in_2,
      input_lwe_dimension, input_lwe_ciphertext_count);
  // 2. Add "true" plaintext
  uint32_t *h_true_plaintext_array =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_true_plaintext_array[index] = PLAINTEXT_TRUE;
  }
  uint32_t *true_plaintext_array = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(true_plaintext_array, h_true_plaintext_array,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  uint32_t *lwe_buffer_2 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_add_lwe_ciphertext_vector_plaintext_vector_32(
      v_stream, gpu_index, lwe_buffer_2, lwe_buffer_1, true_plaintext_array,
      input_lwe_dimension, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_buffer_1, stream, gpu_index);
  cuda_drop_async(true_plaintext_array, stream, gpu_index);
  free(h_true_plaintext_array);

  // 3. Compute a PBS with the LUT created below
  uint32_t *h_pbs_lut = (uint32_t *)malloc((glwe_dimension + 1) *
                                           polynomial_size * sizeof(uint32_t));
  for (uint index = 0; index < (glwe_dimension + 1) * polynomial_size;
       index++) {
    h_pbs_lut[index] =
        index < (glwe_dimension * polynomial_size) ? 0 : PLAINTEXT_TRUE;
  }
  uint32_t *pbs_lut = (uint32_t *)cuda_malloc_async(
      (glwe_dimension + 1) * polynomial_size * sizeof(uint32_t), stream,
      gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut, h_pbs_lut,
                           (glwe_dimension + 1) * polynomial_size *
                               sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *h_pbs_lut_indexes =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_pbs_lut_indexes[index] = 0;
  }
  uint32_t *pbs_lut_indexes = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut_indexes, h_pbs_lut_indexes,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *lwe_pbs_buffer = (uint32_t *)cuda_malloc_async(
      (glwe_dimension * polynomial_size + 1) * input_lwe_ciphertext_count *
          sizeof(uint32_t),
      stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  cuda_bootstrap_amortized_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_pbs_buffer, pbs_lut, pbs_lut_indexes,
      lwe_buffer_2, bootstrapping_key, input_lwe_dimension, glwe_dimension,
      polynomial_size, pbs_base_log, pbs_level_count,
      input_lwe_ciphertext_count, 1, 0, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  cuda_drop_async(lwe_buffer_2, stream, gpu_index);
  cuda_drop_async(pbs_lut, stream, gpu_index);
  cuda_drop_async(pbs_lut_indexes, stream, gpu_index);
  free(h_pbs_lut);
  free(h_pbs_lut_indexes);

  cuda_keyswitch_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_array_out, lwe_pbs_buffer, ksk,
      glwe_dimension * polynomial_size, input_lwe_dimension, ks_base_log,
      ks_level_count, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_pbs_buffer, stream, gpu_index);
}

extern "C" void cuda_boolean_xor_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out,
    void *lwe_array_in_1, void *lwe_array_in_2, void *bootstrapping_key,
    void *ksk, uint32_t input_lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level_count,
    uint32_t ks_base_log, uint32_t ks_level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  uint32_t *lwe_buffer_1 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  // 1. Add the two ciphertexts
  cuda_add_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_buffer_1, lwe_array_in_1, lwe_array_in_2,
      input_lwe_dimension, input_lwe_ciphertext_count);
  // 2. Add "true" plaintext
  uint32_t *h_true_plaintext_array =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_true_plaintext_array[index] = PLAINTEXT_TRUE;
  }
  uint32_t *true_plaintext_array = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(true_plaintext_array, h_true_plaintext_array,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  uint32_t *lwe_buffer_2 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_add_lwe_ciphertext_vector_plaintext_vector_32(
      v_stream, gpu_index, lwe_buffer_2, lwe_buffer_1, true_plaintext_array,
      input_lwe_dimension, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_buffer_1, stream, gpu_index);
  cuda_drop_async(true_plaintext_array, stream, gpu_index);
  free(h_true_plaintext_array);

  // 3. Multiply by 2
  uint32_t *h_cleartext_array =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_cleartext_array[index] = 2;
  }
  uint32_t *cleartext_array = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(cleartext_array, h_cleartext_array,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  uint32_t *lwe_buffer_3 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_mult_lwe_ciphertext_vector_cleartext_vector_32(
      v_stream, gpu_index, lwe_buffer_3, lwe_buffer_2, cleartext_array,
      input_lwe_dimension, input_lwe_ciphertext_count);
  cuda_drop_async(lwe_buffer_2, stream, gpu_index);

  // 4. Compute a PBS with the LUT created below
  uint32_t *h_pbs_lut = (uint32_t *)malloc((glwe_dimension + 1) *
                                           polynomial_size * sizeof(uint32_t));
  for (uint index = 0; index < (glwe_dimension + 1) * polynomial_size;
       index++) {
    h_pbs_lut[index] =
        index < (glwe_dimension * polynomial_size) ? 0 : PLAINTEXT_TRUE;
  }
  uint32_t *pbs_lut = (uint32_t *)cuda_malloc_async(
      (glwe_dimension + 1) * polynomial_size * sizeof(uint32_t), stream,
      gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut, h_pbs_lut,
                           (glwe_dimension + 1) * polynomial_size *
                               sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *h_pbs_lut_indexes =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_pbs_lut_indexes[index] = 0;
  }
  uint32_t *pbs_lut_indexes = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut_indexes, h_pbs_lut_indexes,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *lwe_pbs_buffer = (uint32_t *)cuda_malloc_async(
      (glwe_dimension * polynomial_size + 1) * input_lwe_ciphertext_count *
          sizeof(uint32_t),
      stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  cuda_bootstrap_amortized_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_pbs_buffer, pbs_lut, pbs_lut_indexes,
      lwe_buffer_3, bootstrapping_key, input_lwe_dimension, glwe_dimension,
      polynomial_size, pbs_base_log, pbs_level_count,
      input_lwe_ciphertext_count, 1, 0, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  cuda_drop_async(lwe_buffer_3, stream, gpu_index);
  cuda_drop_async(pbs_lut, stream, gpu_index);
  cuda_drop_async(pbs_lut_indexes, stream, gpu_index);
  free(h_pbs_lut);
  free(h_pbs_lut_indexes);

  cuda_keyswitch_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_array_out, lwe_pbs_buffer, ksk,
      glwe_dimension * polynomial_size, input_lwe_dimension, ks_base_log,
      ks_level_count, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_pbs_buffer, stream, gpu_index);
}

extern "C" void cuda_boolean_xnor_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out,
    void *lwe_array_in_1, void *lwe_array_in_2, void *bootstrapping_key,
    void *ksk, uint32_t input_lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level_count,
    uint32_t ks_base_log, uint32_t ks_level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  uint32_t *lwe_buffer_1 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  // 1. Add the two ciphertexts
  cuda_add_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_buffer_1, lwe_array_in_1, lwe_array_in_2,
      input_lwe_dimension, input_lwe_ciphertext_count);
  // 2. Add "true" plaintext
  uint32_t *h_true_plaintext_array =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_true_plaintext_array[index] = PLAINTEXT_TRUE;
  }
  uint32_t *true_plaintext_array = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(true_plaintext_array, h_true_plaintext_array,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  uint32_t *lwe_buffer_2 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_add_lwe_ciphertext_vector_plaintext_vector_32(
      v_stream, gpu_index, lwe_buffer_2, lwe_buffer_1, true_plaintext_array,
      input_lwe_dimension, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_buffer_1, stream, gpu_index);
  cuda_drop_async(true_plaintext_array, stream, gpu_index);
  free(h_true_plaintext_array);
  // 3. Negate ciphertext
  uint32_t *lwe_buffer_3 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_negate_lwe_ciphertext_vector_32(v_stream, gpu_index, lwe_buffer_3,
                                       lwe_buffer_2, input_lwe_dimension,
                                       input_lwe_ciphertext_count);
  cuda_drop_async(lwe_buffer_2, stream, gpu_index);
  // 4. Multiply by 2
  uint32_t *h_cleartext_array =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_cleartext_array[index] = 2;
  }
  uint32_t *cleartext_array = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(cleartext_array, h_cleartext_array,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  uint32_t *lwe_buffer_4 = (uint32_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * input_lwe_ciphertext_count * sizeof(uint32_t),
      stream, gpu_index);
  cuda_mult_lwe_ciphertext_vector_cleartext_vector_32(
      v_stream, gpu_index, lwe_buffer_4, lwe_buffer_3, cleartext_array,
      input_lwe_dimension, input_lwe_ciphertext_count);
  cuda_drop_async(lwe_buffer_3, stream, gpu_index);

  // 5. Compute a PBS with the LUT created below
  uint32_t *h_pbs_lut = (uint32_t *)malloc((glwe_dimension + 1) *
                                           polynomial_size * sizeof(uint32_t));
  for (uint index = 0; index < (glwe_dimension + 1) * polynomial_size;
       index++) {
    h_pbs_lut[index] =
        index < (glwe_dimension * polynomial_size) ? 0 : PLAINTEXT_TRUE;
  }
  uint32_t *pbs_lut = (uint32_t *)cuda_malloc_async(
      (glwe_dimension + 1) * polynomial_size * sizeof(uint32_t), stream,
      gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut, h_pbs_lut,
                           (glwe_dimension + 1) * polynomial_size *
                               sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *h_pbs_lut_indexes =
      (uint32_t *)malloc(input_lwe_ciphertext_count * sizeof(uint32_t));
  for (uint index = 0; index < input_lwe_ciphertext_count; index++) {
    h_pbs_lut_indexes[index] = 0;
  }
  uint32_t *pbs_lut_indexes = (uint32_t *)cuda_malloc_async(
      input_lwe_ciphertext_count * sizeof(uint32_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(pbs_lut_indexes, h_pbs_lut_indexes,
                           input_lwe_ciphertext_count * sizeof(uint32_t),
                           stream, gpu_index);
  check_cuda_error(cudaGetLastError());
  uint32_t *lwe_pbs_buffer = (uint32_t *)cuda_malloc_async(
      (glwe_dimension * polynomial_size + 1) * input_lwe_ciphertext_count *
          sizeof(uint32_t),
      stream, gpu_index);
  check_cuda_error(cudaGetLastError());

  cuda_bootstrap_amortized_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_pbs_buffer, pbs_lut, pbs_lut_indexes,
      lwe_buffer_4, bootstrapping_key, input_lwe_dimension, glwe_dimension,
      polynomial_size, pbs_base_log, pbs_level_count,
      input_lwe_ciphertext_count, 1, 0, max_shared_memory);
  check_cuda_error(cudaGetLastError());

  cuda_drop_async(lwe_buffer_4, stream, gpu_index);
  cuda_drop_async(pbs_lut, stream, gpu_index);
  cuda_drop_async(pbs_lut_indexes, stream, gpu_index);
  free(h_pbs_lut);
  free(h_pbs_lut_indexes);

  cuda_keyswitch_lwe_ciphertext_vector_32(
      v_stream, gpu_index, lwe_array_out, lwe_pbs_buffer, ksk,
      glwe_dimension * polynomial_size, input_lwe_dimension, ks_base_log,
      ks_level_count, input_lwe_ciphertext_count);

  cuda_drop_async(lwe_pbs_buffer, stream, gpu_index);
}
#endif // CUDA_BOOLEAN_GATES_CU
