#include "addition.cuh"

void cuda_add_lwe_ciphertext_vector_32(void *v_stream, uint32_t gpu_index,
                                       void *lwe_array_out,
                                       void *lwe_array_in_1,
                                       void *lwe_array_in_2,
                                       uint32_t input_lwe_dimension,
                                       uint32_t input_lwe_ciphertext_count) {

  host_addition(v_stream, gpu_index, static_cast<uint32_t *>(lwe_array_out),
                static_cast<uint32_t *>(lwe_array_in_1),
                static_cast<uint32_t *>(lwe_array_in_2), input_lwe_dimension,
                input_lwe_ciphertext_count);
}
void cuda_add_lwe_ciphertext_vector_64(void *v_stream, uint32_t gpu_index,
                                       void *lwe_array_out,
                                       void *lwe_array_in_1,
                                       void *lwe_array_in_2,
                                       uint32_t input_lwe_dimension,
                                       uint32_t input_lwe_ciphertext_count) {

  host_addition(v_stream, gpu_index, static_cast<uint64_t *>(lwe_array_out),
                static_cast<uint64_t *>(lwe_array_in_1),
                static_cast<uint64_t *>(lwe_array_in_2), input_lwe_dimension,
                input_lwe_ciphertext_count);
}
