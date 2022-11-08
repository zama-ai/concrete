#include "multiplication.cuh"

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
