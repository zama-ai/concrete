#include "keyswitch.cuh"
#include "keyswitch.h"
#include "polynomial/parameters.cuh"

#include <cstdint>

/* Perform keyswitch on a batch of input LWE ciphertexts for 32 bits
 *
 *  - lwe_array_out: output batch of num_samples keyswitched ciphertexts c =
 * (a0,..an-1,b) where n is the LWE dimension
 *  - lwe_array_in: input batch of num_samples LWE ciphertexts, containing n
 *            mask values + 1 body value
 *
 * This function calls a wrapper to a device kernel that performs the keyswitch
 * 	- num_samples blocks of threads are launched
 */
void cuda_keyswitch_lwe_ciphertext_vector_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *ksk, uint32_t lwe_dimension_in, uint32_t lwe_dimension_out,
    uint32_t base_log, uint32_t level_count, uint32_t num_samples) {
  cuda_keyswitch_lwe_ciphertext_vector(
      v_stream, gpu_index, static_cast<uint32_t *>(lwe_array_out),
      static_cast<uint32_t *>(lwe_array_in), static_cast<uint32_t *>(ksk),
      lwe_dimension_in, lwe_dimension_out, base_log, level_count, num_samples);
}

/* Perform keyswitch on a batch of input LWE ciphertexts for 64 bits
 *
 *  - lwe_array_out: output batch of num_samples keyswitched ciphertexts c =
 * (a0,..an-1,b) where n is the LWE dimension
 *  - lwe_array_in: input batch of num_samples LWE ciphertexts, containing n
 *            mask values + 1 body value
 *
 * This function calls a wrapper to a device kernel that performs the keyswitch
 * 	- num_samples blocks of threads are launched
 */
void cuda_keyswitch_lwe_ciphertext_vector_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *ksk, uint32_t lwe_dimension_in, uint32_t lwe_dimension_out,
    uint32_t base_log, uint32_t level_count, uint32_t num_samples) {
  cuda_keyswitch_lwe_ciphertext_vector(
      v_stream, gpu_index, static_cast<uint64_t *>(lwe_array_out),
      static_cast<uint64_t *>(lwe_array_in), static_cast<uint64_t *>(ksk),
      lwe_dimension_in, lwe_dimension_out, base_log, level_count, num_samples);
}

void cuda_fp_keyswitch_lwe_to_glwe_32(void *v_stream, void *glwe_array_out,
                                      void *lwe_array_in, void *fp_ksk_array,
                                      uint32_t input_lwe_dimension,
                                      uint32_t output_glwe_dimension,
                                      uint32_t output_polynomial_size,
                                      uint32_t base_log, uint32_t level_count,
                                      uint32_t number_of_input_lwe,
                                      uint32_t number_of_keys) {

  cuda_fp_keyswitch_lwe_to_glwe(
      v_stream, static_cast<uint32_t *>(glwe_array_out),
      static_cast<uint32_t *>(lwe_array_in),
      static_cast<uint32_t *>(fp_ksk_array), input_lwe_dimension,
      output_glwe_dimension, output_polynomial_size, base_log, level_count,
      number_of_input_lwe, number_of_keys);
}
void cuda_fp_keyswitch_lwe_to_glwe_64(void *v_stream, void *glwe_array_out,
                                      void *lwe_array_in, void *fp_ksk_array,
                                      uint32_t input_lwe_dimension,
                                      uint32_t output_glwe_dimension,
                                      uint32_t output_polynomial_size,
                                      uint32_t base_log, uint32_t level_count,
                                      uint32_t number_of_input_lwe,
                                      uint32_t number_of_keys) {

  cuda_fp_keyswitch_lwe_to_glwe(
      v_stream, static_cast<uint64_t *>(glwe_array_out),
      static_cast<uint64_t *>(lwe_array_in),
      static_cast<uint64_t *>(fp_ksk_array), input_lwe_dimension,
      output_glwe_dimension, output_polynomial_size, base_log, level_count,
      number_of_input_lwe, number_of_keys);
}
