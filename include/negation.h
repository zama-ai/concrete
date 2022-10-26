#ifndef CUDA_NEGATE_H_
#define CUDA_NEGATE_H_

#include <cstdint>

extern "C" {

void cuda_negate_lwe_ciphertext_vector_32(void *v_stream, uint32_t gpu_index,
                                          void *lwe_array_out,
                                          void *lwe_array_in,
                                          uint32_t input_lwe_dimension,
                                          uint32_t input_lwe_ciphertext_count);
void cuda_negate_lwe_ciphertext_vector_64(void *v_stream, uint32_t gpu_index,
                                          void *lwe_array_out,
                                          void *lwe_array_in,
                                          uint32_t input_lwe_dimension,
                                          uint32_t input_lwe_ciphertext_count);
}

#endif // CUDA_NEGATE_H_
