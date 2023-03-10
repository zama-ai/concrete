#ifndef CUDA_BIT_EXTRACT_H
#define CUDA_BIT_EXTRACT_H

#include <cstdint>

extern "C" {

void scratch_cuda_extract_bits_32(
    void *v_stream, uint32_t gpu_index, int8_t **bit_extract_buffer,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t number_of_inputs, uint32_t max_shared_memory,
    bool allocate_gpu_memory);

void scratch_cuda_extract_bits_64(
    void *v_stream, uint32_t gpu_index, int8_t **bit_extract_buffer,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t number_of_inputs, uint32_t max_shared_memory,
    bool allocate_gpu_memory);

void cuda_extract_bits_32(void *v_stream, uint32_t gpu_index,
                          void *list_lwe_array_out, void *lwe_array_in,
                          int8_t *bit_extract_buffer, void *ksk,
                          void *fourier_bsk, uint32_t number_of_bits,
                          uint32_t delta_log, uint32_t lwe_dimension_in,
                          uint32_t lwe_dimension_out, uint32_t glwe_dimension,
                          uint32_t polynomial_size, uint32_t base_log_bsk,
                          uint32_t level_count_bsk, uint32_t base_log_ksk,
                          uint32_t level_count_ksk, uint32_t number_of_samples,
                          uint32_t max_shared_memory);

void cuda_extract_bits_64(void *v_stream, uint32_t gpu_index,
                          void *list_lwe_array_out, void *lwe_array_in,
                          int8_t *bit_extract_buffer, void *ksk,
                          void *fourier_bsk, uint32_t number_of_bits,
                          uint32_t delta_log, uint32_t lwe_dimension_in,
                          uint32_t lwe_dimension_out, uint32_t glwe_dimension,
                          uint32_t polynomial_size, uint32_t base_log_bsk,
                          uint32_t level_count_bsk, uint32_t base_log_ksk,
                          uint32_t level_count_ksk, uint32_t number_of_samples,
                          uint32_t max_shared_memory);

void cleanup_cuda_extract_bits(void *v_stream, uint32_t gpu_index,
                               int8_t **bit_extract_buffer);
}

#endif // CUDA_BIT_EXTRACT_H
